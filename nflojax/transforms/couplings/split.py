from __future__ import annotations

from .shared import *

# ===================================================================
# Split Coupling: structured coupling on rank-N events
# ===================================================================
# SplitCoupling is the structured analogue of SplineCoupling. Instead of a
# flat (dim,) binary mask, it splits along a chosen event axis at a chosen
# index, passes the frozen half (flattened) to a conditioner, and applies
# a per-scalar monotonic rational-quadratic spline to the transformed half.
# Log-det is summed over the trailing `event_ndims` axes so that the output
# has the enclosing batch shape.
#
# Design: this class deliberately duplicates the spline-call and identity-init
# bias patching from SplineCoupling rather than extracting a shared helper.
# The duplication is ~30 lines and keeps each coupling variant readable on
# its own; per AGENTS.md "three similar lines beat a premature helper."
@dataclass
class SplitCoupling:
    """
    Axis-based spline coupling for rank-N events.

    Splits input along `split_axis` at `split_index`, feeds the frozen slice
    (flattened) to the conditioner, applies monotonic rational-quadratic
    splines elementwise to the transformed slice, concatenates back together.

    Input/output shape:
      x:         (*batch, *event_shape)   where len(event_shape) == event_ndims
      y:         same as x
      log_det:   batch shape               (summed over all event axes)

    Target use case (particle systems):
      event_shape = (N, d),  split_axis=-2,  split_index=N//2,  event_ndims=2
      frozen slice:      (*batch, N//2, d)      -> conditioner input
                                                   (flattened to (*batch, N//2 * d)
                                                    when flatten_input=True)
      transformed slice: (*batch, N//2, d)      -> elementwise spline

    Fields:
      event_shape: trailing event shape, e.g. (N, d); its rank must equal
                   event_ndims.
      split_axis:  negative int indexing from the end; must lie inside event
                   axes (i.e., abs(split_axis) <= event_ndims).
      split_index: size of the first partition along split_axis.
      event_ndims: number of trailing event axes (>= 1).
      conditioner: Flax module producing a tensor whose total trailing size
                   equals `required_out_dim(transformed_flat, num_bins,
                   boundary_slopes)` = `transformed_flat * (3K-1)` for
                   `linear_tails` or `transformed_flat * 3K` for `circular`.
                   Input shape depends on `flatten_input` (see below).
      swap:        if False, the first partition is frozen; if True, the last.
      flatten_input: when True (default), the frozen slice is flattened to
                   `(*batch, frozen_flat)` before the conditioner call — the
                   contract `MLP` expects. When False, the conditioner sees
                   the structured frozen slice `(*batch, *frozen_shape)`
                   directly; use this for permutation-aware conditioners
                   (`DeepSets`, `Transformer`, `GNN`) that need the particle
                   axis preserved. The output-side reshape only requires the
                   total element count; the conditioner may emit either flat
                   `(*batch, transformed_flat * params_per_scalar)` or
                   structured `(*batch, *transformed_shape, params_per_scalar)`.
      num_bins..max_derivative: same as SplineCoupling.

    Mask semantics: none. The partition is geometric (axis+index), not a
    per-scalar mask. Alternate `swap` between layers to cover all particles.

    Identity initialization: same mechanism as SplineCoupling. The conditioner
    output layer has zeroed kernel and a bias that makes widths=heights=0
    (uniform bins) and derivatives=1 (via the stable_logit of
    (1-min_deriv)/(max_deriv-min_deriv)). Inside [-tail_bound, tail_bound] the
    spline is identity at init.
    """
    event_shape: Tuple[int, ...]
    split_axis: int
    split_index: int
    event_ndims: int
    conditioner: Any
    swap: bool = False
    num_bins: int = 8
    tail_bound: float = 5.0
    min_bin_width: float = 1e-2
    min_bin_height: float = 1e-2
    min_derivative: float = 1e-2
    max_derivative: float = 10.0
    boundary_slopes: str = "linear_tails"
    flatten_input: bool = True

    def __post_init__(self):
        self.event_shape = tuple(int(d) for d in self.event_shape)
        if self.split_axis >= 0:
            raise ValueError(
                f"SplitCoupling: split_axis must be negative, got {self.split_axis}."
            )
        _validate_boundary_slopes(self.boundary_slopes, where="SplitCoupling")
        if self.event_ndims < 1:
            raise ValueError(
                f"SplitCoupling: event_ndims must be >= 1, got {self.event_ndims}."
            )
        if len(self.event_shape) != self.event_ndims:
            raise ValueError(
                f"SplitCoupling: event_shape {self.event_shape} has rank "
                f"{len(self.event_shape)}, expected event_ndims={self.event_ndims}."
            )
        if -self.split_axis > self.event_ndims:
            raise ValueError(
                f"SplitCoupling: split_axis={self.split_axis} lies outside the "
                f"trailing {self.event_ndims} event axes."
            )
        if self.split_index <= 0:
            raise ValueError(
                f"SplitCoupling: split_index must be positive, got {self.split_index}."
            )
        event_axis = self.event_ndims + self.split_axis
        axis_size = self.event_shape[event_axis]
        if self.split_index >= axis_size:
            raise ValueError(
                f"SplitCoupling: split_index={self.split_index} must be < size "
                f"along split_axis ({axis_size})."
            )
        validate_conditioner(self.conditioner, name="SplitCoupling.conditioner")

        # Warn if identity-spline init is unreachable (derivative=1 outside range).
        lo, hi = float(self.min_derivative), float(self.max_derivative)
        if not (lo < 1.0 < hi):
            warnings.warn(
                f"SplitCoupling: derivative range [{lo}, {hi}] excludes 1.0; "
                "identity-like initialization not possible.",
                stacklevel=2,
            )

    @staticmethod
    def required_out_dim(
        transformed_flat: int,
        num_bins: int,
        boundary_slopes: str = "linear_tails",
    ) -> int:
        """Conditioner output dimension for the transformed slice.

        - 'linear_tails': `transformed_flat * (3K - 1)`.
        - 'circular':     `transformed_flat * 3K` (one extra shared boundary slope per scalar).
        """
        return transformed_flat * _params_per_scalar(num_bins, boundary_slopes)

    # ------------------------------------------------------------------
    # Partition sizing. Both `create()` (pre-instance, for MLP sizing) and
    # `init_params` (on an instance) need (frozen_flat, transformed_flat);
    # this staticmethod is the single source of truth for the computation.
    # ------------------------------------------------------------------
    @staticmethod
    def _partition_flats(
        event_shape: Sequence[int],
        split_axis: int,
        split_index: int,
        event_ndims: int,
        swap: bool,
    ) -> Tuple[int, int]:
        event_shape = tuple(int(d) for d in event_shape)
        event_axis = event_ndims + split_axis
        if not (0 <= event_axis < event_ndims):
            raise ValueError(
                f"SplitCoupling: split_axis={split_axis} lies outside the "
                f"trailing {event_ndims} event axes."
            )
        axis_size = event_shape[event_axis]
        frozen_sz = split_index if not swap else axis_size - split_index
        transformed_sz = axis_size - frozen_sz
        other = math.prod(event_shape) // axis_size
        return frozen_sz * other, transformed_sz * other

    @classmethod
    def create(
        cls,
        key: PRNGKey,
        event_shape: Sequence[int],
        split_axis: int,
        split_index: int,
        event_ndims: int,
        hidden_dim: int,
        n_hidden_layers: int,
        *,
        context_dim: int = 0,
        num_bins: int = 8,
        tail_bound: float = 5.0,
        min_bin_width: float = 1e-2,
        min_bin_height: float = 1e-2,
        min_derivative: float = 1e-2,
        max_derivative: float = 10.0,
        swap: bool = False,
        activation: Callable[[Array], Array] = nn.elu,
        res_scale: float = 0.1,
        boundary_slopes: str = "linear_tails",
    ) -> Tuple["SplitCoupling", dict]:
        """
        Factory. Sizes the conditioner MLP for the given event_shape partition,
        initializes with identity-spline bias, returns (coupling, params).
        """
        event_shape = tuple(int(d) for d in event_shape)
        frozen_flat, transformed_flat = cls._partition_flats(
            event_shape, split_axis, split_index, event_ndims, swap
        )

        mlp = MLP(
            x_dim=frozen_flat,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            out_dim=cls.required_out_dim(
                transformed_flat, num_bins, boundary_slopes=boundary_slopes
            ),
            activation=activation,
            res_scale=res_scale,
        )

        coupling = cls(
            event_shape=event_shape,
            split_axis=split_axis,
            split_index=split_index,
            event_ndims=event_ndims,
            conditioner=mlp,
            swap=swap,
            num_bins=num_bins,
            tail_bound=tail_bound,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
            max_derivative=max_derivative,
            boundary_slopes=boundary_slopes,
        )
        params = coupling.init_params(key, context_dim=context_dim)
        return coupling, params

    # ------------------------------------------------------------------
    # Split / combine along split_axis. jnp.split is jit-friendly.
    # ------------------------------------------------------------------
    def _split(self, x: Array) -> Tuple[Array, Array]:
        a, b = jnp.split(x, [self.split_index], axis=self.split_axis)
        return (b, a) if self.swap else (a, b)

    def _combine(self, frozen: Array, transformed_new: Array) -> Array:
        if self.swap:
            return jnp.concatenate([transformed_new, frozen], axis=self.split_axis)
        return jnp.concatenate([frozen, transformed_new], axis=self.split_axis)

    def _check_x(self, x: Array) -> None:
        """Validate trailing shape matches event_shape; mirrors SplineCoupling._check_x."""
        if x.ndim < self.event_ndims:
            raise ValueError(
                f"SplitCoupling: input has rank {x.ndim}, "
                f"need at least event_ndims={self.event_ndims}."
            )
        if x.shape[-self.event_ndims:] != self.event_shape:
            raise ValueError(
                f"SplitCoupling: expected trailing event_shape {self.event_shape}, "
                f"got {x.shape[-self.event_ndims:]}."
            )

    def _forward_or_inverse(
        self,
        params: Any,
        x: Array,
        context: Array | None,
        inverse: bool,
    ) -> Tuple[Array, Array]:
        self._check_x(x)
        mlp_params = params["mlp"]
        frozen, transformed = self._split(x)

        K = self.num_bins
        params_per_scalar = _params_per_scalar(K, self.boundary_slopes)
        transformed_event_shape = transformed.shape[-self.event_ndims:]
        batch_shape = frozen.shape[: -self.event_ndims]

        # Conditioner sees the frozen slice (flat or structured per
        # `flatten_input`); its output is reshaped to per-scalar spline params
        # over the transformed slice. The output reshape depends only on the
        # total element count, so the conditioner is free to emit a flat or
        # structured tensor of the right size.
        cond_input = frozen.reshape(batch_shape + (-1,)) if self.flatten_input else frozen
        theta = self.conditioner.apply({"params": mlp_params}, cond_input, context)
        theta = theta.reshape(batch_shape + transformed_event_shape + (params_per_scalar,))
        widths = theta[..., :K]
        heights = theta[..., K : 2 * K]
        derivatives = theta[..., 2 * K :]

        transformed_new, logabsdet_per_scalar = rational_quadratic_spline(
            inputs=transformed,
            unnormalized_widths=widths,
            unnormalized_heights=heights,
            unnormalized_derivatives=derivatives,
            tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            max_derivative=self.max_derivative,
            inverse=inverse,
            boundary_slopes=self.boundary_slopes,
        )
        y = self._combine(frozen, transformed_new)
        # Sum per-scalar log-det over all event axes => shape == batch_shape.
        log_det = jnp.sum(
            logabsdet_per_scalar,
            axis=tuple(range(-self.event_ndims, 0)),
        )
        return y, log_det

    def forward(
        self,
        params: Any,
        x: Array,
        context: Array | None = None,
    ) -> Tuple[Array, Array]:
        return self._forward_or_inverse(params, x, context, inverse=False)

    def inverse(
        self,
        params: Any,
        y: Array,
        context: Array | None = None,
    ) -> Tuple[Array, Array]:
        return self._forward_or_inverse(params, y, context, inverse=True)

    def _patch_dense_out(self, mlp_params: Any) -> Any:
        """Install an identity-spline bias on the conditioner's `dense_out`.

        The bias length is read from the conditioner itself, so both flat-output
        (`MLP`, bias = `transformed_flat * params_per_scalar`) and per-token
        output (`Transformer`, `GNN`, bias = `d * params_per_scalar`) conditioners
        are handled uniformly. `identity_spline_bias` is the same per-scalar
        pattern tiled across `num_scalars` scalars, so sizing it to match the
        conditioner's bias works for any multiple of `params_per_scalar`.
        """
        if not (hasattr(self.conditioner, "get_output_layer") and
                hasattr(self.conditioner, "set_output_layer")):
            raise RuntimeError(
                "SplitCoupling._patch_dense_out: conditioner must implement "
                "get_output_layer() and set_output_layer()."
            )
        out_layer = self.conditioner.get_output_layer(mlp_params)
        bias_size = int(out_layer["bias"].shape[0])
        params_per_scalar = _params_per_scalar(self.num_bins, self.boundary_slopes)
        if bias_size % params_per_scalar != 0:
            raise ValueError(
                f"SplitCoupling: conditioner dense_out bias shape ({bias_size},) "
                f"is not a multiple of params_per_scalar={params_per_scalar}."
            )
        new_kernel = jnp.zeros_like(out_layer["kernel"])
        new_bias = identity_spline_bias(
            num_scalars=bias_size // params_per_scalar,
            num_bins=self.num_bins,
            min_derivative=self.min_derivative,
            max_derivative=self.max_derivative,
            boundary_slopes=self.boundary_slopes,
        )
        return self.conditioner.set_output_layer(mlp_params, new_kernel, new_bias)

    def init_params(self, key: PRNGKey, context_dim: int = 0) -> dict:
        """
        Initialize conditioner params with an identity-spline output layer.

        Shapes are derived from `self.event_shape` + the partition fields.
        After init, the conditioner's full output shape is checked against the
        coupling's expectation — a per-token conditioner (`Transformer`, `GNN`)
        sized for a symmetric split but used with an asymmetric `split_index`
        raises here, not as a cryptic reshape error during the first forward.
        """
        frozen_flat, transformed_flat = self._partition_flats(
            self.event_shape, self.split_axis, self.split_index,
            self.event_ndims, self.swap,
        )
        if self.flatten_input:
            dummy_x = jnp.zeros((1, frozen_flat), dtype=jnp.float32)
        else:
            event_axis = self.event_ndims + self.split_axis
            axis_size = self.event_shape[event_axis]
            frozen_sz = self.split_index if not self.swap else axis_size - self.split_index
            frozen_shape = list(self.event_shape)
            frozen_shape[event_axis] = frozen_sz
            dummy_x = jnp.zeros((1,) + tuple(frozen_shape), dtype=jnp.float32)
        dummy_context = (
            jnp.zeros((1, context_dim), dtype=jnp.float32) if context_dim > 0 else None
        )
        variables = self.conditioner.init(key, dummy_x, dummy_context)
        mlp_params = variables["params"]
        mlp_params = self._patch_dense_out(mlp_params)

        # Validate the conditioner's output shape against the coupling's
        # expectation. The reshape in `_forward_or_inverse` requires
        # `prod(trailing) == transformed_flat * params_per_scalar`.
        params_per_scalar = _params_per_scalar(self.num_bins, self.boundary_slopes)
        expected = transformed_flat * params_per_scalar
        dummy_out = self.conditioner.apply(
            {"params": mlp_params}, dummy_x, dummy_context
        )
        total_trailing = math.prod(dummy_out.shape[1:])
        if total_trailing != expected:
            axis_size = self.event_shape[self.event_ndims + self.split_axis]
            raise ValueError(
                f"SplitCoupling: conditioner "
                f"{type(self.conditioner).__name__!r} produces output with "
                f"total trailing size {total_trailing}; expected {expected} "
                f"(= transformed_flat={transformed_flat} * "
                f"params_per_scalar={params_per_scalar}). Common cause: a "
                f"per-token conditioner (Transformer, GNN) sized for a "
                f"symmetric split but used with split_index={self.split_index} "
                f"on an axis of size {axis_size} (N_frozen != N_transformed)."
            )
        return {"mlp": mlp_params}

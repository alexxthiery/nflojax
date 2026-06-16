from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from flax import linen as nn

from ..domains import ProductDomain
from ..nets import MLP, Array, PRNGKey, validate_conditioner
from ..splines import rational_quadratic_spline
from .common import _params_per_scalar, identity_spline_bias

# ===================================================================
# CircularCoordinateShift: flat product-domain circular coordinate shift
# ===================================================================
@dataclass
class CircularCoordinateShift:
    """Learnable rigid shift on circular coordinates of a flat product domain.

    Non-circular coordinates are copied through unchanged. Circular coordinates
    are shifted modulo their configured interval. The log-det is identically
    zero because the map is a rigid rotation on each circle.
    """

    domain: ProductDomain

    def __post_init__(self) -> None:
        if not isinstance(self.domain, ProductDomain):
            raise TypeError(
                f"CircularCoordinateShift: domain must be a ProductDomain, "
                f"got {type(self.domain).__name__}."
            )
        circular_mask = np.asarray(
            [d.kind == "circular" for d in self.domain.domains],
            dtype=bool,
        )
        object.__setattr__(self, "_circular_mask", circular_mask)
        lower, upper = self.domain.bounds_arrays()
        object.__setattr__(self, "_lower_j", lower)
        object.__setattr__(self, "_width_j", upper - lower)
        object.__setattr__(self, "_mask_j", jnp.asarray(circular_mask))

    def _check_x(self, x: Array) -> None:
        if x.shape[-1:] != self.domain.event_shape:
            raise ValueError(
                f"CircularCoordinateShift: expected trailing event_shape "
                f"{self.domain.event_shape}, got {x.shape[-1:]}."
            )

    def _wrap_shift(
        self,
        params: Any,
        x: Array,
        sign: float,
        g_value: Array | None,
    ) -> Tuple[Array, Array]:
        self._check_x(x)
        try:
            shift = params["shift"]
        except Exception as e:
            raise KeyError("CircularCoordinateShift expected params['shift'].") from e
        if shift.shape != (self.domain.dim,):
            raise ValueError(
                f"CircularCoordinateShift: expected shift shape ({self.domain.dim},), "
                f"got {shift.shape}."
            )
        shift = jnp.asarray(shift, dtype=x.dtype)
        if g_value is not None:
            shift = shift * g_value[..., None]
        lower = self._lower_j.astype(x.dtype)
        width = self._width_j.astype(x.dtype)
        shifted = jnp.mod(x - lower + sign * shift, width) + lower
        y = jnp.where(self._mask_j, shifted, x)
        return y, jnp.zeros((), dtype=x.dtype)

    def forward(
        self,
        params: Any,
        x: Array,
        context: Array | None = None,
        g_value: Array | None = None,
    ) -> Tuple[Array, Array]:
        del context
        return self._wrap_shift(params, x, sign=+1.0, g_value=g_value)

    def inverse(
        self,
        params: Any,
        y: Array,
        context: Array | None = None,
        g_value: Array | None = None,
    ) -> Tuple[Array, Array]:
        del context
        return self._wrap_shift(params, y, sign=-1.0, g_value=g_value)

    def init_params(self, key: PRNGKey, context_dim: int = 0) -> dict:
        """Zero shift, so the layer is identity at init."""

        del key, context_dim
        return {"shift": jnp.zeros((self.domain.dim,), dtype=jnp.float32)}

    @classmethod
    def create(
        cls,
        key: PRNGKey,
        domain: ProductDomain,
    ) -> Tuple["CircularCoordinateShift", dict]:
        """Factory returning ``(transform, params)``."""

        transform = cls(domain=domain)
        return transform, transform.init_params(key)


# ===================================================================
# ProductSplineCoupling: flat mixed-domain spline coupling
# ===================================================================
@dataclass
class ProductSplineCoupling:
    """RealNVP-style spline coupling on a flat product domain.

    Coordinates marked by ``mask == 1`` are frozen and are mapped to
    conditioner inputs using ``ProductDomain.conditioner_features``, the
    default MLP feature map for this shipped product-domain coupling.
    Coordinates marked by ``mask == 0`` are transformed by scalar
    rational-quadratic splines appropriate to each coordinate domain: linear
    tails for real/interval coordinates and circular boundary slopes for
    circular coordinates.
    """

    domain: ProductDomain
    mask: Array
    conditioner: Any
    num_bins: int = 8
    tail_bound: float = 5.0
    min_bin_width: float = 1e-2
    min_bin_height: float = 1e-2
    min_derivative: float = 1e-2
    max_derivative: float = 10.0
    circular_n_freq: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.domain, ProductDomain):
            raise TypeError(
                f"ProductSplineCoupling: domain must be a ProductDomain, "
                f"got {type(self.domain).__name__}."
            )
        mask = jnp.asarray(self.mask, dtype=jnp.float32)
        mask_tuple = self.domain._check_mask(mask)
        if all(mask_tuple) or not any(mask_tuple):
            raise ValueError(
                "ProductSplineCoupling mask must freeze at least one coordinate "
                "and transform at least one coordinate."
            )
        if self.num_bins <= 0:
            raise ValueError(
                f"ProductSplineCoupling: num_bins must be positive, got {self.num_bins}."
            )
        if self.tail_bound <= 0:
            raise ValueError(
                f"ProductSplineCoupling: tail_bound must be positive, got {self.tail_bound}."
            )
        if self.circular_n_freq <= 0:
            raise ValueError("ProductSplineCoupling: circular_n_freq must be positive.")
        validate_conditioner(self.conditioner, name="ProductSplineCoupling.conditioner")
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "_mask_tuple", mask_tuple)

        frozen_real_indices: list[int] = []
        frozen_real_slots: list[int] = []
        frozen_interval_indices: list[int] = []
        frozen_interval_slots: list[int] = []
        frozen_circular_indices: list[int] = []
        frozen_circular_slots: list[int] = []
        feature_slot = 0
        for i, (frozen, domain) in enumerate(zip(mask_tuple, self.domain.domains)):
            if not frozen:
                continue
            if domain.kind == "real":
                frozen_real_indices.append(i)
                frozen_real_slots.append(feature_slot)
                feature_slot += 1
            elif domain.kind == "interval":
                frozen_interval_indices.append(i)
                frozen_interval_slots.append(feature_slot)
                feature_slot += 1
            else:
                frozen_circular_indices.append(i)
                frozen_circular_slots.extend(
                    range(feature_slot, feature_slot + 2 * self.circular_n_freq)
                )
                feature_slot += 2 * self.circular_n_freq

        object.__setattr__(self, "_feature_dim", feature_slot)
        object.__setattr__(
            self,
            "_frozen_real_indices",
            tuple(frozen_real_indices),
        )
        object.__setattr__(
            self,
            "_frozen_interval_indices",
            tuple(frozen_interval_indices),
        )
        object.__setattr__(
            self,
            "_frozen_circular_indices",
            tuple(frozen_circular_indices),
        )
        object.__setattr__(
            self,
            "_frozen_real_index_array",
            jnp.asarray(frozen_real_indices, dtype=jnp.int32),
        )
        object.__setattr__(
            self,
            "_frozen_real_slot_array",
            jnp.asarray(frozen_real_slots, dtype=jnp.int32),
        )
        object.__setattr__(
            self,
            "_frozen_interval_index_array",
            jnp.asarray(frozen_interval_indices, dtype=jnp.int32),
        )
        object.__setattr__(
            self,
            "_frozen_interval_slot_array",
            jnp.asarray(frozen_interval_slots, dtype=jnp.int32),
        )
        object.__setattr__(
            self,
            "_frozen_circular_index_array",
            jnp.asarray(frozen_circular_indices, dtype=jnp.int32),
        )
        object.__setattr__(
            self,
            "_frozen_circular_slot_array",
            jnp.asarray(frozen_circular_slots, dtype=jnp.int32),
        )
        object.__setattr__(
            self,
            "_frozen_interval_lower",
            jnp.asarray(
                [self.domain.domains[i].lower for i in frozen_interval_indices],
                dtype=jnp.float32,
            ),
        )
        object.__setattr__(
            self,
            "_frozen_interval_width",
            jnp.asarray(
                [
                    self.domain.domains[i].upper - self.domain.domains[i].lower
                    for i in frozen_interval_indices
                ],
                dtype=jnp.float32,
            ),
        )
        object.__setattr__(
            self,
            "_frozen_circular_lower",
            jnp.asarray(
                [self.domain.domains[i].lower for i in frozen_circular_indices],
                dtype=jnp.float32,
            ),
        )
        object.__setattr__(
            self,
            "_frozen_circular_width",
            jnp.asarray(
                [
                    self.domain.domains[i].upper - self.domain.domains[i].lower
                    for i in frozen_circular_indices
                ],
                dtype=jnp.float32,
            ),
        )
        object.__setattr__(
            self,
            "_frozen_circular_freqs",
            jnp.arange(1, self.circular_n_freq + 1, dtype=jnp.float32),
        )
        object.__setattr__(
            self,
            "_frozen_circular_feature_count",
            len(frozen_circular_indices) * 2 * self.circular_n_freq,
        )
        object.__setattr__(
            self,
            "_transformed_indices",
            tuple(i for i, frozen in enumerate(mask_tuple) if not frozen),
        )
        linear_indices = tuple(
            i
            for i in self._transformed_indices
            if self.domain.domains[i].kind != "circular"
        )
        circular_indices = tuple(
            i
            for i in self._transformed_indices
            if self.domain.domains[i].kind == "circular"
        )
        object.__setattr__(self, "_linear_indices", linear_indices)
        object.__setattr__(self, "_circular_indices", circular_indices)
        object.__setattr__(
            self,
            "_linear_index_array",
            jnp.asarray(linear_indices, dtype=jnp.int32),
        )
        object.__setattr__(
            self,
            "_circular_index_array",
            jnp.asarray(circular_indices, dtype=jnp.int32),
        )
        object.__setattr__(
            self,
            "_linear_is_bounded",
            jnp.asarray(
                [
                    self.domain.domains[i].kind != "real"
                    for i in linear_indices
                ],
                dtype=bool,
            ),
        )
        object.__setattr__(
            self,
            "_linear_lower",
            jnp.asarray(
                [
                    0.0 if self.domain.domains[i].lower is None
                    else self.domain.domains[i].lower
                    for i in linear_indices
                ],
                dtype=jnp.float32,
            ),
        )
        object.__setattr__(
            self,
            "_linear_width",
            jnp.asarray(
                [
                    1.0 if self.domain.domains[i].upper is None
                    else self.domain.domains[i].upper - self.domain.domains[i].lower
                    for i in linear_indices
                ],
                dtype=jnp.float32,
            ),
        )
        object.__setattr__(
            self,
            "_circular_lower",
            jnp.asarray(
                [self.domain.domains[i].lower for i in circular_indices],
                dtype=jnp.float32,
            ),
        )
        object.__setattr__(
            self,
            "_circular_width",
            jnp.asarray(
                [
                    self.domain.domains[i].upper - self.domain.domains[i].lower
                    for i in circular_indices
                ],
                dtype=jnp.float32,
            ),
        )
        K = self.num_bins
        linear_width_positions: list[list[int]] = []
        linear_height_positions: list[list[int]] = []
        linear_derivative_positions: list[list[int]] = []
        circular_width_positions: list[list[int]] = []
        circular_height_positions: list[list[int]] = []
        circular_derivative_positions: list[list[int]] = []
        offset = 0
        for i in self._transformed_indices:
            boundary = (
                "circular"
                if self.domain.domains[i].kind == "circular"
                else "linear_tails"
            )
            size = _params_per_scalar(K, boundary)
            if boundary == "circular":
                circular_width_positions.append(list(range(offset, offset + K)))
                circular_height_positions.append(list(range(offset + K, offset + 2 * K)))
                circular_derivative_positions.append(
                    list(range(offset + 2 * K, offset + size))
                )
            else:
                linear_width_positions.append(list(range(offset, offset + K)))
                linear_height_positions.append(list(range(offset + K, offset + 2 * K)))
                linear_derivative_positions.append(
                    list(range(offset + 2 * K, offset + size))
                )
            offset += size
        object.__setattr__(self, "_out_dim", offset)
        object.__setattr__(
            self,
            "_linear_width_positions",
            jnp.asarray(linear_width_positions, dtype=jnp.int32),
        )
        object.__setattr__(
            self,
            "_linear_height_positions",
            jnp.asarray(linear_height_positions, dtype=jnp.int32),
        )
        object.__setattr__(
            self,
            "_linear_derivative_positions",
            jnp.asarray(linear_derivative_positions, dtype=jnp.int32),
        )
        object.__setattr__(
            self,
            "_circular_width_positions",
            jnp.asarray(circular_width_positions, dtype=jnp.int32),
        )
        object.__setattr__(
            self,
            "_circular_height_positions",
            jnp.asarray(circular_height_positions, dtype=jnp.int32),
        )
        object.__setattr__(
            self,
            "_circular_derivative_positions",
            jnp.asarray(circular_derivative_positions, dtype=jnp.int32),
        )
        lo, hi = float(self.min_derivative), float(self.max_derivative)
        if not (lo < 1.0 < hi):
            warnings.warn(
                f"ProductSplineCoupling: derivative range [{lo}, {hi}] excludes 1.0; "
                "identity-like initialization not possible, using midpoint derivative.",
                stacklevel=2,
            )

    @staticmethod
    def required_out_dim(
        domain: ProductDomain,
        mask: Array,
        num_bins: int,
    ) -> int:
        """Return conditioner output width for transformed coordinates."""

        return domain.required_out_dim(mask, num_bins)

    @classmethod
    def create(
        cls,
        key: PRNGKey,
        *,
        domain: ProductDomain,
        mask: Array,
        hidden_dim: int,
        n_hidden_layers: int,
        context_dim: int = 0,
        num_bins: int = 8,
        tail_bound: float = 5.0,
        min_bin_width: float = 1e-2,
        min_bin_height: float = 1e-2,
        min_derivative: float = 1e-2,
        max_derivative: float = 10.0,
        activation: Callable[[Array], Array] = nn.elu,
        res_scale: float = 0.1,
        circular_n_freq: int = 1,
    ) -> Tuple["ProductSplineCoupling", dict]:
        """Factory with the default MLP feature map and identity-spline init."""

        if not isinstance(domain, ProductDomain):
            raise TypeError(
                f"ProductSplineCoupling.create: domain must be a ProductDomain, "
                f"got {type(domain).__name__}."
            )
        feature_dim = domain.conditioner_feature_dim(mask, circular_n_freq)
        out_dim = domain.required_out_dim(mask, num_bins)
        mlp = MLP(
            x_dim=feature_dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            out_dim=out_dim,
            activation=activation,
            res_scale=res_scale,
        )
        coupling = cls(
            domain=domain,
            mask=mask,
            conditioner=mlp,
            num_bins=num_bins,
            tail_bound=tail_bound,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
            max_derivative=max_derivative,
            circular_n_freq=circular_n_freq,
        )
        return coupling, coupling.init_params(key, context_dim=context_dim)

    def _conditioner_params(self, params: Any) -> Any:
        try:
            return params["mlp"]
        except Exception as e:
            raise KeyError("ProductSplineCoupling expected params to contain key 'mlp'.") from e

    def _check_x(self, x: Array) -> None:
        if x.shape[-1:] != self.domain.event_shape:
            raise ValueError(
                f"ProductSplineCoupling: expected trailing event_shape "
                f"{self.domain.event_shape}, got {x.shape[-1:]}."
            )

    def _identity_bias(self, dtype=jnp.float32) -> Array:
        chunks = []
        for i in self._transformed_indices:
            boundary = (
                "circular"
                if self.domain.domains[i].kind == "circular"
                else "linear_tails"
            )
            chunks.append(
                identity_spline_bias(
                    1,
                    self.num_bins,
                    self.min_derivative,
                    self.max_derivative,
                    dtype=dtype,
                    boundary_slopes=boundary,
                )
            )
        return jnp.concatenate(chunks, axis=0)

    def _patch_dense_out(self, mlp_params: Any) -> Any:
        if not (
            hasattr(self.conditioner, "get_output_layer")
            and hasattr(self.conditioner, "set_output_layer")
        ):
            raise RuntimeError(
                "ProductSplineCoupling._patch_dense_out: conditioner must implement "
                "get_output_layer() and set_output_layer() methods."
            )
        out_layer = self.conditioner.get_output_layer(mlp_params)
        bias = out_layer["bias"]
        if bias.shape != (self._out_dim,):
            raise ValueError(
                f"ProductSplineCoupling: conditioner output bias must have shape "
                f"({self._out_dim},), got {bias.shape}."
            )
        return self.conditioner.set_output_layer(
            mlp_params,
            jnp.zeros_like(out_layer["kernel"]),
            self._identity_bias(dtype=bias.dtype),
        )

    def init_params(self, key: PRNGKey, context_dim: int = 0) -> dict:
        """Initialize conditioner params and patch output to identity splines."""

        dummy_x = jnp.zeros((1, self._feature_dim), dtype=jnp.float32)
        dummy_context = (
            jnp.zeros((1, context_dim), dtype=jnp.float32)
            if context_dim > 0
            else None
        )
        variables = self.conditioner.init(key, dummy_x, dummy_context)
        mlp_params = self._patch_dense_out(variables.get("params", {}))
        return {"mlp": mlp_params}

    def _to_linear_canonical(self, x_group: Array) -> Array:
        lower = self._linear_lower.astype(x_group.dtype)
        width = self._linear_width.astype(x_group.dtype)
        bounded = self._linear_is_bounded
        scaled = 2.0 * self.tail_bound * (x_group - lower) / width - self.tail_bound
        return jnp.where(bounded, scaled, x_group)

    def _from_linear_canonical(self, u_group: Array) -> Array:
        lower = self._linear_lower.astype(u_group.dtype)
        width = self._linear_width.astype(u_group.dtype)
        bounded = self._linear_is_bounded
        scaled = lower + (u_group + self.tail_bound) * width / (2.0 * self.tail_bound)
        return jnp.where(bounded, scaled, u_group)

    def _to_circular_canonical(self, x_group: Array) -> Array:
        lower = self._circular_lower.astype(x_group.dtype)
        width = self._circular_width.astype(x_group.dtype)
        return 2.0 * self.tail_bound * (x_group - lower) / width - self.tail_bound

    def _from_circular_canonical(self, u_group: Array) -> Array:
        lower = self._circular_lower.astype(u_group.dtype)
        width = self._circular_width.astype(u_group.dtype)
        x_group = lower + (u_group + self.tail_bound) * width / (2.0 * self.tail_bound)
        return jnp.mod(x_group - lower, width) + lower

    def _conditioner_features(self, x: Array) -> Array:
        features = jnp.zeros(x.shape[:-1] + (self._feature_dim,), dtype=x.dtype)

        if self._frozen_real_indices:
            values = jnp.take(x, self._frozen_real_index_array, axis=-1)
            features = features.at[..., self._frozen_real_slot_array].set(values)

        if self._frozen_interval_indices:
            values = jnp.take(x, self._frozen_interval_index_array, axis=-1)
            lower = self._frozen_interval_lower.astype(x.dtype)
            width = self._frozen_interval_width.astype(x.dtype)
            values = 2.0 * (values - lower) / width - 1.0
            features = features.at[..., self._frozen_interval_slot_array].set(values)

        if self._frozen_circular_indices:
            values = jnp.take(x, self._frozen_circular_index_array, axis=-1)
            lower = self._frozen_circular_lower.astype(x.dtype)
            width = self._frozen_circular_width.astype(x.dtype)
            freqs = self._frozen_circular_freqs.astype(x.dtype)
            phase = 2.0 * jnp.pi * (values - lower) / width
            angles = phase[..., :, None] * freqs
            sin_cos = jnp.stack((jnp.sin(angles), jnp.cos(angles)), axis=-1)
            values = sin_cos.reshape(
                x.shape[:-1] + (self._frozen_circular_feature_count,)
            )
            features = features.at[..., self._frozen_circular_slot_array].set(values)

        return features

    def _theta_group(
        self,
        theta: Array,
        boundary: str,
        g_value: Array | None,
    ) -> tuple[Array, Array, Array]:
        K = self.num_bins
        if boundary == "linear_tails":
            count = len(self._linear_indices)
            widths = jnp.take(theta, self._linear_width_positions, axis=-1)
            heights = jnp.take(theta, self._linear_height_positions, axis=-1)
            derivatives = jnp.take(theta, self._linear_derivative_positions, axis=-1)
        else:
            count = len(self._circular_indices)
            widths = jnp.take(theta, self._circular_width_positions, axis=-1)
            heights = jnp.take(theta, self._circular_height_positions, axis=-1)
            derivatives = jnp.take(theta, self._circular_derivative_positions, axis=-1)

        if g_value is not None:
            g = g_value[..., None, None]
            identity = identity_spline_bias(
                count,
                K,
                self.min_derivative,
                self.max_derivative,
                dtype=theta.dtype,
                boundary_slopes=boundary,
            ).reshape((count, -1))
            widths = g * widths
            heights = g * heights
            derivatives = (1.0 - g) * identity[:, 2 * K :] + g * derivatives

        return widths, heights, derivatives

    def _spline_groups(
        self,
        mlp_params: Any,
        x: Array,
        context: Array | None,
        g_value: Array | None,
    ) -> dict[str, tuple[Array, Array, Array]]:
        features = self._conditioner_features(x)
        theta = self.conditioner.apply({"params": mlp_params}, features, context)
        if theta.shape[-1] != self._out_dim:
            raise ValueError(
                f"ProductSplineCoupling: conditioner output has wrong size. "
                f"Expected {self._out_dim}, got {theta.shape[-1]}."
            )
        result = {}
        if self._linear_indices:
            result["linear_tails"] = self._theta_group(theta, "linear_tails", g_value)
        if self._circular_indices:
            result["circular"] = self._theta_group(theta, "circular", g_value)
        return result

    def _forward_or_inverse(
        self,
        params: Any,
        x: Array,
        context: Array | None,
        inverse: bool,
        g_value: Array | None,
    ) -> Tuple[Array, Array]:
        self._check_x(x)
        mlp_params = self._conditioner_params(params)
        groups = self._spline_groups(mlp_params, x, context, g_value)
        outputs = x
        log_det_total = jnp.zeros(x.shape[:-1], dtype=x.dtype)

        if self._linear_indices:
            chunk = groups["linear_tails"]
            x_group = jnp.take(x, self._linear_index_array, axis=-1)
            u_group = self._to_linear_canonical(x_group)
            y_group, ld_group = rational_quadratic_spline(
                inputs=u_group,
                unnormalized_widths=chunk[0],
                unnormalized_heights=chunk[1],
                unnormalized_derivatives=chunk[2],
                tail_bound=self.tail_bound,
                min_bin_width=self.min_bin_width,
                min_bin_height=self.min_bin_height,
                min_derivative=self.min_derivative,
                max_derivative=self.max_derivative,
                inverse=inverse,
                boundary_slopes="linear_tails",
            )
            outputs = outputs.at[..., self._linear_index_array].set(
                self._from_linear_canonical(y_group)
            )
            log_det_total = log_det_total + jnp.sum(ld_group, axis=-1)

        if self._circular_indices:
            chunk = groups["circular"]
            x_group = jnp.take(x, self._circular_index_array, axis=-1)
            u_group = self._to_circular_canonical(x_group)
            y_group, ld_group = rational_quadratic_spline(
                inputs=u_group,
                unnormalized_widths=chunk[0],
                unnormalized_heights=chunk[1],
                unnormalized_derivatives=chunk[2],
                tail_bound=self.tail_bound,
                min_bin_width=self.min_bin_width,
                min_bin_height=self.min_bin_height,
                min_derivative=self.min_derivative,
                max_derivative=self.max_derivative,
                inverse=inverse,
                boundary_slopes="circular",
            )
            outputs = outputs.at[..., self._circular_index_array].set(
                self._from_circular_canonical(y_group)
            )
            log_det_total = log_det_total + jnp.sum(ld_group, axis=-1)

        return outputs, log_det_total

    def forward(
        self,
        params: Any,
        x: Array,
        context: Array | None = None,
        g_value: Array | None = None,
    ) -> Tuple[Array, Array]:
        return self._forward_or_inverse(params, x, context, inverse=False, g_value=g_value)

    def inverse(
        self,
        params: Any,
        y: Array,
        context: Array | None = None,
        g_value: Array | None = None,
    ) -> Tuple[Array, Array]:
        return self._forward_or_inverse(params, y, context, inverse=True, g_value=g_value)

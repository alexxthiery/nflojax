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
        object.__setattr__(
            self,
            "_transformed_indices",
            tuple(i for i, frozen in enumerate(mask_tuple) if not frozen),
        )
        object.__setattr__(
            self,
            "_feature_dim",
            self.domain.conditioner_feature_dim(mask, self.circular_n_freq),
        )
        object.__setattr__(
            self,
            "_out_dim",
            self.domain.required_out_dim(mask, self.num_bins),
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

    def _to_canonical(self, x_i: Array, domain: Any) -> Array:
        if domain.kind == "real":
            return x_i
        lower = jnp.asarray(domain.lower, dtype=x_i.dtype)
        upper = jnp.asarray(domain.upper, dtype=x_i.dtype)
        return 2.0 * self.tail_bound * (x_i - lower) / (upper - lower) - self.tail_bound

    def _from_canonical(self, u_i: Array, domain: Any) -> Array:
        if domain.kind == "real":
            return u_i
        lower = jnp.asarray(domain.lower, dtype=u_i.dtype)
        upper = jnp.asarray(domain.upper, dtype=u_i.dtype)
        x_i = lower + (u_i + self.tail_bound) * (upper - lower) / (2.0 * self.tail_bound)
        if domain.kind == "circular":
            x_i = jnp.mod(x_i - lower, upper - lower) + lower
        return x_i

    def _spline_chunks(
        self,
        mlp_params: Any,
        x: Array,
        context: Array | None,
        g_value: Array | None,
    ) -> list[tuple[Array, Array, Array]]:
        features = self.domain.conditioner_features(
            x,
            self.mask,
            circular_n_freq=self.circular_n_freq,
        )
        theta = self.conditioner.apply({"params": mlp_params}, features, context)
        if theta.shape[-1] != self._out_dim:
            raise ValueError(
                f"ProductSplineCoupling: conditioner output has wrong size. "
                f"Expected {self._out_dim}, got {theta.shape[-1]}."
            )
        chunks = []
        offset = 0
        K = self.num_bins
        for i in self._transformed_indices:
            domain = self.domain.domains[i]
            boundary = "circular" if domain.kind == "circular" else "linear_tails"
            size = _params_per_scalar(K, boundary)
            theta_i = theta[..., offset : offset + size]
            offset += size
            widths = theta_i[..., :K]
            heights = theta_i[..., K : 2 * K]
            derivatives = theta_i[..., 2 * K :]
            if g_value is not None:
                g = g_value[..., None]
                identity = identity_spline_bias(
                    1,
                    K,
                    self.min_derivative,
                    self.max_derivative,
                    dtype=theta_i.dtype,
                    boundary_slopes=boundary,
                )
                widths = g * widths
                heights = g * heights
                derivatives = (1.0 - g) * identity[2 * K :] + g * derivatives
            chunks.append((widths, heights, derivatives))
        return chunks

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
        chunks = self._spline_chunks(mlp_params, x, context, g_value)
        outputs = [x[..., i] for i in range(self.domain.dim)]
        log_det_total = jnp.zeros(x.shape[:-1], dtype=x.dtype)
        for chunk, i in zip(chunks, self._transformed_indices):
            domain = self.domain.domains[i]
            boundary = "circular" if domain.kind == "circular" else "linear_tails"
            u_i = self._to_canonical(x[..., i], domain)
            y_i, ld_i = rational_quadratic_spline(
                inputs=u_i,
                unnormalized_widths=chunk[0],
                unnormalized_heights=chunk[1],
                unnormalized_derivatives=chunk[2],
                tail_bound=self.tail_bound,
                min_bin_width=self.min_bin_width,
                min_bin_height=self.min_bin_height,
                min_derivative=self.min_derivative,
                max_derivative=self.max_derivative,
                inverse=inverse,
                boundary_slopes=boundary,
            )
            outputs[i] = self._from_canonical(y_i, domain)
            log_det_total = log_det_total + ld_i
        return jnp.stack(outputs, axis=-1), log_det_total

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

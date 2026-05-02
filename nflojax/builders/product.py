from __future__ import annotations

from typing import Any, Callable, Tuple

import jax

from ..distributions import ProductBase
from ..domains import ProductDomain
from ..flows import Bijection, Flow
from ..nets import Array, PRNGKey
from ..transforms import (
    CircularCoordinateShift,
    CompositeTransform,
    ProductSplineCoupling,
)
from .assembly import make_alternating_mask
from .common import (
    _assemble_builder_output,
    _resolve_base,
    _validate_context_options,
    _validate_positive_int,
    _validate_spline_floors,
)


def build_product_spline_flow(
    key: PRNGKey,
    *,
    domain: ProductDomain,
    num_layers: int,
    hidden_dim: int,
    n_hidden_layers: int,
    context_dim: int = 0,
    num_bins: int = 8,
    tail_bound: float = 5.0,
    min_bin_width: float = 1e-2,
    min_bin_height: float = 1e-2,
    min_derivative: float = 1e-2,
    max_derivative: float = 10.0,
    activation: Callable[[Array], Array] = jax.nn.tanh,
    res_scale: float = 0.1,
    circular_n_freq: int = 1,
    use_circular_shift: bool = True,
    base_dist: Any | None = None,
    base_params: Any | None = None,
    return_transform_only: bool = False,
    identity_gate: Callable[[Array], Array] | None = None,
) -> Tuple[Flow | Bijection, Any]:
    """Build a flat spline flow on a product of scalar coordinate domains.

    The event shape is ``(domain.dim,)``. Real coordinates are unconstrained,
    interval coordinates remain inside their bounds, and circular coordinates
    are wrapped to their configured interval. No chemistry or target-specific
    logic is encoded here; the domain only describes coordinate topology. The
    shipped conditioner path uses ``ProductDomain.conditioner_features`` as its
    default MLP feature map.
    """

    if not isinstance(domain, ProductDomain):
        raise TypeError(
            f"build_product_spline_flow: domain must be a ProductDomain, "
            f"got {type(domain).__name__}."
        )
    dim = domain.dim
    if dim < 2:
        raise ValueError("build_product_spline_flow: domain.dim must be at least 2.")
    _validate_positive_int("build_product_spline_flow", "num_layers", num_layers)
    _validate_context_options(
        "build_product_spline_flow",
        context_dim=context_dim,
        identity_gate=identity_gate,
    )
    _validate_spline_floors(
        "build_product_spline_flow",
        num_bins,
        tail_bound,
        min_bin_width,
        min_bin_height,
        min_derivative,
        max_derivative,
    )

    base = None
    base_params_resolved = None
    if not return_transform_only:
        default_base = ProductBase(domain) if base_dist is None else base_dist
        base, base_params_resolved = _resolve_base(
            "build_product_spline_flow",
            default_base,
            base_params,
            domain.event_shape,
        )

    blocks = []
    block_params = []
    n_shift_blocks = num_layers if (use_circular_shift and domain.has_circular) else 0
    keys = jax.random.split(key, num_layers + n_shift_blocks)
    key_idx = 0
    parity = 0
    for _ in range(num_layers):
        if use_circular_shift and domain.has_circular:
            shift, shift_params = CircularCoordinateShift.create(keys[key_idx], domain)
            blocks.append(shift)
            block_params.append(shift_params)
            key_idx += 1

        mask = make_alternating_mask(dim, parity)
        parity = 1 - parity
        coupling, coupling_params = ProductSplineCoupling.create(
            keys[key_idx],
            domain=domain,
            mask=mask,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            context_dim=context_dim,
            num_bins=num_bins,
            tail_bound=tail_bound,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
            max_derivative=max_derivative,
            activation=activation,
            res_scale=res_scale,
            circular_n_freq=circular_n_freq,
        )
        blocks.append(coupling)
        block_params.append(coupling_params)
        key_idx += 1

    transform = CompositeTransform(blocks=blocks)

    return _assemble_builder_output(
        transform=transform,
        block_params=block_params,
        return_transform_only=return_transform_only,
        identity_gate=identity_gate,
        base=base,
        base_params=base_params_resolved,
    )

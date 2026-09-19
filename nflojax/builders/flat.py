from __future__ import annotations

from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp

from ..flows import Bijection, Flow
from ..nets import init_resnet, Array, PRNGKey
from ..transforms import (
    AffineCoupling,
    CompositeTransform,
    LinearTransform,
    LoftTransform,
    OrthogonalTransform,
    Permutation,
    SplineCoupling,
)
from .assembly import analyze_mask_coverage, make_alternating_mask
from .common import (
    _assemble_builder_output,
    _resolve_flat_base,
    _validate_builder_args,
    _validate_spline_floors,
)


def _build_coupling_flow(
    key: PRNGKey,
    dim: int,
    num_layers: int,
    coupling_factory: Callable,
    *,
    func_name: str,
    context_dim: int,
    context_extractor_hidden_dim: int,
    context_extractor_n_layers: int,
    context_feature_dim: int | None,
    activation: Callable[[Array], Array],
    res_scale: float,
    use_permutation: bool,
    use_linear: bool,
    use_orthogonal: bool,
    orthogonal_scale: float,
    use_loft: bool,
    loft_tau: float,
    trainable_base: bool,
    base_dist: Any | None,
    base_params: Any | None,
    return_transform_only: bool,
    identity_gate: Callable[[Array], Array] | None,
) -> Tuple[Flow | Bijection, Any]:
    """
    Shared assembly logic for coupling-based flows.

    Arguments:
      coupling_factory: Callable(key, mask, effective_context_dim) -> (coupling, params).
          Encapsulates coupling-specific creation logic (AffineCoupling vs SplineCoupling).
          effective_context_dim is the context dimension after optional feature extraction.
      All other arguments match build_realnvp / build_spline_realnvp.
    """
    # Context feature extractor (optional)
    key, fe_key = jax.random.split(key)
    if context_extractor_hidden_dim > 0:
        effective_context_dim = context_feature_dim if context_feature_dim is not None else context_dim
        feature_extractor, fe_params = init_resnet(
            fe_key,
            in_dim=context_dim,
            hidden_dim=context_extractor_hidden_dim,
            out_dim=effective_context_dim,
            n_hidden_layers=context_extractor_n_layers,
            activation=activation,
            res_scale=res_scale,
        )
    else:
        feature_extractor, fe_params, effective_context_dim = None, None, context_dim

    # Base distribution (skip if only returning transform)
    base = None
    base_params_resolved = None
    if not return_transform_only:
        base, base_params_resolved = _resolve_flat_base(
            func_name,
            dim,
            trainable_base,
            base_dist,
            base_params,
        )

    # Build transform blocks
    keys = jax.random.split(key, num_layers)
    blocks = []
    block_params = []

    if use_linear:
        key, lin_key = jax.random.split(key)
        lin_block, lin_params = LinearTransform.create(lin_key, dim=dim)
        blocks.append(lin_block)
        block_params.append(lin_params)

    parity = 0
    for layer_idx in range(num_layers):
        mask = make_alternating_mask(dim, parity)
        parity = 1 - parity

        coupling, coupling_params = coupling_factory(
            keys[layer_idx], mask, effective_context_dim,
        )
        blocks.append(coupling)
        block_params.append(coupling_params)

        if use_permutation and layer_idx != num_layers - 1:
            perm = jnp.arange(dim - 1, -1, -1)
            perm_block, perm_params = Permutation.create(keys[layer_idx], perm=perm)
            blocks.append(perm_block)
            block_params.append(perm_params)

    if use_orthogonal:
        # After the couplings (data side), so coordinate-wise structure can be rotated.
        orth_block, orth_params = OrthogonalTransform.create(key, dim=dim, scale=orthogonal_scale)
        blocks.append(orth_block)
        block_params.append(orth_params)

    if use_loft:
        key, loft_key = jax.random.split(key)
        loft_block, loft_params = LoftTransform.create(loft_key, dim=dim, tau=loft_tau)
        blocks.append(loft_block)
        block_params.append(loft_params)

    transform = CompositeTransform(blocks=blocks)
    analyze_mask_coverage(blocks, dim)

    return _assemble_builder_output(
        transform=transform,
        block_params=block_params,
        return_transform_only=return_transform_only,
        identity_gate=identity_gate,
        base=base,
        base_params=base_params_resolved,
        feature_extractor=feature_extractor,
        feature_extractor_params=fe_params,
    )


# ====================================================================
# RealNVP builder
# ====================================================================
def build_realnvp(
    key: PRNGKey,
    dim: int,
    num_layers: int,
    hidden_dim: int,
    n_hidden_layers: int,
    *,
    context_dim: int = 0,
    context_extractor_hidden_dim: int = 0,
    context_extractor_n_layers: int = 2,
    context_feature_dim: int | None = None,
    max_log_scale: float = 5.0,
    res_scale: float = 0.1,
    use_permutation: bool = False,
    use_linear: bool = False,
    use_orthogonal: bool = False,
    orthogonal_scale: float = 10.0,
    use_loft: bool = True,
    trainable_base: bool = False,
    base_dist: Any | None = None,
    base_params: Any | None = None,
    activation: Callable[[Array], Array] = jax.nn.tanh,
    loft_tau: float = 1000.0,
    return_transform_only: bool = False,
    identity_gate: Callable[[Array], Array] | None = None,
) -> Tuple[Flow | Bijection, Any]:
    """Build a flat affine-coupling flow on events of shape ``(dim,)``.

    Args:
        key: PRNG key used to initialize conditioner and optional mixing layers.
        dim: Flat event dimension. Must be positive.
        num_layers: Number of affine coupling layers. Must be positive.
        hidden_dim: Conditioner MLP hidden width.
        n_hidden_layers: Number of conditioner MLP residual blocks.
        context_dim: Optional conditioning width. ``0`` means unconditional.
        context_extractor_hidden_dim: Optional ResNet feature-extractor width.
        context_extractor_n_layers: Number of extractor residual blocks.
        context_feature_dim: Extractor output width. Defaults to ``context_dim``.
        max_log_scale: Bound on affine log-scale.
        res_scale: Conditioner residual scale.
        use_permutation: Insert fixed reverse permutations between couplings.
        use_linear: Prepend a learnable linear transform.
        use_orthogonal: Add a learnable rotation (``OrthogonalTransform``)
            after the couplings, before LOFT. Prefer it to ``use_linear`` when
            the target's structure lies along rotated axes.
        orthogonal_scale: Its generator scale; 10 by default (scale 1 can
            stall on a near-Gaussian plateau, see ``OrthogonalTransform``).
        use_loft: Append a LOFT tail stabilizer.
        trainable_base: Use ``DiagNormal((dim,))`` when no custom base is given.
        base_dist: Optional custom base. If it exposes ``event_shape``, it must
            equal ``(dim,)``. Ignored when ``return_transform_only=True``.
        base_params: Optional params for ``base_dist``.
        activation: Conditioner activation.
        loft_tau: LOFT threshold.
        return_transform_only: Return a ``Bijection`` and omit base validation.
        identity_gate: Optional raw-context gate. Incompatible with
            ``use_permutation=True``.

    Returns:
        ``(Flow, params)`` with params keys ``"base"`` and ``"transform"``, or
        ``(Bijection, params)`` with key ``"transform"`` when
        ``return_transform_only=True``. Feature-extractor params add a
        ``"feature_extractor"`` key.
    """
    _validate_builder_args(
        "build_realnvp", dim, num_layers, context_dim,
        identity_gate, use_permutation, context_extractor_hidden_dim,
    )

    def coupling_factory(layer_key, mask, effective_context_dim):
        return AffineCoupling.create(
            layer_key, dim=dim, mask=mask,
            hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers,
            context_dim=effective_context_dim, activation=activation,
            res_scale=res_scale, max_log_scale=max_log_scale,
        )

    return _build_coupling_flow(
        key, dim, num_layers, coupling_factory,
        func_name="build_realnvp",
        context_dim=context_dim,
        context_extractor_hidden_dim=context_extractor_hidden_dim,
        context_extractor_n_layers=context_extractor_n_layers,
        context_feature_dim=context_feature_dim,
        activation=activation, res_scale=res_scale,
        use_permutation=use_permutation, use_linear=use_linear,
        use_orthogonal=use_orthogonal, orthogonal_scale=orthogonal_scale,
        use_loft=use_loft, loft_tau=loft_tau,
        trainable_base=trainable_base, base_dist=base_dist,
        base_params=base_params, return_transform_only=return_transform_only,
        identity_gate=identity_gate,
    )


# ====================================================================
# Spline RealNVP builder
# ====================================================================
def build_spline_realnvp(
    key: PRNGKey,
    dim: int,
    num_layers: int,
    hidden_dim: int,
    n_hidden_layers: int,
    *,
    context_dim: int = 0,
    context_extractor_hidden_dim: int = 0,
    context_extractor_n_layers: int = 2,
    context_feature_dim: int | None = None,
    num_bins: int = 8,
    tail_bound: float = 5.0,
    min_bin_width: float = 1e-2,
    min_bin_height: float = 1e-2,
    min_derivative: float = 1e-2,
    max_derivative: float = 10.0,
    res_scale: float = 0.1,
    use_permutation: bool = False,
    use_linear: bool = False,
    use_orthogonal: bool = False,
    orthogonal_scale: float = 10.0,
    use_loft: bool = True,
    trainable_base: bool = False,
    base_dist: Any | None = None,
    base_params: Any | None = None,
    activation: Callable[[Array], Array] = jax.nn.tanh,
    loft_tau: float = 1000.0,
    return_transform_only: bool = False,
    identity_gate: Callable[[Array], Array] | None = None,
) -> Tuple[Flow | Bijection, Any]:
    """Build a flat rational-quadratic-spline flow on events ``(dim,)``.

    Args:
        key: PRNG key used to initialize conditioner and optional mixing layers.
        dim: Flat event dimension. Must be positive.
        num_layers: Number of spline coupling layers. Must be positive.
        hidden_dim: Conditioner MLP hidden width.
        n_hidden_layers: Number of conditioner MLP residual blocks.
        context_dim: Optional conditioning width. ``0`` means unconditional.
        context_extractor_hidden_dim: Optional ResNet feature-extractor width.
        context_extractor_n_layers: Number of extractor residual blocks.
        context_feature_dim: Extractor output width. Defaults to ``context_dim``.
        num_bins: Number of spline bins.
        tail_bound: Positive spline half-width.
        min_bin_width: Bin-width simplex floor.
        min_bin_height: Bin-height simplex floor.
        min_derivative: Positive derivative floor.
        max_derivative: Positive derivative cap, at least ``min_derivative``.
        res_scale: Conditioner residual scale.
        use_permutation: Insert fixed reverse permutations between couplings.
        use_linear: Prepend a learnable linear transform.
        use_orthogonal: Add a learnable rotation (``OrthogonalTransform``)
            after the couplings, before LOFT. Prefer it to ``use_linear`` when
            the target's structure lies along rotated axes.
        orthogonal_scale: Its generator scale; 10 by default (scale 1 can
            stall on a near-Gaussian plateau, see ``OrthogonalTransform``).
        use_loft: Append a LOFT tail stabilizer.
        trainable_base: Use ``DiagNormal((dim,))`` when no custom base is given.
        base_dist: Optional custom base. If it exposes ``event_shape``, it must
            equal ``(dim,)``. Ignored when ``return_transform_only=True``.
        base_params: Optional params for ``base_dist``.
        activation: Conditioner activation.
        loft_tau: LOFT threshold.
        return_transform_only: Return a ``Bijection`` and omit base validation.
        identity_gate: Optional raw-context gate. Incompatible with
            ``use_permutation=True``.

    Returns:
        ``(Flow, params)`` with params keys ``"base"`` and ``"transform"``, or
        ``(Bijection, params)`` with key ``"transform"`` when
        ``return_transform_only=True``. Feature-extractor params add a
        ``"feature_extractor"`` key.
    """
    _validate_builder_args(
        "build_spline_realnvp", dim, num_layers, context_dim,
        identity_gate, use_permutation, context_extractor_hidden_dim,
    )
    _validate_spline_floors(
        "build_spline_realnvp",
        num_bins,
        tail_bound,
        min_bin_width,
        min_bin_height,
        min_derivative,
        max_derivative,
    )

    def coupling_factory(layer_key, mask, effective_context_dim):
        return SplineCoupling.create(
            layer_key, dim=dim, mask=mask,
            hidden_dim=hidden_dim, n_hidden_layers=n_hidden_layers,
            context_dim=effective_context_dim, num_bins=num_bins,
            tail_bound=tail_bound, min_bin_width=min_bin_width,
            min_bin_height=min_bin_height, min_derivative=min_derivative,
            max_derivative=max_derivative, activation=activation,
            res_scale=res_scale,
        )

    return _build_coupling_flow(
        key, dim, num_layers, coupling_factory,
        func_name="build_spline_realnvp",
        context_dim=context_dim,
        context_extractor_hidden_dim=context_extractor_hidden_dim,
        context_extractor_n_layers=context_extractor_n_layers,
        context_feature_dim=context_feature_dim,
        activation=activation, res_scale=res_scale,
        use_permutation=use_permutation, use_linear=use_linear,
        use_orthogonal=use_orthogonal, orthogonal_scale=orthogonal_scale,
        use_loft=use_loft, loft_tau=loft_tau,
        trainable_base=trainable_base, base_dist=base_dist,
        base_params=base_params, return_transform_only=return_transform_only,
        identity_gate=identity_gate,
    )


# ====================================================================
# Product-domain spline builder
# ====================================================================

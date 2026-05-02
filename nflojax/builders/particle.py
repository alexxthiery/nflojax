from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import jax

from ..flows import Bijection, Flow
from ..geometry import Geometry
from ..nets import PRNGKey
from ..transforms import (
    CircularShift,
    CoMProjection,
    CompositeTransform,
    Rescale,
    SplitCoupling,
)
from ..transforms.common import _params_per_scalar
from .common import (
    _assemble_builder_output,
    _resolve_base,
    _validate_base_event_shape,
    _validate_positive_int,
)


# ====================================================================
# Particle-flow builder
# ====================================================================
# A `CompositeTransform`-compatible shim for `CoMProjection` used only
# when `use_com_shift=True`. `CompositeTransform` applies `block.forward`
# in order and `block.inverse` in reverse, but for the CoM pattern the
# base lives on the reduced (N-1, d) subspace while samples must come
# out in ambient (N, d). That means the "base -> data" direction needs
# `CoMProjection.inverse` (the expansion), not `.forward` (the
# reduction). This shim swaps the two so the expansion is what
# `transform.forward` sees at the tail of the block list.
@dataclass
class _CoMEmbed:
    """Direction-flipped view of `CoMProjection` (private to the particle builder).

    `forward: (..., N-1, d) -> (..., N, d)` embeds into the zero-CoM subspace.
    `inverse: (..., N, d) -> (..., N-1, d)` projects ambient onto the subspace.

    Not part of the public API; lives here because the particle-flow
    builder is the only place that needs this specific composition.
    """
    event_axis: int = -2

    def __post_init__(self):
        self._proj = CoMProjection(event_axis=self.event_axis)

    def forward(self, params, x, context=None):
        return self._proj.inverse(params, x, context)

    def inverse(self, params, y, context=None):
        return self._proj.forward(params, y, context)

    def init_params(self, key: PRNGKey, context_dim: int = 0) -> dict:
        del key, context_dim
        return {}


def build_particle_flow(
    key: PRNGKey,
    *,
    geometry: Geometry,
    event_shape: Tuple[int, int],
    num_layers: int,
    conditioner: Callable[..., Any],
    base_dist: Any = None,
    base_params: Any = None,
    num_bins: int = 8,
    tail_bound: float = 5.0,
    boundary_slopes: str = "circular",
    use_com_shift: bool = False,
    return_transform_only: bool = False,
) -> Tuple[Flow | Bijection, Dict]:
    """
    Build the canonical particle-flow topology (DM / bgmat-style).

    Architecture (for ``event_shape = (N, d)``, with ``N_eff = N - 1`` if
    ``use_com_shift`` else ``N``)::

        Rescale(geometry -> [-tail_bound, tail_bound], event_shape=(N_eff, d))
        for _ in range(num_layers):
            SplitCoupling(swap=False, boundary_slopes, flatten_input=False)
            SplitCoupling(swap=True,  boundary_slopes, flatten_input=False)
            CircularShift(canonical cube)
        [optional] _CoMEmbed  # base on (N-1, d); embed to ambient (N, d)

    The canonical cube for ``CircularShift`` is
    ``[-tail_bound, tail_bound]^d``, matching the rescaled domain.
    Particle-axis coverage is handled by alternating ``swap``, so no
    per-layer ``Permutation`` is needed. If you want additional particle
    mixing, use ``LatticeBase(permute=True)`` as the base distribution or
    the augmented-coupling pattern (see ``EXTENDING.md``).

    Conditioner contract
    --------------------
    ``conditioner`` is a **keyword-only factory**, called once per
    coupling layer, with three kwargs::

        required_out_dim = N_transformed * d * params_per_scalar
        out_per_particle = d * params_per_scalar
        n_frozen         = N_frozen

    The factory returns a fresh ``flax.linen.Module``. Canonical usages::

        from nflojax.nets import DeepSets, Transformer, GNN

        # Flat-output conditioner (DeepSets): uses required_out_dim.
        lambda *, required_out_dim, **_: DeepSets(
            phi_hidden=(64, 64), rho_hidden=(64,),
            out_dim=required_out_dim,
        )

        # Per-token conditioner (Transformer / GNN): uses out_per_particle.
        lambda *, out_per_particle, **_: Transformer(
            num_layers=2, num_heads=4, embed_dim=64,
            out_per_particle=out_per_particle,
        )

    Per-token conditioners require ``N_frozen == N_transformed`` (a
    half-half split along the particle axis). ``SplitCoupling.init_params``
    validates the conditioner's output size and raises a diagnostic error
    on mismatch.

    Identity at init
    ----------------
    ``SplitCoupling._patch_dense_out`` zero-initialises each conditioner's
    ``dense_out`` so the inner spline composition is identity at init.
    The only non-zero forward log-det at init is the (constant)
    ``Rescale`` contribution.

    Arguments
    ---------
    key: PRNG key for all layer inits.
    geometry: Physical box spec. ``event_shape[-1]`` must equal
        ``geometry.d``.
    event_shape: ``(N, d)`` -- particle count and coord dim.
    num_layers: Number of coupling-pair blocks (swap-False / swap-True).
    conditioner: Keyword-only factory, see contract above.
    base_dist: Mandatory base distribution unless
        ``return_transform_only=True``. For ``use_com_shift=False`` its
        event shape must be ``(N, d)`` (e.g. ``UniformBox(geometry,
        event_shape=(N, d))``). For ``use_com_shift=True`` it must be
        ``(N-1, d)`` (reduced-subspace base).
    base_params: Optional; defaults to ``base_dist.init_params()``.
    num_bins: Spline bin count K. Default 8.
    tail_bound: Spline canonical half-width. Default 5.0.
    boundary_slopes: ``'circular'`` (default) or ``'linear_tails'``.
    use_com_shift: If True, appends a ``_CoMEmbed`` shim so the flow
        acts on the zero-CoM subspace.
    return_transform_only: If True, return ``(Bijection,
        params_without_base)`` instead of ``(Flow, params_with_base)``.

    Returns
    -------
    ``(flow, params)`` where params has keys ``"base"`` and
    ``"transform"`` (or just ``"transform"`` if
    ``return_transform_only=True``).
    """
    if not isinstance(geometry, Geometry):
        raise TypeError(
            f"build_particle_flow: geometry must be a Geometry, got "
            f"{type(geometry).__name__}."
        )
    if len(event_shape) != 2:
        raise ValueError(
            f"build_particle_flow: event_shape must be (N, d); got {event_shape}."
        )
    N, d = int(event_shape[0]), int(event_shape[1])
    if d != geometry.d:
        raise ValueError(
            f"build_particle_flow: event_shape[-1]={d} must equal "
            f"geometry.d={geometry.d}."
        )
    if num_layers <= 0:
        raise ValueError(
            f"build_particle_flow: num_layers must be positive, got {num_layers}."
        )
    _validate_positive_int("build_particle_flow", "num_bins", num_bins)
    if tail_bound <= 0:
        raise ValueError(
            f"build_particle_flow: tail_bound must be positive, got {tail_bound}."
        )
    if N < 2:
        raise ValueError(f"build_particle_flow: N must be >= 2, got {N}.")
    if use_com_shift and N < 3:
        raise ValueError(
            f"build_particle_flow: use_com_shift=True requires N >= 3, got {N} "
            "(one particle is consumed by the CoM projection)."
        )
    if not return_transform_only and base_dist is None:
        raise ValueError(
            "build_particle_flow: base_dist is required when "
            "return_transform_only=False."
        )

    N_eff = N - 1 if use_com_shift else N
    expected_base_shape = (N_eff, d)
    if not return_transform_only:
        _validate_base_event_shape(
            "build_particle_flow",
            base_dist,
            expected_base_shape,
        )
    split_index = N_eff // 2
    params_per_scalar = _params_per_scalar(num_bins, boundary_slopes)
    out_per_particle = d * params_per_scalar

    # The inner stack operates in the rescaled cube [-tail_bound, tail_bound]^d;
    # CircularShift below wraps around that cube, not the physical geometry.
    spline_geometry = Geometry.cubic(
        d=d, side=2.0 * tail_bound, lower=-float(tail_bound),
    )

    # Pre-split the key: one sub-key per SplitCoupling init plus one per
    # CircularShift. Rescale and _CoMEmbed are parameter-free, so they
    # don't consume randomness.
    n_couplings = 2 * num_layers
    n_shifts = num_layers
    keys = jax.random.split(key, n_couplings + n_shifts)

    blocks: List[Any] = []
    params_list: List[Any] = []

    rs_block, rs_params = Rescale.create(
        keys[0],  # unused; Rescale is param-free
        geometry=geometry,
        target=(-float(tail_bound), float(tail_bound)),
        event_shape=(N_eff, d),
    )
    blocks.append(rs_block)
    params_list.append(rs_params)

    key_idx = 0
    for _layer in range(num_layers):
        for swap in (False, True):
            # With an asymmetric split (odd N_eff), swap flips the frozen /
            # transformed sizes, so required_out_dim depends on swap. For
            # even N_eff both values are N_eff // 2 and this is a no-op.
            n_frozen = (N_eff - split_index) if swap else split_index
            n_transformed = N_eff - n_frozen
            required_out_dim = n_transformed * d * params_per_scalar
            cond = conditioner(
                required_out_dim=required_out_dim,
                out_per_particle=out_per_particle,
                n_frozen=n_frozen,
            )
            coupling = SplitCoupling(
                event_shape=(N_eff, d),
                split_axis=-2,
                split_index=split_index,
                event_ndims=2,
                conditioner=cond,
                swap=swap,
                num_bins=num_bins,
                tail_bound=tail_bound,
                boundary_slopes=boundary_slopes,
                flatten_input=False,
            )
            blocks.append(coupling)
            params_list.append(coupling.init_params(keys[key_idx]))
            key_idx += 1

        cs_block, cs_params = CircularShift.create(
            keys[key_idx], geometry=spline_geometry,
        )
        blocks.append(cs_block)
        params_list.append(cs_params)
        key_idx += 1

    if use_com_shift:
        # _CoMEmbed.forward expands (N-1, d) -> (N, d); appending here means
        # transform.forward ends in ambient and transform.inverse starts
        # there -- the directions Flow.sample / Flow.log_prob expect.
        embed = _CoMEmbed(event_axis=-2)
        blocks.append(embed)
        params_list.append(embed.init_params(keys[0]))

    transform = CompositeTransform(blocks=blocks)

    base = None
    base_params_resolved = None
    if not return_transform_only:
        base, base_params_resolved = _resolve_base(
            "build_particle_flow",
            base_dist,
            base_params,
            expected_base_shape,
        )

    return _assemble_builder_output(
        transform=transform,
        block_params=params_list,
        return_transform_only=return_transform_only,
        identity_gate=None,
        base=base,
        base_params=base_params_resolved,
    )

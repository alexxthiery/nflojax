"""Augmented particle flow: Pattern B promoted from a recipe to a builder.

The event is doubled to ``(2N, d)``: ``N`` physical particles followed by ``N``
auxiliary ones, and every ``SplitCoupling`` splits **across that boundary**
rather than along the particle axis. Two properties follow, and they are the
reason to prefer this topology on a strongly coupled system:

  - **each coupling conditions on a full copy of all N particles.** A
    particle-axis split conditions on half of them, which at small N is a
    severe handicap (at N=8, four particles of eight);
  - **both partitions are exactly N**, so per-token conditioners (``Transformer``,
    ``GNN``) work for odd N, which a particle-axis split cannot serve.

It is also fully ``S_N``-equivariant per half when its conditioner is, whereas a
particle-axis split is at most ``S_frozen x S_transformed``-equivariant
(EXTENDING.md, "When axis-split coupling isn't enough").

**What this builder does not decide.** What the auxiliary half *means* is the
application's: the canonical choice (bgmat) is a second copy of the same system
with a conditional target ``p(x, a) = p(x) q(a | x)`` whose conditional is
normalised, so the joint's log Z equals the physical one and free energies need
no marginalisation. A physical-marginal density (for a marginal ESS) is an
inference-time question and also belongs to the application; EXTENDING.md sketches
both. Nothing here assumes any of it.

**Do not stack this with ``CoMProjection`` or add
``CoMProjection.ambient_correction``.** Doubling the degrees of freedom keeps the
ambient dimension intact, so the density is already ambient-valid and the
correction would be counted twice (EXTENDING.md, "CoM handling").
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Tuple

import jax
import numpy as np

from nflojax.geometry import Geometry
from nflojax.transforms import CircularShift, CompositeTransform, Rescale, SplitCoupling
from nflojax.transforms.common import _params_per_scalar, _validate_boundary_slopes
from nflojax.flows import Bijection, Flow
from nflojax.builders.common import (
    _assemble_builder_output,
    _resolve_base,
    _validate_base_event_shape,
    _validate_positive_int,
)

PRNGKey = Any


def build_augmented_flow(
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
    feature_map: Optional[Callable[[Any], Any]] = None,
    return_transform_only: bool = False,
) -> Tuple[Flow | Bijection, Dict]:
    """
    Build the augmented (Pattern B) particle-flow topology.

    Architecture, for a **physical** ``event_shape = (N, d)`` and therefore an
    augmented event ``(2N, d)``::

        Rescale(geometry -> [-tail_bound, tail_bound], event_shape=(2N, d))
        for _ in range(num_layers):
            SplitCoupling(split_axis=-2, split_index=N, swap=False)   # moves physical
            SplitCoupling(split_axis=-2, split_index=N, swap=True)    # moves auxiliary
            CircularShift(canonical cube)

    So one layer transforms every physical particle once, conditioned on all
    ``N`` auxiliary ones, and then the reverse.

    Args:
        key: PRNG key; split internally, one sub-key per coupling and shift.
        geometry: The **physical** box, of dimension ``d``. Both halves live in
            it, so it is applied to all ``2N`` rows.
        event_shape: ``(N, d)``, the physical half. The flow's event, and the
            base's, is ``(2N, d)``.
        num_layers: Number of layers, each two couplings and one shift.
        conditioner: Keyword-only factory, called once per coupling with
            ``required_out_dim = N * d * params_per_scalar``,
            ``out_per_particle = d * params_per_scalar``, ``n_frozen = N`` and
            ``geometry`` (the spline cube, periodic where the physical box is).
            Both sizes are independent of ``swap``, unlike the particle split.
        base_dist: Base distribution on ``(2N, d)``; required unless
            ``return_transform_only``.
        base_params: Its parameters; ``None`` calls ``base_dist.init_params()``.
        num_bins: Spline bins per coordinate.
        tail_bound: Half-width of the spline cube the stack works in.
        boundary_slopes: ``"circular"`` (the default, required on a periodic box)
            or ``"linear_tails"``.
        feature_map: Optional pure function applied to the **structured** frozen
            half before the conditioner sees it (``SplitCoupling.feature_map``),
            for conditioning on something other than the raw coordinates, such
            as a lattice-site encoding. It cannot affect invertibility or the
            log-det, because the frozen half passes through untouched. A map
            that widens the last axis needs a conditioner that does not apply
            its own coordinate embedding (``circular_n_freq`` / ``geometry``),
            which expects exactly ``d`` columns: encode the coordinates inside
            the map instead.
        return_transform_only: Return a ``Bijection`` instead of a ``Flow``.

    Returns:
        ``(flow_or_bijection, params)`` with ``params = {"base", "transform"}``
        (``{"transform"}`` only, for a bijection).

    Raises:
        ValueError: on a non-positive size, ``d`` disagreeing with ``geometry``,
            ``N < 2``, a base whose event shape is not ``(2N, d)``, a missing
            base, or a periodic geometry with ``boundary_slopes="linear_tails"``
            (that target is improper; see ``build_particle_flow``).
    """
    if len(event_shape) != 2:
        raise ValueError(
            f"build_augmented_flow: event_shape must be (N, d), got {event_shape}."
        )
    n_particles, d = int(event_shape[0]), int(event_shape[1])
    if d != geometry.d:
        raise ValueError(
            f"build_augmented_flow: event_shape[-1]={d} must equal "
            f"geometry.d={geometry.d}."
        )
    if n_particles < 2:
        raise ValueError(
            f"build_augmented_flow: N must be >= 2, got {n_particles}."
        )
    if num_layers <= 0:
        raise ValueError(
            f"build_augmented_flow: num_layers must be positive, got {num_layers}."
        )
    _validate_positive_int("build_augmented_flow", "num_bins", num_bins)
    if tail_bound <= 0:
        raise ValueError(
            f"build_augmented_flow: tail_bound must be positive, got {tail_bound}."
        )
    _validate_boundary_slopes(boundary_slopes, where="build_augmented_flow")
    if boundary_slopes != "circular" and any(
        geometry.is_periodic(axis) for axis in range(geometry.d)
    ):
        raise ValueError(
            "build_augmented_flow: geometry has one or more periodic axes but "
            f"boundary_slopes={boundary_slopes!r}. A periodic target on "
            "non-circular (linear-tail) coordinates is improper: its density is "
            "invariant under per-particle box translations (x_i -> x_i + L), so "
            "it has infinitely many identical copies over the unbounded spline "
            "tails and reverse-KL training diverges to infinite entropy. Use "
            "boundary_slopes='circular' for periodic/torus geometries."
        )
    if not return_transform_only and base_dist is None:
        raise ValueError(
            "build_augmented_flow: base_dist is required when "
            "return_transform_only=False."
        )

    augmented_shape = (2 * n_particles, d)
    if not return_transform_only:
        try:
            _validate_base_event_shape(
                "build_augmented_flow", base_dist, augmented_shape,
            )
        except ValueError as exc:                     # add the Pattern B reason
            raise ValueError(
                f"{exc} The augmented flow doubles the event: with "
                f"event_shape={event_shape} (physical) the base must be on "
                f"(2N, d) = {augmented_shape}, holding the physical particles "
                f"first and the auxiliary ones second."
            ) from None

    params_per_scalar = _params_per_scalar(num_bins, boundary_slopes)
    out_per_particle = d * params_per_scalar
    # Both partitions are exactly N, for either swap: that is the whole point of
    # splitting on the physical/auxiliary boundary.
    required_out_dim = n_particles * out_per_particle

    # The stack works in the rescaled cube; CircularShift wraps around that cube,
    # not the physical box. Same convention as build_particle_flow.
    spline_geometry = Geometry.cubic(
        d=d, side=2.0 * tail_bound, lower=-float(tail_bound),
    )
    conditioner_geometry = Geometry(
        lower=np.full(d, -float(tail_bound)), upper=np.full(d, float(tail_bound)),
        periodic=geometry.periodic,
    )

    keys = jax.random.split(key, 3 * num_layers)     # 2 couplings + 1 shift per layer
    blocks: List[Any] = []
    params_list: List[Any] = []

    rs_block, rs_params = Rescale.create(
        keys[0],                                     # unused; Rescale is param-free
        geometry=geometry,
        target=(-float(tail_bound), float(tail_bound)),
        event_shape=augmented_shape,
    )
    blocks.append(rs_block)
    params_list.append(rs_params)

    key_idx = 0
    for _layer in range(num_layers):
        for swap in (False, True):
            cond = conditioner(
                required_out_dim=required_out_dim,
                out_per_particle=out_per_particle,
                n_frozen=n_particles,
                geometry=conditioner_geometry,
            )
            coupling = SplitCoupling(
                event_shape=augmented_shape,
                split_axis=-2,
                split_index=n_particles,
                event_ndims=2,
                conditioner=cond,
                swap=swap,
                num_bins=num_bins,
                tail_bound=tail_bound,
                boundary_slopes=boundary_slopes,
                flatten_input=False,
                feature_map=feature_map,
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

    transform = CompositeTransform(blocks=blocks)

    base = None
    base_params_resolved = None
    if not return_transform_only:
        base, base_params_resolved = _resolve_base(
            "build_augmented_flow", base_dist, base_params, augmented_shape,
        )

    return _assemble_builder_output(
        transform=transform,
        block_params=params_list,
        return_transform_only=return_transform_only,
        identity_gate=None,
        base=base,
        base_params=base_params_resolved,
    )

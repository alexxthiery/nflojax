# nflojax/utils/pbc.py
"""
Orthogonal periodic-box geometry helpers.

Three pure functions used by geometry-aware primitives (GNN conditioners
with neighbour lists, `LatticeBase` with minimum-image fluctuations, any
pairwise distance calculation under PBC):

- `nearest_image(dx, geometry)` — minimum-image wrap of a displacement.
- `pairwise_distance(x, geometry=None)` — pairwise distances for
  `(..., N, d)` positions, with optional PBC.
- `pairwise_distance_sq(x, geometry=None)` — squared version (avoids a
  `sqrt`; preferred for neighbour-list cutoffs and network features).

All three operate on axis-aligned orthogonal boxes only (the `Geometry`
type). Triclinic cells are out of scope (see DESIGN.md §4.8).

Not for: energy evaluation, neighbour lists with physics-specific cutoffs
that encode a force-field's range — those are application code, not flow
code.
"""
from __future__ import annotations

from typing import Optional

import jax.numpy as jnp

from ..geometry import Geometry

Array = jnp.ndarray


def nearest_image(dx: Array, geometry: Geometry) -> Array:
    """Minimum-image wrap of displacement `dx` under `geometry`'s PBC.

    For each periodic axis:
        dx_wrapped[..., j] = dx[..., j] - box[j] * round(dx[..., j] / box[j])
    which lives in `(-box[j] / 2, box[j] / 2]`. Non-periodic axes
    (per `geometry.periodic`) are passed through unchanged.

    Arguments:
        dx:       Displacement tensor of shape `(..., d)`. Last axis is
                  the coord axis; leading axes arbitrary (batch, pair, ...).
        geometry: `Geometry` with `d == dx.shape[-1]`.

    Returns:
        Array of the same shape as `dx`, wrapped on periodic axes.
    """
    if not isinstance(geometry, Geometry):
        raise TypeError(
            f"nearest_image: geometry must be a Geometry instance, "
            f"got {type(geometry).__name__}."
        )
    if dx.shape[-1] != geometry.d:
        raise ValueError(
            f"nearest_image: last axis of dx must be geometry.d={geometry.d}, "
            f"got dx.shape={dx.shape}."
        )
    box = jnp.asarray(geometry.box, dtype=dx.dtype)
    wrapped = dx - box * jnp.round(dx / box)
    if geometry.periodic is None:
        # All-periodic (the default when Geometry is constructed without
        # explicit periodic flags). No per-axis gating needed.
        return wrapped
    periodic = jnp.asarray(geometry.periodic, dtype=bool)
    return jnp.where(periodic, wrapped, dx)


def pairwise_distance_sq(
    x: Array, geometry: Optional[Geometry] = None
) -> Array:
    """Pairwise squared distances for `(..., N, d)` positions.

    Arguments:
        x:        Positions of shape `(..., N, d)`. Must have at least
                  rank 2 (one particle axis + one coord axis).
        geometry: Optional `Geometry`; when provided, displacements are
                  wrapped via `nearest_image` before squaring. `None` gives
                  plain Euclidean squared distances.

    Convention:
        Displacements are computed as `dx[..., i, j, :] = x[..., j, :] -
        x[..., i, :]`. Squared distances are symmetric, so the sign does
        not affect the return value, but it does matter for any caller
        that takes derivatives through the intermediate `dx`.

    Returns:
        Array of shape `(..., N, N)` with `d_sq[i, j]` being the squared
        distance between particles `i` and `j`. Symmetric; diagonal is 0.
    """
    if x.ndim < 2:
        raise ValueError(
            f"pairwise_distance_sq: x must be at least rank 2 with shape "
            f"`(..., N, d)`; got x.shape={x.shape}. (Did you pass `(d,)` "
            f"instead of `(N, d)`?)"
        )
    dx = x[..., None, :, :] - x[..., :, None, :]
    if geometry is not None:
        dx = nearest_image(dx, geometry)
    return jnp.sum(dx * dx, axis=-1)


def pairwise_distance(
    x: Array, geometry: Optional[Geometry] = None
) -> Array:
    """Pairwise distances for `(..., N, d)` positions.

    Thin wrapper around `pairwise_distance_sq`. Prefer the squared version
    in hot paths (neighbour-list cutoffs, MLP inputs) to avoid the `sqrt`.

    Arguments:
        x:        Positions of shape `(..., N, d)`.
        geometry: Optional `Geometry` for minimum-image wrap; `None` gives
                  Euclidean distances.

    Returns:
        Array of shape `(..., N, N)` with Euclidean distances.
    """
    return jnp.sqrt(pairwise_distance_sq(x, geometry))

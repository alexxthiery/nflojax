# nflojax/utils/lattice.py
"""
Lattice-position generators for crystalline particle bases.

Five pure functions returning `(N, 3)` numpy arrays of atom positions in
an orthorhombic box. Used by `LatticeBase` (Stage B3) to define the
mean-site positions of a Gaussian-perturbed lattice base distribution.

Shipped lattices
----------------
- `fcc(n_cells, a)`        -- face-centred cubic, 4 atoms / cubic cell.
- `diamond(n_cells, a)`    -- diamond cubic, 8 atoms / cubic cell.
- `bcc(n_cells, a)`        -- body-centred cubic, 2 atoms / cubic cell.
- `hcp(n_cells, a)`        -- hexagonal close-packed, orthorhombic
  representation, 4 atoms / cell, ideal `c/a = sqrt(8/3)`.
- `hex_ice(n_cells, a)`    -- hexagonal ice (Ice Ih), DM convention,
  8 atoms / cell, with the puckering parameter baked in.

All return numpy arrays of float64; the caller can downcast at use time.
Positions are static configuration data (not JAX-traced) — same backing
type as `nflojax.geometry.Geometry`.

Box conventions
---------------
- `n_cells`: int (uniform per axis) or tuple of length 3.
- `a`: lattice constant (scalar). The orthorhombic cell side along axis
  `i` is `a * cell_aspect[i]`. For cubic lattices `cell_aspect = (1, 1, 1)`;
  for `hcp` and `hex_ice` it is `(1, sqrt(3), sqrt(8/3))`.
- Box returned by `make_box(...)` has corner at the origin and side
  lengths `n_cells[i] * a * cell_aspect[i]`. Build a `Geometry` via
  `Geometry(lower=zeros(3), upper=box)` if you need one.

Not for
-------
- Lattice-specific physics (Madelung constants, defect generation, etc.).
- Triclinic cells. Out of scope (DESIGN.md §4.8).
"""
from __future__ import annotations

from typing import Sequence, Union

import numpy as np

Array = np.ndarray


# ----------------------------------------------------------------------
# Per-lattice unit-cell data (fractional coordinates within one cell)
# ----------------------------------------------------------------------
# Listed in the same order as `flows_for_atomic_solids` where they overlap,
# so test parity is order-independent only via sorting (see test file).

_FCC_FRAC = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
    ],
    dtype=np.float64,
)

_DIAMOND_FRAC = np.array(
    [
        # FCC sub-lattice.
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.5],
        [0.5, 0.0, 0.5],
        [0.5, 0.5, 0.0],
        # Tetrahedral interstitials at +(1/4, 1/4, 1/4).
        [0.25, 0.25, 0.25],
        [0.25, 0.75, 0.75],
        [0.75, 0.25, 0.75],
        [0.75, 0.75, 0.25],
    ],
    dtype=np.float64,
)

_BCC_FRAC = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5],
    ],
    dtype=np.float64,
)

# HCP in the orthorhombic representation:
#   cell_aspect = (1, sqrt(3), sqrt(8/3))
# 4 atoms per orthorhombic cell. Source: Ashcroft & Mermin, "Solid State
# Physics", §4 (close-packed structures); equivalent fractional positions
# at https://en.wikipedia.org/wiki/Close-packing_of_equal_spheres
# (orthorhombic-cell representation, ABAB stacking, ideal c/a = sqrt(8/3)).
_HCP_FRAC = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0],
        [0.5, 1.0 / 6.0, 0.5],
        [0.0, 2.0 / 3.0, 0.5],
    ],
    dtype=np.float64,
)

# Hexagonal ice (Ice Ih), DM convention (Wirnsberger et al. 2022,
# `flows_for_atomic_solids/models/particle_models.py:HexagonalIceLattice`).
# 8 atoms in the orthorhombic cell with cell_aspect = (1, sqrt(3), sqrt(8/3)).
# Puckering parameter `a = 6 * 0.0625` shifts the z-coordinates of the
# upper / lower halves of each layer.
_HEX_ICE_DM_PUCKER = 6.0 * 0.0625
_HEX_ICE_FRAC = (
    np.array(
        [
            [3.0, 5.0, 3.0 + _HEX_ICE_DM_PUCKER],
            [0.0, 4.0, 0.0 + _HEX_ICE_DM_PUCKER],
            [0.0, 2.0, 3.0 + _HEX_ICE_DM_PUCKER],
            [3.0, 1.0, 0.0 + _HEX_ICE_DM_PUCKER],
            [0.0, 2.0, 6.0 - _HEX_ICE_DM_PUCKER],
            [3.0, 1.0, 3.0 - _HEX_ICE_DM_PUCKER],
            [3.0, 5.0, 6.0 - _HEX_ICE_DM_PUCKER],
            [0.0, 4.0, 3.0 - _HEX_ICE_DM_PUCKER],
        ],
        dtype=np.float64,
    )
    / 6.0
)


# Cell-aspect ratios (per-axis side / a).
_CUBIC_ASPECT = np.array([1.0, 1.0, 1.0], dtype=np.float64)
_HEX_ASPECT = np.array(
    [1.0, np.sqrt(3.0), np.sqrt(8.0 / 3.0)], dtype=np.float64
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _as_n_cells(n_cells: Union[int, Sequence[int]]) -> np.ndarray:
    if isinstance(n_cells, int):
        out = np.array([n_cells, n_cells, n_cells], dtype=int)
    else:
        out = np.asarray(n_cells, dtype=int)
        if out.shape != (3,):
            raise ValueError(
                f"n_cells must be int or length-3 sequence; got shape {out.shape}."
            )
    if np.any(out <= 0):
        raise ValueError(f"n_cells must be positive; got {out.tolist()}.")
    return out


def _make_lattice(
    n_cells: Union[int, Sequence[int]],
    a: float,
    cell_aspect: np.ndarray,
    atom_positions_in_cell: np.ndarray,
) -> Array:
    """Tile a unit cell over an `(n_cells[0], n_cells[1], n_cells[2])` grid.

    Returns absolute atom positions in the box `[0, n_cells[i] * a *
    cell_aspect[i]]` per axis, of shape `(N, 3)` where
    `N = prod(n_cells) * len(atom_positions_in_cell)`.
    """
    if a <= 0:
        raise ValueError(f"lattice constant `a` must be positive; got {a}.")
    n_cells = _as_n_cells(n_cells)
    cell_size = float(a) * cell_aspect  # (3,)

    # Cell-corner offsets: (n_total, 3)
    grids = [np.arange(n) for n in n_cells]
    mesh = np.meshgrid(*grids, indexing="ij")
    corners = np.stack([m.ravel() for m in mesh], axis=-1).astype(np.float64)
    corners = corners * cell_size

    # Atom offsets within one cell, in absolute coordinates.
    sites = atom_positions_in_cell * cell_size  # (n_atoms, 3)

    lattice = corners[:, None, :] + sites[None, :, :]  # (n_total, n_atoms, 3)
    return lattice.reshape(-1, 3)


def make_box(
    n_cells: Union[int, Sequence[int]],
    a: float,
    cell_aspect: np.ndarray = _CUBIC_ASPECT,
) -> np.ndarray:
    """Per-axis box length (upper corner) for a given `n_cells / a / aspect`.

    Lower corner is the origin by convention; build a `Geometry` via
    `Geometry(lower=np.zeros(3), upper=make_box(n_cells, a, ...))` when
    you need one.
    """
    n_cells = _as_n_cells(n_cells)
    return n_cells.astype(np.float64) * float(a) * cell_aspect


# ----------------------------------------------------------------------
# Public lattice generators
# ----------------------------------------------------------------------
def fcc(n_cells: Union[int, Sequence[int]], a: float) -> Array:
    """Face-centred cubic lattice. 4 atoms per cubic cell."""
    return _make_lattice(n_cells, a, _CUBIC_ASPECT, _FCC_FRAC)


def diamond(n_cells: Union[int, Sequence[int]], a: float) -> Array:
    """Diamond cubic lattice. 8 atoms per cubic cell."""
    return _make_lattice(n_cells, a, _CUBIC_ASPECT, _DIAMOND_FRAC)


def bcc(n_cells: Union[int, Sequence[int]], a: float) -> Array:
    """Body-centred cubic lattice. 2 atoms per cubic cell."""
    return _make_lattice(n_cells, a, _CUBIC_ASPECT, _BCC_FRAC)


def hcp(n_cells: Union[int, Sequence[int]], a: float) -> Array:
    """Hexagonal close-packed lattice (orthorhombic representation).

    Cell aspect `(1, sqrt(3), sqrt(8/3))` so the conventional `c/a` ratio
    is the ideal `sqrt(8/3)`. 4 atoms per orthorhombic cell.
    """
    return _make_lattice(n_cells, a, _HEX_ASPECT, _HCP_FRAC)


def hex_ice(n_cells: Union[int, Sequence[int]], a: float) -> Array:
    """Hexagonal ice (Ice Ih) lattice, DM convention.

    8 atoms per orthorhombic cell with `cell_aspect = (1, sqrt(3),
    sqrt(8/3))` and the puckering parameter `6 * 0.0625` baked in (matches
    `flows_for_atomic_solids/models/particle_models.py:HexagonalIceLattice`).
    """
    return _make_lattice(n_cells, a, _HEX_ASPECT, _HEX_ICE_FRAC)


# Number of atoms per unit cell (useful for tests and callers).
ATOMS_PER_CELL: dict = {
    "fcc": 4,
    "diamond": 8,
    "bcc": 2,
    "hcp": 4,
    "hex_ice": 8,
}


def cell_aspect(name: str) -> np.ndarray:
    """Per-axis cell-aspect ratio for a named lattice."""
    if name in ("fcc", "diamond", "bcc"):
        return _CUBIC_ASPECT.copy()
    if name in ("hcp", "hex_ice"):
        return _HEX_ASPECT.copy()
    raise ValueError(
        f"cell_aspect: unknown lattice {name!r}; "
        f"expected one of {sorted(ATOMS_PER_CELL)}."
    )

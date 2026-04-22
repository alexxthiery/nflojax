# tests/test_utils_lattice.py
"""Tests for lattice generators (Stage B2)."""
from __future__ import annotations

import numpy as np
import pytest

from nflojax.utils.lattice import (
    ATOMS_PER_CELL,
    bcc,
    cell_aspect,
    diamond,
    fcc,
    hcp,
    hex_ice,
    make_box,
)


GEN = {
    "fcc": fcc,
    "diamond": diamond,
    "bcc": bcc,
    "hcp": hcp,
    "hex_ice": hex_ice,
}


# ---------------------------------------------------------------------
# Counts and shapes
# ---------------------------------------------------------------------
@pytest.mark.parametrize("name,gen", GEN.items())
def test_count_matches_atoms_per_cell(name, gen):
    """N == prod(n_cells) * atoms_per_cell."""
    n_cells = (2, 2, 2)
    pos = gen(n_cells, a=1.0)
    expected = int(np.prod(n_cells)) * ATOMS_PER_CELL[name]
    assert pos.shape == (expected, 3)


@pytest.mark.parametrize("name,gen", GEN.items())
def test_int_n_cells_equivalent_to_uniform_tuple(name, gen):
    """n_cells=2 == n_cells=(2, 2, 2)."""
    a = 1.0
    p_int = gen(2, a=a)
    p_tup = gen((2, 2, 2), a=a)
    assert np.allclose(p_int, p_tup)


@pytest.mark.parametrize("name,gen", GEN.items())
def test_positions_inside_box(name, gen):
    """All positions within [0, box) per axis."""
    n_cells = (3, 2, 4)
    a = 1.5
    pos = gen(n_cells, a=a)
    box = make_box(n_cells, a, cell_aspect(name))
    assert pos.shape[1] == 3
    assert np.all(pos >= 0.0 - 1e-12)
    # Use < box (positions are computed with `endpoint=False` semantics).
    assert np.all(pos < box + 1e-12)


@pytest.mark.parametrize("name,gen", GEN.items())
def test_no_duplicate_positions(name, gen):
    """Distinct atoms within the lattice."""
    pos = gen(2, a=1.0)
    rounded = np.round(pos, decimals=8)
    unique = np.unique(rounded, axis=0)
    assert unique.shape[0] == pos.shape[0]


# ---------------------------------------------------------------------
# Bitwise reference checks (reproduce known unit-cell geometries)
# ---------------------------------------------------------------------
def test_fcc_unit_cell_known_positions():
    """n_cells=1, a=1 == the canonical FCC unit cell."""
    pos = fcc(1, a=1.0)
    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.5, 0.5],
            [0.5, 0.0, 0.5],
            [0.5, 0.5, 0.0],
        ]
    )
    pos_sorted = pos[np.lexsort(pos.T)]
    exp_sorted = expected[np.lexsort(expected.T)]
    assert np.allclose(pos_sorted, exp_sorted, atol=1e-12)


def test_bcc_unit_cell_known_positions():
    pos = bcc(1, a=1.0)
    expected = np.array([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])
    assert np.allclose(np.sort(pos, axis=0), np.sort(expected, axis=0), atol=1e-12)


def test_diamond_unit_cell_size():
    """8 atoms per cubic cell of side 1; all in [0, 1)^3."""
    pos = diamond(1, a=1.0)
    assert pos.shape == (8, 3)
    assert np.all(pos >= 0.0 - 1e-12)
    assert np.all(pos < 1.0 + 1e-12)


def test_diamond_fcc_sublattice_match():
    """First 4 diamond atoms form the FCC sub-lattice."""
    diamond_pos = diamond(1, a=1.0)
    fcc_pos = fcc(1, a=1.0)
    # FCC sub-lattice is the first 4 in our ordering by construction.
    diamond_fcc = diamond_pos[:4]
    assert np.allclose(
        diamond_fcc[np.lexsort(diamond_fcc.T)],
        fcc_pos[np.lexsort(fcc_pos.T)],
        atol=1e-12,
    )


def test_hcp_cell_aspect_is_ideal():
    """cell_aspect = (1, sqrt(3), sqrt(8/3)); ideal c/a = sqrt(8/3)."""
    asp = cell_aspect("hcp")
    assert np.isclose(asp[0], 1.0)
    assert np.isclose(asp[1], np.sqrt(3.0))
    assert np.isclose(asp[2], np.sqrt(8.0 / 3.0))


def test_hex_ice_cell_aspect_matches_hcp():
    """Ice Ih shares HCP's orthorhombic cell aspect."""
    assert np.allclose(cell_aspect("hex_ice"), cell_aspect("hcp"))


def test_hex_ice_8_atoms_per_cell():
    """DM convention: 8 atoms per orthorhombic cell."""
    pos = hex_ice(1, a=1.0)
    assert pos.shape == (8, 3)


# ---------------------------------------------------------------------
# Cross-check against flows_for_atomic_solids (DM reference)
# ---------------------------------------------------------------------
def test_fcc_parity_with_flows_for_atomic_solids():
    """FCC positions match flows_for_atomic_solids.utils.lattice_utils.

    Reproduces their `make_lattice(lower=zeros, upper=ones*N, cell_aspect=ones,
    atom_positions_in_cell=FCC_FRAC, n=4*N**3)` via direct construction.
    """
    n = 2  # 2x2x2 cells, 32 atoms
    a = 1.0
    nflo = fcc(n, a=a)

    # DM convention reproduced inline (no import dependency on the external
    # repo; keeps the test hermetic).
    fcc_frac_dm = np.array(
        [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0]]
    ) / 2
    grids = [np.arange(n) for _ in range(3)]
    mesh = np.meshgrid(*grids, indexing="ij")
    corners = np.stack([m.ravel() for m in mesh], axis=-1).astype(np.float64) * a
    dm = (corners[:, None, :] + (fcc_frac_dm * a)[None, :, :]).reshape(-1, 3)

    nflo_sorted = nflo[np.lexsort(nflo.T)]
    dm_sorted = dm[np.lexsort(dm.T)]
    assert np.allclose(nflo_sorted, dm_sorted, atol=1e-12)


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------
def test_invalid_n_cells_raises():
    with pytest.raises(ValueError, match="n_cells must be int or length-3"):
        fcc((1, 2), a=1.0)
    with pytest.raises(ValueError, match="n_cells must be positive"):
        fcc(0, a=1.0)
    with pytest.raises(ValueError, match="n_cells must be positive"):
        fcc((1, -1, 1), a=1.0)


def test_invalid_lattice_constant_raises():
    with pytest.raises(ValueError, match="lattice constant `a` must be positive"):
        fcc(1, a=0.0)
    with pytest.raises(ValueError, match="lattice constant `a` must be positive"):
        fcc(1, a=-0.5)


def test_make_box_known_value():
    box = make_box((2, 3, 4), a=1.5, cell_aspect=cell_aspect("fcc"))
    assert np.allclose(box, np.array([3.0, 4.5, 6.0]))


def test_cell_aspect_unknown_lattice_raises():
    with pytest.raises(ValueError, match="unknown lattice"):
        cell_aspect("graphene")

"""Periodic (circular) conditioner inputs for particle flows on a torus.

On a periodic box the faces of the cube ``[-B, B)`` are one seam: a particle
just below ``+B`` and one just above ``-B`` are neighbours, but a conditioner
reading raw coordinates sees a jump of ``2B``. With a fully periodic
``geometry``, the particle nets (DeepSets, GNN, Transformer) feed
``circular_embed`` features of the coordinates to their first layer instead
(``circular_n_freq``, default 8 there); the GNN keeps raw coordinates for its
minimum-image neighbour list. ``build_particle_flow`` hands every conditioner
factory the flow-frame cube as ``geometry``.

Oracles: exact invariance under a box translation of one particle,
continuity across the seam with a positive control (raw coordinates jump),
first-layer widths, and the geometry the builder passes.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from nflojax.builders import build_particle_flow
from nflojax.geometry import Geometry
from nflojax.nets import GNN, DeepSets, Transformer

KEY = jax.random.PRNGKey(0)
B = 1.0
CUBE = Geometry.cubic(d=3, side=2 * B, lower=-B)
N = 6
KINDS = ["deepsets", "gnn", "transformer"]
FIRST_LAYER = {"deepsets": "phi_0", "gnn": "embed", "transformer": "input_proj"}


def _net(kind, geometry=CUBE, circular_n_freq=None):
    if kind == "deepsets":
        return DeepSets(phi_hidden=(16,), rho_hidden=(16,), out_dim=5,
                        geometry=geometry, circular_n_freq=circular_n_freq)
    if kind == "gnn":
        return GNN(num_layers=1, hidden=16, out_per_particle=5, num_neighbours=2,
                   geometry=geometry, circular_n_freq=circular_n_freq)
    return Transformer(num_layers=1, num_heads=2, embed_dim=16, out_per_particle=5,
                       geometry=geometry, circular_n_freq=circular_n_freq)


def _configuration():
    return jax.random.uniform(jax.random.PRNGKey(1), (N, 3), minval=-0.8 * B, maxval=0.8 * B)


def _seam_pair(eps=1e-4):
    """The same torus point written on both sides of the seam (up to 2 eps)."""
    x = _configuration()
    return x.at[0, 0].set(B - eps), x.at[0, 0].set(-B + eps)


@pytest.mark.parametrize("kind", KINDS)
class TestPeriodicInputs:
    def test_invariant_under_a_box_translation_of_one_particle(self, kind):
        """Moving one particle by a full box vector is the identity on the
        torus, so the output must not change (random init: every layer counts).
        Tolerance: float32 sin/cos of phases shifted by 2 pi k."""
        net, x = _net(kind), _configuration()
        params = net.init(KEY, x)
        moved = x.at[0, 1].add(2 * B)
        assert jnp.allclose(net.apply(params, x), net.apply(params, moved), atol=1e-4)

    def test_continuous_across_the_seam(self, kind):
        net = _net(kind)
        a, b = _seam_pair()
        params = net.init(KEY, a)
        assert float(jnp.max(jnp.abs(net.apply(params, a) - net.apply(params, b)))) < 1e-2

    def test_raw_coordinates_jump_at_the_seam(self, kind):
        """Positive control: with ``circular_n_freq=0`` the same pair differs
        by far more, so the continuity test above can fail."""
        net = _net(kind, circular_n_freq=0)
        a, b = _seam_pair()
        params = net.init(KEY, a)
        assert float(jnp.max(jnp.abs(net.apply(params, a) - net.apply(params, b)))) > 5e-2


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("geometry, n_freq, width", [
    (CUBE, None, 2 * 8 * 3),                                         # periodic: on, 8 harmonics
    (CUBE, 4, 2 * 4 * 3),
    (CUBE, 0, 3),                                                    # explicitly off
    (None, None, 3),                                                 # no geometry: raw
    (Geometry(lower=[-B] * 3, upper=[B] * 3, periodic=[True, True, False]), None, 3),
])
def test_first_layer_width(kind, geometry, n_freq, width):
    """The auto rule: circular features only on a fully periodic geometry."""
    if kind == "gnn" and geometry is None:
        pytest.skip("covered by the other kinds; GNN without geometry is Euclidean")
    net = _net(kind, geometry=geometry, circular_n_freq=n_freq)
    params = net.init(KEY, _configuration())["params"]
    assert params[FIRST_LAYER[kind]]["kernel"].shape[0] == width


@pytest.mark.parametrize("kind", KINDS)
def test_positive_n_freq_needs_a_geometry(kind):
    with pytest.raises(ValueError, match="geometry"):
        _net(kind, geometry=None, circular_n_freq=4).init(KEY, _configuration())


@pytest.mark.parametrize("periodic", [None, [True, True, False]])
def test_builder_passes_the_flow_frame_cube(periodic):
    """Each factory call gets the cube the conditioner sees, ``[-B, B)^d``
    with ``B = tail_bound``, periodic where the physical box is."""
    seen = []

    def factory(*, required_out_dim, **kw):
        seen.append(kw)
        return DeepSets(phi_hidden=(8,), rho_hidden=(), out_dim=required_out_dim)

    L = 6.2
    geometry = Geometry(lower=[0.0] * 3, upper=[L] * 3, periodic=periodic)
    build_particle_flow(KEY, geometry=geometry, event_shape=(4, 3), num_layers=1,
                        conditioner=factory, tail_bound=L / 2, return_transform_only=True)
    assert len(seen) == 2
    for kw in seen:
        g = kw["geometry"]
        assert np.allclose(g.lower, -L / 2) and np.allclose(g.upper, L / 2)
        assert g.is_periodic() == geometry.is_periodic()
        assert [g.is_periodic(a) for a in range(3)] == [geometry.is_periodic(a) for a in range(3)]

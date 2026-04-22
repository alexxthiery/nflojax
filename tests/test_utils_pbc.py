# tests/test_utils_pbc.py
"""Tests for orthogonal periodic-box helpers (Stage B4)."""
from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

from nflojax.geometry import Geometry
from nflojax.utils.pbc import (
    nearest_image,
    pairwise_distance,
    pairwise_distance_sq,
)


class TestNearestImage:
    def test_diagonal_wrap(self):
        """Displacement (0.8, 0.8, 0.8) in [0, 1]^3 wraps to (-0.2, -0.2, -0.2)."""
        geom = Geometry.cubic(d=3, side=1.0, lower=0.0)
        dx = jnp.array([0.8, 0.8, 0.8])
        wrapped = nearest_image(dx, geom)
        assert jnp.allclose(wrapped, jnp.array([-0.2, -0.2, -0.2]), atol=1e-6)

    def test_displacement_within_half_box_unchanged(self):
        """A displacement already in (-box/2, box/2] is unchanged."""
        geom = Geometry.cubic(d=3, side=2.0, lower=-1.0)  # box length 2 per axis
        dx = jnp.array([0.5, -0.3, 0.9])
        wrapped = nearest_image(dx, geom)
        assert jnp.allclose(wrapped, dx, atol=1e-6)

    def test_slab_geometry_non_periodic_axis_passes_through(self):
        """periodic=[True, True, False]: z-axis is NOT wrapped."""
        geom = Geometry(
            lower=[0.0, 0.0, 0.0], upper=[1.0, 1.0, 1.0],
            periodic=[True, True, False],
        )
        dx = jnp.array([0.8, 0.8, 0.8])
        wrapped = nearest_image(dx, geom)
        # x, y wrap; z stays.
        assert jnp.allclose(wrapped, jnp.array([-0.2, -0.2, 0.8]), atol=1e-6)

    def test_per_axis_box_lengths(self):
        """Rectangular box: per-axis wrap uses the correct length."""
        geom = Geometry(lower=[0.0, 0.0], upper=[1.0, 4.0])  # box = [1, 4]
        dx = jnp.array([0.8, 3.0])
        # x wraps: 0.8 - round(0.8/1)*1 = -0.2
        # y wraps: 3.0 - round(3.0/4)*4 = 3.0 - 4.0 = -1.0
        wrapped = nearest_image(dx, geom)
        assert jnp.allclose(wrapped, jnp.array([-0.2, -1.0]), atol=1e-6)

    def test_batch_shape_preserved(self):
        """Leading batch axes pass through."""
        geom = Geometry.cubic(d=3, side=1.0, lower=0.0)
        dx = jax.random.uniform(jax.random.PRNGKey(0), (7, 5, 3), minval=-5, maxval=5)
        wrapped = nearest_image(dx, geom)
        assert wrapped.shape == dx.shape
        # All wrapped values lie in (-0.5, 0.5].
        assert bool(jnp.all(wrapped > -0.5 - 1e-6))
        assert bool(jnp.all(wrapped <= 0.5 + 1e-6))

    def test_dimension_mismatch_raises(self):
        """dx last axis must equal geometry.d."""
        geom = Geometry.cubic(d=3)
        with pytest.raises(ValueError, match="last axis of dx must be geometry.d"):
            nearest_image(jnp.zeros((4, 2)), geom)

    def test_non_geometry_raises(self):
        with pytest.raises(TypeError, match="geometry must be a Geometry instance"):
            nearest_image(jnp.zeros((3,)), "cube")  # type: ignore[arg-type]

    def test_jit(self):
        geom = Geometry.cubic(d=3, side=1.0, lower=0.0)
        dx = jnp.array([0.8, -0.7, 0.4])
        wrap_jit = jax.jit(lambda d: nearest_image(d, geom))
        assert jnp.allclose(wrap_jit(dx), nearest_image(dx, geom), atol=1e-6)


class TestPairwiseDistance:
    def test_shape(self):
        """Output shape (..., N, N); diagonal zero; symmetric."""
        x = jax.random.normal(jax.random.PRNGKey(0), (4, 6, 3))
        d = pairwise_distance(x)
        assert d.shape == (4, 6, 6)
        assert jnp.allclose(jnp.diagonal(d, axis1=-2, axis2=-1), 0.0, atol=1e-5)
        assert jnp.allclose(d, jnp.swapaxes(d, -1, -2), atol=1e-5)

    def test_euclidean_without_geometry(self):
        """Two particles: plain Euclidean distance."""
        x = jnp.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])  # distance 5
        d = pairwise_distance(x)
        assert jnp.allclose(d, jnp.array([[0.0, 5.0], [5.0, 0.0]]), atol=1e-5)

    def test_pbc_minimum_image(self):
        """Two particles near opposite corners of a cubic box: PBC distance."""
        # Cubic [0, 1]^3. Particles at (0.1, 0.1, 0.1) and (0.9, 0.9, 0.9).
        # Euclidean: sqrt(3 * 0.64) = sqrt(1.92).
        # PBC: displacement (-0.2, -0.2, -0.2), distance = sqrt(3 * 0.04) = sqrt(0.12).
        geom = Geometry.cubic(d=3, side=1.0, lower=0.0)
        x = jnp.array([[0.1, 0.1, 0.1], [0.9, 0.9, 0.9]])
        d_euc = pairwise_distance(x)
        d_pbc = pairwise_distance(x, geometry=geom)
        assert jnp.isclose(d_euc[0, 1], jnp.sqrt(1.92), atol=1e-5)
        assert jnp.isclose(d_pbc[0, 1], jnp.sqrt(0.12), atol=1e-5)

    def test_pairwise_distance_sq_is_square(self):
        """pairwise_distance_sq == pairwise_distance**2."""
        x = jax.random.normal(jax.random.PRNGKey(0), (5, 3))
        d = pairwise_distance(x)
        d_sq = pairwise_distance_sq(x)
        assert jnp.allclose(d_sq, d * d, atol=1e-5)

    def test_ring_configuration(self):
        """4 particles on a ring in [0, 1]^2 at spacing 0.25; nearest-neighbour PBC distance = 0.25."""
        geom = Geometry.cubic(d=2, side=1.0, lower=0.0)
        x = jnp.array([
            [0.125, 0.5],
            [0.375, 0.5],
            [0.625, 0.5],
            [0.875, 0.5],
        ])
        d = pairwise_distance(x, geometry=geom)
        # Nearest neighbours (consecutive indices, including 0-3 via PBC) all at 0.25.
        nn = jnp.array([d[0, 1], d[1, 2], d[2, 3], d[3, 0]])
        assert jnp.allclose(nn, 0.25, atol=1e-5)

    def test_jit(self):
        geom = Geometry.cubic(d=3, side=2.0, lower=-1.0)
        x = jax.random.uniform(jax.random.PRNGKey(0), (6, 3), minval=-1, maxval=1)
        d_jit = jax.jit(lambda pts: pairwise_distance(pts, geom))
        assert jnp.allclose(d_jit(x), pairwise_distance(x, geom), atol=1e-5)

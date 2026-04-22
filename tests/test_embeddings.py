# tests/test_embeddings.py
"""Tests for stateless feature embeddings (Stage C)."""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from nflojax.embeddings import circular_embed, positional_embed
from nflojax.geometry import Geometry


# ============================================================================
# circular_embed
# ============================================================================
class TestCircularEmbed:
    def test_shape_rank1(self):
        """Input (d,) -> output (d * 2 * n_freq,)."""
        geom = Geometry.cubic(d=3, side=2.0, lower=-1.0)
        x = jnp.array([0.1, -0.2, 0.3])
        out = circular_embed(x, geom, n_freq=4)
        assert out.shape == (3 * 2 * 4,)

    def test_shape_rank2_particle_event(self):
        """Input (B, N, d) -> output (B, N, d * 2 * n_freq)."""
        geom = Geometry.cubic(d=3, side=2.0, lower=-1.0)
        x = jax.random.uniform(
            jax.random.PRNGKey(0), (4, 8, 3), minval=-1.0, maxval=1.0
        )
        out = circular_embed(x, geom, n_freq=3)
        assert out.shape == (4, 8, 3 * 2 * 3)

    def test_periodicity_shifts_by_box(self):
        """f(x + box) == f(x): the lowest harmonic exactly tiles the box."""
        geom = Geometry(lower=[-1.0, -2.0, 0.0], upper=[1.0, 2.0, 4.0])
        box = jnp.asarray(geom.box)
        x = jax.random.uniform(jax.random.PRNGKey(0), (10, 3), minval=-1, maxval=1)
        out_x = circular_embed(x, geom, n_freq=4)
        out_x_shift = circular_embed(x + box, geom, n_freq=4)
        assert jnp.allclose(out_x, out_x_shift, atol=1e-5)

    def test_boundary_values_at_lower_corner(self):
        """At x == lower, every cos is 1 and every sin is 0."""
        geom = Geometry.cubic(d=2, side=1.0, lower=0.0)
        x = jnp.zeros((2,))  # x == lower
        out = circular_embed(x, geom, n_freq=3)
        # Layout per coord: [cos_1, sin_1, cos_2, sin_2, cos_3, sin_3], two coords.
        # cos at phase 0 = 1, sin at phase 0 = 0.
        expected = jnp.array(
            [1.0, 0.0, 1.0, 0.0, 1.0, 0.0,   # coord 0
             1.0, 0.0, 1.0, 0.0, 1.0, 0.0],  # coord 1
        )
        assert jnp.allclose(out, expected, atol=1e-6)

    def test_jit(self):
        """Compiles under jit."""
        geom = Geometry.cubic(d=3, side=2.0, lower=-1.0)
        x = jnp.array([0.1, -0.2, 0.3])
        embed_jit = jax.jit(lambda y: circular_embed(y, geom, n_freq=4))
        assert jnp.allclose(embed_jit(x), circular_embed(x, geom, n_freq=4), atol=1e-6)

    def test_n_freq_zero_raises(self):
        """n_freq=0 is rejected to avoid silent zero-width output."""
        geom = Geometry.cubic(d=3, side=2.0, lower=-1.0)
        with pytest.raises(ValueError, match="n_freq must be >= 1"):
            circular_embed(jnp.zeros((3,)), geom, n_freq=0)

    def test_dim_mismatch_raises(self):
        """x last axis must equal geometry.d."""
        geom = Geometry.cubic(d=3, side=1.0, lower=0.0)
        with pytest.raises(ValueError, match="x last axis must equal geometry.d"):
            circular_embed(jnp.zeros((4,)), geom, n_freq=2)

    def test_non_geometry_raises(self):
        with pytest.raises(TypeError, match="geometry must be a Geometry instance"):
            circular_embed(jnp.zeros((3,)), "cube", n_freq=2)  # type: ignore[arg-type]


# ============================================================================
# positional_embed
# ============================================================================
class TestPositionalEmbed:
    def test_shape_rank1(self):
        """Input (B,) -> output (B, 2 * n_freq)."""
        t = jnp.array([0.0, 1.0, 2.0, 3.0])
        out = positional_embed(t, n_freq=8)
        assert out.shape == (4, 16)

    def test_shape_scalar(self):
        """Input scalar (rank-0) -> output (2 * n_freq,)."""
        t = jnp.asarray(0.5)
        out = positional_embed(t, n_freq=4)
        assert out.shape == (8,)

    def test_formula_match(self):
        """Output matches the closed-form sinusoidal expression."""
        t_val = 1.5
        n_freq = 4
        base = 10000.0
        out = positional_embed(jnp.asarray(t_val), n_freq=n_freq, base=base)
        for k in range(n_freq):
            freq = base ** (-k / n_freq)
            assert jnp.isclose(out[2 * k], math.cos(t_val * freq), atol=1e-5)
            assert jnp.isclose(out[2 * k + 1], math.sin(t_val * freq), atol=1e-5)

    def test_at_zero_is_alternating_one_zero(self):
        """positional_embed(0, K) == [1, 0, 1, 0, ..., 1, 0]."""
        out = positional_embed(jnp.asarray(0.0), n_freq=3)
        assert jnp.allclose(out, jnp.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0]), atol=1e-6)

    def test_python_int_input_works(self):
        """Plain Python int t goes through (no float-cast required at call site)."""
        out = positional_embed(0, n_freq=3)
        assert out.shape == (6,)
        assert jnp.allclose(
            out, jnp.array([1.0, 0.0, 1.0, 0.0, 1.0, 0.0]), atol=1e-6
        )
        # Batched int array also works.
        out_batch = positional_embed(jnp.array([0, 1, 2]), n_freq=4)
        assert out_batch.shape == (3, 8)

    def test_jit(self):
        t = jnp.array([0.5, 1.0, 1.5])
        embed_jit = jax.jit(lambda x: positional_embed(x, n_freq=4))
        assert jnp.allclose(embed_jit(t), positional_embed(t, n_freq=4), atol=1e-6)

    def test_n_freq_zero_raises(self):
        with pytest.raises(ValueError, match="n_freq must be >= 1"):
            positional_embed(jnp.asarray(0.5), n_freq=0)

    def test_invalid_base_raises(self):
        with pytest.raises(ValueError, match="base must be positive"):
            positional_embed(jnp.asarray(0.5), n_freq=4, base=0.0)
        with pytest.raises(ValueError, match="base must be positive"):
            positional_embed(jnp.asarray(0.5), n_freq=4, base=-1.0)

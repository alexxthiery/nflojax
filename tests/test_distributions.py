# tests/test_distributions.py
"""Unit tests for base distributions."""
from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

from nflojax.distributions import StandardNormal, DiagNormal


class TestStandardNormal:
    """Tests for StandardNormal distribution."""

    def test_log_prob_shape(self, key, dim, batch_size):
        """log_prob returns correct shape."""
        dist = StandardNormal(dim=dim)
        x = jax.random.normal(key, (batch_size, dim))

        lp = dist.log_prob({}, x)

        assert lp.shape == (batch_size,)
        assert not jnp.isnan(lp).any()

    def test_log_prob_value(self, dim):
        """log_prob matches analytical formula for N(0,I)."""
        dist = StandardNormal(dim=dim)
        x = jnp.zeros((1, dim))

        lp = dist.log_prob({}, x)

        # At x=0: log p(0) = -0.5 * dim * log(2*pi)
        expected = -0.5 * dim * jnp.log(2.0 * jnp.pi)
        assert jnp.allclose(lp, expected, atol=1e-6)

    def test_sample_shape(self, key, dim):
        """sample returns correct shape."""
        dist = StandardNormal(dim=dim)

        samples = dist.sample({}, key, (100,))

        assert samples.shape == (100, dim)

    def test_sample_statistics(self, key, dim):
        """Samples have approximately zero mean and unit variance."""
        dist = StandardNormal(dim=dim)

        samples = dist.sample({}, key, (10000,))
        mean = jnp.mean(samples, axis=0)
        std = jnp.std(samples, axis=0)

        assert jnp.allclose(mean, 0.0, atol=0.1)
        assert jnp.allclose(std, 1.0, atol=0.1)

    def test_log_prob_wrong_dim_raises(self, dim):
        """log_prob raises ValueError for wrong input dimension."""
        dist = StandardNormal(dim=dim)
        x_wrong = jnp.zeros((5, dim + 1))

        with pytest.raises(ValueError, match="event_shape"):
            dist.log_prob({}, x_wrong)


class TestDiagNormal:
    """Tests for DiagNormal distribution."""

    @pytest.fixture
    def diag_params(self, dim):
        """Standard DiagNormal params (loc=0, scale=1)."""
        return {
            "loc": jnp.zeros(dim),
            "log_scale": jnp.zeros(dim),
        }

    @pytest.fixture
    def shifted_params(self, dim):
        """Shifted DiagNormal (loc=2, scale=0.5)."""
        return {
            "loc": jnp.full(dim, 2.0),
            "log_scale": jnp.full(dim, jnp.log(0.5)),
        }

    def test_log_prob_shape(self, key, dim, batch_size, diag_params):
        """log_prob returns correct shape."""
        dist = DiagNormal(dim=dim)
        x = jax.random.normal(key, (batch_size, dim))

        lp = dist.log_prob(diag_params, x)

        assert lp.shape == (batch_size,)
        assert not jnp.isnan(lp).any()

    def test_log_prob_matches_standard_normal(self, key, dim, diag_params):
        """With loc=0, log_scale=0, DiagNormal matches StandardNormal."""
        diag = DiagNormal(dim=dim)
        std = StandardNormal(dim=dim)
        x = jax.random.normal(key, (50, dim))

        lp_diag = diag.log_prob(diag_params, x)
        lp_std = std.log_prob({}, x)

        assert jnp.allclose(lp_diag, lp_std, atol=1e-6)

    def test_log_prob_shifted(self, dim, shifted_params):
        """log_prob at loc should be maximal."""
        dist = DiagNormal(dim=dim)
        loc = shifted_params["loc"]

        # At x = loc, the quadratic term is zero
        lp_at_loc = dist.log_prob(shifted_params, loc.reshape(1, -1))

        # At x = loc + 1, it should be lower
        lp_away = dist.log_prob(shifted_params, (loc + 1.0).reshape(1, -1))

        assert lp_at_loc > lp_away

    def test_sample_shape(self, key, dim, diag_params):
        """sample returns correct shape."""
        dist = DiagNormal(dim=dim)

        samples = dist.sample(diag_params, key, (100,))

        assert samples.shape == (100, dim)

    def test_sample_statistics(self, key, dim, shifted_params):
        """Samples have correct mean and std."""
        dist = DiagNormal(dim=dim)
        expected_loc = shifted_params["loc"]
        expected_scale = jnp.exp(shifted_params["log_scale"])

        samples = dist.sample(shifted_params, key, (10000,))
        mean = jnp.mean(samples, axis=0)
        std = jnp.std(samples, axis=0)

        assert jnp.allclose(mean, expected_loc, atol=0.1)
        assert jnp.allclose(std, expected_scale, atol=0.1)

    def test_log_prob_wrong_dim_raises(self, dim, diag_params):
        """log_prob raises ValueError for wrong input dimension."""
        dist = DiagNormal(dim=dim)
        x_wrong = jnp.zeros((5, dim + 1))

        with pytest.raises(ValueError, match="event_shape"):
            dist.log_prob(diag_params, x_wrong)

    def test_missing_loc_raises(self, dim):
        """Missing 'loc' in params raises KeyError."""
        dist = DiagNormal(dim=dim)
        bad_params = {"log_scale": jnp.zeros(dim)}
        x = jnp.zeros((1, dim))

        with pytest.raises(KeyError, match="loc"):
            dist.log_prob(bad_params, x)

    def test_missing_log_scale_raises(self, dim):
        """Missing 'log_scale' in params raises KeyError."""
        dist = DiagNormal(dim=dim)
        bad_params = {"loc": jnp.zeros(dim)}
        x = jnp.zeros((1, dim))

        with pytest.raises(KeyError, match="log_scale"):
            dist.log_prob(bad_params, x)

    def test_wrong_loc_shape_raises(self, dim):
        """Wrong loc shape raises ValueError."""
        dist = DiagNormal(dim=dim)
        bad_params = {
            "loc": jnp.zeros(dim + 1),
            "log_scale": jnp.zeros(dim),
        }
        x = jnp.zeros((1, dim))

        with pytest.raises(ValueError, match="loc must have shape"):
            dist.log_prob(bad_params, x)

    def test_wrong_log_scale_shape_raises(self, dim):
        """Wrong log_scale shape raises ValueError."""
        dist = DiagNormal(dim=dim)
        bad_params = {
            "loc": jnp.zeros(dim),
            "log_scale": jnp.zeros(dim + 1),
        }
        x = jnp.zeros((1, dim))

        with pytest.raises(ValueError, match="log_scale must have shape"):
            dist.log_prob(bad_params, x)

    def test_jit_compatible(self, key, dim, diag_params):
        """DiagNormal works under JIT."""
        dist = DiagNormal(dim=dim)
        x = jax.random.normal(key, (10, dim))

        log_prob_jit = jax.jit(dist.log_prob)
        sample_jit = jax.jit(lambda p, k: dist.sample(p, k, (10,)))

        lp = log_prob_jit(diag_params, x)
        s = sample_jit(diag_params, key)

        assert lp.shape == (10,)
        assert s.shape == (10, dim)


# ----------------------------------------------------------------------
# Rank-N (structured) event_shape support.
#
# The event may have rank >= 1; log_prob sums over all trailing event axes.
# For rank-1 the behavior is bit-identical to the original `dim=int` path.
# For rank >= 2 this unlocks particle-system flows like (B, N, d).
# ----------------------------------------------------------------------
class TestStandardNormalStructured:
    """StandardNormal with rank-N event_shape."""

    def test_event_shape_int_matches_dim(self, key, batch_size):
        """event_shape=N is identical to dim=N (back-compat)."""
        dist_int = StandardNormal(event_shape=5)
        dist_dim = StandardNormal(dim=5)
        x = jax.random.normal(key, (batch_size, 5))

        assert dist_int.event_shape == (5,)
        assert dist_dim.event_shape == (5,)
        assert jnp.allclose(
            dist_int.log_prob({}, x), dist_dim.log_prob({}, x), atol=1e-6
        )

    def test_rank2_sample_shape(self, key):
        """event_shape=(N, d) produces samples of shape (*batch, N, d)."""
        dist = StandardNormal(event_shape=(8, 3))
        samples = dist.sample({}, key, (16,))
        assert samples.shape == (16, 8, 3)

    def test_rank2_log_prob_shape(self, key):
        """log_prob for rank-2 event returns a scalar per batch element."""
        dist = StandardNormal(event_shape=(8, 3))
        x = jax.random.normal(key, (16, 8, 3))
        lp = dist.log_prob({}, x)
        assert lp.shape == (16,)
        assert not jnp.isnan(lp).any()

    def test_rank2_log_prob_value_at_zero(self):
        """At x=0 the log-prob equals the analytic log-normalizer of N(0, I_{N*d})."""
        N, d = 8, 3
        dist = StandardNormal(event_shape=(N, d))
        x = jnp.zeros((1, N, d))

        expected = -0.5 * (N * d) * jnp.log(2.0 * jnp.pi)
        assert jnp.allclose(dist.log_prob({}, x), expected, atol=1e-6)

    def test_rank2_equals_flattened(self, key):
        """log_prob of rank-2 event = log_prob of flattened rank-1 event."""
        N, d = 4, 3
        dist_struct = StandardNormal(event_shape=(N, d))
        dist_flat = StandardNormal(event_shape=N * d)
        x = jax.random.normal(key, (10, N, d))

        lp_struct = dist_struct.log_prob({}, x)
        lp_flat = dist_flat.log_prob({}, x.reshape(10, N * d))
        assert jnp.allclose(lp_struct, lp_flat, atol=1e-6)

    def test_rank2_wrong_trailing_shape_raises(self):
        """log_prob raises when trailing axes don't match event_shape."""
        dist = StandardNormal(event_shape=(8, 3))
        x_wrong = jnp.zeros((4, 8, 4))
        with pytest.raises(ValueError, match="event_shape"):
            dist.log_prob({}, x_wrong)


class TestDiagNormalStructured:
    """DiagNormal with rank-N event_shape."""

    def test_event_shape_int_matches_dim(self, key, batch_size):
        """event_shape=N is identical to dim=N (back-compat)."""
        dist_int = DiagNormal(event_shape=5)
        dist_dim = DiagNormal(dim=5)
        params = {"loc": jnp.zeros(5), "log_scale": jnp.zeros(5)}
        x = jax.random.normal(key, (batch_size, 5))

        assert dist_int.event_shape == (5,)
        assert dist_dim.event_shape == (5,)
        assert jnp.allclose(
            dist_int.log_prob(params, x), dist_dim.log_prob(params, x), atol=1e-6
        )

    def test_rank2_param_shapes(self):
        """loc and log_scale have shape event_shape for rank-2."""
        N, d = 8, 3
        dist = DiagNormal(event_shape=(N, d))
        params = dist.init_params()
        assert params["loc"].shape == (N, d)
        assert params["log_scale"].shape == (N, d)

    def test_rank2_sample_shape(self, key):
        """Samples of rank-2 distribution have the right shape."""
        dist = DiagNormal(event_shape=(8, 3))
        params = dist.init_params()
        samples = dist.sample(params, key, (16,))
        assert samples.shape == (16, 8, 3)

    def test_rank2_log_prob_matches_standard_normal(self, key):
        """loc=0, log_scale=0 collapses DiagNormal to StandardNormal on the same event."""
        N, d = 4, 3
        diag = DiagNormal(event_shape=(N, d))
        std = StandardNormal(event_shape=(N, d))
        params = diag.init_params()
        x = jax.random.normal(key, (16, N, d))

        assert jnp.allclose(
            diag.log_prob(params, x), std.log_prob({}, x), atol=1e-6
        )

    def test_rank2_log_prob_shifted(self, key):
        """log_prob at loc is greater than at loc + 1 (per-batch-element)."""
        N, d = 4, 3
        dist = DiagNormal(event_shape=(N, d))
        loc = jnp.ones((N, d)) * 2.0
        params = {"loc": loc, "log_scale": jnp.zeros((N, d))}

        lp_at_loc = dist.log_prob(params, loc[None])
        lp_away = dist.log_prob(params, (loc + 1.0)[None])
        assert lp_at_loc[0] > lp_away[0]


class TestDistributionRepr:
    """repr/equality should reflect the canonical event_shape, not legacy dim."""

    def test_standard_normal_repr_has_event_shape(self):
        r = repr(StandardNormal(event_shape=(8, 3)))
        assert "event_shape=(8, 3)" in r
        assert "dim=" not in r  # dim is a property, not a dataclass field

    def test_diag_normal_repr_has_event_shape(self):
        r = repr(DiagNormal(event_shape=(8, 3)))
        assert "event_shape=(8, 3)" in r
        assert "dim=" not in r

    def test_equality_by_event_shape(self):
        assert StandardNormal(dim=4) == StandardNormal(event_shape=(4,))
        assert DiagNormal(dim=4) == DiagNormal(event_shape=4)
        assert StandardNormal(event_shape=(8, 3)) != StandardNormal(event_shape=(3, 8))

# tests/test_distributions.py
"""Unit tests for base distributions."""
from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

from nflojax.distributions import StandardNormal, DiagNormal, UniformBox, LatticeBase
from nflojax.geometry import Geometry


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


# ============================================================================
# UniformBox (Stage B1)
# ============================================================================
class TestUniformBox:
    """Per-axis uniform base distribution on `geometry.box`."""

    def test_sample_shape_and_bounds_rank1(self):
        """Rank-1 event: samples shape `(batch, d)` and lie inside the box."""
        geom = Geometry(lower=[-2.0, -1.0, 0.0], upper=[3.0, 4.0, 5.0])
        dist = UniformBox(geometry=geom, event_shape=(3,))
        x = dist.sample(None, jax.random.PRNGKey(0), (200,))
        assert x.shape == (200, 3)
        lower = jnp.asarray(geom.lower)
        upper = jnp.asarray(geom.upper)
        assert bool(jnp.all(x >= lower))
        assert bool(jnp.all(x <= upper))

    def test_sample_shape_rank2_particle_event(self):
        """Rank-2 event: samples shape `(batch, N, d)`; each particle per-axis in box."""
        geom = Geometry.cubic(d=3, side=2.0, lower=-1.0)
        N = 8
        dist = UniformBox(geometry=geom, event_shape=(N, 3))
        x = dist.sample(None, jax.random.PRNGKey(0), (16,))
        assert x.shape == (16, N, 3)
        assert bool(jnp.all(x >= -1.0)) and bool(jnp.all(x <= 1.0))

    def test_log_prob_closed_form_rank1(self):
        """Constant log-density matches `-sum(log(box))`."""
        geom = Geometry(lower=[-2.0, -1.0, 0.0], upper=[3.0, 4.0, 5.0])
        dist = UniformBox(geometry=geom, event_shape=(3,))
        box = jnp.asarray(geom.box)
        expected = -jnp.sum(jnp.log(box))
        x = dist.sample(None, jax.random.PRNGKey(0), (10,))
        log_p = dist.log_prob(None, x)
        assert log_p.shape == (10,)
        assert jnp.allclose(log_p, expected)

    def test_log_prob_scales_with_particle_count(self):
        """Rank-2 event: log_prob = -N * sum(log(box))."""
        geom = Geometry.cubic(d=3, side=2.0, lower=-1.0)
        N = 5
        dist = UniformBox(geometry=geom, event_shape=(N, 3))
        box = jnp.asarray(geom.box)
        expected = -N * jnp.sum(jnp.log(box))
        x = dist.sample(None, jax.random.PRNGKey(0), (4,))
        log_p = dist.log_prob(None, x)
        assert log_p.shape == (4,)
        assert jnp.allclose(log_p, expected)

    def test_log_prob_out_of_box_is_neg_inf(self):
        """Samples outside the box get -inf log_prob."""
        geom = Geometry.cubic(d=2, side=1.0, lower=0.0)  # [0, 1]^2
        dist = UniformBox(geometry=geom, event_shape=(2,))
        x = jnp.stack(
            [
                jnp.array([0.5, 0.5]),   # in
                jnp.array([1.5, 0.5]),   # out (x>1)
                jnp.array([-0.1, 0.5]),  # out (x<0)
                jnp.array([0.5, 0.5]),   # in
            ]
        )
        log_p = dist.log_prob(None, x)
        assert log_p.shape == (4,)
        assert jnp.isfinite(log_p[0])
        assert jnp.isneginf(log_p[1])
        assert jnp.isneginf(log_p[2])
        assert jnp.isfinite(log_p[3])

    def test_log_prob_at_box_edges_is_finite(self):
        """The boundary [lower, upper] is inclusive."""
        geom = Geometry.cubic(d=2, side=1.0, lower=0.0)
        dist = UniformBox(geometry=geom, event_shape=(2,))
        x = jnp.stack(
            [
                jnp.array([0.0, 0.0]),
                jnp.array([1.0, 1.0]),
                jnp.array([0.0, 1.0]),
            ]
        )
        log_p = dist.log_prob(None, x)
        assert bool(jnp.all(jnp.isfinite(log_p)))

    def test_sample_and_log_prob_consistency(self):
        """log_prob(sample(...)) is finite and constant."""
        geom = Geometry.cubic(d=3, side=2.0, lower=-1.0)
        dist = UniformBox(geometry=geom, event_shape=(4, 3))
        x = dist.sample(None, jax.random.PRNGKey(42), (64,))
        log_p = dist.log_prob(None, x)
        # All finite (samples are all in-box by construction).
        assert bool(jnp.all(jnp.isfinite(log_p)))
        # All equal (constant density).
        assert jnp.allclose(log_p, log_p[0])

    def test_jit(self):
        """log_prob and sample compile under jit."""
        geom = Geometry.cubic(d=3, side=2.0, lower=-1.0)
        dist = UniformBox(geometry=geom, event_shape=(3,))
        sample = jax.jit(dist.sample, static_argnums=(2,))
        log_prob = jax.jit(dist.log_prob)
        x = sample(None, jax.random.PRNGKey(0), (10,))
        log_p = log_prob(None, x)
        assert x.shape == (10, 3)
        assert log_p.shape == (10,)
        assert bool(jnp.all(jnp.isfinite(log_p)))

    def test_init_params_returns_none(self):
        geom = Geometry.cubic(d=3)
        dist = UniformBox(geometry=geom, event_shape=(3,))
        assert dist.init_params() is None

    def test_invalid_event_shape_raises(self):
        """event_shape last axis must equal geometry.d."""
        geom = Geometry.cubic(d=3)
        with pytest.raises(ValueError, match="event_shape must end in the coord dim"):
            UniformBox(geometry=geom, event_shape=(4, 2))
        with pytest.raises(ValueError, match="event_shape must end in the coord dim"):
            UniformBox(geometry=geom, event_shape=())

    def test_invalid_geometry_raises(self):
        """Non-Geometry raises."""
        with pytest.raises(TypeError, match="geometry must be a Geometry instance"):
            UniformBox(geometry="cube", event_shape=(3,))  # type: ignore[arg-type]


# ============================================================================
# LatticeBase (Stage B3)
# ============================================================================
import math as _math


_FACTORIES = ("fcc", "diamond", "bcc", "hcp", "hex_ice")


class TestLatticeBaseFactories:
    """Smoke tests across all 5 named factories."""

    @pytest.mark.parametrize("name", _FACTORIES)
    def test_factory_shapes(self, name):
        """LatticeBase.<name>(2, 1.0, 0.1) returns the expected (N, 3) event."""
        lb = getattr(LatticeBase, name)(n_cells=2, a=1.0, noise_scale=0.1)
        from nflojax.utils.lattice import ATOMS_PER_CELL
        expected_n = 8 * ATOMS_PER_CELL[name]
        assert lb.event_shape == (expected_n, 3)
        assert lb.N == expected_n
        assert lb.d == 3

    @pytest.mark.parametrize("name", _FACTORIES)
    def test_factory_geometry_extent(self, name):
        """Geometry box matches `n_cells * a * cell_aspect`."""
        from nflojax.utils.lattice import cell_aspect, make_box
        lb = getattr(LatticeBase, name)(n_cells=2, a=1.5, noise_scale=0.1)
        expected = make_box(2, 1.5, cell_aspect(name))
        assert jnp.allclose(jnp.asarray(lb.geometry.box), expected, atol=1e-6)

    @pytest.mark.parametrize("name", _FACTORIES)
    def test_sample_then_log_prob_finite(self, name):
        """sample(...) + log_prob round-trip is finite."""
        lb = getattr(LatticeBase, name)(n_cells=2, a=1.0, noise_scale=0.1)
        x = lb.sample(None, jax.random.PRNGKey(0), (4,))
        assert x.shape == (4, lb.N, lb.d)
        log_p = lb.log_prob(None, x)
        assert log_p.shape == (4,)
        assert bool(jnp.all(jnp.isfinite(log_p)))


class TestLatticeBaseDensity:
    """log_prob math, sampling stats, permutation correction."""

    def test_log_prob_closed_form_at_lattice_sites(self):
        """At x == positions: log_prob == -0.5*N*d*log(2π) - N*d*log(σ)."""
        lb = LatticeBase.fcc(n_cells=1, a=1.0, noise_scale=0.2)
        x = lb.positions[None, ...]   # (1, 4, 3)
        nd = lb.N * lb.d
        expected = -0.5 * nd * jnp.log(2 * jnp.pi) - nd * jnp.log(0.2)
        log_p = lb.log_prob(None, x)
        assert jnp.allclose(log_p[0], expected, atol=1e-5)

    def test_permute_subtracts_log_n_factorial(self):
        """log_prob with permute=True is the labelled value minus log(N!)."""
        N = 4  # FCC n_cells=1 -> N=4.
        lb_label = LatticeBase.fcc(n_cells=1, a=1.0, noise_scale=0.1, permute=False)
        lb_perm = LatticeBase.fcc(n_cells=1, a=1.0, noise_scale=0.1, permute=True)
        x = lb_label.sample(None, jax.random.PRNGKey(0), (8,))
        log_label = lb_label.log_prob(None, x)
        log_perm = lb_perm.log_prob(None, x)
        expected_diff = -_math.lgamma(N + 1)
        assert jnp.allclose(log_perm - log_label, expected_diff, atol=1e-5)

    def test_sample_centred_on_lattice(self):
        """Mean of sampled positions converges to lattice positions."""
        lb = LatticeBase.fcc(n_cells=2, a=1.0, noise_scale=0.05)
        x = lb.sample(None, jax.random.PRNGKey(0), (4096,))
        mean = jnp.mean(x, axis=0)  # (N, d)
        assert jnp.allclose(mean, lb.positions, atol=5e-3)

    def test_sample_std_matches_noise_scale(self):
        """Standard deviation of samples matches noise_scale."""
        lb = LatticeBase.fcc(n_cells=2, a=1.0, noise_scale=0.05)
        x = lb.sample(None, jax.random.PRNGKey(1), (4096,))
        std = jnp.std(x, axis=0)  # (N, d)
        assert jnp.allclose(std, 0.05, atol=5e-3)

    def test_permute_sample_shuffles_particle_axis(self):
        """With permute=True, the per-batch particle order is shuffled."""
        lb_no = LatticeBase.fcc(n_cells=2, a=1.0, noise_scale=0.0001, permute=False)
        lb_yes = LatticeBase.fcc(n_cells=2, a=1.0, noise_scale=0.0001, permute=True)
        x_no = lb_no.sample(None, jax.random.PRNGKey(0), (32,))
        x_yes = lb_yes.sample(None, jax.random.PRNGKey(0), (32,))
        # Without permute: every batch sample's particles are in site order.
        nearest_no = jnp.allclose(x_no, lb_no.positions, atol=5e-3)
        assert nearest_no
        # With permute: not equal to lattice (almost always shuffled).
        nearest_yes = jnp.allclose(x_yes, lb_yes.positions, atol=5e-3)
        assert not nearest_yes
        # Sorted positions still match (permutation is the only change).
        x_yes_sorted = jnp.sort(x_yes.reshape(32, -1), axis=-1)
        x_no_sorted = jnp.sort(x_no.reshape(32, -1), axis=-1)
        assert jnp.allclose(x_yes_sorted, x_no_sorted, atol=5e-3)


class TestLatticeBaseValidation:
    def test_non_geometry_raises(self):
        with pytest.raises(TypeError, match="geometry must be a Geometry instance"):
            LatticeBase(positions=jnp.zeros((4, 3)), geometry="cube",  # type: ignore[arg-type]
                        noise_scale=0.1)

    def test_positions_2d_required(self):
        geom = Geometry(lower=[0., 0., 0.], upper=[1., 1., 1.])
        with pytest.raises(ValueError, match="positions must be 2-D"):
            LatticeBase(positions=jnp.zeros((4, 3, 1)), geometry=geom, noise_scale=0.1)

    def test_positions_match_geometry_d(self):
        geom = Geometry(lower=[0., 0., 0.], upper=[1., 1., 1.])  # d=3
        with pytest.raises(ValueError, match="positions.shape\\[-1\\]"):
            LatticeBase(positions=jnp.zeros((4, 2)), geometry=geom, noise_scale=0.1)

    def test_noise_scale_positive(self):
        geom = Geometry(lower=[0., 0., 0.], upper=[1., 1., 1.])
        with pytest.raises(ValueError, match="noise_scale must be positive"):
            LatticeBase(positions=jnp.zeros((4, 3)), geometry=geom, noise_scale=0.0)


class TestLatticeBaseJit:
    def test_log_prob_jit(self):
        lb = LatticeBase.bcc(n_cells=2, a=1.0, noise_scale=0.1)
        x = lb.sample(None, jax.random.PRNGKey(0), (4,))
        log_prob_jit = jax.jit(lb.log_prob)
        assert jnp.allclose(log_prob_jit(None, x), lb.log_prob(None, x), atol=1e-5)

    def test_sample_jit_no_permute(self):
        lb = LatticeBase.bcc(n_cells=2, a=1.0, noise_scale=0.1, permute=False)
        sample_jit = jax.jit(lb.sample, static_argnums=(2,))
        x_jit = sample_jit(None, jax.random.PRNGKey(0), (4,))
        x_ref = lb.sample(None, jax.random.PRNGKey(0), (4,))
        assert jnp.allclose(x_jit, x_ref, atol=1e-5)

    def test_sample_jit_with_permute(self):
        """permute=True path also compiles."""
        lb = LatticeBase.bcc(n_cells=2, a=1.0, noise_scale=0.1, permute=True)
        sample_jit = jax.jit(lb.sample, static_argnums=(2,))
        x = sample_jit(None, jax.random.PRNGKey(0), (4,))
        assert x.shape == (4, lb.N, lb.d)
        log_p = lb.log_prob(None, x)
        assert bool(jnp.all(jnp.isfinite(log_p)))

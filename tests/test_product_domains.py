"""Tests for flat product-domain flows."""
from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

pytestmark = pytest.mark.slow

from conftest import check_logdet_vs_autodiff
from nflojax.builders import build_product_spline_flow
from nflojax.distributions import ProductBase, StandardNormal
from nflojax.domains import ProductDomain, ScalarDomain
from nflojax.flows import Bijection, Flow
from nflojax.transforms import CircularCoordinateShift, ProductSplineCoupling


def mixed_domain() -> ProductDomain:
    """Small mixed product domain used by tests."""

    return ProductDomain(
        [
            ScalarDomain.real(),
            ScalarDomain.interval(-2.0, 3.0),
            ScalarDomain.circular(-jnp.pi, jnp.pi),
            ScalarDomain.real(),
        ]
    )


class TestScalarDomain:
    """Validation and feature tests for scalar domain specifications."""

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError, match="kind"):
            ScalarDomain("bounded")

    def test_real_with_bounds_raises(self):
        with pytest.raises(ValueError, match="must not have bounds"):
            ScalarDomain("real", lower=0.0, upper=1.0)

    def test_invalid_interval_raises(self):
        with pytest.raises(ValueError, match="lower"):
            ScalarDomain.interval(1.0, 1.0)

    def test_invalid_product_domain_raises(self):
        with pytest.raises(ValueError, match="at least one"):
            ProductDomain([])
        with pytest.raises(TypeError, match="ScalarDomain"):
            ProductDomain([ScalarDomain.real(), "not-a-domain"])

    def test_product_domain_properties(self):
        domain = mixed_domain()

        assert domain.dim == 4
        assert domain.has_circular
        assert domain.params_per_scalar(num_bins=5) == (14, 14, 15, 14)

    def test_invalid_product_masks_raise(self):
        domain = mixed_domain()
        with pytest.raises(ValueError, match="shape"):
            domain.required_out_dim(jnp.array([1.0, 0.0]), num_bins=5)
        with pytest.raises(ValueError, match="0 or 1"):
            domain.required_out_dim(jnp.array([1.0, 0.5, 0.0, 1.0]), num_bins=5)
        with pytest.raises(ValueError, match="positive"):
            domain.conditioner_feature_dim(jnp.array([1.0, 0.0, 1.0, 0.0]), 0)

    def test_features_are_finite_and_expand_circular(self):
        domain = mixed_domain()
        x = jnp.array([[0.5, 0.5, 0.0, -1.0]])
        mask = jnp.array([1.0, 1.0, 1.0, 0.0])

        features = domain.conditioner_features(x, mask, circular_n_freq=2)

        # real + interval + circular sin/cos for two frequencies
        assert features.shape == (1, 1 + 1 + 4)
        assert bool(jnp.all(jnp.isfinite(features)))


class TestProductBase:
    """Tests for ProductBase distribution."""

    def test_sample_log_prob_shape_and_domain(self, key):
        domain = mixed_domain()
        base = ProductBase(domain)
        params = base.init_params()

        samples = base.sample(params, key, (128,))
        log_prob = base.log_prob(params, samples)

        assert samples.shape == (128, domain.dim)
        assert log_prob.shape == (128,)
        assert bool(jnp.all(jnp.isfinite(log_prob)))
        assert bool(jnp.all(samples[:, 1] >= -2.0))
        assert bool(jnp.all(samples[:, 1] <= 3.0))
        assert bool(jnp.all(samples[:, 2] >= -jnp.pi))
        assert bool(jnp.all(samples[:, 2] < jnp.pi))

    def test_sample_respects_default_float_precision(self, key):
        domain = mixed_domain()
        base = ProductBase(domain)

        samples = base.sample(base.init_params(), key, (8,))

        expected_dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
        assert samples.dtype == expected_dtype

    def test_out_of_domain_gets_negative_infinity(self):
        domain = mixed_domain()
        base = ProductBase(domain)
        x = jnp.array([[0.0, 4.0, 0.0, 0.0]])

        log_prob = base.log_prob(base.init_params(), x)

        assert jnp.isneginf(log_prob[0])

    def test_jit_compatible(self, key):
        domain = mixed_domain()
        base = ProductBase(domain)
        params = base.init_params()

        sample = jax.jit(lambda k: base.sample(params, k, (5,)))(key)
        log_prob = jax.jit(base.log_prob)(params, sample)

        assert sample.shape == (5, domain.dim)
        assert log_prob.shape == (5,)


class TestCircularCoordinateShift:
    """Tests for circular-only coordinate shifts on flat events."""

    def test_round_trip_and_non_circular_unchanged(self, key):
        domain = mixed_domain()
        shift, params = CircularCoordinateShift.create(key, domain)
        params = {"shift": params["shift"].at[2].set(1.25)}
        x = jnp.array([[0.2, 0.5, 2.8, -0.1]])

        y, ld_f = shift.forward(params, x)
        x_back, ld_i = shift.inverse(params, y)

        assert jnp.allclose(y[:, [0, 1, 3]], x[:, [0, 1, 3]])
        assert jnp.all(y[:, 2] >= -jnp.pi)
        assert jnp.all(y[:, 2] < jnp.pi)
        assert jnp.allclose(x_back, x, atol=1e-6)
        assert jnp.allclose(ld_f + ld_i, 0.0)

    def test_wraps_near_circular_seam(self, key):
        domain = mixed_domain()
        shift, params = CircularCoordinateShift.create(key, domain)
        params = {"shift": params["shift"].at[2].set(0.25)}
        eps = jnp.asarray(1e-5)
        x = jnp.array([[0.0, 0.0, jnp.pi - eps, 0.0], [0.0, 0.0, -jnp.pi + eps, 0.0]])

        y, _ = shift.forward(params, x)
        x_back, _ = shift.inverse(params, y)

        assert bool(jnp.all(y[:, 2] >= -jnp.pi))
        assert bool(jnp.all(y[:, 2] < jnp.pi))
        assert jnp.allclose(x_back, x, atol=1e-5)


class TestProductSplineCoupling:
    """Tests for mixed-domain spline coupling."""

    def test_fast_feature_map_matches_domain_helper(self, key):
        domain = mixed_domain()
        mask = jnp.array([1.0, 1.0, 1.0, 0.0])
        coupling, _ = ProductSplineCoupling.create(
            key,
            domain=domain,
            mask=mask,
            hidden_dim=8,
            n_hidden_layers=1,
            num_bins=4,
            circular_n_freq=2,
        )
        x = jnp.array([[0.5, 0.5, 0.0, -1.0], [0.2, -0.5, 1.0, 0.3]])

        expected = domain.conditioner_features(x, mask, circular_n_freq=2)
        actual = coupling._conditioner_features(x)

        assert jnp.allclose(actual, expected)

    def test_all_frozen_or_all_transformed_masks_raise(self, key):
        domain = mixed_domain()
        with pytest.raises(ValueError, match="freeze at least one"):
            ProductSplineCoupling.create(
                key,
                domain=domain,
                mask=jnp.ones((domain.dim,)),
                hidden_dim=16,
                n_hidden_layers=2,
            )
        with pytest.raises(ValueError, match="freeze at least one"):
            ProductSplineCoupling.create(
                key,
                domain=domain,
                mask=jnp.zeros((domain.dim,)),
                hidden_dim=16,
                n_hidden_layers=2,
            )

    def test_create_identity_and_round_trip(self, key):
        domain = mixed_domain()
        coupling, params = ProductSplineCoupling.create(
            key,
            domain=domain,
            mask=jnp.array([1.0, 0.0, 1.0, 0.0]),
            hidden_dim=8,
            n_hidden_layers=1,
            num_bins=4,
        )
        x = jnp.array([[0.3, -0.5, 2.8, 0.7], [-1.2, 2.5, -2.7, 0.1]])

        y, ld_f = coupling.forward(params, x)
        x_back, ld_i = coupling.inverse(params, y)

        assert jnp.allclose(y, x, atol=2e-2)
        assert jnp.allclose(x_back, x, atol=2e-4)
        assert jnp.allclose(ld_f + ld_i, 0.0, atol=2e-4)

    @pytest.mark.slow
    def test_logdet_vs_autodiff_away_from_circular_seam(self, key):
        domain = mixed_domain()
        coupling, params = ProductSplineCoupling.create(
            key,
            domain=domain,
            mask=jnp.array([1.0, 0.0, 1.0, 0.0]),
            hidden_dim=8,
            n_hidden_layers=1,
            num_bins=4,
        )
        params = jax.tree_util.tree_map(
            lambda p: p + 0.1 * jax.random.normal(key, p.shape),
            params,
        )
        x = jnp.array([0.2, 0.1, 0.3, -0.4])

        result = check_logdet_vs_autodiff(
            lambda z: coupling.forward(params, z),
            x,
            atol=2e-3,
        )

        assert result["error"] < 2e-3

    @pytest.mark.slow
    def test_interval_logdet_matches_autodiff(self, key):
        domain = ProductDomain(
            [
                ScalarDomain.real(),
                ScalarDomain.interval(-2.0, 3.0),
            ]
        )
        coupling, params = ProductSplineCoupling.create(
            key,
            domain=domain,
            mask=jnp.array([1.0, 0.0]),
            hidden_dim=8,
            n_hidden_layers=1,
            num_bins=4,
        )
        params = jax.tree_util.tree_map(
            lambda p: p + 0.1 * jax.random.normal(key, p.shape),
            params,
        )
        x = jnp.array([0.3, 0.25])

        result = check_logdet_vs_autodiff(
            lambda z: coupling.forward(params, z),
            x,
            atol=2e-3,
        )

        assert result["error"] < 2e-3

    def test_jit_round_trip(self, key):
        domain = mixed_domain()
        coupling, params = ProductSplineCoupling.create(
            key,
            domain=domain,
            mask=jnp.array([0.0, 1.0, 0.0, 1.0]),
            hidden_dim=8,
            n_hidden_layers=1,
            num_bins=4,
        )
        x = jnp.array([[0.1, -0.3, 0.5, 0.7]])

        y, _ = jax.jit(coupling.forward)(params, x)
        x_back, _ = jax.jit(coupling.inverse)(params, y)

        assert jnp.allclose(x_back, x, atol=2e-4)


class TestProductSplineBuilder:
    """Tests for the high-level product-domain builder."""

    def test_rejects_custom_base_with_wrong_event_shape(self, key):
        domain = mixed_domain()
        with pytest.raises(ValueError, match="base_dist.event_shape"):
            build_product_spline_flow(
                key,
                domain=domain,
                num_layers=2,
                hidden_dim=4,
                n_hidden_layers=1,
                base_dist=StandardNormal(dim=domain.dim + 1),
            )

    def test_return_transform_only_skips_custom_base_validation(self, key):
        domain = mixed_domain()
        bijection, params = build_product_spline_flow(
            key,
            domain=domain,
            num_layers=2,
            hidden_dim=4,
            n_hidden_layers=1,
            base_dist=StandardNormal(dim=domain.dim + 1),
            return_transform_only=True,
        )

        assert isinstance(bijection, Bijection)
        assert set(params.keys()) == {"transform"}

    def test_invalid_tail_bound_raises(self, key):
        domain = mixed_domain()
        with pytest.raises(ValueError, match="tail_bound must be positive"):
            build_product_spline_flow(
                key,
                domain=domain,
                num_layers=2,
                hidden_dim=4,
                n_hidden_layers=1,
                tail_bound=0.0,
            )

    def test_builds_flow_and_preserves_domain(self, key):
        domain = mixed_domain()
        flow, params = build_product_spline_flow(
            key,
            domain=domain,
            num_layers=2,
            hidden_dim=4,
            n_hidden_layers=1,
            num_bins=3,
        )

        assert isinstance(flow, Flow)
        samples = flow.sample(params, key, (32,))
        log_prob = flow.log_prob(params, samples)

        assert samples.shape == (32, domain.dim)
        assert log_prob.shape == (32,)
        assert bool(jnp.all(jnp.isfinite(log_prob)))
        assert bool(jnp.all(samples[:, 1] >= -2.0))
        assert bool(jnp.all(samples[:, 1] <= 3.0))
        assert bool(jnp.all(samples[:, 2] >= -jnp.pi))
        assert bool(jnp.all(samples[:, 2] < jnp.pi))

    def test_identity_gate_zero_is_identity(self, key):
        domain = mixed_domain()
        flow, params = build_product_spline_flow(
            key,
            domain=domain,
            num_layers=1,
            hidden_dim=4,
            n_hidden_layers=1,
            context_dim=1,
            identity_gate=lambda ctx: ctx[0],
        )
        x = jnp.array([[0.1, 0.2, -0.3, 0.4]])
        context = jnp.array([[0.0]])

        y, log_det = flow.forward(params, x, context=context)

        assert jnp.allclose(y, x, atol=1e-6)
        assert jnp.allclose(log_det, 0.0, atol=1e-6)

    def test_context_shape_mismatch_raises(self, key):
        domain = mixed_domain()
        flow, params = build_product_spline_flow(
            key,
            domain=domain,
            num_layers=1,
            hidden_dim=4,
            n_hidden_layers=1,
            context_dim=2,
        )
        x = jnp.array([[0.1, 0.2, -0.3, 0.4]])
        bad_context = jnp.array([[0.0]])

        with pytest.raises(ValueError, match="context"):
            flow.forward(params, x, context=bad_context)

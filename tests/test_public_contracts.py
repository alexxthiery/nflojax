"""Smoke tests for blessed public entry points.

These tests intentionally exercise user-facing construction paths rather than
implementation details. Keep them small and stable.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

import nflojax.transforms as transforms
from nflojax.builders import (
    assemble_flow,
    build_particle_flow,
    build_product_spline_flow,
    build_realnvp,
    build_spline_realnvp,
    make_alternating_mask,
)
from nflojax.distributions import ProductBase, StandardNormal, UniformBox
from nflojax.domains import ProductDomain, ScalarDomain
from nflojax.flows import Bijection, Flow
from nflojax.geometry import Geometry
from nflojax.nets import DeepSets
from nflojax.transforms import AffineCoupling


def test_transform_facade_exports_public_helpers_only():
    """Transform facade keeps public helpers, not underscore internals."""
    assert "identity_spline_bias" in transforms.__all__
    assert "stable_logit" in transforms.__all__
    assert "validate_identity_gate" in transforms.__all__
    assert "_compute_gate_value" not in transforms.__all__
    assert "_params_per_scalar" not in transforms.__all__
    assert "_validate_boundary_slopes" not in transforms.__all__


def test_direct_module_paths_match_facades():
    """Documented direct module paths resolve to facade objects."""
    from nflojax.builders.flat import build_realnvp as direct_realnvp
    from nflojax.builders.particle import build_particle_flow as direct_particle
    from nflojax.builders.product import build_product_spline_flow as direct_product
    from nflojax.transforms.couplings.affine import AffineCoupling as direct_affine
    from nflojax.transforms.couplings.spline import SplineCoupling as direct_spline
    from nflojax.transforms.couplings.split import SplitCoupling as direct_split
    from nflojax.transforms import SplineCoupling, SplitCoupling

    assert direct_realnvp is build_realnvp
    assert direct_product is build_product_spline_flow
    assert direct_particle is build_particle_flow
    assert direct_affine is AffineCoupling
    assert direct_spline is SplineCoupling
    assert direct_split is SplitCoupling


def _assert_finite_flow(flow: Flow, params, key, x, *, context=None) -> None:
    log_prob = jax.jit(lambda p, z, c: flow.log_prob(p, z, context=c))(
        params, x, context
    )
    samples = jax.jit(lambda p, k: flow.sample(p, k, shape=(4,)))(params, key)

    assert log_prob.shape == x.shape[:-1]
    assert samples.shape[0] == 4
    assert bool(jnp.all(jnp.isfinite(log_prob)))
    assert bool(jnp.all(jnp.isfinite(samples)))


def test_build_realnvp_public_contract(key):
    """Affine RealNVP builds, evaluates, samples, and jits."""
    flow, params = build_realnvp(
        key,
        dim=4,
        num_layers=2,
        hidden_dim=8,
        n_hidden_layers=1,
        use_loft=False,
    )
    x = jax.random.normal(key, (3, 4))

    assert isinstance(flow, Flow)
    _assert_finite_flow(flow, params, key, x)


def test_build_spline_realnvp_public_contract(key):
    """Spline RealNVP builds, evaluates, samples, and jits."""
    flow, params = build_spline_realnvp(
        key,
        dim=4,
        num_layers=2,
        hidden_dim=8,
        n_hidden_layers=1,
        num_bins=5,
        use_loft=False,
    )
    x = jax.random.normal(key, (3, 4))

    assert isinstance(flow, Flow)
    _assert_finite_flow(flow, params, key, x)


def test_build_product_spline_flow_public_contract(key):
    """Product-domain flow preserves mixed real/interval/circular events."""
    domain = ProductDomain(
        [
            ScalarDomain.real(),
            ScalarDomain.interval(-1.0, 2.0),
            ScalarDomain.circular(-jnp.pi, jnp.pi),
            ScalarDomain.real(),
        ]
    )
    flow, params = build_product_spline_flow(
        key,
        domain=domain,
        num_layers=2,
        hidden_dim=8,
        n_hidden_layers=1,
        num_bins=5,
        base_dist=ProductBase(domain),
    )
    x = ProductBase(domain).sample(None, key, (3,))

    assert isinstance(flow, Flow)
    _assert_finite_flow(flow, params, key, x)


def test_build_particle_flow_public_contract(key):
    """Particle-flow builder works with the structured conditioner contract."""
    geometry = Geometry.cubic(d=2, side=2.0)
    event_shape = (4, 2)

    def conditioner_factory(*, required_out_dim, **_):
        return DeepSets(phi_hidden=(8,), rho_hidden=(8,), out_dim=required_out_dim)

    flow, params = build_particle_flow(
        key,
        geometry=geometry,
        event_shape=event_shape,
        num_layers=1,
        conditioner=conditioner_factory,
        base_dist=UniformBox(geometry, event_shape=event_shape),
        num_bins=5,
    )
    x = jax.random.uniform(key, (3,) + event_shape, minval=0.0, maxval=2.0)
    log_prob = jax.jit(lambda p, z: flow.log_prob(p, z))(params, x)

    assert isinstance(flow, Flow)
    assert log_prob.shape == (3,)
    assert bool(jnp.all(jnp.isfinite(log_prob)))


def test_transform_only_bijection_public_contract(key):
    """Transform-only mode returns a jit-compatible Bijection."""
    bijection, params = build_spline_realnvp(
        key,
        dim=4,
        num_layers=2,
        hidden_dim=8,
        n_hidden_layers=1,
        num_bins=5,
        use_loft=False,
        return_transform_only=True,
    )
    x = jax.random.normal(key, (3, 4))
    y, log_det = jax.jit(lambda p, z: bijection.forward(p, z))(params, x)

    assert isinstance(bijection, Bijection)
    assert y.shape == x.shape
    assert log_det.shape == (3,)
    assert bool(jnp.all(jnp.isfinite(y)))
    assert bool(jnp.all(jnp.isfinite(log_det)))


def test_custom_assembly_public_contract(key):
    """Manual assembly remains a supported mid-level construction path."""
    keys = jax.random.split(key, 2)
    blocks_and_params = [
        AffineCoupling.create(
            keys[0],
            dim=4,
            mask=make_alternating_mask(4, parity=0),
            hidden_dim=8,
            n_hidden_layers=1,
        ),
        AffineCoupling.create(
            keys[1],
            dim=4,
            mask=make_alternating_mask(4, parity=1),
            hidden_dim=8,
            n_hidden_layers=1,
        ),
    ]
    flow, params = assemble_flow(blocks_and_params, base=StandardNormal(dim=4))
    x = jax.random.normal(key, (3, 4))

    assert isinstance(flow, Flow)
    _assert_finite_flow(flow, params, key, x)

# tests/test_particle_smoke.py
"""
Stage-E smoke tests.

Drives `build_particle_flow` with each of the three permutation-aware
reference conditioners (`DeepSets`, `Transformer`, `GNN`) and asserts
the three properties a real particle flow needs at init:

  - identity-on-couplings (forward == Rescale scaling);
  - jit-invertible round-trip after perturbing each `dense_out`;
  - non-zero gradient from a trivial loss involving `log_det`.

Separate from `test_particle_integration.py` (which drives the same
coverage through raw `SplitCoupling` assembly) -- this file proves the
builder composes cleanly with every reference conditioner.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.slow

from nflojax.builders import assemble_flow, build_particle_flow
from nflojax.distributions import DiagNormal, UniformBox
from nflojax.geometry import Geometry
from nflojax.nets import DeepSets, GNN, Transformer
from nflojax.transforms import SplitCoupling
from conftest import requires_x64


@pytest.fixture
def key():
    return jax.random.PRNGKey(0)


@pytest.fixture
def cfg():
    return {"N": 4, "d": 2, "K": 4, "num_layers": 1, "tail_bound": 5.0}


def _make_conditioner_factory(name, *, geometry):
    """Build a keyword-only conditioner factory matching the builder contract."""
    if name == "DeepSets":
        return lambda *, required_out_dim, **_: DeepSets(
            phi_hidden=(8,), rho_hidden=(8,), out_dim=required_out_dim,
        )
    if name == "Transformer":
        return lambda *, out_per_particle, **_: Transformer(
            num_layers=1, num_heads=2, embed_dim=8,
            out_per_particle=out_per_particle,
        )
    if name == "GNN":
        return lambda *, out_per_particle, **_: GNN(
            num_layers=1, hidden=8, out_per_particle=out_per_particle,
            num_neighbours=1, geometry=geometry,
        )
    raise ValueError(f"unknown conditioner: {name}")


def _build(name, cfg, key):
    N, d = cfg["N"], cfg["d"]
    g = Geometry.cubic(d=d, side=2.0, lower=-1.0)
    factory = _make_conditioner_factory(name, geometry=g)
    return build_particle_flow(
        key,
        geometry=g,
        event_shape=(N, d),
        num_layers=cfg["num_layers"],
        conditioner=factory,
        base_dist=UniformBox(geometry=g, event_shape=(N, d)),
        num_bins=cfg["K"],
        tail_bound=cfg["tail_bound"],
    )


@pytest.mark.parametrize("conditioner_name", ["DeepSets", "Transformer", "GNN"])
class TestParticleFlowBuilderSmoke:
    """Build the canonical topology with each reference conditioner."""

    def test_identity_on_couplings(self, conditioner_name, cfg, key):
        """At init the couplings are identity; forward == Rescale scaling."""
        flow, params = _build(conditioner_name, cfg, key)
        N, d = cfg["N"], cfg["d"]
        scale = cfg["tail_bound"] / 1.0  # box half-side = 1.0
        x = jax.random.uniform(key, (3, N, d), minval=-1.0, maxval=1.0)
        y, log_det = flow.forward(params, x)
        assert jnp.allclose(y, scale * x, atol=1e-5)
        expected_ld = N * d * math.log(scale)
        assert jnp.allclose(log_det, expected_ld, atol=1e-4)

    @requires_x64
    def test_jit_round_trip_after_perturbation(self, conditioner_name, cfg, key):
        """Randomise every dense_out then round-trip under jit.

        Skipped under float32: with circular splines + Rescale scaling,
        RQS-inverse roundoff accumulates above 1e-3 across four stacked
        couplings. Same failure mode as the ``@requires_x64`` round-trip
        tests in ``test_transforms.py`` and ``test_splines.py``.
        """
        flow, params = _build(conditioner_name, cfg, key)
        N, d = cfg["N"], cfg["d"]
        coupling_indices = [
            i for i, b in enumerate(flow.transform.blocks)
            if hasattr(b, "conditioner")
        ]
        sub_keys = jax.random.split(key, len(coupling_indices))
        for i, k in zip(coupling_indices, sub_keys):
            cond = flow.transform.blocks[i].conditioner
            current = cond.get_output_layer(params["transform"][i]["mlp"])
            kernel = jax.random.normal(k, current["kernel"].shape) * 0.05
            bias = jax.random.normal(k, current["bias"].shape) * 0.05
            params["transform"][i]["mlp"] = cond.set_output_layer(
                params["transform"][i]["mlp"], kernel, bias,
            )
        fwd = jax.jit(lambda p, z: flow.forward(p, z))
        inv = jax.jit(lambda p, z: flow.inverse(p, z))
        x = jax.random.uniform(key, (3, N, d), minval=-0.9, maxval=0.9)
        y, _ = fwd(params, x)
        x_back, _ = inv(params, y)
        # Same tolerance as test_particle_integration.py (float32 RQS-inverse
        # roundoff accumulates).
        assert jnp.allclose(x_back, x, atol=1e-3)

    @pytest.mark.slow
    def test_gradient_flows_end_to_end(self, conditioner_name, cfg, key):
        """Non-zero gradient on at least one dense_out kernel."""
        flow, params = _build(conditioner_name, cfg, key)
        N, d = cfg["N"], cfg["d"]
        x = jax.random.uniform(key, (2, N, d), minval=-0.9, maxval=0.9)

        def loss(p):
            y, log_det = flow.forward(p, x)
            return jnp.sum((y - x) ** 2) + jnp.sum(log_det)

        grads = jax.grad(loss)(params)
        total = 0.0
        for i, block in enumerate(flow.transform.blocks):
            if not hasattr(block, "conditioner"):
                continue
            g_out = block.conditioner.get_output_layer(grads["transform"][i]["mlp"])
            total += float(jnp.sum(jnp.abs(g_out["kernel"])))
        assert total > 0.0


@pytest.mark.slow
def test_free_cluster_assemble(key):
    """Non-periodic structured-particle pipeline composes end-to-end.

    Validates the path free-cluster benchmarks depend on: `Geometry` with
    `periodic=[False]*d`, `DiagNormal` base on `(N, d)`, a stack of
    `SplitCoupling(boundary_slopes='linear_tails', flatten_input=False)`
    with `DeepSets` conditioners, assembled via `assemble_flow`. Bypasses
    `build_particle_flow` (which hard-wires `CircularShift`) because a
    free cluster has no box to wrap around.

    Asserts identity-at-init (linear tails are identity inside
    `[-tail_bound, tail_bound]^d`) and an inverse round-trip. Gradient and JIT
    coverage lives in the smaller structured-particle tests.
    """
    N, d = 5, 3
    num_bins, num_layers = 4, 2
    split_index = N // 2  # asymmetric 6/7 split; swap flips the sizing.
    tail_bound = 5.0

    # Type-level check that the non-periodic Geometry path works at all;
    # the flow below does not consume the geometry (no PBC, no Rescale).
    g = Geometry(
        lower=[-2.0] * d, upper=[2.0] * d, periodic=[False] * d,
    )
    assert not g.is_periodic()

    base = DiagNormal(event_shape=(N, d))

    keys = jax.random.split(key, num_layers)
    blocks_and_params = []
    for i, k in enumerate(keys):
        swap = bool(i % 2)
        n_frozen = N - split_index if swap else split_index
        n_transformed = N - n_frozen
        out_dim = SplitCoupling.required_out_dim(
            n_transformed * d, num_bins, boundary_slopes="linear_tails",
        )
        cond = DeepSets(
            phi_hidden=(8,), rho_hidden=(8,), out_dim=out_dim,
        )
        coupling = SplitCoupling(
            event_shape=(N, d),
            split_axis=-2,
            split_index=split_index,
            event_ndims=2,
            conditioner=cond,
            swap=swap,
            num_bins=num_bins,
            tail_bound=tail_bound,
            boundary_slopes="linear_tails",
            flatten_input=False,
        )
        blocks_and_params.append((coupling, coupling.init_params(k)))

    # validate=False because analyze_mask_coverage only understands flat masks
    # (SplitCoupling has no `mask` attribute; alternating swap covers
    # particles geometrically).
    flow, params = assemble_flow(blocks_and_params, base=base, validate=False)

    # Identity-at-init: linear-tail splines are identity inside the active
    # region at zero-kernel dense_out + identity-spline bias.
    x = jax.random.uniform(key, (3, N, d), minval=-0.9, maxval=0.9)
    y, log_det = flow.forward(params, x)
    assert jnp.allclose(y, x, atol=1e-5)
    assert jnp.allclose(log_det, 0.0, atol=1e-4)

    # Round-trip at init: the assembled free-cluster topology inverts cleanly.
    x_back, _ = flow.inverse(params, y)
    assert jnp.allclose(x_back, x, atol=1e-5)


class TestParticleFlowBuilderCoMShift:
    """Cover the CoM-shift branch with one conditioner (branch is conditioner-
    agnostic, so one is enough)."""

    def test_sample_is_zero_com(self, cfg, key):
        N, d = cfg["N"], cfg["d"]
        g = Geometry.cubic(d=d, side=2.0, lower=-1.0)
        factory = _make_conditioner_factory("DeepSets", geometry=g)
        flow, params = build_particle_flow(
            key, geometry=g, event_shape=(N, d),
            num_layers=cfg["num_layers"], conditioner=factory,
            base_dist=UniformBox(geometry=g, event_shape=(N - 1, d)),
            num_bins=cfg["K"], tail_bound=cfg["tail_bound"],
            use_com_shift=True,
        )
        samples = flow.sample(params, key, (4,))
        assert samples.shape == (4, N, d)
        assert jnp.allclose(jnp.sum(samples, axis=-2), 0.0, atol=1e-5)
        logp = flow.log_prob(params, samples)
        assert logp.shape == (4,)
        assert bool(jnp.all(jnp.isfinite(logp)))

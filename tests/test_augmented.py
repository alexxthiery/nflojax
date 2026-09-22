# tests/test_augmented.py
"""``build_augmented_flow``: the Pattern B composition, promoted to a builder.

Pattern B doubles the event to ``(2N, d)``: ``N`` physical particles and ``N``
auxiliary ones, split **across that boundary** rather than along the particle
axis. Two consequences are the reason it exists, and both are asserted here:

  - every coupling conditions on a **full copy of all N particles**, where a
    particle-axis split conditions on half of them. That is the gap a downstream
    consumer measured at mW N=8 (4 particles of 8 visible to the conditioner);
  - both partitions are exactly ``N``, so per-token conditioners work for **odd
    N** too, which the particle split cannot do.

The rest is the same contract as ``build_particle_flow``: identity on the
couplings at init, an invertible stack, a log-det that matches autodiff, and the
periodic/linear-tails guard. The auxiliary *target* (what the second half means
physically) belongs to the application, not here; see EXTENDING.md.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from nflojax.builders import build_augmented_flow, build_particle_flow
from nflojax.distributions import DiagNormal
from nflojax.geometry import Geometry
from nflojax.nets import Transformer
from nflojax.transforms import Rescale, SplitCoupling
from conftest import check_logdet_vs_autodiff, requires_x64

N, D, BINS, LAYERS = 4, 3, 4, 2
BOX = 6.0


def _geometry(periodic=None):
    return Geometry.cubic(d=D, side=BOX, lower=0.0) if periodic is None else Geometry(
        lower=jnp.zeros(D), upper=jnp.full(D, BOX), periodic=periodic)


def _conditioner(*, out_per_particle, geometry, **_):
    return Transformer(num_layers=1, num_heads=2, embed_dim=8,
                       out_per_particle=out_per_particle, geometry=geometry)


def _flow(key, n=N, num_layers=LAYERS, **kw):
    base = DiagNormal(event_shape=(2 * n, D))
    kw.setdefault("num_bins", BINS)
    kw.setdefault("conditioner", _conditioner)
    return build_augmented_flow(
        key, geometry=_geometry(), event_shape=(n, D), num_layers=num_layers,
        base_dist=base, base_params=base.init_params(), **kw)


def _perturb(params, key, scale=0.1):
    leaves = jax.tree_util.tree_leaves(params)
    keys = list(jax.random.split(key, len(leaves)))
    return jax.tree_util.tree_map(
        lambda p, k: p + scale * jax.random.normal(k, p.shape), params,
        jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(params), keys))


class TestTheSplitIsPhysicalVersusAuxiliary:
    def test_every_coupling_splits_at_the_boundary(self, key):
        """The defining property. A particle-axis split would show
        ``split_index = N // 2`` on an event of ``(N, d)``; Pattern B must show
        ``split_index = N`` on ``(2N, d)`` in **every** coupling, both swaps."""
        flow, _ = _flow(key)
        couplings = [b for b in flow.transform.blocks if isinstance(b, SplitCoupling)]
        assert len(couplings) == 2 * LAYERS
        for c in couplings:
            assert c.event_shape == (2 * N, D)
            assert c.split_index == N and c.split_axis == -2 and c.event_ndims == 2
        assert {c.swap for c in couplings} == {False, True}   # both halves transform

    def test_both_halves_have_the_same_size_for_either_swap(self, key):
        """Why per-token conditioners always fit here: frozen and transformed are
        both N whichever way the swap falls, so ``required_out_dim`` does not
        depend on it."""
        for swap in (False, True):
            frozen, transformed = SplitCoupling._partition_flats(
                (2 * N, D), -2, N, 2, swap)
            assert frozen == transformed == N * D

    def test_odd_particle_counts_work_here_but_not_with_a_particle_split(self, key):
        """The concrete advantage over ``build_particle_flow``: with an odd N a
        per-token conditioner cannot serve a particle-axis split (the two halves
        differ in size), while the physical/auxiliary split is always even."""
        flow, params = _flow(key, n=5)
        x, _ = flow.sample_and_log_prob(params, key, (2,))
        assert x.shape == (2, 10, D)
        base = DiagNormal(event_shape=(5, D))
        with pytest.raises(ValueError):
            build_particle_flow(
                key, geometry=_geometry(), event_shape=(5, D), num_layers=1,
                conditioner=_conditioner, base_dist=base,
                base_params=base.init_params(), num_bins=BINS)


class TestContract:
    def test_samples_are_the_augmented_event(self, key):
        flow, params = _flow(key)
        x, log_q = flow.sample_and_log_prob(params, key, (8,))
        assert x.shape == (8, 2 * N, D) and log_q.shape == (8,)
        assert jnp.all(jnp.isfinite(log_q))

    def test_identity_on_the_couplings_at_init(self, key):
        """At init the couplings are the identity, so the whole forward map is
        just the ``Rescale`` into the spline cube. Pins the composition."""
        flow, params = _flow(key)
        rescale = next(b for b in flow.transform.blocks if isinstance(b, Rescale))
        z = jax.random.uniform(key, (4, 2 * N, D), minval=0.5, maxval=BOX - 0.5)
        expected, _ = rescale.forward(params["transform"][0], z)
        got, _ = flow.transform.forward(params["transform"], z)
        assert jnp.allclose(got, expected, atol=1e-5)

    @requires_x64
    def test_round_trip_with_perturbed_parameters(self, key):
        flow, params = _flow(key)
        noisy = _perturb(params["transform"], key)
        z = jax.random.uniform(key, (4, 2 * N, D), minval=0.5, maxval=BOX - 0.5)
        y, ld_f = flow.transform.forward(noisy, z)
        back, ld_i = flow.transform.inverse(noisy, y)
        assert jnp.allclose(back, z, atol=1e-6)
        assert jnp.allclose(ld_f, -ld_i, atol=1e-6)

    @requires_x64
    def test_log_det_matches_autodiff(self, key):
        """The log-det must be summed over both event axes of the transformed
        half only; autodiff on the full Jacobian is the independent oracle."""
        flow, params = _flow(key, num_layers=1)
        noisy = _perturb(params["transform"], key)
        z = jax.random.uniform(key, (2 * N, D), minval=0.5, maxval=BOX - 0.5)
        flat = z.reshape(-1)

        def forward_flat(v):
            y, ld = flow.transform.forward(noisy, v.reshape(2 * N, D))
            return y.reshape(-1), ld

        got = check_logdet_vs_autodiff(forward_flat, flat, atol=1e-5)
        assert got["error"] < 1e-5, got

    def test_gradients_reach_every_coupling(self, key):
        """A loss through ``log_q`` must move every coupling's parameters, or a
        layer is silently dead."""
        flow, params = _flow(key)
        grads = jax.grad(lambda p: jnp.sum(flow.sample_and_log_prob(p, key, (4,))[1]))(params)
        for i, block in enumerate(flow.transform.blocks):
            if isinstance(block, SplitCoupling):
                leaves = jax.tree_util.tree_leaves(grads["transform"][i])
                assert any(float(jnp.max(jnp.abs(g))) > 0 for g in leaves), f"block {i} dead"

    def test_feature_map_reaches_the_couplings(self, key):
        """The application conditions on something other than the raw frozen
        half (a lattice-site encoding, say), which ``SplitCoupling.feature_map``
        exists for; the builder must pass it through.

        A map that widens the last axis needs a conditioner that does not apply
        its own coordinate embedding, since that embedding expects exactly ``d``
        columns: pass the features yourself or leave ``geometry`` unused."""
        seen = []

        def feature_map(frozen):
            seen.append(frozen.shape)
            return jnp.concatenate([frozen, jnp.sin(frozen)], axis=-1)

        def plain_conditioner(*, out_per_particle, **_):
            return Transformer(num_layers=1, num_heads=2, embed_dim=8,
                               out_per_particle=out_per_particle)

        flow, params = _flow(key, feature_map=feature_map, conditioner=plain_conditioner)
        couplings = [b for b in flow.transform.blocks if isinstance(b, SplitCoupling)]
        assert all(c.feature_map is feature_map for c in couplings)
        flow.sample_and_log_prob(params, key, (2,))
        assert seen and seen[-1][-2:] == (N, D)      # the structured frozen half


class TestValidation:
    def test_a_periodic_box_rejects_linear_tails(self, key):
        with pytest.raises(ValueError, match="improper"):
            _flow(key, boundary_slopes="linear_tails")

    def test_a_base_on_the_physical_event_is_rejected(self, key):
        """The base lives on the augmented event; a base on ``(N, d)`` is the
        obvious mistake and must fail with a message that says so."""
        base = DiagNormal(event_shape=(N, D))
        with pytest.raises(ValueError, match="2N"):
            build_augmented_flow(
                key, geometry=_geometry(), event_shape=(N, D), num_layers=1,
                conditioner=_conditioner, base_dist=base,
                base_params=base.init_params(), num_bins=BINS)

    @pytest.mark.parametrize("kw", [{"num_layers": 0}, {"num_bins": 0},
                                    {"tail_bound": 0.0}])
    def test_nonpositive_sizes_are_rejected(self, key, kw):
        with pytest.raises(ValueError):
            _flow(key, **kw)

    def test_a_single_particle_is_rejected(self, key):
        with pytest.raises(ValueError):
            _flow(key, n=1)


class TestTranslation:
    def test_the_stack_is_not_translation_equivariant(self, key):
        """A documented claim that was false, pinned so it cannot come back.

        EXTENDING.md used to say the augmented pattern makes the joint
        distribution invariant under a simultaneous translation of both halves,
        so no centre-of-mass handling was needed. It does not: the base is a
        zero-centred Gaussian (not translation invariant), and each spline acts
        on the **absolute** coordinates of the transformed half, so
        ``f(x + c) != f(x) + c``. A translation-invariant target therefore still
        needs application-side handling of the flat direction.

        The measured error is of the order of the output itself, not a tolerance
        question.
        """
        flow, params = _flow(key)
        noisy = _perturb(params["transform"], key, scale=0.3)
        x = jax.random.uniform(key, (4, 2 * N, D), minval=0.5, maxval=BOX - 0.5)
        c = jnp.array([0.7, -1.3, 2.1])
        y, _ = flow.transform.forward(noisy, x)
        y_shifted, _ = flow.transform.forward(noisy, jnp.mod(x + c, BOX))
        cube = 2 * 5.0                                   # default tail_bound
        err = jnp.max(jnp.abs(jnp.mod(y_shifted - y - c + cube / 2, cube) - cube / 2))
        assert float(err) > 0.1 * float(jnp.max(jnp.abs(y)))

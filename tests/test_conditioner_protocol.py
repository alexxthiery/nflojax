# tests/test_conditioner_protocol.py
"""
Shared conditioner-contract tests (PLAN.md Stage D4).

Parametrised over `MLP`, `DeepSets`, `Transformer`, and `GNN`; every
built-in conditioner must pass the same contract:

  1. `validate_conditioner` accepts it.
  2. `apply({"params": params}, x, context)` returns a shape whose total
     trailing size matches `SplitCoupling.required_out_dim(...)`.
  3. `get_output_layer` / `set_output_layer` round-trip.
  4. Wiring into `SplitCoupling(flatten_input=...)` initialises to
     identity (the flat/structured path is auto-selected per
     conditioner).
  5. `jax.jit` traces cleanly.

`MLP` is additionally tested with `SplineCoupling` (flat-mask path);
the particle-axis conditioners are tested only with `SplitCoupling`.
"""
from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

from nflojax.nets import (
    MLP,
    DeepSets,
    Transformer,
    GNN,
    init_mlp,
    init_conditioner,
    validate_conditioner,
)
from nflojax.geometry import Geometry
from nflojax.transforms import (
    SplitCoupling,
    SplineCoupling,
    identity_spline_bias,
)


# ---------------------------------------------------------------------------
# Shared problem definition: a half-half SplitCoupling on (N, d) events.
# ---------------------------------------------------------------------------
N = 8
D = 3
K_BINS = 4
PARAMS_PER_SCALAR = 3 * K_BINS - 1    # linear_tails
SPLIT_INDEX = N // 2
N_FROZEN = SPLIT_INDEX
N_TRANSFORMED = N - SPLIT_INDEX
TRANSFORMED_FLAT = N_TRANSFORMED * D
OUT_PER_PARTICLE = D * PARAMS_PER_SCALAR
OUT_DIM_FLAT = TRANSFORMED_FLAT * PARAMS_PER_SCALAR
GEOM = Geometry.cubic(d=D, side=2.0)


def _make_mlp(key):
    # `init_mlp` remains as a full-featured builder (kept for pre-Stage-D
    # `build_realnvp` etc.); the three Stage-D conditioners construct
    # directly via the dataclass + `init_conditioner`.
    mlp, params = init_mlp(
        key, x_dim=N_FROZEN * D, context_dim=0,
        hidden_dim=32, n_hidden_layers=2, out_dim=OUT_DIM_FLAT,
    )
    return mlp, params, {"flatten_input": True}


def _make_deepsets(key):
    ds = DeepSets(phi_hidden=(32, 32), rho_hidden=(32,), out_dim=OUT_DIM_FLAT)
    params = init_conditioner(key, ds, jnp.zeros((1, N_FROZEN, D)))
    return ds, params, {"flatten_input": False}


def _make_transformer(key):
    t = Transformer(
        num_layers=2, num_heads=2, embed_dim=16,
        out_per_particle=OUT_PER_PARTICLE,
    )
    params = init_conditioner(key, t, jnp.zeros((1, N_FROZEN, D)))
    return t, params, {"flatten_input": False}


def _make_gnn(key):
    gnn = GNN(
        num_layers=2, hidden=16, out_per_particle=OUT_PER_PARTICLE,
        num_neighbours=3, geometry=GEOM,
    )
    params = init_conditioner(key, gnn, jnp.zeros((1, N_FROZEN, D)))
    return gnn, params, {"flatten_input": False}


CONDITIONERS = [
    pytest.param(_make_mlp,         id="MLP"),
    pytest.param(_make_deepsets,    id="DeepSets"),
    pytest.param(_make_transformer, id="Transformer"),
    pytest.param(_make_gnn,         id="GNN"),
]


@pytest.fixture(scope="module")
def key():
    return jax.random.PRNGKey(0)


# ---------------------------------------------------------------------------
# Contract tests (run per conditioner)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("factory", CONDITIONERS)
def test_validate_conditioner_accepts(factory, key):
    cond, _params, _ = factory(key)
    validate_conditioner(cond)   # Must not raise.


@pytest.mark.parametrize("factory", CONDITIONERS)
def test_output_total_size_matches_required_out_dim(factory, key):
    """`apply` returns a tensor whose total trailing size matches
    `SplitCoupling.required_out_dim(transformed_flat, num_bins)`."""
    cond, params, cfg = factory(key)
    flatten = cfg["flatten_input"]
    # Pick a minimal dummy matching whichever path this conditioner expects.
    if flatten:
        x = jnp.zeros((1, N_FROZEN * D))
    else:
        x = jnp.zeros((1, N_FROZEN, D))
    out = cond.apply({"params": params}, x)
    # Trailing sizes may differ (flat vs per-particle) but total must match.
    total = int(jnp.prod(jnp.array(out.shape[1:])))
    assert total == SplitCoupling.required_out_dim(
        TRANSFORMED_FLAT, K_BINS, boundary_slopes="linear_tails"
    ) == OUT_DIM_FLAT


@pytest.mark.parametrize("factory", CONDITIONERS)
def test_get_set_output_layer_round_trip(factory, key):
    """get_output_layer then set_output_layer gives back-compatible params."""
    cond, params, _ = factory(key)
    out = cond.get_output_layer(params)
    assert set(out.keys()) == {"kernel", "bias"}
    # set_output_layer with the retrieved kernel/bias (optionally sliced for
    # per-token conditioners) must not raise and must preserve the shape.
    new_params = cond.set_output_layer(params, out["kernel"], out["bias"])
    new_out = cond.get_output_layer(new_params)
    assert new_out["kernel"].shape == out["kernel"].shape
    assert new_out["bias"].shape == out["bias"].shape


@pytest.mark.parametrize("factory", CONDITIONERS)
def test_split_coupling_identity_at_init(factory, key):
    """Identity-at-init holds for every conditioner when wired into a
    SplitCoupling with the matching flatten_input flag."""
    cond, _params_unused, cfg = factory(key)
    coupling = SplitCoupling(
        event_shape=(N, D),
        split_axis=-2,
        split_index=SPLIT_INDEX,
        event_ndims=2,
        conditioner=cond,
        num_bins=K_BINS,
        tail_bound=5.0,
        flatten_input=cfg["flatten_input"],
    )
    params = coupling.init_params(key)
    x = jax.random.uniform(key, (3, N, D), minval=-1.0, maxval=1.0)
    y, log_det = coupling.forward(params, x)
    assert jnp.allclose(y, x, atol=1e-5)
    assert jnp.allclose(log_det, 0.0, atol=1e-5)


@pytest.mark.parametrize("factory", CONDITIONERS)
def test_split_coupling_jits(factory, key):
    cond, _, cfg = factory(key)
    coupling = SplitCoupling(
        event_shape=(N, D),
        split_axis=-2,
        split_index=SPLIT_INDEX,
        event_ndims=2,
        conditioner=cond,
        num_bins=K_BINS,
        flatten_input=cfg["flatten_input"],
    )
    params = coupling.init_params(key)
    x = jax.random.uniform(key, (2, N, D), minval=-1.0, maxval=1.0)
    fwd = jax.jit(lambda p, z: coupling.forward(p, z))
    y, _ = fwd(params, x)
    assert y.shape == x.shape


# ---------------------------------------------------------------------------
# MLP-only: also works with SplineCoupling (flat-mask path).
# ---------------------------------------------------------------------------
def test_mlp_with_spline_coupling_identity(key):
    """MLP is the one conditioner that also plugs into SplineCoupling's
    flat-mask path. Verify identity-at-init there too."""
    dim = 8
    K = 4
    params_per_scalar = 3 * K - 1
    mask = jnp.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=jnp.float32)
    transformed_flat = int(jnp.sum(1 - mask))
    mlp, _ = init_mlp(
        key, x_dim=dim, context_dim=0,
        hidden_dim=32, n_hidden_layers=2,
        out_dim=SplineCoupling.required_out_dim(dim, K, "linear_tails"),
    )
    coupling = SplineCoupling(
        mask=mask, conditioner=mlp,
        num_bins=K, tail_bound=5.0,
    )
    params = coupling.init_params(key)
    x = jax.random.uniform(key, (4, dim), minval=-2.0, maxval=2.0)
    y, log_det = coupling.forward(params, x)
    assert jnp.allclose(y, x, atol=1e-5)
    assert jnp.allclose(log_det, 0.0, atol=1e-5)

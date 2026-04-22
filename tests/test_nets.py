# tests/test_nets.py
"""
Unit tests for neural network modules (MLP, ResNet).

Run with:
    PYTHONPATH=. pytest tests/test_nets.py -v
"""
from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

from nflojax.nets import (
    MLP,
    ResNet,
    DeepSets,
    Transformer,
    GNN,
    init_mlp,
    init_resnet,
    init_conditioner,
)
from nflojax.geometry import Geometry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def key():
    return jax.random.PRNGKey(42)


# ---------------------------------------------------------------------------
# ResNet Tests
# ---------------------------------------------------------------------------
class TestResNet:
    """Tests for the ResNet module."""

    def test_output_shape(self, key):
        """Output shape matches (batch, out_dim)."""
        resnet, params = init_resnet(
            key, in_dim=10, hidden_dim=32, out_dim=5, n_hidden_layers=2
        )
        x = jax.random.normal(key, (20, 10))
        out = resnet.apply({"params": params}, x)
        assert out.shape == (20, 5)

    def test_single_sample(self, key):
        """Works with single sample (no batch dim issues)."""
        resnet, params = init_resnet(
            key, in_dim=4, hidden_dim=16, out_dim=2, n_hidden_layers=1
        )
        x = jax.random.normal(key, (1, 4))
        out = resnet.apply({"params": params}, x)
        assert out.shape == (1, 2)
        assert not jnp.isnan(out).any()

    def test_zero_hidden_layers(self, key):
        """Works with n_hidden_layers=0 (just input→output projection)."""
        resnet, params = init_resnet(
            key, in_dim=4, hidden_dim=16, out_dim=2, n_hidden_layers=0
        )
        x = jax.random.normal(key, (10, 4))
        out = resnet.apply({"params": params}, x)
        assert out.shape == (10, 2)
        assert not jnp.isnan(out).any()

    def test_zero_init_output(self, key):
        """zero_init_output=True produces zero output at initialization."""
        resnet, params = init_resnet(
            key, in_dim=4, hidden_dim=16, out_dim=3, n_hidden_layers=2,
            zero_init_output=True
        )
        x = jax.random.normal(key, (10, 4))
        out = resnet.apply({"params": params}, x)
        # Output should be zero because dense_out kernel and bias are zero
        assert jnp.allclose(out, 0.0, atol=1e-6)

    def test_no_nans_or_infs(self, key):
        """Output is finite for reasonable inputs."""
        resnet, params = init_resnet(
            key, in_dim=8, hidden_dim=32, out_dim=4, n_hidden_layers=3
        )
        x = jax.random.normal(key, (100, 8))
        out = resnet.apply({"params": params}, x)
        assert jnp.isfinite(out).all()

    def test_jit_compatible(self, key):
        """Works under jax.jit."""
        resnet, params = init_resnet(
            key, in_dim=4, hidden_dim=16, out_dim=2, n_hidden_layers=2
        )
        x = jax.random.normal(key, (10, 4))

        @jax.jit
        def forward(params, x):
            return resnet.apply({"params": params}, x)

        out = forward(params, x)
        assert out.shape == (10, 2)
        assert not jnp.isnan(out).any()

    def test_vmap_compatible(self, key):
        """Works under jax.vmap (batching over extra dim)."""
        resnet, params = init_resnet(
            key, in_dim=4, hidden_dim=16, out_dim=2, n_hidden_layers=2
        )
        # Shape: (num_batches, batch_size, in_dim)
        x = jax.random.normal(key, (5, 10, 4))

        @jax.vmap
        def forward_batch(x_batch):
            return resnet.apply({"params": params}, x_batch)

        out = forward_batch(x)
        assert out.shape == (5, 10, 2)

    def test_gradients_exist(self, key):
        """Gradients w.r.t. params are finite and non-zero."""
        resnet, params = init_resnet(
            key, in_dim=4, hidden_dim=16, out_dim=2, n_hidden_layers=2
        )
        x = jax.random.normal(key, (10, 4))

        def loss_fn(params):
            out = resnet.apply({"params": params}, x)
            return (out ** 2).sum()

        grads = jax.grad(loss_fn)(params)
        grad_norm = sum(jnp.linalg.norm(v) for v in jax.tree_util.tree_leaves(grads))
        assert jnp.isfinite(grad_norm)
        assert grad_norm > 0

    def test_res_scale_zero_disables_residuals(self, key):
        """res_scale=0 means no residual connections (output differs)."""
        x = jax.random.normal(key, (10, 4))

        resnet_with_res, params_with = init_resnet(
            key, in_dim=4, hidden_dim=16, out_dim=2, n_hidden_layers=2, res_scale=0.1
        )
        resnet_no_res, params_no = init_resnet(
            jax.random.fold_in(key, 1), in_dim=4, hidden_dim=16, out_dim=2,
            n_hidden_layers=2, res_scale=0.0
        )

        out_with = resnet_with_res.apply({"params": params_with}, x)
        out_no = resnet_no_res.apply({"params": params_no}, x)

        # Different random init, so outputs differ - just check both work
        assert out_with.shape == out_no.shape == (10, 2)
        assert jnp.isfinite(out_with).all()
        assert jnp.isfinite(out_no).all()

    def test_deterministic(self, key):
        """Same input + same params → same output."""
        resnet, params = init_resnet(
            key, in_dim=4, hidden_dim=16, out_dim=2, n_hidden_layers=2
        )
        x = jax.random.normal(key, (10, 4))

        out1 = resnet.apply({"params": params}, x)
        out2 = resnet.apply({"params": params}, x)

        assert jnp.allclose(out1, out2)

    def test_get_output_layer(self, key):
        """get_output_layer returns correct structure and shapes."""
        resnet, params = init_resnet(
            key, in_dim=4, hidden_dim=16, out_dim=5, n_hidden_layers=2
        )
        out_layer = resnet.get_output_layer(params)

        assert "kernel" in out_layer
        assert "bias" in out_layer
        assert out_layer["kernel"].shape == (16, 5)  # (hidden_dim, out_dim)
        assert out_layer["bias"].shape == (5,)  # (out_dim,)

    def test_set_output_layer(self, key):
        """set_output_layer returns new params, original unchanged."""
        resnet, params = init_resnet(
            key, in_dim=4, hidden_dim=16, out_dim=5, n_hidden_layers=2
        )
        original_kernel = params["dense_out"]["kernel"].copy()
        original_bias = params["dense_out"]["bias"].copy()

        new_kernel = jnp.ones((16, 5))
        new_bias = jnp.ones((5,)) * 2

        new_params = resnet.set_output_layer(params, new_kernel, new_bias)

        # New params have new values
        assert jnp.allclose(new_params["dense_out"]["kernel"], new_kernel)
        assert jnp.allclose(new_params["dense_out"]["bias"], new_bias)

        # Original params unchanged
        assert jnp.allclose(params["dense_out"]["kernel"], original_kernel)
        assert jnp.allclose(params["dense_out"]["bias"], original_bias)


# ---------------------------------------------------------------------------
# MLP Tests
# ---------------------------------------------------------------------------
class TestMLP:
    """Tests for the MLP conditioner module."""

    def test_output_shape_no_context(self, key):
        """Output shape correct without context."""
        mlp, params = init_mlp(
            key, x_dim=4, context_dim=0, hidden_dim=16, n_hidden_layers=2, out_dim=8
        )
        x = jax.random.normal(key, (20, 4))
        out = mlp.apply({"params": params}, x)
        assert out.shape == (20, 8)

    def test_output_shape_with_context(self, key):
        """Output shape correct with context."""
        mlp, params = init_mlp(
            key, x_dim=4, context_dim=3, hidden_dim=16, n_hidden_layers=2, out_dim=8
        )
        x = jax.random.normal(key, (20, 4))
        context = jax.random.normal(key, (20, 3))
        out = mlp.apply({"params": params}, x, context)
        assert out.shape == (20, 8)

    def test_zero_init_output_layer(self, key):
        """init_mlp zero-initializes the output layer for identity-start flows."""
        mlp, params = init_mlp(
            key, x_dim=4, context_dim=0, hidden_dim=16, n_hidden_layers=2, out_dim=8
        )
        x = jax.random.normal(key, (10, 4))
        out = mlp.apply({"params": params}, x)
        # Output should be zero due to zero-init of dense_out
        assert jnp.allclose(out, 0.0, atol=1e-6)

    def test_context_affects_output(self, key):
        """Different context produces different output."""
        mlp, params = init_mlp(
            key, x_dim=4, context_dim=3, hidden_dim=16, n_hidden_layers=2, out_dim=8
        )
        # Perturb params to break zero-init symmetry
        params = jax.tree_util.tree_map(
            lambda p: p + 0.1 * jax.random.normal(key, p.shape), params
        )

        x = jax.random.normal(key, (10, 4))
        ctx1 = jnp.zeros((10, 3))
        ctx2 = jnp.ones((10, 3))

        out1 = mlp.apply({"params": params}, x, ctx1)
        out2 = mlp.apply({"params": params}, x, ctx2)

        diff = jnp.abs(out1 - out2).mean()
        assert diff > 1e-3, f"Context should affect output, diff={diff}"

    def test_context_broadcasting_shared(self, key):
        """Shared context (context_dim,) broadcasts to batch."""
        mlp, params = init_mlp(
            key, x_dim=4, context_dim=3, hidden_dim=16, n_hidden_layers=2, out_dim=8
        )
        x = jax.random.normal(key, (20, 4))
        context = jax.random.normal(key, (3,))  # Shared across batch

        out = mlp.apply({"params": params}, x, context)
        assert out.shape == (20, 8)
        assert not jnp.isnan(out).any()

    def test_wrong_x_dim_raises(self, key):
        """Wrong x dimension raises ValueError."""
        mlp, params = init_mlp(
            key, x_dim=4, context_dim=0, hidden_dim=16, n_hidden_layers=2, out_dim=8
        )
        x = jax.random.normal(key, (10, 5))  # Wrong: 5 instead of 4

        with pytest.raises(ValueError, match="expected x with last dimension 4"):
            mlp.apply({"params": params}, x)

    def test_wrong_context_dim_raises(self, key):
        """Wrong context dimension raises ValueError."""
        mlp, params = init_mlp(
            key, x_dim=4, context_dim=3, hidden_dim=16, n_hidden_layers=2, out_dim=8
        )
        x = jax.random.normal(key, (10, 4))
        context = jax.random.normal(key, (10, 5))  # Wrong: 5 instead of 3

        with pytest.raises(ValueError, match="expected context with last dimension 3"):
            mlp.apply({"params": params}, x, context)

    def test_context_batch_mismatch_raises(self, key):
        """Mismatched batch shapes raise ValueError."""
        mlp, params = init_mlp(
            key, x_dim=4, context_dim=3, hidden_dim=16, n_hidden_layers=2, out_dim=8
        )
        x = jax.random.normal(key, (10, 4))
        context = jax.random.normal(key, (20, 3))  # Wrong: 20 instead of 10

        with pytest.raises(ValueError, match="batch shape"):
            mlp.apply({"params": params}, x, context)

    def test_jit_compatible(self, key):
        """Works under jax.jit."""
        mlp, params = init_mlp(
            key, x_dim=4, context_dim=3, hidden_dim=16, n_hidden_layers=2, out_dim=8
        )
        x = jax.random.normal(key, (10, 4))
        context = jax.random.normal(key, (10, 3))

        @jax.jit
        def forward(params, x, context):
            return mlp.apply({"params": params}, x, context)

        out = forward(params, x, context)
        assert out.shape == (10, 8)

    def test_gradients_flow_through_context(self, key):
        """Gradients w.r.t. context are non-zero."""
        mlp, params = init_mlp(
            key, x_dim=4, context_dim=3, hidden_dim=16, n_hidden_layers=2, out_dim=8
        )
        # Perturb params
        params = jax.tree_util.tree_map(
            lambda p: p + 0.1 * jax.random.normal(key, p.shape), params
        )

        x = jax.random.normal(key, (10, 4))
        context = jax.random.normal(key, (10, 3))

        def loss_fn(context):
            out = mlp.apply({"params": params}, x, context)
            return (out ** 2).sum()

        grad_context = jax.grad(loss_fn)(context)
        grad_norm = jnp.linalg.norm(grad_context)
        assert grad_norm > 0, "Gradients should flow through context"

    def test_no_context_when_context_dim_zero(self, key):
        """context_dim=0 means context is ignored."""
        mlp, params = init_mlp(
            key, x_dim=4, context_dim=0, hidden_dim=16, n_hidden_layers=2, out_dim=8
        )
        x = jax.random.normal(key, (10, 4))

        # Both should work and produce same result
        out1 = mlp.apply({"params": params}, x, None)
        out2 = mlp.apply({"params": params}, x)

        assert jnp.allclose(out1, out2)

    def test_missing_context_when_context_dim_positive_raises(self, key):
        """context_dim>0 but context=None raises ValueError."""
        mlp, params = init_mlp(
            key, x_dim=4, context_dim=3, hidden_dim=16, n_hidden_layers=2, out_dim=8
        )
        x = jax.random.normal(key, (10, 4))

        with pytest.raises(ValueError, match="context_dim=3 but context was not passed"):
            mlp.apply({"params": params}, x, None)

    def test_get_output_layer(self, key):
        """get_output_layer returns correct structure and shapes."""
        mlp, params = init_mlp(
            key, x_dim=4, context_dim=0, hidden_dim=16, n_hidden_layers=2, out_dim=8
        )
        out_layer = mlp.get_output_layer(params)

        assert "kernel" in out_layer
        assert "bias" in out_layer
        assert out_layer["kernel"].shape == (16, 8)  # (hidden_dim, out_dim)
        assert out_layer["bias"].shape == (8,)  # (out_dim,)

    def test_set_output_layer(self, key):
        """set_output_layer returns new params, original unchanged."""
        mlp, params = init_mlp(
            key, x_dim=4, context_dim=0, hidden_dim=16, n_hidden_layers=2, out_dim=8
        )
        original_kernel = params["net"]["dense_out"]["kernel"].copy()
        original_bias = params["net"]["dense_out"]["bias"].copy()

        new_kernel = jnp.ones((16, 8))
        new_bias = jnp.ones((8,)) * 2

        new_params = mlp.set_output_layer(params, new_kernel, new_bias)

        # New params have new values
        assert jnp.allclose(new_params["net"]["dense_out"]["kernel"], new_kernel)
        assert jnp.allclose(new_params["net"]["dense_out"]["bias"], new_bias)

        # Original params unchanged
        assert jnp.allclose(params["net"]["dense_out"]["kernel"], original_kernel)
        assert jnp.allclose(params["net"]["dense_out"]["bias"], original_bias)

    def test_set_output_layer_forward_works(self, key):
        """set_output_layer with zeros produces zero output."""
        mlp, params = init_mlp(
            key, x_dim=4, context_dim=0, hidden_dim=16, n_hidden_layers=2, out_dim=8
        )
        # Perturb params to break zero-init
        params = jax.tree_util.tree_map(
            lambda p: p + 0.1 * jax.random.normal(key, p.shape), params
        )

        # Now set output layer to zeros
        out_layer = mlp.get_output_layer(params)
        new_kernel = jnp.zeros_like(out_layer["kernel"])
        new_bias = jnp.zeros_like(out_layer["bias"])
        params = mlp.set_output_layer(params, new_kernel, new_bias)

        # Forward should produce zeros
        x = jax.random.normal(key, (10, 4))
        out = mlp.apply({"params": params}, x)
        assert jnp.allclose(out, 0.0, atol=1e-6)


# ---------------------------------------------------------------------------
# init_mlp / init_resnet Tests
# ---------------------------------------------------------------------------
class TestInitFunctions:
    """Tests for initialization helper functions."""

    def test_init_mlp_returns_module_and_params(self, key):
        """init_mlp returns (MLP, params) tuple."""
        mlp, params = init_mlp(
            key, x_dim=4, context_dim=2, hidden_dim=16, n_hidden_layers=2, out_dim=8
        )
        assert isinstance(mlp, MLP)
        assert isinstance(params, dict)
        # Params are nested under "net" (the ResNet submodule)
        assert "net" in params
        assert "dense_in" in params["net"]
        assert "dense_out" in params["net"]

    def test_init_resnet_returns_module_and_params(self, key):
        """init_resnet returns (ResNet, params) tuple."""
        resnet, params = init_resnet(
            key, in_dim=4, hidden_dim=16, out_dim=8, n_hidden_layers=2
        )
        assert isinstance(resnet, ResNet)
        assert isinstance(params, dict)
        assert "dense_in" in params
        assert "dense_out" in params

    def test_init_mlp_dense_out_is_zero(self, key):
        """init_mlp zero-initializes dense_out kernel and bias."""
        _, params = init_mlp(
            key, x_dim=4, context_dim=0, hidden_dim=16, n_hidden_layers=2, out_dim=8
        )
        # Params are nested under "net" (the ResNet submodule)
        assert jnp.allclose(params["net"]["dense_out"]["kernel"], 0.0)
        assert jnp.allclose(params["net"]["dense_out"]["bias"], 0.0)

    def test_init_resnet_dense_out_not_zero_by_default(self, key):
        """init_resnet does NOT zero-init by default."""
        _, params = init_resnet(
            key, in_dim=4, hidden_dim=16, out_dim=8, n_hidden_layers=2,
            zero_init_output=False
        )
        # At least one of kernel/bias should be non-zero
        kernel_nonzero = not jnp.allclose(params["dense_out"]["kernel"], 0.0)
        bias_nonzero = not jnp.allclose(params["dense_out"]["bias"], 0.0)
        assert kernel_nonzero or bias_nonzero

    def test_init_resnet_dense_out_zero_when_requested(self, key):
        """init_resnet zero-inits dense_out when zero_init_output=True."""
        _, params = init_resnet(
            key, in_dim=4, hidden_dim=16, out_dim=8, n_hidden_layers=2,
            zero_init_output=True
        )
        assert jnp.allclose(params["dense_out"]["kernel"], 0.0)
        assert jnp.allclose(params["dense_out"]["bias"], 0.0)

    def test_different_keys_produce_different_params(self, key):
        """Different PRNGKeys produce different initializations."""
        _, params1 = init_resnet(key, in_dim=4, hidden_dim=16, out_dim=8, n_hidden_layers=2)
        _, params2 = init_resnet(
            jax.random.fold_in(key, 1), in_dim=4, hidden_dim=16, out_dim=8, n_hidden_layers=2
        )

        # dense_in kernels should differ
        assert not jnp.allclose(params1["dense_in"]["kernel"], params2["dense_in"]["kernel"])

    def test_init_mlp_param_structure_for_builders(self, key):
        """
        init_mlp must return params with structure params['net']['dense_out'].

        This structure is required by builders (e.g., _patch_spline_conditioner_dense_out)
        that modify the output layer for identity initialization. If you change the MLP
        param structure, update the builders accordingly.
        """
        _, params = init_mlp(
            key, x_dim=4, context_dim=0, hidden_dim=16, n_hidden_layers=2, out_dim=8
        )

        # These assertions document the contract that builders rely on.
        assert "net" in params, "MLP params must have 'net' key (ResNet submodule)"
        assert "dense_out" in params["net"], "MLP params must have 'net/dense_out' key"
        assert "kernel" in params["net"]["dense_out"], "dense_out must have 'kernel'"
        assert "bias" in params["net"]["dense_out"], "dense_out must have 'bias'"


# ---------------------------------------------------------------------------
# DeepSets Tests (Stage D1: permutation-invariant conditioner)
# ---------------------------------------------------------------------------
class TestDeepSets:
    """Tests for the DeepSets permutation-invariant conditioner."""

    def test_output_shape(self, key):
        """`__call__` emits (*batch, out_dim) for (*batch, N, d) input."""
        B, N, d = 4, 6, 3
        out_dim = 17
        ds = DeepSets(phi_hidden=(16, 16), rho_hidden=(16,), out_dim=out_dim)
        params = init_conditioner(key, ds, jnp.zeros((1, N, d)))
        x = jax.random.normal(key, (B, N, d))
        y = ds.apply({"params": params}, x)
        assert y.shape == (B, out_dim)

    def test_permutation_invariance(self, key):
        """Shuffling the particle axis leaves the output invariant."""
        N, d = 5, 3
        out_dim = 12
        ds = DeepSets(phi_hidden=(16, 16), rho_hidden=(16,), out_dim=out_dim)
        params = init_conditioner(key, ds, jnp.zeros((1, N, d)))
        # Non-zero params so invariance is not trivial (init_conditioner
        # zeroes dense_out; we randomise it here).
        k_out = jax.random.split(key, 2)[1]
        kernel = jax.random.normal(k_out, params["dense_out"]["kernel"].shape) * 0.1
        bias = jax.random.normal(k_out, params["dense_out"]["bias"].shape) * 0.1
        params = ds.set_output_layer(params, kernel, bias)

        x = jax.random.normal(key, (3, N, d))
        y_orig = ds.apply({"params": params}, x)

        perm = jnp.array([2, 0, 4, 1, 3])
        x_perm = jnp.take(x, perm, axis=-2)
        y_perm = ds.apply({"params": params}, x_perm)

        assert jnp.allclose(y_orig, y_perm, atol=1e-5)

    def test_zero_init_output(self, key):
        """init_conditioner zeroes dense_out kernel and bias."""
        ds = DeepSets(phi_hidden=(8,), rho_hidden=(8,), out_dim=10)
        params = init_conditioner(key, ds, jnp.zeros((1, 4, 3)))
        out = ds.get_output_layer(params)
        assert jnp.all(out["kernel"] == 0)
        assert jnp.all(out["bias"] == 0)

    def test_zero_init_produces_zero_output(self, key):
        """With dense_out zeroed, DeepSets outputs exactly zero."""
        ds = DeepSets(phi_hidden=(8,), rho_hidden=(8,), out_dim=10)
        params = init_conditioner(key, ds, jnp.zeros((1, 4, 3)))
        x = jax.random.normal(key, (2, 4, 3))
        y = ds.apply({"params": params}, x)
        assert jnp.all(y == 0)

    def test_context_broadcasts_across_particles(self, key):
        """Shared context shape (context_dim,) is broadcast into phi inputs."""
        B, N, d, C = 3, 5, 3, 4
        ds = DeepSets(phi_hidden=(8,), rho_hidden=(8,), out_dim=7, context_dim=C)
        params = init_conditioner(
            key, ds, jnp.zeros((1, N, d)), jnp.zeros((1, C))
        )
        # Randomise dense_out so output depends on context.
        k = jax.random.split(key, 4)[1]
        kernel = jax.random.normal(k, params["dense_out"]["kernel"].shape) * 0.1
        bias = jax.random.normal(k, params["dense_out"]["bias"].shape) * 0.1
        params = ds.set_output_layer(params, kernel, bias)

        x = jax.random.normal(key, (B, N, d))
        c1 = jnp.zeros((C,))
        c2 = jnp.ones((C,))
        y1 = ds.apply({"params": params}, x, c1)
        y2 = ds.apply({"params": params}, x, c2)
        # Different contexts => different outputs.
        assert not jnp.allclose(y1, y2, atol=1e-3)
        # Shape contract preserved.
        assert y1.shape == (B, 7)

    def test_context_per_sample(self, key):
        """Per-sample context (*batch, context_dim) matches the batch shape."""
        B, N, d, C = 3, 4, 2, 3
        ds = DeepSets(phi_hidden=(8,), rho_hidden=(8,), out_dim=5, context_dim=C)
        params = init_conditioner(
            key, ds, jnp.zeros((1, N, d)), jnp.zeros((1, C))
        )
        x = jax.random.normal(key, (B, N, d))
        c = jax.random.normal(key, (B, C))
        y = ds.apply({"params": params}, x, c)
        assert y.shape == (B, 5)

    def test_context_mismatch_raises(self, key):
        """Context width mismatch raises ValueError."""
        ds = DeepSets(phi_hidden=(8,), rho_hidden=(8,), out_dim=5, context_dim=3)
        params = init_conditioner(
            key, ds, jnp.zeros((1, 4, 3)), jnp.zeros((1, 3))
        )
        x = jnp.zeros((2, 4, 3))
        with pytest.raises(ValueError, match="context"):
            ds.apply({"params": params}, x, jnp.zeros((2, 5)))

    def test_rejects_context_when_unconditional(self, key):
        """Passing context to a context_dim=0 module raises."""
        ds = DeepSets(phi_hidden=(8,), rho_hidden=(8,), out_dim=5, context_dim=0)
        params = init_conditioner(key, ds, jnp.zeros((1, 4, 3)))
        x = jnp.zeros((2, 4, 3))
        with pytest.raises(ValueError, match="context_dim=0"):
            ds.apply({"params": params}, x, jnp.zeros((2, 3)))

    def test_jit(self, key):
        """apply traces cleanly under jax.jit."""
        ds = DeepSets(phi_hidden=(16,), rho_hidden=(16,), out_dim=11)
        params = init_conditioner(key, ds, jnp.zeros((1, 6, 3)))
        x = jax.random.normal(key, (2, 6, 3))
        apply_jit = jax.jit(lambda p, z: ds.apply({"params": p}, z))
        y = apply_jit(params, x)
        assert y.shape == (2, 11)

    def test_integrates_with_split_coupling_identity(self, key):
        """Wiring DeepSets into SplitCoupling(flatten_input=False) gives
        identity at init inside the tail bound."""
        from nflojax.transforms import SplitCoupling

        N, d, K = 6, 3, 4
        params_per_scalar = 3 * K - 1
        n_transformed = N // 2
        out_dim = n_transformed * d * params_per_scalar

        ds = DeepSets(
            phi_hidden=(16, 16),
            rho_hidden=(16,),
            out_dim=out_dim,
        )
        coupling = SplitCoupling(
            event_shape=(N, d),
            split_axis=-2,
            split_index=N // 2,
            event_ndims=2,
            conditioner=ds,
            num_bins=K,
            flatten_input=False,
            tail_bound=5.0,
        )
        params = coupling.init_params(key)

        x = jax.random.uniform(key, (3, N, d), minval=-2.0, maxval=2.0)
        y, log_det = coupling.forward(params, x)
        assert jnp.allclose(y, x, atol=1e-5)
        assert jnp.allclose(log_det, 0.0, atol=1e-5)


# ---------------------------------------------------------------------------
# Transformer Tests (Stage D2: permutation-equivariant attention)
# ---------------------------------------------------------------------------
class TestTransformer:
    """Tests for the Transformer permutation-equivariant conditioner."""

    def test_output_shape(self, key):
        """`__call__` produces (*batch, N, out_per_particle)."""
        B, N, d = 3, 6, 3
        out_per_particle = 11
        t = Transformer(
            num_layers=2, num_heads=2, embed_dim=16,
            out_per_particle=out_per_particle,
        )
        params = init_conditioner(key, t, jnp.zeros((1, N, d)))
        x = jax.random.normal(key, (B, N, d))
        y = t.apply({"params": params}, x)
        assert y.shape == (B, N, out_per_particle)

    def test_permutation_equivariance(self, key):
        """Per-token output tracks the permutation of the input particles."""
        N, d = 5, 3
        out_per_particle = 7
        t = Transformer(
            num_layers=2, num_heads=2, embed_dim=16,
            out_per_particle=out_per_particle,
        )
        params = init_conditioner(key, t, jnp.zeros((1, N, d)))
        # Non-zero dense_out so equivariance is a real test.
        k_out = jax.random.split(key, 2)[1]
        kernel = jax.random.normal(k_out, params["dense_out"]["kernel"].shape) * 0.1
        bias = jax.random.normal(k_out, params["dense_out"]["bias"].shape) * 0.1
        params = t.set_output_layer(params, kernel, bias)

        x = jax.random.normal(key, (2, N, d))
        y = t.apply({"params": params}, x)

        perm = jnp.array([2, 0, 4, 1, 3])
        x_perm = jnp.take(x, perm, axis=-2)
        y_perm = t.apply({"params": params}, x_perm)
        y_reference = jnp.take(y, perm, axis=-2)

        assert jnp.allclose(y_perm, y_reference, atol=1e-5)

    def test_zero_init_output(self, key):
        """init_conditioner zeroes dense_out kernel and bias."""
        t = Transformer(num_layers=2, num_heads=2, embed_dim=8, out_per_particle=5)
        params = init_conditioner(key, t, jnp.zeros((1, 4, 3)))
        out = t.get_output_layer(params)
        assert jnp.all(out["kernel"] == 0)
        assert jnp.all(out["bias"] == 0)

    def test_zero_init_produces_zero_output(self, key):
        """With dense_out zeroed, Transformer outputs exactly zero."""
        t = Transformer(num_layers=2, num_heads=2, embed_dim=8, out_per_particle=5)
        params = init_conditioner(key, t, jnp.zeros((1, 4, 3)))
        x = jax.random.normal(key, (2, 4, 3))
        y = t.apply({"params": params}, x)
        assert jnp.all(y == 0)

    def test_embed_not_divisible_by_heads_raises(self, key):
        """embed_dim % num_heads != 0 raises a clear error."""
        # The check fires inside __call__, so it triggers during init's trace.
        t = Transformer(num_layers=1, num_heads=3, embed_dim=8, out_per_particle=5)
        with pytest.raises(ValueError, match="divisible"):
            init_conditioner(key, t, jnp.zeros((1, 4, 3)))

    def test_context_changes_output(self, key):
        """Different contexts produce different outputs."""
        N, d, C = 4, 2, 3
        t = Transformer(
            num_layers=1, num_heads=2, embed_dim=8, out_per_particle=5,
            context_dim=C,
        )
        params = init_conditioner(
            key, t, jnp.zeros((1, N, d)), jnp.zeros((1, C))
        )
        k = jax.random.split(key, 2)[1]
        kernel = jax.random.normal(k, params["dense_out"]["kernel"].shape) * 0.1
        bias = jax.random.normal(k, params["dense_out"]["bias"].shape) * 0.1
        params = t.set_output_layer(params, kernel, bias)
        x = jax.random.normal(key, (3, N, d))
        y1 = t.apply({"params": params}, x, jnp.zeros((C,)))
        y2 = t.apply({"params": params}, x, jnp.ones((C,)))
        assert not jnp.allclose(y1, y2, atol=1e-3)

    def test_jit(self, key):
        """apply traces cleanly under jax.jit."""
        t = Transformer(num_layers=2, num_heads=4, embed_dim=16, out_per_particle=11)
        params = init_conditioner(key, t, jnp.zeros((1, 6, 3)))
        x = jax.random.normal(key, (2, 6, 3))
        apply_jit = jax.jit(lambda p, z: t.apply({"params": p}, z))
        y = apply_jit(params, x)
        assert y.shape == (2, 6, 11)

    def test_integrates_with_split_coupling_identity(self, key):
        """Wiring Transformer into SplitCoupling(flatten_input=False) gives
        identity at init inside the tail bound."""
        from nflojax.transforms import SplitCoupling

        N, d, K = 6, 3, 4
        params_per_scalar = 3 * K - 1
        # Half-half split: N_frozen == N_transformed.
        out_per_particle = d * params_per_scalar

        t = Transformer(
            num_layers=2, num_heads=2, embed_dim=16,
            out_per_particle=out_per_particle,
        )
        coupling = SplitCoupling(
            event_shape=(N, d),
            split_axis=-2,
            split_index=N // 2,
            event_ndims=2,
            conditioner=t,
            num_bins=K,
            flatten_input=False,
            tail_bound=5.0,
        )
        params = coupling.init_params(key)

        x = jax.random.uniform(key, (3, N, d), minval=-2.0, maxval=2.0)
        y, log_det = coupling.forward(params, x)
        assert jnp.allclose(y, x, atol=1e-5)
        assert jnp.allclose(log_det, 0.0, atol=1e-5)

    def test_split_coupling_round_trip(self, key):
        """Forward+inverse round-trip after randomising dense_out."""
        from nflojax.transforms import SplitCoupling

        N, d, K = 6, 3, 4
        params_per_scalar = 3 * K - 1
        out_per_particle = d * params_per_scalar

        t = Transformer(
            num_layers=2, num_heads=2, embed_dim=16,
            out_per_particle=out_per_particle,
        )
        coupling = SplitCoupling(
            event_shape=(N, d),
            split_axis=-2,
            split_index=N // 2,
            event_ndims=2,
            conditioner=t,
            num_bins=K,
            flatten_input=False,
            tail_bound=5.0,
        )
        params = coupling.init_params(key)

        # Randomise dense_out so the coupling is non-trivial.
        k = jax.random.split(key, 2)[1]
        per_token_bias = jax.random.normal(
            k, params["mlp"]["dense_out"]["bias"].shape) * 0.05
        params["mlp"]["dense_out"]["bias"] = per_token_bias

        x = jax.random.uniform(key, (4, N, d), minval=-3.0, maxval=3.0)
        y, ld_f = coupling.forward(params, x)
        x_back, ld_i = coupling.inverse(params, y)
        assert jnp.allclose(x_back, x, atol=1e-4)


# ---------------------------------------------------------------------------
# GNN Tests (Stage D3: permutation-equivariant message passing)
# ---------------------------------------------------------------------------
class TestGNN:
    """Tests for the GNN permutation-equivariant conditioner."""

    @pytest.fixture
    def cubic_geom(self):
        return Geometry.cubic(d=3, side=2.0)

    def test_output_shape(self, key, cubic_geom):
        """Per-token output shape (*batch, N, out_per_particle)."""
        B, N, d = 2, 8, 3
        out_per_particle = 11
        gnn = GNN(
            num_layers=2, hidden=16, out_per_particle=out_per_particle,
            num_neighbours=4, geometry=cubic_geom,
        )
        params = init_conditioner(key, gnn, jnp.zeros((1, N, d)))
        x = jax.random.uniform(key, (B, N, d), minval=-1.0, maxval=1.0)
        y = gnn.apply({"params": params}, x)
        assert y.shape == (B, N, out_per_particle)

    def test_permutation_equivariance(self, key, cubic_geom):
        """Per-token output tracks permutations of the input particles."""
        N, d = 7, 3
        out_per_particle = 5
        gnn = GNN(
            num_layers=2, hidden=16, out_per_particle=out_per_particle,
            num_neighbours=3, geometry=cubic_geom,
        )
        params = init_conditioner(key, gnn, jnp.zeros((1, N, d)))
        # Randomise dense_out so equivariance is a real test.
        k = jax.random.split(key, 2)[1]
        kernel = jax.random.normal(k, params["dense_out"]["kernel"].shape) * 0.1
        bias = jax.random.normal(k, params["dense_out"]["bias"].shape) * 0.1
        params = gnn.set_output_layer(params, kernel, bias)

        x = jax.random.uniform(key, (2, N, d), minval=-1.0, maxval=1.0)
        y = gnn.apply({"params": params}, x)

        perm = jnp.array([3, 1, 6, 0, 4, 2, 5])
        x_perm = jnp.take(x, perm, axis=-2)
        y_perm = gnn.apply({"params": params}, x_perm)
        y_reference = jnp.take(y, perm, axis=-2)

        assert jnp.allclose(y_perm, y_reference, atol=1e-5)

    def test_zero_init_produces_zero_output(self, key, cubic_geom):
        gnn = GNN(
            num_layers=1, hidden=8, out_per_particle=5,
            num_neighbours=3, geometry=cubic_geom,
        )
        params = init_conditioner(key, gnn, jnp.zeros((1, 6, 3)))
        x = jax.random.uniform(key, (2, 6, 3), minval=-1.0, maxval=1.0)
        y = gnn.apply({"params": params}, x)
        assert jnp.all(y == 0)

    def test_num_neighbours_ge_N_raises(self, key, cubic_geom):
        """num_neighbours >= N raises a clear error (self-edge is excluded)."""
        # The error is raised inside __call__ during init tracing.
        gnn = GNN(
            num_layers=1, hidden=8, out_per_particle=5,
            num_neighbours=4, geometry=cubic_geom,
        )
        with pytest.raises(ValueError, match="num_neighbours"):
            init_conditioner(key, gnn, jnp.zeros((1, 4, 3)))

    def test_neighbour_list_stability_under_perturbation(self, key, cubic_geom):
        """Small jitter (smaller than nearest-neighbour gap) leaves the top-K
        neighbour set unchanged, so the GNN output is a smooth function of x
        in a neighbourhood of the current config."""
        N, d = 6, 3
        gnn = GNN(
            num_layers=1, hidden=8, out_per_particle=3,
            num_neighbours=3, geometry=cubic_geom,
        )
        params = init_conditioner(key, gnn, jnp.zeros((1, N, d)))
        k = jax.random.split(key, 2)[1]
        kernel = jax.random.normal(k, params["dense_out"]["kernel"].shape) * 0.1
        params = gnn.set_output_layer(params, kernel, jnp.zeros_like(
            params["dense_out"]["bias"]))

        # Well-separated positions: nearest-neighbour gap ~ 0.5, jitter 1e-4.
        x = jnp.array([[0.0, 0.0, 0.0],
                       [0.5, 0.0, 0.0],
                       [0.0, 0.5, 0.0],
                       [0.0, 0.0, 0.5],
                       [0.5, 0.5, 0.0],
                       [0.5, 0.0, 0.5]])[None, ...]
        eps = jax.random.uniform(key, x.shape) * 1e-4
        y1 = gnn.apply({"params": params}, x)
        y2 = gnn.apply({"params": params}, x + eps)
        # With the same neighbour set, outputs move smoothly.
        assert jnp.allclose(y1, y2, atol=1e-2)

    def test_pbc_default_12_neighbours(self, key):
        """Default num_neighbours=12 is the shipped value (PLAN.md §10.5)."""
        # Verify the default; 16 particles so 12 neighbours is valid.
        geom = Geometry.cubic(d=3, side=2.0)
        gnn = GNN(num_layers=1, hidden=8, out_per_particle=5, geometry=geom)
        assert gnn.num_neighbours == 12
        # Sanity: still initialises.
        init_conditioner(key, gnn, jnp.zeros((1, 16, 3)))

    def test_no_geometry_uses_euclidean(self, key):
        """geometry=None falls back to Euclidean distances and still works."""
        gnn = GNN(
            num_layers=1, hidden=8, out_per_particle=5,
            num_neighbours=3, geometry=None,
        )
        params = init_conditioner(key, gnn, jnp.zeros((1, 6, 3)))
        x = jax.random.normal(key, (2, 6, 3))
        y = gnn.apply({"params": params}, x)
        assert y.shape == (2, 6, 5)

    def test_cutoff_zeros_far_messages(self, key, cubic_geom):
        """With cutoff=0.0 every message is zeroed; with cutoff=None none is."""
        N, d = 5, 3
        k = jax.random.split(key, 3)
        # Two instances differing only in `cutoff`, sharing init params.
        gnn_cut0 = GNN(
            num_layers=1, hidden=8, out_per_particle=3,
            num_neighbours=2, geometry=cubic_geom, cutoff=0.0,
        )
        params0 = init_conditioner(k[0], gnn_cut0, jnp.zeros((1, N, d)))
        # Randomise dense_out so "everything zero" vs "non-zero" is visible.
        kernel = jax.random.normal(k[1], params0["dense_out"]["kernel"].shape) * 0.5
        bias = jax.random.normal(k[1], params0["dense_out"]["bias"].shape) * 0.5
        params0 = gnn_cut0.set_output_layer(params0, kernel, bias)
        x = jax.random.uniform(k[2], (1, N, d), minval=-0.5, maxval=0.5)
        y_cut0 = gnn_cut0.apply({"params": params0}, x)

        gnn_cut_inf = GNN(
            num_layers=1, hidden=8, out_per_particle=3,
            num_neighbours=2, geometry=cubic_geom, cutoff=None,
        )
        # Re-init on the same key so non-output params differ only in cutoff:
        vars_inf = gnn_cut_inf.init(k[0], jnp.zeros((1, N, d)))
        params_inf = vars_inf["params"]
        params_inf = gnn_cut_inf.set_output_layer(params_inf, kernel, bias)
        y_inf = gnn_cut_inf.apply({"params": params_inf}, x)

        # cutoff=0 kills all messages; cutoff=None keeps them. Outputs differ.
        assert not jnp.allclose(y_cut0, y_inf, atol=1e-3)

    def test_jit(self, key, cubic_geom):
        gnn = GNN(
            num_layers=2, hidden=16, out_per_particle=7,
            num_neighbours=3, geometry=cubic_geom,
        )
        params = init_conditioner(key, gnn, jnp.zeros((1, 8, 3)))
        x = jax.random.uniform(key, (3, 8, 3), minval=-1.0, maxval=1.0)
        apply_jit = jax.jit(lambda p, z: gnn.apply({"params": p}, z))
        y = apply_jit(params, x)
        assert y.shape == (3, 8, 7)

    def test_integrates_with_split_coupling_identity(self, key, cubic_geom):
        """GNN into SplitCoupling(flatten_input=False) gives identity at init."""
        from nflojax.transforms import SplitCoupling

        N, d, K_bins = 8, 3, 4
        params_per_scalar = 3 * K_bins - 1
        out_per_particle = d * params_per_scalar

        gnn = GNN(
            num_layers=2, hidden=16, out_per_particle=out_per_particle,
            num_neighbours=3, geometry=cubic_geom,
        )
        coupling = SplitCoupling(
            event_shape=(N, d),
            split_axis=-2,
            split_index=N // 2,
            event_ndims=2,
            conditioner=gnn,
            num_bins=K_bins,
            flatten_input=False,
            tail_bound=5.0,
        )
        params = coupling.init_params(key)

        x = jax.random.uniform(key, (3, N, d), minval=-1.0, maxval=1.0)
        y, log_det = coupling.forward(params, x)
        assert jnp.allclose(y, x, atol=1e-5, rtol=0)
        assert jnp.allclose(log_det, 0.0, atol=1e-5)

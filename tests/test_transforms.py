# tests/test_transforms.py
"""Unit tests for transform layers."""
from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp

from nflojax.transforms import (
    LinearTransform,
    Permutation,
    AffineCoupling,
    SplineCoupling,
    SplitCoupling,
    CompositeTransform,
    LoftTransform,
    CircularShift,
    Rescale,
    CoMProjection,
    identity_spline_bias,
    stable_logit,
)
from nflojax.geometry import Geometry
from nflojax.nets import init_mlp
from conftest import check_logdet_vs_autodiff, requires_x64


class TestIdentitySplineBias:
    """Shared helper used by both SplineCoupling and SplitCoupling for near-identity init."""

    def test_shape(self):
        bias = identity_spline_bias(
            num_scalars=5, num_bins=8, min_derivative=1e-2, max_derivative=10.0
        )
        assert bias.shape == (5 * (3 * 8 - 1),)

    def test_widths_heights_are_zero(self):
        """First 2K entries per scalar (widths + heights) must be zero."""
        K = 4
        S = 3
        bias = identity_spline_bias(
            num_scalars=S, num_bins=K, min_derivative=1e-2, max_derivative=10.0
        )
        b = bias.reshape(S, 3 * K - 1)
        assert jnp.all(b[:, : 2 * K] == 0.0)

    def test_derivative_logit_produces_derivative_one(self):
        """Last K-1 entries per scalar must equal stable_logit(alpha)."""
        K = 4
        lo, hi = 1e-2, 10.0
        alpha = (1.0 - lo) / (hi - lo)
        expected = stable_logit(jnp.asarray(alpha, dtype=jnp.float32))

        bias = identity_spline_bias(
            num_scalars=2, num_bins=K, min_derivative=lo, max_derivative=hi
        )
        b = bias.reshape(2, 3 * K - 1)
        assert jnp.allclose(b[:, 2 * K :], expected)

    def test_out_of_range_derivative_falls_back_to_zero(self):
        """When 1.0 is not in (min, max), derivative bias stays at zero."""
        bias = identity_spline_bias(
            num_scalars=2, num_bins=4, min_derivative=2.0, max_derivative=10.0
        )
        b = bias.reshape(2, 3 * 4 - 1)
        assert jnp.all(b == 0.0)

    def test_circular_shape(self):
        """Circular mode has one extra slot per scalar (K instead of K-1 derivs)."""
        K = 8
        bias = identity_spline_bias(
            num_scalars=5, num_bins=K,
            min_derivative=1e-2, max_derivative=10.0,
            boundary_slopes="circular",
        )
        assert bias.shape == (5 * (3 * K),)

    def test_circular_widths_heights_zero(self):
        """Circular: first 2K entries per scalar still zero."""
        K = 4
        S = 3
        bias = identity_spline_bias(
            num_scalars=S, num_bins=K,
            min_derivative=1e-2, max_derivative=10.0,
            boundary_slopes="circular",
        )
        b = bias.reshape(S, 3 * K)
        assert jnp.all(b[:, : 2 * K] == 0.0)

    def test_circular_all_derivs_logit_alpha(self):
        """Circular: all K deriv slots (K-1 interior + 1 shared boundary) equal stable_logit(alpha)."""
        K = 4
        lo, hi = 1e-2, 10.0
        alpha = (1.0 - lo) / (hi - lo)
        expected = stable_logit(jnp.asarray(alpha, dtype=jnp.float32))

        bias = identity_spline_bias(
            num_scalars=2, num_bins=K,
            min_derivative=lo, max_derivative=hi,
            boundary_slopes="circular",
        )
        b = bias.reshape(2, 3 * K)
        assert jnp.allclose(b[:, 2 * K :], expected)


def check_logdet_vs_autodiff_structured(forward_fn, x, event_ndims):
    """
    Log-det check for a single-sample rank-N event.

    For an event of shape event_shape (rank `event_ndims`), jax.jacfwd produces
    a tensor of shape event_shape + event_shape. Flatten both sides to a 2D
    matrix to take the determinant. Only works for one sample at a time.
    """
    y, ld = forward_fn(x)
    event_shape = x.shape[-event_ndims:]
    event_size = 1
    for s in event_shape:
        event_size *= s

    J = jax.jacfwd(lambda z: forward_fn(z)[0])(x)
    # J shape: event_shape + event_shape. Flatten to (event_size, event_size).
    J_flat = J.reshape(event_size, event_size)
    ld_autodiff = jnp.log(jnp.abs(jnp.linalg.det(J_flat)))

    return {
        "error": float(jnp.abs(ld - ld_autodiff)),
        "ld": float(ld),
        "ld_autodiff": float(ld_autodiff),
    }


# ============================================================================
# LinearTransform Tests
# ============================================================================
class TestLinearTransform:
    """Tests for LinearTransform (LU-parameterized)."""

    @pytest.fixture
    def identity_params(self, dim):
        """Identity transform params (L=I, U=0, s=1).

        For softplus parametrization, raw_diag = log(e-1) gives s = softplus(raw_diag) = 1.
        """
        return {
            "lower": jnp.zeros((dim, dim)),
            "upper": jnp.zeros((dim, dim)),
            "raw_diag": jnp.full((dim,), jnp.log(jnp.e - 1)),
        }

    @pytest.fixture
    def random_params(self, key, dim):
        """Random transform params."""
        k1, k2, k3 = jax.random.split(key, 3)
        return {
            "lower": jax.random.normal(k1, (dim, dim)) * 0.1,
            "upper": jax.random.normal(k2, (dim, dim)) * 0.1,
            "raw_diag": jax.random.normal(k3, (dim,)) * 0.5,
        }

    def test_identity_at_init(self, dim, identity_params):
        """With zero params, transform is identity."""
        transform = LinearTransform(dim=dim)
        x = jnp.arange(dim, dtype=jnp.float32).reshape(1, dim)

        y, ld = transform.forward(identity_params, x)

        assert jnp.allclose(y, x, atol=1e-6)
        assert jnp.allclose(ld, 0.0, atol=1e-6)

    def test_forward_shape(self, key, dim, random_params, batch_size):
        """Forward returns correct shapes."""
        transform = LinearTransform(dim=dim)
        x = jax.random.normal(key, (batch_size, dim))

        y, ld = transform.forward(random_params, x)

        assert y.shape == (batch_size, dim)
        assert ld.shape == (batch_size,)

    def test_inverse_shape(self, key, dim, random_params, batch_size):
        """Inverse returns correct shapes."""
        transform = LinearTransform(dim=dim)
        y = jax.random.normal(key, (batch_size, dim))

        x, ld = transform.inverse(random_params, y)

        assert x.shape == (batch_size, dim)
        assert ld.shape == (batch_size,)

    def test_invertibility(self, key, dim, random_params):
        """forward(inverse(y)) = y and inverse(forward(x)) = x."""
        transform = LinearTransform(dim=dim)
        x = jax.random.normal(key, (50, dim))

        # Forward then inverse
        y, ld_fwd = transform.forward(random_params, x)
        x_rec, ld_inv = transform.inverse(random_params, y)

        # Relaxed tolerance for triangular solve numerical precision
        max_err = float(jnp.abs(x - x_rec).max())
        assert max_err < 1e-2, f"Max reconstruction error: {max_err}"
        assert jnp.allclose(ld_fwd + ld_inv, 0.0, atol=1e-5)

    def test_logdet_consistency(self, key, dim, random_params):
        """log_det_forward = -log_det_inverse at same point."""
        transform = LinearTransform(dim=dim)
        x = jax.random.normal(key, (20, dim))

        y, ld_fwd = transform.forward(random_params, x)
        _, ld_inv = transform.inverse(random_params, y)

        assert jnp.allclose(ld_fwd, -ld_inv, atol=1e-5)

    def test_logdet_vs_autodiff(self, key, dim, random_params):
        """Log-det matches autodiff Jacobian computation."""
        transform = LinearTransform(dim=dim)
        x = jax.random.normal(key, (dim,))  # Single sample

        y, ld = transform.forward(random_params, x)

        # Compute via autodiff
        J = jax.jacfwd(lambda z: transform.forward(random_params, z)[0])(x)
        ld_autodiff = jnp.log(jnp.abs(jnp.linalg.det(J)))

        assert jnp.allclose(ld, ld_autodiff, atol=1e-4), f"ld={ld}, autodiff={ld_autodiff}"

    def test_jit_compatible(self, key, dim, random_params):
        """LinearTransform works under JIT."""
        transform = LinearTransform(dim=dim)
        x = jax.random.normal(key, (10, dim))

        forward_jit = jax.jit(transform.forward)
        inverse_jit = jax.jit(transform.inverse)

        y, ld = forward_jit(random_params, x)
        x_rec, _ = inverse_jit(random_params, y)

        assert y.shape == (10, dim)
        assert jnp.allclose(x, x_rec, atol=1e-5)

    def test_wrong_input_dim_raises(self, dim, identity_params):
        """Wrong input dimension raises ValueError."""
        transform = LinearTransform(dim=dim)
        x_wrong = jnp.zeros((5, dim + 1))

        with pytest.raises(ValueError, match="expected input last dim"):
            transform.forward(identity_params, x_wrong)

    def test_wrong_lower_shape_raises(self, dim):
        """Wrong lower matrix shape raises ValueError."""
        transform = LinearTransform(dim=dim)
        bad_params = {
            "lower": jnp.zeros((dim + 1, dim)),
            "upper": jnp.zeros((dim, dim)),
            "raw_diag": jnp.full((dim,), jnp.log(jnp.e - 1)),
        }
        x = jnp.zeros((1, dim))

        with pytest.raises(ValueError, match="lower must have shape"):
            transform.forward(bad_params, x)

    def test_wrong_upper_shape_raises(self, dim):
        """Wrong upper matrix shape raises ValueError."""
        transform = LinearTransform(dim=dim)
        bad_params = {
            "lower": jnp.zeros((dim, dim)),
            "upper": jnp.zeros((dim, dim + 1)),
            "raw_diag": jnp.full((dim,), jnp.log(jnp.e - 1)),
        }
        x = jnp.zeros((1, dim))

        with pytest.raises(ValueError, match="upper must have shape"):
            transform.forward(bad_params, x)

    def test_wrong_raw_diag_shape_raises(self, dim):
        """Wrong raw_diag shape raises ValueError."""
        transform = LinearTransform(dim=dim)
        bad_params = {
            "lower": jnp.zeros((dim, dim)),
            "upper": jnp.zeros((dim, dim)),
            "raw_diag": jnp.zeros(dim + 1),
        }
        x = jnp.zeros((1, dim))

        with pytest.raises(ValueError, match="raw_diag must have shape"):
            transform.forward(bad_params, x)

    def test_missing_params_raises(self, dim):
        """Missing params keys raises KeyError."""
        transform = LinearTransform(dim=dim)
        x = jnp.zeros((1, dim))

        with pytest.raises(KeyError):
            transform.forward({}, x)


# ============================================================================
# Permutation Tests
# ============================================================================
class TestPermutation:
    """Tests for Permutation transform."""

    @pytest.fixture
    def reverse_perm(self, dim):
        """Reverse permutation [dim-1, dim-2, ..., 0]."""
        return jnp.arange(dim - 1, -1, -1)

    @pytest.fixture
    def identity_perm(self, dim):
        """Identity permutation [0, 1, ..., dim-1]."""
        return jnp.arange(dim)

    def test_forward_reverses(self, dim, reverse_perm):
        """Forward with reverse perm reverses the vector."""
        transform = Permutation(perm=reverse_perm)
        x = jnp.arange(dim, dtype=jnp.float32).reshape(1, dim)

        y, ld = transform.forward({}, x)

        expected = jnp.flip(x, axis=-1)
        assert jnp.allclose(y, expected)
        assert jnp.allclose(ld, 0.0)  # Permutation has unit determinant

    def test_identity_perm(self, key, dim, identity_perm):
        """Identity permutation leaves input unchanged."""
        transform = Permutation(perm=identity_perm)
        x = jax.random.normal(key, (10, dim))

        y, ld = transform.forward({}, x)

        assert jnp.allclose(y, x)
        assert jnp.allclose(ld, 0.0)

    def test_invertibility(self, key, dim, reverse_perm):
        """forward(inverse(y)) = y."""
        transform = Permutation(perm=reverse_perm)
        x = jax.random.normal(key, (50, dim))

        y, _ = transform.forward({}, x)
        x_rec, _ = transform.inverse({}, y)

        assert jnp.allclose(x, x_rec, atol=1e-6)

    def test_inverse_reverses_forward(self, key, dim, reverse_perm):
        """Inverse undoes forward."""
        transform = Permutation(perm=reverse_perm)
        y = jax.random.normal(key, (20, dim))

        x, _ = transform.inverse({}, y)
        y_rec, _ = transform.forward({}, x)

        assert jnp.allclose(y, y_rec, atol=1e-6)

    def test_logdet_always_zero(self, key, dim, reverse_perm):
        """Log-det is always zero for permutations."""
        transform = Permutation(perm=reverse_perm)
        x = jax.random.normal(key, (30, dim))

        _, ld_fwd = transform.forward({}, x)
        _, ld_inv = transform.inverse({}, x)

        assert jnp.allclose(ld_fwd, 0.0)
        assert jnp.allclose(ld_inv, 0.0)

    def test_output_shape(self, key, dim, reverse_perm, batch_size):
        """Forward/inverse return correct shapes."""
        transform = Permutation(perm=reverse_perm)
        x = jax.random.normal(key, (batch_size, dim))

        y, ld = transform.forward({}, x)

        assert y.shape == (batch_size, dim)
        assert ld.shape == (batch_size,)

    def test_jit_compatible(self, key, dim, reverse_perm):
        """Permutation works under JIT."""
        transform = Permutation(perm=reverse_perm)
        x = jax.random.normal(key, (10, dim))

        forward_jit = jax.jit(transform.forward)
        y, _ = forward_jit({}, x)

        assert y.shape == (10, dim)

    def test_wrong_input_dim_raises(self, dim, reverse_perm):
        """Wrong input dimension raises ValueError."""
        transform = Permutation(perm=reverse_perm)
        x_wrong = jnp.zeros((5, dim + 1))

        with pytest.raises(ValueError, match="axis -1 of size"):
            transform.forward({}, x_wrong)

    def test_non_1d_perm_raises(self):
        """Non-1D permutation raises ValueError."""
        with pytest.raises(ValueError, match="must be 1D"):
            Permutation(perm=jnp.zeros((3, 3)))

    def test_non_integer_perm_raises(self):
        """Non-integer permutation raises TypeError."""
        with pytest.raises(TypeError, match="must be integer"):
            Permutation(perm=jnp.array([0.0, 1.0, 2.0]))

    def test_custom_permutation(self, key):
        """Custom permutation works correctly."""
        perm = jnp.array([2, 0, 3, 1])
        transform = Permutation(perm=perm)
        x = jnp.array([[0, 1, 2, 3]], dtype=jnp.float32)

        y, _ = transform.forward({}, x)

        expected = jnp.array([[2, 0, 3, 1]], dtype=jnp.float32)
        assert jnp.allclose(y, expected)

    # --- event_axis generalisation --------------------------------------

    def test_event_axis_particle(self, key):
        """event_axis=-2 permutes particles on (B, N, d), not coordinates."""
        N, d = 4, 3
        perm = jnp.array([3, 0, 1, 2])  # non-trivial rotation of particles
        transform = Permutation(perm=perm, event_axis=-2)
        x = jax.random.normal(key, (5, N, d))

        y, log_det = transform.forward({}, x)

        assert y.shape == x.shape
        # Particles are re-ordered; coordinates within each particle unchanged.
        for i in range(N):
            assert jnp.allclose(y[:, i, :], x[:, int(perm[i]), :])
        assert log_det.shape == (5, d)
        assert jnp.all(log_det == 0.0)

    def test_event_axis_roundtrip(self, key):
        """Round-trip under particle-axis permutation."""
        N, d = 6, 3
        perm = jax.random.permutation(key, N)
        transform = Permutation(perm=perm, event_axis=-2)
        x = jax.random.normal(jax.random.fold_in(key, 1), (4, N, d))

        y, _ = transform.forward({}, x)
        x_back, _ = transform.inverse({}, y)
        assert jnp.allclose(x_back, x, atol=1e-6)

    def test_event_axis_positive_raises(self):
        """Non-negative event_axis is rejected."""
        with pytest.raises(ValueError, match="event_axis"):
            Permutation(perm=jnp.array([0, 1]), event_axis=0)

    def test_event_axis_shape_mismatch_raises(self):
        """Axis size mismatch is caught at forward time."""
        transform = Permutation(perm=jnp.array([2, 0, 1]), event_axis=-2)
        x = jnp.zeros((4, 5, 3))  # axis -2 is 5, but perm is length 3
        with pytest.raises(ValueError, match="axis -2"):
            transform.forward({}, x)


# ============================================================================
# LoftTransform Tests
# ============================================================================
class TestLoftTransform:
    """Tests for LoftTransform."""

    def test_identity_near_zero(self, key, dim):
        """Near origin, LOFT is approximately identity."""
        transform = LoftTransform(dim=dim, tau=5.0)
        x = jax.random.normal(key, (100, dim)) * 0.1  # Small values

        y, ld = transform.forward({}, x)

        # For |x| << tau, loft(x) ≈ x
        assert jnp.allclose(y, x, atol=0.01)

    def test_invertibility(self, key, dim):
        """forward(inverse(y)) = y."""
        transform = LoftTransform(dim=dim, tau=3.0)
        x = jax.random.normal(key, (50, dim)) * 5  # Mix of small and large

        y, ld_fwd = transform.forward({}, x)
        x_rec, ld_inv = transform.inverse({}, y)

        assert jnp.allclose(x, x_rec, atol=1e-5)
        assert jnp.allclose(ld_fwd + ld_inv, 0.0, atol=1e-5)

    def test_logdet_consistency(self, key, dim):
        """log_det_forward = -log_det_inverse."""
        transform = LoftTransform(dim=dim, tau=2.0)
        x = jax.random.normal(key, (30, dim)) * 3

        y, ld_fwd = transform.forward({}, x)
        _, ld_inv = transform.inverse({}, y)

        assert jnp.allclose(ld_fwd, -ld_inv, atol=1e-5)

    def test_compresses_tails(self, dim):
        """LOFT compresses large values (log behavior in tails)."""
        transform = LoftTransform(dim=dim, tau=1.0)
        x_large = jnp.full((1, dim), 100.0)

        y, _ = transform.forward({}, x_large)

        # y should be much smaller than x due to log compression
        # loft(100) = 1 + log(100 - 1 + 1) = 1 + log(100) ≈ 5.6
        assert jnp.all(y < x_large / 10)

    def test_wrong_dim_raises(self, dim):
        """Wrong input dimension raises ValueError."""
        transform = LoftTransform(dim=dim, tau=1.0)
        x_wrong = jnp.zeros((5, dim + 1))

        with pytest.raises(ValueError, match="expected input last dim"):
            transform.forward({}, x_wrong)

    def test_invalid_dim_raises(self):
        """Non-positive dim raises ValueError."""
        with pytest.raises(ValueError, match="dim must be positive"):
            LoftTransform(dim=0, tau=1.0)

    def test_invalid_tau_raises(self):
        """Non-positive tau raises ValueError."""
        with pytest.raises(ValueError, match="tau must be strictly positive"):
            LoftTransform(dim=4, tau=0.0)


# ============================================================================
# CompositeTransform Tests
# ============================================================================
class TestCompositeTransform:
    """Tests for CompositeTransform."""

    def test_single_block(self, key, dim):
        """Composite with single block equals that block."""
        perm = jnp.arange(dim - 1, -1, -1)
        block = Permutation(perm=perm)
        composite = CompositeTransform(blocks=[block])

        x = jax.random.normal(key, (20, dim))

        y_composite, ld_composite = composite.forward([{}], x)
        y_block, ld_block = block.forward({}, x)

        assert jnp.allclose(y_composite, y_block)
        assert jnp.allclose(ld_composite, ld_block)

    def test_two_reverses_is_identity(self, key, dim):
        """Two reverse permutations = identity."""
        perm = jnp.arange(dim - 1, -1, -1)
        blocks = [Permutation(perm=perm), Permutation(perm=perm)]
        composite = CompositeTransform(blocks=blocks)

        x = jax.random.normal(key, (30, dim))

        y, ld = composite.forward([{}, {}], x)

        assert jnp.allclose(y, x, atol=1e-6)
        assert jnp.allclose(ld, 0.0)

    def test_invertibility(self, key, dim):
        """Composite transform is invertible."""
        perm = jnp.arange(dim - 1, -1, -1)
        loft = LoftTransform(dim=dim, tau=5.0)
        blocks = [Permutation(perm=perm), loft]
        composite = CompositeTransform(blocks=blocks)

        x = jax.random.normal(key, (40, dim))
        params = [{}, {}]

        y, ld_fwd = composite.forward(params, x)
        x_rec, ld_inv = composite.inverse(params, y)

        assert jnp.allclose(x, x_rec, atol=1e-5)
        assert jnp.allclose(ld_fwd + ld_inv, 0.0, atol=1e-5)

    def test_wrong_params_length_raises(self, dim):
        """Wrong number of param sets raises ValueError."""
        perm = jnp.arange(dim - 1, -1, -1)
        blocks = [Permutation(perm=perm), Permutation(perm=perm)]
        composite = CompositeTransform(blocks=blocks)

        x = jnp.zeros((1, dim))

        with pytest.raises(ValueError, match="expected 2 param sets"):
            composite.forward([{}], x)  # Only 1 param set for 2 blocks


# ============================================================================
# AffineCoupling Error Handling Tests
# ============================================================================
class TestAffineCouplingErrors:
    """Error handling tests for AffineCoupling."""

    def test_non_1d_mask_raises(self):
        """Non-1D mask raises ValueError."""
        from nflojax.nets import MLP
        mlp = MLP(x_dim=4, hidden_dim=8, n_hidden_layers=1, out_dim=8)

        with pytest.raises(ValueError, match="must be 1D"):
            AffineCoupling(mask=jnp.zeros((2, 2)), conditioner=mlp)

    def test_missing_mlp_key_raises(self, dim):
        """Missing 'mlp' in params raises KeyError."""
        from nflojax.nets import MLP
        mask = jnp.array([1, 0, 1, 0], dtype=jnp.float32)
        mlp = MLP(x_dim=dim, hidden_dim=8, n_hidden_layers=1, out_dim=2 * dim)
        coupling = AffineCoupling(mask=mask, conditioner=mlp)

        x = jnp.zeros((1, dim))

        with pytest.raises(KeyError, match="mlp"):
            coupling.forward({}, x)

    def test_wrong_input_dim_raises(self, key, dim):
        """Wrong input dimension raises ValueError."""
        mask = jnp.array([1, 0, 1, 0], dtype=jnp.float32)
        mlp, mlp_params = init_mlp(key, x_dim=dim, context_dim=0, hidden_dim=8, n_hidden_layers=1, out_dim=2 * dim)
        coupling = AffineCoupling(mask=mask, conditioner=mlp)

        x_wrong = jnp.zeros((5, dim + 1))

        with pytest.raises(ValueError, match="expected input with last dimension"):
            coupling.forward({"mlp": mlp_params}, x_wrong)


# ============================================================================
# SplineCoupling Error Handling Tests
# ============================================================================
class TestSplineCouplingErrors:
    """Error handling tests for SplineCoupling."""

    def test_non_1d_mask_raises(self):
        """Non-1D mask raises ValueError."""
        from nflojax.nets import MLP
        mlp = MLP(x_dim=4, hidden_dim=8, n_hidden_layers=1, out_dim=92)  # 4 * (3*8 - 1)

        with pytest.raises(ValueError, match="must be 1D"):
            SplineCoupling(mask=jnp.zeros((2, 2)), conditioner=mlp, num_bins=8)

    def test_missing_mlp_key_raises(self, dim):
        """Missing 'mlp' in params raises KeyError."""
        from nflojax.nets import MLP
        mask = jnp.array([1, 0, 1, 0], dtype=jnp.float32)
        num_bins = 8
        out_dim = dim * (3 * num_bins - 1)
        mlp = MLP(x_dim=dim, hidden_dim=8, n_hidden_layers=1, out_dim=out_dim)
        coupling = SplineCoupling(mask=mask, conditioner=mlp, num_bins=num_bins)

        x = jnp.zeros((1, dim))

        with pytest.raises(KeyError, match="mlp"):
            coupling.forward({}, x)


# ============================================================================
# init_params Tests
# ============================================================================
class TestInitParams:
    """Tests for init_params methods on all transforms."""

    def test_linear_transform_init_params_structure(self, key, dim):
        """LinearTransform.init_params returns correct structure."""
        transform = LinearTransform(dim=dim)
        params = transform.init_params(key)

        assert "lower" in params
        assert "upper" in params
        assert "raw_diag" in params
        assert params["lower"].shape == (dim, dim)
        assert params["upper"].shape == (dim, dim)
        assert params["raw_diag"].shape == (dim,)

    def test_linear_transform_init_params_identity(self, key, dim):
        """LinearTransform default init produces identity transform."""
        transform = LinearTransform(dim=dim)
        params = transform.init_params(key)
        x = jax.random.normal(key, (10, dim))

        y, ld = transform.forward(params, x)

        assert jnp.allclose(y, x, atol=1e-5)
        assert jnp.allclose(ld, 0.0, atol=1e-5)

    def test_permutation_init_params_empty(self, key, dim):
        """Permutation.init_params returns empty dict."""
        perm = jnp.arange(dim - 1, -1, -1)
        transform = Permutation(perm=perm)
        params = transform.init_params(key)

        assert params == {}

    def test_loft_init_params_empty(self, key, dim):
        """LoftTransform.init_params returns empty dict."""
        transform = LoftTransform(dim=dim, tau=5.0)
        params = transform.init_params(key)

        assert params == {}

    def test_affine_coupling_init_params_structure(self, key, dim):
        """AffineCoupling.init_params returns correct structure."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        mlp, _ = init_mlp(
            key, x_dim=dim, context_dim=0, hidden_dim=16, n_hidden_layers=1, out_dim=2 * dim
        )
        coupling = AffineCoupling(mask=mask, conditioner=mlp)

        params = coupling.init_params(key)

        assert "mlp" in params
        assert "net" in params["mlp"]
        assert "dense_out" in params["mlp"]["net"]

    def test_affine_coupling_init_params_identity(self, key, dim):
        """AffineCoupling default init produces near-identity transform."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        mlp, _ = init_mlp(
            key, x_dim=dim, context_dim=0, hidden_dim=16, n_hidden_layers=1, out_dim=2 * dim
        )
        coupling = AffineCoupling(mask=mask, conditioner=mlp)

        k1, k2 = jax.random.split(key)
        params = coupling.init_params(k1)
        x = jax.random.normal(k2, (10, dim))

        y, ld = coupling.forward(params, x)

        assert jnp.allclose(y, x, atol=1e-5)
        assert jnp.allclose(ld, 0.0, atol=1e-5)

    def test_affine_coupling_init_params_with_context(self, key, dim, context_dim):
        """AffineCoupling.init_params works with context_dim > 0."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        mlp, _ = init_mlp(
            key, x_dim=dim, context_dim=context_dim, hidden_dim=16, n_hidden_layers=1, out_dim=2 * dim
        )
        coupling = AffineCoupling(mask=mask, conditioner=mlp)

        k1, k2 = jax.random.split(key)
        params = coupling.init_params(k1, context_dim=context_dim)
        x = jax.random.normal(k2, (10, dim))
        ctx = jax.random.normal(k2, (10, context_dim))

        y, ld = coupling.forward(params, x, context=ctx)

        assert y.shape == x.shape
        assert ld.shape == (10,)

    def test_spline_coupling_init_params_structure(self, key, dim):
        """SplineCoupling.init_params returns correct structure."""
        from nflojax.nets import MLP
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        num_bins = 8
        out_dim = dim * (3 * num_bins - 1)
        mlp = MLP(x_dim=dim, context_dim=0, hidden_dim=16, n_hidden_layers=1, out_dim=out_dim)
        coupling = SplineCoupling(mask=mask, conditioner=mlp, num_bins=num_bins)

        params = coupling.init_params(key)

        assert "mlp" in params
        assert "net" in params["mlp"]
        assert "dense_out" in params["mlp"]["net"]

    def test_spline_coupling_init_params_near_identity(self, key, dim):
        """SplineCoupling default init produces near-identity transform."""
        from nflojax.nets import MLP
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        num_bins = 8
        out_dim = dim * (3 * num_bins - 1)
        mlp = MLP(x_dim=dim, context_dim=0, hidden_dim=16, n_hidden_layers=1, out_dim=out_dim)
        coupling = SplineCoupling(mask=mask, conditioner=mlp, num_bins=num_bins)

        k1, k2 = jax.random.split(key)
        params = coupling.init_params(k1)
        # Use values within spline bounds
        x = jax.random.uniform(k2, (10, dim), minval=-4.0, maxval=4.0)

        y, ld = coupling.forward(params, x)

        # Should be close to identity (within tolerance for spline approx)
        assert jnp.allclose(y, x, atol=0.01)
        assert jnp.allclose(ld, 0.0, atol=0.01)

    def test_spline_coupling_init_params_with_context(self, key, dim, context_dim):
        """SplineCoupling.init_params works with context_dim > 0."""
        from nflojax.nets import MLP
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        num_bins = 8
        out_dim = dim * (3 * num_bins - 1)
        mlp = MLP(x_dim=dim, context_dim=context_dim, hidden_dim=16, n_hidden_layers=1, out_dim=out_dim)
        coupling = SplineCoupling(mask=mask, conditioner=mlp, num_bins=num_bins)

        k1, k2 = jax.random.split(key)
        params = coupling.init_params(k1, context_dim=context_dim)
        x = jax.random.uniform(k2, (10, dim), minval=-4.0, maxval=4.0)
        ctx = jax.random.normal(k2, (10, context_dim))

        y, ld = coupling.forward(params, x, context=ctx)

        assert y.shape == x.shape
        assert ld.shape == (10,)

    def test_composite_init_params_structure(self, key, dim):
        """CompositeTransform.init_params returns list of correct length."""
        perm = jnp.arange(dim - 1, -1, -1)
        loft = LoftTransform(dim=dim, tau=5.0)
        blocks = [Permutation(perm=perm), loft, Permutation(perm=perm)]
        composite = CompositeTransform(blocks=blocks)

        params = composite.init_params(key)

        assert isinstance(params, list)
        assert len(params) == 3
        assert params[0] == {}  # Permutation
        assert params[1] == {}  # LoftTransform
        assert params[2] == {}  # Permutation

    def test_composite_init_params_with_coupling(self, key, dim):
        """CompositeTransform.init_params works with coupling blocks."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        mlp, _ = init_mlp(
            key, x_dim=dim, context_dim=0, hidden_dim=16, n_hidden_layers=1, out_dim=2 * dim
        )
        coupling = AffineCoupling(mask=mask, conditioner=mlp)
        perm = jnp.arange(dim - 1, -1, -1)
        blocks = [coupling, Permutation(perm=perm)]
        composite = CompositeTransform(blocks=blocks)

        params = composite.init_params(key)

        assert len(params) == 2
        assert "mlp" in params[0]  # AffineCoupling params
        assert params[1] == {}      # Permutation params

    def test_composite_init_params_with_context(self, key, dim, context_dim):
        """CompositeTransform.init_params passes context_dim to sub-blocks."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        mlp, _ = init_mlp(
            key, x_dim=dim, context_dim=context_dim, hidden_dim=16, n_hidden_layers=1, out_dim=2 * dim
        )
        coupling = AffineCoupling(mask=mask, conditioner=mlp)
        composite = CompositeTransform(blocks=[coupling])

        params = composite.init_params(key, context_dim=context_dim)
        x = jax.random.normal(key, (10, dim))
        ctx = jax.random.normal(key, (10, context_dim))

        # Should work with context
        y, ld = composite.forward(params, x, context=ctx)

        assert y.shape == x.shape
        assert ld.shape == (10,)


# ============================================================================
# Factory Method Tests (required_out_dim and create)
# ============================================================================
class TestAffineCouplingFactory:
    """Tests for AffineCoupling.required_out_dim and create."""

    def test_required_out_dim(self):
        """required_out_dim returns 2 * dim."""
        assert AffineCoupling.required_out_dim(4) == 8
        assert AffineCoupling.required_out_dim(10) == 20
        assert AffineCoupling.required_out_dim(1) == 2

    def test_create_returns_coupling_and_params(self, key, dim):
        """create returns (coupling, params) tuple."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        coupling, params = AffineCoupling.create(
            key, dim=dim, mask=mask, hidden_dim=16, n_hidden_layers=2
        )

        assert isinstance(coupling, AffineCoupling)
        assert isinstance(params, dict)
        assert "mlp" in params

    def test_create_works_without_context_dim_in_init_params(self, key, dim):
        """create + init_params works without specifying context_dim (inferred)."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        coupling, params = AffineCoupling.create(
            key, dim=dim, mask=mask, hidden_dim=16, n_hidden_layers=2
        )

        x = jax.random.normal(key, (10, dim))
        y, ld = coupling.forward(params, x)

        assert y.shape == x.shape
        assert ld.shape == (10,)

    def test_create_identity_at_init(self, key, dim):
        """create produces identity transform at init."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        coupling, params = AffineCoupling.create(
            key, dim=dim, mask=mask, hidden_dim=16, n_hidden_layers=2
        )

        x = jax.random.normal(key, (10, dim))
        y, ld = coupling.forward(params, x)

        assert jnp.allclose(y, x, atol=1e-5)
        assert jnp.allclose(ld, 0.0, atol=1e-5)

    def test_create_with_context(self, key, dim, context_dim):
        """create works with context_dim > 0."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        coupling, params = AffineCoupling.create(
            key, dim=dim, mask=mask, hidden_dim=16, n_hidden_layers=2,
            context_dim=context_dim,
        )

        x = jax.random.normal(key, (10, dim))
        ctx = jax.random.normal(key, (10, context_dim))
        y, ld = coupling.forward(params, x, context=ctx)

        assert y.shape == x.shape
        assert ld.shape == (10,)

    def test_create_invalid_dim_raises(self, key):
        """create raises ValueError for dim <= 0."""
        mask = jnp.array([1, 0], dtype=jnp.float32)
        with pytest.raises(ValueError, match="dim must be positive"):
            AffineCoupling.create(key, dim=0, mask=mask, hidden_dim=16, n_hidden_layers=2)

    def test_create_mask_mismatch_raises(self, key, dim):
        """create raises ValueError when mask doesn't match dim."""
        wrong_mask = jnp.array([1, 0, 1], dtype=jnp.float32)
        with pytest.raises(ValueError, match="mask shape"):
            AffineCoupling.create(key, dim=dim, mask=wrong_mask, hidden_dim=16, n_hidden_layers=2)

    def test_create_invalid_hidden_dim_raises(self, key, dim):
        """create raises ValueError for hidden_dim <= 0."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        with pytest.raises(ValueError, match="hidden_dim must be positive"):
            AffineCoupling.create(key, dim=dim, mask=mask, hidden_dim=0, n_hidden_layers=2)


class TestSplineCouplingFactory:
    """Tests for SplineCoupling.required_out_dim and create."""

    def test_required_out_dim(self):
        """required_out_dim returns dim * (3K - 1)."""
        assert SplineCoupling.required_out_dim(4, num_bins=8) == 4 * (3 * 8 - 1)  # 92
        assert SplineCoupling.required_out_dim(2, num_bins=4) == 2 * (3 * 4 - 1)  # 22
        assert SplineCoupling.required_out_dim(1, num_bins=2) == 1 * (3 * 2 - 1)  # 5

    def test_create_returns_coupling_and_params(self, key, dim):
        """create returns (coupling, params) tuple."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        coupling, params = SplineCoupling.create(
            key, dim=dim, mask=mask, hidden_dim=16, n_hidden_layers=2, num_bins=8
        )

        assert isinstance(coupling, SplineCoupling)
        assert isinstance(params, dict)
        assert "mlp" in params

    def test_create_works_without_context_dim_in_init_params(self, key, dim):
        """create + init_params works without specifying context_dim (inferred)."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        coupling, params = SplineCoupling.create(
            key, dim=dim, mask=mask, hidden_dim=16, n_hidden_layers=2, num_bins=8
        )

        x = jax.random.uniform(key, (10, dim), minval=-4.0, maxval=4.0)
        y, ld = coupling.forward(params, x)

        assert y.shape == x.shape
        assert ld.shape == (10,)

    def test_create_near_identity_at_init(self, key, dim):
        """create produces near-identity transform at init."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        coupling, params = SplineCoupling.create(
            key, dim=dim, mask=mask, hidden_dim=16, n_hidden_layers=2, num_bins=8
        )

        x = jax.random.uniform(key, (10, dim), minval=-4.0, maxval=4.0)
        y, ld = coupling.forward(params, x)

        assert jnp.allclose(y, x, atol=0.01)
        assert jnp.allclose(ld, 0.0, atol=0.01)

    def test_create_with_context(self, key, dim, context_dim):
        """create works with context_dim > 0."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        coupling, params = SplineCoupling.create(
            key, dim=dim, mask=mask, hidden_dim=16, n_hidden_layers=2,
            num_bins=8, context_dim=context_dim,
        )

        x = jax.random.uniform(key, (10, dim), minval=-4.0, maxval=4.0)
        ctx = jax.random.normal(key, (10, context_dim))
        y, ld = coupling.forward(params, x, context=ctx)

        assert y.shape == x.shape
        assert ld.shape == (10,)

    def test_create_invalid_dim_raises(self, key):
        """create raises ValueError for dim <= 0."""
        mask = jnp.array([1, 0], dtype=jnp.float32)
        with pytest.raises(ValueError, match="dim must be positive"):
            SplineCoupling.create(key, dim=0, mask=mask, hidden_dim=16, n_hidden_layers=2)

    def test_create_mask_mismatch_raises(self, key, dim):
        """create raises ValueError when mask doesn't match dim."""
        wrong_mask = jnp.array([1, 0, 1], dtype=jnp.float32)
        with pytest.raises(ValueError, match="mask shape"):
            SplineCoupling.create(key, dim=dim, mask=wrong_mask, hidden_dim=16, n_hidden_layers=2)

    def test_create_invalid_num_bins_raises(self, key, dim):
        """create raises ValueError for num_bins <= 0."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        with pytest.raises(ValueError, match="num_bins must be positive"):
            SplineCoupling.create(key, dim=dim, mask=mask, hidden_dim=16, n_hidden_layers=2, num_bins=0)


class TestCustomConditionerWithRequiredOutDim:
    """Tests that raw constructor + required_out_dim works for custom architectures."""

    def test_affine_coupling_custom_conditioner(self, key, dim):
        """Raw constructor works with custom MLP using required_out_dim."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        out_dim = AffineCoupling.required_out_dim(dim)

        # Custom MLP with different config
        from nflojax.nets import MLP
        custom_mlp = MLP(
            x_dim=dim, context_dim=0, hidden_dim=128, n_hidden_layers=5, out_dim=out_dim
        )

        coupling = AffineCoupling(mask=mask, conditioner=custom_mlp)
        params = coupling.init_params(key)  # Should infer context_dim=0

        x = jax.random.normal(key, (10, dim))
        y, ld = coupling.forward(params, x)

        assert y.shape == x.shape

    def test_spline_coupling_custom_conditioner(self, key, dim):
        """Raw constructor works with custom MLP using required_out_dim."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        num_bins = 8
        out_dim = SplineCoupling.required_out_dim(dim, num_bins)

        # Custom MLP with different config
        from nflojax.nets import MLP
        custom_mlp = MLP(
            x_dim=dim, context_dim=0, hidden_dim=128, n_hidden_layers=5, out_dim=out_dim
        )

        coupling = SplineCoupling(mask=mask, conditioner=custom_mlp, num_bins=num_bins)
        params = coupling.init_params(key)  # Should infer context_dim=0

        x = jax.random.uniform(key, (10, dim), minval=-4.0, maxval=4.0)
        y, ld = coupling.forward(params, x)

        assert y.shape == x.shape


class TestConditionerInterfaceFallback:
    """Tests for conditioner interface fallback behavior."""

    def test_affine_coupling_skips_init_without_interface(self, key, dim):
        """AffineCoupling skips auto-init if conditioner lacks interface methods."""
        from flax import linen as nn

        # Minimal conditioner without get_output_layer/set_output_layer
        class MinimalConditioner(nn.Module):
            out_dim: int
            context_dim: int = 0

            @nn.compact
            def __call__(self, x, context=None):
                return nn.Dense(self.out_dim)(x)

        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        out_dim = AffineCoupling.required_out_dim(dim)
        conditioner = MinimalConditioner(out_dim=out_dim)

        coupling = AffineCoupling(mask=mask, conditioner=conditioner)
        params = coupling.init_params(key)

        # Should work without error (skips auto-init)
        x = jax.random.normal(key, (10, dim))
        y, ld = coupling.forward(params, x)
        assert y.shape == x.shape

    def test_spline_coupling_raises_without_interface(self, key, dim):
        """SplineCoupling raises error if conditioner lacks interface methods."""
        from flax import linen as nn

        # Minimal conditioner without get_output_layer/set_output_layer
        class MinimalConditioner(nn.Module):
            out_dim: int
            context_dim: int = 0

            @nn.compact
            def __call__(self, x, context=None):
                return nn.Dense(self.out_dim)(x)

        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        num_bins = 8
        out_dim = SplineCoupling.required_out_dim(dim, num_bins)
        conditioner = MinimalConditioner(out_dim=out_dim)

        coupling = SplineCoupling(mask=mask, conditioner=conditioner, num_bins=num_bins)

        with pytest.raises(RuntimeError, match="get_output_layer.*set_output_layer"):
            coupling.init_params(key)


# ============================================================================
# Factory Tests for Simple Transforms (LinearTransform, Permutation, LoftTransform)
# ============================================================================
class TestLinearTransformFactory:
    """Tests for LinearTransform.create."""

    def test_create_returns_transform_and_params(self, key, dim):
        """create returns (transform, params) tuple."""
        transform, params = LinearTransform.create(key, dim=dim)

        assert isinstance(transform, LinearTransform)
        assert isinstance(params, dict)
        assert "lower" in params
        assert "upper" in params
        assert "raw_diag" in params

    def test_create_identity_at_init(self, key, dim):
        """create produces identity transform at init."""
        transform, params = LinearTransform.create(key, dim=dim)

        x = jax.random.normal(key, (10, dim))
        y, ld = transform.forward(params, x)

        assert jnp.allclose(y, x, atol=1e-5)
        assert jnp.allclose(ld, 0.0, atol=1e-5)

    def test_create_invalid_dim_raises(self, key):
        """create raises ValueError for dim <= 0."""
        with pytest.raises(ValueError, match="dim must be positive"):
            LinearTransform.create(key, dim=0)

    def test_create_with_context(self, key, dim):
        """create with context_dim > 0 produces conditioner."""
        context_dim = 8
        transform, params = LinearTransform.create(
            key, dim=dim, context_dim=context_dim, hidden_dim=32, n_hidden_layers=2
        )

        assert transform.conditioner is not None
        assert transform.context_dim == context_dim
        assert "mlp" in params

    def test_create_with_context_identity_at_init(self, key, dim):
        """Conditional LinearTransform is identity at initialization."""
        context_dim = 8
        transform, params = LinearTransform.create(
            key, dim=dim, context_dim=context_dim, hidden_dim=32, n_hidden_layers=2
        )

        x = jax.random.normal(key, (10, dim))
        context = jax.random.normal(key, (10, context_dim))
        y, ld = transform.forward(params, x, context)

        assert jnp.allclose(y, x, atol=1e-5)
        assert jnp.allclose(ld, 0.0, atol=1e-5)

    def test_create_with_context_invertibility(self, key, dim):
        """Conditional LinearTransform is invertible."""
        context_dim = 8
        transform, params = LinearTransform.create(
            key, dim=dim, context_dim=context_dim, hidden_dim=32, n_hidden_layers=2
        )

        # Perturb params away from identity
        k1, k2 = jax.random.split(key)
        params["lower"] = jax.random.normal(k1, (dim, dim)) * 0.1
        params["upper"] = jax.random.normal(k2, (dim, dim)) * 0.1

        x = jax.random.normal(key, (10, dim))
        context = jax.random.normal(key, (10, context_dim))

        y, ld_fwd = transform.forward(params, x, context)
        x_rec, ld_inv = transform.inverse(params, y, context)

        assert jnp.allclose(x, x_rec, atol=1e-5)
        assert jnp.allclose(ld_fwd + ld_inv, 0.0, atol=1e-5)

    def test_create_with_context_different_contexts_give_different_outputs(self, key, dim):
        """Different contexts produce different transformations."""
        context_dim = 8
        transform, params = LinearTransform.create(
            key, dim=dim, context_dim=context_dim, hidden_dim=32, n_hidden_layers=2
        )

        # Perturb MLP output layer to get non-zero conditioner output
        # (at init, output layer is zero so all contexts give same result)
        k1 = jax.random.PRNGKey(999)
        net_params = params["mlp"]["net"]
        net_params["dense_out"]["kernel"] = jax.random.normal(
            k1, net_params["dense_out"]["kernel"].shape
        ) * 0.5

        x = jax.random.normal(key, (5, dim))
        context1 = jax.random.normal(jax.random.PRNGKey(1), (5, context_dim))
        context2 = jax.random.normal(jax.random.PRNGKey(2), (5, context_dim))

        y1, _ = transform.forward(params, x, context1)
        y2, _ = transform.forward(params, x, context2)

        # Outputs should differ for different contexts
        assert not jnp.allclose(y1, y2, atol=1e-3)

    def test_create_invalid_context_dim_raises(self, key, dim):
        """create raises ValueError for context_dim < 0."""
        with pytest.raises(ValueError, match="context_dim must be non-negative"):
            LinearTransform.create(key, dim=dim, context_dim=-1)


class TestPermutationFactory:
    """Tests for Permutation.create."""

    def test_create_returns_transform_and_params(self, key, dim):
        """create returns (transform, params) tuple."""
        perm = jnp.arange(dim - 1, -1, -1)
        transform, params = Permutation.create(key, perm=perm)

        assert isinstance(transform, Permutation)
        assert params == {}

    def test_create_works(self, key, dim):
        """create produces working transform."""
        perm = jnp.arange(dim - 1, -1, -1)  # reverse
        transform, params = Permutation.create(key, perm=perm)

        x = jax.random.normal(key, (10, dim))
        y, ld = transform.forward(params, x)

        # Reverse permutation should reverse the last dimension
        assert jnp.allclose(y, x[..., ::-1])
        assert jnp.allclose(ld, 0.0)


class TestLoftTransformFactory:
    """Tests for LoftTransform.create."""

    def test_create_returns_transform_and_params(self, key, dim):
        """create returns (transform, params) tuple."""
        transform, params = LoftTransform.create(key, dim=dim, tau=5.0)

        assert isinstance(transform, LoftTransform)
        assert params == {}

    def test_create_near_identity_near_zero(self, key, dim):
        """create produces near-identity transform for small inputs."""
        transform, params = LoftTransform.create(key, dim=dim, tau=5.0)

        # Small inputs should be nearly unchanged
        x = jax.random.normal(key, (10, dim)) * 0.1
        y, ld = transform.forward(params, x)

        assert jnp.allclose(y, x, atol=1e-4)

    def test_create_invalid_dim_raises(self, key):
        """create raises ValueError for dim <= 0."""
        with pytest.raises(ValueError, match="dim must be positive"):
            LoftTransform.create(key, dim=0)

    def test_create_invalid_tau_raises(self, key, dim):
        """create raises ValueError for tau <= 0."""
        with pytest.raises(ValueError, match="tau must be positive"):
            LoftTransform.create(key, dim=dim, tau=0.0)


# ============================================================================
# LOFT overflow regression tests (C1)
# ============================================================================
class TestLoftOverflow:
    """Regression tests for LOFT inverse overflow with large inputs."""

    def test_loft_inv_large_input_finite(self):
        """loft_inv must return finite values for |y| >> tau."""
        from nflojax.scalar_function import loft_inv
        y = jnp.array([2000.0, -2000.0, 1100.0, -1100.0])
        x = loft_inv(y, tau=1000.0)
        assert jnp.all(jnp.isfinite(x)), f"loft_inv produced non-finite: {x}"

    def test_loft_roundtrip_large_z(self):
        """loft(loft_inv(y)) should roundtrip for y within recoverable range.

        The clamp at 80 means exact roundtrip works for |y| <= tau + 80.
        Values beyond that saturate but remain finite.
        """
        from nflojax.scalar_function import loft, loft_inv
        # Values within recoverable range (tau=1000, max exact = 1080)
        y = jnp.array([1050.0, -1050.0, 1079.0, -1079.0])
        x = loft_inv(y, tau=1000.0)
        y_rec = loft(x, tau=1000.0)
        assert jnp.all(jnp.isfinite(x)), f"loft_inv produced non-finite: {x}"
        assert jnp.allclose(y_rec, y, atol=1e-3), (
            f"roundtrip failed: max err={jnp.max(jnp.abs(y_rec - y))}"
        )

    def test_loft_inv_beyond_clamp_saturates_finitely(self):
        """Values beyond tau+80 saturate but remain finite (no inf/nan)."""
        from nflojax.scalar_function import loft_inv
        y = jnp.array([1200.0, -1200.0, 5000.0, -5000.0])
        x = loft_inv(y, tau=1000.0)
        assert jnp.all(jnp.isfinite(x)), f"loft_inv produced non-finite: {x}"

    def test_loft_transform_inverse_large(self, key):
        """LoftTransform.inverse must handle large values without overflow."""
        dim = 4
        transform, params = LoftTransform.create(key, dim=dim, tau=1000.0)
        y = jnp.full((5, dim), 2000.0)
        x, log_det = transform.inverse(params, y)
        assert jnp.all(jnp.isfinite(x)), f"inverse produced non-finite x: {x}"
        assert jnp.all(jnp.isfinite(log_det)), f"inverse produced non-finite log_det: {log_det}"


# ============================================================================
# Log-det vs autodiff correctness tests (C3)
# ============================================================================
class TestLogdetVsAutodiff:
    """Verify hand-derived log-det formulas match autodiff Jacobian.

    Uses check_logdet_vs_autodiff from conftest.py on single samples (no batch).
    """

    def test_affine_coupling(self):
        """AffineCoupling log-det matches autodiff."""

        key = jax.random.PRNGKey(0)
        dim = 4
        mask = jnp.array([1, 0, 1, 0], dtype=jnp.float32)
        coupling, params = AffineCoupling.create(
            key, dim=dim, mask=mask, hidden_dim=16, n_hidden_layers=1
        )
        x = jax.random.normal(jax.random.PRNGKey(1), (dim,))
        result = check_logdet_vs_autodiff(
            lambda z: coupling.forward(params, z), x
        )
        assert result["error"] < 1e-4, (
            f"AffineCoupling log-det error: {result['error']}"
        )

    def test_affine_coupling_conditional(self):
        """AffineCoupling with context: log-det matches autodiff."""

        key = jax.random.PRNGKey(0)
        dim = 4
        mask = jnp.array([1, 0, 1, 0], dtype=jnp.float32)
        coupling, params = AffineCoupling.create(
            key, dim=dim, mask=mask, hidden_dim=16, n_hidden_layers=1,
            context_dim=2
        )
        x = jax.random.normal(jax.random.PRNGKey(1), (dim,))
        ctx = jax.random.normal(jax.random.PRNGKey(2), (2,))
        result = check_logdet_vs_autodiff(
            lambda z: coupling.forward(params, z, ctx), x
        )
        assert result["error"] < 1e-4, (
            f"AffineCoupling conditional log-det error: {result['error']}"
        )

    def test_spline_coupling(self):
        """SplineCoupling log-det matches autodiff."""

        key = jax.random.PRNGKey(0)
        dim = 4
        mask = jnp.array([1, 0, 1, 0], dtype=jnp.float32)
        coupling, params = SplineCoupling.create(
            key, dim=dim, mask=mask, hidden_dim=16, n_hidden_layers=1,
            num_bins=8
        )
        x = jax.random.normal(jax.random.PRNGKey(1), (dim,)) * 0.5
        result = check_logdet_vs_autodiff(
            lambda z: coupling.forward(params, z), x
        )
        assert result["error"] < 1e-3, (
            f"SplineCoupling log-det error: {result['error']}"
        )

    def test_linear_transform(self):
        """LinearTransform log-det matches autodiff."""

        key = jax.random.PRNGKey(0)
        dim = 4
        transform, _ = LinearTransform.create(key, dim=dim)
        params = {
            "lower": jax.random.normal(key, (dim, dim)) * 0.1,
            "upper": jax.random.normal(jax.random.PRNGKey(1), (dim, dim)) * 0.1,
            "raw_diag": jax.random.normal(jax.random.PRNGKey(2), (dim,)) * 0.5,
        }
        x = jax.random.normal(jax.random.PRNGKey(3), (dim,))
        result = check_logdet_vs_autodiff(
            lambda z: transform.forward(params, z), x
        )
        assert result["error"] < 1e-4, (
            f"LinearTransform log-det error: {result['error']}"
        )

    def test_loft_transform(self):
        """LoftTransform log-det matches autodiff."""

        key = jax.random.PRNGKey(0)
        dim = 4
        transform, params = LoftTransform.create(key, dim=dim, tau=5.0)
        x = jax.random.normal(jax.random.PRNGKey(1), (dim,)) * 3.0
        result = check_logdet_vs_autodiff(
            lambda z: transform.forward(params, z), x
        )
        assert result["error"] < 1e-4, (
            f"LoftTransform log-det error: {result['error']}"
        )

    def test_permutation(self):
        """Permutation log-det is zero (matches autodiff)."""

        dim = 4
        perm = jnp.array([2, 0, 3, 1])
        transform, params = Permutation.create(jax.random.PRNGKey(0), perm=perm)
        x = jax.random.normal(jax.random.PRNGKey(1), (dim,))
        result = check_logdet_vs_autodiff(
            lambda z: transform.forward(params, z), x
        )
        assert result["error"] < 1e-6, (
            f"Permutation log-det error: {result['error']}"
        )

    def test_composite_transform(self):
        """CompositeTransform (affine + linear) log-det matches autodiff."""

        key = jax.random.PRNGKey(0)
        dim = 4
        mask = jnp.array([1, 0, 1, 0], dtype=jnp.float32)
        k1, k2 = jax.random.split(key)
        coupling, c_params = AffineCoupling.create(
            k1, dim=dim, mask=mask, hidden_dim=16, n_hidden_layers=1
        )
        linear, _ = LinearTransform.create(k2, dim=dim)
        l_params = {
            "lower": jax.random.normal(k2, (dim, dim)) * 0.1,
            "upper": jax.random.normal(jax.random.PRNGKey(10), (dim, dim)) * 0.1,
            "raw_diag": jax.random.normal(jax.random.PRNGKey(11), (dim,)) * 0.3,
        }
        composite = CompositeTransform(blocks=[coupling, linear])
        all_params = [c_params, l_params]
        x = jax.random.normal(jax.random.PRNGKey(1), (dim,))
        result = check_logdet_vs_autodiff(
            lambda z: composite.forward(all_params, z), x
        )
        assert result["error"] < 1e-3, (
            f"CompositeTransform log-det error: {result['error']}"
        )


# ============================================================================
# SplitCoupling Tests (structured rank-N events)
# ============================================================================
#
# SplitCoupling is the structured analogue of SplineCoupling. It splits along
# a tensor axis (not a flat mask) and sums log-det over the trailing
# event_ndims axes. Target use case: (B, N, d) particle systems with
# split_axis=-2 (particle axis), event_ndims=2.
class TestSplitCoupling:
    """Tests for SplitCoupling on rank-2 events shaped (N, d)."""

    @pytest.fixture
    def small_event(self):
        """Small particle system: 6 particles in 3D."""
        return {"N": 6, "d": 3}

    def test_create_returns_coupling_and_params(self, key, small_event):
        """create returns (coupling, params) with the expected structure."""
        N, d = small_event["N"], small_event["d"]
        coupling, params = SplitCoupling.create(
            key,
            event_shape=(N, d),
            split_axis=-2,
            split_index=N // 2,
            event_ndims=2,
            hidden_dim=16,
            n_hidden_layers=2,
            num_bins=4,
            tail_bound=5.0,
        )
        assert isinstance(coupling, SplitCoupling)
        assert isinstance(params, dict)
        assert "mlp" in params

    def test_forward_preserves_shape(self, key, small_event):
        """Forward input (B, N, d) -> output (B, N, d)."""
        N, d = small_event["N"], small_event["d"]
        coupling, params = SplitCoupling.create(
            key, event_shape=(N, d), split_axis=-2, split_index=N // 2,
            event_ndims=2, hidden_dim=16, n_hidden_layers=2, num_bins=4,
        )
        x = jax.random.uniform(key, (8, N, d), minval=-4.0, maxval=4.0)
        y, log_det = coupling.forward(params, x)
        assert y.shape == x.shape
        assert log_det.shape == (8,)

    def test_frozen_slice_is_identity(self, key, small_event):
        """The frozen particle slice is passed through unchanged."""
        N, d = small_event["N"], small_event["d"]
        split_index = N // 2
        coupling, params = SplitCoupling.create(
            key, event_shape=(N, d), split_axis=-2, split_index=split_index,
            event_ndims=2, hidden_dim=16, n_hidden_layers=2, num_bins=4,
        )
        x = jax.random.uniform(key, (4, N, d), minval=-4.0, maxval=4.0)
        y, _ = coupling.forward(params, x)

        # swap=False => first `split_index` particles are frozen.
        assert jnp.allclose(y[:, :split_index, :], x[:, :split_index, :])

    def test_swap_flips_frozen_slice(self, key, small_event):
        """swap=True freezes the second slice instead of the first."""
        N, d = small_event["N"], small_event["d"]
        split_index = N // 2
        coupling, params = SplitCoupling.create(
            key, event_shape=(N, d), split_axis=-2, split_index=split_index,
            event_ndims=2, hidden_dim=16, n_hidden_layers=2, num_bins=4,
            swap=True,
        )
        x = jax.random.uniform(key, (4, N, d), minval=-4.0, maxval=4.0)
        y, _ = coupling.forward(params, x)

        # swap=True => last `N - split_index` particles are frozen.
        assert jnp.allclose(y[:, split_index:, :], x[:, split_index:, :])

    def test_near_identity_at_init(self, key, small_event):
        """Zero-bias spline init => forward is near-identity inside [-B, B]."""
        N, d = small_event["N"], small_event["d"]
        coupling, params = SplitCoupling.create(
            key, event_shape=(N, d), split_axis=-2, split_index=N // 2,
            event_ndims=2, hidden_dim=16, n_hidden_layers=2, num_bins=4,
            tail_bound=5.0,
        )
        x = jax.random.uniform(key, (10, N, d), minval=-4.0, maxval=4.0)
        y, log_det = coupling.forward(params, x)

        assert jnp.allclose(y, x, atol=0.01)
        assert jnp.allclose(log_det, 0.0, atol=0.01)

    @requires_x64
    def test_inverse_roundtrip(self, key, small_event):
        """inverse(forward(x)) == x."""
        N, d = small_event["N"], small_event["d"]
        coupling, params = SplitCoupling.create(
            key, event_shape=(N, d), split_axis=-2, split_index=N // 2,
            event_ndims=2, hidden_dim=16, n_hidden_layers=2, num_bins=4,
        )
        # Break identity init by adding noise to params so the test exercises a
        # non-trivial transform.
        params = jax.tree_util.tree_map(lambda p: p + 0.3 * jax.random.normal(key, p.shape), params)

        x = jax.random.uniform(key, (4, N, d), minval=-3.5, maxval=3.5)
        y, ld_fwd = coupling.forward(params, x)
        x_back, ld_inv = coupling.inverse(params, y)

        assert jnp.allclose(x_back, x, atol=1e-4)
        assert jnp.allclose(ld_fwd + ld_inv, 0.0, atol=1e-4)

    def test_log_det_vs_autodiff(self, key, small_event):
        """log_det matches the autodiff Jacobian determinant on a single sample."""
        N, d = small_event["N"], small_event["d"]
        coupling, params = SplitCoupling.create(
            key, event_shape=(N, d), split_axis=-2, split_index=N // 2,
            event_ndims=2, hidden_dim=16, n_hidden_layers=2, num_bins=4,
        )
        params = jax.tree_util.tree_map(lambda p: p + 0.3 * jax.random.normal(key, p.shape), params)

        def fwd_single(z):
            # z: (N, d). coupling.forward expects a batch axis.
            y, ld = coupling.forward(params, z[None])
            return y[0], ld[0]

        x = jax.random.uniform(key, (N, d), minval=-3.0, maxval=3.0)
        result = check_logdet_vs_autodiff_structured(fwd_single, x, event_ndims=2)
        assert result["error"] < 1e-3, f"SplitCoupling log-det error: {result['error']}"

    def test_alternating_swap_covers_all(self, key, small_event):
        """Two layers with opposite swap transform every scalar."""
        N, d = small_event["N"], small_event["d"]
        split_index = N // 2
        c_a, p_a = SplitCoupling.create(
            key, event_shape=(N, d), split_axis=-2, split_index=split_index,
            event_ndims=2, hidden_dim=16, n_hidden_layers=2, num_bins=4,
            swap=False,
        )
        c_b, p_b = SplitCoupling.create(
            jax.random.split(key)[0], event_shape=(N, d), split_axis=-2,
            split_index=split_index, event_ndims=2, hidden_dim=16,
            n_hidden_layers=2, num_bins=4, swap=True,
        )
        # Break identity init.
        p_a = jax.tree_util.tree_map(lambda p: p + 0.3 * jax.random.normal(key, p.shape), p_a)
        p_b = jax.tree_util.tree_map(lambda p: p + 0.3 * jax.random.normal(key, p.shape), p_b)

        x = jax.random.uniform(key, (1, N, d), minval=-2.0, maxval=2.0)
        y1, _ = c_a.forward(p_a, x)
        y2, _ = c_b.forward(p_b, y1)

        # After two layers every particle must differ from the input (no particle
        # is left identity-mapped).
        diff = jnp.abs(y2 - x)
        per_particle_diff = jnp.sum(diff, axis=-1)  # (1, N)
        assert jnp.all(per_particle_diff > 1e-6), (
            f"Some particles untouched; per-particle diffs: {per_particle_diff}"
        )

    def test_jit_compatible(self, key, small_event):
        """SplitCoupling forward/inverse are JIT-compatible."""
        N, d = small_event["N"], small_event["d"]
        coupling, params = SplitCoupling.create(
            key, event_shape=(N, d), split_axis=-2, split_index=N // 2,
            event_ndims=2, hidden_dim=16, n_hidden_layers=2, num_bins=4,
        )
        x = jax.random.uniform(key, (4, N, d), minval=-3.0, maxval=3.0)
        fwd = jax.jit(coupling.forward)
        inv = jax.jit(coupling.inverse)
        y, ld_f = fwd(params, x)
        x_back, ld_i = inv(params, y)
        assert y.shape == x.shape
        assert jnp.allclose(x_back, x, atol=1e-4)

    def test_direct_construction_init_params(self, key, small_event):
        """Constructing SplitCoupling directly and calling init_params works
        without needing frozen_flat/transformed_flat kwargs."""
        from nflojax.nets import MLP
        N, d = small_event["N"], small_event["d"]
        K = 4
        frozen_flat = (N // 2) * d
        transformed_flat = (N // 2) * d
        mlp = MLP(
            x_dim=frozen_flat,
            context_dim=0,
            hidden_dim=16,
            n_hidden_layers=2,
            out_dim=transformed_flat * (3 * K - 1),
        )
        coupling = SplitCoupling(
            event_shape=(N, d),
            split_axis=-2,
            split_index=N // 2,
            event_ndims=2,
            conditioner=mlp,
            num_bins=K,
        )
        params = coupling.init_params(key)
        x = jax.random.uniform(key, (3, N, d), minval=-3.0, maxval=3.0)
        y, ld = coupling.forward(params, x)
        assert y.shape == x.shape
        assert ld.shape == (3,)

    def test_event_shape_rank_mismatch_raises(self, key):
        """event_shape rank must equal event_ndims."""
        from nflojax.nets import MLP
        mlp = MLP(x_dim=4, context_dim=0, hidden_dim=8, n_hidden_layers=1, out_dim=1)
        with pytest.raises(ValueError, match="event_shape"):
            SplitCoupling(
                event_shape=(4,),      # rank 1
                split_axis=-2,
                split_index=1,
                event_ndims=2,          # but claims rank 2
                conditioner=mlp,
            )

    def test_forward_wrong_shape_raises(self, key, small_event):
        """forward on input with trailing shape != event_shape raises cleanly."""
        N, d = small_event["N"], small_event["d"]
        coupling, params = SplitCoupling.create(
            key, event_shape=(N, d), split_axis=-2, split_index=N // 2,
            event_ndims=2, hidden_dim=16, n_hidden_layers=2, num_bins=4,
        )
        # Correct rank, wrong trailing axis size.
        x_bad = jax.random.uniform(key, (4, N, d + 1), minval=-3.0, maxval=3.0)
        with pytest.raises(ValueError, match="event_shape"):
            coupling.forward(params, x_bad)
        with pytest.raises(ValueError, match="event_shape"):
            coupling.inverse(params, x_bad)
        # Too-low rank.
        x_low = jax.random.uniform(key, (N * d,), minval=-3.0, maxval=3.0)
        with pytest.raises(ValueError, match="event_ndims"):
            coupling.forward(params, x_low)


# ============================================================================
# CircularShift Tests (rigid torus rotation)
# ============================================================================
#
# CircularShift is the rotation half of a torus diffeomorphism. It applies a
# per-coordinate learnable shift with modular wrap: y = (x - lower + shift)
# mod (upper - lower) + lower. Log-det = 0 by construction (rigid shift).
#
# Composed with a circular-mode spline coupling, CircularShift + spline
# cover general torus diffeomorphisms (the global rotation comes from the
# shift; local deformation from the spline).
class TestCircularShift:
    """Tests for CircularShift (rigid shift mod L)."""

    def test_forward_preserves_shape(self, key):
        """Rank-3 input (B, N, d) yields output of identical shape."""
        N, d = 6, 3
        shift = CircularShift.from_scalar_box(coord_dim=d, lower=-1.0, upper=1.0)
        params = {"shift": jnp.array([0.3, -0.2, 0.5])}
        x = jax.random.uniform(key, (8, N, d), minval=-1.0, maxval=1.0)
        y, log_det = shift.forward(params, x)
        assert y.shape == x.shape

    def test_round_trip(self, key):
        """inverse(forward(x)) == x to numerical tolerance."""
        N, d = 6, 3
        shift = CircularShift.from_scalar_box(coord_dim=d, lower=-1.0, upper=1.0)
        params = {"shift": jnp.array([0.3, -0.2, 0.5])}
        x = jax.random.uniform(key, (8, N, d), minval=-1.0, maxval=1.0)
        y, _ = shift.forward(params, x)
        x_back, _ = shift.inverse(params, y)
        assert jnp.allclose(x, x_back, atol=1e-6)

    def test_zero_shift_is_identity(self, key):
        """With shift=0, forward is identity on in-box inputs."""
        d = 3
        shift = CircularShift.from_scalar_box(coord_dim=d, lower=-1.0, upper=1.0)
        params = shift.init_params(key)  # init_params returns zero shift
        x = jax.random.uniform(key, (4, 5, d), minval=-0.9, maxval=0.9)
        y, _ = shift.forward(params, x)
        assert jnp.allclose(y, x, atol=1e-6)

    def test_shift_by_L_is_identity(self, key):
        """Shift by the box length equals identity (modular wrap)."""
        d = 3
        # lower=-1, upper=1 so L = 2.
        shift = CircularShift.from_scalar_box(coord_dim=d, lower=-1.0, upper=1.0)
        params = {"shift": jnp.full((d,), 2.0)}
        x = jax.random.uniform(key, (4, 5, d), minval=-0.9, maxval=0.9)
        y, _ = shift.forward(params, x)
        assert jnp.allclose(y, x, atol=1e-6)

    def test_log_det_is_zero(self, key):
        """Log-det is exactly zero (scalar or broadcast-compatible)."""
        d = 3
        shift = CircularShift.from_scalar_box(coord_dim=d, lower=-1.0, upper=1.0)
        params = {"shift": jnp.array([0.3, -0.2, 0.5])}
        x = jax.random.uniform(key, (8, 4, d), minval=-0.9, maxval=0.9)
        _, log_det = shift.forward(params, x)
        assert jnp.all(log_det == 0.0)
        _, log_det_inv = shift.inverse(params, x)
        assert jnp.all(log_det_inv == 0.0)

    def test_jit(self, key):
        """forward/inverse JIT-compatible."""
        d = 3
        shift = CircularShift.from_scalar_box(coord_dim=d, lower=-1.0, upper=1.0)
        params = {"shift": jnp.array([0.3, -0.2, 0.5])}
        x = jax.random.uniform(key, (4, 5, d), minval=-0.9, maxval=0.9)
        fwd = jax.jit(shift.forward)
        inv = jax.jit(shift.inverse)
        y, _ = fwd(params, x)
        x_back, _ = inv(params, y)
        assert jnp.allclose(x_back, x, atol=1e-6)

    def test_out_of_range_input_is_wrapped(self):
        """Input slightly outside the box wraps back in."""
        d = 1
        shift = CircularShift.from_scalar_box(coord_dim=d, lower=0.0, upper=1.0)
        params = {"shift": jnp.array([0.0])}  # no shift
        # 1.1 is 0.1 above upper; mod wrap brings it to 0.1 above lower.
        x = jnp.array([[1.1]])
        y, _ = shift.forward(params, x)
        assert jnp.allclose(y, jnp.array([[0.1]]), atol=1e-6)

    def test_inverted_box_raises(self):
        """Constructing with upper <= lower raises at geometry-construction time."""
        with pytest.raises(ValueError, match="side must be positive"):
            CircularShift.from_scalar_box(coord_dim=3, lower=1.0, upper=0.0)


# ============================================================================
# SplineCoupling with boundary_slopes='circular'
# ============================================================================
class TestSplineCouplingCircular:
    """Flat spline coupling with matched-boundary-slope (circular) mode."""

    def test_required_out_dim_circular(self):
        """out_dim = dim * 3K for circular, dim * (3K-1) for linear_tails."""
        assert SplineCoupling.required_out_dim(4, num_bins=8, boundary_slopes="circular") == 4 * 3 * 8
        assert SplineCoupling.required_out_dim(4, num_bins=8) == 4 * (3 * 8 - 1)

    def test_create_circular_near_identity(self, key, dim):
        """create with circular mode + zero-init gives near-identity transform."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        coupling, params = SplineCoupling.create(
            key, dim=dim, mask=mask, hidden_dim=16, n_hidden_layers=2,
            num_bins=8, boundary_slopes="circular",
        )
        # Inputs inside [-tail_bound, tail_bound] = [-5, 5].
        x = jax.random.uniform(key, (10, dim), minval=-4.0, maxval=4.0)
        y, ld = coupling.forward(params, x)
        assert jnp.allclose(y, x, atol=0.01)
        assert jnp.allclose(ld, 0.0, atol=0.01)

    def test_circular_round_trip(self, key, dim):
        """After breaking identity init, inverse(forward(x)) == x."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        coupling, params = SplineCoupling.create(
            key, dim=dim, mask=mask, hidden_dim=16, n_hidden_layers=2,
            num_bins=8, boundary_slopes="circular",
        )
        params = jax.tree_util.tree_map(
            lambda p: p + 0.2 * jax.random.normal(key, p.shape), params
        )
        x = jax.random.uniform(key, (8, dim), minval=-3.5, maxval=3.5)
        y, ld_f = coupling.forward(params, x)
        x_back, ld_i = coupling.inverse(params, y)
        assert jnp.allclose(x_back, x, atol=1e-4)
        assert jnp.allclose(ld_f + ld_i, 0.0, atol=1e-4)

    def test_circular_with_gvalue_zero_is_identity(self, key, dim):
        """Circular coupling + g_value=0 must still collapse to identity.

        Closes the coverage gap where the identity-gate interpolation path
        reshapes derivatives as (..., dim, K) instead of (..., dim, K-1).
        """
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        coupling, params = SplineCoupling.create(
            key, dim=dim, mask=mask, hidden_dim=16, n_hidden_layers=2,
            context_dim=1, num_bins=8, boundary_slopes="circular",
        )
        # Perturb so the "ungated" transform is non-identity; the gate must
        # still drive the output back to x.
        params = jax.tree_util.tree_map(
            lambda p: p + 0.2 * jax.random.normal(key, p.shape), params
        )
        x = jax.random.uniform(key, (10, dim), minval=-4.0, maxval=4.0)
        context = jnp.zeros((10, 1))
        g_value = jnp.zeros(10)
        y, log_det = coupling.forward(params, x, context, g_value=g_value)
        assert jnp.allclose(y, x, atol=1e-5)
        assert jnp.allclose(log_det, 0.0, atol=1e-5)

    def test_circular_with_context_round_trip(self, key, dim):
        """Conditional circular coupling: inverse(forward(x, ctx)) == x."""
        mask = jnp.array([1, 0] * (dim // 2), dtype=jnp.float32)
        coupling, params = SplineCoupling.create(
            key, dim=dim, mask=mask, hidden_dim=16, n_hidden_layers=2,
            context_dim=3, num_bins=8, boundary_slopes="circular",
        )
        params = jax.tree_util.tree_map(
            lambda p: p + 0.2 * jax.random.normal(key, p.shape), params
        )
        x = jax.random.uniform(key, (8, dim), minval=-3.5, maxval=3.5)
        context = jax.random.normal(jax.random.fold_in(key, 1), (8, 3))
        y, ld_f = coupling.forward(params, x, context)
        x_back, ld_i = coupling.inverse(params, y, context)
        assert jnp.allclose(x_back, x, atol=1e-4)
        # atol=5e-4 for log-det: float32 RQS inverse accumulates ~3e-4 of
        # roundoff across the forward+inverse composition.
        assert jnp.allclose(ld_f + ld_i, 0.0, atol=5e-4)


# ============================================================================
# SplitCoupling with boundary_slopes='circular'
# ============================================================================
class TestSplitCouplingCircular:
    """Structured spline coupling with matched-boundary-slope (circular) mode."""

    def test_required_out_dim_circular(self):
        """out_dim = transformed_flat * 3K for circular."""
        assert SplitCoupling.required_out_dim(
            transformed_flat=6, num_bins=4, boundary_slopes="circular"
        ) == 6 * 3 * 4
        assert SplitCoupling.required_out_dim(
            transformed_flat=6, num_bins=4
        ) == 6 * (3 * 4 - 1)

    def test_create_circular_near_identity(self, key):
        """Zero-init + circular mode → near-identity on rank-3 input."""
        N, d = 6, 3
        coupling, params = SplitCoupling.create(
            key, event_shape=(N, d), split_axis=-2, split_index=N // 2,
            event_ndims=2, hidden_dim=16, n_hidden_layers=2,
            num_bins=4, tail_bound=5.0, boundary_slopes="circular",
        )
        x = jax.random.uniform(key, (4, N, d), minval=-3.5, maxval=3.5)
        y, ld = coupling.forward(params, x)
        assert jnp.allclose(y, x, atol=0.01)
        assert jnp.allclose(ld, 0.0, atol=0.01)

    def test_circular_round_trip(self, key):
        """Non-trivial circular SplitCoupling: forward/inverse invertible.

        Tolerance 1e-3 and a gentler 0.2*randn perturbation (vs 0.3 earlier):
        the circular shared boundary slope can take large values under big
        perturbations, amplifying float32 RQS-inverse roundoff past 1e-3.
        """
        N, d = 6, 3
        coupling, params = SplitCoupling.create(
            key, event_shape=(N, d), split_axis=-2, split_index=N // 2,
            event_ndims=2, hidden_dim=16, n_hidden_layers=2,
            num_bins=4, boundary_slopes="circular",
        )
        params = jax.tree_util.tree_map(
            lambda p: p + 0.2 * jax.random.normal(key, p.shape), params
        )
        x = jax.random.uniform(key, (4, N, d), minval=-3.5, maxval=3.5)
        y, ld_f = coupling.forward(params, x)
        x_back, ld_i = coupling.inverse(params, y)
        assert jnp.allclose(x_back, x, atol=1e-3)
        assert jnp.allclose(ld_f + ld_i, 0.0, atol=1e-3)


# ============================================================================
# Periodic-flow composition: CircularShift + SplitCoupling(circular)
# ============================================================================
# End-to-end unit test for the decomposition (rigid rotation) ∘ (seam-smooth
# deformation). Together these primitives should cover any torus
# diffeomorphism; individually they do not. This test catches bugs that only
# surface when they're composed — e.g., broadcast mistakes in log-det shape,
# a SplitCoupling that can't digest the output of a CircularShift, or
# inputs that drift outside [lower, upper] after the shift.
class TestPeriodicComposition:
    """CircularShift + SplitCoupling(boundary_slopes='circular') in a stack."""

    @pytest.fixture
    def periodic_stack(self, key):
        """Build a 4-layer stack on event_shape=(N=6, d=3) and perturb away
        from identity init so the composition exercises a non-trivial map."""
        N, d = 6, 3
        B = 1.0
        keys = jax.random.split(key, 4)
        blocks = []
        params_list = []
        for i, k in enumerate(keys):
            shift = CircularShift.from_scalar_box(coord_dim=d, lower=-B, upper=B)
            s_params = shift.init_params(k)
            # Break identity by injecting a non-zero shift.
            s_params = {"shift": s_params["shift"] + 0.3}
            blocks.append(shift)
            params_list.append(s_params)

            coupling, c_params = SplitCoupling.create(
                jax.random.fold_in(k, 11),
                event_shape=(N, d),
                split_axis=-2,
                split_index=N // 2,
                event_ndims=2,
                hidden_dim=16,
                n_hidden_layers=2,
                num_bins=8,
                tail_bound=B,
                boundary_slopes="circular",
                swap=(i % 2 == 1),
            )
            # Perturb to move away from the identity-init state.
            c_params = jax.tree_util.tree_map(
                lambda p: p + 0.2 * jax.random.normal(k, p.shape), c_params
            )
            blocks.append(coupling)
            params_list.append(c_params)
        composite = CompositeTransform(blocks=blocks)
        return {"composite": composite, "params": params_list, "N": N, "d": d, "B": B}

    def test_stays_in_box(self, key, periodic_stack):
        """Output of the full stack is always inside [-B, B]."""
        c = periodic_stack["composite"]
        params = periodic_stack["params"]
        N, d, B = periodic_stack["N"], periodic_stack["d"], periodic_stack["B"]
        x = jax.random.uniform(
            jax.random.fold_in(key, 1), (8, N, d), minval=-B * 0.99, maxval=B * 0.99
        )
        y, _ = c.forward(params, x)
        assert jnp.all(y >= -B)
        assert jnp.all(y <= B)

    def test_round_trip(self, key, periodic_stack):
        """inverse(forward(x)) ≈ x on in-box inputs."""
        c = periodic_stack["composite"]
        params = periodic_stack["params"]
        N, d, B = periodic_stack["N"], periodic_stack["d"], periodic_stack["B"]
        x = jax.random.uniform(
            jax.random.fold_in(key, 2), (4, N, d), minval=-B * 0.9, maxval=B * 0.9
        )
        y, ld_f = c.forward(params, x)
        x_back, ld_i = c.inverse(params, y)
        assert jnp.allclose(x_back, x, atol=1e-3)
        assert jnp.allclose(ld_f + ld_i, 0.0, atol=1e-3)

    def test_log_det_is_batch_shape(self, key, periodic_stack):
        """log_det has batch shape (B,), not (B, N) or (B, N, d).

        Catches silent broadcast bugs where a non-batch-reducing transform
        widens the accumulator.
        """
        c = periodic_stack["composite"]
        params = periodic_stack["params"]
        N, d, B = periodic_stack["N"], periodic_stack["d"], periodic_stack["B"]
        batch = 5
        x = jax.random.uniform(
            jax.random.fold_in(key, 3), (batch, N, d), minval=-B * 0.9, maxval=B * 0.9
        )
        _, log_det = c.forward(params, x)
        assert log_det.shape == (batch,), (
            f"expected log_det shape ({batch},), got {log_det.shape}"
        )

    def test_jit(self, key, periodic_stack):
        """Full stack forward/inverse compile under jit."""
        c = periodic_stack["composite"]
        params = periodic_stack["params"]
        N, d, B = periodic_stack["N"], periodic_stack["d"], periodic_stack["B"]
        x = jax.random.uniform(
            jax.random.fold_in(key, 4), (4, N, d), minval=-B * 0.9, maxval=B * 0.9
        )
        fwd = jax.jit(c.forward)
        inv = jax.jit(c.inverse)
        y, _ = fwd(params, x)
        x_back, _ = inv(params, y)
        assert y.shape == x.shape
        assert jnp.allclose(x_back, x, atol=1e-3)


# ============================================================================
# Rescale (fixed per-axis affine from geometry.box to canonical range)
# ============================================================================
class TestRescale:
    """Fixed per-axis affine from Geometry to a target range."""

    def test_round_trip_rank1(self, key):
        """Rank-1 event (B, d): inverse(forward(x)) == x."""
        geom = Geometry(lower=[-2.0, -2.0, -2.0], upper=[3.0, 3.0, 3.0])
        rescale = Rescale(geometry=geom)
        x = jax.random.uniform(key, (8, 3), minval=-2.0, maxval=3.0)
        y, _ = rescale.forward({}, x)
        x_back, _ = rescale.inverse({}, y)
        assert jnp.allclose(x, x_back, atol=1e-6)

    def test_round_trip_rank2_particle_event(self, key):
        """Rank-2 event (B, N, d): inverse(forward(x)) == x."""
        N, d = 5, 3
        geom = Geometry(lower=[-2.0, -2.0, -2.0], upper=[3.0, 3.0, 3.0])
        rescale = Rescale(geometry=geom, event_shape=(N, d))
        x = jax.random.uniform(key, (4, N, d), minval=-2.0, maxval=3.0)
        y, _ = rescale.forward({}, x)
        x_back, _ = rescale.inverse({}, y)
        assert jnp.allclose(x, x_back, atol=1e-6)

    def test_forward_output_in_target_range(self, key):
        """Forward maps geometry.box onto [-1, 1] by default."""
        geom = Geometry(lower=[-2.0, -4.0, 0.0], upper=[3.0, 4.0, 10.0])
        rescale = Rescale(geometry=geom)
        x = jnp.stack(
            [
                jnp.array([-2.0, -4.0, 0.0]),  # box lower corner
                jnp.array([3.0, 4.0, 10.0]),   # box upper corner
                jnp.array([0.5, 0.0, 5.0]),    # midpoint
            ]
        )
        y, _ = rescale.forward({}, x)
        assert jnp.allclose(y[0], jnp.array([-1.0, -1.0, -1.0]), atol=1e-6)
        assert jnp.allclose(y[1], jnp.array([1.0, 1.0, 1.0]), atol=1e-6)
        assert jnp.allclose(y[2], jnp.array([0.0, 0.0, 0.0]), atol=1e-6)

    def test_log_det_closed_form_rank1(self, key):
        """log_det == sum(log(scale)) on rank-1 event."""
        geom = Geometry(lower=[-2.0, -4.0, 0.0], upper=[3.0, 4.0, 10.0])
        rescale = Rescale(geometry=geom)
        box = jnp.asarray(geom.box)
        scale = 2.0 / box  # target span (2) / box span
        expected = jnp.sum(jnp.log(scale))
        x = jax.random.uniform(key, (8, 3), minval=-2.0, maxval=3.0)
        _, log_det_f = rescale.forward({}, x)
        _, log_det_i = rescale.inverse({}, x)
        assert jnp.allclose(log_det_f, expected, atol=1e-6)
        assert jnp.allclose(log_det_i, -expected, atol=1e-6)

    def test_log_det_autodiff_rank1(self, key):
        """log_det agrees with autodiff Jacobian on rank-1 event."""
        geom = Geometry(lower=[-2.0, -4.0, 0.0], upper=[3.0, 4.0, 10.0])
        rescale = Rescale(geometry=geom)
        x = jax.random.uniform(key, (3,), minval=-2.0, maxval=3.0)
        result = check_logdet_vs_autodiff(
            lambda z: rescale.forward({}, z), x, atol=1e-5
        )
        assert result["error"] < 1e-5, result

    def test_log_det_rank2_scales_with_N(self, key):
        """log_det on (N, d) event equals N * sum(log(scale))."""
        N, d = 6, 3
        geom = Geometry(lower=[-2.0, -4.0, 0.0], upper=[3.0, 4.0, 10.0])
        rescale = Rescale(geometry=geom, event_shape=(N, d))
        scale = 2.0 / jnp.asarray(geom.box)
        expected = N * jnp.sum(jnp.log(scale))
        x = jax.random.uniform(key, (4, N, d), minval=-2.0, maxval=3.0)
        _, log_det_f = rescale.forward({}, x)
        assert jnp.allclose(log_det_f, expected, atol=1e-5)

        # Cross-check via autodiff on a single flattened sample.
        def flat_forward(x_flat):
            x_shape = x_flat.reshape(N, d)
            y, ld = rescale.forward({}, x_shape)
            return y.reshape(-1), ld

        x_single = x[0].reshape(-1)
        result = check_logdet_vs_autodiff(flat_forward, x_single, atol=1e-5)
        assert result["error"] < 1e-4, result

    def test_identity_when_box_matches_target(self, key):
        """Geometry.cubic([-1, 1]^d) + target=(-1, 1) -> identity."""
        d = 3
        geom = Geometry.cubic(d=d, side=2.0, lower=-1.0)
        rescale = Rescale(geometry=geom, target=(-1.0, 1.0))
        x = jax.random.uniform(key, (10, d), minval=-0.9, maxval=0.9)
        y, log_det = rescale.forward({}, x)
        assert jnp.allclose(y, x, atol=1e-6)
        assert jnp.allclose(log_det, 0.0, atol=1e-6)

    def test_per_axis_target(self, key):
        """Per-axis target arrays map each axis to its own range."""
        geom = Geometry(lower=[-2.0, -2.0, -2.0], upper=[2.0, 2.0, 2.0])
        tl = jnp.array([-1.0, -2.0, -3.0])
        tu = jnp.array([1.0, 2.0, 3.0])
        rescale = Rescale(geometry=geom, target=(tl, tu))
        x = jax.random.uniform(key, (8, 3), minval=-2.0, maxval=2.0)
        y, _ = rescale.forward({}, x)
        # Per-axis expected scale: (tu-tl)/box = (2, 4, 6)/4 = (0.5, 1.0, 1.5)
        scale = (tu - tl) / 4.0
        expected_y = tl + (x - jnp.array([-2.0, -2.0, -2.0])) * scale
        assert jnp.allclose(y, expected_y, atol=1e-6)

        x_back, _ = rescale.inverse({}, y)
        assert jnp.allclose(x_back, x, atol=1e-6)

    def test_jit(self, key):
        """forward and inverse compile under jit."""
        geom = Geometry(lower=[-2.0, -2.0, -2.0], upper=[3.0, 3.0, 3.0])
        rescale = Rescale(geometry=geom)
        x = jax.random.uniform(key, (4, 3), minval=-2.0, maxval=3.0)
        fwd = jax.jit(rescale.forward)
        inv = jax.jit(rescale.inverse)
        y, _ = fwd({}, x)
        x_back, _ = inv({}, y)
        assert jnp.allclose(x_back, x, atol=1e-6)

    def test_create_factory(self, key):
        """create returns (transform, empty-params)."""
        geom = Geometry.cubic(d=3, side=5.0, lower=-1.0)
        transform, params = Rescale.create(key, geometry=geom)
        assert params == {}
        x = jax.random.uniform(key, (4, 3), minval=-1.0, maxval=4.0)
        y_direct, _ = Rescale(geometry=geom).forward({}, x)
        y_factory, _ = transform.forward(params, x)
        assert jnp.allclose(y_direct, y_factory, atol=1e-6)

    def test_invalid_target_raises(self):
        """target_upper <= target_lower raises ValueError."""
        geom = Geometry.cubic(d=3)
        with pytest.raises(ValueError, match="target lower must be strictly"):
            Rescale(geometry=geom, target=(1.0, -1.0))
        with pytest.raises(ValueError, match="target lower must be strictly"):
            Rescale(geometry=geom, target=(0.0, 0.0))

    def test_mismatched_target_shape_raises(self):
        """Per-axis target with wrong shape raises ValueError."""
        geom = Geometry.cubic(d=3)
        with pytest.raises(ValueError, match="target bounds must be scalar"):
            Rescale(geometry=geom, target=(jnp.array([-1.0, -1.0]), jnp.array([1.0, 1.0])))

    def test_invalid_event_shape_raises(self):
        """event_shape whose last axis != geometry.d raises."""
        geom = Geometry.cubic(d=3)
        with pytest.raises(ValueError, match="event_shape must end in the coord dim"):
            Rescale(geometry=geom, event_shape=(4, 2))
        with pytest.raises(ValueError, match="event_shape must end in the coord dim"):
            Rescale(geometry=geom, event_shape=())

    def test_invalid_geometry_raises(self):
        """Non-Geometry argument raises TypeError."""
        with pytest.raises(TypeError, match="geometry must be a Geometry instance"):
            Rescale(geometry="cube")  # type: ignore[arg-type]


# ============================================================================
# CoMProjection: (N, d) <-> (N-1, d) translation-gauge projection.
# Convention (1): log-det is zero on the (N-1)d subspace.
# See class docstring WARNING block; the constant (d/2)*log(N) lives on
# `CoMProjection.ambient_correction(N, d)` and is the caller's to apply.
# ============================================================================
class TestCoMProjection:
    """Translation-gauge bijection with zero log-det convention."""

    def test_round_trip_zero_com(self, key):
        """inverse(forward(x)) == x for zero-CoM input (the bijection's domain)."""
        N, d = 6, 3
        proj = CoMProjection()
        x = jax.random.normal(key, (4, N, d))
        x_zero_com = x - jnp.mean(x, axis=-2, keepdims=True)
        y, _ = proj.forward({}, x_zero_com)
        x_back, _ = proj.inverse({}, y)
        assert y.shape == (4, N - 1, d)
        assert x_back.shape == (4, N, d)
        assert jnp.allclose(x_back, x_zero_com, atol=1e-6)

    def test_inverse_forward_on_zero_com(self, key):
        """forward(inverse(y)) == y for any y (the canonical round-trip)."""
        N, d = 5, 3
        proj = CoMProjection()
        y = jax.random.normal(key, (4, N - 1, d))
        x = proj.inverse({}, y)[0]
        assert jnp.allclose(jnp.sum(x, axis=-2), 0.0, atol=1e-6)  # zero-CoM by construction
        y_back, _ = proj.forward({}, x)
        assert jnp.allclose(y_back, y, atol=1e-6)

    def test_forward_centers_nonzero_com_input(self, key):
        """forward on non-zero-CoM input centres it; the original CoM is discarded (lossy)."""
        N, d = 4, 2
        proj = CoMProjection()
        x = jax.random.normal(key, (3, N, d)) + 10.0  # shifted; non-zero CoM
        y, _ = proj.forward({}, x)
        # Reconstruct: the ambient x that inverse(y) gives must be zero-CoM.
        x_rec = proj.inverse({}, y)[0]
        assert jnp.allclose(jnp.sum(x_rec, axis=-2), 0.0, atol=1e-5)
        # The reconstruction matches the centred version of the original.
        x_centered = x - jnp.mean(x, axis=-2, keepdims=True)
        assert jnp.allclose(x_rec, x_centered, atol=1e-5)

    def test_inverse_output_is_zero_com(self, key):
        """Every output of inverse has sum-along-particle-axis == 0."""
        N, d = 7, 3
        proj = CoMProjection()
        y = jax.random.normal(key, (2, N - 1, d)) * 3.0  # arbitrary scale
        x = proj.inverse({}, y)[0]
        assert jnp.allclose(jnp.sum(x, axis=-2), 0.0, atol=1e-5)

    def test_log_det_is_zero(self, key):
        """Convention (1): log-det is exactly zero both directions."""
        N, d = 5, 3
        proj = CoMProjection()
        x = jax.random.normal(key, (4, N, d))
        x_zero_com = x - jnp.mean(x, axis=-2, keepdims=True)
        _, ld_fwd = proj.forward({}, x_zero_com)
        _, ld_inv = proj.inverse({}, proj.forward({}, x_zero_com)[0])
        assert jnp.all(ld_fwd == 0.0)
        assert jnp.all(ld_inv == 0.0)

    def test_ambient_correction_value(self):
        """ambient_correction(N, d) == (d/2) * log(N)."""
        import math as _math
        assert jnp.isclose(
            CoMProjection.ambient_correction(4, 3), 1.5 * _math.log(4)
        )
        assert jnp.isclose(
            CoMProjection.ambient_correction(10, 1), 0.5 * _math.log(10)
        )
        assert jnp.isclose(
            CoMProjection.ambient_correction(2, 3), 1.5 * _math.log(2)
        )

    def test_ambient_correction_validation(self):
        """N >= 2, d >= 1 required."""
        with pytest.raises(ValueError, match="N must be >= 2"):
            CoMProjection.ambient_correction(1, 3)
        with pytest.raises(ValueError, match="d must be >= 1"):
            CoMProjection.ambient_correction(4, 0)

    def test_jit(self, key):
        """forward and inverse compile under jit."""
        N, d = 5, 3
        proj = CoMProjection()
        x = jax.random.normal(key, (4, N, d))
        x_zero_com = x - jnp.mean(x, axis=-2, keepdims=True)
        fwd = jax.jit(proj.forward)
        inv = jax.jit(proj.inverse)
        y, _ = fwd({}, x_zero_com)
        x_back, _ = inv({}, y)
        assert jnp.allclose(x_back, x_zero_com, atol=1e-6)

    def test_custom_N_d(self, key):
        """Works with varied (N, d) combinations."""
        for N, d in [(2, 3), (32, 3), (8, 2), (4, 1)]:
            proj = CoMProjection()
            y = jax.random.normal(jax.random.fold_in(key, N * d), (N - 1, d))
            x = proj.inverse({}, y)[0]
            assert x.shape == (N, d)
            y_back, _ = proj.forward({}, x)
            assert jnp.allclose(y_back, y, atol=1e-6)

    def test_create_factory(self, key):
        """create returns (transform, {})."""
        transform, params = CoMProjection.create(key)
        assert params == {}
        y = jax.random.normal(key, (4, d := 3))  # placeholder; not used
        # The returned transform is functionally equivalent to the direct
        # construction:
        assert transform.event_axis == CoMProjection().event_axis

    def test_invalid_event_axis_raises(self):
        """event_axis must be negative and not -1."""
        with pytest.raises(ValueError, match="event_axis must be negative"):
            CoMProjection(event_axis=0)
        with pytest.raises(ValueError, match="event_axis must be negative"):
            CoMProjection(event_axis=2)
        with pytest.raises(ValueError, match="event_axis=-1 is the coord axis"):
            CoMProjection(event_axis=-1)

    def test_custom_event_axis(self, key):
        """event_axis=-3 works on (B, species, N, d) events."""
        N, d = 6, 3
        proj = CoMProjection(event_axis=-3)
        y = jax.random.normal(key, (4, N - 1, 2, d))  # (B, N-1, species=2, d)
        x = proj.inverse({}, y)[0]
        assert x.shape == (4, N, 2, d)
        assert jnp.allclose(jnp.sum(x, axis=-3), 0.0, atol=1e-5)

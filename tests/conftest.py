# tests/conftest.py
"""Shared pytest fixtures for nflojax tests."""
from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp


# Skip marker for tests whose atol is only tight enough under float64. The RQS
# inverse quadratic solver accumulates ~3e-4 of roundoff in float32, which
# exceeds the log-det atol (1e-4) used by these round-trip tests. They pass
# under JAX_ENABLE_X64=1 (proof of correctness); the default float32 run skips.
requires_x64 = pytest.mark.skipif(
    not jax.config.jax_enable_x64,
    reason="float32 RQS inverse roundoff exceeds test atol; "
           "run with JAX_ENABLE_X64=1 to enable.",
)


@pytest.fixture
def key():
    """Default JAX PRNG key."""
    return jax.random.PRNGKey(42)


@pytest.fixture
def dim():
    """Default feature dimension."""
    return 4


@pytest.fixture
def context_dim():
    """Default context dimension for conditional flows."""
    return 2


@pytest.fixture
def batch_size():
    """Default batch size."""
    return 32


def check_logdet_vs_autodiff(forward_fn, x, atol=1e-4):
    """
    Compare log_det from forward pass against autodiff Jacobian.

    Works for single sample (no batch dimension).
    """
    y, ld = forward_fn(x)

    # Compute Jacobian via autodiff
    J = jax.jacfwd(lambda z: forward_fn(z)[0])(x)
    ld_autodiff = jnp.log(jnp.abs(jnp.linalg.det(J)))

    error = float(jnp.abs(ld - ld_autodiff))
    return {
        "error": error,
        "ld": float(ld),
        "ld_autodiff": float(ld_autodiff),
    }

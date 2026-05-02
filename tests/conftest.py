# tests/conftest.py
"""Shared pytest fixtures for nflojax tests."""
from __future__ import annotations

import pytest
import jax
import jax.numpy as jnp


def requires_x64(test_item):
    """Mark precision-proof tests that should run only under JAX x64.

    The marker is selectable with ``pytest -m requires_x64``. The skip keeps
    the default float32 suite green while still making these tests easy to run
    as a targeted x64 gate.
    """

    marked = pytest.mark.requires_x64(test_item)
    return pytest.mark.skipif(
        not jax.config.jax_enable_x64,
        reason="float32 RQS inverse roundoff exceeds test atol; "
        "run with JAX_ENABLE_X64=1 to enable.",
    )(marked)


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

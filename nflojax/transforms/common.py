# nflojax/transforms/common.py
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

from ..nets import Array


def validate_identity_gate(identity_gate, context_dim: int) -> None:
    """
    Check that an identity_gate function is written for single samples.

    WHY THIS CHECK EXISTS
    ---------------------
    _compute_gate_value applies the gate via jax.vmap, which maps over the
    leading (batch) axis and calls the gate once per sample with shape
    (context_dim,). If a user writes a gate that *expects* batched input
    (batch, context_dim), vmap silently feeds it a 1-D vector instead. The
    gate still runs, but it interprets axis 0 as features rather than
    samples, producing wrong scalar values with no error.

    HOW THE CHECK WORKS
    -------------------
    We use jax.eval_shape to trace the gate on two abstract inputs, without
    doing any actual computation (zero FLOPs):

      1. Single sample:  shape (context_dim,)   -> must produce shape ()
      2. Fake batch:     shape (2, context_dim)  -> tells us how the gate
         responds to a rank-2 input.

    A correct per-sample gate either:
      (a) errors on the rank-2 input (it was written for 1-D only), or
      (b) returns shape (2,), meaning it naturally broadcasts across batch.

    A batch-reducing gate returns shape () for both inputs, which is the
    red flag: it collapses the leading axis, treating it as batch.

    WHEN TO CALL
    ------------
    Called at build time (before jit), so tracing costs are negligible.
    The high-level builders (build_realnvp, build_spline_realnvp) call this
    automatically. If using the lower-level assemble_* API, call this
    explicitly since those functions don't know context_dim.

    Arguments:
        identity_gate: Callable mapping a single context vector -> scalar.
        context_dim:   Dimensionality of the raw context vector.

    Raises:
        ValueError: If the gate's output shape is inconsistent with
                    single-sample usage.
    """
    if identity_gate is None:
        return

    dtype = jnp.float32

    # --- Check 1: single sample must produce scalar ---
    single_shape = jax.ShapeDtypeStruct((context_dim,), dtype)
    try:
        out_single = jax.eval_shape(identity_gate, single_shape)
    except Exception as e:
        raise ValueError(
            f"identity_gate failed on single-sample input of shape "
            f"({context_dim},): {e}"
        ) from e

    if out_single.shape != ():
        raise ValueError(
            f"identity_gate must return a scalar for a single context vector "
            f"of shape ({context_dim},), but returned shape {out_single.shape}."
        )

    # --- Check 2: rank-2 input distinguishes per-sample vs batch-reducing ---
    #
    # We trace the gate on shape (2, context_dim). Three outcomes:
    #
    #   - Exception:  gate only accepts 1-D input. This is correct usage;
    #                 vmap will handle batching.
    #
    #   - Shape (2,): gate naturally maps across the leading axis, returning
    #                 one scalar per row. Compatible with vmap (both give
    #                 the same result).
    #
    #   - Shape ():   gate collapses the leading axis, treating it as the
    #                 batch dimension. Under vmap it receives (context_dim,)
    #                 and still returns (), but the *values* will be wrong
    #                 because it operates on features instead of samples.
    #                 This is the failure mode we want to catch.
    batch_shape = jax.ShapeDtypeStruct((2, context_dim), dtype)
    try:
        out_batch = jax.eval_shape(identity_gate, batch_shape)
    except Exception:
        # Gate errors on 2-D input: it was written for single samples. Good.
        return

    if out_batch.shape == ():
        raise ValueError(
            "identity_gate appears to reduce over the input's leading axis: "
            f"it returns shape () for both input shapes ({context_dim},) and "
            f"(2, {context_dim}). This means it treats axis 0 as a batch "
            "dimension. Under jax.vmap (used internally), the gate receives "
            "one sample at a time with shape (context_dim,), so a "
            "batch-reducing gate silently produces wrong values. "
            "Rewrite the gate to operate on a single vector of shape "
            f"({context_dim},) and return a scalar."
        )

    # Any other shape (e.g. (2,)) is acceptable.


def _compute_gate_value(identity_gate, context):
    """
    Compute gate value from context, handling batching via vmap.

    When identity_gate(context) = 0, the transform should be the identity.
    When identity_gate(context) = 1, the transform acts normally.

    This function is called with RAW context (before feature extraction).
    The gate function must be written for a single sample of shape
    (context_dim,); batched inputs are handled via jax.vmap.

    Arguments:
        identity_gate: Callable that maps a single context vector -> scalar, or None.
        context: Raw context tensor of shape (context_dim,) or (batch, context_dim), or None.

    Returns:
        Gate value array of shape () or (batch,), or None if identity_gate is None.

    Raises:
        ValueError: If identity_gate returns non-scalar output.
    """
    if identity_gate is None or context is None:
        return None

    # Handle single sample vs batch
    if context.ndim == 1:
        g_val = identity_gate(context)
    else:
        g_val = jax.vmap(identity_gate)(context)

    g_val = jnp.asarray(g_val)

    # Validate: should be scalar per sample
    if context.ndim == 1 and g_val.ndim > 0:
        raise ValueError(
            f"identity_gate must return scalar, got shape {g_val.shape}"
        )
    if context.ndim > 1 and g_val.ndim > 1:
        raise ValueError(
            f"identity_gate must return scalar per sample, got shape {g_val.shape}"
        )

    return g_val


def stable_logit(p: Array) -> Array:
    """
    Numerically stable logit function: logit(p) = log(p / (1 - p)).

    Clips input to [1e-6, 1 - 1e-6] to avoid log(0) or log(inf).

    Arguments:
        p: Probability values in (0, 1).

    Returns:
        Logit of p.
    """
    p = jnp.clip(p, 1e-6, 1.0 - 1e-6)
    return jnp.log(p) - jnp.log1p(-p)


_BOUNDARY_SLOPE_MODES = ("linear_tails", "circular")


def _params_per_scalar(num_bins: int, boundary_slopes: str) -> int:
    """Per-scalar spline param count.

    - 'linear_tails': K widths + K heights + (K-1) interior derivs = 3K-1.
    - 'circular':     K widths + K heights + K derivs (K-1 interior + 1
                      shared boundary) = 3K.
    """
    if boundary_slopes == "linear_tails":
        return 3 * num_bins - 1
    if boundary_slopes == "circular":
        return 3 * num_bins
    raise ValueError(
        f"_params_per_scalar: unknown boundary_slopes={boundary_slopes!r}. "
        f"Expected one of {_BOUNDARY_SLOPE_MODES}."
    )


def _validate_boundary_slopes(boundary_slopes: str, *, where: str) -> None:
    """Raise if `boundary_slopes` is not a supported mode.

    `where` identifies the call site for the error message (e.g. 'SplineCoupling').
    """
    if boundary_slopes not in _BOUNDARY_SLOPE_MODES:
        raise ValueError(
            f"{where}: boundary_slopes must be one of {_BOUNDARY_SLOPE_MODES}, "
            f"got {boundary_slopes!r}."
        )


def identity_spline_bias(
    num_scalars: int,
    num_bins: int,
    min_derivative: float,
    max_derivative: float,
    dtype=jnp.float32,
    boundary_slopes: str = "linear_tails",
) -> Array:
    """
    Conditioner-output bias that yields an identity spline per scalar.

    With a zero-kernel final layer in the conditioner MLP, this bias makes
    the spline parameters evaluate to:
      - widths=0 (uniform bins after softmax)
      - heights=0 (uniform bins)
      - all derivatives=1 (via stable_logit of (1 - min_d) / (max_d - min_d))
    giving an identity spline on [-tail_bound, tail_bound].

    Per-scalar param count:
      - 'linear_tails': 3K - 1 (K widths + K heights + K-1 interior derivs).
      - 'circular':     3K     (K widths + K heights + K-1 interior + 1
                                shared boundary derivative).

    If 1.0 is not strictly inside (min_derivative, max_derivative) the
    derivative bias cannot reach 1 and we emit zeros; caller should warn.

    Returns:
        bias vector of shape `(num_scalars * params_per_scalar,)`.
    """
    K = num_bins
    params_per_scalar = _params_per_scalar(K, boundary_slopes)
    bias = jnp.zeros((num_scalars, params_per_scalar), dtype=dtype)
    lo, hi = float(min_derivative), float(max_derivative)
    if lo < 1.0 < hi:
        alpha = (1.0 - lo) / (hi - lo)
        u0 = stable_logit(jnp.asarray(alpha, dtype=bias.dtype))
        # All derivative slots (interior + any boundary entry) get the same
        # logit, so they all evaluate to 1 after the sigmoid bound. The
        # widths/heights stay zero.
        bias = bias.at[:, 2 * K :].set(u0)
    return bias.reshape((-1,))

# nflojax/transforms.py
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Callable, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp
import numpy as np
from flax import linen as nn

from .nets import MLP, Array, PRNGKey, validate_conditioner
from . import scalar_function
from .splines import rational_quadratic_spline
from .geometry import Geometry


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


# ===================================================================
# Linear Transform with LU-style parameterization
# ===================================================================
@dataclass
class LinearTransform:
    """
    Global linear (or affine, when conditional) transform with LU-style
    parameterization.

    We parameterize an invertible matrix W in R^{dim x dim} as:

      L = tril(lower_raw, k = -1) + I        (unit-diagonal lower)
      U = triu(upper_raw, k = 1)             (zero diagonal upper)
      s = softplus(raw_diag + delta)         (positive diagonal entries)
      T = U + diag(s)                        (upper-triangular)
      W = L @ T

    Where delta is either 0 (unconditional) or produced by a conditioner
    network that takes context as input (conditional).

    Unconditional forward map:  y = x W^T
    Conditional forward map:    y = x W^T + shift(context)

    The shift is produced by the same conditioner MLP and does not affect the
    log-determinant (Jacobian is still W).

    log |det W| = sum(log(s)) = sum(log(softplus(raw_diag + delta))).

    Inverse uses triangular solves (after subtracting shift if conditional):

      L a = (y - shift)^T    (forward substitution, unit diagonal)
      T u = a                (back substitution)

    Parameters (per block):
      params["lower"]:    unconstrained raw lower-tri part, shape (dim, dim)
      params["upper"]:    unconstrained raw upper-tri part, shape (dim, dim)
      params["raw_diag"]: unconstrained diagonal params, shape (dim,)
      params["mlp"]:      conditioner params (only if context_dim > 0)

    Conditional flows:
      When context_dim > 0, a conditioner MLP maps context to (delta_diag, shift).
      delta_diag is added to raw_diag before softplus; shift is added after the
      linear map. The MLP is zero-initialized so the transform starts at identity.

    Identity gating:
      When g_value is provided, shift is scaled by g: shift_eff = g * shift.
      At g=0 the full transform (including shift) collapses to identity.

    This yields O(dim^2) apply / inverse and O(dim) log-det, without any
    repeated matrix factorizations inside the forward pass.
    """
    dim: int
    conditioner: MLP | None = None  # None if context_dim=0
    context_dim: int = 0

    def _get_raw_params(self, params: Any) -> Tuple[Array, Array, Array]:
        """Extract and validate raw parameters from params dict."""
        try:
            lower_raw = jnp.asarray(params["lower"])
            upper_raw = jnp.asarray(params["upper"])
            raw_diag = jnp.asarray(params["raw_diag"])
        except Exception as e:
            raise KeyError(
                "LinearTransform: params must contain 'lower', 'upper', 'raw_diag'"
            ) from e

        if lower_raw.shape != (self.dim, self.dim):
            raise ValueError(
                f"LinearTransform: lower must have shape ({self.dim}, {self.dim}), "
                f"got {lower_raw.shape}"
            )
        if upper_raw.shape != (self.dim, self.dim):
            raise ValueError(
                f"LinearTransform: upper must have shape ({self.dim}, {self.dim}), "
                f"got {upper_raw.shape}"
            )
        if raw_diag.shape != (self.dim,):
            raise ValueError(
                f"LinearTransform: raw_diag must have shape ({self.dim},), "
                f"got {raw_diag.shape}"
            )

        return lower_raw, upper_raw, raw_diag

    def _compute_conditioner_outputs(
        self,
        params: Any,
        raw_diag: Array,
        context: Array | None,
    ) -> Tuple[Array, Array | None]:
        """
        Compute diagonal scaling s and optional shift from raw_diag and context.

        If conditioner exists and context is provided, the MLP outputs 2*dim
        values split into (delta_diag, shift). delta_diag is added to raw_diag
        before softplus; shift is returned separately.

        Returns:
            s: positive diagonal scaling, shape (dim,) or (batch, dim).
            shift: context-dependent shift, shape (batch, dim) or None.
        """
        if self.conditioner is not None and context is not None:
            mlp_params = params["mlp"]
            out = self.conditioner.apply({"params": mlp_params}, context, None)
            # out shape: (batch, 2*dim) or (2*dim,)
            delta_diag, shift = jnp.split(out, 2, axis=-1)
            s = jax.nn.softplus(raw_diag + delta_diag)
            return s, shift
        else:
            s = jax.nn.softplus(raw_diag)
            return s, None

    def _forward_batched_gate(
        self,
        x: Array,
        lower_raw: Array,
        upper_raw: Array,
        s: Array,
        shift: Array | None,
        g_value: Array,
        batch_shape: tuple,
    ) -> Tuple[Array, Array]:
        """Forward pass with per-sample gating via vmap.

        When g_value is batched, each sample needs its own L, U matrices
        constructed with its gate value. This is slower than the shared-matrix
        path but correctly handles per-sample identity interpolation.
        """
        # Gate the diagonal: s_gated = 1 + g * (s - 1)
        g_diag = g_value[:, None]  # (B, 1)
        s_gated = 1.0 - g_diag + g_diag * s  # broadcasts for both (dim,) and (B, dim)

        # Gate the shift: shift_gated = g * shift
        if shift is not None:
            shift_gated = g_diag * shift
        else:
            shift_gated = None

        dim = self.dim
        dtype = lower_raw.dtype

        def forward_single(x_i, g_i, s_i, shift_i):
            # Build gated LU factors: when g=0, L=I and U=0, so W=I (identity).
            # When g=1, we get the full learned transform.
            L_i = jnp.tril(g_i * lower_raw, k=-1) + jnp.eye(dim, dtype=dtype)
            U_i = jnp.triu(g_i * upper_raw, k=1)
            T_i = U_i + jnp.diag(s_i)
            y_i = L_i @ T_i @ x_i + shift_i
            log_det_i = jnp.sum(jnp.log(s_i))
            return y_i, log_det_i

        # Flatten batch dims for vmap, then reshape back.
        x_flat = x.reshape((-1, dim))
        g_flat = g_value.reshape((-1,))
        s_flat = s_gated.reshape((-1, dim))
        if shift_gated is not None:
            shift_flat = shift_gated.reshape((-1, dim))
        else:
            shift_flat = jnp.zeros_like(x_flat)
        y_flat, log_det_flat = jax.vmap(forward_single)(x_flat, g_flat, s_flat, shift_flat)
        y = y_flat.reshape(batch_shape + (dim,))
        log_det_forward = log_det_flat.reshape(batch_shape)
        return y, log_det_forward

    def _inverse_batched_gate(
        self,
        y: Array,
        lower_raw: Array,
        upper_raw: Array,
        s: Array,
        shift: Array | None,
        g_value: Array,
        batch_shape: tuple,
    ) -> Tuple[Array, Array]:
        """Inverse pass with per-sample gating via vmap.

        When g_value is batched, each sample needs its own L, U matrices
        constructed with its gate value. This is slower than the shared-matrix
        path but correctly handles per-sample identity interpolation.
        """
        # Gate the diagonal: s_gated = 1 + g * (s - 1)
        g_diag = g_value[:, None]  # (B, 1)
        s_gated = 1.0 - g_diag + g_diag * s  # broadcasts for both (dim,) and (B, dim)

        # Gate the shift: shift_gated = g * shift
        if shift is not None:
            shift_gated = g_diag * shift
        else:
            shift_gated = None

        dim = self.dim
        dtype = lower_raw.dtype

        def inverse_single(y_i, g_i, s_i, shift_i):
            # Build gated LU factors (same as forward).
            L_i = jnp.tril(g_i * lower_raw, k=-1) + jnp.eye(dim, dtype=dtype)
            U_i = jnp.triu(g_i * upper_raw, k=1)
            T_i = U_i + jnp.diag(s_i)
            # Subtract shift, then solve L @ T @ x = (y - shift).
            z_i = y_i - shift_i
            a_i = jsp.solve_triangular(L_i, z_i, lower=True, unit_diagonal=True)
            x_i = jsp.solve_triangular(T_i, a_i, lower=False)
            log_det_i = -jnp.sum(jnp.log(s_i))
            return x_i, log_det_i

        # Flatten batch dims for vmap, then reshape back.
        y_flat = y.reshape((-1, dim))
        g_flat = g_value.reshape((-1,))
        s_flat = s_gated.reshape((-1, dim))
        if shift_gated is not None:
            shift_flat = shift_gated.reshape((-1, dim))
        else:
            shift_flat = jnp.zeros_like(y_flat)
        x_flat, log_det_flat = jax.vmap(inverse_single)(y_flat, g_flat, s_flat, shift_flat)
        x = x_flat.reshape(batch_shape + (dim,))
        log_det_inverse = log_det_flat.reshape(batch_shape)
        return x, log_det_inverse

    def forward(
        self,
        params: Any,
        x: Array,
        context: Array | None = None,
        g_value: Array | None = None,
    ) -> Tuple[Array, Array]:
        """
        Forward map: x -> y, returning (y, log_det_forward).

        Unconditional: y = x W^T.
        Conditional:   y = x W^T + shift(context).

        Arguments:
          params: PyTree with leaves 'lower', 'upper', 'raw_diag', and optionally 'mlp'.
          x: input tensor of shape (..., dim).
          context: optional conditioning tensor, shape (..., context_dim).
          g_value: optional gate value for identity_gate. When g_value=0, returns identity.

        Returns:
          y: transformed tensor of shape (..., dim).
          log_det: log |det ∂y/∂x| = sum(log(s)), shape x.shape[:-1].
                   Shift does not affect log-det.
        """
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"LinearTransform: expected input last dim {self.dim}, "
                f"got {x.shape[-1]}"
            )

        # Get raw parameters with validation
        lower_raw, upper_raw, raw_diag = self._get_raw_params(params)
        batch_shape = x.shape[:-1]

        # Compute diagonal scaling and optional shift
        s, shift = self._compute_conditioner_outputs(params, raw_diag, context)

        # Batched gate requires per-sample L, U - use dedicated vmap path
        if g_value is not None and g_value.ndim > 0:
            return self._forward_batched_gate(
                x, lower_raw, upper_raw, s, shift, g_value, batch_shape
            )

        # Gate shift (scalar gate or no gate)
        if shift is not None and g_value is not None:
            shift = g_value * shift

        # Fast path: shared L, U (possibly scaled by scalar gate)
        if g_value is not None:
            # Scalar gate - scale L, U and interpolate s
            lower_raw = g_value * lower_raw
            upper_raw = g_value * upper_raw
            s = 1.0 - g_value + g_value * s

        # Reconstruct L, U
        L = jnp.tril(lower_raw, k=-1) + jnp.eye(self.dim, dtype=lower_raw.dtype)
        U = jnp.triu(upper_raw, k=1)

        # Handle batched s (when context is batched)
        if s.ndim == 1:
            # s is (dim,) - shared across batch
            T = U + jnp.diag(s)
            x_flat = x.reshape((-1, self.dim))  # (B, dim)
            u = x_flat.T                        # (dim, B)
            a = T @ u
            u_prime = L @ a
            y_flat = u_prime.T                  # (B, dim)
            y = y_flat.reshape(batch_shape + (self.dim,))
            # log |det W| = sum(log(s))
            log_det_scalar = jnp.sum(jnp.log(s))
            log_det_forward = jnp.broadcast_to(log_det_scalar, batch_shape)
        else:
            # s is (batch, dim) - different per sample, use vmap
            def forward_single(x_i, s_i):
                T_i = U + jnp.diag(s_i)
                a_i = T_i @ x_i
                y_i = L @ a_i
                log_det_i = jnp.sum(jnp.log(s_i))
                return y_i, log_det_i

            x_flat = x.reshape((-1, self.dim))
            s_flat = s.reshape((-1, self.dim))
            y_flat, log_det_flat = jax.vmap(forward_single)(x_flat, s_flat)
            y = y_flat.reshape(batch_shape + (self.dim,))
            log_det_forward = log_det_flat.reshape(batch_shape)

        # Add shift (does not affect log-det)
        if shift is not None:
            y = y + shift

        return y, log_det_forward

    def inverse(
        self,
        params: Any,
        y: Array,
        context: Array | None = None,
        g_value: Array | None = None,
    ) -> Tuple[Array, Array]:
        """
        Inverse map: y -> x, returning (x, log_det_inverse).

        Unconditional: x = W^{-T} y.
        Conditional:   x = W^{-T} (y - shift(context)).

        Arguments:
          params: PyTree with leaves 'lower', 'upper', 'raw_diag', and optionally 'mlp'.
          y: input tensor of shape (..., dim).
          context: optional conditioning tensor, shape (..., context_dim).
          g_value: optional gate value for identity_gate. When g_value=0, returns identity.

        Returns:
          x: inverse-transformed tensor of shape (..., dim).
          log_det: log |det ∂x/∂y| = -sum(log(s)), shape y.shape[:-1].
                   Shift does not affect log-det.
        """
        if y.shape[-1] != self.dim:
            raise ValueError(
                f"LinearTransform: expected input last dim {self.dim}, "
                f"got {y.shape[-1]}"
            )

        # Get raw parameters with validation
        lower_raw, upper_raw, raw_diag = self._get_raw_params(params)
        batch_shape = y.shape[:-1]

        # Compute diagonal scaling and optional shift
        s, shift = self._compute_conditioner_outputs(params, raw_diag, context)

        # Batched gate requires per-sample L, U - use dedicated vmap path
        if g_value is not None and g_value.ndim > 0:
            return self._inverse_batched_gate(
                y, lower_raw, upper_raw, s, shift, g_value, batch_shape
            )

        # Gate shift (scalar gate or no gate)
        if shift is not None and g_value is not None:
            shift = g_value * shift

        # Subtract shift before linear inverse (does not affect log-det)
        if shift is not None:
            y = y - shift

        # Fast path: shared L, U (possibly scaled by scalar gate)
        if g_value is not None:
            # Scalar gate - scale L, U and interpolate s
            lower_raw = g_value * lower_raw
            upper_raw = g_value * upper_raw
            s = 1.0 - g_value + g_value * s

        # Reconstruct L, U
        L = jnp.tril(lower_raw, k=-1) + jnp.eye(self.dim, dtype=lower_raw.dtype)
        U = jnp.triu(upper_raw, k=1)

        # Handle batched s (when context is batched)
        if s.ndim == 1:
            # s is (dim,) - shared across batch
            T = U + jnp.diag(s)
            y_flat = y.reshape((-1, self.dim))  # (B, dim)
            u_prime = y_flat.T                  # (dim, B)

            # Column-style inverse:
            # 1) L a = u'   -> a
            # 2) T u = a    -> u
            a = jsp.solve_triangular(L, u_prime, lower=True, unit_diagonal=True)
            u = jsp.solve_triangular(T, a, lower=False)

            x_flat = u.T
            x = x_flat.reshape(batch_shape + (self.dim,))
            log_det_scalar = jnp.sum(jnp.log(s))
            log_det_inverse = jnp.broadcast_to(-log_det_scalar, batch_shape)
        else:
            # s is (batch, dim) - different per sample, use vmap
            def inverse_single(y_i, s_i):
                T_i = U + jnp.diag(s_i)
                a_i = jsp.solve_triangular(L, y_i, lower=True, unit_diagonal=True)
                x_i = jsp.solve_triangular(T_i, a_i, lower=False)
                log_det_i = -jnp.sum(jnp.log(s_i))
                return x_i, log_det_i

            y_flat = y.reshape((-1, self.dim))
            s_flat = s.reshape((-1, self.dim))
            x_flat, log_det_flat = jax.vmap(inverse_single)(y_flat, s_flat)
            x = x_flat.reshape(batch_shape + (self.dim,))
            log_det_inverse = log_det_flat.reshape(batch_shape)

        return x, log_det_inverse

    def init_params(self, key: PRNGKey, context_dim: int = 0) -> dict:
        """
        Initialize parameters for this transform.

        Returns identity transform params (L=I, U=0, s=1 => W=I, shift=0).
        For softplus parametrization, raw_diag is initialized so softplus(raw_diag) = 1.
        Conditioner MLP output layer is zero-initialized so both delta_diag and shift start at 0.

        Arguments:
            key: JAX PRNGKey for conditioner initialization.
            context_dim: Context dimension (must match self.context_dim).

        Returns:
            Dict with keys 'lower', 'upper', 'raw_diag', and 'mlp' if context_dim > 0.
        """
        # softplus(x) = 1 when x = log(e - 1) ≈ 0.541
        raw_diag_init = jnp.full((self.dim,), jnp.log(jnp.e - 1), dtype=jnp.float32)

        params = {
            "lower": jnp.zeros((self.dim, self.dim), dtype=jnp.float32),
            "upper": jnp.zeros((self.dim, self.dim), dtype=jnp.float32),
            "raw_diag": raw_diag_init,
        }

        # Initialize conditioner if present
        if self.conditioner is not None:
            dummy_context = jnp.zeros((1, self.context_dim), dtype=jnp.float32)
            variables = self.conditioner.init(key, dummy_context, None)
            mlp_params = variables["params"]

            # Zero-init output layer so delta=0 at init => identity transform
            if hasattr(self.conditioner, "get_output_layer") and hasattr(self.conditioner, "set_output_layer"):
                out_layer = self.conditioner.get_output_layer(mlp_params)
                kernel = jnp.zeros_like(out_layer["kernel"])
                bias = jnp.zeros_like(out_layer["bias"])
                mlp_params = self.conditioner.set_output_layer(mlp_params, kernel, bias)

            params["mlp"] = mlp_params

        return params

    @classmethod
    def create(
        cls,
        key: PRNGKey,
        dim: int,
        *,
        context_dim: int = 0,
        hidden_dim: int = 64,
        n_hidden_layers: int = 2,
        activation: Callable[[Array], Array] = nn.tanh,
        res_scale: float = 0.1,
    ) -> Tuple["LinearTransform", dict]:
        """
        Factory method to create LinearTransform and initialize params.

        Arguments:
            key: JAX PRNGKey for parameter initialization.
            dim: Dimensionality of the transform.
            context_dim: Context dimension (0 for unconditional).
            hidden_dim: Width of hidden layers in conditioner MLP (if context_dim > 0).
            n_hidden_layers: Number of residual blocks in conditioner MLP.
            activation: Activation function for conditioner MLP.
            res_scale: Residual connection scale for conditioner MLP.

        Returns:
            Tuple of (transform, params) ready to use.

        Raises:
            ValueError: If dim <= 0 or context_dim < 0.

        Example:
            >>> # Unconditional
            >>> transform, params = LinearTransform.create(key, dim=4)
            >>> y, log_det = transform.forward(params, x)

            >>> # Conditional on context
            >>> transform, params = LinearTransform.create(
            ...     key, dim=4, context_dim=8, hidden_dim=64, n_hidden_layers=2
            ... )
            >>> y, log_det = transform.forward(params, x, context)
        """
        if dim <= 0:
            raise ValueError(f"LinearTransform.create: dim must be positive, got {dim}.")
        if context_dim < 0:
            raise ValueError(f"LinearTransform.create: context_dim must be non-negative, got {context_dim}.")

        # Create conditioner if context_dim > 0
        conditioner = None
        if context_dim > 0:
            if hidden_dim <= 0:
                raise ValueError(f"LinearTransform.create: hidden_dim must be positive, got {hidden_dim}.")
            # MLP with x_dim=context_dim, context_dim=0: context goes in x slot
            # Output 2*dim: first dim entries are delta_diag, last dim are shift
            conditioner = MLP(
                x_dim=context_dim,
                context_dim=0,
                hidden_dim=hidden_dim,
                n_hidden_layers=n_hidden_layers,
                out_dim=2 * dim,
                activation=activation,
                res_scale=res_scale,
            )

        transform = cls(dim=dim, conditioner=conditioner, context_dim=context_dim)
        params = transform.init_params(key, context_dim=context_dim)
        return transform, params


# ===================================================================
# Affine Coupling Layer
# ===================================================================
@dataclass
class AffineCoupling:
    """
    RealNVP-style affine coupling layer.

    This layer splits the input vector x into two parts using a binary mask m.
    Masked dimensions (where m = 1) pass through unchanged. Unmasked dimensions
    (where m = 0) are transformed using parameters produced by a conditioner
    network.
    
    It roughly works as follows:
    * Split x into x1 = x * m and x2 = x * (1 - m)
    * Use x1 as input to a conditioner network to produce shift and log_scale for transforming x2.
    * Apply the elementwise affine transform on x2: y2 = x2 * exp(log_scale) + shift.
      Note that the shift and log_scale are zeroed out on the masked dimensions.
    * Combine y1 = x1 and y2 to produce output y = y1 + y2. This only modifies the unmasked dimensions.

    Forward transformation y = T(x):
    x1 = x * m
    x2 = x * (1 - m)
    (shift, log_scale) = conditioner(x1) * (1 - m)
    y1 = x1
    y2 = (x2 * exp(log_scale) + shift) 
    y = y1 + y2
    The returned log_det is log |det ∂y/∂x|, equal to the sum of log_scale on the
    unmasked coordinates.

    Inverse transformation x = T^{-1}(y):
    y1 = y * m
    y2 = y * (1 - m)
    (shift, log_scale) = conditioner(y1) * (1 - m)
    x2 = (y2 - shift) * exp(-log_scale)
    x = y1 + x2
    The returned log_det is log |det ∂x/∂y| = -sum(log_scale).

    Parameters:
    params["mlp"]: PyTree containing the Flax parameters of the conditioner.

    All operations act along the last dimension. The mask must be one-dimensional
    with the same length as the feature dimension.
    
    Note:
    In this implementation, the conditioner network is typically initialized
    such that its output is identically zero at initialization. In that case,
    shift = 0 and log_scale = 0, so this layer is exactly the identity map
    at the start of training.
    
    Conditional flows:
      The optional `context` argument enables conditional density estimation p(x|c).
      When provided, context is concatenated to the masked input before being passed
      to the conditioner network. The conditioner MLP must be initialized with
      `context_dim` matching the size of the context vector.

      Context shape: (batch, context_dim) or (context_dim,) for a single sample.
      The same context is used for all coupling layers in a flow.

    References:
      - Dinh, Krueger, Bengio (2017). "NICE: Non-linear Independent Components Estimation"
      - Dinh, Sohl-Dickstein, Bengio (2017). "Density estimation using Real NVP"
    """
    mask: Array          # shape (dim,), values 0 or 1
    conditioner: MLP     # Flax MLP module (definition, no params inside)
    max_log_scale: float = 5.0
    max_shift: float | None = None  # Default: exp(max_log_scale)

    def __post_init__(self):
        # Ensure mask is a 1D array.
        self.mask = jnp.asarray(self.mask)
        if self.mask.ndim != 1:
            raise ValueError(
                f"AffineCoupling mask must be 1D, got shape {self.mask.shape}."
            )
        # Validate conditioner interface.
        validate_conditioner(self.conditioner, name="AffineCoupling.conditioner")

    @property
    def dim(self) -> int:
        return int(self.mask.shape[0])

    @staticmethod
    def required_out_dim(dim: int) -> int:
        """
        Return required conditioner output dimension for AffineCoupling.

        The conditioner must output shift and log_scale for each dimension,
        so out_dim = 2 * dim.

        Arguments:
            dim: Input/output dimensionality.

        Returns:
            Required output dimension for conditioner (2 * dim).
        """
        return 2 * dim

    @classmethod
    def create(
        cls,
        key: PRNGKey,
        dim: int,
        mask: Array,
        hidden_dim: int,
        n_hidden_layers: int,
        *,
        context_dim: int = 0,
        activation: Callable[[Array], Array] = nn.elu,
        res_scale: float = 0.1,
        max_log_scale: float = 5.0,
        max_shift: float | None = None,
    ) -> Tuple["AffineCoupling", dict]:
        """
        Factory method to create AffineCoupling with properly configured MLP.

        This handles the output dimension calculation internally and initializes
        parameters, returning both the coupling and its params ready to use.

        Arguments:
            key: JAX PRNGKey for parameter initialization.
            dim: Input/output dimensionality.
            mask: Binary mask of shape (dim,). 1 = frozen, 0 = transformed.
            hidden_dim: Width of hidden layers in conditioner MLP.
            n_hidden_layers: Number of residual blocks in conditioner MLP.
            context_dim: Context dimension (0 for unconditional).
            activation: Activation function for MLP (default: elu).
            res_scale: Residual connection scale (default: 0.1).
            max_log_scale: Bound on |log_scale| via tanh (default: 5.0).
            max_shift: Bound on |shift| via tanh (default: exp(max_log_scale)).

        Returns:
            Tuple of (coupling, params) ready to use.

        Raises:
            ValueError: If mask length doesn't match dim, or dim <= 0.

        Example:
            >>> coupling, params = AffineCoupling.create(
            ...     key, dim=4, mask=jnp.array([1, 0, 1, 0]),
            ...     hidden_dim=64, n_hidden_layers=2
            ... )
            >>> y, log_det = coupling.forward(params, x)
        """
        # Validate inputs
        if dim <= 0:
            raise ValueError(f"AffineCoupling.create: dim must be positive, got {dim}.")
        if hidden_dim <= 0:
            raise ValueError(f"AffineCoupling.create: hidden_dim must be positive, got {hidden_dim}.")

        mask = jnp.asarray(mask)
        if mask.shape != (dim,):
            raise ValueError(
                f"AffineCoupling.create: mask shape {mask.shape} doesn't match (dim,) = ({dim},)."
            )

        # Create MLP with correct output dimension
        out_dim = cls.required_out_dim(dim)
        mlp = MLP(
            x_dim=dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            out_dim=out_dim,
            activation=activation,
            res_scale=res_scale,
        )

        # Create coupling
        coupling = cls(
            mask=mask,
            conditioner=mlp,
            max_log_scale=max_log_scale,
            max_shift=max_shift,
        )

        # Initialize params
        params = coupling.init_params(key, context_dim=context_dim)

        return coupling, params

    def _condition(
        self,
        params: dict,
        x: Array,
        context: Array | None = None,
        g_value: Array | None = None,
    ) -> Tuple[Array, Array]:
        """
        Run the conditioner network and produce shift and log_scale.

        params:
          dict with key "mlp" containing the conditioner parameters.
        x:
          input tensor of shape (..., dim).
        context:
          optional conditioning tensor of shape (..., context_dim) or (context_dim,).
        g_value:
          optional gate value from identity_gate(context). When g_value=0, the
          transform should be identity, so shift and log_scale are zeroed.
        """
        if "mlp" not in params:
            raise KeyError(
                "AffineCoupling expected params to contain key 'mlp'."
            )

        if x.shape[-1] != self.dim:
            raise ValueError(
                f"AffineCoupling expected input with last dimension {self.dim}, "
                f"got {x.shape[-1]}."
            )

        # Use only the masked part as input to the conditioner.
        # Broadcasting: mask has shape (dim,), x has shape (..., dim).
        x_masked = x * self.mask

        # Apply the MLP. We expect output of size 2 * dim
        # which we split into shift and log_scale_raw.
        mlp_params = params["mlp"]
        out = self.conditioner.apply({"params": mlp_params}, x_masked, context)

        if out.shape[-1] != 2 * self.dim:
            raise ValueError(
                f"Conditioner output last dimension should be 2 * dim = {2 * self.dim}, "
                f"got {out.shape[-1]}."
            )

        shift, log_scale_raw = jnp.split(out, 2, axis=-1)

        # Only transform the unmasked part: zero out contributions on masked dims.
        # (1 - mask) has 1 for transformed dims, 0 otherwise.
        m_unmasked = 1.0 - self.mask

        # Bound both shift and log_scale to avoid numerical explosions.
        # Default max_shift = exp(max_log_scale) matches the maximum scale factor.
        max_shift = self.max_shift if self.max_shift is not None else jnp.exp(self.max_log_scale)
        shift = jnp.tanh(shift / max_shift) * max_shift * m_unmasked
        log_scale = jnp.tanh(log_scale_raw / self.max_log_scale) * self.max_log_scale * m_unmasked

        # Apply identity gate: when g_value=0, shift=0 and log_scale=0 => identity.
        if g_value is not None:
            g = g_value[..., None]  # broadcast to (..., 1) for element-wise multiply
            shift = g * shift
            log_scale = g * log_scale

        return shift, log_scale

    def forward(
        self,
        params: dict,
        x: Array,
        context: Array | None = None,
        g_value: Array | None = None,
    ) -> Tuple[Array, Array]:
        """
        Forward transform: x -> y, returning (y, log_det).

        params:
          dict with key "mlp" for conditioner parameters.
        x:
          input tensor of shape (..., dim).
        context:
          optional conditioning tensor passed to the conditioner.
        g_value:
          optional gate value for identity_gate. When g_value=0, returns identity.

        Returns:
          y: transformed tensor of shape (..., dim).
          log_det: log |det J| with shape x.shape[:-1].
        """
        shift, log_scale = self._condition(params, x, context, g_value=g_value)

        x1 = x * self.mask
        x2 = x * (1.0 - self.mask)

        y2 = x2 * jnp.exp(log_scale) + shift
        y = x1 + y2

        # Sum log_scale over transformed dimensions.
        log_det = jnp.sum(log_scale, axis=-1)
        return y, log_det

    def inverse(
        self,
        params: dict,
        y: Array,
        context: Array | None = None,
        g_value: Array | None = None,
    ) -> Tuple[Array, Array]:
        """
        Inverse transform: y -> x, returning (x, log_det).

        params:
          dict with key "mlp" for conditioner parameters.
        y:
          input tensor of shape (..., dim).
        context:
          optional conditioning tensor passed to the conditioner.
        g_value:
          optional gate value for identity_gate. When g_value=0, returns identity.

        Returns:
          x: inverse-transformed tensor of shape (..., dim).
          log_det: log |det d x / d y| with shape y.shape[:-1].
        """
        shift, log_scale = self._condition(params, y, context, g_value=g_value)

        y1 = y * self.mask
        y2 = y * (1.0 - self.mask)

        x2 = (y2 - shift) * jnp.exp(-log_scale)
        x = y1 + x2

        # Inverse log-det is negative of forward log-det.
        log_det = -jnp.sum(log_scale, axis=-1)
        return x, log_det

    def init_params(self, key: PRNGKey, context_dim: int = 0) -> dict:
        """
        Initialize parameters for this transform.

        Uses Flax init to create MLP parameters. With zero-initialized final layer,
        the transform starts at identity.

        Arguments:
            key: JAX PRNGKey for parameter initialization.
            context_dim: Context dimension (0 for unconditional).

        Returns:
            Dict with key 'mlp' containing MLP parameters.
        """
        dummy_x = jnp.zeros((1, self.dim), dtype=jnp.float32)
        dummy_context = jnp.zeros((1, context_dim), dtype=jnp.float32) if context_dim > 0 else None
        variables = self.conditioner.init(key, dummy_x, dummy_context)
        mlp_params = variables["params"]

        # Zero-init final layer for identity-start (if conditioner supports it).
        if hasattr(self.conditioner, "get_output_layer") and hasattr(self.conditioner, "set_output_layer"):
            out_layer = self.conditioner.get_output_layer(mlp_params)
            kernel = jnp.zeros_like(out_layer["kernel"])
            bias = jnp.zeros_like(out_layer["bias"])
            mlp_params = self.conditioner.set_output_layer(mlp_params, kernel, bias)

        return {"mlp": mlp_params}


# ===================================================================
# Spline Coupling Layer
# ===================================================================
@dataclass
class SplineCoupling:
    """
    RealNVP-style coupling layer with monotonic rational-quadratic splines
    (Durkan et al., 2019).
    
    It roughly works as follows:
    * Split input x into two parts using a binary mask.
    * Use the masked part as input to a conditioner network to produce spline parameters for the unmasked part.
    * Apply elementwise monotonic RQ splines on the unmasked part. The masked part remains unchanged.
    
    The spline: rational_quadratic_spline in splines.py implements the actual spline logic.
    It roughly works as follows:
    * Given K bins, the spline is defined by K widths, K heights, and K-1 internal derivatives.
    * The spline is monotonic and C^1 continuous.
    * Outside the interval [-tail_bound, tail_bound], the spline is the identity map.

    Mask semantics:
      - mask[i] == 1: dimension i is *conditioned on* (left unchanged)
      - mask[i] == 0: dimension i is *transformed* by a spline

    Conditioner:
      - A Flax module (e.g. MLP) that maps x_cond = x * mask to spline parameters.
      - Params are provided via params["mlp"].

    Parameterization per dimension (K bins):
      - widths:      K
      - heights:     K
      - derivatives: K-1  (internal knot derivatives; boundary derivatives are fixed to 1)

      => params_per_dim = 3K - 1
      => conditioner output dimension = dim * (3K - 1)

    Forward / inverse:
      - Applies elementwise monotonic RQ spline on the transformed dimensions.
      - Identity tails outside [-tail_bound, tail_bound] are handled inside splines.py.

    Conditional flows:
      The optional `context` argument enables conditional density estimation p(x|c).
      When provided, context is concatenated to the masked input before being passed
      to the conditioner network. The conditioner MLP must be initialized with
      `context_dim` matching the size of the context vector.

      Context shape: (batch, context_dim) or (context_dim,) for a single sample.
      The same context is used for all coupling layers in a flow.

    Returns:
      - output with shape (..., dim)
      - log_det with shape (...,) corresponding to forward or inverse Jacobian.
    """
    mask: Array                 # shape (dim,), values in {0, 1}
    conditioner: Any            # Flax module, called via conditioner.apply
    num_bins: int = 8
    tail_bound: float = 5.0
    min_bin_width: float = 1e-2
    min_bin_height: float = 1e-2
    min_derivative: float = 1e-2
    max_derivative: float = 10.0
    boundary_slopes: str = "linear_tails"

    def __post_init__(self):
        self.mask = jnp.asarray(self.mask, dtype=jnp.float32)
        if self.mask.ndim != 1:
            raise ValueError(
                f"SplineCoupling: mask must be 1D, got shape {self.mask.shape}."
            )
        _validate_boundary_slopes(self.boundary_slopes, where="SplineCoupling")
        # Validate conditioner interface.
        validate_conditioner(self.conditioner, name="SplineCoupling.conditioner")

        # Warn if identity-like initialization is not possible.
        lo, hi = float(self.min_derivative), float(self.max_derivative)
        if not (lo < 1.0 < hi):
            warnings.warn(
                f"SplineCoupling: derivative range [{lo}, {hi}] excludes 1.0; "
                "identity-like initialization not possible, using midpoint derivative.",
                stacklevel=2
            )

        # Precompute the identity derivative logit for gating.
        # When gated to zero, derivatives should interpolate to this value
        # so that the spline derivative equals 1 (identity behavior).
        if lo < 1.0 < hi:
            alpha = (1.0 - lo) / (hi - lo)
            self._identity_deriv_logit = stable_logit(jnp.array(alpha))
        else:
            # If 1.0 is outside the range, use midpoint (cannot achieve identity)
            self._identity_deriv_logit = jnp.array(0.0)

    @staticmethod
    def required_out_dim(
        dim: int, num_bins: int, boundary_slopes: str = "linear_tails"
    ) -> int:
        """
        Return required conditioner output dimension for SplineCoupling.

        - 'linear_tails': `dim * (3K - 1)` (K widths + K heights + K-1 interior).
        - 'circular':     `dim * 3K`       (K widths + K heights + K derivs).
        """
        return dim * _params_per_scalar(num_bins, boundary_slopes)

    @classmethod
    def create(
        cls,
        key: PRNGKey,
        dim: int,
        mask: Array,
        hidden_dim: int,
        n_hidden_layers: int,
        *,
        context_dim: int = 0,
        num_bins: int = 8,
        tail_bound: float = 5.0,
        min_bin_width: float = 1e-2,
        min_bin_height: float = 1e-2,
        min_derivative: float = 1e-2,
        max_derivative: float = 10.0,
        activation: Callable[[Array], Array] = nn.elu,
        res_scale: float = 0.1,
        boundary_slopes: str = "linear_tails",
    ) -> Tuple["SplineCoupling", dict]:
        """
        Factory method to create SplineCoupling with properly configured MLP.

        This handles the output dimension calculation internally and initializes
        parameters, returning both the coupling and its params ready to use.

        Arguments:
            key: JAX PRNGKey for parameter initialization.
            dim: Input/output dimensionality.
            mask: Binary mask of shape (dim,). 1 = frozen, 0 = transformed.
            hidden_dim: Width of hidden layers in conditioner MLP.
            n_hidden_layers: Number of residual blocks in conditioner MLP.
            context_dim: Context dimension (0 for unconditional).
            num_bins: Number of spline bins (default: 8).
            tail_bound: Spline acts on [-B, B]; identity outside (default: 5.0).
            min_bin_width: Minimum bin width for stability (default: 1e-2).
            min_bin_height: Minimum bin height for stability (default: 1e-2).
            min_derivative: Minimum derivative for stability (default: 1e-2).
            max_derivative: Maximum derivative for stability (default: 10.0).
            activation: Activation function for MLP (default: elu).
            res_scale: Residual connection scale (default: 0.1).

        Returns:
            Tuple of (coupling, params) ready to use.

        Raises:
            ValueError: If mask length doesn't match dim, or invalid parameters.

        Example:
            >>> coupling, params = SplineCoupling.create(
            ...     key, dim=4, mask=jnp.array([1, 0, 1, 0]),
            ...     hidden_dim=64, n_hidden_layers=2, num_bins=8
            ... )
            >>> y, log_det = coupling.forward(params, x)
        """
        # Validate inputs
        if dim <= 0:
            raise ValueError(f"SplineCoupling.create: dim must be positive, got {dim}.")
        if hidden_dim <= 0:
            raise ValueError(f"SplineCoupling.create: hidden_dim must be positive, got {hidden_dim}.")
        if num_bins <= 0:
            raise ValueError(f"SplineCoupling.create: num_bins must be positive, got {num_bins}.")

        mask = jnp.asarray(mask)
        if mask.shape != (dim,):
            raise ValueError(
                f"SplineCoupling.create: mask shape {mask.shape} doesn't match (dim,) = ({dim},)."
            )

        # Create MLP with correct output dimension
        out_dim = cls.required_out_dim(dim, num_bins, boundary_slopes=boundary_slopes)
        mlp = MLP(
            x_dim=dim,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            out_dim=out_dim,
            activation=activation,
            res_scale=res_scale,
        )

        # Create coupling
        coupling = cls(
            mask=mask,
            conditioner=mlp,
            num_bins=num_bins,
            tail_bound=tail_bound,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
            max_derivative=max_derivative,
            boundary_slopes=boundary_slopes,
        )

        # Initialize params
        params = coupling.init_params(key, context_dim=context_dim)

        return coupling, params

    def _conditioner_params(self, params: Any) -> Any:
        try:
            return params["mlp"]
        except Exception as e:
            raise KeyError("SplineCoupling expected params to contain key 'mlp'.") from e

    def _check_x(self, x: Array) -> int:
        if x.ndim < 1:
            raise ValueError(
                f"SplineCoupling expected input with at least 1 dimension, got {x.shape}."
            )
        dim = x.shape[-1]
        if self.mask.shape != (dim,):
            raise ValueError(
                f"SplineCoupling: mask shape {self.mask.shape} does not match "
                f"input dim {dim}."
            )
        if self.num_bins < 1:
            raise ValueError(
                f"SplineCoupling: num_bins must be >= 1, got {self.num_bins}."
            )
        return dim

    def _compute_spline_params(
        self,
        mlp_params: Any,
        x: Array,
        context: Array | None = None,
        g_value: Array | None = None,
    ) -> Tuple[Array, Array, Array]:
        """
        Compute raw spline parameters from the conditioner and reshape them to:
          widths:      (..., dim, K)
          heights:     (..., dim, K)
          derivatives: (..., dim, K-1)

        Arguments:
          mlp_params: parameters for the conditioner network.
          x: input tensor of shape (..., dim).
          context: optional conditioning tensor passed to the conditioner.
          g_value: optional gate value. When g_value=0, spline params are set
              to produce identity transform (uniform bins, derivative=1).
        """
        dim = x.shape[-1]
        K = self.num_bins
        params_per_dim = _params_per_scalar(K, self.boundary_slopes)
        expected_out_dim = dim * params_per_dim

        # Conditioner sees only the masked (conditioning) part.
        x_cond = x * self.mask

        theta = self.conditioner.apply({"params": mlp_params}, x_cond, context)  # (..., expected_out_dim)
        if theta.shape[-1] != expected_out_dim:
            raise ValueError(
                "SplineCoupling: conditioner output has wrong size. "
                f"Expected last dim {expected_out_dim}, got {theta.shape[-1]}."
            )

        theta = theta.reshape(theta.shape[:-1] + (dim, params_per_dim))

        widths = theta[..., :K]                 # (..., dim, K)
        heights = theta[..., K : 2 * K]         # (..., dim, K)
        derivatives = theta[..., 2 * K :]       # (..., dim, K-1) or (..., dim, K)

        # Apply identity gate: when g_value=0, interpolate to identity spline params.
        # Identity spline: widths=heights=0 (uniform bins after softmax), derivatives → 1.
        if g_value is not None:
            # g_value shape: () or (batch,). Broadcast to (..., 1, 1) for element-wise.
            g = g_value[..., None, None]  # (..., 1, 1)

            # widths/heights → 0 when g → 0 (uniform bins)
            widths = g * widths
            heights = g * heights

            # derivatives: interpolate from identity_deriv_logit (gives d=1) to learned value
            identity_d = self._identity_deriv_logit
            derivatives = (1 - g) * identity_d + g * derivatives

        return widths, heights, derivatives

    def _apply_splines(self, x: Array, widths: Array, heights: Array, derivatives: Array, inverse: bool) -> Tuple[Array, Array]:
        """
        Apply scalar splines per dimension using vmap.

        Returns:
          y: (..., dim)
          logabsdet_per_dim: (..., dim)
        """
        K = self.num_bins  # for readability only

        def per_dim_fn(x_d: Array, w_d: Array, h_d: Array, d_d: Array) -> Tuple[Array, Array]:
            return rational_quadratic_spline(
                inputs=x_d,
                unnormalized_widths=w_d,
                unnormalized_heights=h_d,
                unnormalized_derivatives=d_d,
                tail_bound=self.tail_bound,
                min_bin_width=self.min_bin_width,
                min_bin_height=self.min_bin_height,
                min_derivative=self.min_derivative,
                max_derivative=self.max_derivative,
                inverse=inverse,
                boundary_slopes=self.boundary_slopes,
            )

        # vmap over the feature dimension:
        #   x: (..., dim)          -> map axis -1
        #   widths/heights: (..., dim, K)   -> map axis -2
        #   derivatives:    (..., dim, K-1) -> map axis -2
        y, logabsdet = jax.vmap(
            per_dim_fn,
            in_axes=(-1, -2, -2, -2),
            out_axes=(-1, -1),
        )(x, widths, heights, derivatives)

        return y, logabsdet

    def forward(
        self,
        params: Any,
        x: Array,
        context: Array | None = None,
        g_value: Array | None = None,
    ) -> Tuple[Array, Array]:
        """
        Forward map: x -> y with log_det_forward = log|det ∂y/∂x|.

        Arguments:
          params: dict with key "mlp" for conditioner parameters.
          x: input tensor of shape (..., dim).
          context: optional conditioning tensor passed to the conditioner.
          g_value: optional gate value for identity_gate. When g_value=0, returns identity.
        """
        self._check_x(x)
        mlp_params = self._conditioner_params(params)

        widths, heights, derivatives = self._compute_spline_params(
            mlp_params, x, context, g_value=g_value
        )
        y_spline, logabsdet_per_dim = self._apply_splines(
            x, widths, heights, derivatives, inverse=False
        )

        # Only transform unmasked dims; masked dims stay identity.
        inv_mask = 1.0 - self.mask
        y = x * self.mask + y_spline * inv_mask
        log_det = jnp.sum(logabsdet_per_dim * inv_mask, axis=-1)

        return y, log_det

    def inverse(
        self,
        params: Any,
        y: Array,
        context: Array | None = None,
        g_value: Array | None = None,
    ) -> Tuple[Array, Array]:
        """
        Inverse map: y -> x with log_det_inverse = log|det ∂x/∂y|.

        Note: the conditioner depends only on the masked (unchanged) subset.
        Since masked dimensions are copied through exactly, y * mask == x * mask.

        Arguments:
          params: dict with key "mlp" for conditioner parameters.
          y: input tensor of shape (..., dim).
          context: optional conditioning tensor passed to the conditioner.
          g_value: optional gate value for identity_gate. When g_value=0, returns identity.
        """
        self._check_x(y)
        mlp_params = self._conditioner_params(params)

        widths, heights, derivatives = self._compute_spline_params(
            mlp_params, y, context, g_value=g_value
        )
        x_spline, logabsdet_per_dim = self._apply_splines(
            y, widths, heights, derivatives, inverse=True
        )

        inv_mask = 1.0 - self.mask
        x = y * self.mask + x_spline * inv_mask
        log_det = jnp.sum(logabsdet_per_dim * inv_mask, axis=-1)

        return x, log_det

    def _patch_dense_out(self, mlp_params: Any) -> Any:
        """
        Install an identity-spline bias on the conditioner's `dense_out`.

        The bias length is read from the conditioner itself, so any conditioner
        whose `dense_out` bias is a multiple of `params_per_scalar` works —
        `MLP` (flat, `dim * params_per_scalar`) and any hypothetical per-scalar
        variant (`params_per_scalar` repeated over `dim` scalars) are handled
        uniformly. `identity_spline_bias` is the same per-scalar pattern tiled
        across `num_scalars`.
        """
        if not (hasattr(self.conditioner, "get_output_layer") and
                hasattr(self.conditioner, "set_output_layer")):
            raise RuntimeError(
                "SplineCoupling._patch_dense_out: conditioner must implement "
                "get_output_layer() and set_output_layer() methods."
            )

        out_layer = self.conditioner.get_output_layer(mlp_params)
        bias_size = int(out_layer["bias"].shape[0])
        params_per_scalar = _params_per_scalar(self.num_bins, self.boundary_slopes)
        if bias_size % params_per_scalar != 0:
            raise ValueError(
                f"SplineCoupling: conditioner dense_out bias shape ({bias_size},) "
                f"is not a multiple of params_per_scalar={params_per_scalar}."
            )
        new_kernel = jnp.zeros_like(out_layer["kernel"])
        new_bias = identity_spline_bias(
            num_scalars=bias_size // params_per_scalar,
            num_bins=self.num_bins,
            min_derivative=self.min_derivative,
            max_derivative=self.max_derivative,
            boundary_slopes=self.boundary_slopes,
        )
        return self.conditioner.set_output_layer(mlp_params, new_kernel, new_bias)

    def init_params(self, key: PRNGKey, context_dim: int = 0) -> dict:
        """
        Initialize parameters for this transform.

        Uses Flax init + patches final layer for near-identity spline init.

        Arguments:
            key: JAX PRNGKey for parameter initialization.
            context_dim: Context dimension (0 for unconditional).

        Returns:
            Dict with key 'mlp' containing MLP parameters.
        """
        dim = int(self.mask.shape[0])
        dummy_x = jnp.zeros((1, dim), dtype=jnp.float32)
        dummy_context = jnp.zeros((1, context_dim), dtype=jnp.float32) if context_dim > 0 else None
        variables = self.conditioner.init(key, dummy_x, dummy_context)
        mlp_params = variables["params"]
        mlp_params = self._patch_dense_out(mlp_params)
        return {"mlp": mlp_params}


# ===================================================================
# Split Coupling: structured coupling on rank-N events
# ===================================================================
# SplitCoupling is the structured analogue of SplineCoupling. Instead of a
# flat (dim,) binary mask, it splits along a chosen event axis at a chosen
# index, passes the frozen half (flattened) to a conditioner, and applies
# a per-scalar monotonic rational-quadratic spline to the transformed half.
# Log-det is summed over the trailing `event_ndims` axes so that the output
# has the enclosing batch shape.
#
# Design: this class deliberately duplicates the spline-call and identity-init
# bias patching from SplineCoupling rather than extracting a shared helper.
# The duplication is ~30 lines and keeps each coupling variant readable on
# its own; per AGENTS.md "three similar lines beat a premature helper."
@dataclass
class SplitCoupling:
    """
    Axis-based spline coupling for rank-N events.

    Splits input along `split_axis` at `split_index`, feeds the frozen slice
    (flattened) to the conditioner, applies monotonic rational-quadratic
    splines elementwise to the transformed slice, concatenates back together.

    Input/output shape:
      x:         (*batch, *event_shape)   where len(event_shape) == event_ndims
      y:         same as x
      log_det:   batch shape               (summed over all event axes)

    Target use case (particle systems):
      event_shape = (N, d),  split_axis=-2,  split_index=N//2,  event_ndims=2
      frozen slice:      (*batch, N//2, d)      -> conditioner input
                                                   (flattened to (*batch, N//2 * d)
                                                    when flatten_input=True)
      transformed slice: (*batch, N//2, d)      -> elementwise spline

    Fields:
      event_shape: trailing event shape, e.g. (N, d); its rank must equal
                   event_ndims.
      split_axis:  negative int indexing from the end; must lie inside event
                   axes (i.e., abs(split_axis) <= event_ndims).
      split_index: size of the first partition along split_axis.
      event_ndims: number of trailing event axes (>= 1).
      conditioner: Flax module producing a tensor whose total trailing size
                   equals `required_out_dim(transformed_flat, num_bins,
                   boundary_slopes)` = `transformed_flat * (3K-1)` for
                   `linear_tails` or `transformed_flat * 3K` for `circular`.
                   Input shape depends on `flatten_input` (see below).
      swap:        if False, the first partition is frozen; if True, the last.
      flatten_input: when True (default), the frozen slice is flattened to
                   `(*batch, frozen_flat)` before the conditioner call — the
                   contract `MLP` expects. When False, the conditioner sees
                   the structured frozen slice `(*batch, *frozen_shape)`
                   directly; use this for permutation-aware conditioners
                   (`DeepSets`, `Transformer`, `GNN`) that need the particle
                   axis preserved. The output-side reshape only requires the
                   total element count; the conditioner may emit either flat
                   `(*batch, transformed_flat * params_per_scalar)` or
                   structured `(*batch, *transformed_shape, params_per_scalar)`.
      num_bins..max_derivative: same as SplineCoupling.

    Mask semantics: none. The partition is geometric (axis+index), not a
    per-scalar mask. Alternate `swap` between layers to cover all particles.

    Identity initialization: same mechanism as SplineCoupling. The conditioner
    output layer has zeroed kernel and a bias that makes widths=heights=0
    (uniform bins) and derivatives=1 (via the stable_logit of
    (1-min_deriv)/(max_deriv-min_deriv)). Inside [-tail_bound, tail_bound] the
    spline is identity at init.
    """
    event_shape: Tuple[int, ...]
    split_axis: int
    split_index: int
    event_ndims: int
    conditioner: Any
    swap: bool = False
    num_bins: int = 8
    tail_bound: float = 5.0
    min_bin_width: float = 1e-2
    min_bin_height: float = 1e-2
    min_derivative: float = 1e-2
    max_derivative: float = 10.0
    boundary_slopes: str = "linear_tails"
    flatten_input: bool = True

    def __post_init__(self):
        self.event_shape = tuple(int(d) for d in self.event_shape)
        if self.split_axis >= 0:
            raise ValueError(
                f"SplitCoupling: split_axis must be negative, got {self.split_axis}."
            )
        _validate_boundary_slopes(self.boundary_slopes, where="SplitCoupling")
        if self.event_ndims < 1:
            raise ValueError(
                f"SplitCoupling: event_ndims must be >= 1, got {self.event_ndims}."
            )
        if len(self.event_shape) != self.event_ndims:
            raise ValueError(
                f"SplitCoupling: event_shape {self.event_shape} has rank "
                f"{len(self.event_shape)}, expected event_ndims={self.event_ndims}."
            )
        if -self.split_axis > self.event_ndims:
            raise ValueError(
                f"SplitCoupling: split_axis={self.split_axis} lies outside the "
                f"trailing {self.event_ndims} event axes."
            )
        if self.split_index <= 0:
            raise ValueError(
                f"SplitCoupling: split_index must be positive, got {self.split_index}."
            )
        event_axis = self.event_ndims + self.split_axis
        axis_size = self.event_shape[event_axis]
        if self.split_index >= axis_size:
            raise ValueError(
                f"SplitCoupling: split_index={self.split_index} must be < size "
                f"along split_axis ({axis_size})."
            )
        validate_conditioner(self.conditioner, name="SplitCoupling.conditioner")

        # Warn if identity-spline init is unreachable (derivative=1 outside range).
        lo, hi = float(self.min_derivative), float(self.max_derivative)
        if not (lo < 1.0 < hi):
            warnings.warn(
                f"SplitCoupling: derivative range [{lo}, {hi}] excludes 1.0; "
                "identity-like initialization not possible.",
                stacklevel=2,
            )

    @staticmethod
    def required_out_dim(
        transformed_flat: int,
        num_bins: int,
        boundary_slopes: str = "linear_tails",
    ) -> int:
        """Conditioner output dimension for the transformed slice.

        - 'linear_tails': `transformed_flat * (3K - 1)`.
        - 'circular':     `transformed_flat * 3K` (one extra shared boundary slope per scalar).
        """
        return transformed_flat * _params_per_scalar(num_bins, boundary_slopes)

    # ------------------------------------------------------------------
    # Partition sizing. Both `create()` (pre-instance, for MLP sizing) and
    # `init_params` (on an instance) need (frozen_flat, transformed_flat);
    # this staticmethod is the single source of truth for the computation.
    # ------------------------------------------------------------------
    @staticmethod
    def _partition_flats(
        event_shape: Sequence[int],
        split_axis: int,
        split_index: int,
        event_ndims: int,
        swap: bool,
    ) -> Tuple[int, int]:
        event_shape = tuple(int(d) for d in event_shape)
        event_axis = event_ndims + split_axis
        if not (0 <= event_axis < event_ndims):
            raise ValueError(
                f"SplitCoupling: split_axis={split_axis} lies outside the "
                f"trailing {event_ndims} event axes."
            )
        axis_size = event_shape[event_axis]
        frozen_sz = split_index if not swap else axis_size - split_index
        transformed_sz = axis_size - frozen_sz
        other = math.prod(event_shape) // axis_size
        return frozen_sz * other, transformed_sz * other

    @classmethod
    def create(
        cls,
        key: PRNGKey,
        event_shape: Sequence[int],
        split_axis: int,
        split_index: int,
        event_ndims: int,
        hidden_dim: int,
        n_hidden_layers: int,
        *,
        context_dim: int = 0,
        num_bins: int = 8,
        tail_bound: float = 5.0,
        min_bin_width: float = 1e-2,
        min_bin_height: float = 1e-2,
        min_derivative: float = 1e-2,
        max_derivative: float = 10.0,
        swap: bool = False,
        activation: Callable[[Array], Array] = nn.elu,
        res_scale: float = 0.1,
        boundary_slopes: str = "linear_tails",
    ) -> Tuple["SplitCoupling", dict]:
        """
        Factory. Sizes the conditioner MLP for the given event_shape partition,
        initializes with identity-spline bias, returns (coupling, params).
        """
        event_shape = tuple(int(d) for d in event_shape)
        frozen_flat, transformed_flat = cls._partition_flats(
            event_shape, split_axis, split_index, event_ndims, swap
        )

        mlp = MLP(
            x_dim=frozen_flat,
            context_dim=context_dim,
            hidden_dim=hidden_dim,
            n_hidden_layers=n_hidden_layers,
            out_dim=cls.required_out_dim(
                transformed_flat, num_bins, boundary_slopes=boundary_slopes
            ),
            activation=activation,
            res_scale=res_scale,
        )

        coupling = cls(
            event_shape=event_shape,
            split_axis=split_axis,
            split_index=split_index,
            event_ndims=event_ndims,
            conditioner=mlp,
            swap=swap,
            num_bins=num_bins,
            tail_bound=tail_bound,
            min_bin_width=min_bin_width,
            min_bin_height=min_bin_height,
            min_derivative=min_derivative,
            max_derivative=max_derivative,
            boundary_slopes=boundary_slopes,
        )
        params = coupling.init_params(key, context_dim=context_dim)
        return coupling, params

    # ------------------------------------------------------------------
    # Split / combine along split_axis. jnp.split is jit-friendly.
    # ------------------------------------------------------------------
    def _split(self, x: Array) -> Tuple[Array, Array]:
        a, b = jnp.split(x, [self.split_index], axis=self.split_axis)
        return (b, a) if self.swap else (a, b)

    def _combine(self, frozen: Array, transformed_new: Array) -> Array:
        if self.swap:
            return jnp.concatenate([transformed_new, frozen], axis=self.split_axis)
        return jnp.concatenate([frozen, transformed_new], axis=self.split_axis)

    def _check_x(self, x: Array) -> None:
        """Validate trailing shape matches event_shape; mirrors SplineCoupling._check_x."""
        if x.ndim < self.event_ndims:
            raise ValueError(
                f"SplitCoupling: input has rank {x.ndim}, "
                f"need at least event_ndims={self.event_ndims}."
            )
        if x.shape[-self.event_ndims:] != self.event_shape:
            raise ValueError(
                f"SplitCoupling: expected trailing event_shape {self.event_shape}, "
                f"got {x.shape[-self.event_ndims:]}."
            )

    def _forward_or_inverse(
        self,
        params: Any,
        x: Array,
        context: Array | None,
        inverse: bool,
    ) -> Tuple[Array, Array]:
        self._check_x(x)
        mlp_params = params["mlp"]
        frozen, transformed = self._split(x)

        K = self.num_bins
        params_per_scalar = _params_per_scalar(K, self.boundary_slopes)
        transformed_event_shape = transformed.shape[-self.event_ndims:]
        batch_shape = frozen.shape[: -self.event_ndims]

        # Conditioner sees the frozen slice (flat or structured per
        # `flatten_input`); its output is reshaped to per-scalar spline params
        # over the transformed slice. The output reshape depends only on the
        # total element count, so the conditioner is free to emit a flat or
        # structured tensor of the right size.
        cond_input = frozen.reshape(batch_shape + (-1,)) if self.flatten_input else frozen
        theta = self.conditioner.apply({"params": mlp_params}, cond_input, context)
        theta = theta.reshape(batch_shape + transformed_event_shape + (params_per_scalar,))
        widths = theta[..., :K]
        heights = theta[..., K : 2 * K]
        derivatives = theta[..., 2 * K :]

        transformed_new, logabsdet_per_scalar = rational_quadratic_spline(
            inputs=transformed,
            unnormalized_widths=widths,
            unnormalized_heights=heights,
            unnormalized_derivatives=derivatives,
            tail_bound=self.tail_bound,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            max_derivative=self.max_derivative,
            inverse=inverse,
            boundary_slopes=self.boundary_slopes,
        )
        y = self._combine(frozen, transformed_new)
        # Sum per-scalar log-det over all event axes => shape == batch_shape.
        log_det = jnp.sum(
            logabsdet_per_scalar,
            axis=tuple(range(-self.event_ndims, 0)),
        )
        return y, log_det

    def forward(
        self,
        params: Any,
        x: Array,
        context: Array | None = None,
    ) -> Tuple[Array, Array]:
        return self._forward_or_inverse(params, x, context, inverse=False)

    def inverse(
        self,
        params: Any,
        y: Array,
        context: Array | None = None,
    ) -> Tuple[Array, Array]:
        return self._forward_or_inverse(params, y, context, inverse=True)

    def _patch_dense_out(self, mlp_params: Any) -> Any:
        """Install an identity-spline bias on the conditioner's `dense_out`.

        The bias length is read from the conditioner itself, so both flat-output
        (`MLP`, bias = `transformed_flat * params_per_scalar`) and per-token
        output (`Transformer`, `GNN`, bias = `d * params_per_scalar`) conditioners
        are handled uniformly. `identity_spline_bias` is the same per-scalar
        pattern tiled across `num_scalars` scalars, so sizing it to match the
        conditioner's bias works for any multiple of `params_per_scalar`.
        """
        if not (hasattr(self.conditioner, "get_output_layer") and
                hasattr(self.conditioner, "set_output_layer")):
            raise RuntimeError(
                "SplitCoupling._patch_dense_out: conditioner must implement "
                "get_output_layer() and set_output_layer()."
            )
        out_layer = self.conditioner.get_output_layer(mlp_params)
        bias_size = int(out_layer["bias"].shape[0])
        params_per_scalar = _params_per_scalar(self.num_bins, self.boundary_slopes)
        if bias_size % params_per_scalar != 0:
            raise ValueError(
                f"SplitCoupling: conditioner dense_out bias shape ({bias_size},) "
                f"is not a multiple of params_per_scalar={params_per_scalar}."
            )
        new_kernel = jnp.zeros_like(out_layer["kernel"])
        new_bias = identity_spline_bias(
            num_scalars=bias_size // params_per_scalar,
            num_bins=self.num_bins,
            min_derivative=self.min_derivative,
            max_derivative=self.max_derivative,
            boundary_slopes=self.boundary_slopes,
        )
        return self.conditioner.set_output_layer(mlp_params, new_kernel, new_bias)

    def init_params(self, key: PRNGKey, context_dim: int = 0) -> dict:
        """
        Initialize conditioner params with an identity-spline output layer.

        Shapes are derived from `self.event_shape` + the partition fields.
        After init, the conditioner's full output shape is checked against the
        coupling's expectation — a per-token conditioner (`Transformer`, `GNN`)
        sized for a symmetric split but used with an asymmetric `split_index`
        raises here, not as a cryptic reshape error during the first forward.
        """
        frozen_flat, transformed_flat = self._partition_flats(
            self.event_shape, self.split_axis, self.split_index,
            self.event_ndims, self.swap,
        )
        if self.flatten_input:
            dummy_x = jnp.zeros((1, frozen_flat), dtype=jnp.float32)
        else:
            event_axis = self.event_ndims + self.split_axis
            axis_size = self.event_shape[event_axis]
            frozen_sz = self.split_index if not self.swap else axis_size - self.split_index
            frozen_shape = list(self.event_shape)
            frozen_shape[event_axis] = frozen_sz
            dummy_x = jnp.zeros((1,) + tuple(frozen_shape), dtype=jnp.float32)
        dummy_context = (
            jnp.zeros((1, context_dim), dtype=jnp.float32) if context_dim > 0 else None
        )
        variables = self.conditioner.init(key, dummy_x, dummy_context)
        mlp_params = variables["params"]
        mlp_params = self._patch_dense_out(mlp_params)

        # Validate the conditioner's output shape against the coupling's
        # expectation. The reshape in `_forward_or_inverse` requires
        # `prod(trailing) == transformed_flat * params_per_scalar`.
        params_per_scalar = _params_per_scalar(self.num_bins, self.boundary_slopes)
        expected = transformed_flat * params_per_scalar
        dummy_out = self.conditioner.apply(
            {"params": mlp_params}, dummy_x, dummy_context
        )
        total_trailing = math.prod(dummy_out.shape[1:])
        if total_trailing != expected:
            axis_size = self.event_shape[self.event_ndims + self.split_axis]
            raise ValueError(
                f"SplitCoupling: conditioner "
                f"{type(self.conditioner).__name__!r} produces output with "
                f"total trailing size {total_trailing}; expected {expected} "
                f"(= transformed_flat={transformed_flat} * "
                f"params_per_scalar={params_per_scalar}). Common cause: a "
                f"per-token conditioner (Transformer, GNN) sized for a "
                f"symmetric split but used with split_index={self.split_index} "
                f"on an axis of size {axis_size} (N_frozen != N_transformed)."
            )
        return {"mlp": mlp_params}


# ===================================================================
# Permutation Transform: Fixed permutation of dimensions
# ===================================================================
@dataclass
class Permutation:
    """
    Permutation along a chosen event axis.

    Forward:  y = jnp.take(x, perm,     axis=event_axis)
    Inverse:  x = jnp.take(y, inv_perm, axis=event_axis)

    Both Jacobians are permutation matrices with unit determinant, so
    `log_det` is exactly zero. Log-det shape is the input's shape minus
    the permuted axis (i.e., full batch + remaining event axes), kept
    consistent with other rank-N-aware transforms.

    `event_axis` must be negative (an offset from the end). Default `-1`
    preserves the historic "last-axis only" behaviour: with input shape
    `(..., dim)`, `perm.shape == (dim,)` permutes coordinates. Setting
    `event_axis=-2` on `(B, N, d)` input permutes particles instead —
    the standard pattern for rank-N particle flows.

    `perm` must be a 1-D integer array whose length matches the target
    axis; an inverse permutation is precomputed at construction.
    """
    perm: Array
    event_axis: int = -1

    def __post_init__(self):
        self.perm = jnp.asarray(self.perm)
        if self.perm.ndim != 1:
            raise ValueError(
                f"Permutation perm must be 1D, got shape {self.perm.shape}."
            )
        if not jnp.issubdtype(self.perm.dtype, jnp.integer):
            raise TypeError(
                f"Permutation perm must be integer dtype, got {self.perm.dtype}."
            )
        if self.event_axis >= 0:
            raise ValueError(
                f"Permutation event_axis must be negative (offset from end), "
                f"got {self.event_axis}."
            )

        n = self.perm.shape[0]
        inv_perm = jnp.empty_like(self.perm)
        inv_perm = inv_perm.at[self.perm].set(jnp.arange(n))
        self._inv_perm = inv_perm

    @property
    def dim(self) -> int:
        """Length of the permuted axis (the size of `perm`)."""
        return int(self.perm.shape[0])

    def _check_axis_size(self, x: Array) -> None:
        axis = self.event_axis
        if -axis > x.ndim:
            raise ValueError(
                f"Permutation: event_axis={axis} lies outside input rank {x.ndim}."
            )
        if x.shape[axis] != self.dim:
            raise ValueError(
                f"Permutation expected input with axis {axis} of size {self.dim}, "
                f"got shape {x.shape}."
            )

    def _zero_logdet(self, x: Array) -> Array:
        """Return a zero log-det matching the shape of `x` minus the permuted axis."""
        remaining = list(x.shape)
        remaining.pop(self.event_axis)
        return jnp.zeros(tuple(remaining), dtype=x.dtype)

    def forward(
        self, params: Any, x: Array, context: Array | None = None
    ) -> Tuple[Array, Array]:
        """Forward permutation along `event_axis`."""
        del context
        self._check_axis_size(x)
        y = jnp.take(x, self.perm, axis=self.event_axis)
        return y, self._zero_logdet(x)

    def inverse(
        self, params: Any, y: Array, context: Array | None = None
    ) -> Tuple[Array, Array]:
        """Inverse permutation along `event_axis`."""
        del context
        self._check_axis_size(y)
        x = jnp.take(y, self._inv_perm, axis=self.event_axis)
        return x, self._zero_logdet(y)

    def init_params(self, key: PRNGKey, context_dim: int = 0) -> dict:
        """Permutation has no learnable parameters."""
        del key, context_dim
        return {}

    @classmethod
    def create(
        cls,
        key: PRNGKey,
        perm: Array,
        event_axis: int = -1,
    ) -> Tuple["Permutation", dict]:
        """
        Factory method to create Permutation and initialize params.

        Arguments:
            key: JAX PRNGKey (unused; included for interface consistency).
            perm: Permutation indices, shape `(k,)` where `k` is the size of
                the permuted axis.
            event_axis: Negative integer selecting which axis is permuted.
                Default `-1` (historic coordinate-axis behaviour). Use
                `event_axis=-2` for particle-axis permutation on `(B, N, d)`.

        Example:
            >>> # Reverse particle order on (B, N, d):
            >>> transform, params = Permutation.create(
            ...     key, perm=jnp.arange(N)[::-1], event_axis=-2,
            ... )
        """
        del key
        transform = cls(perm=perm, event_axis=event_axis)
        params = transform.init_params(None)  # type: ignore
        return transform, params


# ===================================================================
# Circular Shift: rigid torus rotation
# ===================================================================
# Per-coordinate learnable shift with modular wrap:
#   y = (x - lower + shift) mod (upper - lower) + lower.
# Log-det is identically zero (rigid translation).
#
# This is the "rotation" half of a torus diffeomorphism. Compose with a
# circular-mode spline coupling to get full torus-bijection expressivity:
# the shift moves the seam freely around the circle, and the spline
# deforms locally (matched slopes at the seam → C^1 on the torus).
@dataclass
class CircularShift:
    """Rigid shift modulo the box length, per-coordinate learnable.

    Input shape: `(*batch, ..., coord_dim)`, where `coord_dim == geometry.d`.
    Only the last axis is "coord-like"; the shift vector has shape `(d,)`
    and broadcasts across all preceding axes (particles, batch, etc.).
    This lets a single CircularShift layer rotate a whole rank-N particle
    configuration by the same per-coord displacement, as a rigid-body
    operation on the torus.

    Log-det: scalar zero. Rigid shift has unit Jacobian.

    Compose with a `SplineCoupling` (or `SplitCoupling`) whose inner
    spline uses `boundary_slopes='circular'` to model general torus
    diffeomorphisms.
    """
    geometry: Geometry

    def __post_init__(self):
        if not isinstance(self.geometry, Geometry):
            raise TypeError(
                f"CircularShift: geometry must be a Geometry instance, "
                f"got {type(self.geometry).__name__}. "
                f"Use CircularShift.from_scalar_box(...) for the legacy construction."
            )
        # Precompute jnp arrays for the hot path. These are constants (not
        # traced) so materialising once is fine.
        object.__setattr__(
            self, "_lower_j", jnp.asarray(self.geometry.lower, dtype=jnp.float32)
        )
        object.__setattr__(
            self, "_box_j", jnp.asarray(self.geometry.box, dtype=jnp.float32)
        )

    @property
    def coord_dim(self) -> int:
        """Number of coordinate axes — equals `geometry.d`."""
        return self.geometry.d

    def _wrap_shift(
        self, params: Any, x: Array, sign: float
    ) -> Tuple[Array, Array]:
        shift = params["shift"]
        d = self.geometry.d
        if shift.shape != (d,):
            raise ValueError(
                f"CircularShift: expected shift of shape ({d},), got {shift.shape}."
            )
        y = jnp.mod(x - self._lower_j + sign * shift, self._box_j) + self._lower_j
        # Scalar zero: composes safely in CompositeTransform whose accumulator
        # is scalar zero + block log-dets broadcast up.
        log_det = jnp.zeros((), dtype=x.dtype)
        return y, log_det

    def forward(
        self, params: Any, x: Array, context: Array | None = None
    ) -> Tuple[Array, Array]:
        del context  # Unused.
        return self._wrap_shift(params, x, sign=+1.0)

    def inverse(
        self, params: Any, y: Array, context: Array | None = None
    ) -> Tuple[Array, Array]:
        del context  # Unused.
        return self._wrap_shift(params, y, sign=-1.0)

    def init_params(self, key: PRNGKey, context_dim: int = 0) -> dict:
        """Zero shift → layer is identity at init."""
        del key, context_dim
        return {"shift": jnp.zeros((self.geometry.d,), dtype=jnp.float32)}

    @classmethod
    def create(
        cls,
        key: PRNGKey,
        geometry: Geometry,
    ) -> Tuple["CircularShift", dict]:
        """Factory. Returns (transform, zero-initialized params)."""
        transform = cls(geometry=geometry)
        params = transform.init_params(key)
        return transform, params

    @classmethod
    def from_scalar_box(
        cls, coord_dim: int, lower: float, upper: float
    ) -> "CircularShift":
        """Convenience factory for a cubic box with scalar bounds.

        Equivalent to `CircularShift(geometry=Geometry.cubic(d=coord_dim,
        side=upper-lower, lower=lower))`. Provided so callers migrating from
        the pre-Geometry API don't need to import `Geometry` just to
        construct a cube.
        """
        geom = Geometry.cubic(
            d=coord_dim, side=float(upper) - float(lower), lower=float(lower)
        )
        return cls(geometry=geom)


# ===================================================================
# Rescale: fixed per-axis affine from geometry.box to a canonical range.
#
#   y_i = target_lower_i + (x_i - lower_i) * scale_i,
#       scale_i = (target_upper_i - target_lower_i) / (upper_i - lower_i).
#
# Non-learnable, non-conditional; carries no parameters. Typical use is
# the first layer of a particle flow, mapping a physical Geometry.box
# onto the canonical spline range [-1, 1] so every downstream spline /
# coupling can assume a fixed domain. For a learnable affine, use
# LinearTransform.
# ===================================================================
@dataclass
class Rescale:
    """Fixed per-axis affine from `geometry.box` to a canonical range.

    Input shape: `(*batch, *event_shape)`, with the last axis of
    `event_shape` being the coord axis of length `geometry.d`. Each
    coord scalar is rescaled independently; any leading event axes
    (particles, species, ...) are replicated identically.

    Log-det: scalar
        `event_factor * sum_i log(scale_i)`,
    where `event_factor = prod(event_shape[:-1])` counts the non-coord
    event axes. For a rank-1 event `(d,)` this is 1; for a rank-2 event
    `(N, d)` this is `N`. Scalar log-dets broadcast through
    `CompositeTransform`'s accumulator.
    """

    geometry: Geometry
    target: Tuple[Any, Any] = (-1.0, 1.0)
    event_shape: Tuple[int, ...] | None = None

    def __post_init__(self):
        if not isinstance(self.geometry, Geometry):
            raise TypeError(
                f"Rescale: geometry must be a Geometry instance, "
                f"got {type(self.geometry).__name__}."
            )
        d = self.geometry.d

        tl_raw, tu_raw = self.target
        tl = np.asarray(tl_raw, dtype=np.float32)
        tu = np.asarray(tu_raw, dtype=np.float32)
        if tl.ndim == 0:
            tl = np.full((d,), float(tl), dtype=np.float32)
        if tu.ndim == 0:
            tu = np.full((d,), float(tu), dtype=np.float32)
        if tl.shape != (d,) or tu.shape != (d,):
            raise ValueError(
                f"Rescale: target bounds must be scalar or shape ({d},); "
                f"got lower.shape={tl.shape}, upper.shape={tu.shape}."
            )
        if np.any(tl >= tu):
            raise ValueError(
                f"Rescale: target lower must be strictly < target upper "
                f"element-wise; got lower={tl}, upper={tu}."
            )
        object.__setattr__(self, "target", (tl, tu))

        if self.event_shape is None:
            event_shape: Tuple[int, ...] = (d,)
        else:
            event_shape = tuple(int(s) for s in self.event_shape)
        if len(event_shape) == 0 or event_shape[-1] != d:
            raise ValueError(
                f"Rescale: event_shape must end in the coord dim {d}; "
                f"got event_shape={event_shape}."
            )
        object.__setattr__(self, "event_shape", event_shape)

        # Precomputed constants (numpy -> jnp). Cheap and not traced.
        scale = (tu - tl) / self.geometry.box  # (d,) numpy
        object.__setattr__(
            self, "_lower_j", jnp.asarray(self.geometry.lower, dtype=jnp.float32)
        )
        object.__setattr__(
            self, "_target_lower_j", jnp.asarray(tl, dtype=jnp.float32)
        )
        object.__setattr__(
            self, "_scale_j", jnp.asarray(scale, dtype=jnp.float32)
        )
        event_factor = 1
        for s in event_shape[:-1]:
            event_factor *= int(s)
        object.__setattr__(
            self, "_log_det_fwd", float(event_factor) * float(np.sum(np.log(scale)))
        )

    def forward(
        self, params: Any, x: Array, context: Array | None = None
    ) -> Tuple[Array, Array]:
        del params, context  # Unused; Rescale is non-learnable, non-conditional.
        y = self._target_lower_j + (x - self._lower_j) * self._scale_j
        log_det = jnp.asarray(self._log_det_fwd, dtype=x.dtype)
        return y, log_det

    def inverse(
        self, params: Any, y: Array, context: Array | None = None
    ) -> Tuple[Array, Array]:
        del params, context  # Unused.
        x = self._lower_j + (y - self._target_lower_j) / self._scale_j
        log_det = jnp.asarray(-self._log_det_fwd, dtype=y.dtype)
        return x, log_det

    def init_params(self, key: PRNGKey, context_dim: int = 0) -> dict:
        """No learnable parameters."""
        del key, context_dim
        return {}

    @classmethod
    def create(
        cls,
        key: PRNGKey,
        geometry: Geometry,
        target: Tuple[Any, Any] = (-1.0, 1.0),
        event_shape: Tuple[int, ...] | None = None,
    ) -> Tuple["Rescale", dict]:
        """Factory. Returns `(transform, empty-params)`."""
        transform = cls(geometry=geometry, target=target, event_shape=event_shape)
        return transform, transform.init_params(key)


# ===================================================================
# CoMProjection: (N, d) <-> (N-1, d) translation-gauge projection.
#
# Forward  : x -> y where y_i = x_i - mean(x),  i in [0, N-1)  (drop last)
# Inverse  : y -> x where x_i = y_i for i < N-1, x_{N-1} = -sum(y)
#
# Domains differ in intrinsic dimension: (N, d) ambient vs (N-1, d) reduced.
# This is therefore NOT a bijection on R^(Nd); it is a bijection between
# R^((N-1)d) and the zero-CoM subspace of R^(Nd). The log-det stored on this
# class is **zero in both directions** — see the class docstring for the
# convention, and `CoMProjection.ambient_correction(N, d)` for the constant
# a caller must add when they need an ambient-space log-density.
# ===================================================================
@dataclass
class CoMProjection:
    """Translation-gauge projection: (N, d) <-> (N-1, d).

    Drops the centre-of-mass degree of freedom from a particle
    configuration. Useful when training a flow on a `T(d)`-invariant
    target (most materials Boltzmann generators): the base distribution
    lives on the reduced `(N-1, d)` space and a final `CoMProjection`
    inverse embeds samples back into the ambient zero-CoM subspace.

    Shapes
    ------
    - Forward takes `(..., N, d_axis)` with the particle axis at
      `event_axis` (default `-2`) and returns `(..., N-1, d_axis)`.
    - Inverse takes `(..., N-1, d_axis)` and returns `(..., N, d_axis)`
      whose sum along the particle axis is identically zero.

    Forward behaviour
    -----------------
    Forward **subtracts the per-axis mean along the particle axis
    before dropping the last particle.** An input with non-zero CoM is
    therefore centred; the original CoM is discarded (lossy). For a
    flow round-trip, inputs arriving at `forward` will already be
    zero-CoM if they came from `inverse`.

    ---------------------------------------------------------------
    WARNING — LOG-DET CONVENTION (READ THIS)
    ---------------------------------------------------------------
    The log-det returned by `forward` and `inverse` is **identically
    zero**. This class uses *Convention (1)*: the bijection is treated
    as a relabelling of two `(N-1)d`-dimensional Euclidean spaces. The
    flow produced by composing a `(N-1, d)` base with this bijection's
    inverse yields a density **on the reduced `(N-1, d)` space**.

    If you need a density on the **ambient zero-CoM subspace of
    `R^(Nd)`** (the usual case for reverse-KL training against an
    ambient energy `E(x)`), you must add a constant volume-element
    correction:

        log q_ambient(x) = log q_reduced(y) + (d / 2) * log(N)

    where `y = forward(x)`. The helper is

        CoMProjection.ambient_correction(N, d)  # returns (d/2) * log(N)

    See `REFERENCE.md#comprojection`, `INTERNALS.md` (math derivation),
    and `EXTENDING.md` (when to apply the correction vs. use augmented
    coupling instead) for full guidance.

    When the constant matters
    -------------------------
    - Yes — importance weights / SNIS / ESS / logZ / direct density
      comparisons against an ambient reference measure.
    - No — gradient-based training loss (it is a constant, so it has
      zero gradient). Drop or keep; the optimisation is invariant.

    ---------------------------------------------------------------
    Math (short)
    ---------------------------------------------------------------
    Parameterise the zero-CoM subspace by `y = (x_1, ..., x_{N-1})`,
    `x_N = -sum(y)`. The embedding's per-axis Jacobian matrix `J` has
    `J^T J = I + 1 1^T` whose determinant is `1 + (N-1) = N`. The
    volume scaling is therefore `sqrt(N)` per coordinate axis, and
    `sqrt(N)^d = N^(d/2)` across `d` axes. `(d/2) * log(N)` is the
    constant relating reduced-space and ambient-subspace densities.

    Parameters
    ----------
    event_axis : int, default -2
        Negative axis along which particles are stacked. Must be
        negative and not -1 (the coord axis). For a `(B, N, d)` event
        use the default; for a `(B, species, N, d)` event use `-2` as
        well (still the second-to-last).
    """

    event_axis: int = -2

    def __post_init__(self):
        if self.event_axis >= 0:
            raise ValueError(
                f"CoMProjection: event_axis must be negative (standard trailing-"
                f"axes convention); got {self.event_axis}."
            )
        if self.event_axis == -1:
            raise ValueError(
                f"CoMProjection: event_axis=-1 is the coord axis. "
                f"Use -2 (default) for the particle axis."
            )

    def forward(
        self, params: Any, x: Array, context: Array | None = None
    ) -> Tuple[Array, Array]:
        del params, context  # Unused; CoMProjection is non-learnable.
        mean = jnp.mean(x, axis=self.event_axis, keepdims=True)
        x_centered = x - mean
        # Drop the last particle along event_axis. Using slice_in_dim
        # because event_axis is negative.
        n = x.shape[self.event_axis]
        y = jax.lax.slice_in_dim(x_centered, 0, n - 1, axis=self.event_axis)
        # Convention (1): log-det on the (N-1)d subspace is zero. See
        # class docstring WARNING block for when the caller must add
        # `ambient_correction(N, d) = (d/2) * log(N)`.
        log_det = jnp.zeros((), dtype=x.dtype)
        return y, log_det

    def inverse(
        self, params: Any, y: Array, context: Array | None = None
    ) -> Tuple[Array, Array]:
        del params, context
        last = -jnp.sum(y, axis=self.event_axis, keepdims=True)
        x = jnp.concatenate([y, last], axis=self.event_axis)
        # Same convention; see WARNING in the class docstring.
        log_det = jnp.zeros((), dtype=y.dtype)
        return x, log_det

    def init_params(self, key: PRNGKey, context_dim: int = 0) -> dict:
        """No learnable parameters."""
        del key, context_dim
        return {}

    @classmethod
    def create(
        cls, key: PRNGKey, event_axis: int = -2
    ) -> Tuple["CoMProjection", dict]:
        """Factory. Returns `(transform, empty-params)`."""
        transform = cls(event_axis=event_axis)
        return transform, transform.init_params(key)

    @staticmethod
    def ambient_correction(N: int, d: int) -> float:
        """Constant log-density correction between reduced and ambient measures.

        Returns `(d / 2) * log(N)`, the log of the volume scaling between a
        density on the `(N-1, d)` reduced space and the same density expressed
        on the zero-CoM subspace of `R^(Nd)`. Apply when you need an ambient
        log-density (e.g. reverse-KL training against ambient `E(x)` and
        you care about the absolute value, not just the gradient):

            log_q_ambient = log_q_reduced + CoMProjection.ambient_correction(N, d)

        For gradient-only training, the constant is irrelevant.

        See the class docstring WARNING block for the convention rationale.
        """
        if N <= 1:
            raise ValueError(
                f"CoMProjection.ambient_correction: N must be >= 2, got {N}."
            )
        if d <= 0:
            raise ValueError(
                f"CoMProjection.ambient_correction: d must be >= 1, got {d}."
            )
        return 0.5 * int(d) * math.log(int(N))


# ===================================================================
# Composite Transform: Sequential composition of multiple transforms
# ===================================================================
def _block_supports_gvalue(block: Any) -> bool:
    """Check if a transform block supports the g_value parameter."""
    return isinstance(block, (AffineCoupling, SplineCoupling, LinearTransform, LoftTransform))


@dataclass
class CompositeTransform:
    """
    Sequential composition of multiple transforms.

    Given transforms T_1, T_2, ..., T_n, this object represents the composite
    mapping:
    T(x) = T_n(... T_2(T_1(x)) ...)

    Each block must implement forward(params, x) and inverse(params, y), returning
    the output and the corresponding log-Jacobian determinant.

    Forward propagation:
    y = x
    log_det_total = sum_i log |det ∂T_i/∂(input_i)|
    where the blocks are applied in their listed order.

    Inverse propagation:
    x = y
    log_det_total = sum_i log |det ∂T_i⁻¹/∂(output_i)|
    where the blocks are applied in reverse order.

    Parameters must be a sequence whose length matches that of blocks, where the
    i-th entry contains the parameter PyTree for the i-th transform.
    """
    blocks: List[Any]  # list of AffineCoupling, Permutation, etc

    def forward(
        self,
        params: Sequence[Any],
        x: Array,
        context: Array | None = None,
        g_value: Array | None = None,
    ) -> Tuple[Array, Array]:
        """
        Forward composition: x -> y, applying blocks in order.

        params:
          sequence of parameter objects, one per block.
        x:
          input tensor of shape (..., dim).
        context:
          optional conditioning tensor, passed to all sub-blocks.
        g_value:
          optional gate value for identity_gate. Passed to all sub-blocks that
          support it (couplings, linear). When g_value=0, returns identity.

        Returns:
          y: transformed tensor of shape (..., dim).
          log_det: sum of all block log-dets, shape x.shape[:-1].
        """
        if len(params) != len(self.blocks):
            raise ValueError(
                f"CompositeTransform expected {len(self.blocks)} param sets, "
                f"got {len(params)}."
            )

        y = x
        # Use float64 for log-det accumulation to avoid precision loss in deep flows.
        # Only use float64 if JAX x64 mode is enabled, otherwise fall back silently.
        # Start as scalar zero; each block's log_det has shape = batch_shape and
        # broadcasts the accumulator up. This keeps CompositeTransform agnostic
        # to the event rank (rank-1 flat flows, rank-2 particle flows, ...).
        use_f64 = jax.config.read("jax_enable_x64")
        accum_dtype = jnp.float64 if use_f64 else x.dtype
        log_det_total = jnp.zeros((), dtype=accum_dtype)

        for block, p in zip(self.blocks, params):
            # Pass g_value to blocks that support it (check for keyword argument)
            if g_value is not None and _block_supports_gvalue(block):
                y, log_det = block.forward(p, y, context, g_value=g_value)
            else:
                y, log_det = block.forward(p, y, context)
            log_det_total = log_det_total + log_det.astype(accum_dtype)

        return y, log_det_total.astype(x.dtype)

    def inverse(
        self,
        params: Sequence[Any],
        y: Array,
        context: Array | None = None,
        g_value: Array | None = None,
    ) -> Tuple[Array, Array]:
        """
        Inverse composition: y -> x, applying blocks in reverse order.

        params:
          sequence of parameter objects, one per block (same order as forward).
        y:
          input tensor of shape (..., dim).
        context:
          optional conditioning tensor, passed to all sub-blocks.
        g_value:
          optional gate value for identity_gate. Passed to all sub-blocks that
          support it. When g_value=0, returns identity.

        Returns:
          x: inverse-transformed tensor of shape (..., dim).
          log_det: sum of all block inverse log-dets, shape y.shape[:-1].
        """
        if len(params) != len(self.blocks):
            raise ValueError(
                f"CompositeTransform expected {len(self.blocks)} param sets, "
                f"got {len(params)}."
            )

        x = y
        # Use float64 for log-det accumulation to avoid precision loss in deep flows.
        # Only use float64 if JAX x64 mode is enabled, otherwise fall back silently.
        # Scalar zero initializer; broadcasts to whatever batch shape the blocks return.
        use_f64 = jax.config.read("jax_enable_x64")
        accum_dtype = jnp.float64 if use_f64 else y.dtype
        log_det_total = jnp.zeros((), dtype=accum_dtype)

        # Reverse both blocks and parameter sequence.
        for block, p in zip(reversed(self.blocks), reversed(params)):
            # Pass g_value to blocks that support it
            if g_value is not None and _block_supports_gvalue(block):
                x, log_det = block.inverse(p, x, context, g_value=g_value)
            else:
                x, log_det = block.inverse(p, x, context)
            log_det_total = log_det_total + log_det.astype(accum_dtype)

        return x, log_det_total.astype(y.dtype)

    def init_params(self, key: PRNGKey, context_dim: int = 0) -> list:
        """
        Initialize parameters for all blocks in this composite transform.

        Arguments:
            key: JAX PRNGKey for parameter initialization.
            context_dim: Context dimension (0 for unconditional).

        Returns:
            List of parameter dicts, one per block.
        """
        keys = jax.random.split(key, len(self.blocks))
        params = []
        for k, block in zip(keys, self.blocks):
            if hasattr(block, "init_params"):
                p = block.init_params(k, context_dim=context_dim)
                params.append(p)
            else:
                params.append({})
        return params


# ===================================================================
# LOFT Transform: Coordinate-wise log-soft extension
# ===================================================================
@dataclass
class LoftTransform:
    """
    Coordinate-wise LOFT (log soft extension) transform. It is used to
    stabilize training of normalizing flows in high-dimensional settings.
    This prevents numerical issues arising from extremely small or large
    log-densities in high dimensions by modifying the tails of the
    transformation to be logarithmic instead of linear beyond a threshold.

    Parameters
    ----------
    dim : int
        Feature dimension (size of the last axis).
    tau : float
        Positive threshold where the behavior transitions from linear to
        logarithmic tails.

    Notes
    -----
    - This transform is strictly monotone and C^1 for tau > 0.
    - params is currently unused, kept only for interface compatibility.
      If you later want a learnable tau, you can route it through params.
      
    References
    ----------
    "STABLE TRAINING OF NORMALIZING FLOWS FOR HIGH-DIMENSIONAL VARIATIONAL INFERENCE" by DANIEL ANDRADE
    """
    dim: int
    tau: float

    def __post_init__(self):
        if self.dim <= 0:
            raise ValueError(
                f"LoftTransform: dim must be positive, got {self.dim}."
            )
        if self.tau <= 0.0:
            raise ValueError(
                f"LoftTransform: tau must be strictly positive, got {self.tau}."
            )

    def forward(
        self,
        params: Any,
        x: Array,
        context: Array | None = None,
        g_value: Array | None = None,
    ) -> Tuple[Array, Array]:
        """
        Forward map: x -> y, returning (y, log_det_forward).

        Arguments
        ---------
        params : Any
            Ignored (kept for interface compatibility).
        x : Array
            Input tensor of shape (..., dim).
        context : Array | None
            Ignored (accepted for interface compatibility).
        g_value : Array | None
            Gate value for identity gating. Shape x.shape[:-1].
            When g=0, returns identity. When g=1, returns full LOFT.

        Returns
        -------
        y : Array
            Transformed tensor of shape (..., dim).
        log_det_forward : Array
            log |det ∂y/∂x|, shape x.shape[:-1].
        """
        del context  # Unused in LoftTransform.
        x = jnp.asarray(x)

        if x.shape[-1] != self.dim:
            raise ValueError(
                f"LoftTransform: expected input last dim {self.dim}, "
                f"got {x.shape[-1]}."
            )

        # Forward LOFT (elementwise)
        y_loft = scalar_function.loft(x, self.tau)
        # Elementwise log |loft'(x_i)|
        log_abs_jac = scalar_function.loft_log_abs_det_jac(x, self.tau)

        if g_value is not None:
            g = g_value[..., None]  # (..., 1) for broadcasting over dim
            # Gated forward: y = (1-g)*x + g*loft(x)
            y = (1.0 - g) * x + g * y_loft
            # dy/dx element-wise = (1-g) + g*loft'(x)
            # loft'(x) = exp(log_abs_jac) element-wise
            loft_deriv = jnp.exp(log_abs_jac)
            gated_deriv = (1.0 - g) + g * loft_deriv
            log_det_forward = jnp.sum(jnp.log(jnp.abs(gated_deriv)), axis=-1)
        else:
            y = y_loft
            log_det_forward = jnp.sum(log_abs_jac, axis=-1)

        return y, log_det_forward

    def inverse(
        self,
        params: Any,
        y: Array,
        context: Array | None = None,
        g_value: Array | None = None,
    ) -> Tuple[Array, Array]:
        """
        Inverse map: y -> x, returning (x, log_det_inverse).

        Arguments
        ---------
        params : Any
            Ignored (kept for interface compatibility).
        y : Array
            Input tensor of shape (..., dim).
        context : Array | None
            Ignored (accepted for interface compatibility).
        g_value : Array | None
            Gate value for identity gating. Must match the value used in forward.

        Returns
        -------
        x : Array
            Inverse-transformed tensor of shape (..., dim).
        log_det_inverse : Array
            log |det ∂x/∂y|, shape y.shape[:-1].
        """
        del context  # Unused in LoftTransform.
        y = jnp.asarray(y)

        if y.shape[-1] != self.dim:
            raise ValueError(
                f"LoftTransform: expected input last dim {self.dim}, "
                f"got {y.shape[-1]}."
            )

        if g_value is not None:
            g = g_value[..., None]  # (..., 1)
            # Solve y = (1-g)*x + g*loft(x, tau) for x via Newton iteration.
            # f(x) = (1-g)*x + g*loft(x) - y = 0
            # f'(x) = (1-g) + g*loft'(x)
            # Newton: x_{n+1} = x_n - f(x_n)/f'(x_n)
            x = y  # initial guess (exact when g=0)
            for _ in range(10):
                loft_x = scalar_function.loft(x, self.tau)
                log_jac = scalar_function.loft_log_abs_det_jac(x, self.tau)
                loft_deriv = jnp.exp(log_jac)
                f_val = (1.0 - g) * x + g * loft_x - y
                f_deriv = (1.0 - g) + g * loft_deriv
                x = x - f_val / f_deriv

            # Compute log-det at the converged x
            log_jac_x = scalar_function.loft_log_abs_det_jac(x, self.tau)
            loft_deriv_x = jnp.exp(log_jac_x)
            gated_deriv = (1.0 - g) + g * loft_deriv_x
            log_det_inverse = -jnp.sum(jnp.log(jnp.abs(gated_deriv)), axis=-1)
        else:
            x = scalar_function.loft_inv(y, self.tau)
            log_abs_jac_x = scalar_function.loft_log_abs_det_jac(x, self.tau)
            log_det_inverse = -jnp.sum(log_abs_jac_x, axis=-1)

        return x, log_det_inverse

    def init_params(self, key: PRNGKey, context_dim: int = 0) -> dict:
        """
        Initialize parameters for this transform.

        LoftTransform has no learnable parameters.

        Arguments:
            key: JAX PRNGKey (unused).
            context_dim: Context dimension (unused, included for interface consistency).

        Returns:
            Empty dict.
        """
        del key, context_dim  # Unused.
        return {}

    @classmethod
    def create(
        cls, key: PRNGKey, dim: int, tau: float = 1000.0
    ) -> Tuple["LoftTransform", dict]:
        """
        Factory method to create LoftTransform and initialize params.

        Arguments:
            key: JAX PRNGKey for parameter initialization (unused, for consistency).
            dim: Dimensionality of the transform.
            tau: Threshold parameter for LOFT transition (default: 1000.0).

        Returns:
            Tuple of (transform, params) ready to use.

        Raises:
            ValueError: If dim <= 0 or tau <= 0.

        Example:
            >>> transform, params = LoftTransform.create(key, dim=4, tau=5.0)
            >>> y, log_det = transform.forward(params, x)
        """
        if dim <= 0:
            raise ValueError(f"LoftTransform.create: dim must be positive, got {dim}.")
        if tau <= 0:
            raise ValueError(f"LoftTransform.create: tau must be positive, got {tau}.")

        del key  # Unused for LoftTransform.
        transform = cls(dim=dim, tau=tau)
        params = transform.init_params(None)  # type: ignore
        return transform, params
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Tuple

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsp
from flax import linen as nn

from ..nets import MLP, Array, PRNGKey

# ===================================================================
# Linear Transform with LU-style parameterization
# ===================================================================
@dataclass
class LinearTransform:
    """Global LU-parameterized linear transform.

    The invertible matrix is ``W = L @ (U + diag(softplus(raw_diag + delta)))``,
    with unit-diagonal ``L`` and zero-diagonal ``U``. Unconditional calls apply
    ``y = x W.T``; conditional calls add a context-dependent shift and diagonal
    delta from an MLP. The log determinant is the sum of the log positive
    diagonal entries. Inverses use triangular solves, so no matrix
    factorization is repeated inside forward/inverse.

    Parameter keys are ``lower``, ``upper``, ``raw_diag``, and optionally
    ``mlp`` when ``context_dim > 0``. If ``g_value`` is supplied, the transform
    interpolates to identity at ``g_value=0``.
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

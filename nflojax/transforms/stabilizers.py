from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Tuple

import jax.numpy as jnp

from .. import scalar_function
from ..nets import Array, PRNGKey

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

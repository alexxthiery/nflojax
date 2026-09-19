from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Sequence, Tuple

import jax
import jax.numpy as jnp

from ..nets import Array, PRNGKey
from .couplings import AffineCoupling, SplineCoupling
from .linear import LinearTransform, OrthogonalTransform
from .product import CircularCoordinateShift, ProductSplineCoupling
from .stabilizers import LoftTransform

# ===================================================================
# Composite Transform: Sequential composition of multiple transforms
# ===================================================================
def _block_supports_gvalue(block: Any) -> bool:
    """Check if a transform block supports the g_value parameter."""
    return isinstance(
        block,
        (
            AffineCoupling,
            SplineCoupling,
            ProductSplineCoupling,
            CircularCoordinateShift,
            LinearTransform,
            OrthogonalTransform,
            LoftTransform,
        ),
    )


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

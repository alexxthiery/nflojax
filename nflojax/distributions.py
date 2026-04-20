# nflojax/distributions.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Tuple

import jax
import jax.numpy as jnp

from .nets import Array, PRNGKey


# ----------------------------------------------------------------------
# Shape convention
# ----------------------------------------------------------------------
# An event may have rank >= 1. The canonical representation is a tuple:
# `event_shape = (d1, d2, ..., dk)`. For rank-1 events, passing `dim=N` or
# `event_shape=N` is sugar for `event_shape=(N,)`. A `.dim` property on each
# distribution returns `event_shape[-1]` (the last-axis size) for back-compat
# with the historic rank-1 `dim=int` API and error messages.
def _as_event_tuple(raw: Any) -> Tuple[int, ...]:
    return (raw,) if isinstance(raw, int) else tuple(int(d) for d in raw)


# ----------------------------------------------------------------------
# Standard Normal
# ----------------------------------------------------------------------
@dataclass(init=False)
class StandardNormal:
    """
    Isotropic Gaussian N(0, I) on an event of shape `event_shape`.

    `x` is expected to have shape `(*batch, *event_shape)`; `log_prob`
    returns shape `batch`. `params` is ignored (no learnable parameters).

    Construct with either form:
      StandardNormal(event_shape=(N, d))   # rank-N
      StandardNormal(event_shape=N)        # rank-1 (int is sugar)
      StandardNormal(dim=N)                # legacy; equivalent to the above
    """
    event_shape: Tuple[int, ...]

    def __init__(self, event_shape: Any = None, *, dim: Any = None):
        if event_shape is None and dim is None:
            raise TypeError("StandardNormal requires `event_shape` or `dim`.")
        raw = event_shape if event_shape is not None else dim
        self.event_shape = _as_event_tuple(raw)

    @property
    def dim(self) -> int:
        """Last-axis size of event_shape; kept for back-compat."""
        return self.event_shape[-1]

    def _check_event_shape(self, x: Array) -> None:
        rank = len(self.event_shape)
        if x.shape[-rank:] != self.event_shape:
            raise ValueError(
                f"StandardNormal: expected trailing event_shape {self.event_shape}, "
                f"got {x.shape[-rank:]}"
            )

    def log_prob(self, params: Any, x: Array) -> Array:
        self._check_event_shape(x)
        axes = tuple(range(-len(self.event_shape), 0))
        quad = jnp.sum(x * x, axis=axes)
        log_norm = 0.5 * math.prod(self.event_shape) * jnp.log(2.0 * jnp.pi)
        return -0.5 * quad - log_norm

    def sample(self, params: Any, key: PRNGKey, shape: Tuple[int, ...]) -> Array:
        return jax.random.normal(key, shape=shape + self.event_shape)

    def init_params(self) -> None:
        """StandardNormal has no learnable parameters."""
        return None


# ----------------------------------------------------------------------
# Diagonal Gaussian
# ----------------------------------------------------------------------
@dataclass(init=False)
class DiagNormal:
    """
    Diagonal-covariance Gaussian on an event of shape `event_shape`.

    Required params leaves:
      params["loc"]       shape event_shape
      params["log_scale"] shape event_shape

    `log_prob` sums over all event axes and returns shape `batch`.
    Construction mirrors StandardNormal (accepts `event_shape` or `dim`).
    """
    event_shape: Tuple[int, ...]

    def __init__(self, event_shape: Any = None, *, dim: Any = None):
        if event_shape is None and dim is None:
            raise TypeError("DiagNormal requires `event_shape` or `dim`.")
        raw = event_shape if event_shape is not None else dim
        self.event_shape = _as_event_tuple(raw)

    @property
    def dim(self) -> int:
        """Last-axis size of event_shape; kept for back-compat."""
        return self.event_shape[-1]

    def _extract_params(self, params: Any) -> Tuple[Array, Array]:
        try:
            loc = jnp.asarray(params["loc"])
            log_scale = jnp.asarray(params["log_scale"])
        except Exception as e:
            raise KeyError(
                "DiagNormal expected params to contain 'loc' and 'log_scale'"
            ) from e

        if loc.shape != self.event_shape:
            raise ValueError(
                f"DiagNormal: loc must have shape {self.event_shape}, got {loc.shape}"
            )
        if log_scale.shape != self.event_shape:
            raise ValueError(
                f"DiagNormal: log_scale must have shape {self.event_shape}, got {log_scale.shape}"
            )

        return loc, log_scale

    def _check_event_shape(self, x: Array) -> None:
        rank = len(self.event_shape)
        if x.shape[-rank:] != self.event_shape:
            raise ValueError(
                f"DiagNormal: expected trailing event_shape {self.event_shape}, "
                f"got {x.shape[-rank:]}"
            )

    def log_prob(self, params: Any, x: Array) -> Array:
        self._check_event_shape(x)
        loc, log_scale = self._extract_params(params)
        scale = jnp.exp(log_scale)

        z = (x - loc) / scale
        axes = tuple(range(-len(self.event_shape), 0))
        quad = jnp.sum(z * z, axis=axes)
        log_norm = 0.5 * math.prod(self.event_shape) * jnp.log(2.0 * jnp.pi)
        log_det = jnp.sum(log_scale)  # scalar; same for every batch element

        return -0.5 * quad - log_norm - log_det

    def sample(self, params: Any, key: PRNGKey, shape: Tuple[int, ...]) -> Array:
        loc, log_scale = self._extract_params(params)
        scale = jnp.exp(log_scale)
        eps = jax.random.normal(key, shape=shape + self.event_shape)
        return loc + eps * scale

    def init_params(self) -> dict:
        """Zero loc and log_scale; evaluates to StandardNormal on the same event."""
        return {
            "loc": jnp.zeros(self.event_shape),
            "log_scale": jnp.zeros(self.event_shape),
        }

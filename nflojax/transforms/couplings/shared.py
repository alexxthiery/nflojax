from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Any, Callable, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from ...nets import MLP, Array, PRNGKey, validate_conditioner
from ...splines import rational_quadratic_spline
from ..common import (
    _params_per_scalar,
    _validate_boundary_slopes,
    identity_spline_bias,
    stable_logit,
)

__all__ = [
    "Any",
    "Callable",
    "Sequence",
    "Tuple",
    "dataclass",
    "math",
    "warnings",
    "jax",
    "jnp",
    "nn",
    "MLP",
    "Array",
    "PRNGKey",
    "validate_conditioner",
    "rational_quadratic_spline",
    "_params_per_scalar",
    "_validate_boundary_slopes",
    "identity_spline_bias",
    "stable_logit",
]

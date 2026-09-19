# nflojax/transforms/__init__.py
"""Public transform facade.

Implementation lives in focused sibling modules. This facade preserves the
historical import path::

    from nflojax.transforms import SplineCoupling, CompositeTransform
"""
from __future__ import annotations

from .common import (
    validate_identity_gate,
    stable_logit,
    identity_spline_bias,
)
from .linear import LinearTransform, OrthogonalTransform
from .couplings import AffineCoupling, SplineCoupling, SplitCoupling
from .geometry import Permutation, CircularShift, Rescale, CoMProjection
from .product import CircularCoordinateShift, ProductSplineCoupling
from .composition import CompositeTransform
from .stabilizers import LoftTransform

__all__ = [
    "validate_identity_gate",
    "stable_logit",
    "identity_spline_bias",
    "LinearTransform",
    "OrthogonalTransform",
    "AffineCoupling",
    "SplineCoupling",
    "SplitCoupling",
    "Permutation",
    "CircularShift",
    "Rescale",
    "CoMProjection",
    "CircularCoordinateShift",
    "ProductSplineCoupling",
    "CompositeTransform",
    "LoftTransform",
]

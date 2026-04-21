# nflojax/geometry.py
"""
Box geometry for particle-flow primitives.

This module defines `Geometry`, a value object that encapsulates per-axis
box bounds and per-axis periodicity flags. Multiple bijections and base
distributions consume it (`CircularShift`, future `Rescale`, `UniformBox`,
`LatticeBase`, `utils.pbc.nearest_image`); having one source of truth
avoids each primitive growing its own `lower`/`upper`/`box` plumbing.

What `Geometry` IS
------------------
- An **axis-aligned rectangular domain in R^d** with per-axis periodicity.
- Covers: cubic boxes, rectangular (orthogonal) boxes, fully periodic
  torus, slab geometries (some axes periodic, others open), fully open
  domains.
- Numpy-backed configuration, not a PyTree. Fields are constants; they
  are not traced by JAX transforms and do not participate in params.

What `Geometry` is NOT
----------------------
- NOT a general differential-geometric object: no metric, no curvature,
  no coordinate chart, no `Manifold` interface.
- NOT triclinic: there is no cell matrix. `box = upper - lower` assumes
  the axes are orthogonal. Triclinic / non-orthogonal cells would
  require a separate type (or an extended `Geometry` with a `cell` field,
  at which point the `box` property needs reinterpretation).
- NOT curved: spheres, non-rectangular tori, configuration manifolds of
  molecules (torsions on S^1) do not fit.
- NOT dynamic: there is no time-dependence, no deforming box (NPT with
  learnable volume lives elsewhere).

The name is mildly aspirational — `AxisAlignedBox` or `BoxSpec` would be
more literal — but we keep `Geometry` so the type can grow (via sibling
classes or added fields) when those extensions become real.

Conventions
-----------
- `lower, upper`: 1D arrays of shape `(d,)`. Scalar inputs are broadcast
  in the `Geometry.cubic` factory but the stored value is always 1D.
- `periodic`: 1D bool array of shape `(d,)`, or `None`. `None` is
  interpreted as "all axes periodic", which matches the common PBC case.
- Derived quantities: `box = upper - lower`, `d = len(lower)`,
  `volume = prod(box)`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np


# Geometry carries plain numpy arrays, not jnp, because the box is a
# *constant* specification (not a traced value) and downstream consumers
# materialise jnp arrays from it at call time. Using numpy here keeps the
# dataclass cheap to construct and hashable.
Array = np.ndarray


@dataclass(frozen=True)
class Geometry:
    """Per-axis box bounds + periodicity flags.

    Fields:
      lower, upper: `(d,)` arrays; the box is `[lower, upper]` per axis.
      periodic: `(d,)` bool array, or `None` (interpreted as all-periodic).

    Prefer one of the factory classmethods over the raw constructor:
      Geometry.cubic(d=3, side=1.0, lower=-0.5)           # cubic box
      Geometry.box(lower=[-1, -1, -1], upper=[1, 1, 2])   # rectangular
      Geometry.box(..., periodic=[True, True, False])     # slab (1 open axis)
    """
    lower: Any
    upper: Any
    periodic: Any = None

    def __post_init__(self):
        lower = np.asarray(self.lower, dtype=np.float32)
        upper = np.asarray(self.upper, dtype=np.float32)
        if lower.ndim != 1:
            raise ValueError(
                f"Geometry.lower must be 1D, got shape {lower.shape}. "
                f"Use `Geometry.cubic(d=...)` for a cubic box."
            )
        if upper.ndim != 1:
            raise ValueError(
                f"Geometry.upper must be 1D, got shape {upper.shape}."
            )
        if lower.shape != upper.shape:
            raise ValueError(
                f"Geometry: lower.shape {lower.shape} != upper.shape {upper.shape}."
            )
        if np.any(lower >= upper):
            raise ValueError(
                f"Geometry: lower must be strictly < upper element-wise; "
                f"got lower={lower}, upper={upper}."
            )
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)
        if self.periodic is not None:
            periodic = np.asarray(self.periodic, dtype=bool)
            if periodic.shape != lower.shape:
                raise ValueError(
                    f"Geometry: periodic.shape {periodic.shape} "
                    f"!= lower.shape {lower.shape}."
                )
            object.__setattr__(self, "periodic", periodic)

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------
    @property
    def d(self) -> int:
        """Spatial dimensionality."""
        return int(self.lower.shape[0])

    @property
    def box(self) -> Array:
        """Per-axis box length, shape `(d,)`."""
        return self.upper - self.lower

    @property
    def volume(self) -> float:
        """Total box volume (product of axis lengths)."""
        return float(np.prod(self.box))

    def is_periodic(self, axis: Optional[int] = None) -> bool:
        """True if the given axis (or every axis when `axis is None`) is periodic.

        With the default `periodic=None` the geometry is all-periodic.
        """
        if self.periodic is None:
            return True
        if axis is None:
            return bool(np.all(self.periodic))
        return bool(self.periodic[axis])

    # ------------------------------------------------------------------
    # Factory classmethods
    # ------------------------------------------------------------------
    @classmethod
    def cubic(cls, d: int, side: float = 1.0, lower: float = -0.5) -> "Geometry":
        """Cubic box with uniform side length.

        Example:
          >>> Geometry.cubic(d=3)                   # [-0.5, 0.5]^3
          >>> Geometry.cubic(d=3, side=2.0)         # [-1.0, 1.0]^3
          >>> Geometry.cubic(d=3, side=L, lower=0)  # [0, L]^3
        """
        if d <= 0:
            raise ValueError(f"Geometry.cubic: d must be positive, got {d}.")
        if side <= 0:
            raise ValueError(f"Geometry.cubic: side must be positive, got {side}.")
        lo = np.full((d,), float(lower), dtype=np.float32)
        up = lo + float(side)
        return cls(lower=lo, upper=up)

    # NOTE: A `Geometry.box(lower, upper, periodic)` factory intentionally
    # does NOT exist. It would shadow the `@property box` (per-axis box
    # lengths). Use the bare constructor instead:
    #     Geometry(lower=[0, 0, 0], upper=[Lx, Ly, Lz], periodic=[...])

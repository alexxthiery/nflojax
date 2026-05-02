from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from ..geometry import Geometry
from ..nets import Array, PRNGKey

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
        """Return a zero log-det with batch shape = `x.shape[:event_axis]`.

        Per DESIGN.md §5.5, log-det carries batch shape only (all axes at or
        after ``event_axis`` are event axes). ``event_axis`` is negative, so
        ``x.shape[:event_axis]`` drops the permuted axis *and* every axis to
        its right. For rank-1 ``(B, dim)`` with ``event_axis=-1`` this is
        ``(B,)``; for rank-2 ``(B, N, d)`` with ``event_axis=-2`` this is
        also ``(B,)``.
        """
        return jnp.zeros(x.shape[: self.event_axis], dtype=x.dtype)

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

# nflojax/distributions.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from .geometry import Geometry
from .nets import Array, PRNGKey
from .utils import lattice as _lattice


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


# ----------------------------------------------------------------------
# UniformBox
# ----------------------------------------------------------------------
@dataclass(init=False)
class UniformBox:
    """Per-axis uniform base distribution on `geometry.box`.

    For event_shape `(d,)` each sample is a single point uniform in the box.
    For event_shape `(N, d)` each sample is `N` i.i.d. uniform points.
    `event_shape[-1]` must equal `geometry.d`; leading event axes (e.g. the
    particle axis) contribute `event_factor = prod(event_shape[:-1])` to the
    constant log-density.

    log_prob(x)
    -----------
      -inf                             if any coord of x lies outside the box
      -event_factor * sum(log(box_j))  otherwise (constant; broadcasts)

    sample(key, shape)
    ------------------
      jax.random.uniform in per-axis `[lower, upper]`, returning
      `shape + event_shape`. i.i.d. across every leading batch and event axis.

    Non-learnable — `init_params()` returns `None`.
    """
    geometry: Geometry
    event_shape: Tuple[int, ...]

    def __init__(self, geometry: Geometry, event_shape: Any):
        if not isinstance(geometry, Geometry):
            raise TypeError(
                f"UniformBox: geometry must be a Geometry instance, "
                f"got {type(geometry).__name__}."
            )
        shape = _as_event_tuple(event_shape)
        d = geometry.d
        if len(shape) == 0 or shape[-1] != d:
            raise ValueError(
                f"UniformBox: event_shape must end in the coord dim {d}; "
                f"got event_shape={shape}."
            )
        self.geometry = geometry
        self.event_shape = shape

        # Precompute jnp constants for the hot path.
        self._lower_j = jnp.asarray(geometry.lower, dtype=jnp.float32)
        self._upper_j = jnp.asarray(geometry.upper, dtype=jnp.float32)
        box = jnp.asarray(geometry.box, dtype=jnp.float32)
        event_factor = 1
        for s in shape[:-1]:
            event_factor *= int(s)
        self._event_factor = event_factor
        self._log_norm = -float(event_factor) * float(jnp.sum(jnp.log(box)))

    @property
    def d(self) -> int:
        """Last-axis size of event_shape; kept for back-compat."""
        return self.event_shape[-1]

    def _check_event_shape(self, x: Array) -> None:
        rank = len(self.event_shape)
        if x.shape[-rank:] != self.event_shape:
            raise ValueError(
                f"UniformBox: expected trailing event_shape {self.event_shape}, "
                f"got {x.shape[-rank:]}"
            )

    def log_prob(self, params: Any, x: Array) -> Array:
        del params  # Unused; non-learnable.
        self._check_event_shape(x)
        # in-box test: reduce over every event axis.
        axes = tuple(range(-len(self.event_shape), 0))
        in_box = jnp.all(
            (x >= self._lower_j) & (x <= self._upper_j), axis=axes
        )
        log_norm = jnp.asarray(self._log_norm, dtype=x.dtype)
        neg_inf = jnp.asarray(-jnp.inf, dtype=x.dtype)
        return jnp.where(in_box, log_norm, neg_inf)

    def sample(self, params: Any, key: PRNGKey, shape: Tuple[int, ...]) -> Array:
        del params
        return jax.random.uniform(
            key,
            shape=shape + self.event_shape,
            minval=self._lower_j,
            maxval=self._upper_j,
        )

    def init_params(self) -> None:
        """UniformBox has no learnable parameters."""
        return None


# ----------------------------------------------------------------------
# LatticeBase
# ----------------------------------------------------------------------
@dataclass(init=False)
class LatticeBase:
    """Gaussian-perturbed lattice base distribution for crystalline solids.

    Each particle `i` is drawn from `N(positions[i], noise_scale^2 * I_d)`,
    independently across particles. The lattice sites `positions: (N, d)`
    are static configuration data (numpy at construction; cast to jnp at
    call time).

    Indistinguishability via `permute`
    ---------------------------------
    With `permute=True`:
      - `sample` shuffles the particle axis per batch sample (so the user
        cannot infer which site each output came from);
      - `log_prob` subtracts `log(N!)` from the labelled Gaussian density,
        the standard distinguishable -> indistinguishable correction.
    The constant doesn't affect gradients of the training loss (zero
    gradient) but matters for absolute densities, ESS, `logZ` -- same
    bookkeeping issue documented for `CoMProjection.ambient_correction`.

    The labelled Gaussian density itself is not invariant under permuting
    `x`'s particle order; the `-log(N!)` correction makes the *expectation
    over uniform-permutation augmentations* match the indistinguishable
    convention. For high-noise regimes where multiple permutations have
    comparable likelihood, this is a *single-dominant-permutation*
    approximation.

    Factories
    ---------
    Use the named factories instead of building positions by hand:

        LatticeBase.fcc(n_cells=2, a=1.5, noise_scale=0.1)
        LatticeBase.diamond(n_cells=(3, 3, 3), a=1.0, noise_scale=0.05)
        LatticeBase.bcc(n_cells=4, a=1.2, noise_scale=0.08)
        LatticeBase.hcp(n_cells=2, a=1.0, noise_scale=0.1)
        LatticeBase.hex_ice(n_cells=2, a=1.0, noise_scale=0.1)

    Each builds the lattice via `nflojax.utils.lattice` and a `Geometry`
    with origin at zero whose extent is `n_cells * a * cell_aspect`.
    """
    positions: Array
    geometry: Geometry
    noise_scale: float
    permute: bool

    def __init__(
        self,
        positions: Any,
        geometry: Geometry,
        noise_scale: float,
        permute: bool = False,
    ):
        if not isinstance(geometry, Geometry):
            raise TypeError(
                f"LatticeBase: geometry must be a Geometry instance, "
                f"got {type(geometry).__name__}."
            )
        pos = jnp.asarray(positions, dtype=jnp.float32)
        if pos.ndim != 2:
            raise ValueError(
                f"LatticeBase: positions must be 2-D (N, d), got shape {pos.shape}."
            )
        if pos.shape[-1] != geometry.d:
            raise ValueError(
                f"LatticeBase: positions.shape[-1]={pos.shape[-1]} must equal "
                f"geometry.d={geometry.d}."
            )
        if noise_scale <= 0:
            raise ValueError(
                f"LatticeBase: noise_scale must be positive, got {noise_scale}."
            )
        self.positions = pos
        self.geometry = geometry
        self.noise_scale = float(noise_scale)
        self.permute = bool(permute)

        # Cache the indistinguishability constant log(N!) = lgamma(N+1).
        self._log_n_factorial = float(math.lgamma(self.N + 1))

    @property
    def event_shape(self) -> Tuple[int, ...]:
        return tuple(self.positions.shape)

    @property
    def N(self) -> int:
        return int(self.positions.shape[0])

    @property
    def d(self) -> int:
        return int(self.positions.shape[1])

    def _check_event_shape(self, x: Array) -> None:
        if x.shape[-2:] != self.event_shape:
            raise ValueError(
                f"LatticeBase: expected trailing event_shape {self.event_shape}, "
                f"got {x.shape[-2:]}."
            )

    def log_prob(self, params: Any, x: Array) -> Array:
        del params  # Non-learnable.
        self._check_event_shape(x)
        z = (x - self.positions) / self.noise_scale
        quad = jnp.sum(z * z, axis=(-2, -1))
        nd = self.N * self.d
        log_norm = 0.5 * nd * jnp.log(2.0 * jnp.pi) + nd * jnp.log(self.noise_scale)
        log_p = -0.5 * quad - log_norm
        if self.permute:
            log_p = log_p - self._log_n_factorial
        return log_p

    def sample(self, params: Any, key: PRNGKey, shape: Tuple[int, ...]) -> Array:
        del params
        if not self.permute:
            eps = jax.random.normal(key, shape=shape + self.event_shape)
            return self.positions + self.noise_scale * eps

        # permute=True: per-batch random shuffle of the particle axis.
        n_total = 1
        for s in shape:
            n_total *= int(s)
        keys = jax.random.split(key, n_total + 1)
        noise_key, perm_keys = keys[0], keys[1:]
        eps = jax.random.normal(noise_key, shape=shape + self.event_shape)
        x = self.positions + self.noise_scale * eps
        if n_total == 0:
            return x
        x_flat = x.reshape((n_total,) + self.event_shape)
        # One independent permutation per flat sample.
        N = self.N
        perms = jax.vmap(lambda k: jax.random.permutation(k, N))(perm_keys)
        x_perm = jax.vmap(lambda xx, p: xx[p])(x_flat, perms)
        return x_perm.reshape(shape + self.event_shape)

    def init_params(self) -> None:
        """LatticeBase has no learnable parameters."""
        return None

    # ------------------------------------------------------------------
    # Factories (one per shipped lattice in nflojax.utils.lattice)
    # ------------------------------------------------------------------
    @classmethod
    def _from_lattice(
        cls,
        name: str,
        n_cells,
        a: float,
        noise_scale: float,
        permute: bool,
    ) -> "LatticeBase":
        gen = getattr(_lattice, name)
        positions = gen(n_cells, a)
        box = _lattice.make_box(n_cells, a, _lattice.cell_aspect(name))
        geom = Geometry(lower=np.zeros(3), upper=box)
        return cls(
            positions=positions,
            geometry=geom,
            noise_scale=noise_scale,
            permute=permute,
        )

    @classmethod
    def fcc(cls, n_cells, a: float, noise_scale: float, permute: bool = False):
        """Face-centred cubic lattice, 4 atoms / cubic cell."""
        return cls._from_lattice("fcc", n_cells, a, noise_scale, permute)

    @classmethod
    def diamond(cls, n_cells, a: float, noise_scale: float, permute: bool = False):
        """Diamond cubic lattice, 8 atoms / cubic cell."""
        return cls._from_lattice("diamond", n_cells, a, noise_scale, permute)

    @classmethod
    def bcc(cls, n_cells, a: float, noise_scale: float, permute: bool = False):
        """Body-centred cubic lattice, 2 atoms / cubic cell."""
        return cls._from_lattice("bcc", n_cells, a, noise_scale, permute)

    @classmethod
    def hcp(cls, n_cells, a: float, noise_scale: float, permute: bool = False):
        """Hexagonal close-packed lattice (orthorhombic), 4 atoms / cell, ideal c/a."""
        return cls._from_lattice("hcp", n_cells, a, noise_scale, permute)

    @classmethod
    def hex_ice(cls, n_cells, a: float, noise_scale: float, permute: bool = False):
        """Hexagonal ice (Ice Ih), DM convention, 8 atoms / orthorhombic cell."""
        return cls._from_lattice("hex_ice", n_cells, a, noise_scale, permute)

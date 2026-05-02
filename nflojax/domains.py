"""Flat product-domain specifications for mixed-coordinate flows.

The objects in this module describe coordinate domains only. They do not know
about targets, energies, particles, or application semantics. A product domain
is a rank-1 event space whose coordinates can be real-valued, bounded to an
interval, or circular on an interval with identified endpoints.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence, Tuple

import jax.numpy as jnp
import numpy as np

from .nets import Array


_DOMAIN_KINDS = ("real", "interval", "circular")


@dataclass(frozen=True)
class ScalarDomain:
    """Domain specification for one scalar coordinate.

    Attributes:
        kind: One of ``"real"``, ``"interval"``, or ``"circular"``.
        lower: Lower endpoint for interval/circular coordinates. ``None`` for
            real coordinates.
        upper: Upper endpoint for interval/circular coordinates. ``None`` for
            real coordinates.
    """

    kind: str
    lower: float | None = None
    upper: float | None = None

    def __post_init__(self) -> None:
        if self.kind not in _DOMAIN_KINDS:
            raise ValueError(
                f"ScalarDomain.kind must be one of {_DOMAIN_KINDS}, got {self.kind!r}."
            )
        if self.kind == "real":
            if self.lower is not None or self.upper is not None:
                raise ValueError("ScalarDomain.real() must not have bounds.")
            return
        if self.lower is None or self.upper is None:
            raise ValueError(f"ScalarDomain.{self.kind} requires lower and upper.")
        lower = float(self.lower)
        upper = float(self.upper)
        if not math.isfinite(lower) or not math.isfinite(upper):
            raise ValueError("ScalarDomain bounds must be finite.")
        if lower >= upper:
            raise ValueError(
                f"ScalarDomain lower must be < upper; got lower={lower}, upper={upper}."
            )
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)

    @classmethod
    def real(cls) -> "ScalarDomain":
        """Return an unconstrained real-line coordinate domain."""

        return cls(kind="real")

    @classmethod
    def interval(cls, lower: Any, upper: Any) -> "ScalarDomain":
        """Return a bounded interval coordinate domain ``[lower, upper]``."""

        return cls(kind="interval", lower=float(lower), upper=float(upper))

    @classmethod
    def circular(cls, lower: Any, upper: Any) -> "ScalarDomain":
        """Return a circular coordinate domain with identified endpoints."""

        return cls(kind="circular", lower=float(lower), upper=float(upper))


@dataclass(frozen=True)
class ProductDomain:
    """Flat product of scalar coordinate domains.

    A product domain is coordinate-topology metadata for flat rank-1 events.
    It does not encode target, molecule, energy, or optimizer semantics. The
    ``conditioner_features`` helper below is the shipped default feature map
    for MLP-based product-domain conditioners, not an intrinsic domain rule.

    Args:
        domains: Non-empty sequence of ``ScalarDomain`` instances.

    Attributes:
        domains: Tuple of scalar domains. The event shape is ``(len(domains),)``.
    """

    domains: Sequence[ScalarDomain]

    def __post_init__(self) -> None:
        domains = tuple(self.domains)
        if not domains:
            raise ValueError("ProductDomain requires at least one scalar domain.")
        for i, domain in enumerate(domains):
            if not isinstance(domain, ScalarDomain):
                raise TypeError(
                    f"ProductDomain entries must be ScalarDomain instances; "
                    f"entry {i} has type {type(domain).__name__}."
                )
        object.__setattr__(self, "domains", domains)
        object.__setattr__(
            self,
            "_lower",
            np.asarray(
                [0.0 if d.lower is None else d.lower for d in domains],
                dtype=np.float64,
            ),
        )
        object.__setattr__(
            self,
            "_upper",
            np.asarray(
                [1.0 if d.upper is None else d.upper for d in domains],
                dtype=np.float64,
            ),
        )

    @property
    def dim(self) -> int:
        """Number of scalar coordinates."""

        return len(self.domains)

    @property
    def event_shape(self) -> tuple[int, ...]:
        """Rank-1 event shape."""

        return (self.dim,)

    @property
    def has_circular(self) -> bool:
        """Whether at least one coordinate is circular."""

        return any(domain.kind == "circular" for domain in self.domains)

    def _check_x(self, x: Array) -> None:
        if x.shape[-1:] != self.event_shape:
            raise ValueError(
                f"ProductDomain: expected trailing event_shape {self.event_shape}, "
                f"got {x.shape[-1:]}."
            )

    def _check_mask(self, mask: Array) -> tuple[bool, ...]:
        mask_np = np.asarray(mask, dtype=np.float32)
        if mask_np.shape != (self.dim,):
            raise ValueError(
                f"ProductDomain mask must have shape ({self.dim},), got {mask_np.shape}."
            )
        if not np.all((mask_np == 0.0) | (mask_np == 1.0)):
            raise ValueError("ProductDomain mask entries must be 0 or 1.")
        return tuple(bool(v) for v in mask_np)

    def params_per_scalar(self, num_bins: int) -> tuple[int, ...]:
        """Return spline parameter count per coordinate for ``num_bins`` bins."""

        if num_bins <= 0:
            raise ValueError(f"num_bins must be positive, got {num_bins}.")
        return tuple(
            3 * num_bins if domain.kind == "circular" else 3 * num_bins - 1
            for domain in self.domains
        )

    def required_out_dim(self, mask: Array, num_bins: int) -> int:
        """Return conditioner output width for transformed coordinates."""

        mask_tuple = self._check_mask(mask)
        per_scalar = self.params_per_scalar(num_bins)
        return int(
            sum(size for frozen, size in zip(mask_tuple, per_scalar) if not frozen)
        )

    def conditioner_feature_dim(self, mask: Array, circular_n_freq: int = 1) -> int:
        """Return default MLP feature width for frozen coordinates."""

        if circular_n_freq <= 0:
            raise ValueError("circular_n_freq must be positive.")
        mask_tuple = self._check_mask(mask)
        width = 0
        for frozen, domain in zip(mask_tuple, self.domains):
            if not frozen:
                continue
            width += 2 * circular_n_freq if domain.kind == "circular" else 1
        return width

    def conditioner_features(
        self,
        x: Array,
        mask: Array,
        circular_n_freq: int = 1,
    ) -> Array:
        """Map frozen coordinates to conditioner features.

        This is the default feature map used by product-domain MLP
        conditioners, not a universal embedding policy. Real coordinates are
        passed through. Interval coordinates are normalized to ``[-1, 1]``.
        Circular coordinates are represented by sine/cosine features for
        frequencies ``1, ..., circular_n_freq``.

        Args:
            x: Array with shape ``(*batch, dim)``.
            mask: Array with shape ``(dim,)`` where ``1`` marks frozen
                conditioner-input coordinates.
            circular_n_freq: Number of sine/cosine frequency pairs per frozen
                circular coordinate.

        Returns:
            Feature array with shape
            ``(*batch, conditioner_feature_dim(mask, circular_n_freq))``.
        """

        self._check_x(x)
        if circular_n_freq <= 0:
            raise ValueError("circular_n_freq must be positive.")
        mask_tuple = self._check_mask(mask)
        features = []
        for i, (frozen, domain) in enumerate(zip(mask_tuple, self.domains)):
            if not frozen:
                continue
            x_i = x[..., i : i + 1]
            if domain.kind == "real":
                features.append(x_i)
            elif domain.kind == "interval":
                lower = jnp.asarray(domain.lower, dtype=x.dtype)
                upper = jnp.asarray(domain.upper, dtype=x.dtype)
                features.append(2.0 * (x_i - lower) / (upper - lower) - 1.0)
            else:
                lower = jnp.asarray(domain.lower, dtype=x.dtype)
                upper = jnp.asarray(domain.upper, dtype=x.dtype)
                phase = 2.0 * jnp.pi * (x_i - lower) / (upper - lower)
                for freq in range(1, circular_n_freq + 1):
                    features.append(jnp.sin(float(freq) * phase))
                    features.append(jnp.cos(float(freq) * phase))
        if not features:
            return jnp.zeros(x.shape[:-1] + (0,), dtype=x.dtype)
        return jnp.concatenate(features, axis=-1)

    def bounds_arrays(self, dtype=None) -> Tuple[Array, Array]:
        """Return lower and upper arrays with shape ``(dim,)``."""

        if dtype is None:
            dtype = jnp.asarray(0.0).dtype
        return jnp.asarray(self._lower, dtype=dtype), jnp.asarray(self._upper, dtype=dtype)

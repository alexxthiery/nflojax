# nflojax/embeddings.py
"""
Stateless feature transforms for conditioner inputs.

Two pure functions, both jit-friendly, used by Stage D conditioners
(Transformer, GNN) and by any user-supplied conditioner that wants
ready-made periodic / scalar features:

- `circular_embed(x, geometry, n_freq)` -- per-coord Fourier features
  on a periodic box.
- `positional_embed(t, n_freq, base=10_000)` -- sinusoidal scalar
  context embedding (transformer-style, adapted to continuous `t`).

Both are stateless: no learnable parameters, no random state. For
learnable embeddings, use a Flax `nn.Embed` inside a custom
conditioner in `nets.py` (see DESIGN.md §8).
"""
from __future__ import annotations

import jax.numpy as jnp

from .geometry import Geometry
from .nets import Array


def circular_embed(x: Array, geometry: Geometry, n_freq: int) -> Array:
    """Per-coord Fourier feature embedding on a periodic box.

    For each coord axis ``j``, emit harmonics ``1`` through ``n_freq``
    (loop index ``k in [0, n_freq)``)::

        phase[..., j, k] = 2 * pi * (k + 1) * (x[..., j] - lower[j]) / box[j]
        out[..., j, 2k]     = cos(phase[..., j, k])
        out[..., j, 2k + 1] = sin(phase[..., j, k])

    Output is flattened along the last two axes, so the embedded last
    axis grows from ``d`` to ``d * 2 * n_freq``. Periodic in each coord
    axis with period ``geometry.box[j]`` (so the lowest harmonic
    exactly tiles the box once).

    Arguments:
        x:        Coord tensor of shape ``(..., d)`` where
                  ``d == geometry.d``.
        geometry: Box defining the per-axis period. Non-periodic axes
                  (per ``geometry.periodic``) are not gated -- the math
                  is well-defined for any ``box[j]``, but using
                  ``circular_embed`` on a non-periodic axis is the
                  caller's choice and usually wrong.
        n_freq:   Number of cos/sin pairs per coord (must be >= 1).

    Returns:
        Array of shape ``(..., d * 2 * n_freq)``.
    """
    if not isinstance(geometry, Geometry):
        raise TypeError(
            f"circular_embed: geometry must be a Geometry instance, "
            f"got {type(geometry).__name__}."
        )
    if n_freq < 1:
        raise ValueError(
            f"circular_embed: n_freq must be >= 1, got {n_freq}. "
            f"(Zero-frequency features would silently produce a "
            f"zero-width last axis and break downstream concat.)"
        )
    if x.shape[-1] != geometry.d:
        raise ValueError(
            f"circular_embed: x last axis must equal geometry.d="
            f"{geometry.d}, got x.shape={x.shape}."
        )

    lower = jnp.asarray(geometry.lower, dtype=x.dtype)  # (d,)
    box = jnp.asarray(geometry.box, dtype=x.dtype)      # (d,)
    k = jnp.arange(1, n_freq + 1, dtype=x.dtype)        # (n_freq,)

    # phase shape: (..., d, n_freq)
    scaled = (x - lower) / box                          # (..., d)
    phase = (2.0 * jnp.pi) * scaled[..., None] * k      # (..., d, n_freq)

    cos = jnp.cos(phase)
    sin = jnp.sin(phase)
    # Interleave cos / sin along a new last axis: (..., d, n_freq, 2)
    pairs = jnp.stack([cos, sin], axis=-1)
    # Flatten the trailing (d, n_freq, 2) into (d * 2 * n_freq,).
    return pairs.reshape(*x.shape[:-1], x.shape[-1] * 2 * n_freq)


def positional_embed(
    t: Array, n_freq: int, base: float = 10_000.0
) -> Array:
    """Sinusoidal scalar embedding (transformer-style, continuous ``t``).

    For each frequency ``k in [0, n_freq)``::

        freq[k]            = base ** (-k / n_freq)
        phase[..., k]      = t[..., None] * freq[k]
        out[..., 2k]       = cos(phase[..., k])
        out[..., 2k + 1]   = sin(phase[..., k])

    Adapts the "Attention Is All You Need" positional encoding to a
    continuous scalar ``t`` (use case: temperature, density, MD step).
    Use ``circular_embed`` instead when the input is a periodic coord
    and you know its period.

    Arguments:
        t:       Scalar context tensor of shape ``(...,)``.
        n_freq:  Number of cos/sin pairs (must be >= 1).
        base:    Base of the exponential frequency schedule (default
                 10000, matching the original paper).

    Returns:
        Array of shape ``(..., 2 * n_freq)``.
    """
    if n_freq < 1:
        raise ValueError(
            f"positional_embed: n_freq must be >= 1, got {n_freq}."
        )
    if base <= 0:
        raise ValueError(
            f"positional_embed: base must be positive, got {base}."
        )

    t = jnp.asarray(t)
    k = jnp.arange(n_freq, dtype=t.dtype)                       # (n_freq,)
    freq = jnp.asarray(base, dtype=t.dtype) ** (-k / n_freq)    # (n_freq,)
    phase = t[..., None] * freq                                 # (..., n_freq)

    cos = jnp.cos(phase)
    sin = jnp.sin(phase)
    pairs = jnp.stack([cos, sin], axis=-1)                      # (..., n_freq, 2)
    return pairs.reshape(*t.shape, 2 * n_freq)

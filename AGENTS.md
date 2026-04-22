# AGENTS.md

Project context for coding agents (Claude Code, Cursor, Copilot, etc.).

## Project Summary

Minimal normalizing flows library in JAX. Provides RealNVP and spline flow builders, conditional flows, identity gating, structured rank-N couplings for particle-system events, and an assembly API for custom architectures. Particle-system toolkit (Stages A + B): `Geometry` value object, `Rescale` / `CircularShift` / `CoMProjection` bijections, `UniformBox` / `LatticeBase` (5 crystal factories) base distributions, and `utils/pbc` + `utils/lattice` helpers. Not a pip package; clone and import directly.

For current stage status and what's next, read [PLAN.md §0](PLAN.md). For
the design philosophy and what nflojax refuses to build, read
[DESIGN.md](DESIGN.md). Before any change, run [Testing Strategy](#testing-strategy)
checks.

## Philosophy

This is JAX scientific computing code. Every decision follows from that.

- **Lean and hackable.** Small codebase a researcher can read in an afternoon. No framework magic, no plugin systems, no registries. A user who wants to add a new transform reads one file and follows the pattern.
- **Readable over clever.** Plain functions and dataclasses. If a piece of code needs a comment to explain what it does (not why), rewrite it.
- **No unnecessary abstractions.** One level of indirection is fine; two needs justification. Don't wrap things that don't need wrapping. Three similar lines beat a premature helper.
- **JIT-friendly throughout.** All numerical code must trace cleanly under `jax.jit`. No Python-level control flow on array values. No side effects in forward/inverse paths. Pure functions operating on explicit PyTree params.
- **Numerically robust.** Clamp exponents before `exp`. Use `jax.nn.log_sigmoid` not `log(sigmoid(x))`. Test log-det against full Jacobian autodiff. Treat NaN/Inf as bugs, not edge cases.
- **Documented at the right level.** Docstrings explain the math and the interface contract. Comments explain "why", never "what". AGENTS.md and REFERENCE.md carry the rest.
- **Tests prove correctness, not coverage.** Every transform gets its log-det checked against full Jacobian autodiff. Round-trip `forward(inverse(x)) == x` is tested. Property-based checks over random inputs. Don't write tests for the sake of lines; write tests that catch real bugs.
- **Match the math.** Variable names, function signatures, and docstrings should map clearly to the underlying equations. A reader familiar with the normalizing flows literature should recognize the notation. Don't rename standard quantities for "readability".

## Tech Stack

- **JAX** (core compute, JIT, vmap, autodiff)
- **Flax** (conditioner MLPs via `linen`)
- **Python 3.10+** (type unions with `|`)
- No pip package, no `__init__.py` exports

## Project Structure

```
nflojax/
  __init__.py          empty
  builders.py          High-level constructors + assembly API
  flows.py             Flow and Bijection classes
  transforms.py        All transform types + CompositeTransform
  distributions.py     StandardNormal, DiagNormal, UniformBox
  nets.py              MLP conditioner, ResNet init
  splines.py           Rational-quadratic spline primitives
  scalar_function.py   LOFT forward/inverse scalar functions
  geometry.py          Geometry value object (box bounds + per-axis periodicity)
  utils/
    __init__.py        empty
    pbc.py             nearest_image, pairwise_distance(_sq) under PBC
    lattice.py         fcc / diamond / bcc / hcp / hex_ice generators
tests/
  conftest.py          Shared fixtures + check_logdet_vs_autodiff + requires_x64
  test_builders.py
  test_transforms.py
  test_identity_gate.py
  test_conditional_flow.py
  test_splines.py
  test_distributions.py
  test_nets.py
  test_utils_pbc.py
  test_utils_lattice.py
```

## Module Dependency Graph

```
builders     -> flows, transforms, distributions, nets
flows        -> transforms (gate), nets (types)
transforms   -> nets (MLP), splines, scalar_function, geometry
distributions -> geometry (UniformBox), nets (types)
utils.pbc    -> geometry, nets (types)
utils.lattice -> numpy (no JAX / Flax — static lattice positions)
geometry     -> numpy (no JAX / Flax — configuration values only)
nets         -> flax.linen
```

## Entry Points

- **User entry**: `build_realnvp()`, `build_spline_realnvp()` in `builders.py`
- **Low-level**: `TransformClass.create()` + `assemble_bijection()`/`assemble_flow()`
- **Core types**: `Flow`, `Bijection` in `flows.py`

## Key Patterns

- **Explicit params**: no state in objects. All params passed as PyTree dicts.
- **Transform interface**: `forward(params, x, context=None, g_value=None) -> (y, log_det)`. Some transforms (e.g. `SplitCoupling`) intentionally omit `g_value` when not needed — CompositeTransform detects this via `_block_supports_gvalue`.
- **Zero-init**: conditioner output layers initialized to zero so flows start as identity. Shared helper `identity_spline_bias(num_scalars, num_bins, min_d, max_d)` produces the RQS bias for both `SplineCoupling` and `SplitCoupling`.
- **Mask convention** (flat couplings): `mask=1` means frozen (passed through), `mask=0` means transformed. Alternating parity between layers.
- **Split convention** (structured couplings): `SplitCoupling` partitions along `split_axis` at `split_index` instead of using a scalar mask. Alternate `swap` between layers to cover all slots; there is no `analyze_mask_coverage` equivalent.
- **Event shape**: base distributions and structured couplings accept `event_shape: int | tuple[int, ...]`. Canonical internal form is a tuple. Rank-1 uses `(dim,)`; rank-N uses e.g. `(N, d)`. See REFERENCE.md "Event Shape Convention".
- **Rank-polymorphic composition**: `CompositeTransform` initializes its log-det accumulator as a scalar zero so it works for any event rank. Don't assume `x.shape[:-1]` is the batch shape.
- **Gate contract**: `identity_gate` callable must be written for single sample `(context_dim,)`; batching via `jax.vmap`.
- **Feature extractor split**: gate sees raw context, couplings see extracted features.

## Testing Strategy

**One rule: after any code edit, run `pytest tests/`.** The full suite is
parallel by default (via `pytest-xdist`, configured in `pyproject.toml`) and
runs in ~85s under float32, ~95s under x64 on a multicore machine. Do not
pick a subset — the cost of a missed regression is larger than the minute
you save.

```bash
# Default: run everything in parallel (~85s)
pytest tests/

# At stage close (closing a PLAN.md task): also check float64 (~95s)
JAX_ENABLE_X64=1 pytest tests/

# Iterating on ONE failure you're debugging — narrow with -k, then re-run full
pytest tests/ -k "Rescale and round_trip"
```

Two commands total for the full float32+x64 check (~3 min combined). If you
find yourself wanting "just this file", you're probably over-triaging; run
the full suite unless you have a specific reason.

### Float32 skips

Six tests carry `@requires_x64` and skip under float32 (RQS-inverse and
triangular-solve roundoff exceeds their `atol`); all pass under
`JAX_ENABLE_X64=1`. Expect `pytest tests/` to report `... passed, 6 skipped`.

## Known Issues

No critical or high-priority issues open.

Previously fixed:
- **C1** (fixed `765a278`): LOFT inverse overflow, clamped exponent to 80.0
- **C2** (fixed `765a278`): LoftTransform now supports `g_value` gating
- **C3** (fixed `765a278`): `TestLogdetVsAutodiff` in `test_transforms.py` + spline autodiff tests
- **H2** (fixed `d8446f2`): `max_log_scale` aligned to 5.0 across dataclass, `.create()`, and builders

## Gotchas

- `identity_gate` single-sample contract: gate function receives `(context_dim,)`, not batched. `jax.vmap` handles batching. Writing a batch-aware gate silently produces wrong results. Validated at build time via `jax.eval_shape`.
- Raw context vs extracted: when using a feature extractor, the gate still gets raw context.
- No `__init__.py` exports: must use `from nflojax.builders import build_realnvp`.
- **`CoMProjection` log-det is zero by design** (Convention 1: density on the `(N−1, d)` reduced space). If you need an ambient log-density (reverse-KL with ambient `E(x)`, ESS, `logZ`), add `CoMProjection.ambient_correction(N, d) = (d/2)·log(N)`. Do **not** stack `CoMProjection` with an augmented-coupling pattern — they double-count. See [REFERENCE.md — CoMProjection](REFERENCE.md#comprojection) and [EXTENDING.md — CoM handling](EXTENDING.md#com-handling).

## Documentation Map

| Need | Read |
|------|------|
| Scientific context, Boltzmann-generator primer, vocabulary | [BACKGROUND.md](BACKGROUND.md) |
| Vision, scope, what to build / refuse to build | [DESIGN.md](DESIGN.md) |
| Implementation plan, stage status, long-term trajectory | [PLAN.md](PLAN.md) |
| Design-rationale audit (advisory; not canonical) | [audit.md](audit.md) |
| Quick start, install | [README.md](README.md) |
| How to do X (examples) | [USAGE.md](USAGE.md) |
| API signatures, options tables | [REFERENCE.md](REFERENCE.md) |
| Math, design decisions | [INTERNALS.md](INTERNALS.md) |
| Adding transforms/distributions | [EXTENDING.md](EXTENDING.md) |

If you do not know what a Boltzmann generator is or what nflojax is *for*, start with `BACKGROUND.md`. Before adding any new code, read `DESIGN.md` §§1–4 (vision, philosophy, scope) and run the §9 heuristics. `PLAN.md` tells you what stage is in flight and what v1.0 means. `audit.md` is working opinion — see its "How to read this" preamble before treating anything there as canonical.

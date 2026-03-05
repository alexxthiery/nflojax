# AGENTS.md

Project context for coding agents (Claude Code, Cursor, Copilot, etc.).

## Project Summary

Minimal normalizing flows library in JAX. Provides RealNVP and spline flow builders, conditional flows, identity gating, and an assembly API for custom architectures. Not a pip package; clone and import directly.

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
  distributions.py     StandardNormal, DiagNormal
  nets.py              MLP conditioner, ResNet init
  splines.py           Rational-quadratic spline primitives
  scalar_function.py   LOFT forward/inverse scalar functions
tests/
  conftest.py          Shared fixtures + check_logdet_vs_autodiff
  test_builders.py
  test_transforms.py
  test_identity_gate.py
  test_conditional_flow.py
  test_splines.py
  test_distributions.py
  test_nets.py
```

## Module Dependency Graph

```
builders -> flows, transforms, distributions, nets
flows    -> transforms (gate), nets (types)
transforms -> nets (MLP), splines, scalar_function
nets     -> flax.linen
```

## Entry Points

- **User entry**: `build_realnvp()`, `build_spline_realnvp()` in `builders.py`
- **Low-level**: `TransformClass.create()` + `assemble_bijection()`/`assemble_flow()`
- **Core types**: `Flow`, `Bijection` in `flows.py`

## Key Patterns

- **Explicit params**: no state in objects. All params passed as PyTree dicts.
- **Transform interface**: `forward(params, x, context=None, g_value=None) -> (y, log_det)`
- **Zero-init**: conditioner output layers initialized to zero so flows start as identity.
- **Mask convention**: `mask=1` means frozen (passed through), `mask=0` means transformed. Alternating parity between layers.
- **Gate contract**: `identity_gate` callable must be written for single sample `(context_dim,)`; batching via `jax.vmap`.
- **Feature extractor split**: gate sees raw context, couplings see extracted features.

## Dev Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_builders.py -v

# Run tests matching pattern
python -m pytest tests/ -k "identity_gate" -v

# Run with float64 enabled
JAX_ENABLE_X64=1 python -m pytest tests/ -v
```

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

## Documentation Map

| Need | Read |
|------|------|
| Quick start, install | [README.md](README.md) |
| How to do X (examples) | [USAGE.md](USAGE.md) |
| API signatures, options tables | [REFERENCE.md](REFERENCE.md) |
| Math, design decisions | [INTERNALS.md](INTERNALS.md) |
| Adding transforms/distributions | [EXTENDING.md](EXTENDING.md) |

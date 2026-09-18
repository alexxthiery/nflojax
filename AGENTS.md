# AGENTS.md

Project context for coding agents (Claude Code, Cursor, Copilot, etc.).
**This file is the single, provider-agnostic source of truth.** `CLAUDE.md` and
`.github/copilot-instructions.md` only route here — put all guidance in this file.

## Project Summary

Minimal normalizing flows library in JAX. Provides RealNVP and spline flow builders, product-domain flows for flat mixed real/interval/circular events, conditional flows, identity gating, structured rank-N couplings for particle-system events, and an assembly API for custom architectures. Particle-system toolkit (Stages A + B): `Geometry` value object, `Rescale` / `CircularShift` / `CoMProjection` bijections, `UniformBox` / `LatticeBase` (5 crystal factories) base distributions, and `utils/pbc` + `utils/lattice` helpers. Reference conditioners (Stage D): `MLP` + `DeepSets` (invariant) + `Transformer` (pre-norm, equivariant) + `GNN` (KNN under PBC, equivariant); plug into `SplitCoupling(flatten_input=False)` for the structured-input path. Use as an editable source package (`pip install -e ".[test]"`). There are no top-level `nflojax.__init__` exports; import from submodules.

For current stage status and what's next, read [PLAN.md §0](PLAN.md). For
the design philosophy and what nflojax refuses to build, read
[DESIGN.md](DESIGN.md). Before any change, run [Testing Strategy](#testing-strategy)
checks.

## Philosophy

This is JAX scientific computing code. Every decision follows from that.

- **Lean and hackable.** Small codebase a researcher can read in an afternoon. No framework magic, no plugin systems, no registries. A user who wants to add a new transform reads the matching file under `nflojax/transforms/` and follows the pattern.
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
- Editable source package, no `__init__.py` exports

## Project Structure

```
nflojax/
  __init__.py          empty
  builders/            Public builder facade plus assembly/flat/product/particle modules
  flows.py             Flow and Bijection classes
  transforms/          Public transform facade plus focused implementation modules
  distributions.py     StandardNormal, DiagNormal, UniformBox
  domains.py           ScalarDomain, ProductDomain for flat mixed-domain flows
  nets.py              MLP conditioner, ResNet init
  splines.py           Rational-quadratic spline primitives
  scalar_function.py   LOFT forward/inverse scalar functions
  geometry.py          Geometry value object (box bounds + per-axis periodicity)
  embeddings.py        circular_embed, positional_embed (stateless feature transforms)
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
  test_embeddings.py
  test_utils_pbc.py
  test_utils_lattice.py
```

## Module Dependency Graph

```
builders     -> flows, transforms, distributions, domains, geometry, nets
flows        -> transforms (gate), nets (types)
transforms   -> nets (MLP), splines, scalar_function, geometry, domains
distributions -> geometry (UniformBox), domains (ProductBase), utils.lattice (LatticeBase factories), nets (types)
domains      -> numpy, jax.numpy (flat coordinate topology metadata)
embeddings   -> geometry (circular_embed), nets (types)
utils.pbc    -> geometry, nets (types)
utils.lattice -> numpy (no JAX / Flax — static lattice positions)
geometry     -> numpy (no JAX / Flax — configuration values only)
nets         -> flax.linen
```

## Entry Points

- **User entry**: builder facade `nflojax.builders`; direct modules under
  `nflojax.builders.{assembly,flat,product,particle}` are also supported.
- **Low-level**: `TransformClass.create()` + `assemble_bijection()`/`assemble_flow()`
- **Core types**: `Flow`, `Bijection` in `flows.py`

Builder choice is explicit; do not copy options between rows unless the row
already supports them.

| Builder | Event type | Domain | Base default | Conditioner style | Unsupported by design |
|---------|------------|--------|--------------|-------------------|-----------------------|
| `build_realnvp` | flat rank-1 | all real | `StandardNormal` or `DiagNormal` | MLP | product bounds, rank-N events |
| `build_spline_realnvp` | flat rank-1 | all real | `StandardNormal` or `DiagNormal` | MLP | product bounds, rank-N events |
| `build_product_spline_flow` | flat rank-1 | mixed real/interval/circular | `ProductBase` | MLP with default product feature map | LOFT, linear mixing, permutations |
| `build_particle_flow` | structured rank-N | box/torus particle events | caller-provided | keyword-only conditioner factory | flat masks, feature extractor |
| `assemble_bijection` / `assemble_flow` | custom | caller-defined | caller-provided | caller-defined | automatic topology decisions |

Builder option sets are intentionally different. Do not add an option to a
builder just because another builder supports it.

| Option family | Flat RealNVP builders | Product-domain builder | Particle builder | Assembly API |
|---------------|-----------------------|------------------------|------------------|--------------|
| Custom base | yes, flat `(dim,)` | yes, `domain.event_shape` | yes, `(N, d)` or `(N-1, d)` with CoM | caller-defined |
| Context / identity gate | yes | yes | no | caller-defined |
| Context feature extractor | yes | no | no | yes |
| LOFT / linear / flat permutations | yes | no | no | caller-defined |
| Circular coordinate shifts | no | yes | no | caller-defined |
| Particle circular shifts / CoM | no | no | yes | caller-defined |

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

**Default rule: after any routine code edit, run only the fast suite with
`pytest tests/`.** The default suite excludes tests marked `slow` and is meant
to finish in under a minute. Do not run the full suite by habit while iterating:
the slow tests are reserved for explicit user requests, serious commits,
release/stage-close checks, or broad numerical refactors.

When the user asks to commit, prepare a serious commit, or says the change is
ready, remind them that running the full float32 suite, and full x64 if
precision-sensitive code changed, is a good idea before committing. Only run
those full suites when the user explicitly asks or confirms.

`pytest-xdist` is an optional speedup, not a required test-runner dependency.

```bash
# Default fast suite
pytest tests/

# Optional speedup if pytest-xdist is installed
pytest -n auto tests/

# Slow integration and full-Jacobian proofs only
pytest -o addopts='-q' -m slow tests/

# Targeted x64 precision proofs
JAX_ENABLE_X64=1 pytest -m requires_x64 tests/

# If product-domain dtype/geometry changed, also check that focused suite
JAX_ENABLE_X64=1 pytest -o addopts='-q' tests/test_product_domains.py

# Full float32/x64 are release/stage-close checks, not routine edit gates
pytest -o addopts='-q' tests/
JAX_ENABLE_X64=1 pytest -o addopts='-q' tests/

# Iterating on ONE fast-suite failure you're debugging
pytest tests/ -k "Rescale and round_trip"

# Iterating on ONE slow-suite failure you're debugging
pytest -o addopts='-q' tests/ -k "log_det_vs_autodiff"
```

For routine edits, run the default fast suite plus the targeted x64 command
when precision-sensitive code changed. Full float32 and full x64 take several
minutes serially; reserve them for stage close, release checks, or changes that
touch shared numerical kernels.

### Float32 skips

Several tests carry `@requires_x64` and skip under float32 (RQS-inverse and
triangular-solve roundoff exceeds their `atol`); all pass under
`JAX_ENABLE_X64=1`. The GNN stacked-spline round-trip perturbation is also an
x64 precision proof; default precision has a separate finite/JIT smoke check.
Run only these proofs with `JAX_ENABLE_X64=1 pytest -m requires_x64 tests/`.

## Known Issues

No critical or high-priority code issues open. Audit follow-ups are tracked in
`PLAN.md` under "Audit remediation".

Previously fixed:
- **C1** (fixed `6351357`): LOFT inverse overflow, clamped exponent to 80.0
- **C2** (fixed `6351357`): LoftTransform now supports `g_value` gating
- **C3** (fixed `6351357`): `TestLogdetVsAutodiff` in `test_transforms.py` + spline autodiff tests
- **H2** (fixed `c025b24`): `max_log_scale` aligned to 5.0 across dataclass, `.create()`, and builders

## Gotchas

- `identity_gate` single-sample contract: gate function receives `(context_dim,)`, not batched. `jax.vmap` handles batching. Writing a batch-aware gate silently produces wrong results. Validated at build time via `jax.eval_shape`.
- Raw context vs extracted: when using a feature extractor, the gate still gets raw context.
- No `__init__.py` exports: must use `from nflojax.builders import build_realnvp`.
- **`CoMProjection` log-det is zero by design** (Convention 1: density on the `(N−1, d)` reduced space). If you need an ambient log-density (reverse-KL with ambient `E(x)`, ESS, `logZ`), add `CoMProjection.ambient_correction(N, d) = (d/2)·log(N)`. Do **not** stack `CoMProjection` with an augmented-coupling pattern — they double-count. See [REFERENCE.md — CoMProjection](REFERENCE.md#comprojection) and [EXTENDING.md — CoM handling](EXTENDING.md#com-handling).
- **`SplitCoupling.flatten_input` is True by default.** The flat contract matches `MLP`. To plug in a permutation-aware conditioner (`DeepSets`, `Transformer`, `GNN`, or a user's own), construct `SplitCoupling(..., flatten_input=False)` so the conditioner sees `(*batch, N_frozen, d)`. `SplitCoupling.create()` always uses the MLP path and ignores this.
- **`GNN` self-edge masking uses `jnp.where`, not multiplication.** `jnp.eye(N) * jnp.inf` gives `0 * inf = NaN` off-diagonal and silently poisons the neighbour list. Use `jnp.where(eye_bool, jnp.inf, d_sq)` instead.
- **`GNN` neighbour distance uses `sqrt(d_sq + eps)`, not `sqrt(d_sq)`.** `sqrt` has an infinite gradient at 0, so two coincident particles NaN the *gradient* while the forward stays finite (so it hides until you backprop) — this silently breaks reverse-KL training the moment the flow samples a close pair. The `+1e-12` floor keeps it finite; regression at `tests/test_nets.py::TestGNN::test_gradient_finite_with_coincident_particles`.
- **Product-domain feature maps are builder defaults.** `ProductDomain` is
  coordinate-topology metadata; `conditioner_features` is only the default MLP
  feature map for `ProductSplineCoupling`. See REFERENCE.md "Product Domains"
  before changing it.
- **Periodic target means circular splines, not an option.** If a particle
  target is periodic (a box/torus), the flow density must be periodic too.
  Use `build_particle_flow` with its default `boundary_slopes='circular'`
  (plus the built-in `CircularShift`). `boundary_slopes='linear_tails'` puts
  unbounded tails on a periodic target: the density is invariant under
  per-particle box translations (`x_i -> x_i + L`), so it has infinitely many
  identical copies over the tails and reverse-KL diverges to infinite entropy.
  `build_particle_flow` now **raises** if a periodic `Geometry` is paired with
  `linear_tails`. Use `linear_tails` only for open/free systems (free clusters
  like LJ13 / DW4); for a bounded non-periodic box, build the `Geometry` with
  `periodic=[False, ...]`. Also: when a torus flow uses a (non-periodic)
  Gaussian `LatticeBase`, score samples with the forward `log_q` from
  `sample_and_log_prob`, not `log_prob(x)` — the inverse path is unreliable for
  samples that wrap across the box seam.

## Sibling repos

These live next to nflojax on disk and are load-bearing for testing particle flows end-to-end. They are **not** dependencies — nflojax has no runtime coupling to either.

- **`../jax-pdf/`** — benchmark target log-densities. Provides 11 distributions with a unified `__call__(x) -> log_p` API (on its `feature/periodic-lj-target` branch; `main` has 8). Particle targets (free-cluster `LennardJones`, `DW4`; periodic `PeriodicLennardJones`, `MonatomicWater`, `HarmonicCrystal`) accept structured `(..., n_particles, spatial_dim)` input and plug directly into flows built with `build_particle_flow` — no reshape. Generic targets (`Banana2D`, `NealFunnel`, `LGCP`, `MullerBrown`, `PhiFour`, `DoubleWell`) take flat `(..., dim)` input. Top-level import: `from jax_pdf import LennardJones, DW4, ...`. Use for reverse-KL smoke tests, regression targets, and worked examples.
- **`../bgmat-clean/`** — downstream application repo driving Stage G validation. Clean-room Boltzmann-generator rebuild on top of nflojax. MS2 (DW4) closed as partial success; periodic LJ solid closed 2026-09-18; MS3 (mW water, periodic) is active, with north stars in bgmat-clean's `AGENTS.md` and PLAN.md §7. If nflojax-side friction surfaces in bgmat-clean, file it as a PLAN.md §1–§5 follow-up before declaring the milestone closed (PLAN.md §7 acceptance).

## Documentation Map

| Need | Read |
|------|------|
| Scientific context, Boltzmann-generator primer, vocabulary | [BACKGROUND.md](BACKGROUND.md) |
| Vision, scope, what to build / refuse to build | [DESIGN.md](DESIGN.md) |
| Implementation plan, stage status, long-term trajectory | [PLAN.md](PLAN.md) |
| Local audit notes and remediation rationale | `audit/` (ignored) |
| Quick start, install | [README.md](README.md) |
| How to do X (examples) | [USAGE.md](USAGE.md) |
| API signatures, options tables | [REFERENCE.md](REFERENCE.md) |
| Math, design decisions | [INTERNALS.md](INTERNALS.md) |
| Adding transforms/distributions | [EXTENDING.md](EXTENDING.md) |

If you do not know what a Boltzmann generator is or what nflojax is *for*, start with `BACKGROUND.md`. Before adding any new code, read `DESIGN.md` §§1–4 (vision, philosophy, scope) and run the §9 heuristics. `PLAN.md` tells you what stage is in flight and what v1.0 means. Local audit notes may exist under ignored `audit/`; if they conflict with DESIGN.md, update the canonical docs before changing code.

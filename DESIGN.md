# nflojax — Design Document

**Scope of this document.** A durable, self-contained description of what nflojax is, what it is not, and the principles that tell us where new code belongs. It is written to be read by humans and by coding agents with no prior session context. It is *not* an implementation plan. It tells an agent *what to build and what to refuse to build*, and *why*.

**Status.** Living design. Revise when a real use case breaks an existing rule; do not revise to chase novelty.

---

## 1. Motivation

nflojax is a normalizing-flow framework. Its original purpose was pedagogy and research: a small JAX codebase that a researcher can read in an afternoon, hack in a week, and trust under `jax.jit`.

We now want the same library to be the flow-side foundation for particle-system applications: crystalline solids, liquids, disordered materials, and — eventually — molecular / chemistry workflows. The downstream apps we have in mind today:

- **DeepMind *Flows for Atomic Solids*** (Wirnsberger et al., 2022) — Lennard-Jones and monatomic-water crystals with a Transformer conditioner.
- **bgmat** (Schebek, Noé, Rogal, 2025) — the same systems plus silicon, with a local MPNN conditioner and augmented-coupling architecture.
- **Future** — molecular systems, multi-species solids, liquids, disordered phases.

These apps share a lot of non-physics machinery (periodic boxes, lattice bases, permutation-equivariant conditioners, circular splines, rank-N couplings). Each one currently reimplements most of it. nflojax should absorb the shared machinery *so that every new app writes only the physics*.

The frame we want:

> nflojax = **normalizing-flow framework for particle systems.**
> Applications = **energies + training + observables + domain-specific conditioners + research innovations.**

nflojax does not know that particles interact. It knows how to parameterise, sample, and score a density on a particle configuration space.

---

## 2. Design philosophy

These are non-negotiable. Every concrete decision downstream must pass through them.

1. **Lean and hackable.** Plain dataclasses and pure functions; no framework magic, no plugin registries, no runtime dispatch. The unit of surveyability is the *concept*, not the file. A single-concept module (one class, one numerical kernel) should be surveyable in one read — approximately ≤ 500 LOC. *Catalogue* modules (`transforms.py` gathering all bijections, `nets.py` gathering all conditioners) are lists, not abstractions; they can grow as new entries land, but each *entry* in a catalogue is held to the same ≤ 500-LOC bar. An entry exceeding that bar is the real signal that the abstraction is wrong or the entry should split into its own file.

2. **Readable over clever.** Plain dataclasses and pure functions. If a piece of code needs a comment to explain what it does (as opposed to why), rewrite it.

3. **Explicit parameters.** No hidden state. All learnable parameters are PyTree dicts passed into forward / inverse / sample. This matches JAX and makes everything trivially vmap-able, jittable, and pmappable.

4. **JIT-native throughout.** No Python-level control flow on traced values. No side effects in forward / inverse. Test every new primitive under `jax.jit`.

5. **Numerically robust.** Clamp exponents before `exp`. Use `jax.nn.log_sigmoid` instead of `log(sigmoid(x))`. Test every transform's log-det against full Jacobian autodiff. Treat NaN / Inf as bugs, not edge cases.

6. **Tests prove correctness, not coverage.** For every bijection: round-trip, log-det-vs-autodiff, jit. For every base distribution: `sample_and_log_prob` consistency. Tests exist because a specific failure mode exists — never because we want more green dots.

7. **Match the math.** Variable names and signatures echo the normalizing-flows literature. `x_k`, `y_k`, `widths`, `heights`, `derivatives`, `log_det`. Do not rename standard quantities for "readability".

8. **float32 is the default; float64 is the proof.** All tests pass under `JAX_ENABLE_X64=1`. Tolerance-sensitive tests that fail under float32 carry a documented `@requires_x64` marker and a one-line reason.

9. **No domain knowledge inside flow code.** `SplineCoupling` does not know particles exist. `LatticeBase` does not know what an energy is. Specialisation happens at composition time, not inside primitives.

10. **Every new primitive must be useful beyond the use case that motivated it.** A bijection that only makes sense for LJ crystals belongs in an application. A bijection that makes sense for *any* torus flow belongs in nflojax.

---

## 3. What nflojax IS

nflojax is the toolkit for building, sampling, and scoring a normalizing flow on a particle configuration space. The library delivers:

### 3.1 Core flow machinery

- **Bijections** covering the Cartesian and torus regimes. Today: `AffineCoupling`, `SplineCoupling`, `SplitCoupling` (rank-N), `LinearTransform`, `Permutation` (axis-aware), `CircularShift`, `LoftTransform`, `CompositeTransform`. Planned in Stage A: `Rescale`, `CoMProjection`.
- **Base distributions.** Today: `StandardNormal`, `DiagNormal`. Planned in Stage B: `UniformBox`, `LatticeBase` with factories for FCC, diamond, hexagonal ice, BCC, HCP.
- **Composition** primitives: `CompositeTransform`, the assembly API (`assemble_flow` / `assemble_bijection`), builders for common topologies (`build_realnvp`, `build_spline_realnvp` today; `build_particle_flow` in Stage E).

### 3.2 Conditioner infrastructure

A *conditioner* is the neural module that turns the frozen half of a coupling into per-scalar spline (or affine) parameters. nflojax provides:

- A **conditioner contract** (`validate_conditioner` in `nets.py`) that describes the expected signature, output shape, and identity-at-init requirement.
- **Reference conditioners** today: `MLP` (no equivariance), `ResNet` (building block for MLP). Planned in Stage D at increasing equivariance levels: `DeepSets` (permutation-invariant), `Transformer` (permutation-equivariant attention), `GNN` (permutation-equivariant message passing). Users can always bring their own conditioner that satisfies the contract; the scheduled reference implementations serve as starting points, not as the only option.
- **Feature helpers** (planned in Stage C): circular Fourier embeddings, positional (sinusoidal) embeddings for scalar context, in a new `embeddings.py`.

Reference conditioners are genuinely usable, but nothing in the library prevents a user from writing their own. The contract is the product; the implementations are examples.

### 3.3 Particle-system geometry

- **Periodic box** utilities: `nearest_image`, pairwise distances, under orthogonal boxes. Triclinic boxes are out of current scope (see §4).
- **Lattice generation** utilities: functions that emit `(N, d)` lattice positions for the supported crystal structures, parameterised by cell counts and lattice constants.
- **Circular spline** mode with matched boundary slopes (already landed in `feature/particle-events`), paired with `CircularShift` for full torus diffeomorphisms.
- **Rescale** between physical boxes and the canonical spline range `[-1, 1]` (or `[-B, B]`).
- **Centre-of-mass** bijector for translation-invariant flows.

### 3.4 Assembly

A user should be able to build a particle flow in ~10 lines: pick a base, pick a conditioner, pick a box, call one builder, get back a `Flow` with initialized params. The builder is a convenience, not the only way in; the underlying assembly API (`assemble_flow`, `assemble_bijection`) remains available for users who want full control.

---

## 4. What nflojax IS NOT

Each exclusion is a deliberate boundary. If a contributor proposes moving one of these *in*, they must show that every downstream app would use it, that it is domain-agnostic, and that it does not change the library's shape.

1. **No energies.** Not LJ, not mW, not Stillinger-Weber, not Buckingham, not any force field. Energies are application code. nflojax does not have a `systems/` directory.

2. **No training loops.** No reverse-KL helper, no forward-KL helper, no SNIS, no optax chain construction, no learning-rate schedules, no training step factory. These are one-screen pieces of code that every application writes in its own idiom. We do not own the user's optimiser, their data pipeline, or their loss function.

3. **No observables for physics.** No RDF, no structure factor, no ESS, no `logZ` estimator, no free-energy helpers. ESS and SNIS are generic *once weights exist* — and weights only exist in the application, because the application brings the target density.

4. **No config system.** No `ml_collections.ConfigDict` integration, no Hydra, no TOML parsing. Users pass arguments to Python functions.

5. **No checkpointing, no logging, no distributed training.** These are infrastructure the caller owns.

6. **No marginal inference, no free-energy integration, no phase-diagram machinery.** Research-layer utilities stay with the research.

7. **No SE(3) or E(3) equivariant architectures.** Permutation equivariance is in; rotation / reflection equivariance is out. Implementing a good E(3) MPNN is a research product in its own right (EGNN, NequIP, MACE, PaiNN) and each has opinions we do not want to relitigate. If a user needs E(3), they bring their own conditioner.

8. **No triclinic / non-orthogonal boxes** — yet. bgmat's in-progress support stays there until someone lands a clean orthogonal / triclinic abstraction that is provably bug-free on the orthogonal path. Adding broken triclinic support to nflojax is worse than no triclinic support.

9. **No augmented-coupling "framework".** Augmented coupling is a *composition pattern*: double the base dimension, run a flow over the augmented state, marginalize. This is expressible today with `UniformBox` / `DiagNormal` + `SplitCoupling`. It stays documented in `EXTENDING.md`, not added as a dedicated class. Patterns live in docs; primitives live in code.

10. **No heavy dependency surface.** Current deps: JAX, Flax (for `linen` modules). We do not adopt Distrax, Haiku, TFP-bijectors, e3nn, Jraph, or Equinox. Any new dep must replace substantial home-grown code and survive review by the "lean and hackable" rule.

---

## 5. Core abstractions

These are the contracts that make the whole thing unified.

### 5.1 `Bijection` / `Flow`

A bijection is a dataclass with `forward(params, x, context=None, ...)` and `inverse(params, y, context=None, ...)`. Both return `(output, log_det_jacobian)`. Log-det can be scalar, batch-shaped, or broadcast-compatible with the batch shape.

A `Flow` couples a bijection with a base distribution and exposes `sample`, `log_prob`, `sample_and_log_prob`.

### 5.2 Context is a PyTree

Context is any JAX PyTree or `None`. A conditioner reads whatever structure it wants. nflojax flow layers pass `context` through to every block that accepts it; they do not inspect it. This is how temperature / density / lattice / box / species are threaded to conditioners without the flow layer caring.

**Current implementation.** As of this writing, the built-in `MLP` conditioner concatenates `context` into its input tensor, and `_compute_gate_value` indexes `context.ndim` — both assume `context` is a single `Array` (or `None`). A user-supplied conditioner that takes a structured PyTree context works fine at the flow level, but the built-in path does not yet. Widening `MLP` (e.g. via `jax.flatten_util.ravel_pytree`) and loosening `_compute_gate_value` is tracked in `PLAN.md` Stage A4. Until that lands, treat the "Context is a PyTree" rule as aspirational for the library core, and as true today only when a user brings their own conditioner.

### 5.3 Event shape and event axes

All rank-N-aware primitives consume an `event_shape: tuple[int, ...]` and operate on the trailing `len(event_shape)` axes. A rank-1 event (`(dim,)`) is the Cartesian case; `(N, d)` is the particle case; `(N_a, N_b, d)` would be two-species. A primitive that only works on the last axis must say so.

### 5.4 Conditioner contract

A conditioner is a Flax module with `__call__(x, context=None) -> (..., out_dim)`. The output dimension matches `required_out_dim` of the coupling it serves. Zero-initialised output layer so the whole flow is identity at init. `validate_conditioner` traces the module against abstract shapes to check the contract; for conditioners whose input is not a flat vector (GNN, Transformer), validation is opt-out.

### 5.5 Log-det conventions

- Scalar log-det (e.g., `CircularShift`) returns a scalar and broadcasts up through `CompositeTransform`'s accumulator.
- Batch-shaped log-det returns shape `batch_shape` (everything except event axes).
- Never return event-shaped log-det; this is a bug that silently widens the accumulator.

### 5.6 Identity at init

Every new coupling / conditioner ships with a zero-init output layer so that `forward(params_init, x) == x`. This is how training stabilises. It is also the most aggressive correctness check a new primitive can have; the "near-identity at init" test catches more bugs than any other single test.

---

## 6. How to think about "particle systems" in nflojax

A particle system in nflojax-speak is an event of shape `(N, d)` (optionally with more leading event axes for species / chains). nflojax does not know whether those particles are atoms, coarse-grained beads, residues, or abstract markers. It knows:

- How to **partition** an `(N, d)` event into frozen vs transformed slices (`SplitCoupling`).
- How to **permute** particles (`Permutation(event_axis=-2)`).
- How to **shift** them rigidly around a box (`CircularShift`).
- How to **remove translational degrees of freedom** (`ShiftCenterOfMass`).
- How to **build a base distribution** that samples particle configurations: uniform in a box, Gaussian-perturbed lattice sites, pre-equilibrated point clouds (the last of which users would bring themselves via `DiagNormal` or a custom base).
- How to **parameterise a conditioner** that maps particle features to per-scalar bijection parameters.

A solid is a particle system whose base distribution is a lattice. A liquid is a particle system whose base distribution is uniform (or a pre-equilibrated snapshot). The flow-side machinery is identical in both cases; only the base changes. This is the core reason nflojax should be base-agnostic: we do not discriminate between solids and liquids.

Design implications:

- Bases must be **pluggable at assembly time**. A user swapping `LatticeBase.fcc(...)` for `UniformBox(...)` gets a liquid-capable flow by changing one argument.
- Conditioners must be **equivariant where the system is symmetric, free where it is not**. Permutation equivariance is the minimum for a homogeneous particle system. For multi-species systems we may want product-permutation equivariance later; the conditioner contract already accommodates this through context.
- Box handling must be **explicit, per-axis, and optional**. Orthogonal boxes supported now; triclinic later. `nearest_image(dx, box)` takes a per-axis box so a cubic call is `box=L`, a rectangular call is `box=(Lx, Ly, Lz)`. Non-periodic flows skip it.

---

## 7. Forward-thinking design principles

These are the questions we should ask before every new primitive, and the defaults we want to keep.

### 7.1 Bases

- Solids and liquids both need a particle base. Defaults:
  - **`LatticeBase`** for crystalline: lattice positions + per-site Gaussian + optional random permutation. `LatticeBase.fcc / .diamond / .hex_ice / .bcc / .hcp`.
  - **`UniformBox`** for liquids / unstructured starts.
- Keep `LatticeBase` generic (one class + factory functions). Resist adding species-dependent lattices until a concrete application exists.

### 7.2 Equivariance ladder

Pick the minimum equivariance the application requires, not the maximum we can express.

- `MLP` — no equivariance. Use for low-d flat flows.
- `DeepSets` — permutation invariance over particles.
- `Transformer` — permutation equivariance with global information flow.
- `GNN (MPNN)` — permutation equivariance with locality.

Choose `DeepSets` when the per-particle update depends only on aggregate statistics, `Transformer` when you need fully-connected attention (DM paper's regime), `GNN` when locality matters and the particle count is large (bgmat's regime). Never add an equivariant architecture to nflojax without a downstream use case that is present *today*.

### 7.3 Groups beyond Sn

E(3) / SE(3) equivariance is out of scope by design (see §4). If a future application needs it, the design handles it via:

- A user-supplied conditioner that is E(3) equivariant (EGNN, NequIP, MACE, PaiNN — these exist as libraries; we depend on none of them).
- Flow layers that are equivariant under the relevant group: for E(3), this means coupling layers that transform invariant features (distances, angles) rather than raw coordinates. This is expressible with the current machinery but is a user-level construction.

### 7.4 Boundaries

The current flow covers fully periodic boxes (torus) via `CircularShift` + circular-mode splines, and open / infinite space via LOFT. Slab geometries (periodic in 2 axes, open in 1) fall out of per-axis box handling and do not need new primitives. Fixed-boundary non-periodic (confined particles) is a research problem — the conditioner can see the boundary as context, but nflojax does not provide a canonical "confined" bijection.

### 7.5 Multi-species

Not in scope today. The design leaves this open:

- Event shape `(N, d)` is generic; users can already handle multi-species by flattening species into the particle axis and using species identity as context.
- `Permutation(event_axis=-2)` currently permutes all particles uniformly. A future `BlockPermutation` that permutes within species blocks would be ~50 LOC and fits the existing pattern.
- `LatticeBase` today builds a single-species lattice. A future `HeteronuclearLattice` would compose per-species lattices; out of scope until there is an application.

### 7.6 Long-range effects

Ewald, PPPM, and any O(N log N) or O(N²) infrastructure for long-range energies — out. These are energy-side, not flow-side.

### 7.7 Context as the interface

Every non-trivial feature (temperature, pressure, density, lattice bounds, species, external field) is passed as context. This keeps the flow API tiny (one argument) and pushes complexity to the conditioner, which is where the domain lives. When in doubt, put it in context; resist adding new kwargs to flow methods.

---

## 8. Module charter

For every file in `nflojax/`, here is what it is for and what it is not for. An agent adding code to nflojax should locate its work in the correct file by reading this charter; if none fits, that is a signal to ask whether the code belongs in nflojax at all.

*Legend:* entries tagged **(planned, Stage X)** are modules that do not yet exist; their charter is reserved so that when the stage lands the code has a predetermined home. See `PLAN.md` for stage definitions.

- `flows.py` — `Flow` and `Bijection`. Wrappers that compose a base with a transform and expose sample / log-prob. *Not for* new bijections, new bases, new losses.
- `transforms.py` — all bijections. *Not for* conditioner code, base-distribution code, or anything that reads energy.
- `splines.py` — rational-quadratic spline math. *Not for* coupling / distribution / flow logic; this file is a numerical kernel.
- `distributions.py` — base distributions: `StandardNormal`, `DiagNormal`, and new additions (`UniformBox`, `LatticeBase`). *Not for* unnormalised target densities (those live in applications).
- `nets.py` — conditioner modules: `MLP`, `ResNet`, `DeepSets`, `Transformer`, `GNN`. *Not for* generic neural-network utilities unused by conditioners.
- `embeddings.py` (new) — stateless feature transforms: `circular_embed`, `positional_embed`. *Not for* learnable embeddings (those live in `nets.py` as modules).
- `builders.py` — assembly helpers: `build_realnvp`, `build_spline_realnvp`, `build_particle_flow`. *Not for* primitive construction logic; each builder is a thin composition.
- `scalar_function.py` — LOFT numerical kernel, parallel role to `splines.py`.
- `geometry.py` — `Geometry` value object: per-axis box bounds + per-axis periodicity flags. Numpy-backed configuration (not a PyTree, not traced). Consumed by `CircularShift` today and by upcoming geometry-aware primitives (`Rescale`, `UniformBox`, `LatticeBase`, `utils/pbc`). *Not for* metrics, cell matrices for triclinic cells, curved manifolds — those need sibling types.
- `utils/pbc.py` **(planned, Stage B)** — periodic-box geometry: nearest image, pairwise distances. *Not for* forces, energies, or neighbour lists with cutoffs that are used by energies (those are application code).
- `utils/lattice.py` **(planned, Stage B)** — lattice-position generators used by `LatticeBase`. *Not for* lattice-specific physics (Madelung constants, defect structures, etc.).

If a proposed piece of code does not have a home in the list above, that is the first test. Do not create a new module to justify the code; reconsider whether the code belongs.

---

## 9. Decision heuristics

When an agent (or a human) is unsure whether something belongs, apply these tests in order. Stop at the first "no".

1. **Does every target application need it?** If only one app needs it, it lives in the app.
2. **Is it domain-agnostic?** Does it work for LJ crystals, mW ice, and (plausibly) a chemistry workflow without change? If it hardcodes a physics constant, a cutoff radius, a species count, or a thermodynamic state variable — it is domain-specific.
3. **Does it fit an existing module charter?** See §8. If not, reconsider.
4. **Does it pass the "read in an afternoon" bar?** Adding this, is the total nflojax surface still surveyable end-to-end?
5. **Is there a test for the failure mode it is intended to prevent?** Every new primitive lands with round-trip, log-det, and jit tests. Anything that cannot be tested this way either doesn't belong or needs a testable redesign.
6. **Would a fresh contributor understand the math from the code + docstring, without reading a paper?** If not, tighten the docstring or simplify the code.

If the answer to all six is yes, it is nflojax code.

---

## 10. What stays in applications — and why

We enumerate these here so an agent reading this document can immediately answer "why is X in bgmat and not nflojax?"

- **Lennard-Jones, monatomic water, silicon, Buckingham, Tersoff, Stillinger-Weber, ReaxFF, MLIP wrappers** — energies have parameters specific to the target system; parameter choices are research decisions; energies reference cutoffs and neighbour-lists that are physics-specific. *Applications.*
- **Reverse-KL / forward-KL / SNIS / score-matching losses** — each loss bakes in an assumption about the target. nflojax ships the flow primitives; the caller wires the loss because only the caller knows whether they have samples, densities, or energies. *Applications.*
- **Optax chain and LR schedules** — these are four lines of user code and are never the same twice. *Applications.*
- **GNN with top-N neighbour selection + e3nn + lattice-relative features (bgmat's actual GNN)** — a research artifact. The reference MPNN in nflojax is a *baseline*; bgmat's GNN is the *contribution*. *Applications.*
- **Augmented coupling flow (bgmat)** — a composition pattern, not a new primitive. *Documented in nflojax's `EXTENDING.md`, implemented in bgmat.*
- **Marginal inference (bgmat)** — estimates physical quantities from augmented samples. Physics on the loop. *Applications.*
- **Triclinic boxes** — in-progress; lands in nflojax only once orthogonal + triclinic share a clean API. *Applications for now.*
- **Observables (RDF, structure factor, ESS, logZ estimates)** — all rely on physics context (cutoffs, weights, box shape) or on the existence of a trained target density. *Applications.*
- **Config systems, checkpointing, multi-device training, experiment tracking** — infrastructure. *Applications.*

---

## 11. How to verify we stayed in scope

An agent (or human) should be able to run these checks after any change and be confident the library hasn't drifted.

1. **No energy term anywhere in `nflojax/`.** Grep: no `epsilon`, `sigma`, `cutoff`, `beta*`, `lambda_lj`, `kT`, `kcal`, `angstrom`, `hartree`.
2. **No training-loop primitive.** Grep: no `optax`, no `reverse_kl`, no `loss`, no `train_step`, no `checkpoint`.
3. **No observable.** Grep: no `rdf`, no `ess`, no `log_partition`, no `free_energy`, no `logZ`.
4. **No dependency on Distrax, Haiku, TFP, e3nn, Jraph, Equinox.** `pyproject.toml` lists only JAX and Flax.
5. **No `__init__.py` exports.** The library remains navigable by filename.
6. **Every new primitive has a round-trip + log-det + jit test** in `tests/`.
7. **Every new primitive satisfies identity-at-init** where applicable (couplings, LinearTransform).
8. **All tests pass** under both default float32 and `JAX_ENABLE_X64=1`, or carry an explicit `@requires_x64` with a one-line reason.
9. **`USAGE.md` and `REFERENCE.md` mention every new public name.** Undocumented public names are a lint error.
10. **Total LOC in `nflojax/` is still surveyable**: ballpark ≤ 5000 LOC excluding tests. If this number is about to be exceeded, revisit §4 and §8 before adding more.

---

## 12. Glossary

- **Particle system** — an event of shape `(N, d)` interpreted as `N` positions in `d` dimensions. nflojax is agnostic to what those positions mean physically.
- **Solid vs liquid** — a distinction at the *base distribution* level, not the flow level. Solid ↔ `LatticeBase`; liquid ↔ `UniformBox` or a custom point-cloud base.
- **Periodic box / torus** — a rectangular region with periodic boundary conditions. Supported today via `CircularShift`, circular-mode splines, and `utils/pbc`.
- **Conditioner** — the neural module inside a coupling that maps the frozen half to per-scalar bijection parameters. Pluggable; contract defined in `transforms.py`.
- **Context** — any PyTree passed into a flow forward / inverse call and threaded to every conditioner. The generic slot for temperature, density, lattice, species, external field, etc.
- **Event shape vs batch shape** — the trailing `len(event_shape)` axes are event; everything leading is batch. All nflojax primitives use this convention.
- **Identity at init** — the property that `forward(init_params, x) == x`. Every coupling in nflojax ships with this.
- **Reference conditioner** — one of `MLP`, `DeepSets`, `Transformer`, `GNN` shipped with nflojax. Usable out of the box, replaceable at will; *not* authoritative.

---

## 13. Pointers for agents

If you are an agent picking up nflojax work with no context, read these sections in this order:

1. `AGENTS.md` — tactical project guidance (dev commands, known issues, gotchas).
2. §§1–4 of this document — vision, philosophy, scope.
3. §§5, 8 — abstractions and module charter.
4. `REFERENCE.md` — API surface.
5. `USAGE.md` — how to use the API.
6. `EXTENDING.md` — how to add a new primitive the right way.

Before adding code, run the §9 heuristics. Before declaring work done, run the §11 checks.

---

## 14. Review log

Keep a running log of decisions that changed the shape of the library.

- *2026-04-21* — this document drafted after landing `feature/particle-events` (circular splines + `CircularShift`). Established the "normalizing-flow framework for particle systems" framing and the explicit exclusion list (no energies, no training, no observables, no E(3) equivariance, no triclinic yet).
- *2026-04-21* — §2.1 "500 LOC" rule amended. It now applies per *concept*, not per file. `transforms.py` and `nets.py` are explicitly catalogue files; their total size can grow as new entries land, but each entry is still bound to the 500-LOC bar. The amendment matches reality (`transforms.py` is 2.5 k LOC of ten bijections) without loosening the bar on new code.
- *2026-04-21* — `Geometry` dataclass introduced (new `geometry.py`). Axis-aligned box + per-axis periodicity; numpy-backed, not a PyTree. Landed *before* Stage A to avoid retrofitting every geometry-consuming primitive (`Rescale`, `UniformBox`, `LatticeBase`, `utils/pbc` are all future consumers). `CircularShift` retrofitted to carry a single `geometry: Geometry` field; legacy-ergonomic `CircularShift.from_scalar_box(coord_dim, lower, upper)` classmethod preserved for tests / callers.
- *2026-04-21* — `Permutation` generalised with `event_axis: int = -1` (default preserves historic last-axis behaviour). Enables particle-axis shuffles on `(B, N, d)` events. Rank-N particle flows now have their permutation primitive ready.

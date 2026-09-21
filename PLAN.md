# nflojax — Implementation Plan

**Purpose.** A living, actionable plan for evolving nflojax into the flow-side framework described in [DESIGN.md](DESIGN.md). Scope, philosophy, and boundaries are fixed in DESIGN.md; this file tracks *what we do next, in what order, and how we know it's done*.

**Update discipline.**
- Edit freely when plans change. Record significant shifts in the decision log at the bottom.
- A task is `[ ]` pending, `[~]` in-progress, `[x]` done, `[-]` dropped (with a one-line reason).
- Mark a stage complete only when every task passes its acceptance criteria *and* DESIGN.md §11 checks still hold.
- When a stage reveals a new need, add it as a follow-up task or open a parking-lot entry. Never let undocumented work float.

**Relationship to DESIGN.md.** DESIGN.md is *why*. PLAN.md is *what and when*. If a task in PLAN.md requires relaxing a rule in DESIGN.md, update DESIGN.md first, then cite the change here.

---

## 0. Current status

- Branch: `feature/particle-events` — merged work `a5081ea` landed:
  - `boundary_slopes='circular'` on rational-quadratic splines.
  - `CircularShift` rigid-rotation bijector.
  - Internal spline-parameter helpers de-duplicating 6 sites.
  - Gate + context tests on `SplineCoupling` in circular mode.
  - `@requires_x64` skip marker covering 5 pre-existing float32-only round-trip failures.
- **Stage A fully closed** (`c5aca3d`). A1 (`Rescale`, `2d0cf9c`), A3 (`Permutation` event_axis, Stage-0), A4 (context-type story, §5.2 contract narrowed), A2 (`CoMProjection`, Convention (1) zero log-det + `ambient_correction` helper).
- **Stage B fully closed** (`fa682da`). B1 (`UniformBox`), B4 (`utils/pbc.py`), B2 (`utils/lattice.py` — 5 generators), B3 (`LatticeBase` + 5 factories). New `nflojax/utils/` subdir.
- **Stage C fully closed.** C1 (`circular_embed`), C2 (`positional_embed`). New `nflojax/embeddings.py` (stateless feature transforms for conditioner inputs).
- Test infrastructure: serial `pytest tests/` is the default fast contract
  (`addopts = "-q -m 'not slow'"` in `pyproject.toml`); `pytest-xdist`
  remains an optional speedup via `pytest -n auto tests/`. Slow integration and
  full-Jacobian proof tests run only when explicitly selected. X64 precision
  proofs are selectable via `JAX_ENABLE_X64=1 pytest -m requires_x64 tests/`;
  full float32/x64 is reserved for stage-close/release checks.
- **Stage D fully closed.** `SplitCoupling.flatten_input` hatch unlocks structured-input conditioners. D1 (`DeepSets`, permutation-invariant), D2 (`Transformer`, pre-norm, permutation-equivariant per-token), D3 (`GNN` with top-K neighbours under PBC, `num_neighbours=12`), D4 (shared contract fixture `tests/test_conditioner_protocol.py`). §10.4 and §10.5 resolved (see decision log). Top-level `Dense(..., name="dense_out")` convention across all four conditioners; `SplitCoupling._patch_dense_out` infers the bias size from the conditioner so flat-output and per-token-output modules plug in the same way.
- Current verification on the audit-remediation working tree:
  **646 passed / 10 skipped** under float32 via
  `/home/statah/miniconda3/envs/autodiff/bin/python -m pytest -o addopts='-q' tests/ -p no:cacheprovider`;
  **9 passed / 647 deselected** under targeted x64 via
  `JAX_ENABLE_X64=1 /home/statah/miniconda3/envs/autodiff/bin/python -m pytest -m requires_x64 tests/ -p no:cacheprovider`.
- DESIGN.md checked in; `AGENTS.md` updated to point at it.

- **Stage E fully closed.** E1 (`build_particle_flow`) + E2 (`tests/test_particle_smoke.py`). The builder uses alternating `SplitCoupling.swap` coverage rather than per-layer `Permutation`; `Permutation._zero_logdet` now returns batch shape only for rank-2 particle events, so the earlier accumulator-shape bug is resolved. The `use_com_shift=True` branch appends a private `_CoMEmbed` shim that swaps `CoMProjection.forward` / `.inverse` so `Flow.sample` hits the expansion direction; `EXTENDING.md` now documents that manual-assembly pattern. See the Stage-E decision-log entry for the factory contract and the three kwargs (`required_out_dim`, `out_per_particle`, `n_frozen`).

**Next up: Stage G2 (mW water).** Stage F and the audit-remediation checklist below are closed. Stage G runs in **bgmat-clean** (`../bgmat-clean/`), a clean-room Boltzmann-generator rebuild on top of nflojax. Status as of 2026-09-18:
- **G1 (LJ13 free cluster)** is parked as a post-v1 showcase. MS2 closed as a partial success on DW4.
- **Periodic LJ solid** is closed on its branch-closing gate; the full bgmat LJ-256 match is parked.
- **G2 (mW water, bgmat-clean MS3)** is active, with two north stars (§7): NS1 reproduces bgmat's mW cubic-ice βF/N at N=216, and NS2 (= G3) transfers that model to N=512.
- Its first step, an N=8 head-to-head against bgmat's own `mw_cubic_8`, decides whether Pattern B (§8b) is needed.

See the §11 entry of 2026-09-18.

Branch strategy resolved (§10): `feature/particle-events` was fast-forwarded into `main` on 2026-09-18. New work branches from `main`.

### Audit remediation

Local current-state audit notes live under ignored `audit/`; this checklist is
the committed execution surface.

- [x] **Development contract.** Default pytest no longer requires xdist;
  source/editable install and submodule-only imports are documented; MIT
  `LICENSE` exists.
- [x] **Float32/x64 test policy.** GNN stacked-spline round-trip perturbation is
  an explicit x64 precision proof; default precision has a finite/JIT smoke
  test. X64-only proofs are selectable by pytest marker so agents do not run
  full serial x64 for routine edits.
- [x] **Product-domain hardening.** Added validation coverage for malformed
  domains, masks, base event shape, context shape, dtype policy, circular
  seam behavior, and interval-coordinate log-det autodiff. The default MLP
  feature map remains on `ProductDomain.conditioner_features` for now, but is
  documented as a builder default rather than intrinsic domain semantics.
- [x] **Public-contract tests.** Added smoke tests for flat affine, flat spline,
  product-domain, particle, transform-only, and custom-assembly entry points.
- [x] **Documentation synchronization.** README/AGENTS/REFERENCE/USAGE/
  INTERNALS/EXTENDING now state their source-of-truth roles; REFERENCE has a
  builder decision table and option-support matrix. AGENTS has the same
  builder decision table and option-support matrix; EXTENDING has
  product-domain extension guidance and primitive/builder/pattern/application
  labels. README is intentionally limited to install, quick start, and links.
- [x] **Builder helper consolidation.** Shared the flat-base resolution,
  base event-shape validation, context/gate checks, spline option validation,
  feature-extractor pairing, and builder output packaging paths.
- [x] **Transform catalogue split.** Split the former monolithic transforms
  file into a `nflojax/transforms/` package facade with focused implementation
  modules; public imports from `nflojax.transforms` are preserved for public
  names.
- [x] **F4 source surveyability.** Trimmed `LinearTransform` below 500 LOC,
  split coupling implementations into `nflojax/transforms/couplings/`, and
  split builders into `nflojax/builders/` while preserving facade imports.

### Stage-0 pre-work (landed in the same session, before Stage A)

After the first + second-pass local audits, three "tensions" were lifted directly into the work:

- [x] **Amend DESIGN.md §2.1 "500 LOC" rule** to distinguish single-concept files from package facades/catalogue modules (`nflojax/transforms/`, `nets.py`). Each entry still targets ≤ 500 LOC; catalogue total can grow.
- [x] **Introduce `nflojax/geometry.py`** with the `Geometry(lower, upper, periodic=None)` value object — numpy-backed configuration, not a PyTree. Factory `Geometry.cubic(d, side, lower)`. Derived `box`, `d`, `volume`, `is_periodic()`. Landed *before* Stage A so every upcoming geometry-consuming primitive (`Rescale`, `UniformBox`, `LatticeBase`, `utils/pbc`) targets a single type from day one.
- [x] **Retrofit `CircularShift`** to carry a single `geometry: Geometry` field. `create(key, geometry)` factory; `from_scalar_box(coord_dim, lower, upper)` classmethod for legacy-ergonomic construction.
- [x] **Generalise `Permutation`** with `event_axis: int = -1`. Default preserves historic last-axis behaviour; `event_axis=-2` unlocks particle-axis shuffles on `(B, N, d)`. This satisfies Stage A3 early (see §1 below).

Verification: 401 passed / 5 skipped (float32); 406 passed (x64). Four new `event_axis` tests added.

Deferred from the proposed Stage 0:
- [x] Split `transforms.py` into a `transforms/` package facade with focused
  implementation modules while preserving public imports from
  `nflojax.transforms` for public names.
- [x] Resolve the context-type story. DESIGN.md now uses a two-tier contract:
  built-in MLP conditioners take array context, while custom conditioners may
  use PyTree context through the lower-level flow path.
- [ ] `LinearTransform` remains a single-concept module but is still slightly
  above the 500-line target. Audit whether its long docstring and helper
  structure can be tightened without changing behavior.

---

## 1. Stage A — bijection extensions

Small, high-leverage bijections that every particle-system flow needs. Public
imports live under `nflojax.transforms`; implementations now live in focused
modules under `nflojax/transforms/`.

### Tasks

- [x] **A1. `Rescale(geometry, target=(-1, 1))`** per-axis affine that maps `geometry.box` to the canonical spline range.
  - Dataclass takes a `Geometry` (per Stage-0 retrofit pattern) plus a scalar or per-axis `target` pair.
  - Closed-form log-det (sum of `log(scale_i)` over event axes).
  - Supports arbitrary trailing-axis rescaling; default last axis.
  - Tests: round-trip, log-det-vs-autodiff, jit, `target` default vs explicit.
  - Landed `2d0cf9c`: dataclass with `Geometry` + per-axis `target` + `event_shape`; no params; closed-form log-det; 14 tests green under both dtypes.
- [x] **A2. `CoMProjection(event_axis=-2)`** (replaces the earlier `ShiftCenterOfMass` proposal per audit §9.3).
  - `(N, d)` ↔ `(N-1, d)` bijection: subtract the mean along `event_axis`, drop the last slot, reconstruct from the invariance.
  - Log-det: constant correction `-½ · d · log(N)` (or whichever sign the convention demands; derive explicitly and document on the class).
  - Blocked on picking one CoM strategy; DM / bgmat use different mechanisms. Audit §9.3 recommends B (CoM projection) as the ship-today primitive.
  - Tests: round-trip via the full `(N, d)` ambient space, autodiff-Jacobian determinant sanity on the subspace, jit.
  - **Resolved via Convention (1): log-det is zero on the `(N-1)d` subspace; the `(d/2)·log(N)` volume correction is a caller-applied constant exposed as `CoMProjection.ambient_correction(N, d)`.** Heavy documentation at six contact points: class docstring WARNING block, `REFERENCE.md` subsection with decision box, `USAGE.md` pointer + recipe, new `EXTENDING.md` §"CoM handling" with augmented-coupling alternative and a **do-not-stack** warning, `INTERNALS.md` full derivation (Gram matrix `I + 11^T`, `det = N`), `AGENTS.md` Gotcha one-liner. 12 tests green under both dtypes.
- [x] **A3. `Permutation` generalised to non-last axes.** Landed in Stage-0. `event_axis: int = -1` default preserves last-axis behaviour; `event_axis=-2` shuffles particles on `(B, N, d)`. 4 new tests.
- [x] **A4. Context-type story.** First-pass audit flagged DESIGN.md §5.2 ("context is PyTree") is not matched by the internal gate helper (indexes `.ndim`) or `MLP` (concatenates). Decide: (a) narrow the doc claim to "Array for built-in conditioners, PyTree for custom"; (b) accept PyTree in the built-in path (flatten via `ravel_pytree` in MLP). Update docstrings + `validate_conditioner.validate=False` opt-out accordingly.
  - Tests: if (b), pytree context (dict with two arrays) traces through an MLP conditioner without error; opt-out path covered by a custom-conditioner test.
  - **Resolved: option (a).** DESIGN.md §5.2 rewritten as a two-tier contract (PyTree at flow layer; Array for built-in MLP; PyTree for custom conditioners). `MLP.__call__` docstring tightened. `tests/test_conditional_flow.py::TestCustomConditionerPyTreeContext` locks in the custom-conditioner PyTree path (round-trip + jit).

### Acceptance

- All tasks complete, `pytest tests/` green under both dtype modes.
- Each new bijection mentioned in `REFERENCE.md` and has a one-liner in `USAGE.md`.
- DESIGN.md §11 checks still pass.

### Commit plan

One commit per bijection (A1, A2, A3); one smaller commit for A4.

---

## 2. Stage B — particle-aware base distributions & utils

Bases and geometry helpers that unlock both solids and liquids. Files touched: `nflojax/distributions.py`, new `nflojax/utils/pbc.py`, new `nflojax/utils/lattice.py`.

### Tasks

- [x] **B1. `UniformBox(geometry, event_shape)`** per-axis uniform base.
  - Scalar log-density `-event_factor * sum(log(box))` broadcast over batch; `-inf` for out-of-box `x`.
  - `sample(key, shape)` returns `shape + event_shape`.
  - Tests: sample lies in box, `log_prob` matches closed form, `sample_and_log_prob` consistency, jit.
  - **Landed**: dataclass `UniformBox(geometry: Geometry, event_shape)` (consumes the Stage-0 `Geometry`); `event_factor = prod(event_shape[:-1])` accumulates the constant for rank-N events. 11 tests green under both dtypes.
- [x] **B2. Lattice generators** in `utils/lattice.py`.
  - Pure functions returning `(N, 3)` lattice positions: `fcc`, `diamond`, `hex_ice`, `bcc`, `hcp`.
  - Each takes `n_cells` (int or tuple) and lattice constant(s).
  - Cross-reference `flows_for_atomic_solids/utils/lattice_utils.py` for shape + position agreement.
  - Tests: particle count matches expected, positions inside the claimed box, total volume correct.
  - **Landed**: numpy-backed pure functions; `cell_aspect("fcc")` etc. helper; `make_box(n_cells, a, cell_aspect)` factory for the matching `Geometry`. FCC parity check against the DM unit cell (sorted positions match `1e-12`). 32 tests green.
- [x] **B3. `LatticeBase`** base distribution on top of the generators.
  - One class + five factory methods (`.fcc`, `.diamond`, `.hex_ice`, `.bcc`, `.hcp`).
  - Fields: positions, box, noise scale, optional spherical truncation, optional random permutation.
  - `log_prob` includes `-log N!` when `permute=True`.
  - Tests: `sample_and_log_prob` round-trip, permutation invariance of `log_prob` under `permute=True`, jit.
  - **Landed**: dataclass `LatticeBase(positions, geometry, noise_scale, permute=False)` + 5 `@classmethod` factories. `log_prob` is the labelled Gaussian centred at sites; with `permute=True`, sample shuffles the particle axis per-batch via `jax.vmap(jax.random.permutation)` and `log_prob` subtracts `log(N!)`. **Spherical truncation deferred** (PLAN.md follow-up). 27 tests green.
- [x] **B4. `utils/pbc.py`** orthogonal-box geometry.
  - `nearest_image(dx, box)` — `dx - box * round(dx / box)`; box scalar or per-axis.
  - `pairwise_distance(x, box=None)` and `pairwise_distance_sq(x, box=None)` for `(..., N, d) → (..., N, N)`.
  - Tests: known configurations (two particles on diagonal, ring, lattice), box=None falls back to ordinary distance, jit.
  - **Landed**: `nearest_image(dx, geometry)`, `pairwise_distance(x, geometry=None)`, `pairwise_distance_sq(x, geometry=None)` in `nflojax/utils/pbc.py`. Consumes `Geometry`; non-periodic axes (per `geometry.periodic`) pass through unchanged. 14 tests green.

### Acceptance

- `pytest tests/` green both dtype modes.
- Lattice cell counts and positions match the reference `flows_for_atomic_solids` values bitwise (or within `1e-6` after rescale).
- `USAGE.md` gains a "Particle systems" section showing how to pick a base.
- DESIGN.md §11 checks still pass; no `physics-ish` constants leaked into nflojax.

### Commit plan

One commit per task. B4 before B3 if the lattice factories reuse any PBC helpers.

---

## 3. Stage C — embeddings

Stateless feature transforms used by all non-MLP conditioners. New file `nflojax/embeddings.py`.

### Tasks

- [x] **C1. `circular_embed(x, geometry, n_freq)`** stack of `cos / sin(2π(k+1)(x - lower) / box)`.
  - Shape-preserving except last axis grows `× 2*n_freq`.
  - Vectorised; jit-friendly.
  - Tests: correct shape, periodic output, `n_freq=0` is a degenerate pass-through (or explicit error), jit.
  - **Landed**: API signature changed from `(x, n_freq, lower, upper)` to `(x, geometry, n_freq)` to match the Stage-0 `Geometry`-first convention. `n_freq=0` raises `ValueError` (no silent zero-width output). Non-periodic axes are not gated — caller's responsibility (post-v1 `mask_non_periodic` knob if needed). 8 tests green.
- [x] **C2. `positional_embed(t, n_freq, base=10_000)`** sinusoidal scalar embedding.
  - Output shape `(..., 2*n_freq)`; used for temperature / density / step context.
  - Tests: shape, consistent with the standard sinusoidal positional-encoding formula, jit.
  - **Landed**: standard "Attention Is All You Need" formula adapted for continuous `t`; `n_freq=0` and `base<=0` raise `ValueError`. 7 tests green.

### Acceptance

- Tests green under both dtype modes.
- `REFERENCE.md` mentions both functions; `USAGE.md` shows a one-liner consuming them inside an MLP conditioner.

### Commit plan

One commit.

---

## 4. Stage D — reference conditioners

Permutation-{invariant, equivariant} conditioners in `nflojax/nets.py`. All satisfy the conditioner contract described in DESIGN.md §5.4.

### Tasks

- [x] **D1. `DeepSets(phi_hidden, rho_hidden, out_dim)`** permutation-invariant.
  - `phi` per-particle MLP → sum-pool → `rho` MLP → `dense_out`.
  - `SplitCoupling.init_params` zeroes `dense_out` kernel and patches bias via `identity_spline_bias` for identity-at-init.
  - **Landed**: consumes `(*batch, N_frozen, d)` via `SplitCoupling(flatten_input=False)`. Standalone init via the generic `init_conditioner(key, module, dummy_x)` helper. 10 new tests in `tests/test_nets.py::TestDeepSets` (shape, permutation invariance, context broadcasting, per-sample context, jit, identity-at-init via SplitCoupling).
- [x] **D2. `Transformer(num_layers, num_heads, embed_dim, out_per_particle)`** minimal self-attention stack.
  - Pre-norm residual blocks: `h = h + attn(LN(h)); h = h + ffn(LN(h))`. Flax `nn.SelfAttention` and `nn.LayerNorm`; no masked attention. Per-token `dense_out`.
  - `set_output_layer` is the trivial dict-update (same as `DeepSets`). `SplitCoupling._patch_dense_out` reads the bias length from the conditioner and sizes `identity_spline_bias` to match, so flat and per-token `dense_out` plug in the same way.
  - **Landed**: 9 new tests in `tests/test_nets.py::TestTransformer` (permutation equivariance per-token, SplitCoupling round-trip + identity-at-init, jit, context handling).
- [x] **D3. `GNN(num_layers, hidden, out_per_particle, num_neighbours=12, cutoff=None, geometry=None)`** reference MPNN.
  - Edge index built per-forward via `jax.lax.top_k` on `-d_sq` from `nflojax.utils.pbc.pairwise_distance_sq(x, geometry)` (self-edge pinned to +∞ via `jnp.where`). Messages = Dense-act-Dense over `[h_i, h_j, d_ij]`; aggregate = `jnp.sum` over the neighbour axis; node update = residual MLP. Optional `cutoff` zero-weights distant messages.
  - Permutation-equivariant, **not** SE(3).
  - **Landed**: 10 new tests in `tests/test_nets.py::TestGNN` including permutation equivariance, neighbour-list stability, Euclidean fallback, cutoff behaviour, SplitCoupling round-trip.
- [x] **D4. Shared conditioner contract test fixture** `tests/test_conditioner_protocol.py`.
  - Parametrised over `MLP`, `DeepSets`, `Transformer`, `GNN`. Each asserts: `validate_conditioner` accepts; `apply` returns the `SplitCoupling.required_out_dim` total size; `get_output_layer`/`set_output_layer` round-trip; `SplitCoupling(flatten_input=...)` identity-at-init and jit. `MLP` additionally tested with `SplineCoupling` (flat-mask path).
  - **Landed**: 21 tests (5 contract checks × 4 conditioners + 1 MLP-SplineCoupling).

### Acceptance

- Tests green both dtype modes.
- `REFERENCE.md` documents constructor signatures and equivariance properties.
- DESIGN.md §11 line 4 still holds (no new heavy dependencies).

### Commit plan

One commit per conditioner (D1, D2, D3); D4 in its own commit so the shared fixture lands as a reusable asset.

---

## 5. Stage E — particle-flow builder

A single entry-point that assembles the canonical DM / bgmat topology. Public
facade: `nflojax.builders`; implementation: `nflojax/builders/particle.py`.

### Tasks

- [x] **E1. `build_particle_flow(...)`** topology:
  ```
  Rescale(box -> [-tail_bound, tail_bound])
  for i in range(num_layers):
      SplitCoupling(swap=False, boundary_slopes='circular', conditioner=...)
      SplitCoupling(swap=True,  boundary_slopes='circular', conditioner=...)
      CircularShift(coord_dim=d)
      Permutation(event_axis=-2)
  if use_com_shift:
      prepend ShiftCenterOfMass(event_axis=-2)
  ```
  - Signature: `event_shape=(N, d), box, num_layers, num_bins, conditioner (factory / Module class), boundary_slopes='circular', use_com_shift=False, trainable_base=False`.
  - `conditioner` is a factory (`functools.partial` of a conditioner class or a custom callable returning a conditioner instance).
  - Tests: identity at init on `(B, N, d)`, round-trip, jit.
- [x] **E2. Integration smoke test** `tests/test_particle_smoke.py`.
  - `(N=8, d=3)` flow built with each of `DeepSets` / `Transformer` / `GNN`.
  - Asserts identity at init (tolerant threshold), jit-invertible round-trip, non-zero gradient from a trivial `jnp.sum(x**2)` scalar loss.
  - Purpose: prove all three conditioners compose cleanly with the builder.

### Acceptance

- Smoke test passes both dtype modes.
- `USAGE.md` gains a "Build a particle flow" walkthrough.
- DESIGN.md §11 LOC budget still green.

### Commit plan

One commit for E1 (builder + its own tests), one commit for E2 (cross-conditioner integration test).

---

## 6. Stage F — docs refresh

Post-landing housekeeping.

### Tasks

- [x] **F1.** `USAGE.md` — add "Particle flows" section tying the new primitives together. *Landed progressively in Stages B / C / D / E: "Structured (rank-N) Flows", "Particle bases", "Conditioner features", "Build a particle flow".*
- [x] **F2.** `REFERENCE.md` — add entries for every new public name. *Stage-F sweep added dedicated `### MLP` and `### init_conditioner` sections (both were only mentioned inline before). Every Stage-A–E new public name has a dedicated heading.*
- [x] **F3.** `EXTENDING.md` — augmented-coupling composition pattern and BYO conditioner recipe. *Pattern A rewritten in Stage-E pre-push cleanup; Pattern B expanded from a comment sketch to concrete working code (DiagNormal(event_shape=(2N, d)) + SplitCoupling(split_axis=-2, split_index=N) + MLP) and spot-run end-to-end. BYO conditioner skeleton was already in place — left as the canonical generic recipe.*
- [x] **F4.** `AGENTS.md` — module dependency graph. *Updated in Stage B (embeddings.py, utils/pbc.py, utils/lattice.py all listed).*
- [x] **F5.** `INTERNALS.md` — conditioner-protocol section. *Added "Conditioner protocol" section documenting the minimal contract, the optional identity-init half, the `_patch_dense_out` bias-size auto-inference, and the "reference conditioners are examples, not authoritative" framing (placed before the existing Transformer pre-norm section).*

### Acceptance

- Every new public symbol has an entry in `REFERENCE.md`.
- `USAGE.md` examples actually run (doctest-grade preferred; at minimum spot-executed from the repo).

---

## 7. Stage G — bgmat-clean downstream validation

An off-nflojax deliverable in `../bgmat-clean/`. Not part of nflojax; a proof the abstraction holds when a real Boltzmann-generator application is built on top. Supersedes the earlier "`bgmat/flow_on_nflojax.py` parity test" framing (dropped — see §11 rescope entry 2026-04-23).

### Tasks

- [-] **G1.** Parked 2026-06-15 as a post-v1 showcase: free-cluster LJ13 carries the full `T(3) × SO(3) × S_13` symmetry, which vanilla coupling flows cannot carry. MS2 closed as a partial success on DW4 (§11 2026-04-23). Original task: **LJ13 end-to-end** in `../bgmat-clean/` (bgmat-clean milestone **MS2**). 13-atom free-cluster Lennard-Jones at kT = 0.1 with DeepSets conditioner. Two variants trained and compared:
  - **Variant A** — Pattern A from `EXTENDING.md` (`CoMProjection` + `_CoMEmbed` shim on reduced (12, 3) subspace) + training-time S₁₃ permutation augmentation.
  - **Variant C** — ambient (13, 3) flow + soft CoM-spring `(k/2)|CoM|²` in the energy (full structural S₁₃ via DeepSets).
  First real external use of the `EXTENDING.md` Pattern A recipe. First non-periodic application of any nflojax primitive. Bypasses `build_particle_flow` (which hard-wires `CircularShift`) via manual `assemble_flow`.
- [x] **G1b. Periodic LJ solid** (added 2026-06-15, closed 2026-09-18). A branch-closing periodic validation in `../bgmat-clean/lj_solid/`: torus flow on `build_particle_flow` (circular splines + `CircularShift` + `LatticeBase.fcc`) against `jax_pdf.PeriodicLennardJones`, which is bit-for-bit equal to bgmat's energy. Gate met: stable training to physical energy, ESS rising to about 2% at N=32, and the two nflojax fixes it surfaced landed (§11 2026-06-16). The full bgmat LJ-256 match is parked (`../bgmat-clean/docs/gpu-pattern-b-plan.md`).
- [~] **G2.** **mW water port** in `../bgmat-clean/` (bgmat-clean milestone **MS3**). Reproduces bgmat's monatomic-water results using nflojax's `LatticeBase.diamond` (cubic ice; `hex_ice` later), a GNN conditioner, and `build_particle_flow` (periodic path). Replaces the old parity-test idea.
  - **North star NS1:** absolute βF/N of mW cubic ice at N=216 matches the reference −25.082 within its error bar. bgmat's pretrained model gives −25.083, with 27.6% ESS in the joint (augmented) space.
  - First step: an N=8 head-to-head of a non-augmented nflojax flow against bgmat's `mw_cubic_8` at a matched budget. It decides the Pattern B question (§8b).
- [ ] **G3.** **Transferability** (bgmat-clean milestone **MS4**). Train at one N, evaluate at a larger N.
  - **North star NS2:** the N=216 model, evaluated at N=512, matches the reference −25.062. bgmat gives −25.061 with 4.2% ESS.

### Acceptance

- Each milestone runs end-to-end in `bgmat-clean/` without requiring a change to nflojax source.
- If a gap surfaces (primitive missing, contract awkward, docs misleading), file it as a PLAN.md follow-up task in §1–§5 or §9 and fix it **before** declaring the milestone closed.
- Retrospective entry in §11 after each milestone closes.

### Commit plan

Lives in `../bgmat-clean/` (separate repo, separate commit history). Track milestones here so they aren't forgotten; track per-milestone nflojax-side feedback in §11.

---

## 8. Cross-cutting checklist (run after every stage)

Stage A checklist (2026-04-21):

- [x] `pytest tests/ -q` green under default float32 (428 passed / 6 skipped; the 6 float32 skips carry `@requires_x64`).
- [x] `JAX_ENABLE_X64=1 pytest tests/ -q` green (434 passed).
- [x] Every new primitive satisfies identity-at-init where applicable (`Rescale` with target=source is identity; `CoMProjection` is non-learnable).
- [x] No new energy / training / observable term (DESIGN.md §11 greps).
- [x] No new heavy dependency (DESIGN.md §4 item 10). `pytest-xdist` added to **test** extras only, not runtime deps.
- [x] Total nflojax LOC (excluding tests) ≤ 5000.
- [x] Every public name in `REFERENCE.md` (`Rescale`, `CoMProjection`).

Stage C checklist (2026-04-22):

- [x] `pytest tests/ -q` green under default float32 (527 passed / 6 skipped).
- [x] `JAX_ENABLE_X64=1 pytest tests/ -q` green (533 passed).
- [x] Identity-at-init not applicable (embeddings are stateless, no params).
- [x] No new energy / training / observable term (DESIGN.md §11 greps).
- [x] No new dependency.
- [x] Total nflojax LOC (excluding tests) ≤ 5000.
- [x] Every public name in `REFERENCE.md` (`circular_embed`, `positional_embed` under `nflojax.embeddings`).

Stage E checklist (2026-04-22):

- [x] `pytest tests/ -q` green under default float32 (expected ~600 passed / 7+ skipped — Stage-E adds 3 `@requires_x64` skips for `jit_round_trip_after_perturbation` which accumulates RQS-inverse roundoff through Rescale + 4 circular couplings above 1e-3).
- [x] `JAX_ENABLE_X64=1 pytest tests/ -q` green (10/10 Stage-E tests pass).
- [x] Identity-on-couplings holds: `forward(x) == tail_bound * x` at init (Rescale scaling only; couplings identity via `SplitCoupling._patch_dense_out`).
- [x] No new energy / training / observable term (DESIGN.md §11 greps).
- [x] No new heavy dependency. `build_particle_flow` reuses existing primitives.
- [x] Every public name in `REFERENCE.md` (`build_particle_flow`) and `USAGE.md` (liquid + solid recipes, both spot-run end-to-end).

Stage D checklist (2026-04-22):

- [x] `pytest tests/ -q` green under default float32 (592 passed / 6 skipped).
- [x] `JAX_ENABLE_X64=1 pytest tests/ -q` green (598 passed).
- [x] Every new conditioner satisfies identity-at-init: `DeepSets`, `Transformer`, `GNN` each pass their SplitCoupling round-trip at init. Covered in `tests/test_conditioner_protocol.py::test_split_coupling_identity_at_init` (parametrized).
- [x] No new energy / training / observable term (DESIGN.md §11 greps). Docstring mentions of "training" / "energy" (in `CoMProjection`) are pre-existing.
- [x] No new heavy dependency. All three conditioners use `flax.linen` primitives already in scope.
- [!] Total `nflojax/` LOC excluding tests: **6653**. Exceeds the §11 item 10
  ballpark of 5000. This was later mitigated by splitting the former
  monolithic transforms file into focused modules while preserving the
  `nflojax.transforms` public facade. The later F4 pass split coupling
  implementations and builders into focused submodules; `nets.py` remains the
  main catalogue file to keep an eye on.
- [x] Every public name in `REFERENCE.md` (`DeepSets`, `Transformer`, `GNN`, `init_conditioner`, new `SplitCoupling.flatten_input` field).

---

## 8b. Long-term trajectory

Stages A–G get nflojax to the point where a user can reassemble the DeepMind *Flows for Atomic Solids* paper and bgmat's mW flow with no code in the library that is specific to either. That is the bar for **v1.0**.

### v1.0 — "particle-flow framework"

Definition:

- Stages A–G all pass their acceptance criteria.
- A user can build the DM paper's topology with nflojax primitives + a user-side Transformer conditioner + user-side energy / training / observables, in ≲ 500 lines of app-side code.
- A user can build `bgmat-clean`'s flows (DW4 free cluster at MS2, periodic LJ solid, mW water at MS3) using nflojax primitives + application-side GNN conditioner + application-side energy / training / marginal-inference, without modifying nflojax. LJ13 moved to a post-v1 showcase on 2026-06-15 (§7 G1).
- Doc set (DESIGN / PLAN / BACKGROUND / REFERENCE / USAGE / AGENTS / EXTENDING / audit) is self-contained for a fresh agent. `DESIGN.md` §11 checks still pass.
- No energy, no loss, no training loop, no observable, no physics constant inside `nflojax/`.

Scope signature at v1.0:
- Coupling flows (not autoregressive, not CNF, not flow matching).
- Event shape `(N, d)` with trailing event axes; rank-polymorphic composition.
- Bases: Gaussian, diagonal Gaussian, uniform-on-box, Gaussian-perturbed lattice (FCC / diamond / hex-ice / BCC / HCP).
- Equivariance: permutation (Sn) via base + conditioner cooperation; translation (T(d)) via `CoMProjection` or augmented-coupling pattern; gauge via `CircularShift`. No O(d) / SO(d) / point group.
- Conditioners: `MLP` + at least `DeepSets` reference shipped (audit §12.3 may defer Transformer / GNN to v2).
- Orthogonal boxes only (triclinic parking lot).

### Post-v1 — conditional roadmap

Each post-v1 item lands only if a named trigger fires. No speculative extensions.

- **Transformer / GNN reference conditioners** — if a third-party application other than DM / bgmat asks for one of them, ship it. Otherwise stay at `DeepSets`. Audit §12.3.
- **Triclinic boxes** — if bgmat stabilises its triclinic path and a downstream app asks, generalise `Geometry` to carry an optional `cell: Array | None` field and retrofit every consumer. Audit §4 item 8.
- **Pattern B promoted to a primitive** — *trigger is now partially fired* (bgmat-clean MS2g showed axis-split `SplitCoupling` can't match LJ13-type targets; `EXTENDING.md` Pattern B is the documented escape route and already has one consumer in bgmat). The promotion is: a `build_augmented_flow(*, base, num_layers, conditioner, ...)` builder in `nflojax.builders`, a `marginalise_aux_half(...)` inference-time helper, and the private `_CoMEmbed` shim kept as-is (augmented flows don't need it). Full trigger fires when a **second external consumer** requests augmented coupling — expected to be bgmat-clean MS2h.Variant-D on LJ13 or MS3 mW if `SplitCoupling` + GNN alone under-performs. *Status 2026-09-18:* the periodic LJ solid argued for it (§11 2026-06-16), but its full match is parked. The deciding evidence is now the bgmat-clean MS3 N=8 mW head-to-head: a non-augmented nflojax flow against bgmat's augmented `mw_cubic_8` at a matched budget. Estimated scope: 2–3 days. Strictly precedes the E(n)-equivariant-coupling item below: Pattern B fixes `S_N` without touching `SO(d)`, and is far cheaper.
- **E(3) / SE(3) bijections** — *trigger is not yet fired*. Requires (a) Pattern B primitive landed; (b) a downstream application that still misses its success criteria **specifically** because of broken `SO(d)` equivariance, not because of `S_N` (Pattern B should close `S_N` on its own). bgmat-clean LJ13 and DW4 do not yet establish this — LJ13 is blocked on Pattern B first; DW4's reverse-KL mode-ratio gap is objective-bias, not rotation. If triggered, introduce `transforms/equivariant.py` with EGNN-style couplings under the existing `nflojax/transforms/` package. Estimated scope: multi-week; DESIGN.md §4 item 7 and §7.3 document why this is non-trivial.
- **Block permutation / heteronuclear lattices** — if a multi-species materials application lands, generalise `Permutation` and `LatticeBase`. Audit §7.5.
- **Flow matching / diffusion** — **not** shipped in nflojax; a sibling library. Audit §12.17.

### Stopping / sunsetting criteria

nflojax is a niche tool. It is acceptable to archive / sunset it if any of the following happen:

- **Paradigm shift.** Flow matching or diffusion replaces coupling flows for materials Boltzmann sampling and no downstream application is still training reverse-KL coupling flows. Audit §12.1.
- **Ecosystem absorption.** Distrax, `tfp.bijectors`, or a successor adopts first-class PBC / torus support + rank-N event handling. At that point, nflojax becomes vestigial.
- **No usage.** Six months with no active downstream user and no planned user. Keep the branch, stop maintaining.
- **Scope drift.** If one maintainer can no longer read the library end-to-end in an afternoon, the scope has drifted — revisit `DESIGN.md` §4 as a family (per the "revisit as family" clause).

### Review cadence

- **After each stage.** Write a one-paragraph retrospective in `DESIGN.md` §14 review log. What landed, what was deferred, what was learnt.
- **After Stage E.** Design review against the audit's amendment list. Are the primitives carrying the weight we thought they would? Any that should be merged, deprecated, renamed?
- **After Stage G.** v1.0 release decision. If bgmat prototype passes parity, tag v1.0 and freeze the API for at least one minor-version cycle. If it fails, add the missing primitive to Stage A' and repeat.
- **Every 6 months.** Cross-check against the stopping criteria above.

### North-star test

A concrete, falsifiable success criterion: **a graduate student who has never used nflojax can read the docs in half a day and reproduce a known Boltzmann-generator result** (e.g. DeepMind's 32-particle LJ run) using nflojax primitives + ≲ 500 lines of their own code. If that test fails, the library has a discoverability or abstraction problem.

---

## 9. Parking lot

Things explicitly deferred. Each entry has a one-line reason.

- **Triclinic / non-orthogonal boxes.** Lands only once orthogonal + triclinic share a clean API and bgmat's WIP settles. See DESIGN.md §4 item 8.
- **SE(3) / E(3) equivariant conditioner.** User brings EGNN / NequIP / MACE when needed at the *conditioner* level. The *coupling*-level equivalent is §8b "E(3) / SE(3) bijections" (trigger-gated, needs Pattern B promoted first). DESIGN.md §4 item 7.
- **Block permutation (multi-species).** Waits for a concrete multi-species application. DESIGN.md §7.5.
- **Heteronuclear lattices.** Same. DESIGN.md §7.5.
- **Augmented-coupling primitive (Pattern B).** *Trigger partially fired by bgmat-clean MS2g*; promotion path documented in §8b. Today it stays a recipe in `EXTENDING.md`; full promotion waits on a second external consumer (expected at bgmat-clean MS2h.Variant-D or MS3). As of 2026-09-18 the decision rests on the MS3 N=8 mW head-to-head (§8b). DESIGN.md §4 item 9.
- **Long-range energy support (Ewald / PPPM).** Energy-side; not flow-side. DESIGN.md §7.6.

---

## 10. Open questions

Items that need a decision before the relevant stage can close.

- [x] Branch strategy for Stages A–F. One branch `feature/particle-flow-framework`, or stage-per-branch? (Default: one long-lived branch, stage-per-commit.) **Resolved 2026-09-18: the default.** One long-lived branch (`feature/particle-events`), stage-per-commit, fast-forwarded into `main` after the Stage-G periodic closure with the full float32 and float64 suites green. Later work uses short branches off `main`.
- [x] `ShiftCenterOfMass` log-det convention: document as "zero on the (N−1)d subspace; caller is responsible for the embedding-space correction" vs. "constant $-d \log(N)/2$ correction baked in"? Pick when writing A2. **Resolved 2026-04-21: Convention (1) — zero log-det on the subspace; caller applies `CoMProjection.ambient_correction(N, d) = (d/2)·log(N)` when an ambient density is needed. Rationale + full derivation in §11 decision log and `INTERNALS.md` "CoM Projection and the Volume Correction".**
- [x] `LatticeBase.hex_ice` unit-cell parameters: follow DM's convention (8 atoms per cell) or bgmat's (re-derive)? Likely DM. **Resolved 2026-04-22: DM convention. 8 atoms per orthorhombic cell with `cell_aspect = (1, sqrt(3), sqrt(8/3))` and the puckering parameter `6 * 0.0625` baked in (matches `flows_for_atomic_solids/models/particle_models.py:HexagonalIceLattice`). Atom positions reproduced inline in `nflojax/utils/lattice.py` so the test suite is hermetic.**
- [x] `Transformer` attention norm placement: pre-norm (more stable for deeper stacks) vs. post-norm (closer to DM's original). **Resolved 2026-04-22: pre-norm.** `h = h + attn(LN(h)); h = h + ffn(LN(h))` per block + a final `LN` before `dense_out`. Rationale + background in `INTERNALS.md` "Transformer conditioner: pre-norm choice".
- [x] GNN default `num_neighbours`: 12 (common) vs. 18 (bgmat). **Resolved 2026-04-22: `num_neighbours=12`.** Apps override (bgmat's 18 is app-side, not library default).

---

## 11. Decision log

- *2026-04-21* — Plan drafted alongside DESIGN.md. Adopted: thick-on-flows / thin-on-physics scope; reference conditioner family is MLP + DeepSets + Transformer + MPNN; no energy or training helpers in nflojax.
- *2026-04-21* — Stage A1 closed (`2d0cf9c`). `Rescale` ships as a fixed, non-learnable geometry→canonical affine; `LinearTransform` retains the learnable-affine role. API: `Rescale(geometry, target=(-1, 1), event_shape=None)`; scalar or per-axis target; `event_shape` default `(geometry.d,)` with `event_factor = prod(event_shape[:-1])` so log-det accumulates correctly on rank-N particle events.
- *2026-04-21* — Adopted `pytest-xdist` as default parallel runner (`addopts = "-n auto -q"` in `pyproject.toml`; `pytest-xdist` added to test extras). Rewrote AGENTS.md Dev Commands as a one-command "Testing Strategy" (`pytest tests/` after edits, x64 at stage close). Full suite wall-clock: float32 6:28 → 1:25 (4.5×), x64 12:11 → 1:34 (7.8×). Rationale: cheap full suite removes the agent triage problem; a file→tests mapping would push judgement onto the agent, and agents get that wrong.
- *2026-04-21* — Stage A4 closed via option (a): narrow the built-in contract, keep PyTree at the flow layer. DESIGN.md §5.2 now a two-tier contract (PyTree through flows, Array for built-in `MLP` and the internal gate helper, any PyTree for custom conditioners). Option (b) rejected because target apps (DM, bgmat) bring their own conditioner anyway, so `ravel_pytree` + PyTree-aware batching in the common path would add complexity for no one. A new `TestCustomConditionerPyTreeContext` test in `tests/test_conditional_flow.py` exercises a dict-context conditioner end-to-end (round-trip + jit).
- *2026-04-21* — Stage A2 closed; Stage A fully done. `CoMProjection` ships with **Convention (1)** log-det: the bijection is a relabelling between two `(N-1)d`-dim spaces (reduced Euclidean and zero-CoM subspace of `R^(Nd)`), so `log_det = 0` both directions. The volume-element constant relating the two embeddings is `(d/2)·log(N)` (derived from `det(I + 11^T) = N` for the parameterisation `x_N = -Σy_i`), exposed as `CoMProjection.ambient_correction(N, d)`. Convention (2) — baking the constant into the log-det — was rejected: it silently double-counts in the augmented-coupling composition (bgmat's pattern), where translation invariance is handled separately and densities are already ambient-valid. Explicit caller-applied correction keeps the two patterns cleanly separable. Heavy documentation placed at six contact points to prevent silent misuse: class docstring, REFERENCE.md decision box, USAGE.md recipe, EXTENDING.md "CoM handling" (with do-not-stack warning), INTERNALS.md derivation, AGENTS.md Gotcha.
- *2026-04-22* — **Stage B closed.** Four tasks landing in one session: `UniformBox` (B1, per-axis uniform base on a `Geometry`), `utils/pbc.py` (B4, `nearest_image` + `pairwise_distance(_sq)` consuming `Geometry`), `utils/lattice.py` (B2, pure functions for `fcc / diamond / bcc / hcp / hex_ice` returning `(N, 3)` numpy positions), `LatticeBase` + 5 factories (B3). All consume the Stage-0 `Geometry` value object — no raw `lower/upper` alternatives. `LatticeBase.permute=True` shuffles particle order per-batch via `jax.vmap(jax.random.permutation)` and subtracts `log(N!)` from `log_prob`; the constant has the same caveats as `CoMProjection.ambient_correction` (no gradient effect, matters for absolute densities / ESS / `logZ`). §10.3 resolved with the DM `hex_ice` convention. New `nflojax/utils/` subdir; AGENTS.md dependency graph extended. 84 new tests, 5 minutes total session wall-clock for the Stage. Spherical truncation in `LatticeBase` deferred — no concrete trigger in v1.0 scope.
- *2026-04-22* — **Stage C closed.** Two stateless feature transforms in new `nflojax/embeddings.py`: `circular_embed(x, geometry, n_freq)` (per-coord Fourier features on a periodic box, lowest harmonic tiles `geometry.box`) and `positional_embed(t, n_freq, base=10_000)` (sinusoidal scalar embedding, transformer-style). Both raise `ValueError` on `n_freq=0` to avoid silent zero-width outputs that would break downstream `jnp.concatenate`. **API change vs. PLAN.md spec**: `circular_embed` takes `Geometry` (not raw `(lower, upper)`) to match the Stage-0 retrofit pattern — one path, no overload. Non-periodic axes are not gated; documented as caller's responsibility. 15 new tests; ~80 LOC. Unblocks Stage D conditioners (Transformer, GNN) which both consume these features.
- *2026-04-22* — **Stage D closed.** One preparatory change + three reference conditioners + one shared fixture. The preparatory change: `SplitCoupling.flatten_input: bool = True` hatch (default preserves the flat-`(B, N*d)` contract MLP expects; `False` passes the structured `(B, N_frozen, d)` slice through to permutation-aware conditioners). Four new public names in `nflojax.nets`: `DeepSets` (permutation-invariant aggregator), `Transformer` (pre-norm multi-head self-attention, permutation-equivariant per-token), `GNN` (top-K PBC-aware message passing, `num_neighbours=12` default). All three satisfy the existing conditioner contract (`context_dim` attribute + `apply` + `get_output_layer`/`set_output_layer`) and use a top-level `Dense(..., name="dense_out")` to stay compatible with `SplitCoupling._patch_dense_out`. §10.4 resolved (pre-norm), §10.5 resolved (12 neighbours). New `tests/test_conditioner_protocol.py` locks the contract at 5 checks × 4 conditioners; per-conditioner detail in `tests/test_nets.py`. Two subtle-bug fixes during implementation: (1) self-mask computed via `jnp.where(eye_bool, inf, d_sq)` to avoid `0 * inf = NaN` off-diagonal; (2) test N bumped to 8 particles so `num_neighbours=3` stays < N_frozen=4 at init.
- *2026-04-22* — **Post-close refactor (P1).** Dropped the `set_output_layer` slicing magic from `Transformer` and `GNN`. Instead, `SplitCoupling._patch_dense_out` (and `SplineCoupling._patch_dense_out` for symmetry) now reads the conditioner's current `dense_out` bias length and sizes `identity_spline_bias(num_scalars = bias_size // params_per_scalar, …)` to match. Works for flat and per-token dense_out uniformly because `identity_spline_bias` is a per-scalar pattern tiled across scalars. Net: `Transformer.set_output_layer` / `GNN.set_output_layer` collapsed to the trivial dict-update form (same as `DeepSets`); one Gotcha removed from AGENTS.md; one "library-private convention" line struck; a bias-shape divisibility check added in both `_patch_dense_out` sites. Reason: the slicing hack was a hidden coupling between conditioners and `SplitCoupling`'s internals; inferring from the conditioner keeps the contract local and easier to extend.
- *2026-04-22* — **Post-close hardening.** Four audit items landed as separate changes: (1) `Transformer` uses `nn.MultiHeadDotProductAttention` instead of the now-deprecated `nn.SelfAttention`; (5) `SplitCoupling.init_params` runs a one-sample dummy apply after `_patch_dense_out` and raises a clear, diagnostic `ValueError` when the conditioner's output total-trailing size doesn't match `transformed_flat · params_per_scalar` — catches per-token `Transformer`/`GNN` misconfigured for asymmetric splits at init rather than as a cryptic reshape error at forward time; (4) removed per-conditioner factories `init_deepsets`/`init_transformer`/`init_gnn` (~75 LOC) in favour of a single generic `init_conditioner(key, conditioner, dummy_x, dummy_context=None)` helper (5 LOC) — shrinks the public API, removes three redundant public names, and keeps all init paths uniform; (3) new `tests/test_particle_integration.py` parametrised over `DeepSets`/`Transformer`/`GNN` composing four alternating-swap `SplitCoupling` layers through `CompositeTransform` — asserts identity-at-init, jit round-trip, and non-zero gradient. Reason: each item brings surface down or failure-mode clarity up; together they move Stage D from "works" to "robust + discoverable".
- *2026-04-22* — **Stage E closed.** `build_particle_flow` + cross-conditioner smoke tests landed. Two composition decisions matter for future work:
  1. **No per-layer `Permutation` in the builder.** Alternating `swap` on `SplitCoupling` already covers all particles, so the builder does not add a separate particle-axis permutation between layers. `Permutation._zero_logdet` now returns batch shape only for `(B, N, d)` events with `event_axis=-2`, so a future builder can reintroduce particle permutations if a concrete design calls for them.
  2. **Added a private `_CoMEmbed` shim for `use_com_shift=True`.** `CompositeTransform.forward` applies `block.forward` sequentially; `CoMProjection.forward` reduces `(N, d) → (N-1, d)` (the wrong direction for `Flow.sample`, which goes base-reduced → data-ambient). `_CoMEmbed` flips `.forward`/`.inverse` so the expansion is what `transform.forward` sees at the tail, and the reduction is what `transform.inverse` sees at the head. `EXTENDING.md` now documents the same direction-flipping pattern for manual assembly; `_CoMEmbed` stays private until a second consumer needs a public helper.
  **Conditioner factory contract**: keyword-only callable with three kwargs (`required_out_dim`, `out_per_particle`, `n_frozen`) computed per-layer by the builder. With asymmetric splits (odd `N_eff` under `use_com_shift=True`), `required_out_dim` differs between `swap=False` and `swap=True` layers, so the factory is called with different sizing per-layer — correctly handled by the per-layer recompute inside the swap loop. Per-token conditioners (`Transformer`, `GNN`) require even `N_eff` so `N_frozen == N_transformed`; the Stage-D `SplitCoupling.init_params` sizing check catches the mismatch with a clear diagnostic error. **Stage-E skip count**: 3 new `@requires_x64` skips on the jit round-trip test across the three conditioners (circular RQS-inverse through Rescale + 4 stacked couplings accumulates ~2.4e-3 float32 roundoff, above the 1e-3 round-trip atol). All pass under `JAX_ENABLE_X64=1`. Total Stage-E LOC: ~230 in the particle builder implementation (including the `_CoMEmbed` shim + the builder body + docstring), ~150 in `tests/test_builders.py::TestBuildParticleFlow`, ~140 in `tests/test_particle_smoke.py`. Unblocks Stage G bgmat-parity prototype — all nflojax-side v1.0 primitives are in place.
- *2026-04-22* — **Stage-E pre-push cleanup.** Fixed two of the three audit follow-ups, deferred one:
  1. **`EXTENDING.md` Pattern A** rewritten to a two-part recipe: canonical path points at `build_particle_flow(use_com_shift=True)`; manual-assembly path inlines a ~12-line `CoMEmbed` direction-flipping shim (same pattern as the private `_CoMEmbed` inside the particle builder). `_CoMEmbed` deliberately *not* promoted to public API — one consumer today, copy-paste recipe covers manual-assembly users until a second asks. Recipe spot-executed end-to-end before landing.
  2. **`Permutation._zero_logdet`** changed from `shape.pop(event_axis)` to `x.shape[:event_axis]`, so log-det carries batch shape only per DESIGN.md §5.5. Previous rank-2 behaviour (`(B, d)` on `(B, N, d)` with `event_axis=-2`) was locked in by `test_event_axis_particle`; assertion updated to `(B,)` with an explanatory comment. Side benefit: unblocks future reintroduction of a per-layer `Permutation` inside `build_particle_flow`, though alternating-swap already covers particles so that remains a separate design decision.
  3. **Deferred**: overlap between `test_particle_integration.py` (raw `SplitCoupling` + `CompositeTransform`) and `test_particle_smoke.py` (builder). Both kept as defense-in-depth at different abstraction layers; revisit once real-regression signal tells us which layer catches bugs first.
  Full suite green both dtypes after the fixes: **607 passed / 9 skipped** under float32, **616 passed** under x64 (unchanged vs. Stage-E close — the fixes don't add or remove tests, only update one assertion).
- *2026-04-22* — **Stage F closed.** Mostly a housekeeping sweep — F1 / F4 had already landed during earlier stages (USAGE.md particle-flow content was written during B / C / D / E; AGENTS.md dep-graph was updated in B). The genuinely new work: (F2) audited REFERENCE.md against the public-symbol list from `grep ^class/^def` across `nflojax/*.py` — every Stage-A–E symbol had a dedicated `###` section; the two gaps were `MLP` (only a table row in the Conditioners intro) and `init_conditioner` (only mentioned inline). Added short dedicated sections for both. (F3) Pattern B (augmented coupling) in EXTENDING.md was a one-line comment stub; expanded to a concrete, spot-run-verified recipe showing `DiagNormal(event_shape=(2N, d))` base + `SplitCoupling(split_axis=-2, split_index=N)` over the physical / auxiliary boundary + inference-time marginalisation, with an explicit do-not-stack-with-Pattern-A note. (F5) New "Conditioner protocol" section in INTERNALS.md documenting the minimal `__call__` + `context_dim` contract, the optional `get/set_output_layer` half, the `SplitCoupling._patch_dense_out` bias-size auto-inference (flat + per-token shapes tile through the same `identity_spline_bias` per-scalar pattern), and the "reference conditioners are examples, not authoritative" framing (no plugin registry, no `isinstance` branches, downstream apps own their conditioner). What was learnt: (1) most Stage-F tasks retroactively turned out to be "already done" — documenting as you land primitives, rather than deferring to a post-stage sweep, was the right call; the sweep caught only two genuinely-missed symbols. (2) Writing the augmented-coupling recipe in nflojax's docs rather than leaving it implicit in bgmat clarified the library / application boundary: the composition pattern is nflojax's; the application-side logic (species-aware GNN, marginal-inference, deterministic aux-half construction) explicitly is not. (3) The conditioner-protocol section was the most load-bearing addition — it captures the "no plugin registry" invariant that a fresh agent would otherwise have to reverse-engineer from `validate_conditioner` + `_patch_dense_out` source reads. Acceptance: **pytest unchanged** (docs-only sweep, no test file touched); augmented-coupling recipe spot-run end-to-end; DESIGN.md §11 checks still clean. Unblocks Stage G downstream-validation work.
- *2026-04-23* — **Stage G rescoped.** The original G1 plan — a single-file `bgmat/flow_on_nflojax.py` that reassembles bgmat's mW flow on nflojax primitives and compares bit-for-bit — was dropped. Replaced by a clean-room rebuild in `../bgmat-clean/` (sibling repo, `main` branch, MS1 "toy flow on `(B, 8, 3)`" already landed). The bgmat-clean roadmap is: **MS2 = LJ13 free cluster** (new Stage-G1), **MS3 = mW water port with GNN conditioner** (new Stage-G2), **MS4 = transferability** (new Stage-G3). Motivation: (a) parity against distrax+haiku bgmat is low-signal because bitwise agreement between two different flax/haiku stacks is fragile and isn't the real question — the real question is whether a Boltzmann-generator application can be assembled on nflojax without touching nflojax source; (b) bgmat is **100% periodic**, so a pure-port workflow never exercises the non-periodic path that `Geometry(periodic=False)` / `CoMProjection` / linear-tail splines were designed for; (c) LJ13 is a classic, cheap, literature-benchmarked non-periodic BG target that stress-tests exactly that gap before we get to periodic mW. MS2 comparison strategy: train both **Pattern A** (`CoMProjection` + `_CoMEmbed` shim + S₁₃ permutation augmentation at training) and **CoM-spring** (ambient flow + `(k/2)|CoM|²` in energy + structural S₁₃ via DeepSets) and measure which trade-off wins empirically. Known friction to watch for during MS2: `build_particle_flow` hard-wires `CircularShift` (not valid for a free cluster) — MS2 bypasses it via `assemble_flow`, and only if a second external consumer appears do we promote `use_circular_shift=False` to the builder API. The `_CoMEmbed` shim copy-paste from `EXTENDING.md` Pattern A gets its first outside-nflojax exercise during MS2; if a second consumer appears, promote to public `nflojax.transforms`. Retrospective entry per milestone close.
- *2026-04-23* — **Stage G1 / bgmat-clean MS2 closed.** LJ13 end-to-end landed in `../bgmat-clean/` across eleven commits (MS2.0 → MS2f.3): flow, training loop, eval harness, DEM reference integration, side-by-side `comparison.md`. Final test counts: **nflojax 608 / 9 skipped** (unchanged — MS2 didn't need library edits beyond the one MS2a non-periodic smoke test), **jax-pdf 128 passed** (after MS2b.1 `(..., n, d)` refactor), **bgmat-clean 101 passed / 1 skipped** (TDD throughout). Findings that feed back into nflojax:
  1. **Pattern A recipe (`EXTENDING.md`) survived first external use.** The ~12-line `_CoMEmbed` shim copy-pasted into `bgmat_clean/lj13/flow.py` worked exactly as documented: identity-at-init, jit round-trip, `CoM(sample) = 0` to machine precision. No corrections needed to EXTENDING.md. Still only one consumer, so keep `_CoMEmbed` private; promote if MS3 (mW port) pulls it in again.
  2. **`build_particle_flow` bypass is fine.** `assemble_flow` + a few manual `SplitCoupling` layers was not noticeably more work than a `--no-circular` flag would be. Don't promote `use_circular_shift=False` to the builder API yet.
  3. **jax-pdf `(..., n, d)` refactor was the right call.** Downstream code passes nflojax events directly to `jax_pdf.LennardJones` with no reshape. The DW4 counterpart got the same treatment gratis. AGENTS.md amendment for the particle-distribution tier was uncontroversial.
  4. **Pre-training parameter alignment is load-bearing.** The `mppt` integration surfaced four silent mismatches against the DEM reference: `epsilon=1` vs `=2` (ordered-pair summation convention), `kT=0.1` vs `=1.0`, `trap_mode='com'` vs `'individual'`, and the shape flip. Fixing them was one commit (MS2f.0) but would have been invisible if we'd started with the eval and noticed the metrics didn't agree. Worth a mention in `EXTENDING.md`: "calibrate your target parameters against the reference before comparing; use the reference paper's energy convention, not the library's default".
  5. **Reverse-KL permutation augmentation is unstable** (`bgmat_clean/lj13/train.py:reverse_kl_loss(perm_aug=...)`). Evaluating `log q` at random `S_N` permutations of flow samples drives `log q` toward a point mass under reverse-KL — at kT=1 the loss went `-231 M → +6.7e21` around step 2.5 k. The trick is standard for forward-KL; under reverse-KL it's a footgun. Worth a one-line warning in `EXTENDING.md` / any future S_N-invariance section of nflojax's docs.
  6. **"DeepSets makes SplitCoupling structurally S_N-invariant" is wrong.** This was my implicit claim for Variant C in the MS2 plan. DeepSets makes the *conditioner* S_N-invariant, but `SplitCoupling` partitions atoms by index (first `split_index` frozen, rest transformed), so the full-flow density is only `S_{N_frozen} × S_{N_transformed}`-equivariant. The MS2f.3 `symmetry_diagnostic` on 32 permutations showed Variant C's `var(log q)` at ≈1.3 × 10⁵ (same order of magnitude as Variant A's ≈1.4 × 10⁵), not zero. Worth correcting in `REFERENCE.md`'s `DeepSets` entry and in `EXTENDING.md` any time S_N claims are made. **True structural S_N invariance in a coupling flow requires a coupling that doesn't partition by index** — i.e. a continuous-normalising-flow or an iteratively-reweighting scheme, not `SplitCoupling`. Post-v1 concern.
  7. **Reverse-KL + LJ at kT=1 still needs a lot of steps.** After 5 k steps (the MS2f.2 target) both variants still over-sample the icosahedral basin (Variant A 85 %, Variant C 66 %, DEM reference 36 %) and both have ESS ≲ 0.001. This is an application-layer concern, not a nflojax issue. The fixes — β-annealing, `linearize_below` on the LJ core, forward-KL warmup from DEM samples — all live in `bgmat-clean`'s MS3 backlog.

  **New nflojax tasks filed** (as PLAN.md follow-up items, not blockers):
  - Add one paragraph to `EXTENDING.md` under a new "S_N and symmetry" subsection: SplitCoupling is not structurally S_N-invariant even with a DeepSets conditioner; perm-aug is forward-KL-only.
  - Add a one-line reminder to `REFERENCE.md` `DeepSets` entry: "conditioner-level invariance; the full-flow invariance depends on how `SplitCoupling` partitions the event axes."
  - Revisit `Permutation._zero_logdet` batch-shape fix (already landed in Stage-E cleanup): cover the case where `event_axis=-2` and `split_axis=-2` interact in a future builder (no concrete trigger yet).

  Stage G1 acceptance (from §7): **met** — the milestone ran end-to-end in `../bgmat-clean/` without requiring any nflojax source change. Only doc additions are queued (items 5–7 above).
- *2026-04-23* — **bgmat-clean MS2g — making LJ13 actually match DEM — partial success.** After MS2f.3 showed both variants were mode-seeking (ESS ~ 0, basin over-sampling 85 % vs 36 % ground truth), MS2g worked through five training strategies to try to hit the success criteria. Outcome: significant improvement over MS2f, but **success criteria not met** — bgmat-clean's LJ13 trainer is "good enough for qualitative pair-distance comparison" and not yet "good enough for importance sampling / free-energy estimation". Details live in `bgmat-clean/AGENTS.md` gotcha #7, but the nflojax-relevant takeaways are:
  1. **No nflojax source changes needed for MS2g either.** All the tricks (forward-KL loss, soft-core log_p floor, `--init-from`, β-annealing via chained runs) lived inside `bgmat-clean/`. The nflojax primitives composed cleanly throughout.
  2. **Reverse-KL is the only stable-and-physical training recipe we found for LJ13.** Forward-KL on DEM samples covers DEM *and* unphysical r → 0 spillover — the flow's image contains the reference points but also a ~20 % volume of close-approach leakage. This is a forward-KL-in-coupling-flows pathology, not a nflojax issue, but worth documenting next to Pattern A / Pattern B in `EXTENDING.md` as a "when forward-KL goes wrong" note.
  3. **Soft-core log_p floor is a generic helper, not LJ-specific.** We implement it inside `bgmat-clean/lj13/train.py::reverse_kl_loss` as an optional kwarg. If a future particle-system app (bgmat-clean MS3 mW, or any other user) sees reverse-KL instability from r → 0 spikes, they'll want the same knob. Candidate for promoting into a nflojax utility if a second consumer appears.
  4. **`_CoMEmbed` / Pattern A continues to work.** At no point in MS2g did Pattern A misbehave; all instability came from the training objective, not the flow architecture. Confirms the Stage-F EXTENDING.md recipe is correct as written.
  5. **What's blocking full success**: the remaining gap (bulk U median 6 units cold, icosahedral recovery 2× DEM's baseline, ESS ~ 0) is a research-level issue for reverse-KL coupling flows on LJ13 — wider/deeper architecture, α-divergence or MMD objectives, FAB-style buffers. None of these are nflojax gaps. Parked as a bgmat-clean follow-up; does not block MS3 (mW port) since mW's energy landscape is less adversarial than free-cluster LJ13.
  Stage G1 "nothing new required in nflojax" acceptance still holds through MS2g.
- *2026-04-23* — **Addendum**: post-MS2g discussion confirmed that the MS2f.4 caveat ("`DeepSets` is only conditioner-level S_N-invariant; `SplitCoupling` with an index partition breaks the full-flow density's S_N symmetry") **is the architectural ceiling that stalled MS2g**. Bulk sample observables track DEM because the flow memorises one chart-aligned orbit; ESS stays ~0 and the symmetry diagnostic stays ~30 k under 32 random S_13 permutations because no training run can synthesise an S_13 invariance the coupling structure doesn't have. Consequence: `REFERENCE.md` DeepSets/Transformer/GNN entries updated with scope-of-symmetry sentences, `EXTENDING.md` gains a "When axis-split coupling isn't enough" subsection pointing at Pattern B / post-v1 E(n)-equivariant coupling, and bgmat-clean MS2 retargets to **DW4** (4 particles, 2D, quartic double-well — multimodal target where the same `S_{N_frozen} × S_{N_transformed}` ceiling applies but is tolerable because DW4's bulk distribution doesn't require strict S_N-invariance to be matched on first-moment observables). LJ13 returns as a post-v1 target when Pattern B or E(n)-equivariant flows become available.
- *2026-04-23* — **bgmat-clean MS2h (DW4) closed, MS2 = partial success.** The retarget worked. DW4 is a cleanly trainable benchmark for vanilla coupling flows: **Variant A hits 3 of 4 MS2 success criteria** at 20 k reverse-KL steps. Best numbers (eval at `runs/dw4/comparison.md`):
  - **U median** (pure DW pair energy): Variant A −22.89, Variant C −22.68, DEM −22.80 → within 0.12 of DEM for both (target was ±0.5, **hit**).
  - **Mode coverage** (fraction of samples with a pair in the far-mode band): A 0.997, C 0.993, DEM 0.985 → both within 0.02 of DEM (target ±0.05, **hit**).
  - **ESS** (self-normalised importance weights): A 0.135, C 0.001 → Variant A **hits** the ≥ 0.1 bar; Variant C does not.
  - **Pairwise-distance L1 vs DEM**: A 0.296, C 0.300 → **miss** the < 0.1 bar. Reverse-KL places ~+9 pp at the short-mode peak (r ≈ 2.75) and ~−7 pp at the far-mode peak (r ≈ 5.25) relative to DEM. This is the expected reverse-KL mode-seeking bias and would close with α-divergence / FAB / forward-KL warmup — out of v1.0 scope.
  Stage G1 acceptance is met on DW4: **zero nflojax source changes required** throughout MS2a–MS2h (the only nflojax commit is the `tests/test_particle_smoke.py::test_free_cluster_assemble` addition from MS2a and the three docs files in MS2h.0). The shared training helpers landed cleanly in `bgmat_clean/train.py` (MS2h.1) and were reused verbatim by DW4. Total cross-repo test count: nflojax 608+9, jax-pdf 128, bgmat-clean 168+2 = **906+11 passing**.
  Findings for nflojax that feed forward:
  - **`_CoMEmbed` survived a second consumer.** MS2h.3 copy-pasted the 12-line shim from `lj13/flow.py` to `dw4/flow.py` unchanged. Both consumers are application-side; promoting `_CoMEmbed` to `nflojax.transforms` is now a reasonable v1.1 consideration if a third consumer appears or if `build_particle_flow(use_com_shift=True)`'s private `_CoMEmbed` gets reused externally.
  - **`_fkl_batch` dimension-agnosticism fix.** Moving the CoM-noise helper to `bgmat_clean/train.py` surfaced that the LJ13 version hardcoded `spatial_dim=3`. Fixed to read `shape[-1]` from the reference; now works for DW4 (d=2) with no changes. No parallel fix needed in nflojax — this helper lives application-side.
  - **The L1 mode-ratio gap remains a research direction.** For a post-v1 milestone (or as bgmat-clean MS2.x): test whether α-divergence or FAB-style training closes the +9 / −7 pp ratio gap; if so, the same fix would apply to LJ13 with Pattern B.
  **MS2 verdict: closed as partial success.** DW4 demonstrates that nflojax's primitives compose into a working, useful density estimator for multimodal particle targets without any library-side changes. LJ13 remains parked as a post-v1 target. MS3 (mW water port) is unblocked and can proceed against the larger periodic architecture where `GNN` is designed to shine.
- *2026-04-23* — **AGENTS.md + USAGE.md cross-reference sweep.** Added a "Sibling repos" section to AGENTS.md pointing at `../jax-pdf/` (benchmark target log-densities; `LennardJones` / `DW4` plug into `build_particle_flow`'s `(..., n, d)` events with no reshape) and `../bgmat-clean/` (Stage G application repo). Added a short "Benchmark targets" pointer in USAGE.md's `Build a particle flow` section linking to `bgmat-clean/lj13/` and `bgmat-clean/dw4/` as worked examples. Zero code changes; pure discoverability. Motivation: a fresh agent session found these repos only by reading PLAN.md's decision log, which is both too far down and too long. A top-level pointer in AGENTS.md plus one in USAGE.md means downstream users land on the right cross-references on the first read.
- *2026-06-16* — **Stage-G periodic validation: torus LJ-solid flow works; two nflojax robustness fixes landed (first source changes this milestone needed).** Built a periodic Lennard-Jones solid flow in bgmat-clean on `build_particle_flow` (circular splines + `CircularShift` + `LatticeBase.fcc`) against a new `jax_pdf.PeriodicLennardJones` oracle (bit-for-bit equal to bgmat's `LennardJonesEnergy`, now a committed differential test). The attempt surfaced two genuine fixes: (1) `build_particle_flow` **raises** if a periodic `Geometry` is paired with `boundary_slopes='linear_tails'` — a periodic target on unbounded tails is improper (invariant under per-particle box translations -> infinite copies) and reverse-KL diverges to infinite entropy; (2) the `GNN` neighbour distance now uses `sqrt(d_sq + 1e-12)` — `sqrt(0)` has an infinite gradient that NaN-ed reverse-KL the moment the flow sampled a coincident pair (forward stayed finite, hiding it). Both tested; discoverability follow-ups landed (AGENTS gotchas, REFERENCE crystalline-solid recipe rewrite, USAGE periodic note + free-cluster relabel). **Phase-2a (CPU, non-augmented) findings:** the flow reaches physically-correct energy (betaU/N ~ -1.13, near equilibrium) but ESS plateaus ~1-2% at N=32; the conditioner is NOT the bottleneck at that size (GNN ~= DeepSets once lr is in bgmat's ~7e-5 regime; lr 1e-3 collapses the GNN / over-spreads DeepSets); training must use the forward `log_q` (the inverse is unreliable at the box seam for a Gaussian `LatticeBase` on a torus). The full match (bgmat's 5.84% joint ESS at N=256, 1M steps, augmented coupling) needs **GPU + Pattern B** — this fires the §8b "promote Pattern B" trigger (a second concrete consumer beyond bgmat). Handoff plan committed at `../bgmat-clean/docs/gpu-pattern-b-plan.md`.
- *2026-09-18*: **Periodic LJ solid closed; Stage G retargets to mW with two north stars.**
  - **Closure.** The LJ solid met its branch-closing gate: the flow builds, samples and trains stably to physical energy (βU/N ≈ −1.1), and ESS rises from 0.2% to 2.1% at N=32 with the GNN. The two fixes of 2026-06-16 were the only nflojax source changes.
  - **Deferred.** The reproduction gate is deferred. bgmat's LJ-256 baseline (joint ESS 5.84%, βF/N −3.068) needs GPU + Pattern B, and that plan is parked at `../bgmat-clean/docs/gpu-pattern-b-plan.md`. The N=32 bgmat head-to-head was never run.
  - **By-products.** jax-pdf gained `MonatomicWater` (with a bgmat differential test) and `HarmonicCrystal` (analytic log Z oracle, tested on its own). bgmat-clean gained `free_energy.py` (log Z / FEP estimators), validated end-to-end against an exact harmonic-crystal log Z through the torus flow.
  - **Why stop here.** The LJ solid was chosen on 2026-06-15 as the quick branch-closer. The v1.0 bar (§8b) names bgmat's mW flow, so pushing on to the full LJ-256 match was scope drift.
  - **North stars (§7).** NS1 (G2): βF/N of mW cubic ice at N=216 against the reference −25.082; bgmat gets −25.083 with 27.6% joint ESS. NS2 (G3): the N=216 model at N=512 against −25.062; bgmat gets −25.061 with 4.2% ESS. The bgmat numbers come from its pretrained `params-mw_cubic_216.pkl`, evaluated locally in the `bgmat` env. The v1.0 definition (§8b) now lists DW4, the periodic LJ solid and mW instead of LJ13.
  - **Pattern B.** The decision now rests on the MS3 N=8 mW head-to-head (§8b). At N=8 a non-augmented split coupling transforms half the particles per layer, and its conditioner sees only the 4 frozen ones (GNN `num_neighbours` ≤ 3). bgmat's augmented coupling transforms every physical particle each layer, conditioned on the auxiliary half of a `(2N, d)` event. The head-to-head measures that gap.
  - **Known risk outside nflojax.** jax-pdf `MonatomicWater` builds the three-body term as a dense `(N, N, N)` tensor, about 5.2 GB per tensor at N=216 with batch 128. Training at N=216 needs a neighbour-list version on the jax-pdf side.
  - **Housekeeping.** The remote history of nflojax and jax-pdf was rewritten, changing author metadata only (trees and messages identical), so every commit hash changed. The six hashes cited in `PLAN.md` / `AGENTS.md` now point at the rewritten commits. Older hashes in notes and run manifests map via `../bgmat-clean/docs/archive/commit-map-noreply.tsv` (committed there since 2026-09-19).
  - **Merged.** `feature/particle-events` was fast-forwarded into `main`, together with jax-pdf `feature/periodic-lj-target` and bgmat-clean `feature/lj-solid-flow`. Gate: nflojax 651 passed / 10 skipped in float32 and 661 passed in float64; bgmat-clean 201 passed / 2 skipped in float64; jax-pdf 176 passed.
- *2026-09-19*: **Periodic conditioner inputs (v0.2.0).** Found by the bgmat-clean audit (its CORE-01).
  - **Gap.** On a torus the cube's faces are one seam, but the particle nets read raw coordinates, so a frozen particle just below `+B` and one just above `-B` looked `2B` apart. `LatticeBase` factories put a quarter of the site coordinates exactly on the box edge, so the conditioners were discontinuous where samples are densest. In bgmat-clean, shifting the sites off the seam alone raised mW N=8 ESS from 8.2 % to 18.8 % (3 seeds, 262k samples).
  - **Change.** `DeepSets`, `Transformer` and `GNN` gain `circular_n_freq` (and the first two a `geometry`): `None` means 8 harmonics of `circular_embed` on a fully periodic geometry, else raw coordinates; `0` forces raw. The GNN keeps raw coordinates for its minimum-image distances. `build_particle_flow` passes every conditioner factory the cube it sees as `geometry` (periodic where the physical box is); factories ending in `**_` still work. On by default for torus flows (owner decision). `embeddings.py` no longer imports `nets` (the cycle is gone).
  - **Behaviour change.** A GNN given a periodic geometry now encodes its node features by default, so its parameter layout (the `embed` kernel) and outputs change; pass `circular_n_freq=0` to reproduce old checkpoints. The fixture of `test_neighbour_list_stability_under_perturbation` had ties at the K-th neighbour for 4 of its 6 particles (it passed by luck with raw features); its axes are now scaled to remove them.
  - **Tests.** `tests/test_periodic_inputs.py`: invariance under a box translation of one particle, continuity across the seam with a raw-coordinate positive control, first-layer widths, the builder's geometry. Gate: 166 passed / 1 skipped fast, 513 passed / 10 skipped slow, 689 passed / 1 skipped under x64.
  - **Downstream result (bgmat-clean seam study, mW N=8, 3 seeds, 262k-sample ESS).** Lattice on the seam: raw 4.6 %, circular 10.0 %; off the seam: raw 17.1 %, circular 12.9 %. The site offset is the big effect; once sites are off the seam, 8 harmonics trained 4-5 points worse than raw on every seed at N=8. nflojax's torus default stays on (a function on the torus is the correct default); bgmat-clean passes `circular_n_freq=0` at N=8 and tunes the harmonics in its A7 sweep.
  - **Follow-up (before NS2).** The period of these features is the box, so frequencies must grow with it and a model trained at N=216 does not transfer to N=512 as is. bgmat encodes displacements from the lattice sites (fixed bounds) plus a per-site feature with the unit-cell period. Revisit the input encoding before MS4; at mW N=8 the box is one cell, so the two coincide.
- *2026-09-19*: **`OrthogonalTransform` and `use_orthogonal` (flat builders).** Found by the LTR project (a sampler for multimodal targets) on a rotated product of 1D mixtures, `d = 32`.
  - **Gap.** Axis-aligned couplings cannot fit the rotated target (top-level ESS 0.001 by maximum likelihood on exact samples), and `LinearTransform` cannot learn the rotation: without pivoting, a random 32 x 32 rotation needs LU entries up to ~300, so Adam moves `W` by O(1) per step (a map initialized at the true rotation drifted from ESS 0.79 frozen to 0.68 trained), and TF32 rounding in `L T x` cut the exact rotation's ESS from 1.00 to 0.62.
  - **Change.** `OrthogonalTransform` in `transforms/linear.py`: `W = expm(scale (U - U^T))`, exactly orthogonal, log det 0, identity at init, gate `W(g) = expm(g A)`, unconditional. The flat builders gain `use_orthogonal` (after the couplings, before LOFT) and `orthogonal_scale = 10`: scale 1 stalled on the near-Gaussian plateau at `W = I` (rotation half found); 10 found it within 2500 steps and matched the frozen true rotation. No existing layout or default changes.
  - **Tests.** 22 new tests in `test_transforms.py`, `test_identity_gate.py`, `test_builders.py` (slow tier): closed-form plane rotation, orthogonality and det, round trip, log det vs autodiff, gates (0, 1, scalar, per sample, composite), builder placement, and `q(x) = q_plain(x W)` without LOFT. 7 of 7 mutants killed. Gate: 166 passed / 1 skipped fast; the three edited files 324 passed / 3 skipped; the new tests pass under x64.
- *2026-09-20*: **Stage G / mW N=8 head-to-head: the non-augmented torus flow matches bgmat; Pattern B is not needed yet.** Evidence from bgmat-clean's gate G1 (its plan `docs/ms3-mw-plan.md`, steps A7 to A9; decision D15).
  - **Result.** At N=8, after 50000 reverse-KL steps at batch 128, a flow of nflojax primitives (`build_particle_flow`, circular splines, `LatticeBase.diamond`, `GNN` with `circular_n_freq=2`) reaches 80.4 ± 1.0 % physical-space ESS against bgmat's own `mw_cubic_8` at 83.1 ± 2.3 % (its augmented coupling, same budget, 3 seeds each, 262144-sample scores). With batch 256 ours reaches 83.8 ± 1.7 %. Absolute βF/N agrees with bgmat's joint estimate within 1.6e-4, well inside the paper's 1e-3 tolerance. At the shorter 5000-step budget ours is ahead by a factor 2 to 3 in per-particle log-weight variance: bgmat learns slowly at first.
  - **Consequence for the roadmap.** Pattern B (augmented coupling) stays unbuilt. The decision was agreed in advance on the per-particle log-weight variance v = −ln(ESS)/N, which compares across N: continue non-augmented while v_ours ≤ 2 v_bgmat. It is 1.17 (1.35 with bgmat's v corrected for its finite-sample marginal estimator). The question returns at N=64, where our conditioner sees half the particles against bgmat's full auxiliary copy.
  - **Conditioners.** `DeepSets` led the `GNN` at 5000 steps and trailed it at 50000 (73.7 % against 80.4 %). Since `DeepSets`' parameter count depends on N, the `GNN` is also the only one of the two that can transfer across N, so it is the recipe going forward.
  - **Circular inputs, revisited.** The v0.2.0 default of 8 harmonics is not the best choice here: at mW N=8, 2 harmonics gave 39.8 ± 5.2 % ESS after 2000 steps against 14.2 ± 5.1 % for 8 and 22.7 ± 4.0 % for raw coordinates. At N=8 the box is one unit cell, so those 2 harmonics coincide with bgmat's unit-cell encoding; from N=64 on they do not, which is what the follow-up above is about. bgmat-clean's N=64 rung tests 2 against 4 harmonics.
  - **Nothing needed from nflojax.** A7 to A9 required no library change: the gap to bgmat closed with the application's own tuning.
- *2026-09-21*: **`SplitCoupling(feature_map=...)`: a conditioner may read something other than the transformed variable (v0.3.0).** Found by bgmat-clean's R1b, its crystal-representation rewrite.
  - **Gap.** A coupling handed its conditioner exactly the frozen slice of the variable it was about to transform. A crystal flow wants the opposite pairing: transform each particle's displacement from its lattice site, because then no length in the model depends on the box (which is what transfer across N needs), but condition on positions, because a conditioner with no positional encoding cannot otherwise tell which site a particle occupies or which particles are neighbours. The only routes were a wrapper module, which breaks the `dense_out` identity-at-init patching, or threading a constant context through every call site.
  - **Downstream evidence.** bgmat-clean built the rewrite conditioning on displacements and measured the cost at mW N=8: reverse KL plateaued at −218.1 against the position-conditioned flow's −219.16, and the effective sample size fell from 84% to 1.4%, with the free energy outside its 1e-3 tolerance. Learning rate (7e-5 to 1e-3), spline bins (16, 32), depth (4, 8) and a strictly per-particle scaling were all ruled out as causes first.
  - **Change.** `feature_map: Callable[[Array], Array] | None = None` on `SplitCoupling`, applied to the structured frozen slice before the conditioner and before any flattening. `init_params` now builds its dummy through the same path, so the conditioner is sized from the mapped width and a map that concatenates features widens the first layer. The frozen slice still passes through untouched, so invertibility and the log-det are unaffected by construction.
  - **Tests.** `tests/test_transforms.py::TestSplitCouplingFeatureMap`: identity at init and round trip with a map present; sizing when the map widens the input; both halves of the split; and a behavioural check that the features reach the conditioner, by showing that a map which discards its input makes the transform independent of the frozen particles while the default does not.
  - **Note for whoever owns `OrthogonalTransform`:** `tests/test_transforms.py::TestOrthogonalTransform::test_invertibility` fails on `main` as of `8537cb8` (round-trip error 2.7e-3 against a 1e-5 tolerance, float32). Untouched here and deselected from this branch's runs.

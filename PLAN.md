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

- Branch: `feature/particle-events` — merged work `86be470` landed:
  - `boundary_slopes='circular'` on rational-quadratic splines.
  - `CircularShift` rigid-rotation bijector.
  - `_params_per_scalar` / `_validate_boundary_slopes` helpers de-duplicating 6 sites.
  - Gate + context tests on `SplineCoupling` in circular mode.
  - `@requires_x64` skip marker covering 5 pre-existing float32-only round-trip failures.
- **Stage A fully closed** (`2c94d18`). A1 (`Rescale`, `28b735f`), A3 (`Permutation` event_axis, Stage-0), A4 (context-type story, §5.2 contract narrowed), A2 (`CoMProjection`, Convention (1) zero log-det + `ambient_correction` helper).
- **Stage B fully closed** (`bc90993`). B1 (`UniformBox`), B4 (`utils/pbc.py`), B2 (`utils/lattice.py` — 5 generators), B3 (`LatticeBase` + 5 factories). New `nflojax/utils/` subdir.
- **Stage C fully closed.** C1 (`circular_embed`), C2 (`positional_embed`). New `nflojax/embeddings.py` (stateless feature transforms for conditioner inputs).
- Test infrastructure: `pytest-xdist` adopted as default parallel runner (`addopts = "-n auto -q"` in `pyproject.toml`); AGENTS.md rewritten as a one-command "Testing Strategy". Landed `9ef1141`.
- **Stage D fully closed.** `SplitCoupling.flatten_input` hatch unlocks structured-input conditioners. D1 (`DeepSets`, permutation-invariant), D2 (`Transformer`, pre-norm, permutation-equivariant per-token), D3 (`GNN` with top-K neighbours under PBC, `num_neighbours=12`), D4 (shared contract fixture `tests/test_conditioner_protocol.py`). §10.4 and §10.5 resolved (see decision log). Top-level `Dense(..., name="dense_out")` convention across all four conditioners; `SplitCoupling._patch_dense_out` infers the bias size from the conditioner so flat-output and per-token-output modules plug in the same way.
- Full test suite (parallel): **592 passed / 6 skipped under float32 (~95s); 598 passed under `JAX_ENABLE_X64=1` (~95s).** The 6 float32 skips carry `@requires_x64` (RQS-inverse + LinearTransform triangular-solve roundoff).
- DESIGN.md checked in; `AGENTS.md` updated to point at it.

**Next up — Stage E** (particle-flow builder `build_particle_flow`). All Stage-D primitives are in place; E1/E2 can compose them directly.

Known pending: no branch strategy chosen yet for Stages A–F (see §10.1).

### Stage-0 pre-work (landed in the same session, before Stage A)

After the first + second-pass audits in `audit.md`, three "tensions" were lifted directly into the work:

- [x] **Amend DESIGN.md §2.1 "500 LOC" rule** to distinguish single-concept files from catalogue files (`transforms.py`, `nets.py`). Each entry still ≤ 500 LOC; catalogue total can grow.
- [x] **Introduce `nflojax/geometry.py`** with the `Geometry(lower, upper, periodic=None)` value object — numpy-backed configuration, not a PyTree. Factory `Geometry.cubic(d, side, lower)`. Derived `box`, `d`, `volume`, `is_periodic()`. Landed *before* Stage A so every upcoming geometry-consuming primitive (`Rescale`, `UniformBox`, `LatticeBase`, `utils/pbc`) targets a single type from day one.
- [x] **Retrofit `CircularShift`** to carry a single `geometry: Geometry` field. `create(key, geometry)` factory; `from_scalar_box(coord_dim, lower, upper)` classmethod for legacy-ergonomic construction.
- [x] **Generalise `Permutation`** with `event_axis: int = -1`. Default preserves historic last-axis behaviour; `event_axis=-2` unlocks particle-axis shuffles on `(B, N, d)`. This satisfies Stage A3 early (see §1 below).

Verification: 401 passed / 5 skipped (float32); 406 passed (x64). Four new `event_axis` tests added.

Deferred from the proposed Stage 0 (still open):
- Split `transforms.py` into a `transforms/` subdir, *or* accept the amended rule and leave it as a catalogue file.
- Context-type story (Array vs PyTree): §5.2 of DESIGN.md currently says PyTree; code still assumes Array (`_compute_gate_value.ndim`; `MLP` concat). Decide which wins and reconcile.
- `LinearTransform` (515 LOC) — audit whether it earns its place in `transforms.py` for particle workloads or should extract.

---

## 1. Stage A — bijection extensions

Small, high-leverage bijections that every particle-system flow needs. All live in `nflojax/transforms.py`.

### Tasks

- [x] **A1. `Rescale(geometry, target=(-1, 1))`** per-axis affine that maps `geometry.box` to the canonical spline range.
  - Dataclass takes a `Geometry` (per Stage-0 retrofit pattern) plus a scalar or per-axis `target` pair.
  - Closed-form log-det (sum of `log(scale_i)` over event axes).
  - Supports arbitrary trailing-axis rescaling; default last axis.
  - Tests: round-trip, log-det-vs-autodiff, jit, `target` default vs explicit.
  - Landed `28b735f`: dataclass with `Geometry` + per-axis `target` + `event_shape`; no params; closed-form log-det; 14 tests green under both dtypes.
- [x] **A2. `CoMProjection(event_axis=-2)`** (replaces the earlier `ShiftCenterOfMass` proposal per audit §9.3).
  - `(N, d)` ↔ `(N-1, d)` bijection: subtract the mean along `event_axis`, drop the last slot, reconstruct from the invariance.
  - Log-det: constant correction `-½ · d · log(N)` (or whichever sign the convention demands; derive explicitly and document on the class).
  - Blocked on picking one CoM strategy; DM / bgmat use different mechanisms. Audit §9.3 recommends B (CoM projection) as the ship-today primitive.
  - Tests: round-trip via the full `(N, d)` ambient space, autodiff-Jacobian determinant sanity on the subspace, jit.
  - **Resolved via Convention (1): log-det is zero on the `(N-1)d` subspace; the `(d/2)·log(N)` volume correction is a caller-applied constant exposed as `CoMProjection.ambient_correction(N, d)`.** Heavy documentation at six contact points: class docstring WARNING block, `REFERENCE.md` subsection with decision box, `USAGE.md` pointer + recipe, new `EXTENDING.md` §"CoM handling" with augmented-coupling alternative and a **do-not-stack** warning, `INTERNALS.md` full derivation (Gram matrix `I + 11^T`, `det = N`), `AGENTS.md` Gotcha one-liner. 12 tests green under both dtypes.
- [x] **A3. `Permutation` generalised to non-last axes.** Landed in Stage-0. `event_axis: int = -1` default preserves last-axis behaviour; `event_axis=-2` shuffles particles on `(B, N, d)`. 4 new tests.
- [x] **A4. Context-type story.** First-pass audit flagged DESIGN.md §5.2 ("context is PyTree") is not matched by `_compute_gate_value` (indexes `.ndim`) or `MLP` (concatenates). Decide: (a) narrow the doc claim to "Array for built-in conditioners, PyTree for custom"; (b) accept PyTree in the built-in path (flatten via `ravel_pytree` in MLP). Update docstrings + `validate_conditioner.validate=False` opt-out accordingly.
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

A single entry-point that assembles the canonical DM / bgmat topology. File: `nflojax/builders.py`.

### Tasks

- [ ] **E1. `build_particle_flow(...)`** topology:
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
- [ ] **E2. Integration smoke test** `tests/test_particle_smoke.py`.
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

- [ ] **F1.** `USAGE.md` — add "Particle flows" section tying the new primitives together.
- [ ] **F2.** `REFERENCE.md` — add entries for every new public name.
- [ ] **F3.** `EXTENDING.md` — document the augmented-coupling composition pattern (bases of size `(2N, d)` + `SplitCoupling` across the physical/auxiliary boundary); document the "bring your own conditioner" recipe with a minimal example.
- [ ] **F4.** `AGENTS.md` — update module dependency graph with `embeddings.py`, `utils/pbc.py`, `utils/lattice.py`.
- [ ] **F5.** `INTERNALS.md` — add a short section on the conditioner-protocol abstraction and why reference conditioners are examples, not authoritative.

### Acceptance

- Every new public symbol has an entry in `REFERENCE.md`.
- `USAGE.md` examples actually run (doctest-grade preferred; at minimum spot-executed from the repo).

---

## 7. Stage G — bgmat prototype (validation)

A small off-nflojax deliverable in `bgmat/`. Not part of nflojax; a proof the abstraction holds.

### Tasks

- [ ] **G1.** `bgmat/flow_on_nflojax.py` (location tentative) that reassembles bgmat's mW flow using:
  - nflojax's `LatticeBase.hex_ice(...)` (or bgmat's lattice if factoring differs),
  - nflojax's `build_particle_flow` with `conditioner=bgmat.models.gnn_conditioner.GNNConditioner`,
  - bgmat's own augmented-coupling wrapper, energy, and training loop — untouched.
- [ ] **G2.** Parity test: random input seed compared against bgmat's current flow output within `1e-5` under x64.

### Acceptance

- Parity test passes; if it fails, the gap is a pinpointable missing primitive that goes into the parking lot.
- Nothing new required in nflojax (else we go back and add it, then rerun).

### Commit plan

Lives in `bgmat/`, not nflojax. Track it here so it isn't forgotten.

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

Stage D checklist (2026-04-22):

- [x] `pytest tests/ -q` green under default float32 (592 passed / 6 skipped).
- [x] `JAX_ENABLE_X64=1 pytest tests/ -q` green (598 passed).
- [x] Every new conditioner satisfies identity-at-init: `DeepSets`, `Transformer`, `GNN` each pass their SplitCoupling round-trip at init. Covered in `tests/test_conditioner_protocol.py::test_split_coupling_identity_at_init` (parametrized).
- [x] No new energy / training / observable term (DESIGN.md §11 greps). Docstring mentions of "training" / "energy" (in `CoMProjection`) are pre-existing.
- [x] No new heavy dependency. All three conditioners use `flax.linen` primitives already in scope.
- [!] Total `nflojax/` LOC excluding tests: **6653**. Exceeds the §11 item 10 ballpark of 5000 — needs review. Breakdown: `transforms.py` 2819, `nets.py` 924 (+~640 from Stage D), `builders.py` ~(not counted here), `distributions.py`, `flows.py`, `splines.py`, `embeddings.py`, `geometry.py`, `utils/pbc.py`, `utils/lattice.py`. Per DESIGN.md §2.1 amendment, catalogue files (`transforms.py`, `nets.py`) are allowed to grow; each *concept* within must still fit ≤ 500 LOC. Revisit before Stage E if this drifts further.
- [x] Every public name in `REFERENCE.md` (`DeepSets`, `Transformer`, `GNN`, `init_conditioner`, new `SplitCoupling.flatten_input` field).

---

## 8b. Long-term trajectory

Stages A–G get nflojax to the point where a user can reassemble the DeepMind *Flows for Atomic Solids* paper and bgmat's mW flow with no code in the library that is specific to either. That is the bar for **v1.0**.

### v1.0 — "particle-flow framework"

Definition:

- Stages A–G all pass their acceptance criteria.
- A user can build the DM paper's topology with nflojax primitives + a user-side Transformer conditioner + user-side energy / training / observables, in ≲ 500 lines of app-side code.
- A user can build bgmat's flow using nflojax primitives + bgmat's GNN conditioner + bgmat's augmented-coupling composition + bgmat's energy / training / marginal-inference, without modifying nflojax.
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
- **E(3) / SE(3) bijections** — if a molecular downstream application lands, introduce `transforms/equivariant.py` with EGNN-style couplings. Audit §7.3 documents why this is non-trivial.
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
- **SE(3) / E(3) equivariant conditioner.** Out by design; user brings EGNN / NequIP / MACE when needed. DESIGN.md §4 item 7.
- **Block permutation (multi-species).** Waits for a concrete multi-species application. DESIGN.md §7.5.
- **Heteronuclear lattices.** Same. DESIGN.md §7.5.
- **Augmented-coupling primitive.** Stays a pattern in `EXTENDING.md`, not a class. DESIGN.md §4 item 9.
- **Long-range energy support (Ewald / PPPM).** Energy-side; not flow-side. DESIGN.md §7.6.

---

## 10. Open questions

Items that need a decision before the relevant stage can close.

- [ ] Branch strategy for Stages A–F. One branch `feature/particle-flow-framework`, or stage-per-branch? (Default: one long-lived branch, stage-per-commit.)
- [x] `ShiftCenterOfMass` log-det convention: document as "zero on the (N−1)d subspace; caller is responsible for the embedding-space correction" vs. "constant $-d \log(N)/2$ correction baked in"? Pick when writing A2. **Resolved 2026-04-21: Convention (1) — zero log-det on the subspace; caller applies `CoMProjection.ambient_correction(N, d) = (d/2)·log(N)` when an ambient density is needed. Rationale + full derivation in §11 decision log and `INTERNALS.md` "CoM Projection and the Volume Correction".**
- [x] `LatticeBase.hex_ice` unit-cell parameters: follow DM's convention (8 atoms per cell) or bgmat's (re-derive)? Likely DM. **Resolved 2026-04-22: DM convention. 8 atoms per orthorhombic cell with `cell_aspect = (1, sqrt(3), sqrt(8/3))` and the puckering parameter `6 * 0.0625` baked in (matches `flows_for_atomic_solids/models/particle_models.py:HexagonalIceLattice`). Atom positions reproduced inline in `nflojax/utils/lattice.py` so the test suite is hermetic.**
- [x] `Transformer` attention norm placement: pre-norm (more stable for deeper stacks) vs. post-norm (closer to DM's original). **Resolved 2026-04-22: pre-norm.** `h = h + attn(LN(h)); h = h + ffn(LN(h))` per block + a final `LN` before `dense_out`. Rationale + background in `INTERNALS.md` "Transformer conditioner: pre-norm choice".
- [x] GNN default `num_neighbours`: 12 (common) vs. 18 (bgmat). **Resolved 2026-04-22: `num_neighbours=12`.** Apps override (bgmat's 18 is app-side, not library default).

---

## 11. Decision log

- *2026-04-21* — Plan drafted alongside DESIGN.md. Adopted: thick-on-flows / thin-on-physics scope; reference conditioner family is MLP + DeepSets + Transformer + MPNN; no energy or training helpers in nflojax.
- *2026-04-21* — Stage A1 closed (`28b735f`). `Rescale` ships as a fixed, non-learnable geometry→canonical affine; `LinearTransform` retains the learnable-affine role. API: `Rescale(geometry, target=(-1, 1), event_shape=None)`; scalar or per-axis target; `event_shape` default `(geometry.d,)` with `event_factor = prod(event_shape[:-1])` so log-det accumulates correctly on rank-N particle events.
- *2026-04-21* — Adopted `pytest-xdist` as default parallel runner (`addopts = "-n auto -q"` in `pyproject.toml`; `pytest-xdist` added to test extras). Rewrote AGENTS.md Dev Commands as a one-command "Testing Strategy" (`pytest tests/` after edits, x64 at stage close). Full suite wall-clock: float32 6:28 → 1:25 (4.5×), x64 12:11 → 1:34 (7.8×). Rationale: cheap full suite removes the agent triage problem; a file→tests mapping would push judgement onto the agent, and agents get that wrong.
- *2026-04-21* — Stage A4 closed via option (a): narrow the built-in contract, keep PyTree at the flow layer. DESIGN.md §5.2 now a two-tier contract (PyTree through flows, Array for built-in `MLP` and `_compute_gate_value`, any PyTree for custom conditioners). Option (b) rejected because target apps (DM, bgmat) bring their own conditioner anyway, so `ravel_pytree` + PyTree-aware batching in the common path would add complexity for no one. A new `TestCustomConditionerPyTreeContext` test in `tests/test_conditional_flow.py` exercises a dict-context conditioner end-to-end (round-trip + jit).
- *2026-04-21* — Stage A2 closed; Stage A fully done. `CoMProjection` ships with **Convention (1)** log-det: the bijection is a relabelling between two `(N-1)d`-dim spaces (reduced Euclidean and zero-CoM subspace of `R^(Nd)`), so `log_det = 0` both directions. The volume-element constant relating the two embeddings is `(d/2)·log(N)` (derived from `det(I + 11^T) = N` for the parameterisation `x_N = -Σy_i`), exposed as `CoMProjection.ambient_correction(N, d)`. Convention (2) — baking the constant into the log-det — was rejected: it silently double-counts in the augmented-coupling composition (bgmat's pattern), where translation invariance is handled separately and densities are already ambient-valid. Explicit caller-applied correction keeps the two patterns cleanly separable. Heavy documentation placed at six contact points to prevent silent misuse: class docstring, REFERENCE.md decision box, USAGE.md recipe, EXTENDING.md "CoM handling" (with do-not-stack warning), INTERNALS.md derivation, AGENTS.md Gotcha.
- *2026-04-22* — **Stage B closed.** Four tasks landing in one session: `UniformBox` (B1, per-axis uniform base on a `Geometry`), `utils/pbc.py` (B4, `nearest_image` + `pairwise_distance(_sq)` consuming `Geometry`), `utils/lattice.py` (B2, pure functions for `fcc / diamond / bcc / hcp / hex_ice` returning `(N, 3)` numpy positions), `LatticeBase` + 5 factories (B3). All consume the Stage-0 `Geometry` value object — no raw `lower/upper` alternatives. `LatticeBase.permute=True` shuffles particle order per-batch via `jax.vmap(jax.random.permutation)` and subtracts `log(N!)` from `log_prob`; the constant has the same caveats as `CoMProjection.ambient_correction` (no gradient effect, matters for absolute densities / ESS / `logZ`). §10.3 resolved with the DM `hex_ice` convention. New `nflojax/utils/` subdir; AGENTS.md dependency graph extended. 84 new tests, 5 minutes total session wall-clock for the Stage. Spherical truncation in `LatticeBase` deferred — no concrete trigger in v1.0 scope.
- *2026-04-22* — **Stage C closed.** Two stateless feature transforms in new `nflojax/embeddings.py`: `circular_embed(x, geometry, n_freq)` (per-coord Fourier features on a periodic box, lowest harmonic tiles `geometry.box`) and `positional_embed(t, n_freq, base=10_000)` (sinusoidal scalar embedding, transformer-style). Both raise `ValueError` on `n_freq=0` to avoid silent zero-width outputs that would break downstream `jnp.concatenate`. **API change vs. PLAN.md spec**: `circular_embed` takes `Geometry` (not raw `(lower, upper)`) to match the Stage-0 retrofit pattern — one path, no overload. Non-periodic axes are not gated; documented as caller's responsibility. 15 new tests; ~80 LOC. Unblocks Stage D conditioners (Transformer, GNN) which both consume these features.
- *2026-04-22* — **Stage D closed.** One preparatory change + three reference conditioners + one shared fixture. The preparatory change: `SplitCoupling.flatten_input: bool = True` hatch (default preserves the flat-`(B, N*d)` contract MLP expects; `False` passes the structured `(B, N_frozen, d)` slice through to permutation-aware conditioners). Four new public names in `nflojax.nets`: `DeepSets` (permutation-invariant aggregator), `Transformer` (pre-norm multi-head self-attention, permutation-equivariant per-token), `GNN` (top-K PBC-aware message passing, `num_neighbours=12` default). All three satisfy the existing conditioner contract (`context_dim` attribute + `apply` + `get_output_layer`/`set_output_layer`) and use a top-level `Dense(..., name="dense_out")` to stay compatible with `SplitCoupling._patch_dense_out`. §10.4 resolved (pre-norm), §10.5 resolved (12 neighbours). New `tests/test_conditioner_protocol.py` locks the contract at 5 checks × 4 conditioners; per-conditioner detail in `tests/test_nets.py`. Two subtle-bug fixes during implementation: (1) self-mask computed via `jnp.where(eye_bool, inf, d_sq)` to avoid `0 * inf = NaN` off-diagonal; (2) test N bumped to 8 particles so `num_neighbours=3` stays < N_frozen=4 at init.
- *2026-04-22* — **Post-close refactor (P1).** Dropped the `set_output_layer` slicing magic from `Transformer` and `GNN`. Instead, `SplitCoupling._patch_dense_out` (and `SplineCoupling._patch_dense_out` for symmetry) now reads the conditioner's current `dense_out` bias length and sizes `identity_spline_bias(num_scalars = bias_size // params_per_scalar, …)` to match. Works for flat and per-token dense_out uniformly because `identity_spline_bias` is a per-scalar pattern tiled across scalars. Net: `Transformer.set_output_layer` / `GNN.set_output_layer` collapsed to the trivial dict-update form (same as `DeepSets`); one Gotcha removed from AGENTS.md; one "library-private convention" line struck; a bias-shape divisibility check added in both `_patch_dense_out` sites. Reason: the slicing hack was a hidden coupling between conditioners and `SplitCoupling`'s internals; inferring from the conditioner keeps the contract local and easier to extend.
- *2026-04-22* — **Post-close hardening.** Four audit items landed as separate changes: (1) `Transformer` uses `nn.MultiHeadDotProductAttention` instead of the now-deprecated `nn.SelfAttention`; (5) `SplitCoupling.init_params` runs a one-sample dummy apply after `_patch_dense_out` and raises a clear, diagnostic `ValueError` when the conditioner's output total-trailing size doesn't match `transformed_flat · params_per_scalar` — catches per-token `Transformer`/`GNN` misconfigured for asymmetric splits at init rather than as a cryptic reshape error at forward time; (4) removed per-conditioner factories `init_deepsets`/`init_transformer`/`init_gnn` (~75 LOC) in favour of a single generic `init_conditioner(key, conditioner, dummy_x, dummy_context=None)` helper (5 LOC) — shrinks the public API, removes three redundant public names, and keeps all init paths uniform; (3) new `tests/test_particle_integration.py` parametrised over `DeepSets`/`Transformer`/`GNN` composing four alternating-swap `SplitCoupling` layers through `CompositeTransform` — asserts identity-at-init, jit round-trip, and non-zero gradient. Reason: each item brings surface down or failure-mode clarity up; together they move Stage D from "works" to "robust + discoverable".

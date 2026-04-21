# nflojax — Design & Plan Audit

**Audience.** The person (human or agent) deciding whether `DESIGN.md` and `PLAN.md` are ready to act on.

**Frame.** Reviewer comes from the intersection of normalizing-flow library work (nflows / Distrax / TFP-bijectors / Zuko / FlowJax) and physics-grade Boltzmann sampling of solids, liquids, supercooled / glassy systems. The critique is opinionated and concrete. Items are flagged as:

- **[BLOCK]** — fix before starting code. Ignoring it will cause rework.
- **[STRONG]** — fix soon. Ignoring it will cause pain in 1–2 stages.
- **[NIT]** — wording / polish. Worth fixing; not load-bearing.

A one-line reason accompanies every item.

---

## 0. Executive take

Both documents are unusually clean. The scope framing ("nflojax = normalizing-flow framework for particle systems; everything else stays in apps") is exactly the line a library like this should draw, and the §4 exclusion list is the sort of thing most libraries only write after five years of scope drift.

That said:

1. The **solid-vs-liquid framing is a simplification that glosses over where most of the real pain lives** — supercooled, glassy, and disordered phases whose equilibrium densities are neither single-mode-per-lattice-site nor uniform. nflojax should be honest about what its two bases (`LatticeBase`, `UniformBox`) do and do not cover.
2. The **translation-invariance story is under-specified**. DM and bgmat use two *different* mechanisms for the same goal; proposing a single `ShiftCenterOfMass` primitive without deciding which one is premature.
3. **`DeepSets` is listed as a baseline equivariant conditioner — empirically it will fail at anything interesting**. Calling it pedagogical would be more honest and would save users a training run.
4. **Augmented coupling deserves a thin primitive, not just a pattern**. The recipe is short, but the log-prob accounting is exactly the kind of detail users mis-implement. One shared implementation prevents a class of bugs.
5. **The plan under-weights scaling and over-weights breadth**. Stage D ships three conditioners; at least one of them (GNN under jit with PBC top-k neighbours) is a multi-week engineering problem in isolation, not a sub-stage.
6. The plan has **no API-review gate and no performance benchmark stage**. For a library that will gate GPU wall-clock at N=500+, both are essential.
7. `validate_conditioner` as described is **architecturally brittle**; rethink it now, not after landing three non-MLP conditioners against it.
8. The audit's first pass **under-weighted symmetries and invariances** (permutation, translation / CoM, rotation / O(d), gauge, point group). These are the main thing distinguishing a particle flow from a generic flow and deserve a systematic treatment in DESIGN.md. See §9 below for the full picture and §10 for other crucial topics the first pass brushed past (connectivity discontinuities, log-det sign conventions, NPT, internal-coordinate molecules, golden-sample regression tests, etc.).

None of this invalidates the direction; the direction is good. It tightens the scope and re-orders the work.

---

## 1. Strengths (so you know what not to "fix")

Keep these; they are load-bearing.

- **§1 framing** — "nflojax does not know particles interact" is the right mental model and it should be quoted anywhere a contributor proposes to cross that line.
- **§2.9–10** — the "no domain knowledge" + "generalises beyond its motivating case" rules are worth more than any test.
- **§4 exclusion list with reasons** — most libraries have an implicit exclusion list; writing it down will prevent the most common review arguments.
- **§5.6 identity-at-init as a correctness invariant** — this is the single best test in a flow library; elevating it to a design rule is exactly right.
- **§9 decision heuristics in stop-at-first-no form** — operationalisable; agents can actually apply it.
- **§11 verification checklist with grep targets** — the grep list (`epsilon`, `optax`, `rdf`, …) is the right kind of lightweight enforcement. Keep and extend.
- **PLAN.md's `[ ]/[~]/[x]/[-]` + decision log** — living-plan hygiene most repos don't bother with.

---

## 2. DESIGN.md — section-by-section critique

### §1 Motivation

**[NIT]** The three-app list (DM / bgmat / future) is accurate but understates the research frontier. Add (at minimum, for framing):

- *MuMoNICE* (Moqvist, Brown, Noé, 2024) and *Timewarp* / *ItÔ generator* lineages — which need time-dependent or SDE-ish flows and may or may not fit nflojax's API, but exist and will ask.
- *Equivariant Augmented Flows* (Boyda et al., Köhler et al.) — closer to bgmat; useful reference for how the augmented-coupling literature is structured.

Not every reference belongs in a motivation section, but a one-liner "related work we are not absorbing but that users may come from" anchors expectations.

**[STRONG]** The sentence *"We now want the same library to be the flow-side foundation for … liquids, disordered materials"* is a larger claim than is currently supported. See §3 below — neither `LatticeBase` nor `UniformBox` is a good base for a liquid at freezing or a supercooled glass. Either soften the claim or add a pointer to how users would bring their own empirical base (e.g., a `PointCloudBase` that wraps a stored MD trajectory).

### §2 Philosophy

**[STRONG]** Rule 8 ("float32 is the default; float64 is the proof") is under-specified. The 5 pre-existing failures on `feature/particle-events` are a hint that this regime is fragile. Operationally, the rule should add:

> A new primitive may not gate a user feature on `@requires_x64`. If float64 is needed for functional correctness (not just for tolerance of a test), the primitive is broken or the test is wrong. `@requires_x64` is for tolerance of non-trivial tests only.

Without that qualifier, a contributor could ship a primitive that *only functions* under x64 and claim the rule permits it.

**[NIT]** Rule 10 is actually the most important rule but is last. Move it to position 3 or 4; it's the one a contributor most often forgets.

### §3 What nflojax IS

**[BLOCK]** §3.1 claims bijections cover "Cartesian, torus, and **simplex**". There is no simplex transform in the repo today. Either delete "simplex" or add it to the plan explicitly (Stick-Breaking is a 50-line transform; there is a use case — fractional coordinates of composition). Today this line is a lie.

**[STRONG]** §3.2 lists `Transformer` and `GNN` as reference conditioners. Realistically:

- A Transformer that matches DM performance is ~200 LOC, has non-trivial numerics (attention variance scaling, pre- vs post-norm, rotary embeddings or not), and must scale to N~500 without OOM. This is not a drop-in.
- A reference MPNN under PBC with fixed-cutoff or top-k neighbours, jit-compatible, is a 1–2 week engineering effort before it is trustworthy. Fixed cutoff + padding to `num_neighbours` is the jit-friendly path; top-k is not jit-friendly without `lax.top_k` tricks.

Pick *one* of these for the first release and be explicit in DESIGN.md that "GNN/Transformer are on the roadmap; DeepSets ships first". Listing both as reference conditioners today sets an expectation the plan does not yet meet.

**[STRONG]** §3.3 says "Centre-of-mass bijector for translation-invariant flows" — singular. The translation-invariance problem has *two* established solutions:

1. **Drop-a-particle + sample-random-shift** (DM): the flow operates on `(N-1, d)` coordinates; the Nth is reconstructed; log-prob inflates by `-log V`.
2. **Augmented CoM swap** (bgmat): flow operates on `(2N, d)`; CoM of one half is subtracted from both; log-det constant `±1`.

These are not substitutable. Shipping a single "`ShiftCenterOfMass(event_axis=-2)`" without specifying which job it does will trap whoever uses it. Options:
- (a) Ship neither; document both recipes in `EXTENDING.md`.
- (b) Ship both, clearly named (`ProjectOutCoM` and `SwapCoMAcrossHalves`).
- Do *not* ship one and pretend it covers both.

**[NIT]** §3.4 "build a particle flow in ~10 lines" is aspirational; the reality is closer to 30 if the user has to configure a conditioner. Either deliver on the claim (a `build_particle_flow(...)` with sane defaults) or reword.

### §4 What nflojax IS NOT

**[STRONG]** Item 7 ("No SE(3) / E(3) equivariant architectures") is correct as a starting point but will age poorly. Make it *conditional*: "not today; revisit if one downstream use case requires it." Today it reads like permanent doctrine, which is too much commitment for a decision that depends on what bgmat-v2 decides to ship.

**[STRONG]** Item 9 (augmented coupling as "pattern, not primitive") is the call I'd most strongly urge you to reconsider. The pattern has *three* subtle pieces that users get wrong:

1. Base factorisation: `log p_phys(x_phys) + log p_aux(x_aux)` must be numerically consistent with the aux base's `sample_and_log_prob`.
2. Log-det attribution: when the inner flow mixes physical and aux, the total log-det goes on the physical log-prob; the aux log-prob is the *aux base* evaluated at the *aux pre-flow sample*, not the post-flow sample.
3. Marginalisation: estimating `log p(x_phys)` by marginalising out `x_aux` requires importance sampling with the right proposal.

Items 1 and 2 are where the bugs are, and they are *not* captured by a documentation snippet people will actually read. A 40-line `AugmentedBase(base_phys, base_aux)` dataclass + an explicit `AugmentedFlow(inner_flow)` wrapper resolves them by construction and is well within the "lean and hackable" budget. Patterns in docs are where bugs go to breed; primitives in code are where they get debugged once.

**[NIT]** Item 10 ("no heavy dependency surface") — clarify that Flax's own `linen` → `nnx` transition will eventually force our hand. Add a line acknowledging that we will migrate when `linen` is deprecated and that the API surface of conditioners is expected to change at that point.

### §5 Core abstractions

**[BLOCK]** §5.1 says log-det may be "scalar, batch-shaped, or broadcast-compatible with the batch shape". This is *three* options and I have never seen a flow library survive all three cleanly. Recommend collapsing to two:

- **Scalar** — e.g. `CircularShift`, `Permutation`.
- **Exactly `batch_shape`** — everything else.

Broadcasting "compatible with" is where bugs live (one primitive returns `()`, another returns `(B,)`, `CompositeTransform`'s accumulator is happy to add them, but the downstream user gets a surprising `(B,)` when they expected `()`). Make the rule: scalar or `batch_shape`, nothing in between, and test every new primitive for it.

**[STRONG]** §5.2 ("Context is a PyTree") is correct *if* every existing primitive actually respects it. `MLP`'s current behaviour is to *concatenate* context to the input, which requires context to be a flat array. The plan's task A4 ("context-type docstring sweep") treats this as a docstring change; it is not. It is either:

- (a) An MLP behaviour change — MLP needs to either reject non-array context or flatten via `ravel_pytree`. Either way, non-trivial.
- (b) A clearer rule: "context is either `None`, an Array, or a PyTree; the default reference conditioners accept only Array; advanced conditioners accept any PyTree". Honest, but weakens the headline.

Decide which, then update both docs. The current `A4 = docstring sweep` underestimates this.

**[STRONG]** §5.3 (event shape and event axes) — currently most primitives are last-axis-only and will stay that way. The sentence *"A primitive that only works on the last axis must say so"* reads as if last-axis-only is the exception. In practice, `SplitCoupling` is the only primitive that is genuinely rank-N-aware today. Flip the framing: most primitives are last-axis-only; the ones that are rank-N-aware are flagged.

**[BLOCK]** §5.4 — the conditioner contract is what unifies the whole library; this section is three sentences. It needs to say:

- What the input tensor's shape is (for a flat coupling vs `SplitCoupling` they differ).
- What the output tensor's shape is, and how it relates to `_params_per_scalar`.
- Who owns featurisation (raw coords → something-the-conditioner-can-use). Currently implicit; should be explicit: **the conditioner owns featurisation**. `circular_embed`, `positional_embed`, distance features, and lattice-relative features are all conditioner responsibilities.
- How context is threaded and what the conditioner is allowed to assume about its structure.
- What "identity at init" means when the output is used as spline parameters vs affine parameters (the mechanism differs — spline needs `identity_spline_bias`; affine needs zero scale + zero shift).

A one-page §5.4 rewrite is the single highest-value edit to DESIGN.md.

**[STRONG]** §5.6 ("Identity at init") — does not hold for augmented flows out of the box. The physical sub-flow must be identity at init; the auxiliary sub-flow may or may not be, depending on whether you want the aux base to be Gaussian or something else at init. State the rule more carefully:

> Identity at init applies to the mapping from samples drawn from the base to samples produced by the flow. For augmented flows it applies to the physical-slot projection of the mapping.

### §6 How to think about particle systems

**[STRONG]** The solid/liquid split (§6 paragraph 3) is too binary. Better taxonomy, which still fits the "two bases" story:

- **Localised bases**: `LatticeBase` — modality concentrated near distinct sites. Fine for crystals, defect-free + small thermal fluctuation regime. Breaks when particles interchange (diffusion, defect migration) unless the base includes random permutation sampling (DM does, `LatticeBase` should too — PLAN §2.B3 correctly captures this).
- **Delocalised bases**: `UniformBox` — modality uniform. Fine for dilute gases. For dense liquids, a liquid at the melting point does *not* have uniform density (local correlations dominate) and a uniform base will give a terrible flow to train.
- **Data-driven bases**: **missing today**. Users sampling dense liquids / glasses / metastable configurations will want a base that is an MD-equilibrated snapshot density (Gaussian around equilibrium positions, or a mixture). This is ~40 LOC (`PointCloudBase(positions, noise_scale, permute)`), it fits the charter, and it makes the "liquids supported" claim real. **Strongly recommend adding.**

Also: the sentence *"A solid is a particle system whose base distribution is a lattice"* is not correct in the physics sense (crystallography doesn't care what your flow's base is). Reword to: *"We identify 'solid-class flows' with flows built on a localised lattice-anchored base, and 'liquid-class flows' with flows built on a delocalised base. The physics the user is modelling is their concern; the base is nflojax's."*

### §7 Forward-thinking

**[STRONG]** §7.2 (equivariance ladder) lists `DeepSets` as a valid equivariant conditioner choice *"when the per-particle update depends only on aggregate statistics"*. Fine in theory; in practice for materials, the per-particle update depends on *neighbours*, not on aggregate. DeepSets loses neighbour information by construction. Two better framings:

- Call `DeepSets` **"pedagogical minimum and unit-test baseline"** — usable, not competitive.
- Replace with or add **`InteractionNet` / `AttentiveDeepSets`** — sum-pool with attention weights; `O(N·k)` with a fixed neighbour count; competitive with small transformers.

If you want one equivariant ship with the first release, `AttentiveDeepSets` beats both `DeepSets` (not expressive enough) and `Transformer` (too much code, O(N²) memory).

**[STRONG]** §7.3 (groups beyond Sn) — the claim *"for E(3), this means coupling layers that transform invariant features (distances, angles) rather than raw coordinates. This is expressible with the current machinery but is a user-level construction"* is misleading. Coupling layers that transform distances are well-defined; the round-trip in the ambient `(N, d)` coordinate space is *not* — you cannot recover positions from distances alone. If you want E(3)-equivariant flows in the sense of Köhler–Klein–Noé (CNF on SO(3)-invariant features), you need either CNFs (out of nflojax scope today) or very specific coupling constructions (Garcia Satorras et al.; Köhler et al. 2020). This is *not* trivially expressible today. Honest rewrite: "E(3)-equivariant coupling flows require a dedicated bijection class that operates on invariant features and lifts back to coordinates; we do not ship this, and expressing it via current primitives is not straightforward."

**[NIT]** §7.4 (boundaries) — confined particles (hard walls) can be expressed via `LoftTransform` + clamping; worth a sentence.

**[STRONG]** §7.7 ("Context as the interface") — this is good but undersells a hazard. If you route *everything* through context, the flow layers will each call the conditioner with the *same* context. For expensive conditioners (GNN over 500 particles), this wastes compute. Add a line:

> Conditioners may cache derived features per forward pass via `flax.linen` submodules or a user-managed cache; nflojax does not enforce memoisation.

### §8 Module charter

**[STRONG]** The charter is good but missing a bucket: **fixtures and shared test utilities** currently live in `tests/conftest.py`. If the conditioner contract is going to be tested identically across `MLP`/`DeepSets`/`Transformer`/`GNN`, the shared fixture needs a canonical home. Either say "tests/conftest.py owns cross-primitive fixtures" or add `tests/_protocol.py`.

**[NIT]** `utils/pbc.py` charter says *"Not for forces, energies, or neighbour lists with cutoffs"*. Neighbour-list construction under PBC is geometry, not physics. If a conditioner needs a fixed-cutoff neighbour list (which `GNN` will), that code is flow-side and should live somewhere — probably `utils/pbc.py` or `utils/neighbours.py`. Current charter forbids it there. Clarify.

### §9 Decision heuristics

**[NIT]** Heuristic 1 ("does every target application need it?") is strict; replace "every" with "two or more". Otherwise `LatticeBase.hex_ice` fails the test today (bgmat uses it, DM uses it, chemistry apps don't — and "every" makes the cut ambiguous).

### §10 What stays in applications

**[STRONG]** Clean. Add one more: **energy-dependent bijections** (e.g., umbrella bijections, free-energy bijections, score-based bijections). They reference energies by definition; they are application code.

### §11 Verification checklist

**[STRONG]** The grep targets are smart. Add:

- `jraph`, `e3nn`, `haiku`, `distrax`, `tfp`, `tensorflow_probability` — dependency forbiddance.
- `@pytest.mark.slow` and any other test skip markers that don't match `@requires_x64` — skip drift.

Also: add a rule about **numerical-parity tests** against bgmat or DM reference values for every lattice and embedding; these are load-bearing for the parity claim in PLAN §7.G2 and need to be automated, not manual.

### §12 Glossary

**[NIT]** "Particle system" is defined twice (motivation + glossary) with slightly different emphasis. Unify.

**[STRONG]** Missing entries: "augmented flow", "CoM projection", "spherical truncation", "identity-at-init", "conditioner protocol", "featurisation". All are used elsewhere.

### §13 Pointers for agents / §14 Review log

No issues.

---

## 3. PLAN.md — stage-by-stage critique

### Stage A (bijection extensions)

**[STRONG]** A2 (`ShiftCenterOfMass`) assumes the single-primitive model §3.3 proposes; see the `[STRONG]` item on §3.3 above. **Hold A2 until the translation-invariance design is decided.** Ship A1, A3, A4 first.

**[BLOCK]** A4 ("context-type docstring sweep") — see the §5.2 `[STRONG]` item. This is not a docstring sweep; it is either a behaviour change to `MLP` or a narrower rule. Re-scope A4 accordingly, or it will balloon.

**[NIT]** A1 (`Rescale`) — the log-det contribution is trivially per-axis; worth calling out that when `lower`/`upper` are per-axis arrays, the log-det depends on their shape, which affects the batch-shape convention (§5 item 1 above). Test explicitly.

### Stage B (particle-aware base & utils)

**[STRONG]** Stage B mixes four independent pieces of work. Split into:

- **B-lattice-geom**: `utils/lattice.py` + `LatticeBase`.
- **B-pbc**: `utils/pbc.py`.
- **B-uniform**: `UniformBox`.
- **B-pointcloud** (new): `PointCloudBase` per §6 critique above.

Each is ~1 day; bundled as "Stage B" the work is a week and the tests will get sloppy.

**[STRONG]** B2 (lattice generators): the *hard* part is getting FCC / diamond / hex-ice right to agree with DM's `flows_for_atomic_solids/utils/lattice_utils.py` to machine precision after rescaling. Recommend: *the first PR ships the test (a frozen reference against DM's output) before the implementation*. Red → green. This is exactly where hand-coded off-by-ones live.

**[STRONG]** B3 (`LatticeBase`) — who owns the box? Options:

- (a) `LatticeBase` constructs and owns the box from `(n_cells, a)`; flow inherits it.
- (b) User supplies box; `LatticeBase` checks consistency.
- (c) Both allowed.

**Pick (a)** and document it. Users *will* get the cell-count vs box-length math wrong if you let them. The "override box" hook can be added later if someone actually needs it.

**[NIT]** B3 spherical truncation needs a note that the rejection-sampling cost is baked in (`DM`'s approach resamples until the perturbation fits inside the Voronoi radius); this is technically a hidden control-flow step not compatible with a single jit tracing. Check whether DM's implementation uses truncation via rejection (it does) and whether bgmat matches (it does). Either document the performance characteristic or use a differentiable projection onto the truncation sphere.

### Stage C (embeddings)

Low risk. **[NIT]** Add a test that `circular_embed` is genuinely periodic (value at `x=lower` equals value at `x=upper`) and that `positional_embed` output preserves scalar ordering (monotone in `t` for the first component).

### Stage D (conditioners)

**[BLOCK]** Ship `DeepSets` (D1) *alone* in the first release. Reasons:

- `DeepSets` is the test bed for the conditioner protocol. Shipping it alone + D4 (shared fixture) validates the protocol with minimum surface area.
- `Transformer` at ≤150 LOC and matching DM's numerics is genuine research engineering; budget 1–2 weeks, not 1–3 days.
- `GNN` with PBC top-k neighbours + jit compatibility is a similar 1–2 weeks. The top-k → fixed-cutoff trade-off alone is worth a design memo.

Recommended D revised:

- **D1**. `DeepSets` + shared protocol fixture (D4 merged in).
- **D2**. `AttentiveDeepSets` (or `InteractionNet`) — cheaper and more useful than either full Transformer or full GNN for a first pass. Optional.
- **D3**. `Transformer` — after a round of use with D1/D2.
- **D4**. `GNN` — after triclinic and neighbour-list story settles.

Defer D3/D4 to a "Particle flows v2" stage, documented but not in the current plan.

**[STRONG]** For any GNN reference impl: **use a fixed-cutoff padded neighbour list, not top-k**. Top-k under jit requires `jax.lax.top_k` on distances, which is fine numerically but breaks when multiple distances tie or when particles are exactly at the cutoff (needs stable sort; `top_k` isn't stable). Fixed cutoff + padding to `num_neighbours_max` + masking is the standard jit-friendly path (jraph does this, m3gnet does this). Document explicitly.

### Stage E (builder)

**[STRONG]** E1's topology is opinionated. The DM paper's canonical stack differs from bgmat's (bgmat inserts `CircularShift` and `ShiftCenterOfMass` at different positions; DM alternates permutations between, not after, couplings). Two options:

- (a) Ship only one canonical topology; document it is DM-style; users who want bgmat-style build by hand.
- (b) Parametrise topology via a `layer_sequence` argument or a `TopologyRecipe` helper.

(a) is "lean and hackable"; (b) is "batteries included". I'd pick (a) and add a worked `bgmat-style` assembly recipe to `EXTENDING.md`.

**[NIT]** E1's signature will need careful defaults. `num_layers=24, num_bins=16` are DM's values for large systems; for `N=8` smoke tests they are overkill. Ship with smaller defaults (e.g., `num_layers=4, num_bins=8`) + note in docstring.

### Stage F (docs)

**[STRONG]** F3 ("document augmented-coupling pattern in `EXTENDING.md`") — cross-reference the §4 item 9 call above. If we decide to ship augmented-coupling as a thin primitive, F3 changes from docs to code.

**[NIT]** Add **F6**: update `INTERNALS.md` with the expanded §5.4 conditioner contract spec (assuming DESIGN.md gets rewritten).

### Stage G (bgmat prototype)

**[STRONG]** G1's "parity test within `1e-5` under x64" is optimistic. Parity between nflojax-assembled flow and bgmat's Distrax-based flow will be *architectural* (same topology, same random seed, same init), not bit-level. Loosen to: "parity up to architecture-equivalent random init" — i.e., check that the two trained flows reach the same loss trajectory on a fixed dataset, not that the init outputs match.

**[STRONG]** G is the most important stage, not the last one. Run a minimal version of G *before* E:

- Pick bgmat's simplest flow (e.g., mW at N=8).
- Identify what primitives nflojax needs to reassemble it.
- Verify the reassembly runs before adding the builder.

This is the only way to know if the abstraction is right. Leaving G to the end means discovering a missing primitive after committing to five stages.

### Cross-cutting (§8)

**[BLOCK]** Add two checklist items:

- **Performance regression test.** Every stage measures a baseline flow training-step time at a representative N and compares to the previous stage. A 2× slowdown without explanation blocks the commit. Without this, the library will be beautiful and unusable at N=500.
- **Memory regression test.** Same, for peak GPU memory.

**[STRONG]** Item "Total nflojax LOC ≤ 5000" — generous. Current count is ~5.3k lines already (rough count from `wc -l nflojax/*.py`). Update either the budget or the rule.

### Parking lot (§9)

**[STRONG]** Add:

- **Flax `linen` → `nnx` migration.** Flax has announced the migration; when it lands, every conditioner and every `MLP`-using assembly is touched. Not a surprise; plan for it.
- **Checkpoint / API stability policy.** Today there is none. When bgmat depends on nflojax, "we moved an output layer and your checkpoint no longer loads" is a real failure mode.
- **Multi-device / pmap.** Users will want this at some point. Document whether nflojax flows are pmap-friendly (they currently are, because of explicit params; worth stating).
- **E(3) equivariant flow bijections** (not conditioners). Distinct from the conditioner parking-lot item; a `ECouplingLayer` analogue of Garcia Satorras et al. may one day belong in `transforms.py`.

### Open questions (§10)

**[NIT]** Question on `Transformer` attention norm placement — add: also decide attention-weights scaling (`1/sqrt(d_head)` vs `1/d_head`), position encoding (add vs concat), and whether to support cross-attention for context-heavy conditioners. These are choices, not free variables.

**[STRONG]** Missing question: **what is the public API surface?** `__init__.py` is empty today. Do we keep it empty (navigate-by-filename) or do we export the 20 most-used names? Users will patch this themselves otherwise. Pick one.

### Decision log (§11)

No issues. Keep updating it.

---

## 4. Physics / materials-sampling concerns DESIGN.md under-addresses

These are pitfalls experienced in real Boltzmann-generator work that the docs haven't acknowledged yet. Each is a candidate for either a DESIGN.md amendment or a parking-lot entry.

1. **Defect sampling** — a flow trained on a perfect crystal lattice will not produce defect configurations (vacancies, interstitials). This is a base-distribution limitation, not a flow limitation, and the user will be surprised. Document under `LatticeBase`.

2. **Mode collapse in reverse-KL training** — endemic to Boltzmann generators on multi-modal targets. nflojax doesn't train; still, the assembly-time choice of base interacts with mode coverage. If `LatticeBase` samples random permutations, mode coverage is easier; if not, the flow will lock onto one labelling. Worth flagging in `USAGE.md`.

3. **Finite-size effects at small N** — flows trained at N=64 extrapolate poorly to N=512 unless the conditioner is explicitly transferable. bgmat documents this; nflojax should note it in the GNN / DeepSets docstring (not every equivariant conditioner transfers).

4. **Low-temperature instability** — at low T the energy landscape becomes sharp and the RQ-spline may need more bins. `num_bins` defaults to 8; the DM paper uses 16 for mW. Note in the builder defaults.

5. **Rotational symmetry breaking** — for crystals, the user typically fixes the lattice orientation. Rotating the ground-truth crystal produces a new mode that the flow does not cover. SE(3) equivariance would fix this; we are not shipping it. Worth a one-line warning.

6. **Pressure / volume coupling** — NPT ensembles require the flow to act on both coordinates and box volume. bgmat's `ShapeFlow` is exactly this. Out of scope today, but the parking lot should name it explicitly.

7. **Long autocorrelations at the melting transition** — liquids near freezing have long-range correlations that local GNNs with small cutoffs cannot capture. The reference GNN should ship with a `num_neighbours` default large enough to cover typical first coordination shells (≥12 for FCC, ≥18 for bcc).

8. **Supercooled / glassy systems** — non-ergodic on timescales accessible to a sampler; a trained Boltzmann flow might sample configurations MD would never reach (which is a feature, not a bug, but the user needs to validate carefully). Parking-lot entry: *"quality metrics for glass samples vs MD samples — application code".*

9. **Mixtures (binary LJ, multi-species mW)** — require species-aware conditioner and species-aware base. bgmat hasn't done this yet; nflojax's `BlockPermutation` parking-lot entry is on the right track but under-specified. Worth a one-paragraph expansion of §7.5.

10. **Numerical stability at large N** — `SplitCoupling` at N=1000 requires the MLP/Transformer to emit `transformed_flat * (3K)` outputs; at K=16 and N=500 and d=3 that is 750 * 48 = 36000 scalar outputs per forward pass per layer. Memory-ok, but NaN-risky without care. Worth a numerical-robustness amendment to §2 item 5.

---

## 5. Proposed amendments (concrete, prioritised)

The amendments below can be applied in one pass to the two documents. Each has a one-line rationale and a rough size.

### Must-do before starting code

1. **Rewrite DESIGN.md §5.4 conditioner contract** to include input shape, output shape, featurisation ownership, context contract, and identity-at-init-by-layer-type. [~1 page.]
2. **Replace DESIGN.md §3.3 "Centre-of-mass bijector" with a decision**: ship neither / both / one-and-document-the-other. Update PLAN.md Stage A2 accordingly. [2 lines in DESIGN.md; 3 lines in PLAN.md.]
3. **Decide: augmented-coupling primitive vs pattern.** If primitive: add to PLAN.md as a new stage between D and E. If pattern: expand `EXTENDING.md` with the full recipe including the three gotchas. [~30 LOC of code either way; ~1 page of docs.]
4. **Split PLAN.md Stage B** into four subtasks (lattice-geom, pbc, uniform, pointcloud) and Stage D into four subtasks (deepsets, attentive-deepsets or defer, transformer-defer, gnn-defer). [~5 min.]
5. **Add `PointCloudBase` to DESIGN.md §7.1 and PLAN.md Stage B.** Makes "liquids supported" real. [~40 LOC + tests.]
6. **Re-scope PLAN.md A4** (context-type). Decide: MLP behaviour change or narrower rule. [~10 min.]
7. **Fix DESIGN.md §5.1 log-det convention** to two allowed shapes (scalar OR batch). [2 lines.]

### Should-do before starting code

8. **Add G-as-first-stage**: a dry-run bgmat-on-nflojax assembly with the current feature-set to catch missing primitives before investing in E. [~2 hours.]
9. **Add benchmark + memory regression to §8 cross-cutting checklist.** [~5 lines.]
10. **Expand §11 grep targets** to catch Jraph / e3nn / distrax / tfp. [1 line.]
11. **Demote `DeepSets` to "pedagogical" in DESIGN.md §7.2**; add `AttentiveDeepSets` as the realistic reference or defer non-MLP reference conditioners to v2. [2 lines in DESIGN.md; PLAN.md Stage D adjusted.]
12. **Rewrite DESIGN.md §6 solid-vs-liquid framing** in terms of localised vs delocalised vs data-driven bases. [~½ page.]

### Nice-to-have before starting code

13. **Add DESIGN.md section on numerical stability at scale** (memory, NaN detection, GPU-specific gotchas). [½ page.]
14. **Unify glossary entries** (particle system, context, augmented flow, CoM projection, spherical truncation, identity-at-init, featurisation). [½ page.]
15. **Add `DESIGN.md` entries for checkpoint-stability policy and Flax `linen→nnx` migration.** [2 lines each.]
16. **Public-API-surface decision** (empty `__init__.py` vs curated exports). [1-line rule + implementation.]

---

## 6. Things to keep thinking about (open problems)

- How to evaluate a flow library that does *not* ship training or losses. The natural answer is "with the downstream app"; but then every change to nflojax requires a downstream regression. Need a lightweight *internal* quality signal that does not depend on an energy.
- How to version conditioner architectures. If `Transformer` ships and someone trains a 10⁶-step mW flow with it, and we later refactor attention normalisation, their checkpoint is dead. Do we lock reference architectures at 1.0 and branch from them only in new classes (`TransformerV2`)? Ordinary semver is too coarse.
- Relationship to continuous-time flows. `CNF`, `FFJORD`, neural-ODE couplings don't fit the current `forward/inverse → (y, log_det)` contract cleanly (log-det is an ODE solve). Out of scope today, but a future maintainer may want them. Parking-lot entry worth writing.
- Whether to absorb score-based / diffusion-ish primitives. bgmat may or may not want these; "Boltzmann generator" is a research field moving towards CNF / diffusion hybrids. Worth one sentence in §4 saying "not today; reconsider in 2027".
- Whether `LatticeBase` and `PointCloudBase` are actually the same class with different noise profiles. A case can be made.

---

## 7. Recommended action sequence

Treat this as an ordered checklist applied before resuming coding.

1. Apply amendments 1, 2, 7 to DESIGN.md. (~1 hour.)
2. Apply amendment 4 to PLAN.md. (~15 min.)
3. Apply amendments 3, 5, 6 — decisions first, then text. (~1 hour of thinking, ~½ hour of writing.)
4. Apply amendment 8 — one day of exploratory work; may surface new primitives. (~1 day.)
5. Apply amendments 9, 10, 11 to DESIGN.md / PLAN.md. (~30 min.)
6. Apply amendments 12, 13 (re-frame solid/liquid, add numerical stability). (~1 hour.)
7. Commit DESIGN.md, PLAN.md, `audit.md` as a `docs:` chunk on `feature/particle-events` (or a new `docs/design-charter` branch) **before** any Stage A work starts.
8. Open Stage A (A1, A3 only — A2 blocked by amendment 2; A4 re-scoped by amendment 6).

Total: ~2 days of focused work before any new primitive lands. In exchange: the conditioner contract, the CoM story, the augmented-flow story, the solid/liquid framing, and the stage sizing are all clean before code is written against them.

---

## 8. One-paragraph bottom line (first pass)

The design and plan are solid starts and unusually disciplined. Before executing, do a *second* pass on four specific surfaces — the conditioner contract (§5.4), the translation-invariance mechanism (§3.3 + §3.3's proposed `ShiftCenterOfMass`), the augmented-coupling decision (§4 item 9), and the solid/liquid framing (§6) — and re-size the work items in PLAN.md so that each stage is actually a week of work, not a week of work compressed into "one commit". Do amendment 8 (bgmat dry-run) early; everything else follows from what you learn there.

**Addendum after §9 and §10 below.** The first pass was honest but not meticulous on symmetry. The expanded bottom line: CoM becomes *three* primitives, not one (`CoMProjection` ships, the other two are recipes); permutation-invariance gains a named test fixture; rotation / point-group symmetry stays out of nflojax but gets a clean "why and what to do instead" subsection in DESIGN.md; a new DESIGN.md "Symmetries we care about" section is the single most impactful addition. Total added work vs. the first pass: ~1 day of docs + ~150 LOC of tests; no new major primitives beyond what amendments 1–16 already imply.

---

## 9. Symmetries & invariances — the full picture

The first pass mentioned permutation, translation, and rotation in passing. Those three, plus two others (gauge, point group), are the defining concerns of a particle flow; the rest of the library is in service of them. Getting any of them wrong produces a flow that *appears to work* (trains, samples without NaN, round-trips cleanly) while silently producing a different distribution than the user thinks. This section treats them one by one.

### 9.1 The five groups that actually matter

| Group | What it is | Target density | Is flow typically equivariant? | Where is the responsibility |
|---|---|---|---|---|
| **Sn** | Permutation of indistinguishable particles | Invariant | Sometimes | Shared: base samples permutation; conditioner is equivariant |
| **T(d)** | Continuous translation by a box vector | Invariant under PBC | Rarely | Base (sample a uniform shift) OR flow (project out CoM) |
| **O(d)** / **SO(d)** | Rotation + optional inversion | Invariant for fluids; broken for crystals | Almost never | User, via architecture choice (out of nflojax scope) |
| **Lattice point group** | Residual discrete symmetry after the crystal breaks O(d) | Invariant for the chosen crystal | Never (in practice) | User, and almost always deliberately ignored |
| **Gauge (box origin)** | Continuous Sd of "where is the torus origin" | Invariant | Yes, if the flow wraps correctly | `CircularShift` + base construction |

The rest of §9 treats each row.

### 9.2 Permutation (Sn) — the indistinguishability contract

Classical particles are indistinguishable. `p(x_1, ..., x_N) = p(x_{π(1)}, ..., x_{π(N)})` for every permutation π. A flow is *Sn-invariant in log-prob* iff:

1. The **base** is Sn-invariant: `p_base(x) = (1/N!) Σ_π p_base(π⁻¹ x)` (a fact, not a test). This is achieved by sampling a random permutation inside the base's `sample`; the base's `log_prob` includes `-log N!`.
2. The **flow** is Sn-equivariant: `f(πx) = πf(x)` as a mapping. A coupling with a permutation-equivariant conditioner (`Transformer`, `GNN`, `DeepSets` all qualify; `MLP` does not) is equivariant. A `Permutation(event_axis=-2)` layer composed into the stack is equivariant **only if** it applies the *same* permutation on the forward and inverse paths (jit will enforce this provided the permutation is a dataclass field, not randomly resampled at each call).

Five concrete failure modes worth testing:

- **Half-wired Sn.** User writes a `DiagNormal` on `(N, d)` as base (not Sn-invariant; mean/scale are per-particle) + a Transformer conditioner (Sn-equivariant). The flow is Sn-equivariant as a map but not Sn-invariant in density. `log_prob(x) ≠ log_prob(π x)`. The user may not notice until they see a free-energy comparison fail.
- **Shuffled permutations across forward/inverse.** A dynamically-sampled `Permutation(key)` that doesn't thread its key deterministically will produce inconsistent permutations; round-trip tests pass *per call* but not across calls.
- **Silent break via context.** The conditioner sees a context that is not itself Sn-equivariant (e.g., per-particle species label with a fixed order). Sn-invariance is quietly lost; tests may not catch it unless they permute context too.
- **Lattice-particle assignment.** `LatticeBase.fcc(permute=False)` pins particle `i` to lattice site `i`. A flow built on this base is distinguishable-particle, not indistinguishable-particle. Training against an indistinguishable-particle energy will over-count by `N!` (constant, harmless for most losses but catastrophic for log-partition estimates).
- **Finite-sum symmetrisation** — users trying to patch Sn-invariance post-hoc by averaging `log_prob` over a few random permutations. Statistically biased unless averaged over the full group; this is a bad habit worth warning against in docs.

**Recommendations.**

- **Add a test fixture** `flow_is_permutation_invariant(flow, params, key, atol)` that draws random permutations and checks `log_prob`. Cheap; catches all five failure modes.
- **Document the two-bit table** (base Sn-invariant × conditioner Sn-equivariant) in DESIGN.md: only the `(✓, ✓)` quadrant yields an Sn-invariant density.
- **Be explicit in `LatticeBase`**: `permute=True` is Sn-invariant; `permute=False` is not. Default should be `True` for materials workflows. Warn (not raise) when `permute=False` is combined with an Sn-equivariant conditioner — it's legal but usually not intended.
- **Document that `MLP` on particle events breaks Sn-equivariance** (because concatenating an `(N, d)` input to a flat vector assigns meaning to particle index). This already happens in current code; users need to know.

### 9.3 Translation (T(d) on the torus) — CoM, realistically

Under PBC, the target density satisfies `p(x) = p(x + t · 𝟙_N)` for every box vector `t`. This is a gauge-like symmetry: the `Nd` ambient coordinates carry only `(N-1)d` internal-motion dof plus `d` gauge dof. What to do with those `d` gauge dof defines four architectures seen in the wild. The first-pass audit's one-liner (DM vs bgmat) is a caricature; here is the complete story:

**Strategy A — Do nothing.** Train the flow on the full `(N, d)` with no CoM handling. The density is already translation-invariant (if the base is), so the flow trains on a redundant parametrisation. Works; wastes `d` learnable dof; empirically converges slower by a small constant factor. This is what you get if you compose `LatticeBase(permute=True) → SplineCouplings → ...` with nothing else.

**Strategy B — CoM projection (Köhler et al.-style).** Make the flow a bijection from `(N-1)d` internal dof to `(N-1)d` internal dof; reconstruct the Nth particle as `-Σ_{i<N} x_i`. Log-prob accounting: the projection is *not* length-preserving in ambient space, so `log p_ambient(x) = log p_internal(P(x)) + log det P'`, where `P` is the projection and `log det P'` includes the `-½ log N` term from the projection matrix plus any translation uniformity. Köhler et al. (2020 "Equivariant Flows on Lie Groups") works this out explicitly. **Clean** as a bijection; **non-trivial** as an accounting exercise.

**Strategy C — Sample translation explicitly (DM-style).** Flow operates on the `(N-1)d` internal dof; the CoM is sampled separately from `Uniform(box)`. Total log-prob: `log p_internal(x − ⟨x⟩) + (−log V)`. The `-log V` term is the flat density of the uniform CoM. This is what `flows_for_atomic_solids/models/particle_models.py::TranslationInvariant` does. Less elegant than Strategy B but simpler to audit.

**Strategy D — Augmented CoM swap (bgmat-style).** Flow operates on `2N·d` augmented coordinates; at designated layers, a `ShiftCenterOfMass(swap=True)` bijector subtracts one half's CoM from both halves. Log-det per swap: constant `±1` (the Jacobian is a permutation of axes). Does not project out the CoM globally; instead, the augmented structure maintains a constraint that the physical half has zero CoM, with the auxiliary half absorbing the translation. **Specific to augmented flows**; only makes sense if you also adopt Strategy 4 of §4 item 9 (augmented coupling).

**Recommendations, concrete.**

- **Ship `CoMProjection` as the default primitive.** Forward: `(N, d) → (N-1, d)`. Inverse: `(N-1, d) → (N, d)` reconstructing the last particle. Log-det: closed form (`-½ log N · d`, up to sign and conventions). Covers Strategy B cleanly as a bijection.
- **Do not ship a generic `ShiftCenterOfMass(event_axis=-2)` primitive.** The name is ambiguous; users will mis-apply it to Strategy A, C, or D and get silently wrong results. **Retire this name** from PLAN.md Stage A2 in favour of `CoMProjection`.
- **Document Strategy C as a recipe** in `EXTENDING.md`: a `TranslationInvariantFlow(inner_flow, box)` wrapper that samples uniform CoM. 20 LOC; not a primitive.
- **Document Strategy D as part of the augmented-coupling pattern** (or primitive, per §4 item 9): a `CoMSwapAugmented(swap: bool)` bijector acting on `(2N, d)`. Strictly specific to augmented setups.
- **Warn against Strategy A in docs**: it works but wastes capacity; mention the alternatives first.

### 9.4 Rotation / O(d) / SO(d) / inversion

This is the group most often confused in this subfield. The key distinction is **whether the target density is itself O(d)-invariant**.

**Isotropic fluids (liquid, gas, dilute).** Target is O(d)-invariant. A flow that is not O(d)-equivariant *can still train and sample correctly* — it's just using 1/|O(d)| of its effective capacity per configuration. At large N, this is fine (the lost capacity is small relative to `N!`); at small N (clusters), it's crippling.

**Crystals.** Target is NOT O(d)-invariant, because the lattice spontaneously breaks O(d) to the point group (T_d for diamond, O_h for FCC/cubic, C_6v for hex-ice). The user has made a choice of lattice orientation; the flow should match that choice. O(d)-equivariance in the flow would be *wrong* — it would try to symmetrise away the orientation the user fixed. This is a source of endless confusion: "equivariance is good" is true in general but false for crystals in the naive sense.

**Clusters, molecules (isolated).** Target IS O(d)-invariant (no lattice to break the symmetry). E(3)-equivariant flows (Köhler et al. 2020; Garcia Satorras et al. 2021; Bose et al. 2022) are the right tool. None of them is trivially expressible in nflojax's current primitive set; users must bring the architecture.

**What nflojax ships.** Nothing for O(d)-equivariance. Users who need it bring an EGNN / NequIP / MACE / spherical-CNF conditioner. The `Conditioner` protocol handles this through context (invariant features in, scalar params out), but nflojax does not provide the conditioner.

**What nflojax COULD usefully ship.**

- **`rotation_symmetrised_log_prob(flow, params, x, n_rotations, key)`** — a validation-time utility for isotropic fluids: sample `n_rotations` Haar-random rotations, average `log_prob`. ~10 LOC. Documented as "validation / analysis, not training" because it makes training prohibitively expensive.
- **Documentation** that makes clear (a) for crystals, you want the flow to respect the chosen orientation — do not symmetrise; (b) for fluids, you can optionally symmetrise in `log_prob` evaluations; (c) for clusters, you need equivariant primitives that nflojax does not ship.

### 9.5 Gauge (box origin)

The torus has no preferred origin. Under PBC, the map `x → x + t (mod box)` is a symmetry of the density. This is *not the same* as translation invariance (§9.3): §9.3 is about translating *all particles together*, which is a symmetry of the *absolute* positions; §9.5 is about the choice of where the box's origin lies, which affects the *representation* of the same physical state.

In practice the two become entangled under PBC. A flow whose coupling layers are origin-dependent (because they split along `x < 0` vs `x > 0` in some arbitrary gauge) can over-fit to the gauge. `CircularShift` with a *learnable* per-coord shift discovers the optimal gauge during training; this is effectively gauge-fixing.

**Observations that should be documented.**

- If the user does CoM projection (§9.3 B), they have already fixed the translation gauge. Adding a learnable `CircularShift` on top becomes a redundant parameter (gradient will be zero modulo numerical noise). Not wrong, wasteful.
- If the user does NOT do CoM projection, `CircularShift` is load-bearing: it lets the flow choose the gauge. Keep it.
- The audit's amendment 2 should clarify the intended interaction with `CoMProjection`: one or the other, not both.

### 9.6 Lattice point group — why we do not touch it

After O(d) is broken to the lattice point group (48 for O_h; 24 for T_d; 12 for C_6v), a further discrete symmetrisation is possible. In principle a `PointGroupSymmetrised(flow, group_ops)` wrapper would symmetrise `log_prob` over the 48 copies. In practice:

- The base (`LatticeBase`) has a fixed crystallographic orientation, so symmetrising over the point group would force averaging over 48 different base distributions — expensive, does not match what the user wants.
- The conditioner is not point-group-equivariant by construction; it would need to be designed specifically (via invariant-feature extractors).
- Nobody in the literature has seriously deployed point-group-equivariant flows for crystals. The marginal benefit is small because the flow already sees many equivalent configurations through Sn-invariance + permutation sampling.

**Call.** Explicit non-goal in DESIGN.md. Do not add `PointGroupSymmetrised` even as a recipe. If an application needs it, they write a ~20-line wrapper over `log_prob`.

### 9.7 Interactions: the working symmetry set

For nflojax today:

- **Sn × T(d)** is the realistic operating regime. The base provides Sn-invariance (via `permute=True`) and optionally T(d)-invariance (via built-in CoM projection); the conditioner provides Sn-equivariance; `CircularShift` + circular splines provide torus wrapping.
- **Everything else** (rotation, inversion, point group) is user-provided or deliberately not enforced.

This needs to be a table, not a sentence. In a new DESIGN.md "Symmetries" section, ship something like:

```
              base          conditioner     flow
Sn            ✓ (permute)   ✓ (equivariant)  → Sn-invariant density
T(d)          ✓ (CoM proj) OR ✓ (uniform CoM)   → T(d)-invariant density
O(d)          ✗             ✗                 → user responsibility
Point group   ✗             ✗                 → non-goal
Gauge         ✓ (CircularShift)              → torus-correct
```

### 9.8 Concrete additions to DESIGN.md

A new section **§15. Symmetries we care about** (after the glossary) with the content of §9.1, §9.7 table, §9.2 two-bit table, and one paragraph each on T(d), O(d), gauge, point group. ~1.5 pages. This is the single highest-value DESIGN.md addition the audit has identified.

---

## 10. Other topics the first pass under-weighted

Less meticulous than §9; more of a punch list with one paragraph each. Each is a candidate DESIGN.md addition, a PLAN.md parking-lot entry, or a test.

### 10.1 Connectivity / neighbour-list as a discrete jump variable

GNNs that build their edge list from a cutoff or top-k lookup at every forward pass produce `log_prob` that is **non-smooth** when a pair of particles crosses the cutoff: the neighbour list changes, so the message graph changes, so the output changes discontinuously. Gradients through `lax.top_k` or a hard cutoff do not exist at the cutoff boundary. Mitigations: smooth cutoff functions (`f(r) = tanh((r − r_c)/δ)` weighting messages instead of hard inclusion); continuous top-k (Sinkhorn-style). Add to DESIGN.md §7.2 caveat on the GNN reference impl.

### 10.2 Log-det sign conventions

Every primitive must return `log|det J_f|` for `forward` and `log|det J_{f⁻¹}| = −log|det J_f|` for `inverse`. The current codebase gets this right but it is not documented as a contract. Add to DESIGN.md §5.5. Add a per-primitive assertion `log_det_forward(params, x) + log_det_inverse(params, forward(x)) ≈ 0` to the test battery (most tests assert this implicitly; make it named and shared).

### 10.3 Reversibility under jit

Every primitive is a bijection in principle; not every primitive is invertible under jit without warnings (LOFT has branches; spline inverse requires a quadratic solver). The existing tests cover invertibility but not "invertible + jittable + shape-stable under arbitrary batch dims". Worth an explicit contract test.

### 10.4 Mixed precision

Default is float32; `@requires_x64` handles the x64 test carve-out. Bfloat16 / float16 are not supported: `_normalize_bin_params` uses `jnn.sigmoid` in a way that can underflow in bf16. Document in DESIGN.md §2.8 as an explicit non-goal; do not promise bf16 correctness.

### 10.5 Determinism

For a fixed PRNG key, fixed params, and fixed input, a flow's outputs should be bit-identical across runs. This holds under jit with autotune disabled. Worth a one-liner in DESIGN.md §11 checks: `XLA_FLAGS=--xla_gpu_autotune_level=0` for reproducibility.

### 10.6 NPT / volume coupling

The target density for NPT ensembles is `p(x, V) ∝ e^{-β(E(x) + PV)}`. Flows for NPT need to act on both coordinates and box volume. bgmat's `ShapeFlow` does this; nflojax does not. Parking-lot entry with a pointer to bgmat's construction; not in scope today.

### 10.7 Internal coordinates (molecules)

Bonds, angles, torsions — a completely different parametrisation of the configuration space. E(3)-invariant by construction. Relevant flow libraries: Köhler–Klein–Noé 2020 for `SO(3)` and torus distributions; Midgley et al. 2023 for boltzmann generators on molecules. Out of scope today; parking-lot entry.

### 10.8 Isotope / mass differences

Classical positions-only flows don't care about masses. Path-integral flows (quantum thermal averages) do. Out of scope today; mention parking-lot.

### 10.9 Normal-mode / mass-weighted coordinates

For low-T crystal sampling, transforming Cartesian → normal modes gives a diagonal base distribution (independent Gaussians per mode) that is orders of magnitude easier to sample. This is a *base-distribution trick*, not a flow trick, and it belongs in nflojax if anywhere. A `NormalModeBase(lattice, phonon_frequencies)` would be ~80 LOC and would immediately unlock efficient low-T materials sampling. **Worth considering for Stage B**.

### 10.10 Phase coexistence / broken ergodicity

A sampler that never jumps between two coexisting phases is wrong at the level that matters. nflojax does not train, so it does not enforce ergodicity; however, the choice of base distribution determines which phases the flow can reach. A `LatticeBase.fcc` flow will not sample hex-ice configurations. This is correct behaviour, but users need to understand it. Document in `USAGE.md` under "Picking a base".

### 10.11 Checkpoint migrations / API stability

When a user trains a flow against `nflojax 1.3` and we refactor attention normalisation in `nflojax 1.4`, their checkpoint's params PyTree will no longer load. Options: (a) freeze reference conditioners architecturally at v1.0 and only add new classes for breaking changes (`Transformer` → `TransformerV2`); (b) provide a `load_legacy(old_params, old_version, new_class)` migration utility. (a) is cheaper and consistent with "reference implementations, not authoritative". Document it.

### 10.12 Hardware specifics: TPU vs GPU

`segment_sum` and `scatter` behave slightly differently on TPU (padded; may have reproducibility differences). Not a showstopper but worth a note if bgmat or downstream apps target TPU.

### 10.13 Golden-sample regression tests

Best practice for a flow library: check in a handful of fixed `(seed, params, expected_sample, expected_log_prob)` tuples as `tests/fixtures/`. Runs under float64 for tight tolerance. Catches silent numerical regressions (e.g., a Flax API change that reorders operations) that unit tests don't. Add to PLAN.md cross-cutting checklist.

### 10.14 Error messages

Current code raises `ValueError` on shape mismatches with useful messages. Spot-audit suggests this is usually good; worth a deliberate pass.

### 10.15 Interoperability with ASE / LAMMPS / OpenMM / GROMACS

Out of scope; the bridge lives in applications. Mention parking-lot.

### 10.16 Visualisation of flow samples

Not in scope (no RDF / structure factor / etc. in nflojax). But a tiny `tools/plot_samples.py` or a notebook in `USAGE.md` would save users days of "is my flow doing anything". Optional; consider as a docs addition.

### 10.17 Composition sanity: shape propagation under jit

A failure mode not covered by current tests: a coupling whose output shape depends on trace time rather than on the primitive's declared event shape. Should add a `test_shape_propagation.py` that traces every primitive under a range of event-shape scenarios and confirms output shape matches the declared contract.

### 10.18 Randomness in base sampling

`LatticeBase.sample(key)` with `permute=True` consumes key bits for both the permutation and the noise. The key-splitting convention matters: if we split `(key_perm, key_noise) = jax.random.split(key, 2)`, this needs to be documented so users who want to reconstruct a specific sample can. Small but load-bearing.

### 10.19 Handling of zero-particle edge cases

`N=0` is an invalid but possible call; `N=1` has no non-trivial couplings; `N=2` has one coupling that is trivially permutation-invariant. These edge cases should raise clear errors or be documented as supported. Add to per-primitive tests.

### 10.20 Documentation for "why this and not X"

Prospective users will ask "why not nflows / Distrax / FlowJax / Zuko / TFP?". DESIGN.md should have a brief comparison table or paragraph. Not a scope question; a positioning question.

---

## 11. Additional amendments (from §9 and §10)

Appending to the amendments list in §5:

17. **[MUST] Add DESIGN.md §15 "Symmetries we care about"** with §9.1–§9.7 content. The single highest-value doc addition.
18. **[MUST] Rename `ShiftCenterOfMass` → `CoMProjection`** as the single shipped primitive (Strategy B). Document Strategies A, C, D as non-primitives. Update PLAN.md Stage A2 language and content.
19. **[MUST] Add permutation-invariance test fixture** `tests/test_permutation_invariance.py` — a helper that, given a flow asserting Sn-invariance, verifies `log_prob(x) ≈ log_prob(π x)`. Reusable across `LatticeBase` and user assemblies.
20. **[SHOULD] Add rotation-symmetrised log-prob utility** `rotation_symmetrised_log_prob(flow, params, x, n_rotations, key)` in `nflojax/observables-validation.py` or a similar file. Or document as a 10-line recipe in `USAGE.md`. (Tension: "no observables" rule. This is an analysis utility, not a physics observable; edge case.)
21. **[SHOULD] Document gauge-vs-CoM interaction** in DESIGN.md §15: `CircularShift` is gauge-fixing; `CoMProjection` is translation-fixing; use one or the other, usually not both.
22. **[SHOULD] Add parking-lot entries to PLAN.md §9**: NPT / `ShapeFlow`, internal-coordinate molecules, normal-mode bases, E(3)-equivariant flows (as bijections, distinct from conditioners), mixed-precision, checkpoint migration policy.
23. **[SHOULD] Add log-det sign-convention assertion** to per-primitive test battery.
24. **[SHOULD] Add golden-sample regression fixtures** for `LatticeBase` factories and `circular_embed`.
25. **[NIT] Add "why not $other_flow_library" paragraph** to DESIGN.md or README.md — one paragraph positions nflojax for users coming from nflows / Distrax / FlowJax.
26. **[NIT] Add smooth-cutoff documentation** for GNN reference impl — neighbour-list discontinuity mitigations.
27. **[SHOULD] Consider `NormalModeBase`** for Stage B — a lattice base in the phonon basis. Biggest efficiency win for low-T crystal sampling; fits the charter.

---

## 12. Looking five steps ahead — second-pass critique

The first pass audited the documents at their own level: is the scope right, are the abstractions right, is the plan right-sized? This second pass changes the frame: *what decisions are we making today that will hurt in 6–18 months, and what can only be fixed now?* Items are tagged by time horizon.

Nothing here is a reason to delay. Several are reasons to adjust defaults before typing the first Stage-A primitive.

### 12.1 [RESEARCH, high confidence] Coupling flows are a shrinking island

Since 2023, the centre of gravity for generative models on particle systems has shifted away from coupling-based normalizing flows toward:

- **Flow matching** (Lipman et al. 2023; Chen & Lipman 2023; Pooladian et al. 2023). O(1) memory in num_layers, cleaner loss landscape, trivially composable. Already the default in TorchCFM.
- **Score-based diffusion** / **Stochastic interpolants** (Song, Sohl-Dickstein, Albergo-Vanden-Eijnden). Different loss; different architectures (U-nets / transformers, not coupling).
- **Continuous normalizing flows on Lie groups** (Köhler–Klein–Noé lineage, MuMoNICE 2024) for molecular workflows.
- **Boltzmann-generator augmentations** (stochastic normalizing flows, BG-diffusion hybrids) — bgmat's augmented-coupling is closer to this lineage than to vanilla coupling flows.

nflojax is a coupling-flow library by construction. The `Bijection` contract (`forward`, `inverse`, `log_det`; O(1) in both directions) encodes this choice. Coupling flows have genuine advantages — exact log-likelihood, cheap inverse, jit-friendly — but they are no longer the dominant paradigm for new materials-sampling work.

**Implication for DESIGN.md.** Add a paragraph to §1 (motivation) that explicitly names the paradigm choice and why: "coupling flows give exact log-likelihood in both directions in a single forward pass, which is what reverse-KL Boltzmann training needs. Diffusion / flow matching give cheaper training at the cost of more expensive log-prob evaluation, a trade-off the target applications can't afford." This positions the library for users who will ask "why not flow matching" and gives the honest answer.

**Implication for PLAN.md.** Parking-lot entry acknowledging the paradigm shift and naming an exit: if in 18 months every target application has migrated to flow matching, nflojax is done. Not a prediction; a contingency.

**Amendment candidate.** [SHOULD] Add DESIGN.md §1 paragraph + PLAN.md parking lot.

### 12.2 [6mo, certainty] The Flax `linen` → `nnx` transition will rewrite every conditioner

Flax has publicly committed to `nnx` as the successor to `linen`. Migration semantics:

- `nnx` uses pure-Python objects with mutable state; `linen` uses pure functions with immutable params.
- Initialization is eager in `nnx`, deferred (via `.init(key, x)`) in `linen`.
- The `.apply({"params": p}, x)` pattern does not exist in `nnx`.
- User code depending on the param-tree structure breaks.

Every conditioner in nflojax (`MLP`, `ResNet`, plus the planned `DeepSets`, `Transformer`, `GNN`) is `linen`. When bgmat depends on nflojax and trains a flow with params in the `linen` layout, a future upgrade to `nnx` invalidates the checkpoint.

**What to do.**

1. **Don't migrate eagerly.** `nnx` is still settling. Migrating in 2026 costs a lot and buys nothing.
2. **Insulate against it now.** Write the reference conditioners so that the neural-network *signature* is independent of whether the backing module is `linen` or `nnx`. A thin wrapper (`nets._linen_mlp`, `nets._nnx_mlp` dispatched by a module-level flag) is ~30 LOC and makes the migration a find-and-replace.
3. **Consider: drop Flax for the trivially-simple parts.** `MLP` is 20 lines of pure JAX. Handwritten `Dense` layers using `jax.random.normal` initialisation and pure `jnp.dot` are another 15 lines. Pure-JAX nets would completely eliminate the Flax dependency for MLP/DeepSets and confine Flax to the Transformer / GNN paths. This is the cleanest long-term bet. Cost: ~40 LOC of initialisation and save-restore utilities.

**Amendment candidate.** [MUST, before Stage D] Decide: eager migration, dual-backend wrapper, or Flax-free for simple nets. The decision constrains the whole conditioner family.

### 12.3 [12mo+, high confidence] Reference conditioners are a maintenance debt trap

Shipping `DeepSets` + `Transformer` + `GNN` as reference conditioners signs nflojax up for maintaining three mini-architectures. In 18 months:

- **`DeepSets`** — stable. Too simple to need changes.
- **`Transformer`** — users will compare against modern transformers (pre-norm / post-norm is already dated; rotary embeddings are now standard; flash-attention is the baseline on large-N; grouped-query-attention for memory). Each of these is a "shouldn't we add..." request with a PR. Either nflojax keeps up and balloons, or it falls behind and users replace it anyway.
- **`GNN`** — even worse. MPNN variants evolve rapidly (attention-augmented MPNNs, equivariant message passing, higher-order messages). Each is a PR.

**Proposed re-frame.** Ship **one** reference conditioner: `DeepSets`. Document the conditioner protocol exhaustively. Point users at established libraries (or their own code) for anything fancier. bgmat's GNN is already out-of-library in this proposal; that's fine — and it's also fine to never ship a reference Transformer or GNN.

Alternative: ship Transformer and GNN but **mark them explicitly "v1, not intended for cutting-edge performance"** with a link to alternatives. This sets expectations and reduces the "shouldn't we add X" pressure.

**Amendment candidate.** [SHOULD, before Stage D] Reconsider the scope of reference conditioners. Defer Transformer + GNN to a v2 release; ship DeepSets alone in v1; lean into "we are the contract, not the implementations".

### 12.4 [12mo+, medium confidence] The coupling-flow scaling ceiling is paradigm, not engineering

At N = 500, a full-attention Transformer conditioner needs O(N²) memory per layer; 24 layers × 500 atoms × 500 pairwise × ~64 features × 4 bytes = ~3 GB just for attention weights. At N = 2000 this is ~48 GB — exceeds any single GPU.

Mitigations for coupling flows:
- **Sparse attention** (local windows). Hurts expressivity; not a free lunch.
- **Linear attention** (Performer / Linformer). Changes the architecture; nobody in the Boltzmann-sampler literature has validated this works at materials quality.
- **Chunked inversion.** Structural change to the coupling layer.

**The structural cost.** Coupling flows need `num_layers` passes through the full conditioner, forward AND backward. Flow matching avoids this: one pass, one gradient, O(1) conditioner calls. For N > 1000, flow matching is likely the only tractable option.

**Implication.** DESIGN.md should state the target scale honestly:

- N up to ~500 on single GPU with Transformer, ~1500 with GNN + cutoff.
- Beyond that, the coupling-flow paradigm is the bottleneck, not the library.

This matters because bgmat's transferability claim is "train at 64, deploy at 512+". At 512 with their GNN this is achievable; at 5000 nothing in the coupling-flow world works, regardless of the library. Users will blame nflojax for a paradigm limitation unless DESIGN.md is clear.

**Amendment candidate.** [SHOULD] Add a "scalability envelope" subsection to DESIGN.md with a concrete N-vs-conditioner-vs-GPU-memory table (order-of-magnitude, not benchmark).

### 12.5 [6mo, medium confidence] `build_particle_flow` will balloon

Every builder in every flow library of the past decade has ballooned. `build_realnvp` currently takes 10+ kwargs; `build_particle_flow` will start with 10 and hit 20 by Stage G:

- `event_shape`, `box`, `num_layers`, `num_bins`, `conditioner`, `boundary_slopes`, `use_com_shift`, `trainable_base`, `tail_bound`, `min_bin_width`, `min_bin_height`, `min_derivative`, `max_derivative`, `identity_gate`, `activation`, `res_scale`, `permutation_strategy`, `init_fn`, ...

Each user of bgmat / DM / a new app will tweak 2–3 of these and copy the rest. The shared code in `build_particle_flow` becomes a knot of conditionals.

**Better alternative.** Don't ship a builder; ship a **canonical recipe** in `USAGE.md` that compiles to ~30 lines of user code. Users modify fearlessly; no one has to maintain `build_particle_flow`. This is TFP-bijectors' approach.

Compromise: keep `build_particle_flow` but restrict it to the 3–4 knobs that every application actually uses (`event_shape`, `box`, `conditioner`, `num_layers`); everything else goes in `conditioner_kwargs` / advanced-assembly mode. Users who want to tweak spline tails drop to `assemble_flow`.

**Amendment candidate.** [SHOULD, before Stage E] Reduce `build_particle_flow` API surface, or replace with a recipe.

### 12.6 [12mo+, high confidence] The `IS NOT` list is a hostage to fortune

Every `"No X"` in DESIGN.md §4 is a pressure point for future PRs:

- "No energies" — someone will want a trivial harmonic-oscillator energy "just for the integration tests".
- "No training loops" — someone will want a 20-line reverse-KL helper "just for the examples".
- "No observables" — the moment a user ships a flow and can't compute ESS, they'll ask.

The concern isn't any one ask; it's that approving each one in isolation drifts the library into a BG framework, which is the scope the user explicitly rejected. **Individual negotiations fail; systemic rules hold.**

**Proposed governance.** Add a paragraph to DESIGN.md §4:

> Exceptions to this list are considered as a *family*, not individually. Admitting one energy admits the question "which energies?", which pulls in training loops, which pulls in observables. If a pressure mounts to admit one excluded item, the maintainers revisit the full exclusion list in one review and either keep all or collapse all.

This is stronger than the current single-item "every target app must need it" rule because it prevents cumulative drift.

**Amendment candidate.** [MUST, minor edit] Add the "revisited as a family" clause to §4 preamble.

### 12.7 [RESEARCH, medium confidence] The `Bijection` contract locks out autoregressive flows

Current contract:
```
forward(params, x, ...) -> (y, log_det)   # O(1) in N
inverse(params, y, ...) -> (x, log_det)   # O(1) in N
```

This is a *coupling-flow* contract. Autoregressive flows (MAF, IAF, bgflows) have:
- **MAF**: O(1) forward (training), O(N) inverse (sampling).
- **IAF**: O(N) forward (training), O(1) inverse (sampling).

For molecular flows on internal coordinates (bonds / angles / torsions), AR is sometimes the natural choice (torsion angle ordering matters). A future nflojax user who wants an AR flow finds the contract doesn't fit cleanly.

**Two options.**

- Accept: "nflojax is coupling-only; AR is another library's problem". Document explicitly.
- Generalise the `Bijection` contract to allow per-direction cost markers: `forward_cost='O(1)' | 'O(N)'`, `inverse_cost='O(1)' | 'O(N)'`. Compositions would then carry the max. This is a contract change that would need to land before the library is widely adopted.

**Recommendation.** Accept the limitation; be explicit in DESIGN.md. The cost of the contract generalisation isn't justified by current users, and if someone really wants AR flows, they can build a parallel library.

**Amendment candidate.** [SHOULD] Add one sentence to DESIGN.md §5.1: "This contract assumes both forward and inverse are O(1) in the event dimension. Autoregressive flows do not satisfy this; they are out of scope."

### 12.8 [NOW, high confidence] The missing `Geometry` abstraction

Currently `box`, `lower`, `upper`, `periodic_mask` are passed around as separate scalar / tuple arguments. `CircularShift(lower, upper)`, `Rescale(lower, upper)`, `LatticeBase.fcc(..., box=...)`, `utils.pbc.nearest_image(dx, box)`, `UniformBox(lower, upper, ...)`. If the user changes the box shape, four constructors need to update.

**Proposal.** Introduce a small `Geometry` dataclass now:

```python
@dataclass
class Geometry:
    lower: Array           # (d,)
    upper: Array           # (d,)
    periodic: Array | None # (d,) of bools; None means all-True

    @property
    def box(self) -> Array: return self.upper - self.lower
    @property
    def d(self) -> int: return self.lower.shape[0]
```

Every primitive that needs geometry takes a `Geometry` instead of separate args. Benefits:

- Adding non-orthogonal (triclinic) later means adding a `cell: Array | None` field; consumers not using it ignore it.
- Adding slab geometry (periodic in some axes) uses the existing `periodic` field.
- User-facing code is cleaner.

**Cost.** ~50 LOC + touches every geometry-consuming primitive. Cheap *now*; expensive after Stage B.

**Amendment candidate.** [MUST, before Stage A] Introduce `Geometry` before writing any new geometry-consuming primitive. Retrofit `CircularShift` in the same commit.

### 12.9 [6mo, high confidence] Multi-device is harder than "pmap-friendly"

DESIGN.md treats distributed training as "infrastructure the caller owns". Correct for the library boundary. But:

- A user doing `pmap(flow.sample)(keys)` gets **data-parallel sampling** cleanly. Fine.
- A user doing `shard_map(flow.sample)` with a flow sharded over the particle axis hits `SplitCoupling` expecting the full particle axis to be local. Not fine; requires `jax.experimental.shard_map` + explicit collective calls in the bijection.
- A user doing `shard_map` over batch with the conditioner distributed across devices hits the `validate_conditioner` contract (which expects one device).

This is not a library bug, but users assuming "pmap-friendly" means "works out of the box on 8 GPUs" will be frustrated. The library should state what it supports:

- Data-parallel over batch: works via `pmap`.
- Model-parallel over particles: does not work; sharded flows require custom bijections.
- Model-parallel over layers: partial (each layer is a `Bijection`, so you could `pmap` over them, but this is rarely the right pattern).

**Amendment candidate.** [SHOULD] Add "Distributed execution" subsection to DESIGN.md §§3 or §7: what works, what doesn't, where the user goes next.

### 12.10 [12mo+, medium confidence] Reproducibility at the library level

A paper whose results depend on a flow trained with nflojax 1.3 needs to be reproducible in nflojax 2.0. This requires:

- **Pinned deps**: `pyproject.toml` with `jax == X.Y.Z` or narrow ranges. Currently loose.
- **Checkpoint format spec**: which PyTree paths hold which params. Today implicit.
- **Conditioner architecture spec**: "the `Transformer` in version 1.X has pre-norm, no rotary embeddings, `num_heads = X`, ..." Today undocumented.
- **Test fixtures with frozen outputs**: `tests/fixtures/` with `(seed, params, expected_log_prob)` tuples. Today absent.

Without these, any library refactor invalidates every trained checkpoint. With them, the maintenance burden of "can we reproduce Schebek 2025 with nflojax 3.0?" is finite.

**Amendment candidate.** [SHOULD, ≤12mo] Add Stage H to PLAN.md: reproducibility infrastructure. `CITATION.cff`, a `REPRODUCIBILITY.md` spec, pinned test-fixture goldens. Roughly 2 days of work; value grows with age.

### 12.11 [NOW, high confidence] Compositional correctness testing is missing

Per-primitive tests pass; a `CompositeTransform` built from an adversarial (but valid) sequence may still fail. Current coverage doesn't exercise:

- `Rescale` followed by `SplitCoupling` with non-default event shape.
- Two `CircularShift`s in a row (log-det accumulator stays scalar?).
- `ShiftCenterOfMass` combined with `SplitCoupling` (projection subspace interacts with partition).
- Any permutation of the above across event axes.

A **composition fuzz test** that enumerates small valid stacks and asserts round-trip + log-det-vs-autodiff at each would catch the integration bugs that unit tests miss. ~50 LOC of fixture logic.

**Amendment candidate.** [MUST, add to cross-cutting checklist] Introduce `tests/test_composition_fuzz.py` before any new multi-layer bijection ships.

### 12.12 [6mo, medium confidence] Approximation-quality regression tests on analytical targets

Current tests prove *correctness*: `forward(inverse(x)) == x`. They do not prove *quality*: can this flow actually learn the target? For reference conditioners this matters — if a Transformer "passes tests" but can't fit a 2D banana distribution in 10k steps, it's broken in a way unit tests miss.

**Proposal.** Add a small `tests/test_flow_quality_analytical.py` suite:

- 2D banana distribution; train 1k steps; assert loss < threshold.
- 4D double-moon; train 1k steps; assert loss < threshold.
- 8D Gaussian mixture; train 2k steps; assert ESS > threshold against importance weights.

These tests have no physics, no energy in the nflojax sense — they are mathematical target densities. They tell us reference conditioners are actually usable. They take ~30 seconds on a CPU.

**Scope tension.** DESIGN.md §4 says no training. A test that trains for 1k steps uses `optax` transitively. Options:

- Keep the tests in an *optional* test file behind a skip marker that runs only in CI, never counted against the "no training" rule — tests are infrastructure, not library code.
- Or: add an `examples/` directory that contains these as example scripts, not tests.

**Amendment candidate.** [SHOULD] Add the quality tests; mark them explicitly as infrastructure tests, not library code. This is not a scope violation.

### 12.13 [12mo+, low confidence] "Particle system" is narrower than "normalizing flows for condensed matter"

The framing is `(N, d)` = particles in a box. This works for atoms, beads, coarse-grained macromolecules. It does not work natively for:

- Molecules with bonded topology (internal coords).
- Proteins with sequence order.
- Liquid crystals (need orientation per molecule).
- Fields / continuous media (infinite-dim).

The library isn't obliged to cover these, but the name "nflojax" + the "particle systems" framing in DESIGN.md §6 will attract users who expect these use cases. Either expand framing or name the exclusion.

**Amendment candidate.** [NIT] DESIGN.md §6 add one sentence: "nflojax supports particle configurations: N points in d dimensions. Systems with intrinsic internal structure (molecules with torsions, polymers with sequence, fields) are out of scope; consider [X libraries]."

### 12.14 [6mo, high confidence] License and citation infrastructure

DESIGN.md doesn't mention:

- License (MIT per `LICENSE` file — fine, but not stated in DESIGN.md).
- How to cite nflojax (`CITATION.cff` file — absent).
- Contribution policy (who merges, who reviews — absent).

Academic users of bgmat that train flows with nflojax need to know how to cite. Missing `CITATION.cff` is a 10-line fix that pays off for years.

**Amendment candidate.** [SHOULD] Add `CITATION.cff` and one paragraph to README.md on "how to cite".

### 12.15 [6mo, high confidence] Gradient checkpointing and flow-internal training concerns

The library sits at the boundary between "flow primitives" and "training". Three concerns live in the no-man's land:

- **Gradient checkpointing.** At 24 layers × N=500 × Transformer conditioner, backward-pass memory explodes. `jax.checkpoint` must be applied inside `CompositeTransform.forward`, not by the user. Who owns the decision?
- **Gradient clipping per layer.** Total-norm clipping is the user's job. Per-layer clipping requires the user to know the param-tree structure — nflojax's, not their own. A convenience `flow.per_layer_param_tree()` helper makes this feasible without owning training.
- **Stop-gradient for identity gates during warmup.** bgmat may want to freeze `g_value` at 0 for the first N steps. Currently doable but awkward; a `with gate_frozen(flow): ...` context would be clean, though it is close to a training helper.

**Call.** For gradient checkpointing specifically: **expose a `use_checkpoint: bool` kwarg on `CompositeTransform`**. It's one line; it saves users serious memory; it stays within the "flow primitive" scope because it's an execution hint, not a loss.

For the other two: document as user's responsibility with helper utilities (`flow.per_layer_param_tree()`) but no training hooks.

**Amendment candidate.** [SHOULD] Add `use_checkpoint` to `CompositeTransform`; add `per_layer_param_tree` helper; document.

### 12.16 [6mo, low confidence] User learning curve and onboarding

Every user coming from PyTorch flow libraries (`nflows`, `normflows`, `FlowTorch`) will struggle with:

- JAX's PRNG-key discipline.
- Explicit params vs. hidden state.
- Flax module patterns.

nflojax is not a JAX 101; USAGE.md can't teach it. But a brief "coming from PyTorch flows" paragraph + a single worked example that juxtaposes JAX and PyTorch idioms would save users their first day. A `USAGE.md` section "from PyTorch to nflojax" is ~½ page, high-value.

**Amendment candidate.** [NIT] Add USAGE.md section "Coming from PyTorch flows".

### 12.17 [RESEARCH, medium confidence] Flow matching / score-based via the same library?

If the paradigm shifts to flow matching, two options:

- (A) nflojax stays coupling-only; a sibling `flowmatch_jax` is born.
- (B) nflojax grows to cover flow matching, meaning the `Bijection` contract is joined by a `VectorField` / `ODE` contract, loss helpers appear, the library doubles in scope.

(A) is consistent with the current philosophy. (B) is the path of least resistance once the first user asks. Preempt: state in DESIGN.md that nflojax is coupling-only and that flow-matching / diffusion live elsewhere.

**Amendment candidate.** [SHOULD] Add explicit "non-goal: flow matching / diffusion / CNFs" to DESIGN.md §4 list.

### 12.18 [NOW, high confidence] PLAN.md missing explicit "revisit and stop" cadence

The plan marches forward: Stage A → G. It does not check in with itself. In 6 months when three stages are done, it will be unclear whether to keep executing or revisit the design. Good practice:

- At end of each stage, a 1-page retrospective in `DESIGN.md §14` (review log).
- After Stage E, a design review of the whole library against the audit.
- After Stage G (bgmat prototype), a decision on whether to release v1.0.

**Amendment candidate.** [SHOULD] Add retrospective checkpoints to PLAN.md.

### 12.19 [12mo+, medium confidence] Testing boundary for "flow primitives that act on particles"

The library sits in an awkward test boundary: it claims no physics, but the primitives are particle-specific. Concrete consequence:

- We cannot test `CircularShift` on "real" PBC data.
- We cannot test `LatticeBase.fcc` against "real" FCC samples with meaningful tolerance.
- Tests degenerate to shape-and-round-trip.

To tighten this, introduce **synthetic particle benchmarks**:

- A known analytic density on a particle-shaped space (e.g., independent Gaussians per particle, or a pairwise-harmonic density with known normalizer).
- Flows trained against it recover known log_Z and known samples.

This is the "test approximation quality" recommendation of §12.12 applied specifically to particle topologies. Add to that test suite.

### 12.20 [RESEARCH, high confidence] The augmented-coupling decision will bite

First-pass audit (§4 item 9) proposed ship `AugmentedBase + AugmentedFlow` as a thin primitive. If DESIGN.md keeps it as just a pattern, here is what will happen in 6–12 months:

- bgmat's augmented-coupling impl is in bgmat. Fine.
- A new app (say, chemistry flows with auxiliary variables) implements it separately.
- A third app implements it a third way.
- Two of the three have subtle bugs in log-det attribution.
- The user reading the three codebases has to reverse-engineer which one is correct.

Keeping it out of nflojax saves ~40 LOC of library code at the cost of >500 LOC of duplicated app code and several silent bugs. Net: ship the primitive.

**Amendment candidate.** [MUST] Reverse the first-pass's soft-decline on augmented coupling. Ship `AugmentedBase` + `AugmentedFlow` wrapper as ~40-line primitives with explicit log-det accounting. Upgrade first-pass amendment 3 from "decide" to "ship".

---

## 12-prime. Priority summary (second-pass amendments)

Ranked by when-to-act, then confidence.

### [MUST, before Stage A]

- **§12.2** — Decide conditioner backend: `linen` / dual-backend / pure-JAX. Before any new conditioner is written.
- **§12.6** — Add "revisit as a family" clause to DESIGN.md §4.
- **§12.8** — Introduce `Geometry` dataclass before any new geometry-consuming primitive.
- **§12.11** — Add compositional-fuzz test.
- **§12.20** — Ship `AugmentedBase` + `AugmentedFlow` as primitives, reversing the first-pass soft-decline.

### [SHOULD, before Stage D]

- **§12.1** — DESIGN.md §1 paragraph on paradigm choice.
- **§12.3** — Reduce reference-conditioner scope; ship `DeepSets` alone, defer `Transformer` + `GNN` to v2.
- **§12.4** — Scalability envelope in DESIGN.md.
- **§12.5** — Reduce `build_particle_flow` surface or replace with recipe.
- **§12.12** — Analytical-target quality tests.
- **§12.14** — `CITATION.cff` + README cite paragraph.
- **§12.15** — `use_checkpoint` kwarg on `CompositeTransform`.
- **§12.17** — Flow-matching / diffusion explicitly named as non-goals in DESIGN.md §4.

### [SHOULD, ≤12mo]

- **§12.9** — Distributed-execution subsection in DESIGN.md.
- **§12.10** — Stage H: reproducibility infrastructure.
- **§12.18** — Retrospective checkpoints in PLAN.md.
- **§12.19** — Synthetic particle benchmarks.

### [NIT]

- **§12.13** — "What's out of scope" sentence in DESIGN.md §6 for molecules / polymers / fields.
- **§12.16** — "Coming from PyTorch flows" section in USAGE.md.
- **§12.7** — One-sentence note in DESIGN.md §5.1 that AR flows are out of scope.

---

## 13. Revised bottom line (after pass 1 + 2)

Pass 1 said: clean up four surfaces (conditioner contract, CoM, augmented coupling, solid/liquid), re-size PLAN.md stages, do the bgmat dry-run early.

Pass 2 adds:

1. **Five pre-Stage-A must-dos**: introduce `Geometry`, decide conditioner backend for the Flax transition, add the "revisit as family" clause, ship augmented-coupling primitives, introduce compositional fuzz testing.
2. **Reduce reference-conditioner scope**. Ship `DeepSets` alone in v1; defer `Transformer` and `GNN` to v2 (or "bring your own"). This is a bigger scope change than pass-1 recommended.
3. **State explicitly what nflojax is not**: not flow-matching / diffusion / CNFs; not autoregressive flows; not multi-device for model-parallel; not general-purpose normalizing flows library; coupling-flow-only for N ≲ 500 (Transformer) / ≲ 1500 (GNN).
4. **Add reproducibility and governance infrastructure** early (CITATION.cff, pinned deps, family-revisit clause for §4).

**Total incremental work over pass 1**: ~2 more days of doc edits + ~100 LOC of `Geometry` refactor + ~50 LOC of augmented-coupling primitive. No change to the 2-day pre-code budget; the shape of it changes.

Net: the library becomes narrower, more defensible, and more maintainable. It takes on less future-maintenance debt and positions honestly against the paradigm shift happening in the surrounding field.

---

*Audit drafted 2026-04-21 against DESIGN.md + PLAN.md as of the same date, in the session that also landed `feature/particle-events` (commit 86be470). First pass §§0–8; symmetries and missed topics in §§9–11; second pass (five steps ahead) in §§12–13.*

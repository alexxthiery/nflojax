# nflojax — Background

## What this library is for

nflojax is a JAX normalizing-flow framework aimed at sampling equilibrium configurations of many-body systems in materials and molecular science. It is the flow-side half of a **Boltzmann generator**; the energy, training loop, and observables are application code.

This document is for a contributor who knows machine learning but not the condensed-matter / Boltzmann-generator domain. It is meta-level — it explains *why* the library exists and what vocabulary to expect. Design and plan documents (DESIGN.md, PLAN.md) are downstream of this.

## The problem

Given a many-body energy `E(x)` over particle configurations `x ∈ R^{N×d}`, statistical mechanics says the equilibrium distribution at temperature `T` is

```
p(x) ∝ exp(-β E(x)),   β = 1/(k_B T).
```

Computing expectations under `p` — free energies, phase diagrams, radial distribution functions, response properties — is the central computational task in materials science. Traditional methods (molecular dynamics, Markov-chain Monte Carlo) produce *correlated* samples and struggle with multi-modal targets (two competing phases), rare events (nucleation, reaction barriers), and slow dynamics (glasses, low-temperature solids). A generative model that samples `p` *directly* sidesteps all of these.

## Boltzmann generators

The Boltzmann-generator idea (Noé et al. 2019; expanded by the *Flows for Atomic Solids* lineage) is:

- Train a normalizing flow `q_θ(x)` to approximate `p(x) = exp(-β E(x)) / Z`.
- Typically via **reverse KL**:
  ```
  L(θ) = E_{x ~ q_θ} [ log q_θ(x) - log p(x) ]
       = E_{x ~ q_θ} [ β E(x) + log q_θ(x) ] + const.
  ```
- Once trained, `q_θ` produces i.i.d. samples in one forward pass.
- Importance weights `w(x) = exp(-β E(x)) / q_θ(x)` give **self-normalised importance sampling (SNIS)** estimates of expectations under `p`, and of `log Z`.
- The quality metric is the **effective sample size** `ESS = (Σ w_i)^2 / (N · Σ w_i^2)` — how close `q` is to `p`.

Strengths: decorrelated samples, native log-density in both directions, tractable importance weighting.
Weaknesses: training is mode-seeking (reverse KL), expensive in high dimensions, paradigm is being challenged by flow matching and diffusion.

## Why *coupling* flows specifically

nflojax is a coupling-flow library. The `Bijection` contract assumes O(1) cost in *both* `forward` and `inverse`. Coupling flows (RealNVP, spline couplings) meet this: a single forward pass gives both samples and log-likelihood; a single inverse pass gives log-likelihood for arbitrary inputs. This is exactly what reverse-KL training on `E(x)` needs.

Alternatives exist and are specifically *out of scope* for nflojax:
- **Autoregressive flows** (MAF, IAF): O(1) one direction, O(N) the other. Different library.
- **Continuous-time flows / CNFs** (FFJORD): log-det via ODE solve; different contract.
- **Flow matching / diffusion**: cheaper training, more expensive log-prob. Different paradigm; sibling library, not this one.

This choice is documented in `DESIGN.md` §4.

## Particle systems, as nflojax sees them

An event is `(N, d)` — `N` particles in `d` spatial dimensions (`d = 3` in practice). nflojax is agnostic to *what* the particles are (atoms, coarse-grained beads, residues); it knows their geometry and symmetries.

### Core symmetries

The equilibrium density satisfies several invariances that a good flow should respect:

- **Permutation `S_N`** — classical particles are indistinguishable: `p(x_1, ..., x_N) = p(x_{π(1)}, ..., x_{π(N)})`. Enforced via a combination of (a) permutation-invariant base distributions (random permutation at sample time) and (b) permutation-equivariant conditioners. Both halves must hold; see `audit.md` §9.2 for the two-bit table.
- **Translation `T(d)`** — for periodic boundary conditions, shifting every particle by the same vector leaves `p` unchanged. Handled by CoM projection, augmented-coupling CoM swap, or explicit uniform-shift sampling. See `audit.md` §9.3.
- **Rotation `O(d)` / `SO(d)`** — for isotropic fluids, `p` is rotation-invariant. For crystals it is *not* (the lattice pins an orientation). nflojax does not enforce rotation equivariance — see `DESIGN.md` §4 item 7 for the reasoning.

### Boundary conditions

- **Periodic (torus)** — most materials simulations use periodic boundary conditions; the configuration space is `([lower, upper])^d / ~`. Handled by `CircularShift` + `boundary_slopes='circular'` splines (both already in nflojax) plus per-axis `Geometry.periodic` flags.
- **Open** — no periodicity; particles can in principle escape. Handled by `LoftTransform` for numerical stability.
- **Slab** — periodic in some axes, open in others (e.g., thin films). Captured by `Geometry(..., periodic=[True, True, False])`.

### Phases

- **Crystalline solid** — atoms oscillate around discrete lattice sites. The right base distribution is a `LatticeBase` (Gaussian-perturbed lattice positions, optionally randomly permuted). The flow only has to learn the correlated thermal displacements around that base. Example: DeepMind's LJ flows, bgmat's ice flows.
- **Liquid** — delocalised density; no lattice. The right base is `UniformBox` on the simulation cell. The flow has to learn everything. Harder than solids; not demonstrated at competitive scale yet for coupling flows.
- **Glass / supercooled** — formally a liquid (no crystalline order) but with broken ergodicity on MD timescales. A well-trained Boltzmann generator can sample configurations MD wouldn't reach; this is a feature, not a bug, but requires care in validation.
- **Molecular** — internal coordinates (bonds, angles, torsions); E(3)-invariant by construction. Requires non-Cartesian bijections not shipped in nflojax today.

## Downstream applications

In priority order:

1. **DeepMind *Flows for Atomic Solids*** (Wirnsberger et al., 2022). Lennard-Jones and monatomic-water crystals, 32–500 atoms. Transformer conditioner with circular Fourier embeddings. Reverse-KL training. Canonical reference for the crystalline Boltzmann-generator pattern. `../flows_for_atomic_solids/`.

2. **bgmat** (Schebek, Noé, Rogal 2025). The same systems plus silicon, scaled to N > 1000. Swaps the DM Transformer for a local MPNN conditioner. Introduces **augmented coupling flows** (duplicate the degrees of freedom with Gaussian auxiliaries, train a flow on the augmented state) and a **`ShiftCenterOfMass`** bijector. Claims transferability (train at N=64, deploy at N > 512). `../bgmat/`.

3. **Future** — molecular flows (requires internal coords + E(3) equivariance), multi-species solids, disordered phases. Not shipped, not planned in the immediate roadmap; each one has a parking-lot entry in `PLAN.md` §9.

## What success looks like

nflojax is successful when a user can implement the DM paper, bgmat, or a new particle-system Boltzmann generator by writing only application-side code: the energy, the training loop, the observables, and any research-specific conditioner architecture. The flow machinery (bijections, bases, assembly, conditioner contract) comes from nflojax. Target: < 500 lines of app-side code to reproduce a published paper's training pipeline, excluding the conditioner.

A failure mode we are actively avoiding: nflojax becomes a Boltzmann-generator framework. The moment it ships an energy, a loss, a training loop, or an observable, it enters the scope war of "which LJ, which SNIS, which scheduler". `DESIGN.md` §4 is the wall between the two.

## Vocabulary a new contributor will meet

- **β (beta)** — inverse temperature `1 / (k_B T)`. Appears in reverse-KL loss as the prefactor on `E(x)`.
- **Reverse KL** — training objective `E_{x ~ q}[β E(x) + log q(x)]`. Used when you have the energy but no samples from the target.
- **Forward KL** — `E_{x ~ p}[-log q(x)]`. Used when you have samples from `p` but no energy.
- **SNIS** (self-normalised importance sampling) — estimator for `E_p[f(x)]` using weighted samples from `q`.
- **ESS** — effective sample size; `(Σ w_i)^2 / (N · Σ w_i^2)`. Between 0 and 1; a measure of proposal quality.
- **log Z** — log-partition function; estimated via SNIS as `log mean_i w_i`.
- **RDF** — radial distribution function; pairwise-distance histogram characterising a liquid or solid's local structure.
- **Lattice base** — a base distribution on `(N, d)` configurations that puts each particle near a lattice site with a small Gaussian perturbation. Optional random permutation restores `S_N` invariance.
- **Identity gate** — a scalar context-dependent multiplier on a flow layer: gate=0 makes the layer an identity (for training stabilisation / boundary conditions), gate=1 makes it a full learned transform.
- **Circular spline** — rational-quadratic spline with matching boundary slopes at both ends of `[-B, B]`. Makes the flow C¹ on the torus; pairs with `CircularShift`.
- **Augmented coupling flow** — a flow architecture where the state is doubled with auxiliary Gaussian degrees of freedom; the inner flow acts on the augmented state and the auxiliary axes are marginalised at inference. Used in bgmat.
- **CoM projection** — removing the translation-invariant degree of freedom by projecting `(N, d)` onto the `(N-1)·d`-dimensional subspace of zero-centre-of-mass configurations.
- **Boundary slopes (circular vs linear_tails)** — two modes for RQ splines. `linear_tails` pins both boundary derivatives to 1 (identity outside the box). `circular` ties them to one shared learnable value (C¹ wraparound). See `nflojax/splines.py`.

## Papers to read (in priority order for this codebase)

1. **Wirnsberger, P. et al., 2022.** *Normalizing Flows for Atomic Solids.* *Machine Learning: Science and Technology* 3(2), 025009. DOI `10.1088/2632-2153/ac6b16`. The canonical crystalline Boltzmann-generator reference; the architecture nflojax must be able to reproduce.
2. **Schebek, M., Noé, F., Rogal, J., 2025.** *Scalable Boltzmann Generators for equilibrium sampling of large-scale materials.* arXiv:2509.25486. Extends #1 with MPNN + augmented coupling; this is the bgmat paper.
3. **Noé, F., Olsson, S., Köhler, J., Wu, H., 2019.** *Boltzmann Generators: Sampling equilibrium states of many-body systems with deep learning.* *Science* 365, eaaw1147. The field's founding paper; establishes the reverse-KL + flow recipe.
4. **Durkan, C. et al., 2019.** *Neural Spline Flows.* NeurIPS. Rational-quadratic splines; the spline primitive nflojax ships.
5. **Dinh, L., Sohl-Dickstein, J., Bengio, S., 2017.** *Density estimation using Real-NVP.* ICLR. Coupling flows; the affine primitive nflojax ships.

For the flow machinery itself, #4 and #5 are enough. For the domain, read #1 first; read #3 for history; read #2 for the most recent state of the art on scalability.

## When to read this document vs. the others

- **BACKGROUND.md (here)** — you do not know what a Boltzmann generator is, or you want the scientific motivation / vocabulary.
- **DESIGN.md** — you want to know what nflojax *does* (and refuses to do) as a library, and why.
- **PLAN.md** — you want to know what's next and what's already landed.
- **audit.md** — you want the long argument behind the design.
- **USAGE.md / REFERENCE.md** — you want to use or extend the API.
- **INTERNALS.md / EXTENDING.md** — you want to understand or add mathematical machinery.

---

*Version: 2026-04-21. Update when the scientific frame shifts — new phases in scope, new paradigm risks, new downstream apps.*

# Internals

Mathematical foundations and design decisions behind nflojax.

**Contents:**

- [Change of Variables](#change-of-variables)
- [Coupling Layers](#coupling-layers)
- [Structured (Rank-N) Events](#structured-rank-n-events)
- [Composing Transformations](#composing-transformations)
- [Conditional Normalizing Flows](#conditional-normalizing-flows)
- [Identity Gate](#identity-gate)
- [Numerical Stability](#numerical-stability)
- [Density Evaluation](#density-evaluation)
- [CoM Projection and the Volume Correction](#com-projection-and-the-volume-correction)
- [References](#references)

## Change of Variables

Let `z ~ p_0(z)` be a sample from the base distribution (typically standard Gaussian), and let `f: R^d -> R^d` be an invertible transformation.
The transformed variable `x = f(z)` has density:

```
p(x) = p_0(f_inv(x)) * |det df_inv/dx|
```

Taking logarithms:

```
log p(x) = log p_0(z) + log|det dz/dx|
```

where `z = f_inv(x)`.

The computational bottleneck is computing `det df/dz`, which is `O(d^3)` for general transformations.
Modern normalizing flows design transformations with tractable (cheaper-than-cubic) Jacobians.

## Coupling Layers

### Affine Coupling (RealNVP)

The affine coupling layer splits the input into two parts using a binary mask `m in {0,1}^d`.

**Forward** (`z -> x`):

```
x_masked       = z * m
x_transformed  = z * (1-m) * exp(s(z * m)) + t(z * m)
x              = x_masked + x_transformed
```

where `s, t: R^d -> R^d` are scale and shift functions (conditioner MLP outputs).

**Jacobian**: triangular, so its determinant is the product of diagonal elements:

```
log|det dx/dz| = sum_{i: m_i=0} s_i(z * m)
```

**Inverse** (`x -> z`):

```
z_masked       = x * m
z_transformed  = (x * (1-m) - t(x * m)) * exp(-s(x * m))
z              = z_masked + z_transformed
```

The inverse is computed analytically without iterative methods.

### Spline Coupling

Instead of affine transformations, spline coupling uses monotonic rational-quadratic splines [durkan2019].

For each dimension `i` where `m_i = 0`:

```
x_i = RQSpline(z_i; theta_i)
```

where `theta_i = (w_i, h_i, d_i)` are spline parameters (bin widths, heights, knot derivatives) predicted by the conditioner.

**Properties**:
- Monotonic and invertible by construction
- C1 continuous
- More expressive than affine (can model multimodal conditionals)
- Identity outside `[-B, B]` (linear tails)

**Parameterization** (per dimension, `K` bins):
- Widths: `K` parameters -> softmax -> min\_width floor -> scale to `2B`
- Heights: `K` parameters -> softmax -> min\_height floor -> scale to `2B`
- Derivatives: `K-1` internal derivatives (boundary derivatives fixed to 1)

Internal knot derivatives are bounded via sigmoid:

```
d = d_min + (d_max - d_min) * sigmoid(d_raw)
```

where `d_min` and `d_max` are hyperparameters (defaults: 0.01 and 10.0).

Total: `3K - 1` parameters per transformed dimension.

## Structured (Rank-N) Events

Many physical systems (point clouds, particles in a box, sets of tokens)
have events with rank > 1. The natural shape is `(*batch, N, d)` rather
than a flattened `(*batch, N*d)`.

### Event shape convention

The "event" is the trailing sub-tensor on which the density is defined; the
"batch" consists of leading independent samples. We represent the event by
a tuple `event_shape` (rank 1 is the flat special case). For `x` of shape
`(*batch, *event_shape)`:

```
p(x) is defined on R^{prod(event_shape)}
log_prob(x) returns shape `batch`
```

Base distributions (`StandardNormal`, `DiagNormal`) store `event_shape` as
a tuple and sum their log-probability over the last `len(event_shape)`
axes. For `event_shape = (N, d)`:

```
log p(x) = -0.5 * sum_{n,i} x_{n,i}^2  -  0.5 * N * d * log(2*pi)
```

Rank-1 and rank-N use the same code paths: the rank-1 case is just the
special case `event_shape = (dim,)`.

### Structured coupling (SplitCoupling)

The mask-based coupling layers (`AffineCoupling`, `SplineCoupling`) split
dimensions via a binary mask `m in {0,1}^d`. That only makes sense when
the event is a flat vector. For a rank-N event, a natural alternative is
to split along one of the event axes:

```
x: (*batch, N, d)
x_frozen      = x[:, :split_index, :]    # first half of particles
x_transformed = x[:, split_index:, :]    # second half of particles
```

The frozen slice conditions a network that produces per-scalar spline
parameters for every element of the transformed slice, and the spline acts
elementwise:

```
theta = MLP(flatten(x_frozen), context)                  # (*batch, M * (3K-1))
theta = reshape(theta, (*batch, N//2, d, 3K-1))          # per-scalar params
y_transformed = RQSpline(x_transformed; theta)           # elementwise
y = concat(x_frozen, y_transformed, axis=-2)
```

### Log-determinant

A rank-N coupling that acts elementwise on the transformed slice has a
triangular Jacobian in the flattened representation: frozen dimensions are
identity, transformed dimensions depend only on the frozen ones via the
conditioner and on themselves through the elementwise spline. The overall
log-determinant therefore reduces to the sum of per-scalar spline
log-derivatives:

```
log|det dy/dx| = sum_{n, i} log|RQSpline'(x_{n,i}; theta_{n,i})|
             over (n, i) in the transformed slice
```

In code this is `jnp.sum(logabsdet_per_scalar, axis=tuple(range(-event_ndims, 0)))`.

### Alternating coverage

A single `SplitCoupling` only transforms one half of the split axis. To
cover every slot, alternate `swap` between layers (layer 0 freezes slots
`[0, split_index)`, layer 1 freezes slots `[split_index, axis_size)`, etc.).
`SplitCoupling` does not expose a `mask`, so the library's
`analyze_mask_coverage` check does not apply — responsibility for coverage
is on the builder.

### Static shape

`SplitCoupling` stores `event_shape`, `split_axis`, `split_index`, and
`event_ndims` as dataclass fields. The conditioner MLP is sized to the
corresponding frozen / transformed flat sizes. Changing the event shape
requires rebuilding the coupling. For size-transferable flows (e.g. evaluating
weights trained at N=216 on N=512), supply a size-agnostic conditioner
(GNN over particle neighborhoods); the default MLP is specific to its
`x_dim`.

## Composing Transformations

### Composite Transform

Multiple transforms compose sequentially:

```
f = f_n . f_{n-1} . ... . f_1
```

The log-determinant accumulates:

```
log|det df/dz| = sum_{i=1}^{n} log|det df_i/dz_{i-1}|
```

### Mask Alternation

A single coupling layer only transforms dimensions where `m_i = 0`.
To ensure all dimensions are transformed, masks alternate:

- Layer 1: mask = `[1, 0, 1, 0, ...]` (parity=0)
- Layer 2: mask = `[0, 1, 0, 1, ...]` (parity=1)
- Layer 3: parity=0 again, etc.

The builder validates mask coverage at construction time and raises an error if any original dimension is never transformed (accounting for permutations).

### Permutations

Fixed permutations between coupling layers improve mixing.

For a permutation `pi`, the forward transform is `y_i = x_{pi(i)}`.
The Jacobian is a permutation matrix with determinant +/-1, so `log|det| = 0`.

This library uses a reverse permutation `pi(i) = d - 1 - i` between coupling layers when `use_permutation=True`.

### Linear Transform

Optional global linear transformation with LU-style parameterization.

Factor an invertible matrix `W in R^{d x d}` as:

```
W = L * T
```

where:
- `L = tril(L_raw, k=-1) + I`, unit-diagonal lower triangular
- `T = triu(U_raw, k=1) + diag(exp(log_s))`, upper triangular with positive diagonal

**Parameters**: `lower (d, d)`, `upper (d, d)`, `log_diag (d,)`.

**Complexity**: forward/inverse `O(d^2)` via triangular solves; log-determinant `O(d)`:

```
log|det W| = sum_{i=1}^{d} log_diag_i
```

Initialized to identity (`L = I`, `T = I`) via zero initialization.

## Conditional Normalizing Flows

Conditional flows model `p(x | c)` where `c` is a conditioning variable.

### Formulation

The base distribution remains unconditional: `z ~ p_0(z)`.
The transformation becomes context-dependent: `x = f(z; c)`.
For fixed `c`, the map `f(.; c)` must be invertible in `z`.

Conditional density:

```
log p(x | c) = log p_0(z) + log|det dz/dx|
```

where `z = f_inv(x; c)`.
The Jacobian is taken w.r.t. `x` (or `z`), not `c`.
The conditioning variable affects transformation parameters but does not participate in the change of variables.

### Concatenation Strategy

Context is concatenated to the conditioner input:

```
[s, t] = MLP([z_masked, c])
```

The MLP input dimension becomes `dim + context_dim`.
Simple to implement and works well for low-dimensional context, but context influence may be diluted in deep networks.

### Context Broadcasting

Context can be per-sample `(B, context_dim)` or shared `(context_dim,)`.
Shared context is broadcast to match batch dimensions.

## Identity Gate

The identity gate enables smooth interpolation between the identity transform and the learned transform based on context.

### Mechanics

When `identity_gate` is provided, each coupling layer scales its conditioner output (the raw parameters that define the transform) by the gate value before computing the transformation:

- `gate = 0`: conditioner output zeroed -> transform is identity, `log_det = 0`
- `gate = 1`: conditioner output unchanged -> transform acts normally
- `0 < gate < 1`: smooth interpolation

For affine coupling, this means `s -> g * s` and `t -> g * t`.
For spline coupling, the raw spline parameters are scaled by `g`, pulling the spline toward identity.

`LinearTransform` also supports gating via component-wise LU interpolation (see [REFERENCE.md](REFERENCE.md#lineartransform) for the exact formula).
In conditional mode, the conditioner MLP outputs both a diagonal scaling delta and a shift; both vanish at `g=0`.

### Raw Context vs Extracted Features

The gate function always receives raw context, even when a feature extractor is used.
Coupling layers see extracted features, but the gate does not.

This is by design: the gate encodes known structure (e.g., boundary conditions at specific parameter values), so it operates on interpretable inputs rather than a learned representation that changes during training.

### Constraints

- Requires `context_dim > 0` (gate operates on context)
- Incompatible with `use_permutation=True` (permutations cannot be smoothly gated)
- `LoftTransform` supports gating via `y = (1-g)*x + g*loft(x)` with Newton inverse

### Single-Sample Contract

The gate function must be written for a single sample with shape `(context_dim,)`.
Batching is handled internally via `jax.vmap`.
A gate function that expects batched input `(batch, context_dim)` will produce silently wrong results.

The library validates this at build time using `jax.eval_shape` (zero FLOPs).

## Numerical Stability

### Zero Initialization

Conditioner networks are initialized with zero final-layer weights and biases:

```
W_out = 0,  b_out = 0
```

This means:
- Affine coupling: `s = 0, t = 0` -> identity transform
- Spline coupling: uniform bins, unit derivatives -> near-identity

The flow starts as identity, avoiding extreme initial transformations.

### Bounded Log-Scale

For affine coupling, the log-scale is bounded:

```
s = s_max * tanh(s_raw / s_max)
```

This prevents `exp(s)` from exploding or vanishing.

### LOFT Transform

The LOFT (Log-Soft) transform stabilizes high-dimensional flows by modifying the tails:

```
LOFT(x; tau) = x                                            if |x| <= tau
             = sign(x) * (tau + log(|x| - tau + 1))         if |x| > tau
```

Transitions from linear (near origin) to logarithmic (in tails), preventing extreme values from causing numerical issues [andrade2021].

## Density Evaluation

### log_prob

Uses the inverse map and change of variables:

```python
z, log_det_inv = inverse(params, x, context)
log_prob = base_dist.log_prob(z) + log_det_inv
```

### sample_and_log_prob

Avoids redundant inverse computation when both samples and densities are needed:

```python
z = base_dist.sample(key, shape)
x, log_det_fwd = forward(params, z, context)
log_prob = base_dist.log_prob(z) - log_det_fwd
```

Note the sign difference: `log_prob` uses `+ log_det_inv` while `sample_and_log_prob` uses `- log_det_fwd`.
Both are correct because `log|det dz/dx| = -log|det dx/dz|`.

See [REFERENCE.md](REFERENCE.md#forwardinverse-convention) for the full sign convention table.

## CoM Projection and the Volume Correction

`CoMProjection` is a bijection between spaces of different intrinsic embedding
dimension: `(N, d)` ambient (as a subspace of `R^(Nd)`) and `(N−1, d)` reduced
Euclidean space. Because the domains differ, "log-det of the Jacobian" is not
a square-matrix determinant — it is a volume-form ratio that depends on how
the embedding is parameterised. This section derives the `(d/2) · log(N)`
factor and explains why nflojax stores a zero log-det on `CoMProjection` and
leaves the factor to a caller-applied correction.

### Setup

Let `x = (x_1, ..., x_N) ∈ R^(Nd)` with each `x_i ∈ R^d`. Total ambient
dimension is `Nd`. The zero-CoM subspace

```
S = { x ∈ R^(Nd) : Σ_i x_i = 0 }
```

is cut out by `d` linear constraints (one per coord axis), so `dim(S) = (N−1)d`.

Parameterise `S` by dropping the last particle and reconstructing it from the
constraint:

```
y ∈ R^((N−1)d),   y = (x_1, ..., x_{N−1})
x_N = −Σ_{i<N} y_i
```

This `ψ: R^((N−1)d) → S ⊂ R^(Nd)` is a linear bijection between two
`(N−1)d`-dimensional spaces.

### The Jacobian

Consider one coord axis at a time (the `d` axes decouple: `ψ` acts as the
same map independently on each axis). Fix a coord axis; the per-axis
embedding `ψ_1: R^(N−1) → R^N` is

```
ψ_1(y_1, ..., y_{N−1}) = (y_1, ..., y_{N−1}, −Σ y_i).
```

Its Jacobian matrix `J ∈ R^(N × (N−1))` is

```
J_{ij} = δ_{ij}    for i < N,
J_{Nj} = −1        for all j.
```

Since `J` is rectangular (`N > N−1`), the "Jacobian determinant" is replaced
by the volume-scaling factor `sqrt(det(J^T J))` (the Gram determinant). We
compute `J^T J ∈ R^((N−1) × (N−1))`:

```
(J^T J)_{jk} = Σ_i J_{ij} J_{ik}
             = δ_{jk} · (contributions from rows i < N, which equal δ_{jk})
             + 1 · 1  (contribution from row N: (−1)(−1) = 1)
             = δ_{jk} + 1.
```

So `J^T J = I_{N−1} + 11^T`, where `1 ∈ R^(N−1)` is the all-ones column
vector.

### Determinant via the matrix determinant lemma

For a rank-1 update, `det(A + uv^T) = det(A) · (1 + v^T A^{−1} u)`. With
`A = I_{N−1}`, `u = v = 1`:

```
det(I + 11^T) = 1 · (1 + 1^T I^{−1} 1) = 1 + (N−1) = N.
```

Therefore the per-axis volume scaling is `sqrt(det(J^T J)) = sqrt(N)`, and
across the `d` independent coord axes the total volume scaling is

```
sqrt(N)^d = N^(d/2).
```

Taking the log gives the **constant correction** between a density on the
reduced `R^((N−1)d)` space and the same density expressed on the zero-CoM
subspace `S`:

```
log q_ambient(x) = log q_reduced(y) + (d/2) · log(N).
```

### Why `CoMProjection.forward` / `.inverse` both return `log_det = 0`

`CoMProjection` is implemented as a pure relabelling between two
`(N−1)d`-dimensional spaces (the reduced space and the zero-CoM subspace, both
viewed as abstract Euclidean spaces of the same dimension). The Jacobian
determinant of this relabelling, **measured in the intrinsic metric of each
space**, is `1` — hence `log_det = 0`.

The `(d/2) · log(N)` factor appears only when one *re-embeds* the zero-CoM
subspace into its ambient `R^(Nd)` and evaluates the density using the
Euclidean volume form of the larger space. nflojax exposes this as a
separate static helper

```python
CoMProjection.ambient_correction(N, d)  # returns (d/2) * log(N)
```

so that callers can apply it explicitly when (and only when) they need the
ambient density.

### Why not bake the factor into `log_det`

Baking `±(d/2) · log(N)` into `CoMProjection` would double-count in the
augmented-coupling composition (bgmat-style), where translation invariance is
handled by a doubled-DoF construction that already yields ambient-valid
densities. Users of the augmented pattern would have to *subtract* the
correction back out — a fragile, implicit obligation. The explicit helper
pushes the bookkeeping to the call site where the measure convention is
decided. See [EXTENDING.md — CoM handling](EXTENDING.md#com-handling) for
the two composition patterns side-by-side and the "do not stack them"
warning.

### When the correction matters

- **Gradient-based training loss** (reverse-KL, forward-KL): the correction
  is a constant, so its gradient is zero. The loss is unchanged whether
  you add it or not.
- **Importance weights / SNIS / ESS / `logZ` estimates**: the correction
  affects absolute log-density values. Must be applied.
- **Absolute density comparisons across models or reference measures**:
  must be applied, with consistent conventions across all compared models.
- **Relative density within one flow** (ratios, conditional quantities):
  cancels; no correction needed.

## References

1. Dinh, L., Sohl-Dickstein, J., and Bengio, S. (2017). "Density estimation using Real-NVP." ICLR.

2. Durkan, C., Bekasov, A., Murray, I., and Papamakarios, G. (2019). "Neural Spline Flows." NeurIPS.

3. Perez, E., Strub, F., De Vries, H., Dumoulin, V., and Courville, A. (2018). "FiLM: Visual Reasoning with a General Conditioning Layer." AAAI.

4. Andrade, D. (2021). "Stable Training of Normalizing Flows for High-dimensional Variational Inference."

5. Papamakarios, G., Nalisnick, E., Rezende, D.J., Mohamed, S., and Lakshminarayanan, B. (2021). "Normalizing Flows for Probabilistic Modeling and Inference." JMLR.

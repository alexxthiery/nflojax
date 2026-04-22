# Usage

How-to cookbook for nflojax. Each section is self-contained with a copy-pasteable example.
For API details, see [REFERENCE.md](REFERENCE.md). For math, see [INTERNALS.md](INTERNALS.md).

**Contents:**

- [Affine Flow (RealNVP)](#affine-flow-realnvp)
- [Spline Flows](#spline-flows)
- [Conditional Flows](#conditional-flows)
- [Feature Extractor](#feature-extractor)
- [Transform-Only Mode](#transform-only-mode-bijection)
- [Identity Gating](#identity-gating)
- [Custom Architectures](#custom-architectures-assembly-api)
- [Assembly with Context](#assembly-with-context-and-feature-extractor)
- [Structured (rank-N) Flows](#structured-rank-n-flows)
- [Training](#training)

## Affine Flow (RealNVP)

Affine coupling layers as in Dinh et al. (2017). Build a flow, draw samples, evaluate log-density.

```python
import jax
from nflojax.builders import build_realnvp

key = jax.random.PRNGKey(0)
flow, params = build_realnvp(
    key, dim=16, num_layers=8, hidden_dim=256, n_hidden_layers=2,
)

samples = flow.sample(params, key, shape=(1000,))          # (1000, 16)
log_probs = flow.log_prob(params, samples)                  # (1000,)
samples, log_probs = flow.sample_and_log_prob(params, key, shape=(1000,))
```

Full options: [REFERENCE.md#builder-options](REFERENCE.md#builder-options)

## Spline Flows

Rational-quadratic splines are more expressive than affine couplings.

```python
from nflojax.builders import build_spline_realnvp

flow, params = build_spline_realnvp(
    key, dim=16, num_layers=8, hidden_dim=256, n_hidden_layers=2,
    num_bins=8,           # spline resolution
    tail_bound=5.0,       # linear tails outside [-B, B]
)
```

Spline-specific options: [REFERENCE.md#builder-options](REFERENCE.md#builder-options).
How splines work: [INTERNALS.md#spline-coupling](INTERNALS.md#spline-coupling)

## Conditional Flows

Model $p(x \mid \text{context})$ by setting `context_dim > 0`. Context is concatenated to conditioner inputs.

```python
flow, params = build_realnvp(
    key, dim=16, num_layers=8, hidden_dim=256, n_hidden_layers=2,
    context_dim=4,
)

context = jax.random.normal(key, (1000, 4))   # per-sample context
samples = flow.sample(params, key, shape=(1000,), context=context)
log_probs = flow.log_prob(params, samples, context=context)
```

Context can be per-sample `(batch, context_dim)` or shared `(context_dim,)`.

How conditioning works: [INTERNALS.md#conditional-normalizing-flows](INTERNALS.md#conditional-normalizing-flows)

## Feature Extractor

For high-dimensional or heterogeneous context, a learned ResNet can preprocess context before coupling layers.

```python
flow, params = build_realnvp(
    key, dim=16, num_layers=8, hidden_dim=256, n_hidden_layers=2,
    context_dim=64,                      # raw context dimension
    context_extractor_hidden_dim=128,    # >0 enables the extractor
    context_extractor_n_layers=2,        # depth of extractor
    context_feature_dim=16,              # output dim (default: same as context_dim)
)
```

The extractor is shared across all coupling layers. Its params live in `params["feature_extractor"]`.

Extractor options: [REFERENCE.md#context-feature-extractor](REFERENCE.md#context-feature-extractor)

## Transform-Only Mode (Bijection)

When you only need the invertible map with tractable Jacobian, without a base distribution:

```python
bijection, params = build_realnvp(
    key, dim=16, num_layers=8, hidden_dim=256, n_hidden_layers=2,
    context_dim=4,
    return_transform_only=True,
)

x = jax.random.normal(key, (1000, 16))
context = jax.random.normal(key, (1000, 4))

y, log_det_fwd = bijection.forward(params, x, context=context)
x_rec, log_det_inv = bijection.inverse(params, y, context=context)
# log_det_fwd + log_det_inv approx 0 (invertibility)
```

### Use cases

**Change of variables in integration:**

```python
z = sample_base(key, shape)
x, log_det = bijection.forward(params, z, context=context)
# Integrate f(x) * exp(log_det) under base measure
```

**Custom base distribution:**

```python
from nflojax.flows import Flow

bijection, bij_params = build_realnvp(..., return_transform_only=True)
my_flow = Flow(
    base_dist=my_custom_dist,
    transform=bijection.transform,
    feature_extractor=bijection.feature_extractor,
)
```

## Identity Gating

Enforce that the transform is identity at specific context values.
The gate function maps context to a scalar; wherever it returns 0, the transform becomes identity ($x \to x$, $\log\det = 0$).

```python
import jax.numpy as jnp
from nflojax.builders import build_realnvp

# Gate = sin(pi * t): identity at t=0 and t=1, full transform at t=0.5
gate_fn = lambda ctx: jnp.sin(jnp.pi * ctx[0])

flow, params = build_realnvp(
    key, dim=4, num_layers=4, hidden_dim=64, n_hidden_layers=2,
    context_dim=1,
    identity_gate=gate_fn,
)

# At t=0: transform is identity
x = jax.random.normal(key, (100, 4))
ctx_0 = jnp.zeros((100, 1))
y_0, log_det_0 = flow.forward(params, x, context=ctx_0)
# y_0 approx x, log_det_0 approx 0

# At t=0.5: full learned transform
ctx_half = jnp.ones((100, 1)) * 0.5
y_half, _ = flow.forward(params, x, context=ctx_half)
```

**Constraints:**
- Requires `context_dim > 0`
- Incompatible with `use_permutation=True`
- Gate receives **raw context**, even with a feature extractor. Couplings see extracted features; the gate does not. This is by design: the gate encodes known structure (e.g., boundary conditions), so it operates on interpretable inputs.
- Gate must be written for a **single sample** `(context_dim,)`. Batching is handled via `jax.vmap`.

How it works: [INTERNALS.md#identity-gate](INTERNALS.md#identity-gate)

## Custom Architectures (Assembly API)

Mix coupling types and control layer order using the assembly API.

```python
import jax
from nflojax.builders import make_alternating_mask, assemble_bijection, assemble_flow
from nflojax.transforms import AffineCoupling, SplineCoupling, LinearTransform, LoftTransform
from nflojax.distributions import StandardNormal

keys = jax.random.split(jax.random.PRNGKey(0), 5)
dim = 8
mask0 = make_alternating_mask(dim, parity=0)
mask1 = make_alternating_mask(dim, parity=1)

blocks_and_params = [
    AffineCoupling.create(keys[0], dim=dim, mask=mask0, hidden_dim=64, n_hidden_layers=2),
    AffineCoupling.create(keys[1], dim=dim, mask=mask1, hidden_dim=64, n_hidden_layers=2),
    SplineCoupling.create(keys[2], dim=dim, mask=mask0, hidden_dim=64, n_hidden_layers=2, num_bins=8),
    LinearTransform.create(keys[3], dim=dim),
    LoftTransform.create(keys[4], dim=dim),
]

# As Bijection (no base distribution)
bijection, params = assemble_bijection(blocks_and_params)

# As Flow (with base distribution)
flow, params = assemble_flow(blocks_and_params, base=StandardNormal(dim=dim))
```

Assembly API reference: [REFERENCE.md#assembly-api](REFERENCE.md#assembly-api)

## Assembly with Context and Feature Extractor

When using the assembly API with conditioning, create the feature extractor separately and pass the output dimension as `context_dim` to each coupling.

```python
from nflojax.builders import make_alternating_mask, create_feature_extractor, assemble_bijection
from nflojax.transforms import AffineCoupling, LoftTransform

keys = jax.random.split(key, 4)
dim = 8
raw_context_dim = 16
effective_context_dim = 8

# Create feature extractor
fe, fe_params = create_feature_extractor(
    keys[0], in_dim=raw_context_dim, hidden_dim=32, out_dim=effective_context_dim,
)

# Couplings use effective_context_dim (not raw)
mask0 = make_alternating_mask(dim, parity=0)
mask1 = make_alternating_mask(dim, parity=1)

blocks_and_params = [
    AffineCoupling.create(keys[1], dim=dim, mask=mask0, hidden_dim=64,
                          n_hidden_layers=2, context_dim=effective_context_dim),
    AffineCoupling.create(keys[2], dim=dim, mask=mask1, hidden_dim=64,
                          n_hidden_layers=2, context_dim=effective_context_dim),
    LoftTransform.create(keys[3], dim=dim),
]

bijection, params = assemble_bijection(
    blocks_and_params,
    feature_extractor=fe,
    feature_extractor_params=fe_params,
)

# Pass raw context; the extractor transforms it internally
raw_context = jax.random.normal(key, (100, raw_context_dim))
y, log_det = bijection.forward(params, x, context=raw_context)
```

## Structured (rank-N) Flows

When the event has structure beyond a flat vector — for example a particle
system with `N` particles in `d` coordinates — you can work directly on
rank-3 tensors `(*batch, N, d)` instead of flattening.

Two pieces make this possible:

- `StandardNormal` / `DiagNormal` accept an `event_shape` tuple. See
  [REFERENCE.md#event-shape-convention](REFERENCE.md#event-shape-convention).
- `SplitCoupling` is the structured analogue of `SplineCoupling`. It splits
  along a tensor axis (e.g. `-2` for the particle axis) instead of using a
  1-D mask. See
  [REFERENCE.md#splitcoupling](REFERENCE.md#splitcoupling).

### Example: particle flow on `(B, N, d)`

```python
import jax
import jax.numpy as jnp

from nflojax.builders import assemble_flow
from nflojax.distributions import StandardNormal
from nflojax.transforms import SplitCoupling

key = jax.random.PRNGKey(0)
N, d = 8, 3
num_layers = 6

# Rank-3 base distribution.
base = StandardNormal(event_shape=(N, d))

# Alternating-swap SplitCoupling layers covering both halves of particles.
keys = jax.random.split(key, num_layers)
blocks_and_params = []
for i, k in enumerate(keys):
    coupling, params = SplitCoupling.create(
        k,
        event_shape=(N, d),
        split_axis=-2,       # particle axis
        split_index=N // 2,
        event_ndims=2,        # particle + coord axes
        hidden_dim=64,
        n_hidden_layers=2,
        num_bins=8,
        tail_bound=5.0,
        swap=(i % 2 == 1),    # alternate which half is frozen
    )
    blocks_and_params.append((coupling, params))

# SplitCoupling has no 1-D mask, so the coverage analysis is skipped with
# validate=False. Alternating `swap` guarantees every particle is touched.
flow, params = assemble_flow(blocks_and_params, base=base, validate=False)

# Sampling returns a rank-3 tensor.
samples = flow.sample(params, key, shape=(16,))         # (16, N, d)
log_prob = flow.log_prob(params, samples)                # (16,)
```

### Training this flow

Same pattern as [Training](#training). Substitute a rank-3 target, e.g. a
`DiagNormal` with per-particle shifted means:

```python
from nflojax.distributions import DiagNormal

target = DiagNormal(event_shape=(N, d))
shifts = jnp.zeros((N, d)).at[:, 0].set(jnp.arange(N) * 0.5)
target_params = {"loc": shifts, "log_scale": jnp.log(0.5) * jnp.ones((N, d))}
```

and train by forward KL exactly as in the rank-1 recipe.

### Varying event_shape

`event_shape` is static per transform. To evaluate the same conditioner
weights at a different particle count `N'`, rebuild the flow with
`event_shape=(N', d)` and load the old params into the new pytree. This
only works when the conditioner is size-agnostic by construction (e.g. a
GNN over particle neighborhoods). The default MLP conditioner has its
`x_dim` / `out_dim` baked to the original `N` and will not transfer.

### Permutation-aware conditioners (DeepSets, Transformer, GNN)

The default `SplitCoupling.create(...)` uses an MLP, which does not
exploit the particle structure. To plug in a permutation-invariant or
-equivariant conditioner, construct `SplitCoupling` directly with
`flatten_input=False` so the conditioner sees `(*batch, N_frozen, d)`
instead of the flat `(*batch, N_frozen · d)` vector.

```python
from nflojax.nets import DeepSets
from nflojax.transforms import SplitCoupling

N, d, K = 8, 3, 8
params_per_scalar = 3 * K - 1

ds = DeepSets(
    phi_hidden=(64, 64), rho_hidden=(64,),
    out_dim=(N // 2) * d * params_per_scalar,
)

coupling = SplitCoupling(
    event_shape=(N, d), split_axis=-2, split_index=N // 2,
    event_ndims=2,
    conditioner=ds,
    num_bins=K, tail_bound=5.0,
    flatten_input=False,        # <-- structured input path
)
params = coupling.init_params(key)
```

Swap `DeepSets` for `Transformer` or `GNN` to pick a different
equivariance profile; all three satisfy the same
`SplitCoupling.init_params` identity-init pipeline. See
[REFERENCE.md#conditioners](REFERENCE.md#conditioners) for the constructor
signatures.

## Particle bases

For `(N, d)` particle events the base distribution lives on the particle
configuration space. Two ship today:

- `nflojax.distributions.UniformBox(geometry, event_shape)` — per-axis
  uniform on `geometry.box`, i.i.d. over leading event axes. Natural base
  for a **liquid starter kit** (`UniformBox` + `Rescale` → canonical
  range → coupling layers). See
  [REFERENCE.md#uniformbox](REFERENCE.md#uniformbox).
- `nflojax.distributions.LatticeBase` — Gaussian-perturbed crystalline
  lattice via five named factories
  (`fcc / diamond / bcc / hcp / hex_ice`). Natural base for a **solid
  starter kit** (`LatticeBase` → `Rescale` → couplings →
  `CoMProjection.inverse`). Use `permute=True` for indistinguishable
  particles (subtracts `log N!`; shuffles sample order). See
  [REFERENCE.md#latticebase](REFERENCE.md#latticebase).

```python
from nflojax.geometry import Geometry
from nflojax.distributions import UniformBox

# (N, d) liquid base on a cubic box [-L/2, L/2]^3.
geom = Geometry.cubic(d=3, side=2.0, lower=-1.0)
base = UniformBox(geometry=geom, event_shape=(N, 3))

samples = base.sample(None, key, (batch,))         # (batch, N, 3)
log_p   = base.log_prob(None, samples)             # (batch,); constant -N*sum(log(box))
```

`log_prob` returns `-inf` for configurations that fall outside the box on
any coord — useful when evaluating `flow.log_prob` on arbitrary points.

```python
from nflojax.distributions import LatticeBase

# (32, 3) FCC lattice on a [0, 2]^3 cubic box, σ = 0.05; indistinguishable.
lb = LatticeBase.fcc(n_cells=2, a=1.0, noise_scale=0.05, permute=True)

samples = lb.sample(None, key, (batch,))           # (batch, 32, 3); shuffled
log_p   = lb.log_prob(None, samples)               # (batch,); incl. -log(N!)
geom    = lb.geometry                              # box [0, 2]^3 ready for Rescale
```

## Conditioner features (embeddings)

Stateless feature transforms for the input side of a custom conditioner:

- `nflojax.embeddings.circular_embed(x, geometry, n_freq)` — per-coord
  Fourier features on a periodic box; lowest harmonic tiles `geometry.box`
  exactly. Last axis grows from `d` to `d * 2 * n_freq`.
- `nflojax.embeddings.positional_embed(t, n_freq, base=10_000)` —
  sinusoidal scalar embedding for context like temperature, density, or
  MD step. Last axis grows from `()` to `2 * n_freq`.

See [REFERENCE.md#nflojaxembeddings](REFERENCE.md#nflojaxembeddings) for
the full math + shape contracts.

```python
import flax.linen as nn
import jax.numpy as jnp
from nflojax.embeddings import circular_embed, positional_embed
from nflojax.geometry import Geometry

class FourierMLPConditioner(nn.Module):
    """Custom conditioner: Fourier features over particle coords + scalar T."""
    geometry: Geometry
    n_freq_x: int = 4
    n_freq_t: int = 8
    context_dim: int = 0  # for validate_conditioner; not used here
    out_dim: int = 1

    @nn.compact
    def __call__(self, x, context=None):
        # Particle Fourier features: (..., d) -> (..., d * 2 * n_freq_x).
        feat_x = circular_embed(x, self.geometry, self.n_freq_x)
        if context is not None:
            # Sinusoidal temperature embedding.
            feat_t = positional_embed(context, self.n_freq_t)
            feat_t = jnp.broadcast_to(
                feat_t[..., None, :], feat_x.shape[:-1] + feat_t.shape[-1:]
            )
            inp = jnp.concatenate([feat_x, feat_t], axis=-1)
        else:
            inp = feat_x
        return nn.Dense(self.out_dim, kernel_init=nn.initializers.zeros)(inp)
```

This is the canonical "user-supplied conditioner" pattern referenced in
DESIGN.md §5.2 — flows pass any PyTree context through, so a custom
conditioner can choose its own embedding strategy without changes
elsewhere in the stack.

## Periodic boxes and the torus

For flows on a periodic box, two pieces cooperate:

- `nflojax.geometry.Geometry` — an axis-aligned rectangular box + per-axis
  periodicity flags. Configuration object, not a PyTree.
  See [REFERENCE.md#geometry](REFERENCE.md#geometry).
- `nflojax.transforms.Rescale` — fixed per-axis affine from `geometry.box`
  to a canonical range (default `[-1, 1]`); use as the first layer so
  downstream splines / couplings see a fixed domain.
  See [REFERENCE.md#rescale](REFERENCE.md#rescale).
- `nflojax.transforms.CircularShift` — rigid per-coord shift mod the box
  length (the "rotation" half of a torus diffeomorphism).
  See [REFERENCE.md#circularshift](REFERENCE.md#circularshift).
- `nflojax.transforms.CoMProjection` — removes the centre-of-mass degree of
  freedom for `T(d)`-invariant targets: `(N, d) ↔ (N−1, d)`. **Log-det is
  zero on the reduced space**; add `CoMProjection.ambient_correction(N, d)
  = (d/2)·log(N)` when you need an ambient log-density (reverse-KL training
  against ambient `E(x)`, ESS / logZ estimates). See
  [REFERENCE.md#comprojection](REFERENCE.md#comprojection) and
  [EXTENDING.md#com-handling](EXTENDING.md#com-handling) for the "when to
  apply" decision box.

```python
from nflojax.transforms import CoMProjection

# Reduced-space base: (N-1, d). CoMProjection.inverse embeds to (N, d).
proj, params = CoMProjection.create(key)         # event_axis=-2 default

# When you need ambient log-density (e.g. comparing against E(x)):
log_q_ambient = flow.log_prob(params, x) + CoMProjection.ambient_correction(N, d)
```

```python
from nflojax.geometry import Geometry
from nflojax.transforms import CircularShift

# Cubic box [-1, 1]^3.
geom = Geometry.cubic(d=3, side=2.0, lower=-1.0)
shift, params = CircularShift.create(key, geometry=geom)

# Legacy-ergonomic alternative for the same cubic box:
shift = CircularShift.from_scalar_box(coord_dim=3, lower=-1.0, upper=1.0)
params = shift.init_params(key)

# Forward = rigid shift mod box length, log_det = 0.
y, log_det = shift.forward(params, x)
```

For full torus-bijection expressivity, stack `CircularShift` with a
`SplineCoupling` (or `SplitCoupling`) using `boundary_slopes='circular'`:

```python
from nflojax.transforms import CompositeTransform, SplineCoupling

coupling, c_params = SplineCoupling.create(
    key, dim=3, mask=jnp.array([1, 0, 1]),
    hidden_dim=64, n_hidden_layers=2,
    num_bins=16, tail_bound=1.0, boundary_slopes='circular',
)
blocks = [shift, coupling]
block_params = [params, c_params]
torus_flow = CompositeTransform(blocks=blocks)
```

The learnable shift discovers the optimal torus gauge; the circular-mode
spline does the local deformation with matched boundary slopes (C¹ on the
circle).

## Training

The library provides density evaluation; training loops are up to you.

### Forward KL (Maximum Likelihood)

Train on observed data by minimizing negative log-likelihood.

```python
import optax

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

@jax.jit
def loss_fn(params, x, context=None):
    return -flow.log_prob(params, x, context=context).mean()

@jax.jit
def step(params, opt_state, x, context=None):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, context)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

for batch in data_loader:
    params, opt_state, loss = step(params, opt_state, batch)
```

### Reverse KL (Variational Inference)

Train by sampling from the flow and minimizing $\text{KL}(q \| p)$ against an unnormalized target.

```python
import jax.numpy as jnp
import optax

def log_target(x):
    """Unnormalized log density of target."""
    return -0.5 * jnp.sum(x**2, axis=-1)

optimizer = optax.adam(1e-3)
opt_state = optimizer.init(params)

@jax.jit
def loss_fn(params, key):
    x, log_q = flow.sample_and_log_prob(params, key, shape=(256,))
    return jnp.mean(log_q - log_target(x))

@jax.jit
def train_step(params, opt_state, key):
    loss, grads = jax.value_and_grad(loss_fn)(params, key)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

for i in range(1000):
    key, subkey = jax.random.split(key)
    params, opt_state, loss = train_step(params, opt_state, subkey)
```

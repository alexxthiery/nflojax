# Extending nflojax

Recipes for adding custom Transforms, Distributions, and Conditioners, plus
pattern-level guidance for composing library primitives.

**Contents:**

- [Adding a Custom Transform](#adding-a-custom-transform)
- [Adding a Custom Distribution](#adding-a-custom-distribution)
- [Adding a Custom Conditioner](#adding-a-custom-conditioner)
- [CoM handling: projection vs augmented coupling](#com-handling)

---

## Adding a Custom Transform

Transforms must be `@dataclass` classes.
No base class or ABC is required; `CompositeTransform` relies on duck typing.

### Required Methods

```python
def forward(self, params, x, context=None, g_value=None) -> (y, log_det)
def inverse(self, params, y, context=None, g_value=None) -> (x, log_det)
def init_params(self, key, context_dim=0) -> params
```

- `params`: PyTree of learnable parameters (can be empty dict `{}`)
- `x`, `y`: Arrays of shape `(..., dim)`
- `context`: Optional conditioning tensor, shape `(..., context_dim)` or `None`
- `g_value`: Optional identity gate scalar or array (see INTERNALS.md)
- `log_det`: Log absolute Jacobian determinant, shape `(...,)`

All transforms must accept `context_dim=0` in `init_params`, even if unused.
This ensures a uniform interface for `CompositeTransform`.

### Optional Methods

```python
@classmethod
def create(cls, key, ...) -> (transform, params)  # Factory method
```

### Minimal Skeleton

```python
from dataclasses import dataclass
from typing import Any, Tuple
from jaxtyping import Array, PRNGKeyArray as PRNGKey

@dataclass
class MyTransform:
    dim: int

    def forward(
        self, params: Any, x: Array, context: Array | None = None,
        g_value: Array | None = None,
    ) -> Tuple[Array, Array]:
        batch_shape = x.shape[:-1]
        y = ...           # transform x
        log_det = ...     # shape batch_shape
        return y, log_det

    def inverse(
        self, params: Any, y: Array, context: Array | None = None,
        g_value: Array | None = None,
    ) -> Tuple[Array, Array]:
        batch_shape = y.shape[:-1]
        x = ...           # invert y
        log_det = ...     # shape batch_shape (negative of forward log_det)
        return x, log_det

    def init_params(self, key: PRNGKey, context_dim: int = 0) -> dict:
        del key, context_dim  # if unused
        return {}

    @classmethod
    def create(cls, key: PRNGKey, dim: int) -> Tuple["MyTransform", dict]:
        transform = cls(dim=dim)
        params = transform.init_params(key)
        return transform, params
```

### Templates

- **`Permutation`**: no learnable parameters, just shuffles dimensions
- **`LoftTransform`**: parameter-free but with nontrivial computation
- **`AffineCoupling`**: full example with conditioner network and identity gating

### Integration

Use with `CompositeTransform` to chain transforms:

```python
from nflojax.transforms import CompositeTransform

composite = CompositeTransform(blocks=[transform1, transform2, transform3])
y, log_det = composite.forward(params_list, x)
```

---

## Adding a Custom Distribution

Distributions must be `@dataclass` classes.

### Required Methods

```python
def log_prob(self, params, x) -> log_prob
def sample(self, params, key, shape) -> samples
def init_params(self) -> params
```

- `params`: PyTree of parameters (can be `None` for parameter-free distributions)
- `x`: Array of shape `(..., dim)`
- `log_prob`: Log probability, shape `(...,)`
- `key`: JAX PRNGKey
- `shape`: Tuple for batch dimensions (samples will be `shape + (dim,)`)

### Minimal Skeleton

```python
from dataclasses import dataclass
from typing import Any, Tuple

import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray as PRNGKey

@dataclass
class MyDistribution:
    dim: int

    def log_prob(self, params: Any, x: Array) -> Array:
        # x shape: (..., dim)
        log_p = ...  # shape (...,)
        return log_p

    def sample(self, params: Any, key: PRNGKey, shape: Tuple[int, ...]) -> Array:
        # return samples of shape (*shape, dim)
        return ...

    def init_params(self) -> Any:
        # None for parameter-free, dict for learnable
        return None
```

### Templates

- **`StandardNormal`**: isotropic Gaussian, no learnable params
- **`DiagNormal`**: diagonal Gaussian with learnable `loc` and `log_scale`

### Integration

Use with `Flow` as the base distribution:

```python
from nflojax.flows import Flow

flow = Flow(base_dist=my_distribution, transform=my_transform)
log_prob = flow.log_prob(params, x)
samples = flow.sample(params, key, (batch_size,))
```

---

## Adding a Custom Conditioner

Conditioners must be Flax `nn.Module` subclasses.
Coupling layers call `conditioner.apply({"params": params}, x, context)`.

### Required Interface

```python
# Flax __call__ method
def __call__(self, x, context=None) -> output

# Attributes
context_dim: int   # 0 for unconditional
```

- `x`: Input tensor, shape `(..., x_dim)`
- `context`: Optional conditioning tensor or `None`
- `output`: Shape `(..., out_dim)` where `out_dim` depends on the coupling type

### Required Methods (for auto-initialization)

```python
def get_output_layer(self, params) -> {"kernel": Array, "bias": Array}
def set_output_layer(self, params, kernel, bias) -> params
```

Coupling layers use these to zero-initialize the output layer at init:

- `AffineCoupling`: zeros kernel and bias so shift=0, scale=1
- `SplineCoupling`: sets biases for identity spline (raises error if methods are missing)

### Minimal Skeleton

```python
import jax.numpy as jnp
import flax.linen as nn
from jaxtyping import Array

class MyConditioner(nn.Module):
    x_dim: int
    context_dim: int = 0
    hidden_dim: int = 64
    out_dim: int = 1

    @nn.compact
    def __call__(self, x: Array, context: Array | None = None) -> Array:
        # Concatenate context if provided
        if context is not None:
            inp = jnp.concatenate([x, context], axis=-1)
        else:
            inp = x

        # Your network here
        h = nn.Dense(self.hidden_dim, name="dense_in")(inp)
        h = nn.relu(h)
        out = nn.Dense(self.out_dim, name="dense_out")(h)
        return out

    def get_output_layer(self, params: dict) -> dict:
        return params["dense_out"]

    def set_output_layer(self, params: dict, kernel, bias) -> dict:
        params = dict(params)
        params["dense_out"] = {"kernel": kernel, "bias": bias}
        return params
```

### Output Dimensions

Conditioner output size depends on the coupling type:

```python
AffineCoupling.required_out_dim(dim)                        # 2 * dim
SplineCoupling.required_out_dim(dim, num_bins)              # dim * (3 * num_bins - 1)
SplitCoupling.required_out_dim(transformed_flat, num_bins)  # transformed_flat * (3 * num_bins - 1)
```

### Input Shape Contract (Flat vs. Structured)

`AffineCoupling` and `SplineCoupling` pass `x * mask` to the conditioner,
shape `(..., dim)` — the flat mask contract. `SplitCoupling` instead
flattens the structured frozen slice before calling the conditioner:

```python
# Inside SplitCoupling._forward_or_inverse
batch_shape = frozen.shape[: -self.event_ndims]
frozen_flat = frozen.reshape(batch_shape + (-1,))
theta = self.conditioner.apply({"params": mlp_params}, frozen_flat, context)
```

So by default the conditioner receives a **flat** vector even when the
coupling's event is structured. The default `MLP` works unchanged.

If you want a conditioner that exploits the structure (e.g. a GNN over
particle neighborhoods), write a custom Flax module that accepts the
structured shape and handles the reshape itself. Two options:

1. **Reshape inside the conditioner.** Accept `x` of shape `(..., N*d)`,
   reshape to `(..., N, d)` inside `__call__`, and produce output of shape
   `(..., transformed_flat * (3K - 1))` to match `SplitCoupling`'s
   expectation. No changes to `SplitCoupling` needed.
2. **Bypass the default flattening.** Subclass `SplitCoupling` and override
   `_forward_or_inverse` to pass the structured slice directly. This is a
   deeper customization and should be a last resort.

Option 1 is the library default path; most GNN integrations fit it with a
single `x.reshape(*x.shape[:-1], N, d)` at the top of the conditioner.

### Template

- **`MLP`** in `nets.py`: full implementation with residual blocks, context handling, and output layer access

---

## CoM handling

Translation `T(d)` invariance of a particle-system target (no preferred
origin) creates a redundant degree of freedom: `E(x + c · 1)` is the same
energy for any constant shift `c ∈ R^d`. A flow that learns this redundancy
wastes capacity and its log-density is only defined up to the CoM, which
breaks log-density bookkeeping.

Two composition patterns fix this. **Pick one, apply it once, and document
which.** Stacking both silently double-counts the `(d/2) · log(N)` volume
factor and corrupts `logZ` / ESS / absolute density values — the gradient
still flows, so the symptoms are subtle.

### Pattern A: `CoMProjection`

Base distribution lives on the `(N−1, d)` reduced space. The last bijection
before ambient coordinates is `CoMProjection.inverse`, which embeds into
the zero-CoM subspace of `R^(Nd)`.

```python
from nflojax.distributions import DiagNormal
from nflojax.transforms import CoMProjection, CompositeTransform
from nflojax.flows import Flow

# Inner flow operates on (N-1, d).
inner = CompositeTransform(blocks=[...])
proj, proj_params = CoMProjection.create(key)

# Composite: inner first, then CoMProjection.inverse at sampling time.
# (Remember composition is T_n o ... o T_1 — place CoMProjection last.)
flow = Flow(
    base_dist=DiagNormal(event_shape=(N - 1, d)),
    transform=CompositeTransform(blocks=[inner, proj]),
)

# Sampling: base (N-1, d) -> inner -> CoMProjection.inverse -> x in ambient
# zero-CoM subspace.
samples = flow.sample(params, key, (batch_size,))

# Log-prob on the REDUCED space (this is what `flow.log_prob` returns).
log_q_reduced = flow.log_prob(params, samples)

# To compare against an ambient energy E(x), add the correction:
log_q_ambient = log_q_reduced + CoMProjection.ambient_correction(N, d)

# For gradient-based training, the constant is irrelevant — either form
# gives the same gradient. Keep the form that matches the downstream
# quantity you measure.
```

**When to apply the correction**: see the decision box in
[REFERENCE.md#comprojection](REFERENCE.md#comprojection). Short version —
*yes* for importance weights / ESS / `logZ` / absolute density, *no* for
training-loss gradients.

### Pattern B: Augmented coupling (bgmat-style)

Double the degrees of freedom with auxiliary Gaussian variables, train the
inner flow on the `(2N, d)` augmented state, and marginalise the auxiliary
half at inference. Translation invariance is handled *inside* the augmented
flow by centring both halves symmetrically — no `CoMProjection` needed.

Sketch (full implementation lives in the bgmat application, not in
nflojax):

```python
# Base: Gaussian on (2N, d); inner flow: SplitCoupling acting across the
# physical/auxiliary split_axis. At sampling time, drop the auxiliary half.
# At log-prob time, integrate over auxiliaries via closed-form or MC.
# No CoMProjection in the chain.
```

The augmented pattern encodes translation invariance via the symmetric
Gaussian base (both halves zero-centred) rather than explicitly removing a
DoF. The `(d/2) · log(N)` factor is *not* applicable here — doubling the
degrees of freedom keeps the ambient dimension intact, and the auxiliary
marginalisation absorbs the bookkeeping.

> ⚠️  **Do not stack Pattern A and Pattern B.** If the flow already uses
> augmented coupling, do not also wrap it with `CoMProjection`, and do not
> add `ambient_correction` anywhere. The augmented pattern's density is
> already ambient-valid. Adding the correction over-counts.

### Why this distinction is in `EXTENDING.md` and not `transforms.py`

Both patterns are composition recipes, not primitives. nflojax ships the
pieces (`CoMProjection`, the standard coupling/spline bijections, the
structured `SplitCoupling` used inside augmented flows) but does not ship
an "augmented coupling" class — the pattern lives here as documentation.
See DESIGN.md §4 item 9 for why.

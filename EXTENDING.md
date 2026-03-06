# Extending nflojax

Recipes for adding custom Transforms, Distributions, and Conditioners.

**Contents:**

- [Adding a Custom Transform](#adding-a-custom-transform)
- [Adding a Custom Distribution](#adding-a-custom-distribution)
- [Adding a Custom Conditioner](#adding-a-custom-conditioner)

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

### Required Interface

```python
def log_prob(params, x) -> log_prob
def sample(params, key, shape) -> samples
```

- `params`: PyTree of parameters (can be `None` for parameter-free distributions)
- `x`: Array of shape `(..., dim)`
- `log_prob`: Log probability, shape `(...,)`
- `key`: JAX PRNGKey
- `shape`: Tuple for batch dimensions (samples will be `shape + (dim,)`)

### Optional Methods

```python
def init_params() -> params  # Initialize parameters (no key needed)
```

### Templates

- **`StandardNormal`** — Simplest: isotropic Gaussian, no learnable params
- **`DiagNormal`** — Diagonal Gaussian with learnable `loc` and `log_scale`

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

### Required Interface

```python
def apply({"params": params}, x, context) -> output  # Flax convention
context_dim: int  # Attribute (0 for unconditional)
```

- Must follow Flax module conventions
- `x`: Input tensor, shape `(..., x_dim)`
- `context`: Optional conditioning tensor or `None`
- `output`: Shape `(..., out_dim)` where `out_dim` depends on the coupling type

### Optional Methods (for auto-initialization)

```python
def get_output_layer(params) -> {"kernel": Array, "bias": Array}
def set_output_layer(params, kernel, bias) -> params
```

If present, coupling layers will automatically initialize the output layer:
- `AffineCoupling`: Zero-initializes for identity start
- `SplineCoupling`: Sets biases for identity spline (raises error if methods missing)

**Note:** `SplineCoupling` emits a warning if the derivative range `[min_derivative, max_derivative]`
does not contain 1.0, since identity-like initialization requires `derivative ≈ 1`. In this case,
the midpoint derivative is used instead.

### Template

- **`MLP`** in `nets.py` — Full implementation with context handling and output layer access

### Output Dimensions

Conditioner output size depends on the coupling type:

```python
AffineCoupling.required_out_dim(dim)           # Returns 2 * dim
SplineCoupling.required_out_dim(dim, num_bins) # Returns dim * (3 * num_bins - 1)
```

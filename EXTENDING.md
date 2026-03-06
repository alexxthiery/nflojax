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
AffineCoupling.required_out_dim(dim)           # Returns 2 * dim
SplineCoupling.required_out_dim(dim, num_bins) # Returns dim * (3 * num_bins - 1)
```

### Template

- **`MLP`** in `nets.py`: full implementation with residual blocks, context handling, and output layer access

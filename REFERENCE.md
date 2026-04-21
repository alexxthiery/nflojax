# API Reference

**Contents:**

- [Event Shape Convention](#event-shape-convention)
- [Core Classes](#core-classes)
- [Builders](#builders)
- [Assembly API](#assembly-api)
- [Transforms](#transforms)
- [Distributions](#distributions)
- [Utilities](#utilities)
- [Parameter Structure](#parameter-structure)
- [Forward/Inverse Convention](#forwardinverse-convention)
- [Context Feature Extractor](#context-feature-extractor)

## Event Shape Convention

An event may have rank 1 (flat vector) or rank > 1 (structured, e.g. a batch
of `N` particles in `d` dimensions: `event_shape = (N, d)`).

All base distributions accept either form at construction:

| Form | Meaning |
|------|---------|
| `StandardNormal(event_shape=(N, d))` | Rank-N event of shape `(N, d)` |
| `StandardNormal(event_shape=N)` | Rank-1 sugar for `(N,)` |
| `StandardNormal(dim=N)` | Legacy alias, same as `event_shape=N` |

Internally the canonical form is always a tuple. A read-only `dim` property
returns `event_shape[-1]` for back-compat with the rank-1 API.

**Expected tensor shapes** throughout the library:

| Tensor | Shape |
|--------|-------|
| `x` fed to `log_prob` / `forward` / `inverse` | `(*batch, *event_shape)` |
| `log_prob`, `log_det` returned | `batch` |
| Samples from `base_dist.sample(params, key, (B,))` | `(B, *event_shape)` |

Rank-1 and rank-N events use the same code paths. For flat events,
`event_shape = (dim,)` and behavior is bit-identical to the original
`dim=int` API.

For transforms that can act on rank > 1 events, see
[`SplitCoupling`](#splitcoupling).

## Core Classes

### Flow

Normalizing flow distribution: base distribution + invertible transform.

```python
from nflojax.flows import Flow

flow = Flow(base_dist, transform, feature_extractor=None, identity_gate=None)
```

| Method | Signature | Returns |
|--------|-----------|---------|
| `forward` | `(params, z, context=None)` | `(x, log_det)` |
| `inverse` | `(params, x, context=None)` | `(z, log_det)` |
| `log_prob` | `(params, x, context=None)` | `log_prob` |
| `sample` | `(params, key, shape, context=None)` | `x` |
| `sample_and_log_prob` | `(params, key, shape, context=None)` | `(x, log_prob)` |

Shapes: `x`, `z` are `(..., dim)`. `log_det`, `log_prob` are `(...,)`. `context` is `(..., context_dim)` or `None`.

### Bijection

Invertible transform with tractable Jacobian, no base distribution.

```python
from nflojax.flows import Bijection

bijection = Bijection(transform, feature_extractor=None, identity_gate=None)
```

| Method | Signature | Returns |
|--------|-----------|---------|
| `forward` | `(params, x, context=None)` | `(y, log_det)` |
| `inverse` | `(params, y, context=None)` | `(x, log_det)` |

## Builders

### build_realnvp

```python
from nflojax.builders import build_realnvp

flow_or_bijection, params = build_realnvp(
    key, dim, num_layers, hidden_dim, n_hidden_layers, **options
)
```

### build_spline_realnvp

```python
from nflojax.builders import build_spline_realnvp

flow_or_bijection, params = build_spline_realnvp(
    key, dim, num_layers, hidden_dim, n_hidden_layers, **options
)
```

### Builder Options

**Shared options** (both builders):

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `context_dim` | int | 0 | Conditioning variable dimension; 0 = unconditional |
| `context_extractor_hidden_dim` | int | 0 | Feature extractor hidden width; 0 = disabled |
| `context_extractor_n_layers` | int | 2 | Residual blocks in context extractor |
| `context_feature_dim` | int or None | None | Extractor output dim; None = same as context_dim |
| `res_scale` | float | 0.1 | Scale factor for residual connections |
| `activation` | callable | `jax.nn.tanh` | Conditioner MLP activation |
| `use_permutation` | bool | False | Reverse permutations between couplings |
| `use_linear` | bool | False | Prepend LU-parameterized linear transform |
| `use_loft` | bool | True | Append LoftTransform for tail stabilization |
| `loft_tau` | float | 1000.0 | LOFT threshold parameter |
| `trainable_base` | bool | False | Use DiagNormal with learnable loc/scale |
| `base_dist` | object or None | None | Custom base distribution; overrides trainable_base |
| `base_params` | PyTree or None | None | Params for custom base_dist |
| `return_transform_only` | bool | False | Return Bijection instead of Flow |
| `identity_gate` | callable or None | None | Context -> scalar gate for identity interpolation |

**Affine-specific** (`build_realnvp` only):

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `max_log_scale` | float | 5.0 | Bound on log-scale via tanh clamping |

**Spline-specific** (`build_spline_realnvp` only):

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `num_bins` | int | 8 | Number of spline bins (K) |
| `tail_bound` | float | 5.0 | Spline acts on [-B, B]; linear tails outside |
| `min_bin_width` | float | 1e-2 | Floor for bin widths |
| `min_bin_height` | float | 1e-2 | Floor for bin heights |
| `min_derivative` | float | 1e-2 | Lower bound for knot derivatives |
| `max_derivative` | float | 10.0 | Upper bound for knot derivatives |

## Assembly API

For custom architectures (mixing coupling types, non-standard layer order).

### assemble_bijection

```python
from nflojax.builders import assemble_bijection

bijection, params = assemble_bijection(
    blocks_and_params,
    feature_extractor=None,
    feature_extractor_params=None,
    validate=True,
    identity_gate=None,
)
```

`blocks_and_params` is a list of `(transform, params)` tuples from `.create()` calls.
Returns params dict: `{"transform": [...], "feature_extractor": ...}`.

### assemble_flow

```python
from nflojax.builders import assemble_flow

flow, params = assemble_flow(
    blocks_and_params,
    base,
    base_params=None,
    feature_extractor=None,
    feature_extractor_params=None,
    validate=True,
    identity_gate=None,
)
```

Returns params dict: `{"base": ..., "transform": [...], "feature_extractor": ...}`.

### Utilities

```python
from nflojax.builders import make_alternating_mask, create_feature_extractor, analyze_mask_coverage

mask = make_alternating_mask(dim, parity)    # parity: 0 or 1
fe, fe_params = create_feature_extractor(
    key, in_dim, hidden_dim, out_dim,
    n_layers=2, activation=jax.nn.tanh, res_scale=0.1,
)
analyze_mask_coverage(blocks, dim)           # warns if any dimension is never transformed
```

`analyze_mask_coverage` is called automatically by `assemble_bijection`, `assemble_flow`, and the builders. It traces mask/permutation interactions across coupling layers and prints a warning if any original dimension is never in the "transformed" role.

## Transforms

All transforms share this interface:

```python
y, log_det = transform.forward(params, x, context=None, g_value=None)
x, log_det = transform.inverse(params, y, context=None, g_value=None)
transform, params = TransformClass.create(key, **kwargs)
```

Shapes: `x`, `y` are `(..., dim)`. `log_det` is `(...,)` (one scalar per sample). `context` is `(..., context_dim)` or `None`. `g_value` is `()` or `(batch,)` or `None`.

**Mask convention** (couplings): `mask[i] = 1` means dimension `i` is frozen (passed through); `mask[i] = 0` means transformed.

**Identity gating**: all transforms except `Permutation` accept an optional `g_value`. At `g=0` the transform is identity; at `g=1` it acts normally. Intermediate values interpolate smoothly. `Permutation` has no `g_value` parameter (it has no learnable components to gate). `CompositeTransform` skips `g_value` for blocks that don't support it.

**Zero initialization**: conditioner output layers are initialized to zero, so all learnable transforms start as identity.

| Transform | Purpose |
|-----------|---------|
| `AffineCoupling` | RealNVP affine coupling layer |
| `SplineCoupling` | Rational-quadratic spline coupling layer |
| `SplitCoupling` | Spline coupling for rank-N (particle-system) events |
| `LinearTransform` | LU-parameterized invertible linear map |
| `Permutation` | Fixed shuffle along any negative event axis |
| `CircularShift` | Rigid per-coord shift mod box (torus rotation) |
| `LoftTransform` | Log-soft tail compression |
| `CompositeTransform` | Sequential composition of transforms |

### AffineCoupling

**What:** RealNVP-style coupling. Frozen dimensions condition an MLP that produces shift and log-scale for the transformed dimensions.

**Forward:**

```
x1 = x * mask                                    # frozen
(shift, log_scale) = MLP(x1, context) * (1-mask)  # bounded by tanh
y = x1 + (x * (1-mask)) * exp(log_scale) + shift
log_det = sum(log_scale)
```

**Inverse:** `x2 = (y2 - shift) * exp(-log_scale)`, log_det = `-sum(log_scale)`.

**Create:**

```python
coupling, params = AffineCoupling.create(
    key, dim, mask, hidden_dim, n_hidden_layers, **kwargs
)
```

| Param | Type | Default | Meaning |
|-------|------|---------|---------|
| `dim` | int | required | Input/output dimensionality |
| `mask` | Array | required | Binary mask of shape `(dim,)` |
| `hidden_dim` | int | required | Conditioner MLP hidden width |
| `n_hidden_layers` | int | required | Residual blocks in conditioner MLP |
| `context_dim` | int | `0` | Context dimension; 0 = unconditional |
| `activation` | callable | `nn.elu` | MLP activation function |
| `res_scale` | float | `0.1` | Residual connection scale |
| `max_log_scale` | float | `5.0` | Bound on `log_scale` via `tanh(. / max) * max` |
| `max_shift` | float | `None` | Bound on shift; defaults to `exp(max_log_scale)` |

**Params dict:**

| Key | Shape | Description |
|-----|-------|-------------|
| `"mlp"` | PyTree | Flax params for conditioner MLP (output dim = `2 * dim`) |

**Context:** MLP receives `concat(x_masked, context)`. When `g_value` is provided, both shift and log_scale are scaled by `g`.

### SplineCoupling

**What:** Coupling layer using monotone rational-quadratic splines. More expressive than affine: each transformed dimension gets a flexible monotone function parameterized by `K` bins.

**Forward:** Frozen dimensions condition an MLP that outputs spline knot parameters (widths, heights, derivatives) for each transformed dimension. The spline acts on `[-B, B]` with linear identity tails outside.

**Inverse:** Analytical inverse of the rational-quadratic spline (closed-form per bin).

**Create:**

```python
coupling, params = SplineCoupling.create(
    key, dim, mask, hidden_dim, n_hidden_layers, **kwargs
)
```

| Param | Type | Default | Meaning |
|-------|------|---------|---------|
| `dim` | int | required | Input/output dimensionality |
| `mask` | Array | required | Binary mask of shape `(dim,)` |
| `hidden_dim` | int | required | Conditioner MLP hidden width |
| `n_hidden_layers` | int | required | Residual blocks in conditioner MLP |
| `context_dim` | int | `0` | Context dimension; 0 = unconditional |
| `num_bins` | int | `8` | Number of spline bins (K) |
| `tail_bound` | float | `5.0` | Spline domain `[-B, B]`; identity outside |
| `min_bin_width` | float | `1e-2` | Floor for bin widths (numerical stability) |
| `min_bin_height` | float | `1e-2` | Floor for bin heights (numerical stability) |
| `min_derivative` | float | `1e-2` | Lower bound for knot derivatives |
| `max_derivative` | float | `10.0` | Upper bound for knot derivatives |
| `activation` | callable | `nn.elu` | MLP activation function |
| `res_scale` | float | `0.1` | Residual connection scale |

**Params dict:**

| Key | Shape | Description |
|-----|-------|-------------|
| `"mlp"` | PyTree | Flax params for conditioner MLP (output dim = `dim * (3K - 1)`) |

The `3K - 1` per dimension breaks down as: K widths + K heights + (K-1) interior derivatives.

**Note:** The low-level `rational_quadratic_spline` function in `splines.py` defaults to `min_bin_width=1e-3` and `min_bin_height=1e-3`, while `SplineCoupling.create` and `build_spline_realnvp` default to `1e-2`. Users calling the spline primitive directly will get tighter bin floors.

**Context:** Same as AffineCoupling: MLP receives `concat(x_masked, context)`. When `g_value` is provided, spline params interpolate toward identity: widths and heights are scaled by `g` (uniform bins at g=0), and derivatives are interpolated in logit space toward derivative=1.

### SplitCoupling

**What:** Structured analogue of `SplineCoupling` for rank >= 2 events. Instead of a flat 1-D mask, it splits the input along a chosen tensor axis at a chosen index. The frozen slice is flattened and fed to the conditioner; monotone rational-quadratic splines act elementwise on the transformed slice; log-det is summed over all `event_ndims` trailing axes.

**Target use case:** particle systems with `event_shape = (N, d)`, where `split_axis = -2` splits particles into two halves and `event_ndims = 2` tells the log-det to sum over both the particle axis and the coordinate axis.

**Shapes** (for `event_shape = (N, d)`, `split_axis = -2`, `split_index = N // 2`, `event_ndims = 2`):

| Tensor | Shape |
|--------|-------|
| `x` in | `(*batch, N, d)` |
| Frozen slice (passed to conditioner, flattened) | `(*batch, (N//2) * d)` |
| Conditioner output | `(*batch, (N//2) * d * (3K - 1))` |
| Transformed slice | `(*batch, N//2, d)` |
| `y` out | `(*batch, N, d)` |
| `log_det` | `batch` |

**Forward:** `x -> split -> conditioner(frozen_flat, context) -> spline(transformed) -> concat`. Inverse reverses the spline; split/concat are symmetric.

**Mask semantics:** none. The partition is geometric (axis + index), not a per-scalar mask. Alternate `swap` between layers to cover all particles (e.g. layer 0 freezes first half, layer 1 freezes second half).

**Create:**

```python
coupling, params = SplitCoupling.create(
    key,
    event_shape=(N, d),
    split_axis=-2,
    split_index=N // 2,
    event_ndims=2,
    hidden_dim=64,
    n_hidden_layers=2,
    **kwargs,
)
```

| Param | Type | Default | Meaning |
|-------|------|---------|---------|
| `event_shape` | tuple of int | required | Trailing event shape, e.g. `(N, d)`. Rank must equal `event_ndims` |
| `split_axis` | int | required | Negative axis index (e.g. `-2`); must lie inside the event axes |
| `split_index` | int | required | Size of the first partition along `split_axis`; must be `< axis_size` |
| `event_ndims` | int | required | Number of trailing axes that are part of the event (>= 1) |
| `hidden_dim` | int | required | Conditioner MLP hidden width |
| `n_hidden_layers` | int | required | Residual blocks in conditioner MLP |
| `context_dim` | int | `0` | Context dimension; 0 = unconditional |
| `swap` | bool | `False` | If `True`, the last `axis_size - split_index` slots are frozen instead |
| `num_bins` | int | `8` | Spline bins `K` |
| `tail_bound` | float | `5.0` | Spline domain `[-B, B]`; identity outside |
| `min_bin_width` | float | `1e-2` | Floor for bin widths |
| `min_bin_height` | float | `1e-2` | Floor for bin heights |
| `min_derivative` | float | `1e-2` | Lower bound for knot derivatives |
| `max_derivative` | float | `10.0` | Upper bound for knot derivatives |
| `activation` | callable | `nn.elu` | MLP activation function |
| `res_scale` | float | `0.1` | Residual connection scale |

**Params dict:**

| Key | Shape | Description |
|-----|-------|-------------|
| `"mlp"` | PyTree | Flax params for conditioner MLP (output dim = `transformed_flat * (3K - 1)`) |

Where `transformed_flat = prod(event_shape) * (axis_size - split_index) / axis_size` (or the swapped complement).

**Static methods:**

- `SplitCoupling.required_out_dim(transformed_flat, num_bins) -> int`: computes conditioner output size as `transformed_flat * (3K - 1)`.

**Covering all slots:** `SplitCoupling` has no `mask` attribute, so `analyze_mask_coverage` cannot verify that every slot gets transformed. The recommended pattern is alternating `swap` between successive layers. A minimum of 2 layers is required.

**Validation:** `__post_init__` checks rank consistency, axis bounds, and `split_index` range. `forward`/`inverse` additionally check that `x.shape[-event_ndims:] == event_shape`.

### LinearTransform

**What:** LU-parameterized invertible matrix. Mixes all dimensions via a learned linear map `W = L * T` where `L` is unit-diagonal lower triangular and `T = U + diag(s)` is upper triangular with `s = softplus(raw_diag)`.

**Forward (unconditional):** `y = x @ W^T`, log_det = `sum(log(s))`.

**Forward (conditional):** MLP maps context to `(delta_diag, shift)` each of shape `(dim,)`. Then `s = softplus(raw_diag + delta_diag)`, `y = x @ W^T + shift`. The shift does not affect log_det.

**Inverse:** Triangular solves (no explicit matrix inversion). Conditional: subtract shift first.

**Create:**

```python
transform, params = LinearTransform.create(key, dim, **kwargs)
```

| Param | Type | Default | Meaning |
|-------|------|---------|---------|
| `dim` | int | required | Input/output dimensionality |
| `context_dim` | int | `0` | Context dimension; 0 = unconditional |
| `hidden_dim` | int | `64` | Conditioner MLP hidden width (only when `context_dim > 0`) |
| `n_hidden_layers` | int | `2` | Residual blocks in conditioner MLP |
| `activation` | callable | `nn.tanh` | MLP activation function |
| `res_scale` | float | `0.1` | Residual connection scale |

**Params dict:**

| Key | Shape | When | Description |
|-----|-------|------|-------------|
| `"lower"` | `(dim, dim)` | always | Lower triangular matrix (unit diagonal enforced) |
| `"upper"` | `(dim, dim)` | always | Upper triangular matrix (zero diagonal enforced) |
| `"raw_diag"` | `(dim,)` | always | Pre-softplus diagonal entries |
| `"mlp"` | PyTree | `context_dim > 0` | Conditioner MLP (output dim = `2 * dim`) |

**Gating:** The LU factors are gated component-wise: `L_off -> g * L_off`, `U_off -> g * U_off`, `s -> 1 - g + g * s`, `shift -> g * shift`. At `g=0` this gives `L=I`, `T=I`, so the transform is exactly identity. At `g=1` it acts as the full learned transform. The interpolation path is not the same as `g * W + (1-g) * I` due to cross-terms in the LU product.

**Init:** `W = I`, shift = 0 (MLP output layer zero-initialized).

### Permutation

**What:** Fixed shuffle along any negative event axis. No learnable parameters.

**Forward:** `y = jnp.take(x, perm, axis=event_axis)`, log_det = 0 with shape equal to `x.shape` minus the permuted axis.

**Inverse:** `x = jnp.take(y, inv_perm, axis=event_axis)` where `inv_perm` is precomputed at construction.

**Create:**

```python
# Last-axis shuffle (coordinates), the historic default:
transform, params = Permutation.create(key, perm=jnp.arange(dim)[::-1])

# Particle-axis shuffle on (B, N, d) events:
transform, params = Permutation.create(key, perm=perm_N, event_axis=-2)
```

| Param | Type | Default | Meaning |
|-------|------|---------|---------|
| `perm` | Array | required | 1-D integer index array of shape `(k,)` where `k` is the size of the permuted axis |
| `event_axis` | int | `-1` | Negative axis along which to permute. Must be negative. `-1` for coordinate-axis permutation (flat couplings). `-2` for particle-axis permutation on `(B, N, d)` events. |

**Params dict:** `{}` (empty, no learnable parameters).

**Log-det shape:** `x.shape` with the permuted axis removed. For `x.shape = (B, N, d)` and `event_axis=-2`, log-det is `(B, d)`.

### CircularShift

**What:** Rigid per-coordinate shift modulo the box length. The "rotation" half of a torus diffeomorphism — compose with a `boundary_slopes='circular'` spline coupling for full torus-bijection expressivity.

**Forward:**

```
y = (x - lower + shift) mod (upper - lower) + lower
log_det = 0  (scalar)
```

`shift` is a learnable per-coord vector of shape `(d,)` where `d = geometry.d`; it broadcasts across every preceding axis (particles, batch, etc.).

**Inverse:** Same wrap with `-shift`. Log-det is again scalar zero.

**Create:**

```python
from nflojax.geometry import Geometry
from nflojax.transforms import CircularShift

# Preferred: pass a Geometry.
geom = Geometry.cubic(d=3, side=2.0, lower=-1.0)  # [-1, 1]^3
transform, params = CircularShift.create(key, geometry=geom)

# Ergonomic fallback for a simple cubic box:
transform = CircularShift.from_scalar_box(coord_dim=3, lower=-1.0, upper=1.0)
params = transform.init_params(key)
```

| Param | Type | Default | Meaning |
|-------|------|---------|---------|
| `geometry` | `Geometry` | required | Axis-aligned box that defines the torus |

**Params dict:** `{"shift": (d,) float32}` — zero at init (identity at init).

**Log-det:** scalar zero. Composes cleanly inside `CompositeTransform` (whose accumulator starts as scalar zero and broadcasts up).

**Pairing:** For full torus-diffeomorphism expressivity, stack `CircularShift` with a `SplineCoupling` (or `SplitCoupling`) whose inner spline uses `boundary_slopes='circular'`. The shift discovers the optimal gauge; the spline does the local deformation with matched boundary slopes (C¹ on the circle).

### LoftTransform

**What:** Element-wise log-soft tail compression. Linear for `|z| <= tau`, logarithmic for `|z| > tau`. Prevents overflow in high dimensions. No learnable parameters.

**Forward:**

```
g(z) = sign(z) * [ min(|z|, tau) + log( max(|z| - tau, 0) + 1 ) ]
```

`log_det = sum_i log|g'(z_i)|` where `g'(z) = 1 / (max(|z| - tau, 0) + 1)`.

**Inverse:** `g_inv(y) = sign(y) * [min(|y|, tau) + expm1(max(|y| - tau, 0))]`, clamped to prevent overflow.

**Create:**

```python
transform, params = LoftTransform.create(key, dim, tau=1000.0)
```

| Param | Type | Default | Meaning |
|-------|------|---------|---------|
| `dim` | int | required | Input/output dimensionality |
| `tau` | float | `1000.0` | Threshold: linear below, logarithmic above |

**Params dict:** `{}` (empty, no learnable parameters).

**Gating:** When `g_value` is provided, `y = (1-g)*x + g*loft(x)` with matched log_det. Inverse uses 10 Newton iterations (hardcoded, no convergence check) starting from `y` as initial guess. The inverse also clamps the exponent argument to 80.0 to prevent float32 overflow in `expm1`.

### CompositeTransform

**What:** Sequential composition `T = T_n ... T_1`. Forward applies blocks left-to-right, inverse applies them right-to-left.

**Forward:** Chains all blocks, accumulating log_det. Passes `context` and `g_value` to each block that supports them (Permutation does not receive `g_value`). When `jax_enable_x64` is set, log_det is accumulated in float64 for numerical precision, then cast back to input dtype.

**Inverse:** Same blocks in reverse order.

**Create:** No `.create()` class method. Constructed directly:

```python
transform = CompositeTransform(blocks=[t1, t2, t3])
params = [params1, params2, params3]  # list matching block order
```

**Params:** A list of per-block param dicts, one per block in the same order as `blocks`.

**Rank-polymorphic:** `CompositeTransform` works with any event rank. It initializes the log-det accumulator as a scalar zero and lets each block's log-det broadcast up to its own batch shape. This means a `CompositeTransform([SplitCoupling(...), SplitCoupling(...)])` on a rank-3 event returns `log_det` of shape `batch`, same as rank-1.

## Geometry

```python
from nflojax.geometry import Geometry
```

**What:** Value object for an axis-aligned rectangular box in `R^d` plus per-axis periodicity flags. Consumed by any geometry-aware primitive (`CircularShift` today; `Rescale`, `UniformBox`, `LatticeBase`, `utils/pbc.nearest_image` in future stages). Numpy-backed, frozen, not a PyTree.

**Scope:**

- IS: axis-aligned rectangular domain + per-axis periodicity. Covers cubic, rectangular, fully-periodic torus, slab, fully-open.
- IS NOT: triclinic cells, curved manifolds, metric spaces, time-dependent / deforming boxes, internal-coordinate spaces.

**Construct:**

```python
# Cubic box (most common case):
geom = Geometry.cubic(d=3)                    # [-0.5, 0.5]^3
geom = Geometry.cubic(d=3, side=2.0)          # [-1, 1]^3
geom = Geometry.cubic(d=3, side=L, lower=0)   # [0, L]^3

# Rectangular / slab: use the bare constructor.
geom = Geometry(lower=[0, 0, 0], upper=[Lx, Ly, Lz])
geom = Geometry(lower=[0, 0, 0], upper=[Lx, Ly, Lz],
                periodic=[True, True, False])  # periodic in xy, open in z
```

There is intentionally no `Geometry.box(...)` factory — it would shadow the `@property box`.

**Fields:**

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `lower` | `np.ndarray` shape `(d,)` | required | Per-axis lower bounds |
| `upper` | `np.ndarray` shape `(d,)` | required | Per-axis upper bounds |
| `periodic` | `np.ndarray` shape `(d,)` bool or `None` | `None` | Per-axis PBC flags; `None` means all-periodic |

**Derived:**

| Property | Type | Meaning |
|----------|------|---------|
| `d` | `int` | Spatial dimensionality `len(lower)` |
| `box` | `np.ndarray` shape `(d,)` | Per-axis side lengths `upper - lower` |
| `volume` | `float` | `prod(box)` |
| `is_periodic(axis=None)` | `bool` | True if `axis` (or every axis) is periodic |

**Validation:** `__post_init__` enforces `lower.shape == upper.shape`, both 1-D, `lower < upper` element-wise, and (if `periodic` is given) matching shape.

## Distributions

```python
from nflojax.distributions import StandardNormal, DiagNormal
```

Both distributions accept an event of arbitrary rank. See
[Event Shape Convention](#event-shape-convention) for the canonical form.

| Distribution | Constructor forms | Params | Description |
|-------------|-------------------|--------|-------------|
| `StandardNormal` | `(event_shape=(N, d))`, `(event_shape=N)`, `(dim=N)`, `(N)` | `None` | Isotropic `N(0, I)` on the event |
| `DiagNormal` | same | `{"loc": event_shape, "log_scale": event_shape}` | Diagonal-covariance Gaussian on the event |

Both provide: `log_prob(params, x)`, `sample(params, key, shape)`, `init_params()`. For `DiagNormal`, `init_params()` returns zero `loc` and `log_scale`, i.e. a standard Gaussian over the event.

**Attributes:**

- `dist.event_shape`: canonical tuple form.
- `dist.dim`: read-only property equal to `event_shape[-1]` (for back-compat with rank-1 error messages and error paths).

**Examples:**

```python
StandardNormal(dim=4).event_shape               # (4,)
StandardNormal(event_shape=4).event_shape       # (4,)
StandardNormal(event_shape=(8, 3)).event_shape  # (8, 3)

dist = DiagNormal(event_shape=(8, 3))
params = dist.init_params()
params["loc"].shape, params["log_scale"].shape  # (8, 3), (8, 3)
samples = dist.sample(params, key, (16,))       # (16, 8, 3)
dist.log_prob(params, samples).shape            # (16,)
```

Both provide: `log_prob(params, x)`, `sample(params, key, shape)`, `init_params()`.

## Utilities

### identity_spline_bias

```python
from nflojax.transforms import identity_spline_bias

bias = identity_spline_bias(
    num_scalars=N_transformed,
    num_bins=K,
    min_derivative=1e-2,
    max_derivative=10.0,
    dtype=jnp.float32,  # default
)
# bias.shape == (num_scalars * (3*K - 1),)
```

**Purpose:** Conditioner-output bias that, together with a zero-kernel final
layer, produces an identity spline per scalar. Used by both `SplineCoupling`
and `SplitCoupling` for near-identity initialization. Re-use this helper if
you write a custom coupling that consumes RQ-spline params and want to
start at identity like the library couplings do.

**Math:** Widths and heights set to zero (uniform bins after softmax).
Derivatives set to `stable_logit((1 - min_d) / (max_d - min_d))`, which
evaluates to derivative = 1 after the bounded-sigmoid. If `1.0` is not
strictly inside `(min_derivative, max_derivative)`, the helper falls back
to an all-zero bias (identity unreachable; caller should warn).

### stable_logit

```python
from nflojax.transforms import stable_logit

u = stable_logit(p)  # logit with input clipped into [1e-6, 1 - 1e-6]
```

Numerically stable logit used internally for bounding knot derivatives.

## Parameter Structure

### Flow params

```python
params = {
    "base": None,            # StandardNormal (no params)
    # or: {"loc": (dim,), "log_scale": (dim,)}  # DiagNormal
    "transform": [
        {"mlp": {...}},      # coupling layer 0
        {"mlp": {...}},      # coupling layer 1
        {},                  # LoftTransform (no params)
    ],
    # "feature_extractor": {...}  # only if extractor enabled
}
```

### Bijection params

```python
params = {
    "transform": [
        {"mlp": {...}},      # coupling layer 0
        {"mlp": {...}},      # coupling layer 1
        {},                  # LoftTransform (no params)
    ],
    # "feature_extractor": {...}  # only if extractor enabled
}
```

Optional blocks when enabled:
- `use_linear=True`: first entry is `{"lower": (d,d), "upper": (d,d), "raw_diag": (d,)}`
- `use_permutation=True`: empty dict `{}` entries between couplings

## Forward/Inverse Convention

| Method | Direction | Log-det sign |
|--------|-----------|--------------|
| `forward(z)` | `z -> x` | `+log|det dx/dz|` |
| `inverse(x)` | `x -> z` | `+log|det dz/dx|` |

`inverse` returns the log-det of the inverse map (negative of the forward log-det).

Density evaluation uses:
```python
log_prob(x) = base.log_prob(z) + log_det_inv   # where z, log_det_inv = inverse(x)
```

Efficient sampling uses:
```python
log_prob(x) = base.log_prob(z) - log_det_fwd   # where x, log_det_fwd = forward(z)
```

## Context Feature Extractor

When `context_extractor_hidden_dim > 0`, a shared ResNet preprocesses raw context before coupling layers see it.

| Parameter | Description |
|-----------|-------------|
| `context_extractor_hidden_dim` | Hidden layer width; 0 disables the extractor |
| `context_extractor_n_layers` | Residual blocks in the extractor (default: 2) |
| `context_feature_dim` | Output dimension; defaults to `context_dim` |

The extractor params live in `params["feature_extractor"]`.
Coupling layers receive the extracted features (dimension = `context_feature_dim`), not the raw context.

When using the assembly API, create the extractor separately and pass the output dimension as `context_dim` to each coupling:

```python
fe, fe_params = create_feature_extractor(key, in_dim=raw_dim, hidden_dim=32, out_dim=8)
# couplings use context_dim=8 (the extractor output dim)
bijection, params = assemble_bijection(blocks, feature_extractor=fe, feature_extractor_params=fe_params)
```

See [USAGE.md](USAGE.md#assembly-with-context-and-feature-extractor) for full examples.
For the math behind conditioning, see [INTERNALS.md](INTERNALS.md#conditional-normalizing-flows).

## Gotchas

**Identity gate constraints.** `identity_gate` requires `context_dim > 0` and is incompatible with `use_permutation=True`. Both constraints are enforced at build time with clear error messages.

**Identity gate single-sample contract.** The gate function receives a single context vector `(context_dim,)`, not a batch. Batching is handled via `jax.vmap`. Writing a batch-aware gate silently produces wrong results. The builders validate this via `jax.eval_shape`.

**Raw context vs extracted features.** When using a feature extractor, the identity gate still receives the raw context, not extracted features. Coupling layers see extracted features. This is intentional: the gate encodes known structure (e.g., boundary conditions) and operates on interpretable inputs.

**Residual scaling defaults to 0.1.** Both MLP and ResNet conditioners use `res_scale=0.1`, scaling residual branch outputs. Most implementations default to 1.0. This improves stability but can make convergence appear slower. Adjustable via the `res_scale` parameter.

**LOFT tau=1000 barely activates.** The default `loft_tau=1000.0` only compresses values beyond magnitude 1000, acting as a gentle safety net. For active tail compression, lower tau significantly.

**SplitCoupling has no mask, so `analyze_mask_coverage` skips it.** The library's coverage check works only on couplings with a 1-D `mask` attribute. A flow built entirely from `SplitCoupling` layers passes `analyze_mask_coverage` trivially (zero coupling-like blocks found), so it's your responsibility to alternate `swap` between layers to make sure every slot along `split_axis` is transformed.

**SplitCoupling `event_shape` is static.** The `event_shape` is a field on the dataclass and baked into the MLP conditioner's `x_dim` / `out_dim`. To evaluate the same weights at a larger event shape (e.g. more particles), rebuild the coupling with the new `event_shape` and transfer params into the new pytree. This only works if the conditioner is size-agnostic by construction (e.g. a GNN); the default MLP is not.

**MLP context validation is symmetric.** Passing `context` when `context_dim=0` raises `ValueError`. Omitting `context` when `context_dim > 0` also raises `ValueError`. Both cases produce clear error messages.

**Zero-initialized output layers.** `init_mlp` always zero-initializes the conditioner output layer. This ensures identity-start flows but can be surprising if reusing the MLP for other purposes.

**Conditioner receives full-dim masked vector.** Coupling layers pass `x * mask` (shape `(dim,)` with zeros in transformed positions) to the MLP, not a reduced vector of only the frozen dimensions. The MLP's input dimension equals `dim`, not the count of frozen dimensions.

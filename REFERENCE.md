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

### build_particle_flow

The canonical DM / bgmat-style topology for particle systems on `(N, d)` events.

```python
from nflojax.builders import build_particle_flow
from nflojax.distributions import UniformBox, LatticeBase
from nflojax.geometry import Geometry
from nflojax.nets import DeepSets

flow_or_bijection, params = build_particle_flow(
    key,
    geometry=Geometry.cubic(d=3, side=2.0, lower=-1.0),
    event_shape=(N, d),
    num_layers=4,
    conditioner=(lambda *, required_out_dim, **_: DeepSets(
        phi_hidden=(64, 64), rho_hidden=(64,), out_dim=required_out_dim,
    )),
    base_dist=UniformBox(geometry=..., event_shape=(N, d)),
    num_bins=8, tail_bound=5.0,
    boundary_slopes="circular",
    use_com_shift=False,
)
```

**Architecture** (for `N_eff = N - 1` if `use_com_shift` else `N`):

```
Rescale(geometry -> [-tail_bound, tail_bound])
for _ in range(num_layers):
    SplitCoupling(swap=False, flatten_input=False)
    SplitCoupling(swap=True,  flatten_input=False)
    CircularShift(canonical cube)
[optional] _CoMEmbed   # base on (N-1, d); embeds to ambient (N, d)
```

`_CoMEmbed` is a private direction-flipped view of `CoMProjection` so that `Flow.sample` (base → data) hits the expansion direction. Particle coverage comes from alternating `swap` on `SplitCoupling`; no per-layer `Permutation` is inserted.

**Conditioner factory contract.** `conditioner` is a **keyword-only callable** invoked once per coupling layer with three kwargs:

| kwarg | value | used by |
|-------|-------|---------|
| `required_out_dim` | `N_transformed * d * params_per_scalar` | flat-output conditioners (`DeepSets`) |
| `out_per_particle` | `d * params_per_scalar` | per-token conditioners (`Transformer`, `GNN`) |
| `n_frozen` | `N_frozen` | user conditioners that size per-particle context |

The factory returns a fresh `flax.linen.Module`. Absorb unused kwargs with `**_`. With asymmetric splits (odd `N_eff` under `use_com_shift=True`), `required_out_dim` differs between `swap=False` and `swap=True` layers — the builder recomputes per-layer. Per-token conditioners require `N_frozen == N_transformed` (even `N_eff`).

**Base distribution.** Mandatory. Must match the inner event shape:

- `use_com_shift=False`: `(N, d)` (e.g. `UniformBox(geometry, event_shape=(N, d))`, `LatticeBase.fcc(...)`).
- `use_com_shift=True`: `(N-1, d)` reduced-subspace base. Remember `CoMProjection.ambient_correction(N, d) = (d/2)·log(N)` if you need an ambient log-density.

**Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `num_bins` | int | 8 | Spline bins K |
| `tail_bound` | float | 5.0 | Spline canonical half-width |
| `boundary_slopes` | str | `'circular'` | Or `'linear_tails'` |
| `use_com_shift` | bool | False | Append `_CoMEmbed` for zero-CoM subspace flows |
| `base_params` | PyTree or None | None | Defaults to `base_dist.init_params()` |
| `return_transform_only` | bool | False | Return `Bijection` instead of `Flow` |

Identity-at-init holds for the learnable part: `SplitCoupling._patch_dense_out` zero-patches every conditioner's `dense_out`, so the couplings are identity on first forward. The only non-zero init log-det contribution is the constant `Rescale` term.

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
| `Rescale` | Fixed per-axis affine from `geometry.box` to a canonical target range |
| `CoMProjection` | Translation-gauge bijection `(N, d) ↔ (N−1, d)`; zero log-det on subspace |
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

**Direct construction** (for non-MLP conditioners such as `DeepSets`, `Transformer`, `GNN`):

```python
coupling = SplitCoupling(
    event_shape=(N, d), split_axis=-2, split_index=N // 2,
    event_ndims=2, conditioner=my_conditioner,
    num_bins=K, flatten_input=False,   # False = conditioner sees (*batch, N_frozen, d)
)
params = coupling.init_params(key)
```

The additional field `flatten_input: bool = True` selects the conditioner input contract:

| `flatten_input` | Conditioner input shape | When to use |
|-----------------|-------------------------|-------------|
| `True` (default) | `(*batch, frozen_flat)` | Flat-input conditioners: `MLP`, any custom Dense-only network. Matches the contract `SplitCoupling.create()` assumes. |
| `False` | `(*batch, *frozen_shape)` (structured) | Permutation-aware conditioners: `DeepSets`, `Transformer`, `GNN`. Required to preserve the particle axis. |

The output-side reshape only depends on total element count, so the conditioner may emit either a flat `(*batch, transformed_flat * (3K - 1))` or a structured `(*batch, *transformed_shape, 3K - 1)` tensor of the same total size — both unflatten correctly.

**Params dict:**

| Key | Shape | Description |
|-----|-------|-------------|
| `"mlp"` | PyTree | Flax params for conditioner (output dim = `transformed_flat * (3K - 1)`) |

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

### Rescale

**What:** Fixed, non-learnable per-axis affine that maps a physical `Geometry` box onto a canonical target range (default `[-1, 1]`). Typically the first layer of a particle flow, so every downstream spline / coupling can assume a fixed domain. For a learnable affine, use `LinearTransform`.

**Forward:**

```
scale_i = (target_upper_i - target_lower_i) / (upper_i - lower_i)
y_i    = target_lower_i + (x_i - lower_i) * scale_i
log_det = event_factor * sum_i log(scale_i)   (scalar)
```

`event_factor = prod(event_shape[:-1])` accounts for any non-coord event axes: `1` for a rank-1 event `(d,)`, `N` for a rank-2 particle event `(N, d)`.

**Inverse:** `x_i = lower_i + (y_i - target_lower_i) / scale_i`; log-det negated.

**Create:**

```python
from nflojax.geometry import Geometry
from nflojax.transforms import Rescale

geom = Geometry(lower=[-L/2, -L/2, -L/2], upper=[L/2, L/2, L/2])

# Default: map geometry.box to [-1, 1] per axis.
rescale, params = Rescale.create(key, geometry=geom)

# Particle event (N, d): pass event_shape so log-det accumulates correctly.
rescale, params = Rescale.create(key, geometry=geom, event_shape=(N, 3))

# Per-axis target bounds.
import jax.numpy as jnp
rescale = Rescale(
    geometry=geom,
    target=(jnp.array([-1, -2, -3]), jnp.array([1, 2, 3])),
)
```

| Param | Type | Default | Meaning |
|-------|------|---------|---------|
| `geometry` | `Geometry` | required | Source box to map from |
| `target` | scalar pair or `(Array, Array)` | `(-1.0, 1.0)` | Destination bounds, uniform or per-axis |
| `event_shape` | `tuple[int, ...]` or `None` | `(geometry.d,)` | Event shape whose last axis is the coord axis |

**Params dict:** `{}` (empty, no learnable parameters).

**Log-det:** scalar (broadcasts over batch). Constant across all samples.

### CoMProjection

**What:** Translation-gauge bijection between a particle configuration `(N, d)` and its reduced `(N−1, d)` space. Removes the centre-of-mass degree of freedom so a flow can sample a `T(d)`-invariant target without learning the trivial translational redundancy.

**Forward (centre + drop last):**

```
mean   = mean(x, axis=event_axis)
y      = (x - mean)[..., :N-1, :]      # drop the last particle along event_axis
log_det = 0                             # see WARNING below
```

**Inverse (reconstruct from zero-CoM constraint):**

```
x_last = -sum(y, axis=event_axis)
x      = concat([y, x_last], axis=event_axis)
log_det = 0
```

**Create:**

```python
from nflojax.transforms import CoMProjection

transform, params = CoMProjection.create(key)          # event_axis=-2 default
transform, params = CoMProjection.create(key, event_axis=-3)  # (B, species, N, d) events
```

| Param | Type | Default | Meaning |
|-------|------|---------|---------|
| `event_axis` | int | `-2` | Negative axis along which particles are stacked. Must be negative and not `-1` (the coord axis). |

**Params dict:** `{}` (empty; non-learnable).

**Log-det shape:** scalar zero.

---

> ### ⚠️  LOG-DET CONVENTION — READ BEFORE USE
>
> `CoMProjection` is **not** a bijection on `R^(Nd)`. It is a bijection between
> the `(N−1, d)` reduced space and the zero-CoM subspace of `R^(Nd)`. These
> spaces have the same intrinsic dimension but are embedded differently in
> ambient `R^(Nd)`.
>
> `nflojax` uses **Convention (1)**: the log-det is the log-det of the linear
> isomorphism viewed *on the reduced space*, which is **zero**. The flow
> composed with this bijection produces a density on the `(N−1, d)` reduced
> space.
>
> **If you need a density on the ambient zero-CoM subspace** (the usual case
> when training reverse-KL against an ambient energy `E(x)` and you care
> about absolute log-density values), add the constant volume-element
> correction:
>
> ```python
> log_q_ambient = flow.log_prob(params, x) + CoMProjection.ambient_correction(N, d)
> #                                             = (d / 2) * log(N)
> ```
>
> **When the constant matters:** see [Bookkeeping
> constants](#bookkeeping-constants) for the unified rule (this primitive
> and `LatticeBase(permute=True)` follow the same convention).
>
> | Use case | Add the correction? |
> |---|---|
> | Gradient-based training loss (reverse-KL, forward-KL) | **No** — constant → zero gradient |
> | Importance weights `w = exp(−β E − log q)` / SNIS / ESS | **Yes** |
> | `logZ` estimates | **Yes** |
> | Absolute density comparisons across models | **Yes** |
> | Ratios of `q` values (same flow, same CoM treatment) | **No** |
>
> **Why not bake the constant into the log-det?** It would silently double-
> count in the augmented-coupling pattern (bgmat-style) where CoM is handled
> by a different mechanism. See [EXTENDING.md — CoM handling](EXTENDING.md#com-handling)
> for when to use `CoMProjection` vs. augmented coupling.
>
> **Math** (for reference): parameterise the zero-CoM subspace by
> `y ∈ R^((N−1)d)` with `x_N = −Σy_i`. The embedding's per-axis Jacobian `J ∈
> R^(N × (N−1))` has `J^T J = I + 11^T`, so `det(J^T J) = 1 + (N−1) = N`. The
> induced ambient volume element is `sqrt(N)` per coord axis, and
> `sqrt(N)^d = N^(d/2)` across `d` axes; hence the `(d/2) · log(N)` correction.
> See also `INTERNALS.md` for the full derivation.

---

**Forward-on-non-zero-CoM is lossy.** If you pass an input whose per-particle-axis mean is not zero, `forward` silently centres it — the original CoM is discarded. This matches how the bijection is used in a flow (inputs arriving at `forward` for log-prob have typically come from `inverse` and are zero-CoM by construction), but be aware when using `CoMProjection` outside a flow pipeline.

**Pairing with base distributions:** A typical particle-flow stack is

```
base on (N-1, d)  →  inner flow on (N-1, d)  →  CoMProjection.inverse  →  x ambient zero-CoM
```

Sampling chain (left-to-right) yields `x` whose particle-axis sum is identically zero. Evaluate log-prob by running `forward` first to drop back to the reduced space.

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

## Conditioners

Reference neural modules for coupling layers. All satisfy the contract in `nflojax/nets.py` (see [DESIGN.md §5.4](DESIGN.md#54-conditioner-contract)): a Flax `nn.Module` with `context_dim: int`, `apply({"params": params}, x, context=None)`, `get_output_layer(params)`, `set_output_layer(params, kernel, bias)`.

Importable names: `MLP`, `ResNet`, `DeepSets`, `Transformer`, `GNN` (modules); `init_mlp`, `init_resnet` (full-featured builders for the flat MLP path kept for the pre-Stage-D builders); and `init_conditioner` (generic helper that calls `module.init` and zeroes `dense_out` for any module satisfying the optional half of the contract). When wiring a conditioner into `SplitCoupling`, you do not need `init_conditioner` — the coupling's `init_params` handles init and identity patching.

| Conditioner | Equivariance | Input contract | Use with |
|-------------|--------------|----------------|----------|
| `MLP` | none (fully connected) | flat `(*batch, x_dim)` | `SplineCoupling`, `AffineCoupling`, or `SplitCoupling(flatten_input=True)` (default) |
| `DeepSets` | permutation-**invariant** | structured `(*batch, N, in_dim)` | `SplitCoupling(flatten_input=False)` |
| `Transformer` | permutation-**equivariant** (per-token output) | structured `(*batch, N, in_dim)` | `SplitCoupling(flatten_input=False)`; the per-token output lines up with the transformed particles only when `N_frozen == N_transformed` (the default half-half split). Asymmetric splits still work, but coupling-level equivariance is lost. |
| `GNN` | permutation-**equivariant** (per-token output) | structured `(*batch, N, in_dim)` | `SplitCoupling(flatten_input=False)` |

### MLP

```python
from nflojax.nets import MLP

mlp = MLP(
    x_dim=frozen_dim,           # required — flat input size
    context_dim=0,              # 0 for unconditional
    hidden_dim=64,
    n_hidden_layers=2,
    out_dim=required_out_dim,   # e.g. transformed_dim * (3K - 1) for spline couplings
)
```

**What:** Residual-block MLP (wraps `ResNet` under `"net"`). The default flat-path conditioner for `AffineCoupling` / `SplineCoupling` / `SplitCoupling(flatten_input=True)`. The coupling's `.create()` constructs it for you; use `MLP(...)` directly when writing a custom coupling or assembling one via `assemble_flow`.

**Context:** must be a single JAX `Array` (or `None`), shape `(context_dim,)` for shared or `(*batch, context_dim)` for per-sample. PyTree contexts require a custom conditioner (see DESIGN.md §5.2).

**Params:** `{"net": {...}}` — a nested dict produced by `init_mlp(key, …)` or `MLP(...).init(key, dummy_x, dummy_context)["params"]`. Zero-initialised `net/dense_out` keeps the flow at identity on init.

**ResNet.** Underlying building block (`nflojax.nets.ResNet`) — rarely constructed directly. Same fields as `MLP` minus `x_dim` / `context_dim` (no context handling, no concatenation). Useful as a head inside a custom equivariant conditioner.

### init_conditioner

```python
from nflojax.nets import init_conditioner

params = init_conditioner(key, conditioner, dummy_x, dummy_context=None)
```

Generic helper: runs `conditioner.init(key, dummy_x, dummy_context)` and zeroes `dense_out`'s kernel + bias. Use for standalone conditioners (no `SplitCoupling` wrapping); inside a coupling, `init_params` already handles init + identity patching so you do not need this.

**Works with any module satisfying the optional half of the conditioner contract** (`get_output_layer` / `set_output_layer`). For the full-featured flat-MLP path there are also `init_mlp` and `init_resnet` that accept `hidden_dim` / `n_hidden_layers` / etc. directly and return `(module, params)` pairs; use those when you want the pre-Stage-D builder ergonomics.

### DeepSets

```python
from nflojax.nets import DeepSets
from nflojax.transforms import SplitCoupling

ds = DeepSets(
    phi_hidden=(64, 64), rho_hidden=(64,),
    out_dim=N_transformed * d * (3 * K - 1),
    context_dim=0,
)
coupling = SplitCoupling(
    event_shape=(N, d), split_axis=-2, split_index=N // 2, event_ndims=2,
    conditioner=ds, num_bins=K, flatten_input=False,
)
params = coupling.init_params(key)

# Or standalone (no coupling): use `init_conditioner` to get a zero'd dense_out.
from nflojax.nets import init_conditioner
ds_params = init_conditioner(key, ds, jnp.zeros((1, N_frozen, d)))
```

**Architecture:** per-particle `phi` MLP → sum-pool over the particle axis → `rho` MLP → `dense_out`. Permutation-invariant because sum is symmetric.

| Field | Type | Meaning |
|-------|------|---------|
| `phi_hidden` | tuple of int | Hidden widths for the shared per-particle stack |
| `rho_hidden` | tuple of int | Hidden widths for the pooled stack (empty tuple skips) |
| `out_dim` | int | Total flat output size (typically `N_transformed · d · (3K − 1)`) |
| `context_dim` | int | 0 for unconditional; context is broadcast across the particle axis |
| `activation` | callable | Default `nn.elu` |

**Params dict:** `{"phi_0": …, "phi_1": …, …, "rho_0": …, …, "dense_out": {"kernel", "bias"}}`.

### Transformer

```python
from nflojax.nets import Transformer
from nflojax.transforms import SplitCoupling

t = Transformer(
    num_layers=4, num_heads=4, embed_dim=64,
    out_per_particle=d * (3 * K - 1),
    ffn_multiplier=4, context_dim=0,
)
coupling = SplitCoupling(
    event_shape=(N, d), split_axis=-2, split_index=N // 2, event_ndims=2,
    conditioner=t, num_bins=K, flatten_input=False,
)
params = coupling.init_params(key)
```

**Architecture (pre-norm, resolved in PLAN.md §10.4):** per-particle `input_proj` → `L × [attn(LN(h)) + ffn(LN(h))]` with residual connections → final `LN` → per-token `dense_out`. Output shape `(*batch, N, out_per_particle)`. Equivariance holds per-token.

| Field | Type | Meaning |
|-------|------|---------|
| `num_layers` | int | Attention/FFN block count |
| `num_heads` | int | Must divide `embed_dim` |
| `embed_dim` | int | Token feature width |
| `out_per_particle` | int | Per-particle output; `d · (3K − 1)` for a spline coupling |
| `ffn_multiplier` | int | FFN hidden width = `ffn_multiplier · embed_dim` (default 4) |
| `context_dim` | int | 0 for unconditional; context projected to `embed_dim` and broadcast-added to every token |

### GNN

```python
from nflojax.nets import GNN
from nflojax.geometry import Geometry
from nflojax.transforms import SplitCoupling

geom = Geometry.cubic(d=3, side=2.0)
gnn = GNN(
    num_layers=3, hidden=64,
    out_per_particle=3 * (3 * K - 1),
    num_neighbours=12, cutoff=None, geometry=geom,
)
coupling = SplitCoupling(
    event_shape=(N, 3), split_axis=-2, split_index=N // 2, event_ndims=2,
    conditioner=gnn, num_bins=K, flatten_input=False,
)
params = coupling.init_params(key)
```

**Architecture:** per-particle `embed` → for each layer, compute the top-`num_neighbours` neighbour list via `nflojax.utils.pbc.pairwise_distance_sq(x, geometry)` and `jax.lax.top_k(-d_sq)` (self-edge pinned to +∞ via `jnp.where`), run a message MLP over `[h_i, h_j, d_ij]`, sum-aggregate, residual node update → per-token `dense_out`. Equivariance holds per-token.

| Field | Type | Default | Meaning |
|-------|------|---------|---------|
| `num_layers` | int | required | Message-passing depth |
| `hidden` | int | required | Node feature width |
| `out_per_particle` | int | required | Per-particle output |
| `num_neighbours` | int | `12` | K nearest neighbours per forward (resolved in PLAN.md §10.5; apps may override) |
| `cutoff` | float \| None | `None` | If set, messages from neighbours with distance ≥ `cutoff` are zero-weighted |
| `geometry` | `Geometry` \| None | `None` | `None` → Euclidean distances; `Geometry` → PBC minimum-image |
| `context_dim` | int | `0` | 0 for unconditional |
| `activation` | callable | `nn.silu` | Activation inside message and update MLPs |

Note: not SE(3)-equivariant — only permutation-equivariant. For E(3)/SE(3), supply a user-side EGNN / NequIP / MACE conditioner (see [DESIGN.md §4 item 7](DESIGN.md)).

## Geometry

```python
from nflojax.geometry import Geometry
```

**What:** Value object for an axis-aligned rectangular box in `R^d` plus per-axis periodicity flags. Consumed by every geometry-aware primitive shipped today: `CircularShift`, `Rescale`, `CoMProjection` (Stage A), `UniformBox`, `LatticeBase` (Stage B), and `utils.pbc.nearest_image` / `pairwise_distance` (Stage B). Numpy-backed, frozen, not a PyTree.

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
from nflojax.distributions import StandardNormal, DiagNormal, UniformBox
```

All distributions accept an event of arbitrary rank. See
[Event Shape Convention](#event-shape-convention) for the canonical form.

| Distribution | Constructor forms | Params | Description |
|-------------|-------------------|--------|-------------|
| `StandardNormal` | `(event_shape=(N, d))`, `(event_shape=N)`, `(dim=N)`, `(N)` | `None` | Isotropic `N(0, I)` on the event |
| `DiagNormal` | same | `{"loc": event_shape, "log_scale": event_shape}` | Diagonal-covariance Gaussian on the event |
| `UniformBox` | `(geometry=..., event_shape=(N, d))` | `None` | Per-axis uniform on `geometry.box`; i.i.d. over leading event axes |
| `LatticeBase` | `.fcc / .diamond / .bcc / .hcp / .hex_ice(n_cells, a, noise_scale, permute=False)` | `None` | Gaussian-perturbed crystalline lattice |

All provide: `log_prob(params, x)`, `sample(params, key, shape)`, `init_params()`. For `DiagNormal`, `init_params()` returns zero `loc` and `log_scale`, i.e. a standard Gaussian over the event. For `UniformBox` and `LatticeBase`, `init_params()` returns `None`.

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

### UniformBox

**What:** Per-axis uniform distribution on a `Geometry` box. Sample space is the hyperrectangle `[lower_j, upper_j]` for each coord axis; for rank-`≥ 2` events (e.g. particle events `(N, d)`), every leading event entry is i.i.d. uniform.

**Constructor:**

```python
from nflojax.geometry import Geometry
from nflojax.distributions import UniformBox

geom = Geometry(lower=[-L/2, -L/2, -L/2], upper=[L/2, L/2, L/2])

# Single-particle d-vector:
dist = UniformBox(geometry=geom, event_shape=(3,))

# N-particle event:
dist = UniformBox(geometry=geom, event_shape=(N, 3))
```

| Param | Type | Default | Meaning |
|-------|------|---------|---------|
| `geometry` | `Geometry` | required | Defines the per-axis box. |
| `event_shape` | `tuple[int, ...]` or `int` | required | Event shape whose last axis is the coord axis (`geometry.d`). |

**log_prob:**

```
in_box        = all(lower <= x <= upper, reduced over every event axis)
log_prob(x)   = -event_factor * sum_j log(box_j)  if in_box
              = -inf                               otherwise

event_factor  = prod(event_shape[:-1])  # 1 for rank-1, N for (N, d), etc.
```

**sample(params, key, shape):** `jax.random.uniform` with per-axis `lower`/`upper`, returning shape `shape + event_shape`. Every position (including particle axes) is an i.i.d. draw.

**Params dict:** `None` (non-learnable).

**Pairing:** `UniformBox` + `Rescale` is the "liquid starter kit": sample uniform in `geometry.box`, flow into a canonical `[-1, 1]` spline range, apply coupling layers. See [USAGE.md — Particle bases](USAGE.md#particle-bases).

### LatticeBase

**What:** Gaussian-perturbed lattice base distribution for crystalline solids. Each particle `i` is drawn from `N(positions[i], noise_scale^2 · I_d)`, independently across particles. Sites come from one of five named factories wrapping `nflojax.utils.lattice` (`fcc`, `diamond`, `bcc`, `hcp`, `hex_ice`).

**Constructor — use a factory.** Direct construction is allowed but the factories handle positions + geometry consistently:

```python
from nflojax.distributions import LatticeBase

lb = LatticeBase.fcc(n_cells=2, a=1.5, noise_scale=0.1)            # 32 atoms
lb = LatticeBase.diamond(n_cells=(3, 3, 3), a=1.0, noise_scale=0.05)
lb = LatticeBase.bcc(n_cells=4, a=1.2, noise_scale=0.08)
lb = LatticeBase.hcp(n_cells=2, a=1.0, noise_scale=0.1)
lb = LatticeBase.hex_ice(n_cells=2, a=1.0, noise_scale=0.1)        # 64 atoms
```

| Arg | Type | Default | Meaning |
|-----|------|---------|---------|
| `n_cells` | `int` or length-3 tuple | required | Cell repeats per axis. |
| `a` | `float` | required | Lattice constant. |
| `noise_scale` | `float` | required | Per-coord Gaussian σ for the per-site jitter. |
| `permute` | `bool` | `False` | Indistinguishability switch — see below. |

**log_prob:**

```
z         = (x - positions) / noise_scale         # shape (..., N, d)
log_p     = -1/2 * sum(z^2)                       # over all event axes
            -1/2 * N*d * log(2π) - N*d * log(σ)   # Gaussian normaliser
            (-log(N!)  if permute=True)           # indistinguishability constant
```

**sample(params, key, shape):**

```
eps  = jax.random.normal(key, shape + (N, d))
x    = positions + noise_scale * eps
        (per-batch random shuffle along the particle axis if permute=True)
```

**Params dict:** `None` (non-learnable; positions and noise_scale are static configuration).

---

> ### `permute=True` — what it does
>
> - **Sampling:** the particle axis is shuffled per batch sample (so the
>   user cannot infer which lattice site each output came from).
> - **Density:** subtracts `log(N!)` from the labelled-Gaussian density.
>   This is the standard distinguishable → indistinguishable correction.
>
> The labelled Gaussian itself is not exactly invariant under permuting
> `x`'s particle axis; the `-log(N!)` is the correct indistinguishability
> constant only when one permutation dominates the likelihood (the small-
> noise / well-separated-sites regime). For larger `noise_scale` relative
> to lattice spacing, a permutation-marginalised density would differ.
>
> **The `-log(N!)` is a constant**, so it doesn't affect gradients of a
> training loss. It does affect absolute densities, ESS, `logZ`. See
> [Bookkeeping constants](#bookkeeping-constants) for the unified rule
> (this and `CoMProjection.ambient_correction` follow the same convention).

---

**Pairing for a Boltzmann generator on a crystalline solid:**

```
LatticeBase  ->  Rescale (lattice box -> [-1, 1])  ->  inner couplings  ->  CoMProjection.inverse  ->  ambient
```

- The lattice base lives on `(N, d)` reduced coordinates (no CoM removal yet).
- `Rescale` puts coords into the canonical spline range.
- Couplings learn the residual deformation.
- `CoMProjection.inverse` (Stage A) embeds back into the zero-CoM ambient
  subspace, ready to score against an ambient `E(x)`.

For `T(d)`-invariant targets you typically want `permute=True`; for
distinguishable-particle targets (toy benchmarks) leave it `False`.

## Bookkeeping constants

Two primitives in nflojax expose a constant log-density correction that
**the caller is responsible for applying when needed**. Both follow the
same rule: they are no-ops for gradient-based training (zero gradient) but
load-bearing for absolute densities, importance weights, ESS, `logZ`, and
any quantitative cross-model comparison.

| Primitive | Constant | API | When it matters |
|-----------|----------|-----|-----------------|
| [`CoMProjection`](#comprojection) | `(d/2) · log(N)` | `CoMProjection.ambient_correction(N, d)` | Ambient-subspace density vs reduced-space density. Required for reverse-KL against ambient `E(x)` when reporting absolute log-prob. |
| [`LatticeBase`](#latticebase) (`permute=True`) | `-log(N!)` | Built into `log_prob` when `permute=True`; *not* added when `permute=False` | Distinguishable → indistinguishable particle convention. Required for ESS / `logZ` against an indistinguishable target. |

**Rule of thumb:**

| Use case | Apply correction? |
|----------|-------------------|
| Gradient-based training loss (reverse-KL, forward-KL) | **No** — constant → zero gradient |
| Importance weights `w = exp(−β E − log q)` / SNIS / ESS | **Yes** |
| `logZ` estimates | **Yes** |
| Absolute density comparisons across models | **Yes** (apply consistently to every model compared) |
| Ratios within one flow | **No** — cancels |

**Do not stack patterns that solve the same problem twice.** If a flow
already removes translations via the augmented-coupling pattern (see
[EXTENDING.md — CoM handling](EXTENDING.md#com-handling)), do *not* also
apply `CoMProjection.ambient_correction`. Likewise if your inner flow
already permutation-symmetrises samples, do not also subtract `log(N!)`
via `LatticeBase(permute=True)`.

When a future primitive adds a third such constant (most likely candidate:
a unit-of-volume rescale for non-orthogonal cells), document it here and
update the rule table; this is the canonical place to look.

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

### nflojax.utils.pbc

```python
from nflojax.utils.pbc import (
    nearest_image,
    pairwise_distance,
    pairwise_distance_sq,
)
```

Pure functions for orthogonal periodic-box geometry. All consume the
`Geometry` value object; non-periodic axes (per `geometry.periodic`) pass
through unchanged. Triclinic cells are out of scope (DESIGN.md §4.8).

| Function | Signature | Purpose |
|----------|-----------|---------|
| `nearest_image` | `(dx, geometry) -> Array` | Minimum-image wrap of displacement `dx` of shape `(..., d)`. Returns the same shape, with each periodic axis wrapped to `(-box[j]/2, box[j]/2]` and non-periodic axes unchanged. |
| `pairwise_distance` | `(x, geometry=None) -> Array` | Pairwise Euclidean distances for `(..., N, d)` positions; output shape `(..., N, N)`. With `geometry`, wraps displacements via `nearest_image` first. |
| `pairwise_distance_sq` | `(x, geometry=None) -> Array` | Squared version. Avoids `sqrt`; preferred in hot paths (neighbour-list cutoffs, network features). |

**Example:**

```python
from nflojax.geometry import Geometry
from nflojax.utils.pbc import nearest_image, pairwise_distance

geom = Geometry.cubic(d=3, side=1.0, lower=0.0)        # [0, 1]^3, all-periodic
dx   = jnp.array([0.8, 0.8, 0.8])
nearest_image(dx, geom)                                # [-0.2, -0.2, -0.2]

# Slab: periodic in x, y; open in z.
geom = Geometry(lower=[0,0,0], upper=[1,1,1], periodic=[True, True, False])
nearest_image(jnp.array([0.8, 0.8, 0.8]), geom)        # [-0.2, -0.2, 0.8]

# Pairwise PBC distances on (B, N, d) positions.
x  = jax.random.uniform(key, (4, 8, 3))
d  = pairwise_distance(x, geometry=geom)               # (4, 8, 8); diag = 0
```

**Not for:** energy evaluation, neighbour lists with physics-specific
cutoffs (force-field range), or anything that pulls in domain knowledge.
Those live in application code (DESIGN.md §4 item 1).

### nflojax.utils.lattice

```python
from nflojax.utils.lattice import (
    fcc, diamond, bcc, hcp, hex_ice,
    make_box, cell_aspect, ATOMS_PER_CELL,
)
```

Pure functions returning `(N, 3)` numpy arrays of crystalline lattice
positions. Used by `LatticeBase` to define the mean sites of a
Gaussian-perturbed lattice base distribution; can also be called standalone
for visualisation, ground-truth comparisons, or non-flow physics code.

| Lattice | Atoms / cell | Cell aspect | Notes |
|---------|--------------|-------------|-------|
| `fcc(n_cells, a)` | 4 | `(1, 1, 1)` | Face-centred cubic |
| `diamond(n_cells, a)` | 8 | `(1, 1, 1)` | Diamond cubic; first 4 atoms = FCC sub-lattice |
| `bcc(n_cells, a)` | 2 | `(1, 1, 1)` | Body-centred cubic |
| `hcp(n_cells, a)` | 4 | `(1, √3, √(8/3))` | Orthorhombic representation; ideal `c/a = √(8/3)` |
| `hex_ice(n_cells, a)` | 8 | `(1, √3, √(8/3))` | Ice Ih, **DM convention** with the puckering parameter `6 × 0.0625` baked in (matches `flows_for_atomic_solids`) |

`n_cells` is `int` (uniform per axis) or a length-3 sequence; `a` is the
lattice constant (scalar). Box side along axis `i` is `n_cells[i] * a *
cell_aspect[i]`; the box origin is at zero by convention.

```python
import numpy as np
from nflojax.geometry import Geometry
from nflojax.utils.lattice import fcc, make_box, cell_aspect

# 2x2x2 FCC, lattice constant 1.5 -> 32 atoms in a (3, 3, 3) box.
positions = fcc(n_cells=2, a=1.5)
box       = make_box(n_cells=2, a=1.5, cell_aspect=cell_aspect("fcc"))
geom      = Geometry(lower=np.zeros(3), upper=box)
```

`ATOMS_PER_CELL` is a `{name: int}` dict for callers that need to invert
the atom-count formula `N = prod(n_cells) * atoms_per_cell`.

**Hex-ice convention.** PLAN.md §10.3 was resolved in favour of the DM
convention. If you need bgmat's re-derivation instead, build a custom
generator following the `_make_lattice` recipe in `nflojax/utils/lattice.py`.

**Not for:** lattice-specific physics (Madelung sums, defect generation),
non-orthorhombic cells (DESIGN.md §4.8).

### nflojax.embeddings

```python
from nflojax.embeddings import circular_embed, positional_embed
```

Stateless feature transforms used by Stage D conditioners (Transformer,
GNN) and any user-supplied conditioner that wants ready-made periodic /
scalar features. No learnable parameters; no random state.

| Function | Signature | Purpose |
|----------|-----------|---------|
| `circular_embed(x, geometry, n_freq)` | `(..., d), Geometry, int -> (..., d * 2 * n_freq)` | Per-coord Fourier features on a periodic box. The lowest harmonic exactly tiles `geometry.box`. |
| `positional_embed(t, n_freq, base=10_000)` | `(...,), int, float -> (..., 2 * n_freq)` | Sinusoidal scalar embedding (transformer-style, continuous `t`). For temperature, density, MD step. |

`n_freq` must be `>= 1` for both (zero-frequency would silently produce a
zero-width last axis and break downstream `jnp.concatenate`).

```python
import jax.numpy as jnp
from nflojax.geometry import Geometry
from nflojax.embeddings import circular_embed, positional_embed

# Periodic-box features: 32 atoms in a [-1, 1]^3 cubic box, 4 harmonics.
geom = Geometry.cubic(d=3, side=2.0, lower=-1.0)
x    = jax.random.uniform(key, (B, 32, 3), minval=-1, maxval=1)
feat = circular_embed(x, geom, n_freq=4)             # (B, 32, 24)

# Scalar context features: temperature ~ U(0.5, 2.0).
t    = jax.random.uniform(key, (B,), minval=0.5, maxval=2.0)
ctx  = positional_embed(t, n_freq=8)                  # (B, 16)

# Concatenate to form a custom conditioner input.
inp = jnp.concatenate(
    [feat, jnp.broadcast_to(ctx[:, None, :], feat.shape[:-1] + ctx.shape[-1:])],
    axis=-1,
)                                                      # (B, 32, 40)
```

**Math** — `circular_embed`:

```
phase[..., j, k] = 2 * pi * (k + 1) * (x[..., j] - lower[j]) / box[j]
out[..., j, 2k]     = cos(phase[..., j, k])
out[..., j, 2k + 1] = sin(phase[..., j, k])
```

then flatten the trailing `(d, 2 * n_freq)` axes. Periodic in each coord
axis with period `geometry.box[j]`.

**Math** — `positional_embed` (matches "Attention Is All You Need" with
continuous `t`):

```
freq[k]            = base ** (-k / n_freq)         # k in [0, n_freq)
out[..., 2k]       = cos(t * freq[k])
out[..., 2k + 1]   = sin(t * freq[k])
```

**Non-periodic axes.** `circular_embed` does not gate on
`geometry.periodic`. The math is well-defined for any `box[j]`, but using
`circular_embed` on an axis that's *not* periodic in your physical setup
is the caller's mistake — no warning is raised. (This may grow a
`mask_non_periodic` knob post-v1 if a real use case appears.)

**Not for:** learnable embeddings (those go into a Flax `nn.Module` in
`nets.py` per DESIGN.md §8); graph / edge features (the GNN computes
those directly).

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

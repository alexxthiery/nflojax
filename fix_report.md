# Doc Audit Fix Report

Systematic audit of REFERENCE.md and USAGE.md against the nflojax implementation.
Three parallel agents inspected transforms, builders/flows/distributions, and usage examples/internals.

## Files modified

- `REFERENCE.md`: 8 edits (discrepancy fixes + new Gotchas section)
- `nflojax/builders.py`: 2 edits (docstring fixes)
- `nflojax/nets.py`: 1 edit (missing context validation)
- `tests/test_nets.py`: 1 test added

## Discrepancies fixed

### D1. `create_feature_extractor` missing params

REFERENCE.md showed `create_feature_extractor(key, in_dim, hidden_dim, out_dim, n_layers=2)`.
The actual signature also accepts `activation=jax.nn.tanh` and `res_scale=0.1`.
Added both to the docs.

### D2. LinearTransform gating formula was wrong

Docs said `W -> g*W + (1-g)*I`.
The implementation gates LU factors component-wise: `L_off -> g*L_off`, `U_off -> g*U_off`, `s -> 1-g+g*s`.
The resulting matrix `L_gated @ T_gated` differs from `g*W + (1-g)*I` due to cross-terms in the LU product.
Both share the key property (g=0 gives identity, g=1 gives full W), but the interpolation path is different.
Replaced the formula with the accurate component-wise description.

### D3. `base_params` docstring in builders was wrong

Both `build_realnvp` and `build_spline_realnvp` docstrings said "If None and base_dist is provided, defaults to `{}`".
The code actually calls `base_dist.init_params()`.
Fixed both docstrings.

### D4. `analyze_mask_coverage` was undocumented

Public function in `builders.py`, called automatically by `assemble_bijection`, `assemble_flow`, and the builders.
Added to the Utilities section with usage example and description.

### D5. Permutation `g_value` interface claim

Docs said "all transforms" accept `g_value`.
Permutation's `forward`/`inverse` don't accept `g_value` (no learnable components to gate).
`CompositeTransform` handles this via `_block_supports_gvalue`, skipping `g_value` for Permutation.
Updated the identity gating paragraph and CompositeTransform section to reflect this.

### D6. Spline defaults mismatch between layers

The low-level `rational_quadratic_spline` in `splines.py` defaults to `min_bin_width=1e-3` and `min_bin_height=1e-3`.
`SplineCoupling.create` and `build_spline_realnvp` default to `1e-2`.
Added a note to the SplineCoupling section.

### D7. SplineCoupling gating details understated

Docs said "spline params interpolate toward identity".
The actual implementation scales widths/heights by `g` and interpolates derivatives in logit space toward derivative=1.
Added the precise description.

### D8. LOFT inverse details missing

Added: 10 hardcoded Newton iterations (no convergence check), exponent clamped to 80.0 for float32 safety.

### D9. CompositeTransform float64 accumulation undocumented

Added: log_det accumulated in float64 when `jax_enable_x64` is set, then cast back to input dtype.

## Gotchas section added to REFERENCE.md

Eight items covering common pitfalls:

1. **Identity gate constraints** -- requires `context_dim > 0`, incompatible with `use_permutation=True`
2. **Identity gate single-sample contract** -- gate receives `(context_dim,)`, not batched
3. **Raw context vs extracted features** -- gate sees raw context, couplings see extracted features
4. **Residual scaling defaults to 0.1** -- non-standard, adjustable via `res_scale`
5. **LOFT tau=1000 barely activates** -- gentle safety net, lower for active compression
6. **MLP context validation** -- now symmetric (see implementation fix below)
7. **Zero-initialized output layers** -- ensures identity-start, surprising for reuse
8. **Conditioner receives full-dim masked vector** -- shape `(dim,)` with zeros, not reduced

## Confirmed correct (no changes needed)

- All Flow/Bijection constructor args and method signatures
- All builder option names and defaults
- All distribution classes
- Assembly API signatures
- Forward/inverse convention and log_prob formulas
- All USAGE.md code examples
- Feature extractor `zero_init_output` consistency between builder and `create_feature_extractor`

## Implementation fix: MLP missing context validation

The MLP validated `context` passed when `context_dim=0` (raises `ValueError`) but silently accepted `context=None` when `context_dim > 0`. A user calling a conditional flow without passing context would get a cryptic JAX shape mismatch deep in a Dense matmul.

Investigation confirmed no internal code path relies on the silent `None` behavior. The LinearTransform conditioner passes raw context in the MLP's `x` slot with `context_dim=0`, so the new check does not affect it.

**Fix:** Added symmetric validation in `MLP.__call__`: raises `ValueError` with a clear message when `context is None and context_dim > 0`. Added corresponding test. 342/342 tests pass.

## False alarm investigated

The transforms agent flagged that LinearTransform's batched-gate path computes `W @ x` while the unbatched path computes `x @ W^T`.
Manual verification confirmed these are equivalent for 1D vectors: `(W @ x)_j = sum_k W_jk * x_k = (x @ W^T)_j`.
No bug.

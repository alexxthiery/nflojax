# nflojax/nets.py
"""
Neural network modules for normalizing flow conditioners.

Conditioner Interface
---------------------
Conditioners used with coupling layers (`AffineCoupling`, `SplineCoupling`,
`SplitCoupling`) should implement the following interface:

Required:
    - ``apply({"params": params}, x, context) -> output`` (Flax convention)
    - ``context_dim: int`` attribute (0 for unconditional)

Optional (enables automatic output layer initialization):
    - ``get_output_layer(params) -> {"kernel": Array, "bias": Array}``
    - ``set_output_layer(params, kernel, bias) -> params``

If the optional methods are present, coupling layers will automatically
zero-initialize (`AffineCoupling`) or identity-initialize (`SplineCoupling`,
`SplitCoupling`) the output layer for stable training. If missing,
`AffineCoupling` skips auto-init; the spline couplings raise an error.

Identity init infers the bias length from the conditioner's `dense_out`
and sizes `identity_spline_bias` to match, so both flat-output conditioners
(`MLP`, bias of size `num_scalars · params_per_scalar`) and per-token
conditioners (`Transformer`, `GNN`, bias of size `d · params_per_scalar`)
plug in with the same trivial `set_output_layer`.

The built-in `MLP` + `ResNet` (flat) and `DeepSets` / `Transformer` / `GNN`
(structured, use with `SplitCoupling(flatten_input=False)`) all implement
the full interface.
"""
from __future__ import annotations

from typing import Callable, Sequence, Tuple

import jax
import jax.numpy as jnp
from flax import linen as nn

from .geometry import Geometry
from .utils.pbc import pairwise_distance_sq


Array = jnp.ndarray
PRNGKey = jax.Array  # type alias for JAX random keys


# Module-level helper used by GNN to gather per-particle neighbour features
# without vmapping the caller. `signature="(n,h),(n,k)->(n,k,h)"` makes it
# batch-polymorphic over any leading axes.
_gather_neighbours = jnp.vectorize(
    lambda h_row, i_row: jnp.take(h_row, i_row, axis=0),
    signature="(n,h),(n,k)->(n,k,h)",
)


def validate_conditioner(conditioner, name: str = "conditioner") -> None:
    """
    Validate that a conditioner meets the required interface.

    Coupling layers (AffineCoupling, SplineCoupling) require conditioners with:
      - ``context_dim: int`` attribute (0 for unconditional)
      - ``apply({"params": params}, x, context) -> output`` method (Flax convention)

    This function raises helpful errors if the conditioner doesn't meet the contract.

    Arguments:
        conditioner: The conditioner object to validate.
        name: Name to use in error messages (default: "conditioner").

    Raises:
        TypeError: If conditioner is missing required attributes or methods.
    """
    if not hasattr(conditioner, "context_dim"):
        raise TypeError(
            f"{name} must have a 'context_dim' attribute (int, 0 for unconditional). "
            f"See nets.py docstring for the conditioner interface."
        )

    if not hasattr(conditioner, "apply"):
        raise TypeError(
            f"{name} must have an 'apply' method (Flax convention). "
            f"See nets.py docstring for the conditioner interface."
        )


class ResNet(nn.Module):
    """
    Residual MLP block: input → hidden → residual blocks → output.

    A reusable residual network architecture that can be used standalone
    or as a building block within other modules (e.g., MLP, feature extractors).

    Architecture:
      1. Input projection: x → Dense(hidden_dim) → h
      2. Residual trunk: h = h + res_scale * F(h) for each layer
         where F = Dense → activation → Dense
      3. Output projection: Dense(out_dim)

    Attributes:
        hidden_dim: Width of residual hidden stream.
        out_dim: Output dimensionality.
        n_hidden_layers: Number of residual blocks.
        activation: Activation function (default: elu).
        res_scale: Scale applied to residual updates (default: 0.1).
    """

    hidden_dim: int
    out_dim: int
    n_hidden_layers: int = 2
    activation: Callable[[Array], Array] = nn.elu
    res_scale: float = 0.1

    @nn.compact
    def __call__(self, x: Array) -> Array:
        """
        Forward pass through the residual network.

        Arguments:
            x: Input tensor of shape (..., in_dim). Input dimension is inferred.

        Returns:
            Output tensor of shape (..., out_dim).
        """
        # 1. Input projection.
        h = nn.Dense(self.hidden_dim, name="dense_in")(x)

        # 2. Residual trunk.
        for i in range(self.n_hidden_layers):
            r = nn.Dense(self.hidden_dim, name=f"res_{i}_dense0")(h)
            r = self.activation(r)
            r = nn.Dense(self.hidden_dim, name=f"res_{i}_dense1")(r)
            h = h + self.res_scale * r

        # 3. Output projection.
        out = nn.Dense(self.out_dim, name="dense_out")(h)
        return out

    def get_output_layer(self, params: dict) -> dict:
        """
        Return output layer parameters.

        Arguments:
            params: Parameter dict for this ResNet.

        Returns:
            Dict with "kernel" and "bias" arrays for the output layer.
        """
        return params["dense_out"]

    def set_output_layer(self, params: dict, kernel: Array, bias: Array) -> dict:
        """
        Return new params with output layer replaced.

        Arguments:
            params: Parameter dict for this ResNet.
            kernel: New kernel array for output layer.
            bias: New bias array for output layer.

        Returns:
            New parameter dict (original unchanged).
        """
        new_params = dict(params)
        new_params["dense_out"] = {"kernel": kernel, "bias": bias}
        return new_params


class MLP(nn.Module):
    """
    Residual MLP conditioner for flow layers.

    Wraps a ResNet with context handling (validation, broadcasting, concatenation).
    The underlying ResNet is stored under the "net" submodule.

    The final layer ("net/dense_out") should be zero-initialized externally
    (via init_mlp) to start the flow at identity.

    Attributes:
        x_dim: Dimensionality of input x.
        context_dim: Dimensionality of context (0 for unconditional).
        hidden_dim: Width of residual hidden stream.
        n_hidden_layers: Number of residual blocks.
        out_dim: Output dimensionality (e.g., coupling parameters).
        activation: Activation function (default: elu).
        res_scale: Scale applied to residual updates (default: 0.1).
    """
    x_dim: int
    context_dim: int = 0
    hidden_dim: int = 64
    n_hidden_layers: int = 2
    out_dim: int = 1
    activation: Callable[[Array], Array] = nn.elu
    res_scale: float = 0.1

    @nn.compact
    def __call__(self, x: Array, context: Array | None = None) -> Array:
        """
        Forward pass through the residual MLP.

        Arguments:
            x: Input tensor of shape (..., x_dim).
            context: Optional context tensor. Must be a single JAX `Array`
                (or `None`); the MLP concatenates it with `x` and does not
                flatten nested PyTrees. For structured (PyTree) contexts,
                supply a custom conditioner instead — the flow threads
                context through without inspection. See DESIGN.md §5.2.
                - None (unconditional)
                - shape (context_dim,) for shared context
                - shape (..., context_dim) for per-sample context

        Returns:
            Output tensor of shape (..., out_dim).
        """
        # Shape check for x.
        if x.shape[-1] != self.x_dim:
            raise ValueError(
                f"MLP expected x with last dimension {self.x_dim}, got {x.shape[-1]}."
            )

        # Reject context when context_dim=0 (silent misconfiguration).
        if context is not None and self.context_dim == 0:
            raise ValueError(
                "context was passed but context_dim=0. "
                "Set context_dim to match your context features."
            )

        # Reject missing context when context_dim>0 (silent misconfiguration).
        if context is None and self.context_dim > 0:
            raise ValueError(
                f"context_dim={self.context_dim} but context was not passed. "
                "Pass context to all flow methods when using a conditional flow."
            )

        # Context handling.
        if context is not None and self.context_dim > 0:
            # Check context feature dimension.
            if context.shape[-1] != self.context_dim:
                raise ValueError(
                    f"MLP expected context with last dimension {self.context_dim}, "
                    f"got {context.shape[-1]}."
                )
            # Broadcast shared context to batch dimensions.
            if context.ndim == 1:
                context = jnp.broadcast_to(context, x.shape[:-1] + (self.context_dim,))
            elif context.shape[:-1] != x.shape[:-1]:
                raise ValueError(
                    f"Context batch shape {context.shape[:-1]} doesn't match "
                    f"x batch shape {x.shape[:-1]}."
                )
            inp = jnp.concatenate([x, context], axis=-1)
        else:
            inp = x

        # Delegate to ResNet for the actual computation.
        net = ResNet(
            hidden_dim=self.hidden_dim,
            out_dim=self.out_dim,
            n_hidden_layers=self.n_hidden_layers,
            activation=self.activation,
            res_scale=self.res_scale,
            name="net",
        )
        return net(inp)

    def get_output_layer(self, params: dict) -> dict:
        """
        Return output layer parameters.

        Arguments:
            params: Parameter dict for this MLP.

        Returns:
            Dict with "kernel" and "bias" arrays for the output layer.
        """
        return params["net"]["dense_out"]

    def set_output_layer(self, params: dict, kernel: Array, bias: Array) -> dict:
        """
        Return new params with output layer replaced.

        Arguments:
            params: Parameter dict for this MLP.
            kernel: New kernel array for output layer.
            bias: New bias array for output layer.

        Returns:
            New parameter dict (original unchanged).
        """
        new_net = dict(params["net"])
        new_net["dense_out"] = {"kernel": kernel, "bias": bias}
        new_params = dict(params)
        new_params["net"] = new_net
        return new_params


def init_mlp(
    key: PRNGKey,
    x_dim: int,
    context_dim: int,
    hidden_dim: int,
    n_hidden_layers: int,
    out_dim: int,
    activation: Callable[[Array], Array] = nn.elu,
    res_scale: float = 0.1,
) -> Tuple[MLP, dict]:
    """
    Construct a residual MLP module and initialize its parameters.

    Arguments:
        key: JAX PRNGKey used for parameter initialization.
        x_dim: Dimensionality of input x.
        context_dim: Dimensionality of context (0 for unconditional).
        hidden_dim: Width of residual hidden stream.
        n_hidden_layers: Number of residual blocks.
        out_dim: Output dimensionality.
        activation: Activation function (default: elu).
        res_scale: Scale applied to residual updates (default: 0.1).

    Returns:
        mlp: A Flax MLP module (definition only, no params inside).
        params: A PyTree of parameters for this MLP (suitable for mlp.apply).

    Notes:
        The final linear layer ("dense_out") is explicitly zero-initialized
        (kernel and bias set to zero). This makes the initial output of the
        conditioner identically zero, so any flow that interprets the output
        as shift/log_scale starts exactly at the identity transform.
    """
    mlp = MLP(
        x_dim=x_dim,
        context_dim=context_dim,
        hidden_dim=hidden_dim,
        n_hidden_layers=n_hidden_layers,
        out_dim=out_dim,
        activation=activation,
        res_scale=res_scale,
    )

    # Dummy inputs for initialization.
    B = 1
    dummy_x = jnp.zeros((B, x_dim), dtype=jnp.float32)
    dummy_context = jnp.zeros((B, context_dim), dtype=jnp.float32) if context_dim > 0 else None
    variables = mlp.init(key, dummy_x, dummy_context)

    params = variables.get("params", {})

    # Zero-initialize the output layer for identity-start flows.
    out_layer = mlp.get_output_layer(params)
    kernel = jnp.zeros_like(out_layer["kernel"])
    bias = jnp.zeros_like(out_layer["bias"])
    params = mlp.set_output_layer(params, kernel, bias)

    return mlp, params


def init_resnet(
    key: PRNGKey,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    n_hidden_layers: int = 2,
    activation: Callable[[Array], Array] = nn.elu,
    res_scale: float = 0.1,
    zero_init_output: bool = False,
) -> Tuple[ResNet, dict]:
    """
    Construct a ResNet module and initialize its parameters.

    Arguments:
        key: JAX PRNGKey used for parameter initialization.
        in_dim: Dimensionality of input (used for dummy input during init).
        hidden_dim: Width of residual hidden stream.
        out_dim: Output dimensionality.
        n_hidden_layers: Number of residual blocks (default: 2).
        activation: Activation function (default: elu).
        res_scale: Scale applied to residual updates (default: 0.1).
        zero_init_output: If True, zero-initialize the output layer (default: False).

    Returns:
        resnet: A Flax ResNet module (definition only, no params inside).
        params: A PyTree of parameters for this ResNet (suitable for resnet.apply).
    """
    resnet = ResNet(
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        n_hidden_layers=n_hidden_layers,
        activation=activation,
        res_scale=res_scale,
    )

    # Dummy input for initialization.
    dummy_x = jnp.zeros((1, in_dim), dtype=jnp.float32)
    variables = resnet.init(key, dummy_x)

    params = variables.get("params", {})

    # Optionally zero-initialize the output layer.
    if zero_init_output:
        out_layer = resnet.get_output_layer(params)
        kernel = jnp.zeros_like(out_layer["kernel"])
        bias = jnp.zeros_like(out_layer["bias"])
        params = resnet.set_output_layer(params, kernel, bias)

    return resnet, params


# ===================================================================
# DeepSets: permutation-invariant conditioner for particle-axis events
# ===================================================================
# Classic Deep Sets (Zaheer et al., 2017). phi is a per-particle MLP
# applied with shared weights over the particle axis; sum-pool over
# particles yields a permutation-invariant feature; rho then maps it to
# the flow-level output. Intended for use with SplitCoupling when
# flatten_input=False so the particle axis is visible to phi.
class DeepSets(nn.Module):
    """
    Permutation-invariant conditioner for particle-axis events.

    Architecture:
      1. (optional) Broadcast `context` across the particle axis and
         concatenate into each per-particle feature vector.
      2. phi: a Dense stack applied per-particle with shared weights.
      3. Sum-pool over the particle axis.
      4. rho: a Dense stack on the pooled feature.
      5. dense_out: final Dense(out_dim); the layer that
         `SplitCoupling.init_params` zeroes and patches with
         `identity_spline_bias`.

    Output is permutation-invariant with respect to the input particle
    axis because step 3 is a commutative reduction.

    Input shape:  `(*batch, N, in_dim)` when used with
      `SplitCoupling(flatten_input=False)` on a `(*batch, N, d)` event.
      The per-particle feature width is inferred from `x.shape[-1]` on
      first call; there is no `in_dim` constructor argument.

    Output shape: `(*batch, out_dim)`, where `out_dim` is a flat size
      chosen by the caller (typically `N_transformed * d * (3K - 1)`
      for a spline coupling with K bins).

    Attributes:
      phi_hidden: Hidden widths for the per-particle stack. The last
        entry is the per-particle feature width fed to the sum-pool.
      rho_hidden: Hidden widths for the pooled stack. Empty tuple
        skips rho and feeds the pooled features directly to
        `dense_out`.
      out_dim: Total flat output size.
      context_dim: Context width (0 for unconditional). Context is
        broadcast across the particle axis.
      activation: Activation between Dense layers (default: elu).
    """
    phi_hidden: Sequence[int]
    rho_hidden: Sequence[int]
    out_dim: int
    context_dim: int = 0
    activation: Callable[[Array], Array] = nn.elu

    @nn.compact
    def __call__(self, x: Array, context: Array | None = None) -> Array:
        if x.ndim < 2:
            raise ValueError(
                f"DeepSets expects input with rank >= 2 "
                f"(a particle axis is required); got shape {x.shape}."
            )

        if context is not None and self.context_dim == 0:
            raise ValueError(
                "context was passed but context_dim=0. "
                "Set context_dim to match your context features."
            )
        if context is None and self.context_dim > 0:
            raise ValueError(
                f"context_dim={self.context_dim} but context was not passed."
            )
        if context is not None and self.context_dim > 0:
            if context.shape[-1] != self.context_dim:
                raise ValueError(
                    f"DeepSets expected context with last dim {self.context_dim}, "
                    f"got {context.shape[-1]}."
                )
            # Broadcast across the particle axis so phi sees it per-token.
            # x is (*batch, N, in_dim); target context shape: (*batch, N, context_dim).
            target_shape = x.shape[:-1] + (self.context_dim,)
            ctx = jnp.broadcast_to(context[..., None, :], target_shape) \
                if context.ndim == x.ndim - 1 \
                else jnp.broadcast_to(context, target_shape)
            h = jnp.concatenate([x, ctx], axis=-1)
        else:
            h = x

        # phi: per-particle stack with shared weights over the particle axis.
        for i, width in enumerate(self.phi_hidden):
            h = nn.Dense(width, name=f"phi_{i}")(h)
            h = self.activation(h)

        # Permutation-invariant pooling.
        h = jnp.sum(h, axis=-2)

        # rho: stack on the pooled feature.
        for i, width in enumerate(self.rho_hidden):
            h = nn.Dense(width, name=f"rho_{i}")(h)
            h = self.activation(h)

        return nn.Dense(self.out_dim, name="dense_out")(h)

    def get_output_layer(self, params: dict) -> dict:
        return params["dense_out"]

    def set_output_layer(self, params: dict, kernel: Array, bias: Array) -> dict:
        new_params = dict(params)
        new_params["dense_out"] = {"kernel": kernel, "bias": bias}
        return new_params


def init_conditioner(
    key: PRNGKey,
    conditioner,
    dummy_x: Array,
    dummy_context: Array | None = None,
) -> dict:
    """Init a conditioner and zero its `dense_out` for an identity-start flow.

    Works with any conditioner implementing the optional half of the
    conditioner contract (`get_output_layer` / `set_output_layer`). For a
    conditioner plugged into `SplitCoupling` you do not need this — the
    coupling's `init_params` handles init and identity patching on its own.
    Use this helper when you want a pre-initialised conditioner standalone,
    or when writing a test that exercises the conditioner outside a coupling.

    Arguments:
        key: PRNG key.
        conditioner: Flax `nn.Module` satisfying the conditioner contract.
        dummy_x: Example input tensor; its leading axes may be any batch
            shape (e.g. `(1, d)` for flat MLP input, `(1, N, d)` for
            structured particle input).
        dummy_context: Example context tensor, or `None` for unconditional.

    Returns:
        Params PyTree with `dense_out` kernel and bias zeroed.
    """
    variables = conditioner.init(key, dummy_x, dummy_context)
    params = variables.get("params", {})
    out = conditioner.get_output_layer(params)
    return conditioner.set_output_layer(
        params, jnp.zeros_like(out["kernel"]), jnp.zeros_like(out["bias"])
    )


# ===================================================================
# Transformer: permutation-equivariant self-attention conditioner
# ===================================================================
# Pre-norm stack of multi-head self-attention + feed-forward blocks.
# Fully connected attention (no mask) because the particle set is small
# and dense interactions are the whole point. The output is per-token,
# so when used with SplitCoupling(flatten_input=False) and N_frozen ==
# N_transformed (the half-half split), the coupling-level mapping
# input-particle -> output-particle is permutation-equivariant.
#
# §10.4 decision: pre-norm (`h = h + attn(LayerNorm(h))`). More stable
# for deep stacks than the original "Attention Is All You Need" post-norm
# layout; see INTERNALS.md.
class Transformer(nn.Module):
    """
    Permutation-equivariant self-attention conditioner.

    Architecture (pre-norm):
      1. `input_proj`: Dense(embed_dim) applied per-particle.
      2. (optional) Broadcast-add a learned context projection to every
         token.
      3. For each of `num_layers` blocks:
           h = h + SelfAttention(LayerNorm(h))
           h = h + FFN(LayerNorm(h))
         where FFN = Dense(ffn_multiplier * embed_dim) -> gelu ->
         Dense(embed_dim).
      4. Final LayerNorm.
      5. `dense_out`: per-token Dense(out_per_particle); caller reshapes.

    Permutation equivariance. Attention, LayerNorm, and per-token Dense
    layers all commute with particle-axis permutations. Shuffling the
    input along the particle axis shuffles the output the same way.

    Output shape: `(*batch, N, out_per_particle)`. When used with
    `SplitCoupling(flatten_input=False)` and `N_frozen == N_transformed`
    (the half-half split), set `out_per_particle = d * params_per_scalar`
    so the total trailing size matches `required_out_dim`.

    Attributes:
      num_layers: Stacked attention/FFN block count.
      num_heads: Heads in multi-head self-attention; must divide
        `embed_dim`.
      embed_dim: Token feature width.
      out_per_particle: Output width for each particle (e.g.,
        `d * (3K-1)` for a linear-tails spline with K bins).
      ffn_multiplier: FFN hidden width = `ffn_multiplier * embed_dim`
        (default 4, standard Transformer).
      context_dim: Context width (0 for unconditional). Context is
        projected to `embed_dim` and broadcast-added to every token.
    """
    num_layers: int
    num_heads: int
    embed_dim: int
    out_per_particle: int
    ffn_multiplier: int = 4
    context_dim: int = 0

    @nn.compact
    def __call__(self, x: Array, context: Array | None = None) -> Array:
        if x.ndim < 2:
            raise ValueError(
                f"Transformer expects input with rank >= 2 "
                f"(a particle axis is required); got shape {x.shape}."
            )
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"Transformer.embed_dim={self.embed_dim} must be divisible "
                f"by num_heads={self.num_heads}."
            )

        if context is not None and self.context_dim == 0:
            raise ValueError(
                "context was passed but context_dim=0. "
                "Set context_dim to match your context features."
            )
        if context is None and self.context_dim > 0:
            raise ValueError(
                f"context_dim={self.context_dim} but context was not passed."
            )

        h = nn.Dense(self.embed_dim, name="input_proj")(x)

        if context is not None and self.context_dim > 0:
            if context.shape[-1] != self.context_dim:
                raise ValueError(
                    f"Transformer expected context with last dim "
                    f"{self.context_dim}, got {context.shape[-1]}."
                )
            c = nn.Dense(self.embed_dim, name="context_proj")(context)
            if c.ndim == h.ndim - 1:
                c = c[..., None, :]
            c = jnp.broadcast_to(c, h.shape)
            h = h + c

        for i in range(self.num_layers):
            h_norm = nn.LayerNorm(name=f"ln_attn_{i}")(h)
            # Self-attention: pass `h_norm` as `inputs_q`; the module reuses it
            # as keys/values when `inputs_kv` is omitted.
            attn_out = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=self.embed_dim,
                out_features=self.embed_dim,
                use_bias=True,
                name=f"attn_{i}",
            )(h_norm)
            h = h + attn_out
            h_norm = nn.LayerNorm(name=f"ln_ffn_{i}")(h)
            h2 = nn.Dense(self.ffn_multiplier * self.embed_dim,
                          name=f"ffn_in_{i}")(h_norm)
            h2 = nn.gelu(h2)
            h2 = nn.Dense(self.embed_dim, name=f"ffn_out_{i}")(h2)
            h = h + h2

        h = nn.LayerNorm(name="ln_final")(h)
        return nn.Dense(self.out_per_particle, name="dense_out")(h)

    def get_output_layer(self, params: dict) -> dict:
        return params["dense_out"]

    def set_output_layer(self, params: dict, kernel: Array, bias: Array) -> dict:
        new_params = dict(params)
        new_params["dense_out"] = {"kernel": kernel, "bias": bias}
        return new_params


# ===================================================================
# GNN: permutation-equivariant message-passing conditioner
# ===================================================================
# Per-forward top-`num_neighbours` edges under orthogonal-box PBC (via
# nflojax.utils.pbc.pairwise_distance). Messages are MLPs over
# [node_i, node_j, distance_ij]; aggregation is sum over neighbours;
# node update is a residual MLP. Output is per-token — equivariance
# story identical to `Transformer`.
#
# §10.5 decision: default `num_neighbours=12`. Applications (bgmat
# uses 18) override via the constructor.
class GNN(nn.Module):
    """
    Permutation-equivariant message-passing conditioner.

    Architecture:
      1. `embed`: Dense(hidden) applied per-particle.
      2. (optional) Context broadcast-added to each node (projected to
         `hidden`).
      3. Per-forward neighbour list: top-`num_neighbours` nearest
         particles (excluding self) by squared distance. Orthogonal-box
         PBC when `geometry` is provided.
      4. For each of `num_layers` blocks:
           msg = Dense -> activation -> Dense over [h_i, h_j, d_ij]
           agg = sum over the neighbour axis
           h   = h + Dense -> activation -> Dense over [h, agg]
      5. `dense_out`: per-token Dense(out_per_particle).

    Permutation equivariance. Sum aggregation, per-token Dense, and the
    symmetric neighbour list all commute with particle-axis permutations.

    Output shape: `(*batch, N, out_per_particle)`. Use with
    `SplitCoupling(flatten_input=False)` and half-half split.

    Attributes:
      num_layers: Message-passing depth.
      hidden: Node feature width.
      out_per_particle: Output width per particle.
      num_neighbours: K nearest neighbours per forward. Default 12.
      cutoff: Optional distance cutoff; messages from neighbours farther
        than `cutoff` are zero-weighted. `None` keeps all K neighbours.
      geometry: Optional `Geometry` for PBC minimum-image distances.
        `None` uses plain Euclidean distances.
      context_dim: Context width (0 for unconditional).
      activation: Activation inside message and update MLPs.
    """
    num_layers: int
    hidden: int
    out_per_particle: int
    num_neighbours: int = 12
    cutoff: float | None = None
    geometry: Geometry | None = None
    context_dim: int = 0
    activation: Callable[[Array], Array] = nn.silu

    @nn.compact
    def __call__(self, x: Array, context: Array | None = None) -> Array:
        if x.ndim < 2:
            raise ValueError(
                f"GNN expects input with rank >= 2; got shape {x.shape}."
            )
        N = x.shape[-2]
        K = int(self.num_neighbours)
        if K >= N:
            raise ValueError(
                f"GNN.num_neighbours={K} must be < N={N} "
                f"(self-edges are excluded)."
            )

        if context is not None and self.context_dim == 0:
            raise ValueError(
                "context was passed but context_dim=0. "
                "Set context_dim to match your context features."
            )
        if context is None and self.context_dim > 0:
            raise ValueError(
                f"context_dim={self.context_dim} but context was not passed."
            )

        h = nn.Dense(self.hidden, name="embed")(x)

        if context is not None and self.context_dim > 0:
            if context.shape[-1] != self.context_dim:
                raise ValueError(
                    f"GNN expected context with last dim {self.context_dim}, "
                    f"got {context.shape[-1]}."
                )
            c = nn.Dense(self.hidden, name="context_proj")(context)
            if c.ndim == h.ndim - 1:
                c = c[..., None, :]
            c = jnp.broadcast_to(c, h.shape)
            h = h + c

        # Per-forward neighbour list. Remove self-edges by pinning the diagonal
        # to +inf BEFORE top_k. Do NOT use `jnp.eye(N) * jnp.inf`: that gives
        # `0 * inf = NaN` off-diagonal and silently poisons the neighbour list.
        d_sq = pairwise_distance_sq(x, self.geometry)
        eye_bool = jnp.eye(N, dtype=bool)
        d_sq_no_self = jnp.where(eye_bool, jnp.inf, d_sq)
        neg_topk, idx_topk = jax.lax.top_k(-d_sq_no_self, K)  # (..., N, K)
        d_nearest = jnp.sqrt(jnp.maximum(-neg_topk, 0.0))      # (..., N, K)

        for layer_idx in range(self.num_layers):
            neighbours = _gather_neighbours(h, idx_topk)            # (..., N, K, H)
            node_i = jnp.broadcast_to(h[..., :, None, :], neighbours.shape)
            msg_inp = jnp.concatenate(
                [node_i, neighbours, d_nearest[..., None]], axis=-1
            )
            msg = nn.Dense(self.hidden, name=f"msg_in_{layer_idx}")(msg_inp)
            msg = self.activation(msg)
            msg = nn.Dense(self.hidden, name=f"msg_out_{layer_idx}")(msg)

            if self.cutoff is not None:
                cutoff_mask = (d_nearest < float(self.cutoff))[..., None]
                msg = msg * cutoff_mask.astype(msg.dtype)

            agg = jnp.sum(msg, axis=-2)                              # (..., N, H)
            upd_inp = jnp.concatenate([h, agg], axis=-1)
            upd = nn.Dense(self.hidden, name=f"upd_in_{layer_idx}")(upd_inp)
            upd = self.activation(upd)
            upd = nn.Dense(self.hidden, name=f"upd_out_{layer_idx}")(upd)
            h = h + upd

        return nn.Dense(self.out_per_particle, name="dense_out")(h)

    def get_output_layer(self, params: dict) -> dict:
        return params["dense_out"]

    def set_output_layer(self, params: dict, kernel: Array, bias: Array) -> dict:
        new_params = dict(params)
        new_params["dense_out"] = {"kernel": kernel, "bias": bias}
        return new_params



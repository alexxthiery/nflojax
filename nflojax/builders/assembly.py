from __future__ import annotations

from typing import Any, Callable, Dict, List, Tuple

import jax
import jax.numpy as jnp

from ..flows import Bijection, Flow
from ..nets import init_resnet, Array, PRNGKey
from ..transforms import CompositeTransform, Permutation
from .common import _validate_feature_extractor_pair


def analyze_mask_coverage(blocks, dim: int) -> None:
    """
    Check how often each *original* dimension is transformed by coupling-like
    blocks, taking permutations into account.

    A coupling-like block is any block with a 1D `mask` attribute of shape (dim,),
    where mask=1 means frozen and mask=0 means transformed.

    Raises a ValueError if any dimension is never transformed.
    """
    current_pos = jnp.arange(dim)
    transformed_counts = jnp.zeros(dim, dtype=jnp.int32)

    for block in blocks:
        mask = getattr(block, "mask", None)

        if mask is not None:
            mask = jnp.asarray(mask)
            if mask.shape != (dim,):
                raise ValueError(
                    f"Coupling mask shape {mask.shape} incompatible with dim={dim} "
                    f"for block type {type(block).__name__}."
                )

            transformed_positions = jnp.where(mask == 0)[0]
            transformed_original = current_pos[transformed_positions]
            transformed_counts = transformed_counts.at[transformed_original].add(1)

        elif isinstance(block, Permutation):
            current_pos = current_pos[block.perm]

        else:
            # ignore other block types (LinearTransform, LoftTransform, etc.)
            pass

    never = jnp.where(transformed_counts == 0)[0]
    if never.size > 0:
        raise ValueError(
            "\n" + "=" * 80 + "\n"
            f"Mask/permutation schedule leaves some dims never transformed: \n"
            f"indices {never.tolist()}, \n"
            f"counts={transformed_counts.tolist()} \n"
            f"Possible cause: permutation layer and parity masks cancel out. \n"
            f"\n ** Use the option: use_permutation=False or change the number of layers. ** \n"
            "\n" + "=" * 80
        )

# ====================================================================
# Mask construction utility
# ====================================================================
def make_alternating_mask(dim: int, parity: int) -> Array:
    """
    Build a binary mask of length dim with alternating 1/0 pattern.

    In coupling layers, mask=1 means the dimension is frozen (passed through),
    and mask=0 means the dimension is transformed.

    Args:
        dim: Length of the mask (must be positive).
        parity: Which pattern to use (must be 0 or 1).
            parity=0: mask = [1, 0, 1, 0, ...] (even indices frozen)
            parity=1: mask = [0, 1, 0, 1, ...] (odd indices frozen)

    Returns:
        Binary mask array of shape (dim,) with dtype float32.

    Raises:
        ValueError: If dim <= 0 or parity not in {0, 1}.

    Example:
        >>> make_alternating_mask(4, parity=0)
        Array([1., 0., 1., 0.], dtype=float32)
        >>> make_alternating_mask(4, parity=1)
        Array([0., 1., 0., 1.], dtype=float32)
    """
    if dim <= 0:
        raise ValueError(f"make_alternating_mask: dim must be positive, got {dim}.")
    if parity not in (0, 1):
        raise ValueError(f"make_alternating_mask: parity must be 0 or 1, got {parity}.")

    idx = jnp.arange(dim)
    mask = ((idx + parity) % 2 == 0).astype(jnp.float32)
    return mask


# ====================================================================
# Feature extractor creation
# ====================================================================
def create_feature_extractor(
    key: PRNGKey,
    in_dim: int,
    hidden_dim: int,
    out_dim: int,
    n_layers: int = 2,
    activation: Callable[[Array], Array] = jax.nn.tanh,
    res_scale: float = 0.1,
) -> Tuple[Any, Any]:
    """
    Create a context feature extractor network.

    The feature extractor is a ResNet that transforms raw context into learned
    features before they are used by coupling layers. This is useful when raw
    context features are high-dimensional or not directly informative.

    Args:
        key: JAX PRNGKey for initialization.
        in_dim: Input dimension (raw context dimension).
        hidden_dim: Width of hidden layers in the ResNet.
        out_dim: Output dimension (effective context dimension for couplings).
        n_layers: Number of residual blocks (default: 2).
        activation: Activation function (default: tanh).
        res_scale: Scale factor for residual connections (default: 0.1).

    Returns:
        feature_extractor: A Flax ResNet module.
        params: Parameters for the feature extractor.

    Raises:
        ValueError: If any dimension is not positive or n_layers < 1.

    Example:
        >>> fe, fe_params = create_feature_extractor(key, in_dim=8, hidden_dim=32, out_dim=16)
        >>> # Then use out_dim as context_dim for couplings:
        >>> coupling, c_params = AffineCoupling.create(..., context_dim=16)
    """
    if in_dim <= 0:
        raise ValueError(f"create_feature_extractor: in_dim must be positive, got {in_dim}.")
    if hidden_dim <= 0:
        raise ValueError(f"create_feature_extractor: hidden_dim must be positive, got {hidden_dim}.")
    if out_dim <= 0:
        raise ValueError(f"create_feature_extractor: out_dim must be positive, got {out_dim}.")
    if n_layers < 1:
        raise ValueError(f"create_feature_extractor: n_layers must be >= 1, got {n_layers}.")

    feature_extractor, params = init_resnet(
        key,
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        out_dim=out_dim,
        n_hidden_layers=n_layers,
        activation=activation,
        res_scale=res_scale,
        zero_init_output=False,
    )
    return feature_extractor, params


# ====================================================================
# Shared helpers
# ====================================================================
# ====================================================================
# Flow/Bijection assembly from blocks
# ====================================================================
def _validate_blocks_and_params(
    blocks_and_params: List[Tuple[Any, Any]],
    function_name: str,
) -> Tuple[List[Any], List[Any]]:
    """
    Validate and unzip a list of (block, params) tuples.

    Args:
        blocks_and_params: List of (transform, params) tuples.
        function_name: Name of calling function for error messages.

    Returns:
        (blocks, block_params): Unzipped lists.

    Raises:
        ValueError: If validation fails.
    """
    if not blocks_and_params:
        raise ValueError(
            f"{function_name}: blocks_and_params cannot be empty. "
            "Provide at least one (transform, params) tuple."
        )

    blocks = []
    block_params = []

    for i, item in enumerate(blocks_and_params):
        if not isinstance(item, tuple) or len(item) != 2:
            raise ValueError(
                f"{function_name}: Element {i} must be a (transform, params) tuple, "
                f"got {type(item).__name__} with {len(item) if hasattr(item, '__len__') else 'N/A'} elements."
            )
        block, params = item
        blocks.append(block)
        block_params.append(params)

    return blocks, block_params


def _check_dimension_consistency(blocks: List[Any], function_name: str) -> int | None:
    """
    Check that all coupling-like blocks have consistent dimensions.

    Returns:
        The consistent dimension, or None if no coupling blocks.

    Raises:
        ValueError: If dimensions are inconsistent.
    """
    dims = []
    for i, block in enumerate(blocks):
        mask = getattr(block, "mask", None)
        if mask is not None:
            dim = int(jnp.asarray(mask).shape[0])
            dims.append((i, type(block).__name__, dim))

    if not dims:
        return None

    first_dim = dims[0][2]
    inconsistent = [(i, name, d) for i, name, d in dims if d != first_dim]

    if inconsistent:
        raise ValueError(
            f"{function_name}: Inconsistent dimensions across coupling blocks. "
            f"First coupling (index {dims[0][0]}, {dims[0][1]}) has dim={first_dim}, "
            f"but found: {[(f'index {i}, {name}, dim={d}') for i, name, d in inconsistent]}."
        )

    return first_dim


def assemble_bijection(
    blocks_and_params: List[Tuple[Any, Any]],
    feature_extractor: Any = None,
    feature_extractor_params: Any = None,
    validate: bool = True,
    identity_gate: Callable[[Array], Array] | None = None,
) -> Tuple[Bijection, Dict]:
    """
    Assemble a Bijection from a list of (transform, params) tuples.

    This function provides a mid-level API for building custom flow architectures
    by manually composing transform blocks. It handles the bookkeeping of aligning
    blocks with their parameters and constructing the correct params dict structure.

    Args:
        blocks_and_params: List of (transform, params) tuples, as returned by
            Transform.create() factory methods. Example:
                [
                    AffineCoupling.create(key1, dim=4, mask=mask0, ...),
                    SplineCoupling.create(key2, dim=4, mask=mask1, ...),
                    LinearTransform.create(key3, dim=4),
                ]
        feature_extractor: Optional context feature extractor (Flax module).
            If provided, raw context will be transformed before being passed
            to coupling layers.
        feature_extractor_params: Parameters for the feature extractor.
            Required if feature_extractor is provided.
        validate: If True (default), perform validation:
            - Check blocks_and_params is non-empty
            - Check each element is a 2-tuple
            - Check dimension consistency across coupling blocks
            - Run mask coverage analysis

    Returns:
        bijection: Bijection object with the composed transform.
        params: Dict with structure:
            {"transform": [block_params...]}
            or {"transform": [...], "feature_extractor": fe_params}

    Raises:
        ValueError: If validation fails.

    Example:
        >>> from nflojax.builders import make_alternating_mask, assemble_bijection
        >>> from nflojax.transforms import AffineCoupling, SplineCoupling, LoftTransform
        >>>
        >>> keys = jax.random.split(key, 4)
        >>> mask0 = make_alternating_mask(4, parity=0)
        >>> mask1 = make_alternating_mask(4, parity=1)
        >>>
        >>> blocks_and_params = [
        ...     AffineCoupling.create(keys[0], dim=4, mask=mask0, hidden_dim=64, n_hidden_layers=2),
        ...     AffineCoupling.create(keys[1], dim=4, mask=mask1, hidden_dim=64, n_hidden_layers=2),
        ...     SplineCoupling.create(keys[2], dim=4, mask=mask0, hidden_dim=64, n_hidden_layers=2, num_bins=8),
        ...     LoftTransform.create(keys[3], dim=4),
        ... ]
        >>>
        >>> bijection, params = assemble_bijection(blocks_and_params)
        >>> y, log_det = bijection.forward(params, x)
    """
    _validate_feature_extractor_pair(
        "assemble_bijection", feature_extractor, feature_extractor_params
    )

    # Validate and unzip blocks
    blocks, block_params = _validate_blocks_and_params(
        blocks_and_params, "assemble_bijection"
    )

    if validate:
        # Check dimension consistency
        dim = _check_dimension_consistency(blocks, "assemble_bijection")

        # Run mask coverage analysis if we have coupling blocks
        if dim is not None:
            analyze_mask_coverage(blocks, dim)

    # Assemble
    transform = CompositeTransform(blocks=blocks)

    params: Dict[str, Any] = {"transform": block_params}
    if feature_extractor is not None:
        params["feature_extractor"] = feature_extractor_params

    bijection = Bijection(transform=transform, feature_extractor=feature_extractor, identity_gate=identity_gate)
    return bijection, params


def assemble_flow(
    blocks_and_params: List[Tuple[Any, Any]],
    base: Any,
    base_params: Any = None,
    feature_extractor: Any = None,
    feature_extractor_params: Any = None,
    validate: bool = True,
    identity_gate: Callable[[Array], Array] | None = None,
) -> Tuple[Flow, Dict]:
    """
    Assemble a Flow from blocks, base distribution, and optional feature extractor.

    This function provides a mid-level API for building custom flow architectures.
    It combines manually composed transform blocks with a base distribution to
    create a full normalizing flow.

    Args:
        blocks_and_params: List of (transform, params) tuples, as returned by
            Transform.create() factory methods.
        base: Base distribution object (e.g., StandardNormal(dim), DiagNormal(dim)).
            Must have log_prob(), sample(), and init_params() methods.
        base_params: Parameters for the base distribution. If None, calls
            base.init_params() to get default parameters.
        feature_extractor: Optional context feature extractor (Flax module).
        feature_extractor_params: Parameters for the feature extractor.
            Required if feature_extractor is provided.
        validate: If True (default), perform validation checks.

    Returns:
        flow: Flow object with base distribution and composed transform.
        params: Dict with structure:
            {"base": base_params, "transform": [block_params...]}
            or {"base": ..., "transform": [...], "feature_extractor": fe_params}

    Raises:
        ValueError: If validation fails or base is None.

    Example:
        >>> from nflojax.builders import make_alternating_mask, assemble_flow
        >>> from nflojax.transforms import AffineCoupling, LoftTransform
        >>> from nflojax.distributions import StandardNormal
        >>>
        >>> keys = jax.random.split(key, 3)
        >>> mask0 = make_alternating_mask(4, parity=0)
        >>> mask1 = make_alternating_mask(4, parity=1)
        >>>
        >>> blocks_and_params = [
        ...     AffineCoupling.create(keys[0], dim=4, mask=mask0, hidden_dim=64, n_hidden_layers=2),
        ...     AffineCoupling.create(keys[1], dim=4, mask=mask1, hidden_dim=64, n_hidden_layers=2),
        ...     LoftTransform.create(keys[2], dim=4),
        ... ]
        >>>
        >>> flow, params = assemble_flow(blocks_and_params, base=StandardNormal(dim=4))
        >>> log_prob = flow.log_prob(params, x)
        >>> samples = flow.sample(params, key, shape=(100,))
    """
    if base is None:
        raise ValueError(
            "assemble_flow: base distribution is required. "
            "Use assemble_bijection() if you don't need a base distribution."
        )

    _validate_feature_extractor_pair(
        "assemble_flow", feature_extractor, feature_extractor_params
    )

    # Validate and unzip blocks
    blocks, block_params = _validate_blocks_and_params(
        blocks_and_params, "assemble_flow"
    )

    if validate:
        # Check dimension consistency
        dim = _check_dimension_consistency(blocks, "assemble_flow")

        # Run mask coverage analysis if we have coupling blocks
        if dim is not None:
            analyze_mask_coverage(blocks, dim)

    # Resolve base params
    if base_params is None:
        base_params = base.init_params()

    # Assemble
    transform = CompositeTransform(blocks=blocks)

    params: Dict[str, Any] = {
        "base": base_params,
        "transform": block_params,
    }
    if feature_extractor is not None:
        params["feature_extractor"] = feature_extractor_params

    flow = Flow(base_dist=base, transform=transform, feature_extractor=feature_extractor, identity_gate=identity_gate)
    return flow, params


# ====================================================================
# Shared builder helpers
# ====================================================================

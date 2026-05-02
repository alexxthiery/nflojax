from __future__ import annotations

from typing import Any, Callable

from ..distributions import DiagNormal, StandardNormal
from ..flows import Bijection, Flow
from ..nets import Array
from ..transforms import CompositeTransform, validate_identity_gate


def _event_shape_tuple(event_shape: Any) -> tuple[int, ...]:
    """Normalize an event-shape-like value for builder validation."""
    if isinstance(event_shape, int):
        return (int(event_shape),)
    return tuple(int(v) for v in event_shape)


def _validate_positive_int(func_name: str, name: str, value: int) -> None:
    """Validate a positive integer builder argument."""
    if value <= 0:
        raise ValueError(f"{func_name}: {name} must be positive, got {value}.")


def _validate_nonnegative_int(func_name: str, name: str, value: int) -> None:
    """Validate a non-negative integer builder argument."""
    if value < 0:
        raise ValueError(
            f"{func_name}: {name} must be non-negative, got {value}."
        )


def _validate_context_options(
    func_name: str,
    *,
    context_dim: int,
    identity_gate: Callable[[Array], Array] | None,
    context_extractor_hidden_dim: int = 0,
    use_permutation: bool = False,
) -> None:
    """Validate context, feature-extractor, and identity-gate options."""
    _validate_nonnegative_int(func_name, "context_dim", context_dim)
    if identity_gate is not None and use_permutation:
        raise ValueError(
            f"{func_name}: identity_gate is incompatible with use_permutation=True. "
            "Permutation cannot be smoothly interpolated to identity."
        )
    if identity_gate is not None and context_dim == 0:
        raise ValueError(
            f"{func_name}: identity_gate requires context_dim > 0 "
            "(gate function operates on context)."
        )
    validate_identity_gate(identity_gate, context_dim)
    if context_extractor_hidden_dim > 0 and context_dim == 0:
        raise ValueError(
            f"{func_name}: context_extractor_hidden_dim > 0 requires context_dim > 0."
        )


def _validate_feature_extractor_pair(
    func_name: str,
    feature_extractor: Any,
    feature_extractor_params: Any,
) -> None:
    """Validate feature-extractor module/params pairing."""
    if feature_extractor is not None and feature_extractor_params is None:
        raise ValueError(
            f"{func_name}: feature_extractor_params is required when "
            "feature_extractor is provided."
        )
    if feature_extractor is None and feature_extractor_params is not None:
        raise ValueError(
            f"{func_name}: feature_extractor_params provided but "
            "feature_extractor is None."
        )


def _validate_base_event_shape(
    func_name: str,
    base_dist: Any,
    expected_event_shape: tuple[int, ...],
) -> None:
    """Validate a base distribution event shape when the base exposes one."""
    actual = getattr(base_dist, "event_shape", None)
    if actual is None:
        return
    actual_shape = _event_shape_tuple(actual)
    expected_shape = _event_shape_tuple(expected_event_shape)
    if actual_shape != expected_shape:
        raise ValueError(
            f"{func_name}: base_dist.event_shape must match "
            f"{expected_shape}; got {actual_shape}."
        )


def _resolve_base(
    func_name: str,
    base_dist: Any,
    base_params: Any,
    expected_event_shape: tuple[int, ...],
) -> tuple[Any, Any]:
    """Validate and resolve params for an explicit base distribution."""
    _validate_base_event_shape(func_name, base_dist, expected_event_shape)
    return base_dist, base_dist.init_params() if base_params is None else base_params


def _validate_builder_args(
    func_name: str,
    dim: int,
    num_layers: int,
    context_dim: int,
    identity_gate: Callable[[Array], Array] | None,
    use_permutation: bool,
    context_extractor_hidden_dim: int,
) -> None:
    """Validate arguments common to build_realnvp and build_spline_realnvp."""
    _validate_positive_int(func_name, "dim", dim)
    _validate_positive_int(func_name, "num_layers", num_layers)
    _validate_context_options(
        func_name,
        context_dim=context_dim,
        identity_gate=identity_gate,
        context_extractor_hidden_dim=context_extractor_hidden_dim,
        use_permutation=use_permutation,
    )


def _validate_spline_floors(
    func_name: str,
    num_bins: int,
    tail_bound: float,
    min_bin_width: float,
    min_bin_height: float,
    min_derivative: float | None = None,
    max_derivative: float | None = None,
) -> None:
    """Validate spline bin count, bounds, and numerical floors."""
    _validate_positive_int(func_name, "num_bins", num_bins)
    if tail_bound <= 0:
        raise ValueError(
            f"{func_name}: tail_bound must be positive, got {tail_bound}."
        )
    if min_bin_width * num_bins >= 1.0:
        raise ValueError(
            f"{func_name}: min_bin_width * num_bins must be < 1. "
            f"Got {min_bin_width} * {num_bins} = {min_bin_width * num_bins}."
        )
    if min_bin_height * num_bins >= 1.0:
        raise ValueError(
            f"{func_name}: min_bin_height * num_bins must be < 1. "
            f"Got {min_bin_height} * {num_bins} = {min_bin_height * num_bins}."
        )
    if min_derivative is not None and min_derivative <= 0:
        raise ValueError(
            f"{func_name}: min_derivative must be positive, got {min_derivative}."
        )
    if max_derivative is not None and max_derivative <= 0:
        raise ValueError(
            f"{func_name}: max_derivative must be positive, got {max_derivative}."
        )
    if (
        min_derivative is not None
        and max_derivative is not None
        and min_derivative > max_derivative
    ):
        raise ValueError(
            f"{func_name}: min_derivative must be <= max_derivative; got "
            f"{min_derivative} > {max_derivative}."
        )


def _resolve_flat_base(
    func_name: str,
    dim: int,
    trainable_base: bool,
    base_dist: Any | None,
    base_params: Any | None,
) -> tuple[Any, Any]:
    """Resolve the base distribution and params for flat all-real builders."""
    if base_dist is not None:
        return _resolve_base(func_name, base_dist, base_params, (dim,))
    if trainable_base:
        base = DiagNormal(dim=dim)
    else:
        base = StandardNormal(dim=dim)
    return base, base.init_params()


def _assemble_builder_output(
    *,
    transform: CompositeTransform,
    block_params: list[Any],
    return_transform_only: bool,
    identity_gate: Callable[[Array], Array] | None,
    base: Any | None = None,
    base_params: Any | None = None,
    feature_extractor: Any | None = None,
    feature_extractor_params: Any | None = None,
) -> tuple[Flow | Bijection, dict[str, Any]]:
    """Package builder output without changing public parameter structure."""
    params: dict[str, Any] = {"transform": block_params}
    if feature_extractor is not None:
        params["feature_extractor"] = feature_extractor_params

    if return_transform_only:
        return (
            Bijection(
                transform=transform,
                feature_extractor=feature_extractor,
                identity_gate=identity_gate,
            ),
            params,
        )

    params = {"base": base_params, **params}
    return (
        Flow(
            base_dist=base,
            transform=transform,
            feature_extractor=feature_extractor,
            identity_gate=identity_gate,
        ),
        params,
    )

#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch


def lookup_sample_value(key: str, sample: Mapping[str, Any]) -> Any | None:
    """Look up a value from a flat or dotted-path sample mapping.

    Args:
        key: Sample key to look up. The key may either be present directly in the
            mapping or expressed as a dotted path such as `"observation.state"`.
        sample: Sample mapping to search.

    Returns:
        The matching value if found, otherwise `None`.
    """
    if key in sample:
        return sample[key]

    current: Any = sample
    for part in key.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def coerce_to_tensor(value: Any | None) -> torch.Tensor | None:
    """Best-effort conversion from numeric sample content to a tensor.

    Args:
        value: Value that may already be a tensor or may be convertible to one.

    Returns:
        A tensor view of the input value when conversion is possible, otherwise
        `None`.
    """
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        return value
    try:
        return torch.as_tensor(value)
    except (TypeError, ValueError):
        return None


def extract_latest_state_tensor(value: Any | None) -> torch.Tensor | None:
    """Convert a state-like value into a 1D tensor for the latest state.

    CRITICAL FIX: Tightened input validation to prevent misinterpretation of
    high-dimensional data (e.g., images) as state vectors.

    Supported shapes:
    - (D,): 1D state vector → returns as-is
    - (1, D): Batch size 1 state → squeezed to (D,)
    - (T, D): Sequence of states → extracts latest row as (D,)

    Unsupported shapes (rejected):
    - (H, W, C): Images and visual features
    - (B, T, D): Already-batched sequences with explicit batch dim
    - Other high-dimensional tensors

    Args:
        value: State-like content from a sample.

    Returns:
        A 1D state tensor of shape (D,), or None if:
        - value is None
        - value cannot be converted to a tensor
        - value has unsupported shape

    Examples:
        >>> extract_latest_state_tensor([0.1, 0.2, 0.3])  # (D,)
        tensor([0.1, 0.2, 0.3])
        >>> extract_latest_state_tensor([[0.1, 0.2, 0.3]])  # (1, D)
        tensor([0.1, 0.2, 0.3])
        >>> extract_latest_state_tensor([[0.1, 0.2], [0.3, 0.4]])  # (T, D)
        tensor([0.3, 0.4])
    """
    tensor = coerce_to_tensor(value)
    if tensor is None:
        return None

    tensor = tensor.detach()

    # Squeeze leading dimensions of size 1 (e.g., (1, D) → (D,))
    while tensor.ndim > 1 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)

    # Handle valid cases
    if tensor.ndim == 1:
        # (D,) - valid 1D state vector
        return tensor

    if tensor.ndim == 2:
        # (T, D) - sequence of states, take the last one
        return tensor[-1]

    # CRITICAL FIX: Reject high-dimensional inputs (likely images or invalid data)
    if tensor.ndim >= 3:
        # e.g., (H, W, C) or (B, T, D) or other high-dim data
        # This is likely a misuse (visual features instead of state)
        # Log and return None instead of silently reshaping
        import logging
        logging.warning(
            f"extract_latest_state_tensor: Rejecting high-dimensional tensor with shape {tuple(tensor.shape)}. "
            "State vectors should be 1D or 2D (time steps × dimensions). "
            "Received data may be visual features (images) or incorrectly formatted state."
        )
        return None

    return None


def extract_action_chunk_tensor(value: Any | None) -> torch.Tensor | None:
    """Convert an action-like value into a `(T, D)` action-chunk tensor.

    Legacy note:
        This helper remains only for legacy/internal chunk-based code paths.
        Current public hard/soft stage-label generators use
        `extract_action_chunk_with_reason(...)` instead.

    The behavior matches the current hard-label generator:
    - `(D,)` becomes `(1, D)`
    - `(T, D)` stays `(T, D)`
    - higher-rank action inputs are flattened into `(-1, D)`

    Args:
        value: Action-like content from a sample.

    Returns:
        A `(T, D)` action tensor or `None` if conversion is not possible.
    """
    tensor = coerce_to_tensor(value)
    if tensor is None:
        return None
    tensor = tensor.detach()
    while tensor.ndim > 2 and tensor.shape[0] == 1:
        tensor = tensor.squeeze(0)
    if tensor.ndim == 1:
        return tensor.unsqueeze(0)
    if tensor.ndim >= 2:
        return tensor.reshape(-1, tensor.shape[-1])
    return None


def extract_action_chunk_with_reason(
    value: Any | None,
    min_steps: int = 2,
) -> tuple[torch.Tensor | None, str | None, dict[str, Any]]:
    """Convert an action-like value into a local action chunk tensor of shape `(K, D)`.

    The chunk semantics are explicit:
    - arm energy is later computed from `delta_action_chunk = action_chunk[1:] - action_chunk[:-1]`
    - gripper signal is primarily derived from event detection over the chunk

    Args:
        value: Action-chunk-like content from a sample.
        min_steps: Minimum required number of chunk steps.

    Returns:
        A tuple `(action_chunk, reason, debug)` where `action_chunk` is a detached
        `(K, D)` tensor on success. Invalid cases return `None` with one of:
        - `"missing_action_chunk"`
        - `"invalid_action_chunk_shape"`
        - `"insufficient_action_chunk_length"`
    """
    if value is None:
        return None, "missing_action_chunk", {}

    tensor = coerce_to_tensor(value)
    if tensor is None:
        return None, "missing_action_chunk", {"action_chunk_value_type": type(value).__name__}

    tensor = tensor.detach()
    if tensor.ndim != 2:
        return None, "invalid_action_chunk_shape", {"action_chunk_shape": tuple(tensor.shape)}

    chunk_length = int(tensor.shape[0])
    if chunk_length < min_steps:
        return None, "insufficient_action_chunk_length", {"chunk_length": chunk_length}

    return tensor, None, {}


def extract_state_chunk_tensor(
    value: Any | None,
    min_steps: int = 2,
) -> tuple[torch.Tensor | None, str | None, dict[str, Any]]:
    """Convert a state-like value into a local state chunk tensor of shape `(K, S)`.

    This helper is dedicated to chunk-based gripper event extraction. Unlike the
    legacy action-chunk helper, it does not reinterpret 1D inputs as a valid
    chunk because state-event detection requires an explicit sequence.

    Args:
        value: State-chunk-like content from a sample.
        min_steps: Minimum required number of chunk steps.

    Returns:
        A tuple `(state_chunk, reason, debug)` where `state_chunk` is a detached
        `(K, S)` tensor on success. Invalid cases return `None` with one of:
        - `"missing_state_chunk"`
        - `"invalid_state_chunk_shape"`
        - `"insufficient_state_chunk_length"`
    """
    if value is None:
        return None, "missing_state_chunk", {}

    tensor = coerce_to_tensor(value)
    if tensor is None:
        return None, "missing_state_chunk", {"state_chunk_value_type": type(value).__name__}

    tensor = tensor.detach()
    if tensor.ndim != 2:
        return None, "invalid_state_chunk_shape", {"state_chunk_shape": tuple(tensor.shape)}

    chunk_length = int(tensor.shape[0])
    if chunk_length < min_steps:
        return None, "insufficient_state_chunk_length", {"chunk_length": chunk_length}

    return tensor, None, {}


def extract_single_action_tensor_with_reason(
    value: Any | None,
) -> tuple[torch.Tensor | None, str | None, dict[str, Any]]:
    """Convert an action-like value into a single-step action tensor of shape `(D,)`.

    Supported shapes:
    - `(D,)`: returned as-is
    - `(1, D)`: squeezed to `(D,)`

    Rejected shapes:
    - `(T, D)` with `T > 1`
    - any tensor with rank other than 1 or 2

    Args:
        value: Action-like content from a sample.

    Returns:
        A tuple `(action_tensor, reason, debug)` where:
        - `action_tensor` is a detached 1D tensor on success, otherwise `None`
        - `reason` is `None` on success, otherwise one of:
            - `"missing_action"`
            - `"unsupported_multi_step_action"`
            - `"invalid_action_shape"`
        - `debug` carries lightweight shape/type metadata for invalid cases
    """
    if value is None:
        return None, "missing_action", {}

    tensor = coerce_to_tensor(value)
    if tensor is None:
        return None, "missing_action", {"action_value_type": type(value).__name__}

    tensor = tensor.detach()
    if tensor.ndim == 1:
        return tensor, None, {}
    if tensor.ndim == 2 and tensor.shape[0] == 1:
        return tensor.squeeze(0), None, {}
    if tensor.ndim == 2:
        return None, "unsupported_multi_step_action", {"action_shape": tuple(tensor.shape)}

    return None, "invalid_action_shape", {"action_shape": tuple(tensor.shape)}


def extract_single_action_tensor(value: Any | None) -> torch.Tensor | None:
    """Backward-compatible wrapper around single-step action extraction."""
    action_tensor, _, _ = extract_single_action_tensor_with_reason(value)
    return action_tensor


def normalize_index_sequence(indices: Sequence[int] | None, size: int) -> list[int]:
    """Drop invalid indices and return a sorted unique list.

    Args:
        indices: Candidate indices.
        size: Exclusive upper bound for valid indices.

    Returns:
        A sorted list of unique indices within `[0, size)`.
    """
    if not indices:
        return []
    normalized = sorted({int(index) for index in indices if 0 <= int(index) < size})
    return normalized


def resolve_vector_indices(
    *,
    vector_dim: int | None,
    configured_indices: Sequence[int] | None,
) -> list[int]:
    """Resolve configured indices against a known vector size.

    Args:
        vector_dim: Size of the target vector. When `None`, no indices can be
            resolved.
        configured_indices: User-configured indices for the vector.

    Returns:
        A normalized index list, or an empty list when `vector_dim` is unknown.
    """
    if vector_dim is None:
        return []
    return normalize_index_sequence(configured_indices, vector_dim)


def resolve_joint_groups(
    action_dim: int,
    gripper_action_indices: Sequence[int],
    large_joint_indices: Sequence[int] | None = None,
    small_joint_indices: Sequence[int] | None = None,
) -> tuple[list[int], list[int]]:
    """Resolve large/small joint groups for coarse action-motion statistics.

    If configured groups are provided, they are normalized and returned directly.
    Otherwise the non-gripper action dimensions are split in half, with the first
    half treated as large joints and the second half treated as small joints.

    Args:
        action_dim: Size of the action vector.
        gripper_action_indices: Indices reserved for gripper control.
        large_joint_indices: Optional configured large-joint indices.
        small_joint_indices: Optional configured small-joint indices.

    Returns:
        A tuple of `(large_joint_indices, small_joint_indices)`.
    """
    configured_large = normalize_index_sequence(large_joint_indices, action_dim)
    configured_small = normalize_index_sequence(small_joint_indices, action_dim)

    if configured_large or configured_small:
        return configured_large, configured_small

    gripper_index_set = set(gripper_action_indices)
    arm_indices = [index for index in range(action_dim) if index not in gripper_index_set]
    if not arm_indices:
        return [], []

    split_index = max(1, len(arm_indices) // 2)
    resolved_large_joint_indices = arm_indices[:split_index]
    resolved_small_joint_indices = arm_indices[split_index:]
    return resolved_large_joint_indices, resolved_small_joint_indices


def compute_mean_abs_motion(action_chunk: torch.Tensor, indices: Sequence[int]) -> float:
    """Compute mean absolute motion over selected action dimensions.

    Args:
        action_chunk: Action chunk tensor with shape `(T, D)`.
        indices: Action dimensions to include in the statistic.

    Returns:
        Mean absolute value across the selected action dimensions. Returns `0.0`
        when `indices` is empty.
    """
    if not indices:
        return 0.0
    return float(action_chunk[:, list(indices)].abs().mean().item())


def compute_mean_abs_signal(action_vector: torch.Tensor, indices: Sequence[int]) -> float:
    """Compute mean absolute value over selected dimensions of one action vector.

    Args:
        action_vector: Single-step action tensor with shape `(D,)`.
        indices: Action dimensions to include in the statistic.

    Returns:
        Mean absolute value across the selected action dimensions. Returns `0.0`
        when `indices` is empty.

    Raises:
        ValueError: If `action_vector` is not a 1D tensor.
    """
    if not indices:
        return 0.0
    if action_vector.ndim != 1:
        raise ValueError(
            f"`action_vector` must have shape (D,), got {tuple(action_vector.shape)}."
        )
    return float(action_vector[list(indices)].abs().mean().item())


def compute_mean_abs_chunk_delta(
    action_chunk: torch.Tensor,
    indices: Sequence[int],
) -> float:
    """Compute mean absolute delta over selected dimensions of a local action chunk."""
    if not indices:
        return 0.0
    if action_chunk.ndim != 2 or action_chunk.shape[0] < 2:
        raise ValueError(
            f"`action_chunk` must have shape (K, D) with K >= 2, got {tuple(action_chunk.shape)}."
        )
    delta_chunk = action_chunk[1:] - action_chunk[:-1]
    return float(delta_chunk[:, list(indices)].abs().mean().item())


def compute_mean_signed_chunk_delta(
    action_chunk: torch.Tensor,
    indices: Sequence[int],
) -> float:
    """Compute mean signed delta over selected dimensions of a local action chunk."""
    if not indices:
        return 0.0
    if action_chunk.ndim != 2 or action_chunk.shape[0] < 2:
        raise ValueError(
            f"`action_chunk` must have shape (K, D) with K >= 2, got {tuple(action_chunk.shape)}."
        )
    delta_chunk = action_chunk[1:] - action_chunk[:-1]
    return float(delta_chunk[:, list(indices)].mean().item())


def extract_binary_gripper_events_from_state_chunk(
    state_chunk: torch.Tensor,
    gripper_state_indices: Sequence[int],
    *,
    closed_threshold: float | None = None,
    open_threshold: float | None = None,
    is_binary_state: bool = True,
) -> tuple[dict[str, Any], str | None, dict[str, Any]]:
    """Extract binary gripper open/close events from a state chunk.

    The returned event dict is designed for event-driven stage labeling where
    the gripper signal is primarily discrete, while arm motion remains
    delta-action-chunk energy driven.
    """
    if state_chunk.ndim != 2:
        return {}, "invalid_state_chunk_shape", {"state_chunk_shape": tuple(state_chunk.shape)}

    if not gripper_state_indices:
        return {}, None, {}

    values = state_chunk[:, list(gripper_state_indices)].float().mean(dim=1)
    if values.numel() < 2:
        return {}, "insufficient_state_chunk_length", {"chunk_length": int(values.numel())}

    if is_binary_state:
        states = values > 0
    else:
        if closed_threshold is None or open_threshold is None:
            return (
                {},
                "missing_gripper_state_thresholds",
                {
                    "closed_threshold": closed_threshold,
                    "open_threshold": open_threshold,
                },
            )
        start_value = float(values[0].item())
        end_value = float(values[-1].item())
        if start_value >= closed_threshold:
            start_state = True
        elif start_value <= open_threshold:
            start_state = False
        else:
            start_state = start_value > (closed_threshold + open_threshold) / 2.0
        if end_value >= closed_threshold:
            end_state = True
        elif end_value <= open_threshold:
            end_state = False
        else:
            end_state = end_value > (closed_threshold + open_threshold) / 2.0
        states = torch.tensor([start_state, end_state], dtype=torch.bool, device=values.device)

    start_state = bool(states[0].item())
    end_state = bool(states[-1].item())
    changed = start_state != end_state
    return (
        {
            "gripper_start_state": start_state,
            "gripper_end_state": end_state,
            "gripper_changed": changed,
            "closing_event": (not start_state) and end_state,
            "opening_event": start_state and (not end_state),
        },
        None,
        {},
    )


def extract_gripper_events_from_action_chunk(
    action_chunk: torch.Tensor,
    gripper_action_indices: Sequence[int],
    *,
    close_is_negative: bool,
    trend_threshold: float,
) -> dict[str, Any]:
    """Extract auxiliary gripper trend scores from a local action chunk."""
    if not gripper_action_indices:
        return {
            "gripper_signed_trend": 0.0,
            "close_command_score": 0.0,
            "open_command_score": 0.0,
        }

    gripper_signed_trend = compute_mean_signed_chunk_delta(action_chunk, gripper_action_indices)
    if close_is_negative:
        close_command_score = max(0.0, -gripper_signed_trend - trend_threshold)
        open_command_score = max(0.0, gripper_signed_trend - trend_threshold)
    else:
        close_command_score = max(0.0, gripper_signed_trend - trend_threshold)
        open_command_score = max(0.0, -gripper_signed_trend - trend_threshold)

    return {
        "gripper_signed_trend": gripper_signed_trend,
        "close_command_score": close_command_score,
        "open_command_score": open_command_score,
    }

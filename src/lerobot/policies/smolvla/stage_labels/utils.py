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

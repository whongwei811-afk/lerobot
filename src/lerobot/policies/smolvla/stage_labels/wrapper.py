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

from collections.abc import Mapping
from typing import Any

import torch
import torch.utils.data

from lerobot.utils.constants import ACTION, OBS_STATE

from .base import BaseStageLabelGenerator, StageLabelOutput
from .utils import extract_latest_state_tensor, extract_single_action_tensor_with_reason


class StageLabeledDataset(torch.utils.data.Dataset):
    """Dataset wrapper that attaches chunk-based stage labels to each sample.

    This wrapper leaves the underlying dataset unchanged on disk and does not
    modify the base sample in place. Instead, it creates a shallow top-level copy
    of each retrieved sample, injects a local current-to-future action chunk,
    runs a stage-label generator on the enriched sample,
    and appends the normalized stage-label fields to the copied sample.

    By default, debug metadata is excluded from training samples because debug
    dictionaries are often irregular, verbose, and less friendly to dataloader
    collation than tensor fields.

    Examples:
        ```python
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.smolvla.stage_labels.factory import build_stage_label_generator
        from lerobot.policies.smolvla.stage_labels.wrapper import StageLabeledDataset

        base_dataset = LeRobotDataset("my_dataset")
        label_generator = build_stage_label_generator("hard")
        dataset = StageLabeledDataset(base_dataset, label_generator)

        sample = dataset[0]
        ```
    """

    def __init__(
        self,
        base_dataset: Any,
        label_generator: BaseStageLabelGenerator,
        include_debug: bool = False,
        default_num_stages: int = 3,
        action_chunk_size: int = 8,
        action_chunk_anchor: str = "current_to_future",
    ) -> None:
        """Initialize a stage-labeled dataset wrapper.

        Args:
            base_dataset: Dataset-like object implementing `__len__` and
                `__getitem__`.
            label_generator: Stage-label generator compatible with the unified
                `BaseStageLabelGenerator` interface.
            include_debug: Whether to include `stage_debug` in returned samples.
                Defaults to `False`.
            default_num_stages: Default number of stages when inference fails.
                Defaults to 3 (move, grasp, place).
            action_chunk_size: Number of consecutive frames used to build the
                local action chunk anchored at the current sample. Must be >= 2.
            action_chunk_anchor: Chunk anchor mode. Only
                `"current_to_future"` is currently supported.
        """
        if action_chunk_size < 2:
            raise ValueError(
                f"`action_chunk_size` must be >= 2, got {action_chunk_size}."
            )
        if action_chunk_anchor != "current_to_future":
            raise ValueError(
                "`action_chunk_anchor` only supports 'current_to_future'. "
                f"Got {action_chunk_anchor!r}."
            )
        self.base_dataset = base_dataset
        self.label_generator = label_generator
        self.include_debug = include_debug
        self.action_chunk_size = action_chunk_size
        self.action_chunk_anchor = action_chunk_anchor
        self.num_stages = self._infer_num_stages(label_generator)
        if self.num_stages == 0:
            self.num_stages = default_num_stages

    def __len__(self) -> int:
        """Return the number of wrapped samples."""
        return len(self.base_dataset)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return a stage-labeled sample.

        Workflow:
        1. Read the base sample from `base_dataset[idx]`.
        2. Create a top-level copy so the original sample is not modified in place.
        3. Inject local action/state chunk context anchored at `idx`.
        4. Run the stage-label generator on the enriched sample.
        5. Attach normalized stage-label fields to the copied sample.

        Args:
            idx: Sample index.

        Returns:
            A copied sample dictionary augmented with stage-label fields.

        Raises:
            TypeError: If the base dataset does not return a mapping-like sample.
        """
        base_sample = self.base_dataset[idx]
        if not isinstance(base_sample, Mapping):
            raise TypeError(
                "StageLabeledDataset expects `base_dataset[idx]` to return a mapping-like sample. "
                f"Got {type(base_sample).__name__} at index {idx}."
            )

        labeled_sample = dict(base_sample)
        self._inject_local_action_chunk(sample=labeled_sample, idx=idx)
        stage_output = self.label_generator(labeled_sample)
        self._attach_stage_fields(labeled_sample, stage_output)

        if self.include_debug:
            labeled_sample["stage_debug"] = dict(stage_output.stage_debug)

        return labeled_sample

    def __getattr__(self, name: str) -> Any:
        """Delegate unknown attributes to the wrapped base dataset."""
        return getattr(self.base_dataset, name)

    def __repr__(self) -> str:
        """Return a concise representation of the wrapper."""
        return (
            f"{self.__class__.__name__}("
            f"base_dataset={self.base_dataset!r}, "
            f"label_generator={self.label_generator.__class__.__name__}, "
            f"include_debug={self.include_debug}, "
            f"action_chunk_size={self.action_chunk_size}, "
            f"action_chunk_anchor={self.action_chunk_anchor!r})"
        )

    def _inject_local_action_chunk(self, sample: dict[str, Any], idx: int) -> None:
        """Attach local chunk metadata used by chunk-based stage-label generators.

        Semantics:
        - `stage_label_chunk_complete`: actual chunk length equals requested length
        - `stage_label_chunk_valid`: structurally usable chunk, meaning:
          not cross-episode, action chunk exists, and chunk length is at least 2

        A chunk can therefore be valid but incomplete, which is expected near an
        episode tail or dataset boundary.
        """
        episode_index = sample.get("episode_index")
        action_rows: list[torch.Tensor] = []
        state_rows: list[torch.Tensor] = []
        state_chunk_available = True
        cross_episode = False

        for offset in range(self.action_chunk_size):
            current_idx = idx + offset
            if current_idx >= len(self.base_dataset):
                break

            chunk_sample = self.base_dataset[current_idx]
            if not isinstance(chunk_sample, Mapping):
                raise TypeError(
                    "StageLabeledDataset expects chunk samples to be mapping-like. "
                    f"Got {type(chunk_sample).__name__} at index {current_idx}."
                )

            if offset > 0 and episode_index is not None and chunk_sample.get("episode_index") != episode_index:
                cross_episode = True
                break

            action_value = self._lookup_chunk_value(ACTION, "actions", sample=chunk_sample)
            action_row, _, _ = extract_single_action_tensor_with_reason(action_value)
            if action_row is None:
                break
            action_rows.append(action_row)

            if state_chunk_available:
                state_value = self._lookup_chunk_value(OBS_STATE, "observation.state", sample=chunk_sample)
                state_row = extract_latest_state_tensor(state_value)
                if state_row is None:
                    state_chunk_available = False
                    state_rows.clear()
                else:
                    state_rows.append(state_row)

        chunk_size = len(action_rows)
        chunk_complete = chunk_size == self.action_chunk_size
        chunk_valid = (not cross_episode) and chunk_size >= 2
        sample["stage_label_chunk_valid"] = chunk_valid
        sample["stage_label_chunk_size"] = chunk_size
        sample["stage_label_chunk_requested_size"] = self.action_chunk_size
        sample["stage_label_chunk_cross_episode"] = cross_episode
        sample["stage_label_chunk_complete"] = chunk_complete
        sample["stage_label_anchor_index"] = idx

        if action_rows:
            sample["stage_label_action_chunk"] = torch.stack(action_rows, dim=0)
        if state_rows and len(state_rows) == len(action_rows):
            sample["stage_label_state_chunk"] = torch.stack(state_rows, dim=0)

    @staticmethod
    def _lookup_chunk_value(*keys: str, sample: Mapping[str, Any]) -> Any | None:
        """Return the first available chunk source among candidate keys."""
        for key in keys:
            if key in sample:
                return sample[key]
            current: Any = sample
            found = True
            for part in key.split("."):
                if not isinstance(current, Mapping) or part not in current:
                    found = False
                    break
                current = current[part]
            if found:
                return current
        return None

    def _attach_stage_fields(
        self,
        sample: dict[str, Any],
        stage_output: StageLabelOutput,
    ) -> None:
        """Attach normalized stage-label fields to a copied sample."""
        sample["stage_id"] = self._normalize_stage_id(stage_output)
        sample["stage_prob_target"] = self._normalize_stage_prob_target(stage_output)
        sample["stage_valid_mask"] = stage_output.stage_valid_mask
        sample["stage_source"] = stage_output.stage_source

    def _normalize_stage_id(self, stage_output: StageLabelOutput) -> torch.Tensor:
        """Return a collate-safe stage id tensor.

        Missing discrete labels are represented by a `-1` tensor whose shape
        follows `stage_valid_mask`. This keeps the placeholder compatible with
        the current scalar-label setup while also making it safer to extend to
        sequence-level labels in the future.
        """
        if stage_output.stage_id is not None:
            return stage_output.stage_id

        device = self._get_output_device(stage_output)
        return torch.full(
            size=tuple(stage_output.stage_valid_mask.shape),
            fill_value=-1,
            dtype=torch.long,
            device=device,
        )

    def _normalize_stage_prob_target(self, stage_output: StageLabelOutput) -> torch.Tensor:
        """Return a collate-safe stage probability tensor with unified dimensionality.

        CRITICAL FIX: All generators now return tensors with shape (K,), where K is the
        number of stages, regardless of generator type. This ensures:
        - Consistent tensor stacking in DataLoader
        - Safe batch collation without shape mismatches
        - Proper gradient flow during training
        - Validity mask controls whether supervision is used

        For generators that return probabilities (soft label):
            Returns the probability vector directly.

        For generators without probabilities (hard label):
            Returns a zero vector of shape (K,). The stage_valid_mask indicates
            whether this label should contribute to loss.

        Args:
            stage_output: StageLabelOutput from the generator.

        Returns:
            A tensor of shape (K,) with dtype float32, on the appropriate device.
        """
        if stage_output.stage_prob_target is not None:
            return stage_output.stage_prob_target

        device = self._get_output_device(stage_output)
        # ALL generators now return (K,) dimensional tensors for batch collation safety
        return torch.zeros(self.num_stages, dtype=torch.float32, device=device)

    def _get_output_device(self, stage_output: StageLabelOutput) -> torch.device:
        """Infer the most appropriate device for wrapper-created tensors.

        CRITICAL FIX: Enhanced device consistency checking to prevent device mismatch
        errors in multi-GPU training. Prioritizes tensors that are guaranteed to exist.

        Args:
            stage_output: StageLabelOutput instance.

        Returns:
            The torch device for the output tensors.

        Raises:
            RuntimeError: If no valid device can be inferred (should not happen in
                practice since stage_valid_mask is always present).
        """
        # Prioritize stage_valid_mask as it's always guaranteed to exist
        if stage_output.stage_valid_mask is not None:
            return stage_output.stage_valid_mask.device

        # Fallback to other tensors if present (defensive programming)
        if stage_output.stage_id is not None:
            return stage_output.stage_id.device
        if stage_output.stage_prob_target is not None:
            return stage_output.stage_prob_target.device

        raise RuntimeError(
            "Cannot infer device: all tensor fields are None. "
            "This indicates a corrupted StageLabelOutput."
        )

    def _infer_num_stages(self, label_generator: BaseStageLabelGenerator) -> int:
        """Infer the stage count from the generator when available.

        CRITICAL FIX: Improved robustness for num_stages detection to ensure
        stage_prob_target always has correct dimensionality.

        Args:
            label_generator: The stage label generator.

        Returns:
            Number of stages if detectable, otherwise 0 (caller must use default).
        """
        # Try to get stage_names dict
        stage_names = getattr(label_generator, "stage_names", None)
        if isinstance(stage_names, Mapping) and stage_names:
            return len(stage_names)

        # Try to get from config if available
        cfg = getattr(label_generator, "cfg", None)
        if cfg is not None:
            # Check if config has move/grasp/place stage IDs
            move_id = getattr(cfg, "move_stage_id", None)
            grasp_id = getattr(cfg, "grasp_stage_id", None)
            place_id = getattr(cfg, "place_stage_id", None)
            if move_id is not None and grasp_id is not None and place_id is not None:
                unique_ids = len(set(filter(lambda x: x is not None, [move_id, grasp_id, place_id])))
                if unique_ids == 3:
                    return 3

        return 0

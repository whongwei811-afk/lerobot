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

from .base import BaseStageLabelGenerator, StageLabelOutput


class StageLabeledDataset(torch.utils.data.Dataset):
    """Dataset wrapper that attaches action-based stage labels to each sample.

    This wrapper leaves the underlying dataset unchanged on disk and does not
    modify the base sample in place. Instead, it creates a shallow top-level copy
    of each retrieved sample, runs a stage-label generator on the original sample,
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
        """
        self.base_dataset = base_dataset
        self.label_generator = label_generator
        self.include_debug = include_debug
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
        3. Run the stage-label generator on the original sample.
        4. Attach normalized stage-label fields to the copied sample.

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
        stage_output = self.label_generator(base_sample)
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
            f"include_debug={self.include_debug})"
        )

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

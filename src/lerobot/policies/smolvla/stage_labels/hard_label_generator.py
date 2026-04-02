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

from dataclasses import dataclass, fields
from typing import Any, Mapping

import torch

from lerobot.utils.constants import ACTION, OBS_STATE

from .base import BaseStageLabelGenerator, StageLabelOutput
from .utils import (
    compute_mean_abs_motion,
    extract_action_chunk_tensor,
    extract_latest_state_tensor,
    lookup_sample_value,
    normalize_index_sequence,
    resolve_joint_groups,
    resolve_vector_indices,
)


@dataclass
class HardStageLabelGeneratorConfig:
    """Configuration for rule-based hard stage label generation.

    Attributes:
        state_key: Key used to read the current state from a sample.
        action_key_candidates: Candidate keys used to find the current action or
            an action chunk in a sample. The first matching key is used.
        move_stage_id: Integer id used for the `move` stage.
        grasp_stage_id: Integer id used for the `grasp` stage.
        place_stage_id: Integer id used for the `place` stage.
        gripper_state_indices: Optional indices of gripper-related dimensions in
            the state vector. If missing, no gripper-state heuristic is used.
        gripper_action_indices: Optional indices of gripper-related dimensions in
            the action vector. If missing, the last `default_gripper_width`
            dimensions are used as a fallback.
        default_gripper_width: Number of trailing action dimensions to interpret as
            gripper controls when `gripper_action_indices` is not provided.
        large_joint_indices: Optional indices for large/proximal arm joints in the
            action vector. If missing, non-gripper action dimensions are split in
            half to derive large and small joint groups.
        small_joint_indices: Optional indices for small/distal arm joints in the
            action vector. If missing, non-gripper action dimensions are split in
            half to derive large and small joint groups.
        gripper_is_closed_below_threshold: Whether smaller gripper state values
            indicate a more closed gripper.
        gripper_closed_threshold: Threshold for considering the gripper closed.
        gripper_open_threshold: Threshold for considering the gripper open.
        gripper_close_is_negative: Whether negative gripper action values mean the
            gripper is closing.
        gripper_close_action_threshold: Threshold for detecting a meaningful
            closing action.
        gripper_open_action_threshold: Threshold for detecting a meaningful opening
            action.
        require_gripper_state_for_grasp: Whether `grasp` classification requires
            gripper-state evidence in addition to closing-action evidence.
        grasp_max_arm_motion: Maximum average arm motion allowed for a `grasp`
            classification.
        grasp_max_small_joint_motion: Maximum average small-joint motion allowed
            for a `grasp` classification.
        place_max_large_joint_motion: Maximum average large-joint motion allowed
            for a `place` classification.
        place_min_small_joint_motion: Minimum average small-joint motion that can
            support a `place` classification when the gripper is already open.
        place_small_to_large_ratio: Minimum ratio between small-joint and
            large-joint motion that can indicate a fine `place` adjustment.
    """

    state_key: str = OBS_STATE
    action_key_candidates: tuple[str, ...] = (ACTION, "actions", "action_chunk", "future_actions")

    move_stage_id: int = 0
    grasp_stage_id: int = 1
    place_stage_id: int = 2

    gripper_state_indices: tuple[int, ...] | None = None
    gripper_action_indices: tuple[int, ...] | None = None
    default_gripper_width: int = 1

    large_joint_indices: tuple[int, ...] | None = None
    small_joint_indices: tuple[int, ...] | None = None

    gripper_is_closed_below_threshold: bool = False
    gripper_closed_threshold: float = 0.60
    gripper_open_threshold: float = 0.20

    gripper_close_is_negative: bool = True
    gripper_close_action_threshold: float = 0.05
    gripper_open_action_threshold: float = 0.05
    require_gripper_state_for_grasp: bool = True

    grasp_max_arm_motion: float = 0.15
    grasp_max_small_joint_motion: float = 0.10

    place_max_large_joint_motion: float = 0.10
    place_min_small_joint_motion: float = 0.01
    place_small_to_large_ratio: float = 0.80

    def __post_init__(self) -> None:
        """Validate configuration values for consistent hard-label generation."""
        stage_ids = (self.move_stage_id, self.grasp_stage_id, self.place_stage_id)
        if len(set(stage_ids)) != 3:
            raise ValueError(
                "`move_stage_id`, `grasp_stage_id`, and `place_stage_id` must be distinct. "
                f"Got {stage_ids}."
            )
        if self.default_gripper_width < 0:
            raise ValueError(
                f"`default_gripper_width` must be >= 0, got {self.default_gripper_width}."
            )
        if not self.action_key_candidates:
            raise ValueError("`action_key_candidates` must not be empty.")

        self._validate_non_negative("gripper_close_action_threshold", self.gripper_close_action_threshold)
        self._validate_non_negative("gripper_open_action_threshold", self.gripper_open_action_threshold)
        self._validate_non_negative("grasp_max_arm_motion", self.grasp_max_arm_motion)
        self._validate_non_negative("grasp_max_small_joint_motion", self.grasp_max_small_joint_motion)
        self._validate_non_negative("place_max_large_joint_motion", self.place_max_large_joint_motion)
        self._validate_non_negative("place_min_small_joint_motion", self.place_min_small_joint_motion)
        self._validate_non_negative("place_small_to_large_ratio", self.place_small_to_large_ratio)

    def stage_name_by_id(self) -> dict[int, str]:
        """Return the stage-name mapping implied by the configured stage ids."""
        return {
            self.move_stage_id: "move",
            self.grasp_stage_id: "grasp",
            self.place_stage_id: "place",
        }

    @staticmethod
    def _validate_non_negative(name: str, value: float) -> None:
        """Validate that a scalar threshold-like value is non-negative."""
        if value < 0:
            raise ValueError(f"`{name}` must be >= 0, got {value}.")

    @classmethod
    def from_config(cls, cfg: HardStageLabelGeneratorConfig | Mapping[str, Any] | Any | None) -> HardStageLabelGeneratorConfig:
        """Build a config from a dataclass instance, mapping, or config-like object."""
        if cfg is None:
            return cls()
        if isinstance(cfg, cls):
            return cfg

        field_names = {field_.name for field_ in fields(cls)}
        if isinstance(cfg, Mapping):
            data = {key: value for key, value in cfg.items() if key in field_names}
            return cls(**data)

        data = {name: getattr(cfg, name) for name in field_names if hasattr(cfg, name)}
        return cls(**data)


class HardStageLabelGenerator(BaseStageLabelGenerator):
    """Rule-based hard-label generator for coarse SmolVLA stage supervision.

    The first implementation intentionally stays simple and model-agnostic. It only
    consumes sample-level state/action data and emits one of three hard labels:

    - `0 = move`
    - `1 = grasp`
    - `2 = place`

    The generator prefers action-derived signals and uses gripper state as an
    additional hint when available.
    """

    def __init__(self, cfg: HardStageLabelGeneratorConfig | Mapping[str, Any] | Any | None = None):
        self.cfg = HardStageLabelGeneratorConfig.from_config(cfg)
        self.stage_names = self.cfg.stage_name_by_id()

    def generate(self, sample: Mapping[str, Any], **kwargs: Any) -> StageLabelOutput:
        """Generate a hard stage label from a single sample.

        Args:
            sample: A single dataset sample or sample-like mapping.
            **kwargs: Unused extension point for future generator-specific options.

        Returns:
            A `StageLabelOutput` carrying a single hard label when enough
            information is available, otherwise an invalid output.
        """
        del kwargs

        state_value = lookup_sample_value(self.cfg.state_key, sample)
        action_value, action_key = self._find_action_value(sample)

        state_tensor = extract_latest_state_tensor(state_value)
        action_chunk = extract_action_chunk_tensor(action_value)
        if action_chunk is None:
            return self._invalid_output(
                reason="missing_action",
                sample=sample,
                action_key=action_key,
                state_available=state_tensor is not None,
            )

        action_dim = int(action_chunk.shape[-1])
        gripper_action_indices = self._resolve_gripper_action_indices(action_dim)
        if not gripper_action_indices:
            return self._invalid_output(
                reason="missing_gripper_action_dims",
                sample=sample,
                action_key=action_key,
                state_available=state_tensor is not None,
            )

        arm_indices = [index for index in range(action_dim) if index not in gripper_action_indices]
        large_joint_indices, small_joint_indices = resolve_joint_groups(
            action_dim=action_dim,
            gripper_action_indices=gripper_action_indices,
            large_joint_indices=self.cfg.large_joint_indices,
            small_joint_indices=self.cfg.small_joint_indices,
        )

        gripper_action = action_chunk[:, gripper_action_indices]
        gripper_action_mean = float(gripper_action.mean().item())
        if self.cfg.gripper_close_is_negative:
            close_command = gripper_action_mean <= -self.cfg.gripper_close_action_threshold
            open_command = gripper_action_mean >= self.cfg.gripper_open_action_threshold
        else:
            close_command = gripper_action_mean >= self.cfg.gripper_close_action_threshold
            open_command = gripper_action_mean <= -self.cfg.gripper_open_action_threshold

        arm_motion = compute_mean_abs_motion(action_chunk, arm_indices)
        large_joint_motion = compute_mean_abs_motion(action_chunk, large_joint_indices)
        small_joint_motion = compute_mean_abs_motion(action_chunk, small_joint_indices)

        gripper_state_value = None
        gripper_is_closed = None
        gripper_is_open = None
        gripper_state_indices = resolve_vector_indices(
            vector_dim=None if state_tensor is None else int(state_tensor.shape[-1]),
            configured_indices=self.cfg.gripper_state_indices,
        )
        if state_tensor is not None and gripper_state_indices:
            gripper_state_value = float(state_tensor[gripper_state_indices].mean().item())
            gripper_is_closed = self._is_gripper_closed(gripper_state_value)
            gripper_is_open = self._is_gripper_open(gripper_state_value)

        stage_id, decision_reason = self._classify_stage(
            close_command=close_command,
            open_command=open_command,
            arm_motion=arm_motion,
            large_joint_motion=large_joint_motion,
            small_joint_motion=small_joint_motion,
            gripper_is_closed=gripper_is_closed,
            gripper_is_open=gripper_is_open,
        )

        output_device = action_chunk.device
        return StageLabelOutput(
            stage_id=torch.tensor(stage_id, dtype=torch.long, device=output_device),
            stage_prob_target=None,
            stage_valid_mask=torch.tensor(True, dtype=torch.bool, device=output_device),
            stage_source="hard",
            stage_debug={
                "decision_reason": decision_reason,
                "stage_name": self.stage_names.get(stage_id, "unknown"),
                "action_key": action_key,
                "state_key": self.cfg.state_key,
                "gripper_action_mean": gripper_action_mean,
                "arm_motion": arm_motion,
                "large_joint_motion": large_joint_motion,
                "small_joint_motion": small_joint_motion,
                "gripper_state_value": gripper_state_value,
                "gripper_is_closed": gripper_is_closed,
                "gripper_is_open": gripper_is_open,
                "gripper_action_indices": gripper_action_indices,
                "gripper_state_indices": gripper_state_indices,
                "large_joint_indices": large_joint_indices,
                "small_joint_indices": small_joint_indices,
                "arm_indices": arm_indices,
            },
        )

    def _classify_stage(
        self,
        *,
        close_command: bool,
        open_command: bool,
        arm_motion: float,
        large_joint_motion: float,
        small_joint_motion: float,
        gripper_is_closed: bool | None,
        gripper_is_open: bool | None,
    ) -> tuple[int, str]:
        """Classify a sample into move/grasp/place using coarse heuristics."""
        grasp_state_condition = (
            gripper_is_closed is False
            if self.cfg.require_gripper_state_for_grasp
            else gripper_is_closed is not True
        )
        if (
            close_command
            and arm_motion <= self.cfg.grasp_max_arm_motion
            and small_joint_motion <= self.cfg.grasp_max_small_joint_motion
            and grasp_state_condition
        ):
            return self.cfg.grasp_stage_id, "closing_gripper_with_limited_arm_motion"

        place_from_open_command = open_command and (
            large_joint_motion <= self.cfg.place_max_large_joint_motion
            or small_joint_motion >= max(
                self.cfg.place_min_small_joint_motion,
                large_joint_motion * self.cfg.place_small_to_large_ratio,
            )
        )
        if place_from_open_command:
            return self.cfg.place_stage_id, "opening_gripper_near_fine_adjustment"

        place_from_state = (
            gripper_is_open is True
            and large_joint_motion <= self.cfg.place_max_large_joint_motion
            and small_joint_motion >= self.cfg.place_min_small_joint_motion
        )
        if place_from_state:
            return self.cfg.place_stage_id, "gripper_open_with_small_joint_adjustment"

        return self.cfg.move_stage_id, "default_move"

    def _invalid_output(
        self,
        *,
        reason: str,
        sample: Mapping[str, Any],
        action_key: str | None,
        state_available: bool,
    ) -> StageLabelOutput:
        """Return a unified invalid label output when the sample lacks signal."""
        return StageLabelOutput(
            stage_id=None,
            stage_prob_target=None,
            stage_valid_mask=torch.tensor(False, dtype=torch.bool),
            stage_source="hard",
            stage_debug={
                "decision_reason": reason,
                "action_key": action_key,
                "state_key": self.cfg.state_key,
                "available_keys": sorted(sample.keys()),
                "state_available": state_available,
            },
        )

    def _find_action_value(self, sample: Mapping[str, Any]) -> tuple[Any | None, str | None]:
        """Find the first configured action-like field present in the sample."""
        for key in self.cfg.action_key_candidates:
            value = lookup_sample_value(key, sample)
            if value is not None:
                return value, key
        return None, None

    def _resolve_gripper_action_indices(self, action_dim: int) -> list[int]:
        """Resolve gripper action indices with a trailing-dimension fallback."""
        configured = normalize_index_sequence(self.cfg.gripper_action_indices, action_dim)
        if configured:
            return configured
        width = min(max(self.cfg.default_gripper_width, 0), action_dim)
        if width == 0:
            return []
        return list(range(action_dim - width, action_dim))

    def _is_gripper_closed(self, gripper_state_value: float) -> bool:
        """Check whether a gripper state value should be interpreted as closed."""
        if self.cfg.gripper_is_closed_below_threshold:
            return gripper_state_value <= self.cfg.gripper_closed_threshold
        return gripper_state_value >= self.cfg.gripper_closed_threshold

    def _is_gripper_open(self, gripper_state_value: float) -> bool:
        """Check whether a gripper state value should be interpreted as open."""
        if self.cfg.gripper_is_closed_below_threshold:
            return gripper_state_value >= self.cfg.gripper_open_threshold
        return gripper_state_value <= self.cfg.gripper_open_threshold

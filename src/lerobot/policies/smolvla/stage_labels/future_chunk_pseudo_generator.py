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

from lerobot.utils.constants import ACTION

from .base import BaseStageLabelGenerator, StageLabelOutput
from .utils import (
    compute_mean_abs_motion,
    extract_action_chunk_tensor,
    lookup_sample_value,
    normalize_index_sequence,
    resolve_joint_groups,
)


@dataclass
class FutureChunkPseudoLabelGeneratorConfig:
    """Configuration for action-target-chunk pseudo stage label generation.

    Attributes:
        future_action_key_candidates: Candidate keys used to find the action target
            chunk aligned with the current sample. For backward compatibility this
            field keeps the historical `future_*` name, but the matched tensor is
            only guaranteed to be the sample-aligned action target chunk, not
            necessarily an explicitly named `future_actions` tensor.
        min_future_steps: Minimum number of action-target-chunk steps required to
            generate a pseudo label.
        move_stage_id: Integer id used for the `move` stage.
        grasp_stage_id: Integer id used for the `grasp` stage.
        place_stage_id: Integer id used for the `place` stage.
        gripper_action_indices: Optional indices of gripper-related dimensions in
            the future action chunk. If missing, the trailing
            `default_gripper_width` dimensions are used as a fallback.
        default_gripper_width: Number of trailing action dimensions to interpret as
            gripper controls when `gripper_action_indices` is not provided.
        large_joint_indices: Optional indices for large/proximal arm joints in the
            action vector.
        small_joint_indices: Optional indices for small/distal arm joints in the
            action vector.
        gripper_close_is_negative: Whether negative gripper action values mean the
            gripper is closing.
        gripper_close_action_threshold: Threshold for detecting a meaningful
            closing bias from the action target chunk.
        gripper_open_action_threshold: Threshold for detecting a meaningful opening
            bias from the action target chunk.
        gripper_trend_threshold: Threshold for detecting a meaningful signed
            gripper trend across the chunk.
        min_gripper_energy: Minimum gripper motion energy required before grasp or
            place can be emitted.
        min_large_joint_energy: Minimum large-joint motion energy required before
            `move` can be considered dominant.
        min_small_joint_energy: Minimum small-joint motion energy required before
            `grasp` or `place` can be considered dominant.
        large_joint_dominance_ratio: Minimum ratio for large-joint energy to be
            considered dominant over small-joint energy.
        small_joint_dominance_ratio: Minimum ratio for small-joint energy to be
            considered dominant over large-joint energy.
        allow_invalid_when_ambiguous: Whether ambiguous chunks should return an
            invalid output instead of forcing a stage label.
        require_explicit_move_evidence: Whether `move` should only be emitted when
            large-joint dominance is explicitly observed.
    """

    future_action_key_candidates: tuple[str, ...] = (
        "future_actions",
        "action_chunk",
        "actions",
        ACTION,
    )
    min_future_steps: int = 2

    move_stage_id: int = 0
    grasp_stage_id: int = 1
    place_stage_id: int = 2

    gripper_action_indices: tuple[int, ...] | None = None
    default_gripper_width: int = 1

    large_joint_indices: tuple[int, ...] | None = None
    small_joint_indices: tuple[int, ...] | None = None

    gripper_close_is_negative: bool = True
    gripper_close_action_threshold: float = 0.05
    gripper_open_action_threshold: float = 0.05
    gripper_trend_threshold: float = 0.02

    min_gripper_energy: float = 0.01
    min_large_joint_energy: float = 0.01
    min_small_joint_energy: float = 0.01
    large_joint_dominance_ratio: float = 1.20
    small_joint_dominance_ratio: float = 1.10
    allow_invalid_when_ambiguous: bool = True
    require_explicit_move_evidence: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values for consistent pseudo-label generation."""
        stage_ids = (self.move_stage_id, self.grasp_stage_id, self.place_stage_id)
        if len(set(stage_ids)) != 3:
            raise ValueError(
                "`move_stage_id`, `grasp_stage_id`, and `place_stage_id` must be distinct. "
                f"Got {stage_ids}."
            )
        if not self.future_action_key_candidates:
            raise ValueError("`future_action_key_candidates` must not be empty.")
        if self.min_future_steps <= 0:
            raise ValueError(f"`min_future_steps` must be > 0, got {self.min_future_steps}.")
        if self.default_gripper_width < 0:
            raise ValueError(
                f"`default_gripper_width` must be >= 0, got {self.default_gripper_width}."
            )

        self._validate_non_negative("gripper_close_action_threshold", self.gripper_close_action_threshold)
        self._validate_non_negative("gripper_open_action_threshold", self.gripper_open_action_threshold)
        self._validate_non_negative("gripper_trend_threshold", self.gripper_trend_threshold)
        self._validate_non_negative("min_gripper_energy", self.min_gripper_energy)
        self._validate_non_negative("min_large_joint_energy", self.min_large_joint_energy)
        self._validate_non_negative("min_small_joint_energy", self.min_small_joint_energy)
        self._validate_non_negative("large_joint_dominance_ratio", self.large_joint_dominance_ratio)
        self._validate_non_negative("small_joint_dominance_ratio", self.small_joint_dominance_ratio)

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
    def from_config(
        cls, cfg: FutureChunkPseudoLabelGeneratorConfig | Mapping[str, Any] | Any | None
    ) -> FutureChunkPseudoLabelGeneratorConfig:
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


class FutureChunkPseudoLabelGenerator(BaseStageLabelGenerator):
    """Generate pseudo stage labels from sample-aligned action-target-chunk statistics.

    This generator is model-agnostic and only consumes a sample-level action target
    chunk associated with the current sample. For backward compatibility, the
    configuration still refers to `future_action_key_candidates`, but the retrieved
    tensor is best interpreted as a sample-aligned supervision chunk rather than a
    guaranteed explicit `future_actions` field.

    It computes coarse motion-energy statistics for:

    - large-joint action dimensions
    - small-joint action dimensions
    - gripper action dimensions

    The resulting pseudo label is selected using configured dominance heuristics.
    """

    def __init__(
        self, cfg: FutureChunkPseudoLabelGeneratorConfig | Mapping[str, Any] | Any | None = None
    ):
        self.cfg = FutureChunkPseudoLabelGeneratorConfig.from_config(cfg)
        self.stage_names = self.cfg.stage_name_by_id()

    def generate(self, sample: Mapping[str, Any], **kwargs: Any) -> StageLabelOutput:
        """Generate a pseudo stage label from a single sample.

        Args:
            sample: A single dataset sample or sample-like mapping.
            **kwargs: Unused extension point for future generator-specific options.

        Returns:
            A `StageLabelOutput` containing a pseudo hard label when a usable
            sample-aligned action target chunk is available and long enough,
            otherwise an invalid output.
        """
        del kwargs

        future_chunk_value, action_key = self._find_future_chunk_value(sample)
        future_chunk = extract_action_chunk_tensor(future_chunk_value)
        if future_chunk is None:
            return self._invalid_output(
                reason="missing_future_chunk",
                sample=sample,
                action_key=action_key,
            )

        if future_chunk.shape[0] < self.cfg.min_future_steps:
            return self._invalid_output(
                reason="insufficient_future_chunk_length",
                sample=sample,
                action_key=action_key,
                chunk_length=int(future_chunk.shape[0]),
            )

        action_dim = int(future_chunk.shape[-1])
        gripper_action_indices = self._resolve_gripper_action_indices(action_dim)
        if not gripper_action_indices:
            return self._invalid_output(
                reason="missing_gripper_action_dims",
                sample=sample,
                action_key=action_key,
                chunk_length=int(future_chunk.shape[0]),
            )

        arm_indices = [index for index in range(action_dim) if index not in gripper_action_indices]
        large_joint_indices, small_joint_indices = resolve_joint_groups(
            action_dim=action_dim,
            gripper_action_indices=gripper_action_indices,
            large_joint_indices=self.cfg.large_joint_indices,
            small_joint_indices=self.cfg.small_joint_indices,
        )

        large_joint_energy = compute_mean_abs_motion(future_chunk, large_joint_indices)
        small_joint_energy = compute_mean_abs_motion(future_chunk, small_joint_indices)
        gripper_energy = compute_mean_abs_motion(future_chunk, gripper_action_indices)

        gripper_trend_stats = self._compute_gripper_trend_stats(
            future_chunk[:, gripper_action_indices]
        )
        close_command, open_command = self._infer_gripper_direction(
            gripper_energy=gripper_energy,
            gripper_trend_stats=gripper_trend_stats,
        )

        stage_id, decision_reason = self._classify_stage(
            large_joint_energy=large_joint_energy,
            small_joint_energy=small_joint_energy,
            gripper_energy=gripper_energy,
            close_command=close_command,
            open_command=open_command,
        )
        if stage_id is None:
            return self._invalid_output(
                reason=decision_reason,
                sample=sample,
                action_key=action_key,
                chunk_length=int(future_chunk.shape[0]),
                extra_debug={
                    "chunk_semantics": "action_target_chunk",
                    "large_joint_energy": large_joint_energy,
                    "small_joint_energy": small_joint_energy,
                    "gripper_energy": gripper_energy,
                    "close_command": close_command,
                    "open_command": open_command,
                    **gripper_trend_stats,
                    "gripper_action_indices": gripper_action_indices,
                    "large_joint_indices": large_joint_indices,
                    "small_joint_indices": small_joint_indices,
                    "arm_indices": arm_indices,
                },
            )

        output_device = future_chunk.device
        return StageLabelOutput(
            stage_id=torch.tensor(stage_id, dtype=torch.long, device=output_device),
            stage_prob_target=None,
            stage_valid_mask=torch.tensor(True, dtype=torch.bool, device=output_device),
            stage_source="future_chunk_pseudo",
            stage_debug={
                "decision_reason": decision_reason,
                "stage_name": self.stage_names.get(stage_id, "unknown"),
                "action_key": action_key,
                "chunk_semantics": "action_target_chunk",
                "chunk_length": int(future_chunk.shape[0]),
                "large_joint_energy": large_joint_energy,
                "small_joint_energy": small_joint_energy,
                "gripper_energy": gripper_energy,
                "close_command": close_command,
                "open_command": open_command,
                **gripper_trend_stats,
                "gripper_action_indices": gripper_action_indices,
                "large_joint_indices": large_joint_indices,
                "small_joint_indices": small_joint_indices,
                "arm_indices": arm_indices,
            },
        )

    def _classify_stage(
        self,
        *,
        large_joint_energy: float,
        small_joint_energy: float,
        gripper_energy: float,
        close_command: bool,
        open_command: bool,
    ) -> tuple[int | None, str]:
        """Classify an action-target chunk into move/grasp/place using energy dominance."""
        small_joint_dominant = (
            small_joint_energy >= self.cfg.min_small_joint_energy
            and small_joint_energy >= large_joint_energy * self.cfg.small_joint_dominance_ratio
        )
        large_joint_dominant = (
            large_joint_energy >= self.cfg.min_large_joint_energy
            and large_joint_energy >= small_joint_energy * self.cfg.large_joint_dominance_ratio
        )
        has_gripper_signal = gripper_energy >= self.cfg.min_gripper_energy

        if close_command and has_gripper_signal and small_joint_dominant:
            return self.cfg.grasp_stage_id, "closing_gripper_with_small_joint_dominance"

        if open_command and has_gripper_signal and small_joint_dominant:
            return self.cfg.place_stage_id, "opening_gripper_with_small_joint_dominance"

        if large_joint_dominant:
            return self.cfg.move_stage_id, "large_joint_dominance"

        if self.cfg.require_explicit_move_evidence:
            if self.cfg.allow_invalid_when_ambiguous:
                if has_gripper_signal and not (close_command or open_command):
                    return None, "ambiguous_gripper_signal_without_directional_consensus"
                if small_joint_energy >= self.cfg.min_small_joint_energy:
                    return None, "weak_or_non_dominant_small_joint_signal"
                if large_joint_energy >= self.cfg.min_large_joint_energy:
                    return None, "no_dominant_motion_pattern"
                return None, "weak_signal_across_all_motion_groups"
            return self.cfg.move_stage_id, "fallback_move_without_explicit_dominance"

        return self.cfg.move_stage_id, "default_move_without_explicit_requirement"

    def _compute_gripper_trend_stats(self, gripper_chunk: torch.Tensor) -> dict[str, float]:
        """Compute lightweight gripper trend statistics over the chunk.

        Args:
            gripper_chunk: Gripper-only action chunk with shape `(T, D_gripper)`.

        Returns:
            A dictionary of scalar trend statistics used for debugging and for
            joint direction inference.
        """
        if gripper_chunk.ndim != 2:
            raise ValueError(
                f"`gripper_chunk` must have shape (T, D_gripper), got {tuple(gripper_chunk.shape)}."
            )

        num_steps = int(gripper_chunk.shape[0])
        midpoint = max(1, num_steps // 2)
        first_half = gripper_chunk[:midpoint]
        second_half = gripper_chunk[midpoint:] if midpoint < num_steps else gripper_chunk[-1:]

        gripper_mean = float(gripper_chunk.mean().item())
        first_half_mean = float(first_half.mean().item())
        second_half_mean = float(second_half.mean().item())
        end_minus_start = float((gripper_chunk[-1] - gripper_chunk[0]).mean().item())

        if num_steps >= 2:
            mean_diff = float(gripper_chunk.diff(dim=0).mean().item())
        else:
            mean_diff = 0.0

        signed_trend = 0.5 * (second_half_mean - first_half_mean) + 0.5 * mean_diff
        return {
            "gripper_mean": gripper_mean,
            "gripper_first_half_mean": first_half_mean,
            "gripper_second_half_mean": second_half_mean,
            "gripper_end_minus_start": end_minus_start,
            "gripper_mean_diff": mean_diff,
            "gripper_signed_trend": signed_trend,
        }

    def _infer_gripper_direction(
        self,
        *,
        gripper_energy: float,
        gripper_trend_stats: Mapping[str, float],
    ) -> tuple[bool, bool]:
        """Infer closing/opening direction using gripper trend and magnitude signals.

        CRITICAL FIX: Relaxed gripper direction logic from ALL conditions (AND) to
        2-of-3 heuristic (OR-based voting). This prevents false negatives when only
        some directional signals are present.

        The three signals are:
        1. signed_trend: Overall trend across the chunk
        2. end_minus_start: Start-to-end progression
        3. magnitude signals: Whether values cross threshold

        Args:
            gripper_energy: Mean absolute gripper motion.
            gripper_trend_stats: Dict of gripper statistics.

        Returns:
            Tuple of (close_command, open_command) booleans.
        """
        has_gripper_signal = gripper_energy >= self.cfg.min_gripper_energy
        if not has_gripper_signal:
            return False, False

        gripper_mean = gripper_trend_stats["gripper_mean"]
        second_half_mean = gripper_trend_stats["gripper_second_half_mean"]
        signed_trend = gripper_trend_stats["gripper_signed_trend"]
        end_minus_start = gripper_trend_stats["gripper_end_minus_start"]

        if self.cfg.gripper_close_is_negative:
            # For closing: negative direction is closing
            # 2-of-3 voting: need 2 out of 3 signals to agree
            close_trend_signal = signed_trend <= -self.cfg.gripper_trend_threshold
            close_progression_signal = end_minus_start <= -self.cfg.gripper_trend_threshold
            close_magnitude_signal = (
                second_half_mean <= -self.cfg.gripper_close_action_threshold
                or gripper_mean <= -self.cfg.gripper_close_action_threshold
            )
            # Use voting: at least 2 of 3 signals must be true
            close_signal_count = sum([close_trend_signal, close_progression_signal, close_magnitude_signal])
            close_command = close_signal_count >= 2

            # For opening: positive direction is opening
            open_trend_signal = signed_trend >= self.cfg.gripper_trend_threshold
            open_progression_signal = end_minus_start >= self.cfg.gripper_trend_threshold
            open_magnitude_signal = (
                second_half_mean >= self.cfg.gripper_open_action_threshold
                or gripper_mean >= self.cfg.gripper_open_action_threshold
            )
            open_signal_count = sum([open_trend_signal, open_progression_signal, open_magnitude_signal])
            open_command = open_signal_count >= 2
        else:
            # For closing: positive direction is closing (opposite sign convention)
            close_trend_signal = signed_trend >= self.cfg.gripper_trend_threshold
            close_progression_signal = end_minus_start >= self.cfg.gripper_trend_threshold
            close_magnitude_signal = (
                second_half_mean >= self.cfg.gripper_close_action_threshold
                or gripper_mean >= self.cfg.gripper_close_action_threshold
            )
            close_signal_count = sum([close_trend_signal, close_progression_signal, close_magnitude_signal])
            close_command = close_signal_count >= 2

            # For opening: negative direction is opening
            open_trend_signal = signed_trend <= -self.cfg.gripper_trend_threshold
            open_progression_signal = end_minus_start <= -self.cfg.gripper_trend_threshold
            open_magnitude_signal = (
                second_half_mean <= -self.cfg.gripper_open_action_threshold
                or gripper_mean <= -self.cfg.gripper_open_action_threshold
            )
            open_signal_count = sum([open_trend_signal, open_progression_signal, open_magnitude_signal])
            open_command = open_signal_count >= 2

        return close_command, open_command

    def _invalid_output(
        self,
        *,
        reason: str,
        sample: Mapping[str, Any],
        action_key: str | None,
        chunk_length: int | None = None,
        extra_debug: dict[str, Any] | None = None,
    ) -> StageLabelOutput:
        """Return a unified invalid label output when no usable target chunk exists."""
        stage_debug = {
            "decision_reason": reason,
            "action_key": action_key,
            "chunk_semantics": "action_target_chunk",
            "chunk_length": chunk_length,
            "available_keys": sorted(sample.keys()),
        }
        if extra_debug:
            stage_debug.update(extra_debug)
        return StageLabelOutput(
            stage_id=None,
            stage_prob_target=None,
            stage_valid_mask=torch.tensor(False, dtype=torch.bool),
            stage_source="future_chunk_pseudo",
            stage_debug=stage_debug,
        )

    def _find_future_chunk_value(self, sample: Mapping[str, Any]) -> tuple[Any | None, str | None]:
        """Find the first configured action-target-chunk field present in the sample."""
        for key in self.cfg.future_action_key_candidates:
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

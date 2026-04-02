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
class SoftStageLabelGeneratorConfig:
    """Configuration for action-target-chunk soft stage label generation.

    Attributes:
        action_chunk_key_candidates: Candidate keys used to find the sample-aligned
            action target chunk. The first matching key is used.
        min_chunk_steps: Minimum number of chunk steps required to generate a soft
            label distribution.
        move_stage_id: Integer id used for the `move` stage.
        grasp_stage_id: Integer id used for the `grasp` stage.
        place_stage_id: Integer id used for the `place` stage.
            These ids are semantic labels carried in config/debug metadata. The
            generated probability vector itself always follows the semantic class
            order `[move, grasp, place]`, exposed through `stage_order`.
        gripper_action_indices: Optional indices of gripper-related dimensions in
            the action target chunk. If missing, the trailing
            `default_gripper_width` dimensions are used as a fallback.
        default_gripper_width: Number of trailing action dimensions to interpret as
            gripper controls when `gripper_action_indices` is not provided.
        large_joint_indices: Optional indices for large/proximal arm joints in the
            action vector.
        small_joint_indices: Optional indices for small/distal arm joints in the
            action vector.
        gripper_close_is_negative: Whether negative gripper action values mean the
            gripper is closing.
        use_gripper_trend_signal: Whether close/open trend statistics should
            contribute to the soft stage scores.
        gripper_trend_threshold: Threshold used to normalize directional gripper
            trend strength.
        min_total_energy: Minimum combined motion energy required to produce a
            valid soft label distribution.
        temperature: Softmax temperature applied to the stage logits.
        move_bias: Additive bias for the `move` logit.
        grasp_bias: Additive bias for the `grasp` logit.
        place_bias: Additive bias for the `place` logit.
        move_large_joint_weight: Weight for large-joint energy ratio in the `move`
            logit.
        move_small_joint_weight: Weight for small-joint energy ratio in the
            `move` logit.
        move_gripper_weight: Weight for gripper energy ratio in the `move` logit.
        grasp_small_joint_weight: Weight for small-joint energy ratio in the
            `grasp` logit.
        grasp_gripper_weight: Weight for gripper energy ratio in the `grasp`
            logit.
        grasp_close_trend_weight: Weight for closing trend strength in the `grasp`
            logit.
        grasp_open_trend_penalty: Penalty for opening trend strength in the
            `grasp` logit.
        place_small_joint_weight: Weight for small-joint energy ratio in the
            `place` logit.
        place_gripper_weight: Weight for gripper energy ratio in the `place`
            logit.
        place_open_trend_weight: Weight for opening trend strength in the `place`
            logit.
        place_close_trend_penalty: Penalty for closing trend strength in the
            `place` logit.
    """

    action_chunk_key_candidates: tuple[str, ...] = (
        "future_actions",
        "action_chunk",
        "actions",
        ACTION,
    )
    min_chunk_steps: int = 2

    move_stage_id: int = 0
    grasp_stage_id: int = 1
    place_stage_id: int = 2

    gripper_action_indices: tuple[int, ...] | None = None
    default_gripper_width: int = 1

    large_joint_indices: tuple[int, ...] | None = None
    small_joint_indices: tuple[int, ...] | None = None

    gripper_close_is_negative: bool = True
    use_gripper_trend_signal: bool = True
    gripper_trend_threshold: float = 0.02

    min_total_energy: float = 1e-4
    temperature: float = 1.0

    move_bias: float = 0.0
    grasp_bias: float = 0.0
    place_bias: float = 0.0

    move_large_joint_weight: float = 2.0
    move_small_joint_weight: float = 0.7
    move_gripper_weight: float = 0.7

    grasp_small_joint_weight: float = 1.2
    grasp_gripper_weight: float = 1.0
    grasp_close_trend_weight: float = 1.5
    grasp_open_trend_penalty: float = 0.6

    place_small_joint_weight: float = 1.2
    place_gripper_weight: float = 1.0
    place_open_trend_weight: float = 1.5
    place_close_trend_penalty: float = 0.6

    def __post_init__(self) -> None:
        """Validate configuration values for consistent soft-label generation."""
        stage_ids = (self.move_stage_id, self.grasp_stage_id, self.place_stage_id)
        if len(set(stage_ids)) != 3:
            raise ValueError(
                "`move_stage_id`, `grasp_stage_id`, and `place_stage_id` must be distinct. "
                f"Got {stage_ids}."
            )
        if not self.action_chunk_key_candidates:
            raise ValueError("`action_chunk_key_candidates` must not be empty.")
        if self.min_chunk_steps <= 0:
            raise ValueError(f"`min_chunk_steps` must be > 0, got {self.min_chunk_steps}.")
        if self.default_gripper_width < 0:
            raise ValueError(
                f"`default_gripper_width` must be >= 0, got {self.default_gripper_width}."
            )

        self._validate_positive("temperature", self.temperature)
        self._validate_non_negative("gripper_trend_threshold", self.gripper_trend_threshold)
        self._validate_non_negative("min_total_energy", self.min_total_energy)

    @staticmethod
    def _validate_positive(name: str, value: float) -> None:
        """Validate that a scalar parameter is strictly positive."""
        if value <= 0:
            raise ValueError(f"`{name}` must be > 0, got {value}.")

    @staticmethod
    def _validate_non_negative(name: str, value: float) -> None:
        """Validate that a scalar parameter is non-negative."""
        if value < 0:
            raise ValueError(f"`{name}` must be >= 0, got {value}.")

    @classmethod
    def from_config(
        cls, cfg: SoftStageLabelGeneratorConfig | Mapping[str, Any] | Any | None
    ) -> SoftStageLabelGeneratorConfig:
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


class SoftStageLabelGenerator(BaseStageLabelGenerator):
    """Generate soft stage-label distributions from action-target-chunk statistics.

    The generator is model-agnostic and produces a normalized probability vector
    over the three stage classes using coarse action-group statistics and optional
    gripper direction trends.

    The output probability vector always uses the semantic class order
    `[move, grasp, place]`. This ordering is reported in `stage_debug["stage_order"]`
    and is independent from the numeric values stored in
    `move_stage_id/grasp_stage_id/place_stage_id`.
    """

    def __init__(self, cfg: SoftStageLabelGeneratorConfig | Mapping[str, Any] | Any | None = None):
        self.cfg = SoftStageLabelGeneratorConfig.from_config(cfg)
        self.stage_names = {
            self.cfg.move_stage_id: "move",
            self.cfg.grasp_stage_id: "grasp",
            self.cfg.place_stage_id: "place",
        }

    def generate(self, sample: Mapping[str, Any], **kwargs: Any) -> StageLabelOutput:
        """Generate a soft stage label distribution from a single sample.

        Args:
            sample: A single dataset sample or sample-like mapping.
            **kwargs: Unused extension point for future generator-specific options.

        Returns:
            A `StageLabelOutput` containing `stage_prob_target` when a usable
            action target chunk is available, otherwise an invalid output.
        """
        del kwargs

        chunk_value, action_key = self._find_action_chunk_value(sample)
        action_chunk = extract_action_chunk_tensor(chunk_value)
        if action_chunk is None:
            return self._invalid_output(
                reason="missing_action_target_chunk",
                sample=sample,
                action_key=action_key,
            )

        if action_chunk.shape[0] < self.cfg.min_chunk_steps:
            return self._invalid_output(
                reason="insufficient_action_target_chunk_length",
                sample=sample,
                action_key=action_key,
                chunk_length=int(action_chunk.shape[0]),
            )

        action_dim = int(action_chunk.shape[-1])
        gripper_action_indices = self._resolve_gripper_action_indices(action_dim)
        if not gripper_action_indices:
            return self._invalid_output(
                reason="missing_gripper_action_dims",
                sample=sample,
                action_key=action_key,
                chunk_length=int(action_chunk.shape[0]),
            )

        arm_indices = [index for index in range(action_dim) if index not in gripper_action_indices]
        large_joint_indices, small_joint_indices = resolve_joint_groups(
            action_dim=action_dim,
            gripper_action_indices=gripper_action_indices,
            large_joint_indices=self.cfg.large_joint_indices,
            small_joint_indices=self.cfg.small_joint_indices,
        )

        large_joint_energy = compute_mean_abs_motion(action_chunk, large_joint_indices)
        small_joint_energy = compute_mean_abs_motion(action_chunk, small_joint_indices)
        gripper_energy = compute_mean_abs_motion(action_chunk, gripper_action_indices)
        total_energy = large_joint_energy + small_joint_energy + gripper_energy

        # CRITICAL FIX: Check for NaN/inf values that can propagate through computation
        if not (
            all(isinstance(e, float) for e in [large_joint_energy, small_joint_energy, gripper_energy])
        ):
            return self._invalid_output(
                reason="non_numeric_energy_values",
                sample=sample,
                action_key=action_key,
                chunk_length=int(action_chunk.shape[0]),
            )

        import math
        for energy_val, energy_name in [
            (large_joint_energy, "large_joint_energy"),
            (small_joint_energy, "small_joint_energy"),
            (gripper_energy, "gripper_energy"),
            (total_energy, "total_energy"),
        ]:
            if not math.isfinite(energy_val):
                return self._invalid_output(
                    reason=f"non_finite_{energy_name}",
                    sample=sample,
                    action_key=action_key,
                    chunk_length=int(action_chunk.shape[0]),
                    extra_debug={
                        "large_joint_energy": large_joint_energy,
                        "small_joint_energy": small_joint_energy,
                        "gripper_energy": gripper_energy,
                        "total_energy": total_energy,
                        "failing_energy": energy_name,
                    },
                )

        if total_energy < self.cfg.min_total_energy:
            return self._invalid_output(
                reason="insufficient_total_motion_energy",
                sample=sample,
                action_key=action_key,
                chunk_length=int(action_chunk.shape[0]),
                extra_debug={
                    "large_joint_energy": large_joint_energy,
                    "small_joint_energy": small_joint_energy,
                    "gripper_energy": gripper_energy,
                    "total_energy": total_energy,
                },
            )

        energy_stats = self._compute_energy_ratios(
            large_joint_energy=large_joint_energy,
            small_joint_energy=small_joint_energy,
            gripper_energy=gripper_energy,
        )
        gripper_trend_stats = self._compute_gripper_trend_stats(action_chunk[:, gripper_action_indices])
        close_strength, open_strength = self._compute_gripper_direction_strengths(gripper_trend_stats)
        logits = self._compute_stage_logits(
            energy_stats=energy_stats,
            close_strength=close_strength,
            open_strength=open_strength,
            device=action_chunk.device,
            dtype=action_chunk.dtype,
        )
        probabilities = torch.softmax(logits / self.cfg.temperature, dim=0)

        if not torch.isfinite(probabilities).all():
            return self._invalid_output(
                reason="non_finite_stage_probabilities",
                sample=sample,
                action_key=action_key,
                chunk_length=int(action_chunk.shape[0]),
                extra_debug={
                    "logits": logits.detach().cpu().tolist(),
                    "temperature": self.cfg.temperature,
                },
            )

        return StageLabelOutput(
            stage_id=None,
            stage_prob_target=probabilities,
            stage_valid_mask=torch.tensor(True, dtype=torch.bool, device=action_chunk.device),
            stage_source="soft",
            stage_debug={
                "action_key": action_key,
                "chunk_semantics": "action_target_chunk",
                "chunk_length": int(action_chunk.shape[0]),
                "stage_order": self._stage_probability_order(),
                "large_joint_energy": large_joint_energy,
                "small_joint_energy": small_joint_energy,
                "gripper_energy": gripper_energy,
                "total_energy": total_energy,
                **energy_stats,
                **gripper_trend_stats,
                "close_strength": close_strength,
                "open_strength": open_strength,
                "logits": logits.detach().cpu().tolist(),
                "probabilities": probabilities.detach().cpu().tolist(),
                "gripper_action_indices": gripper_action_indices,
                "large_joint_indices": large_joint_indices,
                "small_joint_indices": small_joint_indices,
                "arm_indices": arm_indices,
            },
        )

    def _compute_energy_ratios(
        self,
        *,
        large_joint_energy: float,
        small_joint_energy: float,
        gripper_energy: float,
    ) -> dict[str, float]:
        """Convert raw energy statistics into normalized energy ratios.

        CRITICAL FIX: Added finite value checking to prevent NaN propagation.

        Args:
            large_joint_energy: Energy of large joints.
            small_joint_energy: Energy of small joints.
            gripper_energy: Energy of gripper.

        Returns:
            Dictionary with normalized energy ratios.

        Raises:
            ValueError: If total_energy is not finite or zero.
        """
        total_energy = large_joint_energy + small_joint_energy + gripper_energy

        import math
        if not math.isfinite(total_energy) or total_energy <= 0:
            raise ValueError(
                f"`total_energy` must be finite and positive, got {total_energy}. "
                f"(large={large_joint_energy}, small={small_joint_energy}, gripper={gripper_energy})"
            )

        return {
            "large_joint_energy_ratio": large_joint_energy / total_energy,
            "small_joint_energy_ratio": small_joint_energy / total_energy,
            "gripper_energy_ratio": gripper_energy / total_energy,
        }

    def _compute_gripper_trend_stats(self, gripper_chunk: torch.Tensor) -> dict[str, float]:
        """Compute lightweight gripper trend statistics over the chunk."""
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
        mean_diff = float(gripper_chunk.diff(dim=0).mean().item()) if num_steps >= 2 else 0.0
        signed_trend = 0.5 * (second_half_mean - first_half_mean) + 0.5 * mean_diff

        return {
            "gripper_mean": gripper_mean,
            "gripper_first_half_mean": first_half_mean,
            "gripper_second_half_mean": second_half_mean,
            "gripper_end_minus_start": end_minus_start,
            "gripper_mean_diff": mean_diff,
            "gripper_signed_trend": signed_trend,
        }

    def _compute_gripper_direction_strengths(
        self, gripper_trend_stats: Mapping[str, float]
    ) -> tuple[float, float]:
        """Convert gripper trend statistics into close/open direction strengths."""
        if not self.cfg.use_gripper_trend_signal:
            return 0.0, 0.0

        signed_trend = gripper_trend_stats["gripper_signed_trend"]
        end_minus_start = gripper_trend_stats["gripper_end_minus_start"]
        second_half_mean = gripper_trend_stats["gripper_second_half_mean"]
        gripper_mean = gripper_trend_stats["gripper_mean"]

        if self.cfg.gripper_close_is_negative:
            close_raw = max(
                0.0,
                -signed_trend - self.cfg.gripper_trend_threshold,
            ) + max(0.0, -end_minus_start - self.cfg.gripper_trend_threshold)
            open_raw = max(
                0.0,
                signed_trend - self.cfg.gripper_trend_threshold,
            ) + max(0.0, end_minus_start - self.cfg.gripper_trend_threshold)
            close_bias = max(0.0, -second_half_mean) + max(0.0, -gripper_mean)
            open_bias = max(0.0, second_half_mean) + max(0.0, gripper_mean)
        else:
            close_raw = max(
                0.0,
                signed_trend - self.cfg.gripper_trend_threshold,
            ) + max(0.0, end_minus_start - self.cfg.gripper_trend_threshold)
            open_raw = max(
                0.0,
                -signed_trend - self.cfg.gripper_trend_threshold,
            ) + max(0.0, -end_minus_start - self.cfg.gripper_trend_threshold)
            close_bias = max(0.0, second_half_mean) + max(0.0, gripper_mean)
            open_bias = max(0.0, -second_half_mean) + max(0.0, -gripper_mean)

        close_strength = close_raw + 0.5 * close_bias
        open_strength = open_raw + 0.5 * open_bias
        return close_strength, open_strength

    def _compute_stage_logits(
        self,
        *,
        energy_stats: Mapping[str, float],
        close_strength: float,
        open_strength: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Compute unnormalized stage logits before softmax normalization.

        The returned tensor always follows the semantic probability order
        `[move, grasp, place]`, independent of the configured numeric stage ids.
        """
        large_ratio = energy_stats["large_joint_energy_ratio"]
        small_ratio = energy_stats["small_joint_energy_ratio"]
        gripper_ratio = energy_stats["gripper_energy_ratio"]

        logits = torch.zeros(3, dtype=dtype, device=device)
        logits[0] = (
            self.cfg.move_bias
            + self.cfg.move_large_joint_weight * large_ratio
            - self.cfg.move_small_joint_weight * small_ratio
            - self.cfg.move_gripper_weight * gripper_ratio
        )
        logits[1] = (
            self.cfg.grasp_bias
            + self.cfg.grasp_small_joint_weight * small_ratio
            + self.cfg.grasp_gripper_weight * gripper_ratio
            + self.cfg.grasp_close_trend_weight * close_strength
            - self.cfg.grasp_open_trend_penalty * open_strength
        )
        logits[2] = (
            self.cfg.place_bias
            + self.cfg.place_small_joint_weight * small_ratio
            + self.cfg.place_gripper_weight * gripper_ratio
            + self.cfg.place_open_trend_weight * open_strength
            - self.cfg.place_close_trend_penalty * close_strength
        )
        return logits

    def _stage_probability_order(self) -> list[dict[str, int | str]]:
        """Describe the semantic order used by `stage_prob_target`.

        Returns:
            A length-3 list where each element describes one probability position
            in the returned soft-label vector.
        """
        return [
            {"index": 0, "stage_id": self.cfg.move_stage_id, "stage_name": "move"},
            {"index": 1, "stage_id": self.cfg.grasp_stage_id, "stage_name": "grasp"},
            {"index": 2, "stage_id": self.cfg.place_stage_id, "stage_name": "place"},
        ]

    def _invalid_output(
        self,
        *,
        reason: str,
        sample: Mapping[str, Any],
        action_key: str | None,
        chunk_length: int | None = None,
        extra_debug: dict[str, Any] | None = None,
    ) -> StageLabelOutput:
        """Return a unified invalid output when no usable soft-label signal exists."""
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
            stage_source="soft",
            stage_debug=stage_debug,
        )

    def _find_action_chunk_value(self, sample: Mapping[str, Any]) -> tuple[Any | None, str | None]:
        """Find the first configured action-target-chunk field present in the sample."""
        for key in self.cfg.action_chunk_key_candidates:
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

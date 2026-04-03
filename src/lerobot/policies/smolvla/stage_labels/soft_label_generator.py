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

import math
from dataclasses import dataclass, fields
from typing import Any, Mapping

import torch

from lerobot.utils.constants import ACTION, OBS_STATE

from .base import BaseStageLabelGenerator, StageLabelOutput
from .utils import (
    compute_mean_abs_chunk_delta,
    extract_action_chunk_with_reason,
    extract_binary_gripper_events_from_state_chunk,
    extract_gripper_events_from_action_chunk,
    extract_state_chunk_tensor,
    lookup_sample_value,
    normalize_index_sequence,
    resolve_joint_groups,
    resolve_vector_indices,
)


@dataclass
class SoftStageLabelGeneratorConfig:
    """Configuration for soft labels from local action chunks.

    Arm motion semantics:
    - continuous support is computed only from `delta_action_chunk`
    - only large/small arm joint energy ratios contribute to motion proportions

    Gripper semantics:
    - grasp/place are primarily driven by discrete event strengths
    - gripper action trend remains only a weak auxiliary cue

    Action input semantics:
    - `action_chunk_key_candidates` is the primary path
    - `action_key_candidates` is retained only as a compatibility fallback
    - ordinary single-step action tensors are no longer a valid primary input
    """

    action_key_candidates: tuple[str, ...] = (ACTION, "actions")
    action_chunk_key_candidates: tuple[str, ...] = ("stage_label_action_chunk",)
    state_key: str = OBS_STATE
    state_chunk_key_candidates: tuple[str, ...] = ("stage_label_state_chunk",)

    move_stage_id: int = 0
    grasp_stage_id: int = 1
    place_stage_id: int = 2

    min_chunk_steps: int = 4

    gripper_action_indices: tuple[int, ...] | None = None
    default_gripper_width: int = 1
    large_joint_indices: tuple[int, ...] | None = None
    small_joint_indices: tuple[int, ...] | None = None

    gripper_state_indices: tuple[int, ...] | None = None
    gripper_state_is_binary: bool = True
    gripper_closed_threshold: float | None = None
    gripper_open_threshold: float | None = None

    gripper_close_is_negative: bool = True
    gripper_trend_threshold: float = 0.02

    min_total_arm_energy: float = 1e-4

    move_bias: float = 0.0
    grasp_bias: float = 0.0
    place_bias: float = 0.0

    move_large_joint_weight: float = 2.0
    move_small_joint_weight: float = 0.7
    move_close_event_penalty: float = 1.0
    move_open_event_penalty: float = 1.0

    grasp_small_joint_weight: float = 1.2
    grasp_close_event_weight: float = 1.8
    grasp_open_event_penalty: float = 0.6
    grasp_close_command_weight: float = 0.5

    place_small_joint_weight: float = 1.2
    place_open_event_weight: float = 1.8
    place_close_event_penalty: float = 0.6
    place_open_command_weight: float = 0.5

    temperature: float = 1.0

    def __post_init__(self) -> None:
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
        if self.min_chunk_steps < 2:
            raise ValueError(f"`min_chunk_steps` must be >= 2, got {self.min_chunk_steps}.")
        if not self.action_key_candidates and not self.action_chunk_key_candidates:
            raise ValueError("At least one action or action-chunk key candidate must be configured.")
        if self.temperature <= 0:
            raise ValueError(f"`temperature` must be > 0, got {self.temperature}.")
        if self.gripper_trend_threshold < 0:
            raise ValueError(
                f"`gripper_trend_threshold` must be >= 0, got {self.gripper_trend_threshold}."
            )
        if self.min_total_arm_energy < 0:
            raise ValueError(
                f"`min_total_arm_energy` must be >= 0, got {self.min_total_arm_energy}."
            )

    @classmethod
    def from_config(
        cls, cfg: SoftStageLabelGeneratorConfig | Mapping[str, Any] | Any | None
    ) -> SoftStageLabelGeneratorConfig:
        if cfg is None:
            return cls()
        if isinstance(cfg, cls):
            return cfg

        field_names = {field_.name for field_ in fields(cls)}
        if isinstance(cfg, Mapping):
            filtered = {key: value for key, value in cfg.items() if key in field_names}
            return cls(**filtered)

        data = {name: getattr(cfg, name) for name in field_names if hasattr(cfg, name)}
        return cls(**data)


class SoftStageLabelGenerator(BaseStageLabelGenerator):
    """Generate soft stage probabilities from local action chunks.

    Arm support is continuous and derived from `delta_action_chunk`, while the
    gripper signal is event-driven. The probability vector always follows the
    semantic order `[move, grasp, place]`.
    """

    def __init__(self, cfg: SoftStageLabelGeneratorConfig | Mapping[str, Any] | Any | None = None):
        self.cfg = SoftStageLabelGeneratorConfig.from_config(cfg)
        self.stage_names = {
            self.cfg.move_stage_id: "move",
            self.cfg.grasp_stage_id: "grasp",
            self.cfg.place_stage_id: "place",
        }

    def generate(self, sample: Mapping[str, Any], **kwargs: Any) -> StageLabelOutput:
        del kwargs

        action_chunk_value, action_chunk_key, action_chunk_source = self._find_action_chunk_value(sample)
        state_chunk_value, state_chunk_key = self._find_state_chunk_value(sample)

        if bool(sample.get("stage_label_chunk_cross_episode", False)):
            return self._invalid_output(
                reason="cross_episode_action_chunk",
                sample=sample,
                action_chunk_key=action_chunk_key,
                action_chunk_source=action_chunk_source,
                state_chunk_key=state_chunk_key,
            )

        action_chunk, action_reason, action_debug = extract_action_chunk_with_reason(
            action_chunk_value,
            min_steps=self.cfg.min_chunk_steps,
        )
        if action_chunk is None:
            inferred_chunk_size = self._infer_chunk_size_from_action_debug(action_debug)
            return self._invalid_output(
                reason=action_reason or "missing_action_chunk",
                sample=sample,
                action_chunk_key=action_chunk_key,
                action_chunk_source=action_chunk_source,
                state_chunk_key=state_chunk_key,
                extra_debug={
                    **self._chunk_debug_metadata(
                        sample,
                        action_chunk_source=action_chunk_source,
                        chunk_size=inferred_chunk_size,
                    ),
                    "action_chunk_reason": action_reason or "missing_action_chunk",
                    **action_debug,
                },
            )

        action_dim = int(action_chunk.shape[-1])
        gripper_action_indices = self._resolve_gripper_action_indices(action_dim)
        if not gripper_action_indices:
            return self._invalid_output(
                reason="missing_gripper_action_dims",
                sample=sample,
                action_chunk_key=action_chunk_key,
                action_chunk_source=action_chunk_source,
                state_chunk_key=state_chunk_key,
            )

        arm_indices = [index for index in range(action_dim) if index not in gripper_action_indices]
        large_joint_indices, small_joint_indices = resolve_joint_groups(
            action_dim=action_dim,
            gripper_action_indices=gripper_action_indices,
            large_joint_indices=self.cfg.large_joint_indices,
            small_joint_indices=self.cfg.small_joint_indices,
        )

        large_joint_energy = compute_mean_abs_chunk_delta(action_chunk, large_joint_indices)
        small_joint_energy = compute_mean_abs_chunk_delta(action_chunk, small_joint_indices)
        total_arm_energy = large_joint_energy + small_joint_energy

        if not math.isfinite(large_joint_energy) or not math.isfinite(small_joint_energy):
            return self._invalid_output(
                reason="non_finite_arm_energy",
                sample=sample,
                action_chunk_key=action_chunk_key,
                action_chunk_source=action_chunk_source,
                state_chunk_key=state_chunk_key,
                extra_debug={
                    "large_joint_energy": large_joint_energy,
                    "small_joint_energy": small_joint_energy,
                    "total_arm_energy": total_arm_energy,
                },
            )

        event_stats = self._resolve_gripper_events(
            action_chunk=action_chunk,
            state_chunk_value=state_chunk_value,
            gripper_action_indices=gripper_action_indices,
        )
        closing_event_strength = (
            1.0 if event_stats["closing_event"] else event_stats["close_command_score"]
        )
        opening_event_strength = (
            1.0 if event_stats["opening_event"] else event_stats["open_command_score"]
        )
        total_support = total_arm_energy + closing_event_strength + opening_event_strength
        if total_support <= self.cfg.min_total_arm_energy:
            return self._invalid_output(
                reason=(
                    "missing_gripper_event_signal"
                    if event_stats["gripper_event_source"] == "none"
                    else "insufficient_total_support"
                ),
                sample=sample,
                action_chunk_key=action_chunk_key,
                action_chunk_source=action_chunk_source,
                state_chunk_key=state_chunk_key,
                extra_debug={
                    "large_joint_energy": large_joint_energy,
                    "small_joint_energy": small_joint_energy,
                    "total_arm_energy": total_arm_energy,
                    "closing_event_strength": closing_event_strength,
                    "opening_event_strength": opening_event_strength,
                    **event_stats,
                },
            )

        energy_stats = self._compute_arm_energy_ratios(
            large_joint_energy=large_joint_energy,
            small_joint_energy=small_joint_energy,
        )
        logit_dtype = action_chunk.dtype if torch.is_floating_point(action_chunk) else torch.float32
        logits = self._compute_stage_logits(
            energy_stats=energy_stats,
            closing_event_strength=closing_event_strength,
            opening_event_strength=opening_event_strength,
            close_command_score=event_stats["close_command_score"],
            open_command_score=event_stats["open_command_score"],
            device=action_chunk.device,
            dtype=logit_dtype,
        )
        probabilities = torch.softmax(logits / self.cfg.temperature, dim=0)
        if not torch.isfinite(probabilities).all():
            return self._invalid_output(
                reason="non_finite_stage_probabilities",
                sample=sample,
                action_chunk_key=action_chunk_key,
                action_chunk_source=action_chunk_source,
                state_chunk_key=state_chunk_key,
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
                "action_semantics": "local_action_chunk",
                "energy_semantics": "delta_action_chunk",
                "chunk_size": int(action_chunk.shape[0]),
                "delta_chunk_length": int(action_chunk.shape[0] - 1),
                "action_chunk_key": action_chunk_key,
                "action_chunk_source": action_chunk_source,
                "state_chunk_key": state_chunk_key,
                **self._chunk_debug_metadata(
                    sample,
                    action_chunk_source=action_chunk_source,
                    chunk_size=int(action_chunk.shape[0]),
                ),
                "stage_order": self._stage_probability_order(),
                "large_joint_energy": large_joint_energy,
                "small_joint_energy": small_joint_energy,
                "total_arm_energy": total_arm_energy,
                **energy_stats,
                "gripper_start_state": event_stats["gripper_start_state"],
                "gripper_end_state": event_stats["gripper_end_state"],
                "gripper_changed": event_stats["gripper_changed"],
                "closing_event": event_stats["closing_event"],
                "opening_event": event_stats["opening_event"],
                "closing_event_strength": closing_event_strength,
                "closing_event_strength_source": (
                    "state_event"
                    if event_stats["gripper_event_source"] == "state" and event_stats["closing_event"]
                    else "action_fallback"
                ),
                "opening_event_strength": opening_event_strength,
                "opening_event_strength_source": (
                    "state_event"
                    if event_stats["gripper_event_source"] == "state" and event_stats["opening_event"]
                    else "action_fallback"
                ),
                "gripper_signed_trend": event_stats["gripper_signed_trend"],
                "close_command_score": event_stats["close_command_score"],
                "open_command_score": event_stats["open_command_score"],
                "gripper_event_source": event_stats["gripper_event_source"],
                "state_chunk_available": event_stats["state_chunk_available"],
                "state_chunk_reason": event_stats["state_chunk_reason"],
                "state_event_reason": event_stats["state_event_reason"],
                "state_event_debug": event_stats["state_event_debug"],
                "action_fallback_used": event_stats["action_fallback_used"],
                "logits": logits.detach().cpu().tolist(),
                "probabilities": probabilities.detach().cpu().tolist(),
                "gripper_action_indices": gripper_action_indices,
                "gripper_state_indices": event_stats["gripper_state_indices"],
                "large_joint_indices": large_joint_indices,
                "small_joint_indices": small_joint_indices,
                "arm_indices": arm_indices,
            },
        )

    def _compute_arm_energy_ratios(
        self,
        *,
        large_joint_energy: float,
        small_joint_energy: float,
    ) -> dict[str, float]:
        total_arm_energy = large_joint_energy + small_joint_energy
        if not math.isfinite(total_arm_energy) or total_arm_energy <= 0:
            return {
                "large_joint_energy_ratio": 0.0,
                "small_joint_energy_ratio": 0.0,
            }
        return {
            "large_joint_energy_ratio": large_joint_energy / total_arm_energy,
            "small_joint_energy_ratio": small_joint_energy / total_arm_energy,
        }

    def _compute_stage_logits(
        self,
        *,
        energy_stats: Mapping[str, float],
        closing_event_strength: float,
        opening_event_strength: float,
        close_command_score: float,
        open_command_score: float,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        large_ratio = energy_stats["large_joint_energy_ratio"]
        small_ratio = energy_stats["small_joint_energy_ratio"]

        logits = torch.zeros(3, dtype=dtype, device=device)
        logits[0] = (
            self.cfg.move_bias
            + self.cfg.move_large_joint_weight * large_ratio
            - self.cfg.move_small_joint_weight * small_ratio
            - self.cfg.move_close_event_penalty * closing_event_strength
            - self.cfg.move_open_event_penalty * opening_event_strength
        )
        logits[1] = (
            self.cfg.grasp_bias
            + self.cfg.grasp_small_joint_weight * small_ratio
            + self.cfg.grasp_close_event_weight * closing_event_strength
            - self.cfg.grasp_open_event_penalty * opening_event_strength
            + self.cfg.grasp_close_command_weight * close_command_score
        )
        logits[2] = (
            self.cfg.place_bias
            + self.cfg.place_small_joint_weight * small_ratio
            + self.cfg.place_open_event_weight * opening_event_strength
            - self.cfg.place_close_event_penalty * closing_event_strength
            + self.cfg.place_open_command_weight * open_command_score
        )
        return logits

    def _resolve_gripper_events(
        self,
        *,
        action_chunk: torch.Tensor,
        state_chunk_value: Any | None,
        gripper_action_indices: list[int],
    ) -> dict[str, Any]:
        action_stats = extract_gripper_events_from_action_chunk(
            action_chunk,
            gripper_action_indices,
            close_is_negative=self.cfg.gripper_close_is_negative,
            trend_threshold=self.cfg.gripper_trend_threshold,
        )
        event_stats: dict[str, Any] = {
            "gripper_start_state": None,
            "gripper_end_state": None,
            "gripper_changed": False,
            "closing_event": False,
            "opening_event": False,
            "gripper_event_source": "none",
            "gripper_state_indices": [],
            "state_chunk_available": False,
            "state_chunk_reason": None,
            "state_event_reason": None,
            "state_event_debug": {},
            "action_fallback_used": False,
            **action_stats,
        }

        state_chunk, state_chunk_reason, _ = extract_state_chunk_tensor(state_chunk_value)
        event_stats["state_chunk_reason"] = state_chunk_reason
        if state_chunk is not None:
            event_stats["state_chunk_available"] = True
            gripper_state_indices = resolve_vector_indices(
                vector_dim=int(state_chunk.shape[-1]),
                configured_indices=self.cfg.gripper_state_indices,
            )
            if gripper_state_indices:
                state_events, state_reason, state_event_debug = extract_binary_gripper_events_from_state_chunk(
                    state_chunk,
                    gripper_state_indices,
                    closed_threshold=self.cfg.gripper_closed_threshold,
                    open_threshold=self.cfg.gripper_open_threshold,
                    is_binary_state=self.cfg.gripper_state_is_binary,
                )
                event_stats["state_event_reason"] = state_reason
                event_stats["state_event_debug"] = state_event_debug
                event_stats["gripper_state_indices"] = gripper_state_indices
                if state_reason is None and state_events:
                    event_stats.update(state_events)
                    event_stats["gripper_event_source"] = "state"
                    return event_stats
            else:
                event_stats["state_event_reason"] = "missing_gripper_state_indices"
                event_stats["state_event_debug"] = {
                    "state_dim": int(state_chunk.shape[-1]),
                    "configured_gripper_state_indices": (
                        list(self.cfg.gripper_state_indices)
                        if self.cfg.gripper_state_indices is not None
                        else []
                    ),
                }

        event_stats["action_fallback_used"] = True
        if event_stats["close_command_score"] > 0.0 or event_stats["open_command_score"] > 0.0:
            event_stats["closing_event"] = event_stats["close_command_score"] > 0.0
            event_stats["opening_event"] = event_stats["open_command_score"] > 0.0
            event_stats["gripper_event_source"] = "action_fallback"

        return event_stats

    def _stage_probability_order(self) -> list[dict[str, int | str]]:
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
        action_chunk_key: str | None,
        action_chunk_source: str,
        state_chunk_key: str | None,
        extra_debug: dict[str, Any] | None = None,
    ) -> StageLabelOutput:
        stage_debug = {
            "decision_reason": reason,
            "action_semantics": "local_action_chunk",
            "energy_semantics": "delta_action_chunk",
            "action_chunk_key": action_chunk_key,
            "action_chunk_source": action_chunk_source,
            "state_chunk_key": state_chunk_key,
            "stage_order": self._stage_probability_order(),
            "chunk_cross_episode": bool(sample.get("stage_label_chunk_cross_episode", False)),
            **self._chunk_debug_metadata(sample, action_chunk_source=action_chunk_source),
            "min_chunk_steps": self.cfg.min_chunk_steps,
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

    def _find_action_chunk_value(
        self, sample: Mapping[str, Any]
    ) -> tuple[Any | None, str | None, str]:
        for key in self.cfg.action_chunk_key_candidates:
            value = lookup_sample_value(key, sample)
            if value is not None:
                return value, key, "action_chunk_key"
        for key in self.cfg.action_key_candidates:
            value = lookup_sample_value(key, sample)
            if value is not None:
                return value, key, "action_key_fallback"
        return None, None, "action_chunk_key"

    def _find_state_chunk_value(self, sample: Mapping[str, Any]) -> tuple[Any | None, str | None]:
        for key in self.cfg.state_chunk_key_candidates:
            value = lookup_sample_value(key, sample)
            if value is not None:
                return value, key
        value = lookup_sample_value(self.cfg.state_key, sample)
        if value is not None:
            return value, self.cfg.state_key
        return None, None

    def _resolve_gripper_action_indices(self, action_dim: int) -> list[int]:
        configured = normalize_index_sequence(self.cfg.gripper_action_indices, action_dim)
        if configured:
            return configured
        width = min(max(self.cfg.default_gripper_width, 0), action_dim)
        if width == 0:
            return []
        return list(range(action_dim - width, action_dim))

    def _chunk_debug_metadata(
        self,
        sample: Mapping[str, Any],
        *,
        action_chunk_source: str,
        chunk_size: int | None = None,
    ) -> dict[str, Any]:
        resolved_chunk_size = chunk_size
        if resolved_chunk_size is None:
            resolved_chunk_size = int(sample.get("stage_label_chunk_size", 0))
        chunk_requested_size = int(sample.get("stage_label_chunk_requested_size", 0))
        chunk_complete = bool(
            sample.get(
                "stage_label_chunk_complete",
                chunk_requested_size > 0 and resolved_chunk_size == chunk_requested_size,
            )
        )
        chunk_cross_episode = bool(sample.get("stage_label_chunk_cross_episode", False))
        chunk_valid = bool(
            sample.get(
                "stage_label_chunk_valid",
                (not chunk_cross_episode) and resolved_chunk_size >= 2,
            )
        )
        return {
            "chunk_size": resolved_chunk_size,
            "chunk_requested_size": chunk_requested_size,
            "chunk_complete": chunk_complete,
            "chunk_valid": chunk_valid,
            "action_chunk_source": action_chunk_source,
        }

    @staticmethod
    def _infer_chunk_size_from_action_debug(action_debug: Mapping[str, Any]) -> int | None:
        if "chunk_length" in action_debug:
            return int(action_debug["chunk_length"])
        shape = action_debug.get("action_chunk_shape")
        if isinstance(shape, tuple):
            if len(shape) == 1:
                return 1
            if len(shape) >= 2:
                return int(shape[0])
        return None

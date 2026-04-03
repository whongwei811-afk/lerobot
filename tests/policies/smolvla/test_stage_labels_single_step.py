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

import pytest
import torch

from lerobot.policies.smolvla.stage_labels.analyzer import StageLabelStatistics
from lerobot.policies.smolvla.stage_labels.factory import build_stage_label_generator
from lerobot.policies.smolvla.stage_labels.hard_label_generator import HardStageLabelGenerator
from lerobot.policies.smolvla.stage_labels.soft_label_generator import SoftStageLabelGenerator
from lerobot.policies.smolvla.stage_labels.utils import extract_state_chunk_tensor
from lerobot.policies.smolvla.stage_labels.wrapper import StageLabeledDataset


class _ToyDataset:
    def __init__(self, samples: list[dict[str, object]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        return self.samples[idx]


def _sample(action: list[float], state: list[float], episode_index: int = 0) -> dict[str, object]:
    return {
        "action": torch.tensor(action, dtype=torch.float32),
        "observation": {"state": torch.tensor(state, dtype=torch.float32)},
        "episode_index": episode_index,
    }


def _hard_generator() -> HardStageLabelGenerator:
    return HardStageLabelGenerator(
        {
            "min_chunk_steps": 4,
            "gripper_action_indices": (3,),
            "gripper_state_indices": (0,),
            "large_joint_indices": (0, 1),
            "small_joint_indices": (2,),
            "allow_action_trend_fallback_for_gripper_event": True,
        }
    )


def _soft_generator() -> SoftStageLabelGenerator:
    return SoftStageLabelGenerator(
        {
            "min_chunk_steps": 4,
            "gripper_action_indices": (3,),
            "gripper_state_indices": (0,),
            "large_joint_indices": (0, 1),
            "small_joint_indices": (2,),
        }
    )


def test_wrapper_chunk_valid_and_complete_semantics() -> None:
    dataset = _ToyDataset(
        [
            _sample([0.0, 0.0, 0.0, 0.0], [0.0], 0),
            _sample([0.1, 0.0, 0.0, 0.0], [0.0], 0),
            _sample([0.2, 0.0, 0.0, 0.0], [0.0], 0),
        ]
    )
    wrapped = StageLabeledDataset(
        dataset,
        build_stage_label_generator("hard"),
        include_debug=True,
        action_chunk_size=4,
    )

    head = wrapped[0]
    tail = wrapped[2]

    assert "stage_label_action_chunk" in head
    assert "stage_label_chunk_valid" in head
    assert "stage_label_chunk_size" in head
    assert "stage_label_chunk_complete" in head
    assert tuple(head["stage_label_action_chunk"].shape) == (3, 4)
    assert head["stage_label_chunk_valid"] is True
    assert head["stage_label_chunk_size"] == 3
    assert head["stage_label_chunk_complete"] is False
    assert head["stage_label_chunk_cross_episode"] is False

    assert tuple(tail["stage_label_action_chunk"].shape) == (1, 4)
    assert tail["stage_label_chunk_valid"] is False
    assert tail["stage_label_chunk_size"] == 1
    assert tail["stage_label_chunk_complete"] is False


def test_invalid_debug_contains_full_chunk_metadata() -> None:
    dataset = _ToyDataset(
        [
            _sample([0.0, 0.0, 0.0, 0.0], [0.0], 0),
            _sample([0.1, 0.0, 0.0, 0.0], [0.0], 0),
            _sample([0.2, 0.0, 0.0, 0.0], [0.0], 0),
        ]
    )
    wrapped = StageLabeledDataset(
        dataset,
        _hard_generator(),
        include_debug=True,
        action_chunk_size=4,
    )

    sample = wrapped[0]

    assert bool(sample["stage_valid_mask"].item()) is False
    assert sample["stage_debug"]["decision_reason"] == "insufficient_action_chunk_length"
    assert sample["stage_debug"]["chunk_size"] == 3
    assert sample["stage_debug"]["chunk_requested_size"] == 4
    assert sample["stage_debug"]["chunk_complete"] is False
    assert sample["stage_debug"]["chunk_valid"] is True
    assert sample["stage_debug"]["min_chunk_steps"] == 4


def test_chunk_hard_label_from_closing_event() -> None:
    generator = _hard_generator()
    sample = {
        "stage_label_action_chunk": torch.tensor(
            [
                [0.0, 0.0, 0.00, 0.0],
                [0.0, 0.0, 0.06, 0.0],
                [0.0, 0.0, 0.12, 0.0],
                [0.0, 0.0, 0.18, 0.0],
            ],
            dtype=torch.float32,
        ),
        "stage_label_state_chunk": torch.tensor([[0.0], [0.0], [1.0], [1.0]], dtype=torch.float32),
        "stage_label_chunk_size": 4,
        "stage_label_chunk_complete": True,
        "stage_label_chunk_cross_episode": False,
    }

    output = generator(sample)

    assert bool(output.stage_valid_mask.item()) is True
    assert int(output.stage_id.item()) == 1
    assert output.stage_debug["stage_name"] == "grasp"
    assert output.stage_debug["closing_event"] is True
    assert output.stage_debug["gripper_event_source"] == "state"


def test_chunk_hard_label_from_opening_event() -> None:
    generator = _hard_generator()
    sample = {
        "stage_label_action_chunk": torch.tensor(
            [
                [0.0, 0.0, 0.00, 0.0],
                [0.0, 0.0, 0.05, 0.0],
                [0.0, 0.0, 0.11, 0.0],
                [0.0, 0.0, 0.16, 0.0],
            ],
            dtype=torch.float32,
        ),
        "stage_label_state_chunk": torch.tensor([[1.0], [1.0], [0.0], [0.0]], dtype=torch.float32),
        "stage_label_chunk_size": 4,
        "stage_label_chunk_complete": True,
        "stage_label_chunk_cross_episode": False,
    }

    output = generator(sample)

    assert bool(output.stage_valid_mask.item()) is True
    assert int(output.stage_id.item()) == 2
    assert output.stage_debug["stage_name"] == "place"
    assert output.stage_debug["opening_event"] is True


def test_chunk_hard_label_move_from_large_joint_dominance() -> None:
    generator = _hard_generator()
    sample = {
        "stage_label_action_chunk": torch.tensor(
            [
                [0.0, 0.0, 0.00, 0.0],
                [0.3, 0.2, 0.01, 0.0],
                [0.6, 0.4, 0.02, 0.0],
                [0.9, 0.6, 0.03, 0.0],
            ],
            dtype=torch.float32,
        ),
        "stage_label_state_chunk": torch.tensor([[0.0], [0.0], [0.0], [0.0]], dtype=torch.float32),
        "stage_label_chunk_size": 4,
        "stage_label_chunk_complete": True,
        "stage_label_chunk_cross_episode": False,
    }

    output = generator(sample)

    assert bool(output.stage_valid_mask.item()) is True
    assert int(output.stage_id.item()) == 0
    assert output.stage_debug["stage_name"] == "move"
    assert output.stage_debug["large_joint_dominance"] is True
    assert output.stage_debug["closing_event"] is False
    assert output.stage_debug["opening_event"] is False


def test_chunk_soft_label_uses_event_strengths() -> None:
    generator = _soft_generator()
    closing_sample = {
        "stage_label_action_chunk": torch.tensor(
            [
                [0.0, 0.0, 0.00, 0.0],
                [0.0, 0.0, 0.05, 0.0],
                [0.0, 0.0, 0.10, 0.0],
                [0.0, 0.0, 0.15, 0.0],
            ],
            dtype=torch.float32,
        ),
        "stage_label_state_chunk": torch.tensor([[0.0], [0.0], [1.0], [1.0]], dtype=torch.float32),
        "stage_label_chunk_size": 4,
        "stage_label_chunk_complete": True,
        "stage_label_chunk_cross_episode": False,
    }
    opening_sample = {
        **closing_sample,
        "stage_label_state_chunk": torch.tensor([[1.0], [1.0], [0.0], [0.0]], dtype=torch.float32),
    }

    closing_output = generator(closing_sample)
    opening_output = generator(opening_sample)

    assert bool(closing_output.stage_valid_mask.item()) is True
    assert bool(opening_output.stage_valid_mask.item()) is True
    assert int(torch.argmax(closing_output.stage_prob_target).item()) == 1
    assert int(torch.argmax(opening_output.stage_prob_target).item()) == 2


def test_chunk_soft_label_not_driven_by_gripper_continuous_energy() -> None:
    generator = SoftStageLabelGenerator(
        {
            "min_chunk_steps": 4,
            "gripper_action_indices": (3,),
            "large_joint_indices": (0, 1),
            "small_joint_indices": (2,),
        }
    )
    sample = {
        "stage_label_action_chunk": torch.tensor(
            [
                [0.0, 0.0, 0.00, -1.0],
                [0.4, 0.3, 0.01, -1.0],
                [0.8, 0.6, 0.02, -1.0],
                [1.2, 0.9, 0.03, -1.0],
            ],
            dtype=torch.float32,
        ),
        "stage_label_chunk_size": 4,
        "stage_label_chunk_complete": True,
        "stage_label_chunk_cross_episode": False,
    }

    output = generator(sample)

    assert bool(output.stage_valid_mask.item()) is True
    assert int(torch.argmax(output.stage_prob_target).item()) == 0
    assert output.stage_debug["gripper_event_source"] == "none" or output.stage_debug["gripper_event_source"] == "action_fallback"
    assert output.stage_debug["close_command_score"] == pytest.approx(0.0)
    assert output.stage_debug["open_command_score"] == pytest.approx(0.0)


def test_chunk_invalid_on_cross_episode_boundary() -> None:
    dataset = _ToyDataset(
        [
            _sample([0.0, 0.0, 0.0, 0.0], [0.0], 0),
            _sample([0.1, 0.0, 0.0, 0.0], [0.0], 0),
            _sample([0.2, 0.0, 0.0, 0.0], [0.0], 1),
            _sample([0.3, 0.0, 0.0, 0.0], [0.0], 1),
        ]
    )
    wrapped = StageLabeledDataset(
        dataset,
        _hard_generator(),
        include_debug=True,
        action_chunk_size=4,
    )

    output = wrapped[0]

    assert bool(output["stage_valid_mask"].item()) is False
    assert output["stage_debug"]["decision_reason"] == "cross_episode_action_chunk"
    assert output["stage_debug"]["chunk_size"] == 2
    assert output["stage_debug"]["chunk_requested_size"] == 4
    assert output["stage_debug"]["chunk_complete"] is False
    assert output["stage_debug"]["chunk_valid"] is False


def test_hard_debug_keeps_state_failure_when_action_fallback_succeeds() -> None:
    generator = HardStageLabelGenerator(
        {
            "min_chunk_steps": 4,
            "gripper_action_indices": (3,),
            "gripper_state_indices": (1,),
            "large_joint_indices": (0, 1),
            "small_joint_indices": (2,),
            "allow_action_trend_fallback_for_gripper_event": True,
        }
    )
    sample = {
        "stage_label_action_chunk": torch.tensor(
            [
                [0.0, 0.0, 0.00, 0.0],
                [0.0, 0.0, 0.06, -0.1],
                [0.0, 0.0, 0.12, -0.2],
                [0.0, 0.0, 0.18, -0.3],
            ],
            dtype=torch.float32,
        ),
        "stage_label_state_chunk": torch.tensor([[0.0], [0.0], [0.0], [0.0]], dtype=torch.float32),
    }

    output = generator(sample)

    assert bool(output.stage_valid_mask.item()) is True
    assert int(output.stage_id.item()) == 1
    assert output.stage_debug["gripper_event_source"] == "action_fallback"
    assert output.stage_debug["state_chunk_available"] is True
    assert output.stage_debug["state_chunk_reason"] is None
    assert output.stage_debug["state_event_reason"] == "missing_gripper_state_indices"
    assert output.stage_debug["action_fallback_used"] is True


def test_soft_debug_keeps_state_failure_when_action_fallback_succeeds() -> None:
    generator = SoftStageLabelGenerator(
        {
            "min_chunk_steps": 4,
            "gripper_action_indices": (3,),
            "gripper_state_indices": (1,),
            "large_joint_indices": (0, 1),
            "small_joint_indices": (2,),
        }
    )
    sample = {
        "stage_label_action_chunk": torch.tensor(
            [
                [0.0, 0.0, 0.00, 0.0],
                [0.0, 0.0, 0.06, -0.1],
                [0.0, 0.0, 0.12, -0.2],
                [0.0, 0.0, 0.18, -0.3],
            ],
            dtype=torch.float32,
        ),
        "stage_label_state_chunk": torch.tensor([[0.0], [0.0], [0.0], [0.0]], dtype=torch.float32),
    }

    output = generator(sample)

    assert bool(output.stage_valid_mask.item()) is True
    assert int(torch.argmax(output.stage_prob_target).item()) == 1
    assert output.stage_debug["gripper_event_source"] == "action_fallback"
    assert output.stage_debug["state_chunk_available"] is True
    assert output.stage_debug["state_chunk_reason"] is None
    assert output.stage_debug["state_event_reason"] == "missing_gripper_state_indices"
    assert output.stage_debug["action_fallback_used"] is True
    assert output.stage_debug["closing_event_strength_source"] == "action_fallback"


def test_extract_state_chunk_tensor() -> None:
    state_chunk, reason, debug = extract_state_chunk_tensor(torch.tensor([[0.0], [1.0]]), min_steps=2)
    assert reason is None
    assert debug == {}
    assert tuple(state_chunk.shape) == (2, 1)

    missing_chunk, missing_reason, _ = extract_state_chunk_tensor(None)
    assert missing_chunk is None
    assert missing_reason == "missing_state_chunk"

    invalid_chunk, invalid_reason, invalid_debug = extract_state_chunk_tensor(torch.tensor([0.0, 1.0]))
    assert invalid_chunk is None
    assert invalid_reason == "invalid_state_chunk_shape"
    assert invalid_debug["state_chunk_shape"] == (2,)

    short_chunk, short_reason, short_debug = extract_state_chunk_tensor(torch.tensor([[0.0]]), min_steps=2)
    assert short_chunk is None
    assert short_reason == "insufficient_state_chunk_length"
    assert short_debug["chunk_length"] == 1


@pytest.mark.parametrize("generator", [_hard_generator(), _soft_generator()])
def test_action_key_fallback_debug_behavior(generator: HardStageLabelGenerator | SoftStageLabelGenerator) -> None:
    output = generator({"action": torch.tensor([0.1, 0.0, 0.0, -0.1], dtype=torch.float32)})

    assert bool(output.stage_valid_mask.item()) is False
    assert output.stage_debug["decision_reason"] == "invalid_action_chunk_shape"
    assert output.stage_debug["action_chunk_source"] == "action_key_fallback"
    assert output.stage_debug["action_chunk_reason"] == "invalid_action_chunk_shape"
    assert output.stage_debug["action_chunk_shape"] == (4,)
    assert output.stage_debug["chunk_size"] == 1
    assert output.stage_debug["chunk_valid"] is False


def test_factory_only_supports_hard_and_soft() -> None:
    assert isinstance(build_stage_label_generator("hard"), HardStageLabelGenerator)
    assert isinstance(build_stage_label_generator("soft"), SoftStageLabelGenerator)

    with pytest.raises(ValueError, match="Unsupported stage label generator mode"):
        build_stage_label_generator("future_pseudo")


def test_stage_labeled_dataset_keeps_training_fields() -> None:
    dataset = _ToyDataset(
        [
            _sample([0.0, 0.0, 0.0, 0.0], [0.0], 0),
            _sample([0.0, 0.0, 0.1, 0.0], [0.0], 0),
            _sample([0.0, 0.0, 0.2, 0.0], [1.0], 0),
            _sample([0.0, 0.0, 0.3, 0.0], [1.0], 0),
        ]
    )
    labeled_dataset = StageLabeledDataset(dataset, _soft_generator(), include_debug=True, action_chunk_size=4)

    sample = labeled_dataset[0]

    assert isinstance(sample["stage_id"], torch.Tensor)
    assert isinstance(sample["stage_prob_target"], torch.Tensor)
    assert isinstance(sample["stage_valid_mask"], torch.Tensor)
    assert isinstance(sample["stage_source"], str)
    assert isinstance(sample["stage_debug"], dict)
    assert sample["stage_prob_target"].shape == (3,)
    assert sample["stage_source"] == "soft"
    assert sample["stage_debug"]["action_semantics"] == "local_action_chunk"


def test_analyzer_only_collects_soft_statistics_for_soft_source() -> None:
    hard_dataset = StageLabeledDataset(
        _ToyDataset(
            [
                _sample([0.0, 0.0, 0.0, 0.0], [0.0], 0),
                _sample([0.0, 0.0, 0.1, 0.0], [0.0], 0),
                _sample([0.0, 0.0, 0.2, 0.0], [1.0], 0),
                _sample([0.0, 0.0, 0.3, 0.0], [1.0], 0),
            ]
        ),
        _hard_generator(),
        include_debug=True,
        action_chunk_size=4,
    )

    hard_analyzer = StageLabelStatistics(hard_dataset, verbose=False)
    hard_analyzer.compute()

    assert hard_analyzer.source_counts["hard"] == 4
    assert hard_analyzer.stage_id_counts
    assert hard_analyzer.soft_label_probs == []

    soft_dataset = StageLabeledDataset(
        _ToyDataset(
            [
                _sample([0.0, 0.0, 0.0, 0.0], [0.0], 0),
                _sample([0.0, 0.0, 0.1, 0.0], [0.0], 0),
                _sample([0.0, 0.0, 0.2, 0.0], [1.0], 0),
                _sample([0.0, 0.0, 0.3, 0.0], [1.0], 0),
            ]
        ),
        _soft_generator(),
        include_debug=True,
        action_chunk_size=4,
    )

    soft_analyzer = StageLabelStatistics(soft_dataset, verbose=False)
    soft_analyzer.compute()

    assert soft_analyzer.source_counts["soft"] == 4
    assert soft_analyzer.stage_id_counts == {}
    assert len(soft_analyzer.soft_label_probs) == 1

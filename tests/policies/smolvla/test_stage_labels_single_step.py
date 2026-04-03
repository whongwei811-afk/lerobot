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
from lerobot.policies.smolvla.stage_labels.wrapper import StageLabeledDataset


class _ToyDataset:
    def __init__(self, samples: list[dict[str, object]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, object]:
        return self.samples[idx]


def test_hard_generator_uses_single_step_action() -> None:
    generator = HardStageLabelGenerator()
    sample = {
        "action": torch.tensor([0.25, 0.10, 0.05, 0.00], dtype=torch.float32),
        "observation": {
            "state": torch.tensor([0.2, 0.4, 0.6], dtype=torch.float32),
        },
    }

    output = generator(sample)

    assert output.stage_source == "hard"
    assert bool(output.stage_valid_mask.item()) is True
    assert output.stage_prob_target is None
    assert isinstance(output.stage_id, torch.Tensor)
    assert output.stage_id.dtype == torch.long
    assert output.stage_id.ndim == 0
    assert output.stage_debug["action_semantics"] == "single_step_action"
    assert "chunk_semantics" not in output.stage_debug


def test_soft_generator_uses_single_step_action() -> None:
    generator = SoftStageLabelGenerator()
    sample = {
        "action": torch.tensor([0.05, 0.02, 0.01, -0.40], dtype=torch.float32),
    }

    output = generator(sample)

    assert output.stage_source == "soft"
    assert bool(output.stage_valid_mask.item()) is True
    assert output.stage_id is None
    assert isinstance(output.stage_prob_target, torch.Tensor)
    assert output.stage_prob_target.shape == (3,)
    assert torch.isclose(output.stage_prob_target.sum(), torch.tensor(1.0), atol=1e-6)
    assert output.stage_debug["action_semantics"] == "single_step_action"
    assert output.stage_debug["action_key"] == "action"
    assert "chunk_semantics" not in output.stage_debug
    assert output.stage_debug.get("decision_reason") != "insufficient_action_target_chunk_length"


def test_factory_only_supports_hard_and_soft() -> None:
    assert isinstance(build_stage_label_generator("hard"), HardStageLabelGenerator)
    assert isinstance(build_stage_label_generator("soft"), SoftStageLabelGenerator)

    with pytest.raises(ValueError, match="Unsupported stage label generator mode"):
        build_stage_label_generator("future_pseudo")


@pytest.mark.parametrize("generator_cls", [HardStageLabelGenerator, SoftStageLabelGenerator])
def test_generators_report_unsupported_multi_step_action(generator_cls: type) -> None:
    generator = generator_cls()
    sample = {
        "action": torch.tensor(
            [
                [0.10, 0.05, 0.02, -0.20],
                [0.15, 0.04, 0.01, -0.10],
            ],
            dtype=torch.float32,
        ),
    }

    output = generator(sample)

    assert bool(output.stage_valid_mask.item()) is False
    assert output.stage_debug["decision_reason"] == "unsupported_multi_step_action"
    assert output.stage_debug["action_shape"] == (2, 4)


@pytest.mark.parametrize("generator_cls", [HardStageLabelGenerator, SoftStageLabelGenerator])
def test_generators_report_invalid_action_shape(generator_cls: type) -> None:
    generator = generator_cls()
    sample = {
        "action": torch.zeros(1, 2, 4, dtype=torch.float32),
    }

    output = generator(sample)

    assert bool(output.stage_valid_mask.item()) is False
    assert output.stage_debug["decision_reason"] == "invalid_action_shape"
    assert output.stage_debug["action_shape"] == (1, 2, 4)


@pytest.mark.parametrize("mode", ["hard", "soft"])
def test_stage_labeled_dataset_keeps_training_fields(mode: str) -> None:
    dataset = _ToyDataset(
        [
            {
                "action": torch.tensor([0.1, 0.05, 0.02, -0.2], dtype=torch.float32),
                "observation": {
                    "state": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
                },
            }
        ]
    )

    labeled_dataset = StageLabeledDataset(
        dataset,
        build_stage_label_generator(mode),
        include_debug=True,
    )
    sample = labeled_dataset[0]

    assert isinstance(sample["stage_id"], torch.Tensor)
    assert isinstance(sample["stage_prob_target"], torch.Tensor)
    assert isinstance(sample["stage_valid_mask"], torch.Tensor)
    assert isinstance(sample["stage_source"], str)
    assert isinstance(sample["stage_debug"], dict)
    assert sample["stage_prob_target"].shape == (3,)
    assert sample["stage_source"] == mode
    assert sample["stage_debug"]["action_semantics"] == "single_step_action"


def test_analyzer_only_collects_soft_statistics_for_soft_source() -> None:
    hard_dataset = StageLabeledDataset(
        _ToyDataset(
            [
                {
                    "action": torch.tensor([0.10, 0.05, 0.02, -0.20], dtype=torch.float32),
                    "observation": {
                        "state": torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
                    },
                }
            ]
        ),
        build_stage_label_generator("hard"),
        include_debug=True,
    )

    hard_analyzer = StageLabelStatistics(hard_dataset, verbose=False)
    hard_analyzer.compute()

    assert hard_analyzer.source_counts["hard"] == 1
    assert hard_analyzer.stage_id_counts
    assert hard_analyzer.soft_label_probs == []
    assert hard_analyzer.soft_label_max_probs == []
    assert hard_analyzer.soft_label_entropy_values == []

    soft_dataset = StageLabeledDataset(
        _ToyDataset(
            [
                {
                    "action": torch.tensor([0.05, 0.02, 0.01, -0.40], dtype=torch.float32),
                }
            ]
        ),
        build_stage_label_generator("soft"),
        include_debug=True,
    )

    soft_analyzer = StageLabelStatistics(soft_dataset, verbose=False)
    soft_analyzer.compute()

    assert soft_analyzer.source_counts["soft"] == 1
    assert soft_analyzer.stage_id_counts == {}
    assert soft_analyzer.stage_name_counts == {}
    assert len(soft_analyzer.soft_label_probs) == 1
    assert len(soft_analyzer.soft_label_max_probs) == 1
    assert len(soft_analyzer.soft_label_entropy_values) == 1

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

import pytest
import torch

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import (
    build_lowpass_target,
    compute_component_regularization_losses,
    compute_component_teacher_weight,
    interpolate_trend_from_anchors,
    reduce_action_loss_per_sample,
)


def test_component_config_defaults_keep_feature_disabled():
    config = SmolVLAConfig()

    assert config.enable_action_component_branch is False
    assert config.enable_suffix_component_film is False
    assert config.component_teacher_min_weight == 0.1
    assert config.loss_weight_recon == 0.25
    assert config.loss_weight_comp == 0.05


def test_component_config_rejects_invalid_anchor_count():
    with pytest.raises(ValueError, match="component_num_anchors"):
        SmolVLAConfig(enable_action_component_branch=True, component_num_anchors=1)


def test_component_config_rejects_invalid_teacher_schedule():
    with pytest.raises(ValueError, match="component_teacher_min_weight"):
        SmolVLAConfig(enable_action_component_branch=True, component_teacher_min_weight=1.2)

    with pytest.raises(ValueError, match="component_teacher_warmup_ratio"):
        SmolVLAConfig(
            enable_action_component_branch=True,
            component_teacher_warmup_ratio=0.4,
            component_teacher_decay_ratio=0.8,
        )


def test_component_teacher_weight_has_warmup_cosine_decay_and_floor():
    assert compute_component_teacher_weight(0, 100, 0.1, 0.7, 0.1) == pytest.approx(1.0)
    assert compute_component_teacher_weight(10, 100, 0.1, 0.7, 0.1) == pytest.approx(1.0)

    mid = compute_component_teacher_weight(50, 100, 0.1, 0.7, 0.1)
    end = compute_component_teacher_weight(95, 100, 0.1, 0.7, 0.1)

    assert 0.1 < mid < 1.0
    assert end == pytest.approx(0.1)


def test_interpolate_trend_from_anchors_respects_chunk_shape():
    anchor_values = torch.tensor([[[0.0], [1.0], [0.5]]], dtype=torch.float32)
    anchor_positions = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float32)

    trend = interpolate_trend_from_anchors(anchor_values, anchor_positions, chunk_size=5)

    assert trend.shape == (1, 5, 1)
    assert trend[:, 0].item() == pytest.approx(0.0)
    assert trend[:, -1].item() == pytest.approx(0.5)


def test_component_regularizers_penalize_uniform_gates_and_large_residuals():
    gates = torch.full((2, 2), 0.5, dtype=torch.float32)
    trend = torch.zeros(2, 4, 3, dtype=torch.float32)
    lowpass = torch.ones(2, 4, 3, dtype=torch.float32)
    refined = torch.full((2, 4, 3), 4.0, dtype=torch.float32)

    reg = compute_component_regularization_losses(gates, trend, lowpass, refined)

    assert set(reg) == {
        "loss_reg_gate_entropy",
        "loss_reg_gate_balance",
        "loss_reg_trend_prior",
        "loss_reg_ref_mag",
        "loss_reg_ref_center",
    }
    assert reg["loss_reg_gate_entropy"].shape == (2,)
    assert torch.all(reg["loss_reg_ref_mag"] > 0)


def test_reduce_action_loss_per_sample_respects_action_is_pad():
    losses = torch.tensor(
        [
            [[1.0, 3.0], [10.0, 10.0]],
            [[2.0, 4.0], [6.0, 8.0]],
        ]
    )
    action_is_pad = torch.tensor([[False, True], [False, False]])

    reduced = reduce_action_loss_per_sample(losses, action_is_pad)

    assert torch.allclose(reduced, torch.tensor([2.0, 5.0]))

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

from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig


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

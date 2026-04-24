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
from types import SimpleNamespace
from torch import nn

from lerobot.configs.types import FeatureType, PolicyFeature
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.policies.smolvla.modeling_smolvla import (
    ActionChunkEncoder,
    ActionComponentDecompositionHead,
    ComponentPredictor,
    FiLMConditioner,
    SuffixComponentFiLM,
    build_lowpass_target,
    compute_component_regularization_losses,
    compute_component_teacher_weight,
    interpolate_trend_from_anchors,
    reduce_action_loss_per_sample,
)
from lerobot.policies.smolvla.smolvlm_with_expert import (
    SmolVLMForwardOutput,
    extract_prefix_state_token,
)
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


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


def test_extract_prefix_state_token_uses_last_valid_prefix_token():
    prefix_hidden = torch.tensor(
        [[[1.0, 1.0], [2.0, 2.0], [9.0, 9.0], [0.0, 0.0]]],
        dtype=torch.float32,
    )
    prefix_pad_masks = torch.tensor([[True, True, True, False]])

    state_token = extract_prefix_state_token(prefix_hidden, prefix_pad_masks)

    assert state_token.shape == (1, 2)
    assert torch.equal(state_token, torch.tensor([[9.0, 9.0]]))


def test_smolvlm_forward_output_keeps_optional_prefix_state_token():
    output = SmolVLMForwardOutput(
        outputs_embeds=[torch.ones(1, 3, 4), torch.ones(1, 2, 6)],
        past_key_values={},
        prefix_state_token=torch.zeros(1, 4),
    )

    assert output.prefix_state_token.shape == (1, 4)


def test_action_component_modules_produce_chunk_level_outputs():
    batch_size = 2
    chunk_size = 5
    action_dim = 3
    hidden_dim = 8
    num_anchors = 3

    encoder = ActionChunkEncoder(
        action_dim=action_dim,
        hidden_dim=hidden_dim,
        kernel_size=3,
        pooling="mean",
    )
    film = FiLMConditioner(context_dim=6, target_dim=hidden_dim, hidden_dim=10)
    head = ActionComponentDecompositionHead(
        hidden_dim=hidden_dim,
        action_dim=action_dim,
        chunk_size=chunk_size,
        num_anchors=num_anchors,
    )
    predictor = ComponentPredictor(context_dim=6, hidden_dim=10)
    suffix_film = SuffixComponentFiLM(component_dim=2, hidden_dim=12, target_dim=7)

    actions = torch.randn(batch_size, chunk_size, action_dim)
    context = torch.randn(batch_size, 6)

    encoded = encoder(actions)
    conditioned = film(encoded, context)
    outputs = head(conditioned)
    z_hat = predictor(context)
    gamma, beta = suffix_film(z_hat)

    assert encoded.shape == (batch_size, hidden_dim)
    assert conditioned.shape == (batch_size, hidden_dim)
    assert outputs["gates"].shape == (batch_size, 2)
    assert outputs["anchor_values"].shape == (batch_size, num_anchors, action_dim)
    assert outputs["trend"].shape == (batch_size, chunk_size, action_dim)
    assert outputs["refined"].shape == (batch_size, chunk_size, action_dim)
    assert outputs["reconstruction"].shape == (batch_size, chunk_size, action_dim)
    assert z_hat.shape == (batch_size, 2)
    assert gamma.shape == (batch_size, 1, 7)
    assert beta.shape == (batch_size, 1, 7)


class DummySmolVLM(nn.Module):
    def __init__(self, text_hidden_size: int, expert_hidden_size: int):
        super().__init__()
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=text_hidden_size))
        self.expert_hidden_size = expert_hidden_size
        self.processor = SimpleNamespace(
            tokenizer=SimpleNamespace(fake_image_token_id=0, global_image_token_id=1)
        )
        self.vlm = SimpleNamespace(device=torch.device("cpu"))

    def embed_language_tokens(self, tokens):
        return torch.zeros(tokens.shape[0], tokens.shape[1], self.config.text_config.hidden_size)

    def embed_image(self, image):
        return torch.zeros(image.shape[0], 2, self.config.text_config.hidden_size)

    def forward(
        self,
        attention_mask,
        position_ids,
        past_key_values,
        inputs_embeds,
        use_cache,
        fill_kv_cache,
        prefix_pad_masks=None,
        return_prefix_state_token=False,
    ):
        prefix, suffix = inputs_embeds
        suffix_hidden = None
        if suffix is not None:
            suffix_hidden = torch.zeros(
                prefix.shape[0], suffix.shape[1], self.expert_hidden_size, device=prefix.device
            )
        prefix_state_token = (
            extract_prefix_state_token(prefix, prefix_pad_masks) if return_prefix_state_token else None
        )
        return SmolVLMForwardOutput(
            outputs_embeds=[prefix, suffix_hidden],
            past_key_values={},
            prefix_state_token=prefix_state_token,
        )


def make_component_policy(monkeypatch):
    monkeypatch.setattr(
        "lerobot.policies.smolvla.modeling_smolvla.require_package",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        "lerobot.policies.smolvla.modeling_smolvla.SmolVLMWithExpertModel",
        lambda *args, **kwargs: DummySmolVLM(text_hidden_size=8, expert_hidden_size=6),
    )
    config = SmolVLAConfig(
        max_state_dim=4,
        max_action_dim=3,
        chunk_size=5,
        n_action_steps=5,
        enable_action_component_branch=True,
        enable_suffix_component_film=True,
        component_hidden_dim=8,
        component_predictor_hidden_dim=8,
        suffix_component_film_hidden_dim=8,
    )
    config.input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(4,)),
        "observation.images.base_0_rgb": PolicyFeature(type=FeatureType.VISUAL, shape=(3, 8, 8)),
    }
    config.output_features = {"action": PolicyFeature(type=FeatureType.ACTION, shape=(3,))}
    return SmolVLAPolicy(config)


def test_smolvla_forward_reports_grouped_component_losses(monkeypatch):
    policy = make_component_policy(monkeypatch)
    batch = {
        OBS_STATE: torch.randn(2, 4),
        "observation.images.base_0_rgb": torch.rand(2, 3, 8, 8),
        OBS_LANGUAGE_TOKENS: torch.ones(2, 3, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 3, dtype=torch.bool),
        ACTION: torch.randn(2, 5, 3),
        "action_is_pad": torch.tensor([[False, False, False, False, True], [False, False, False, False, False]]),
    }

    loss, loss_dict = policy.forward(batch, reduction="mean")

    assert torch.isfinite(loss)
    assert {"loss_flow", "loss_recon", "loss_comp", "loss_reg"} <= set(loss_dict)
    assert {"loss_reg_gate_entropy", "loss_reg_gate_balance", "loss_reg_trend_prior"} <= set(loss_dict)


def test_smolvla_forward_reduction_none_keeps_per_sample_loss(monkeypatch):
    policy = make_component_policy(monkeypatch)
    batch = {
        OBS_STATE: torch.randn(2, 4),
        "observation.images.base_0_rgb": torch.rand(2, 3, 8, 8),
        OBS_LANGUAGE_TOKENS: torch.ones(2, 3, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(2, 3, dtype=torch.bool),
        ACTION: torch.randn(2, 5, 3),
        "action_is_pad": torch.zeros(2, 5, dtype=torch.bool),
    }

    losses, loss_dict = policy.forward(batch, reduction="none")

    assert losses.shape == (2,)
    assert torch.isfinite(losses).all()
    assert loss_dict["loss"] == pytest.approx(losses.mean().item())

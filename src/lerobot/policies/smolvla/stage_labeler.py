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

"""Stage labeler for SmolVLA future action chunks.

This module augments the future-action stage labeler with a multimodal
conditioning branch that encodes:

1. image observations ``ob.i`` with a frozen SmolVLM vision encoder,
2. robot state ``ob.s`` with a small trainable MLP,
3. task tokens with a frozen SmolVLM text encoder,

then fuses them into a conditioning vector ``et``. The pooled action
representation is modulated by ``et`` through FiLM before the downstream
mixture, coarse-anchor, and residual heads.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

try:
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.policies.smolvla.smolvlm_with_expert import SmolVLMWithExpertModel
except ModuleNotFoundError:
    from configuration_smolvla import SmolVLAConfig
    from smolvlm_with_expert import SmolVLMWithExpertModel


@dataclass(slots=True)
class StageLabelerConfig:
    """Configuration for :class:`FutureChunkStageLabeler`.

    Attributes:
        chunk_size: Temporal horizon ``H`` of the future action chunk.
        action_dim: Action dimensionality ``D``.
        encoder_hidden_dim: Hidden width used by the temporal encoder and heads.
        encoder_num_layers: Number of temporal convolution layers.
        encoder_kernel_size: Kernel size of each temporal convolution.
        encoder_dropout: Dropout applied inside the encoder and MLP heads.
        num_anchors: Number of coarse anchors ``K`` used for interpolation.
        residual_l1_weight: Weight applied to the residual sparsity loss.
        entropy_weight: Weight applied to the posterior entropy term.
        balance_weight: Weight applied to the batch-level mixture balancing loss.
        spacing_weight: Weight applied to the anchor spacing penalty.
        target_probs: Desired batch-average mixture probabilities
            ``(p_global, p_local)``.
        min_anchor_spacing_ratio: Minimum adjacent spacing, expressed as a
            fraction of the uniform spacing ``(H - 1) / (K - 1)``.
        eps: Numerical epsilon used in divisions and logarithms.
        condition_hidden_dim: Output width of the fused conditioning vector.
        state_hidden_dim: Hidden width of the robot-state encoder MLP.
        freeze_condition_encoders: Whether the SmolVLM image/text encoders stay
            frozen and in eval mode.
        vlm_model_name: SmolVLM checkpoint used for image/text encoding.
        image_proj_dim: Projected width of the pooled image feature.
        text_proj_dim: Projected width of the pooled text feature.
        film_hidden_dim: Hidden width of the FiLM MLP that modulates the
            action chunk representation.
        use_condition_film: Whether to apply FiLM modulation when a condition
            embedding is available.
    """

    chunk_size: int
    action_dim: int
    encoder_hidden_dim: int = 256
    encoder_num_layers: int = 3
    encoder_kernel_size: int = 5
    encoder_dropout: float = 0.1
    num_anchors: int = 4
    residual_l1_weight: float = 0.05
    entropy_weight: float = 0.01
    balance_weight: float = 0.1
    spacing_weight: float = 0.1
    target_probs: tuple[float, float] = (0.5, 0.5)
    min_anchor_spacing_ratio: float = 0.5
    eps: float = 1e-8
    condition_hidden_dim: int = 256
    state_hidden_dim: int = 256
    freeze_condition_encoders: bool = True
    vlm_model_name: str = SmolVLAConfig.vlm_model_name
    image_proj_dim: int | None = None
    text_proj_dim: int | None = None
    film_hidden_dim: int = 256
    use_condition_film: bool = True

    def __post_init__(self) -> None:
        if self.chunk_size <= 1:
            raise ValueError(f"`chunk_size` must be greater than 1, got {self.chunk_size}.")
        if self.action_dim <= 0:
            raise ValueError(f"`action_dim` must be positive, got {self.action_dim}.")
        if self.encoder_hidden_dim <= 0:
            raise ValueError(
                f"`encoder_hidden_dim` must be positive, got {self.encoder_hidden_dim}."
            )
        if self.encoder_num_layers <= 0:
            raise ValueError(
                f"`encoder_num_layers` must be positive, got {self.encoder_num_layers}."
            )
        if self.encoder_kernel_size <= 0:
            raise ValueError(
                f"`encoder_kernel_size` must be positive, got {self.encoder_kernel_size}."
            )
        if self.encoder_kernel_size % 2 == 0:
            raise ValueError(
                "`encoder_kernel_size` must be odd so the temporal encoder preserves chunk length."
            )
        if self.num_anchors < 2:
            raise ValueError(f"`num_anchors` must be at least 2, got {self.num_anchors}.")
        if len(self.target_probs) != 2:
            raise ValueError(
                f"`target_probs` must contain exactly 2 entries, got {len(self.target_probs)}."
            )
        target_probs_sum = sum(self.target_probs)
        if target_probs_sum <= 0.0:
            raise ValueError(
                "`target_probs` must sum to a positive value so it can represent a valid mixture."
            )
        if any(prob < 0.0 for prob in self.target_probs):
            raise ValueError("`target_probs` must be non-negative.")
        if self.min_anchor_spacing_ratio < 0.0:
            raise ValueError(
                "`min_anchor_spacing_ratio` must be non-negative, "
                f"got {self.min_anchor_spacing_ratio}."
            )
        if self.eps <= 0.0:
            raise ValueError(f"`eps` must be positive, got {self.eps}.")
        if self.condition_hidden_dim <= 0:
            raise ValueError(
                f"`condition_hidden_dim` must be positive, got {self.condition_hidden_dim}."
            )
        if self.state_hidden_dim <= 0:
            raise ValueError(f"`state_hidden_dim` must be positive, got {self.state_hidden_dim}.")
        if self.image_proj_dim is not None and self.image_proj_dim <= 0:
            raise ValueError(f"`image_proj_dim` must be positive, got {self.image_proj_dim}.")
        if self.text_proj_dim is not None and self.text_proj_dim <= 0:
            raise ValueError(f"`text_proj_dim` must be positive, got {self.text_proj_dim}.")
        if self.film_hidden_dim <= 0:
            raise ValueError(f"`film_hidden_dim` must be positive, got {self.film_hidden_dim}.")

        normalized_probs = tuple(prob / target_probs_sum for prob in self.target_probs)
        object.__setattr__(self, "target_probs", normalized_probs)
        if self.image_proj_dim is None:
            object.__setattr__(self, "image_proj_dim", self.condition_hidden_dim)
        if self.text_proj_dim is None:
            object.__setattr__(self, "text_proj_dim", self.condition_hidden_dim)


@dataclass(slots=True)
class StageLabelerOutput:
    """Container returned by :class:`FutureChunkStageLabeler`.

    The reconstruction tensors are always returned with a batch dimension. When
    the input action chunk is provided as ``[H, D]``, the model internally
    expands it to ``[1, H, D]`` and all output tensors follow that shape.

    Loss fields follow the requested reduction:
    - ``*_per_sample`` always have shape ``[B]``.
    - ``loss``, ``recon_l1``, ``residual_l1``, ``entropy`` and ``spacing_loss``
      match the ``reduction`` argument passed to ``forward``.
    - ``balance_loss`` is batch-level by definition and is always scalar.
    """

    loss: Tensor
    loss_per_sample: Tensor
    total_loss: Tensor
    total_loss_per_sample: Tensor
    recon_l1: Tensor
    recon_l1_per_sample: Tensor
    residual_l1: Tensor
    residual_l1_per_sample: Tensor
    entropy: Tensor
    entropy_per_sample: Tensor
    balance_loss: Tensor
    balance_loss_per_sample: Tensor
    spacing_loss: Tensor
    spacing_loss_per_sample: Tensor
    encoded_chunk: Tensor
    action_chunk: Tensor
    valid_mask: Tensor
    mixture_logits: Tensor
    stage_probs: Tensor
    mean_probs: Tensor
    target_probs: Tensor
    p_global: Tensor
    p_local: Tensor
    anchor_values: Tensor
    anchor_positions: Tensor
    anchor_gaps: Tensor
    coarse_chunk: Tensor
    residual_chunk: Tensor
    refined_chunk: Tensor
    mixed_chunk: Tensor
    image_embedding: Tensor | None = None
    state_embedding: Tensor | None = None
    text_embedding: Tensor | None = None
    condition_embedding: Tensor | None = None
    conditioned_chunk: Tensor | None = None


class ChunkEncoder1D(nn.Module):
    """Temporal 1D convolution encoder for future action chunks."""

    def __init__(self, config: StageLabelerConfig) -> None:
        super().__init__()
        padding = config.encoder_kernel_size // 2

        layers: list[nn.Module] = []
        in_channels = config.action_dim
        for _ in range(config.encoder_num_layers):
            layers.extend(
                [
                    nn.Conv1d(
                        in_channels=in_channels,
                        out_channels=config.encoder_hidden_dim,
                        kernel_size=config.encoder_kernel_size,
                        padding=padding,
                    ),
                    nn.GELU(),
                    nn.Dropout(config.encoder_dropout),
                ]
            )
            in_channels = config.encoder_hidden_dim

        self.temporal_conv = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.output_proj = nn.Sequential(
            nn.Linear(config.encoder_hidden_dim, config.encoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.encoder_dropout),
        )

    def forward(self, action_chunk: Tensor, action_is_pad: Tensor | None = None) -> Tensor:
        """Encode an action chunk into a single chunk-level representation."""

        if action_chunk.ndim != 3:
            raise ValueError(
                f"`action_chunk` must have shape [B, H, D], got {tuple(action_chunk.shape)}."
            )

        masked_chunk = action_chunk
        if action_is_pad is not None:
            valid_mask = (~action_is_pad).unsqueeze(-1).to(dtype=action_chunk.dtype)
            masked_chunk = action_chunk * valid_mask

        conv_input = masked_chunk.transpose(1, 2)
        hidden = self.temporal_conv(conv_input)
        if action_is_pad is None:
            pooled = self.pool(hidden).squeeze(-1)
        else:
            pooled_mask = (~action_is_pad).unsqueeze(1).to(dtype=hidden.dtype)
            denom = pooled_mask.sum(dim=-1).clamp_min(1.0)
            pooled = (hidden * pooled_mask).sum(dim=-1) / denom
        return self.output_proj(pooled)


class MixtureHead(nn.Module):
    """Predict the global/local option posterior for a chunk."""

    def __init__(self, config: StageLabelerConfig) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(config.encoder_hidden_dim, config.encoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.encoder_dropout),
            nn.Linear(config.encoder_hidden_dim, 2),
        )

    def forward(self, encoded_chunk: Tensor) -> tuple[Tensor, Tensor]:
        logits = self.proj(encoded_chunk)
        probs = torch.softmax(logits, dim=-1)
        return logits, probs


class CoarseAnchorDecoder(nn.Module):
    """Decode a coarse chunk using learned anchors and fixed interpolation."""

    def __init__(self, config: StageLabelerConfig) -> None:
        super().__init__()
        self.chunk_size = config.chunk_size
        self.action_dim = config.action_dim
        self.num_anchors = config.num_anchors
        self.min_anchor_spacing_ratio = config.min_anchor_spacing_ratio
        self.eps = config.eps

        self.anchor_value_head = nn.Sequential(
            nn.Linear(config.encoder_hidden_dim, config.encoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.encoder_dropout),
            nn.Linear(config.encoder_hidden_dim, config.num_anchors * config.action_dim),
        )
        self.anchor_position_head = nn.Sequential(
            nn.Linear(config.encoder_hidden_dim, config.encoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.encoder_dropout),
            nn.Linear(config.encoder_hidden_dim, config.num_anchors),
        )

    def forward(self, encoded_chunk: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch_size = encoded_chunk.shape[0]
        anchor_values = self.anchor_value_head(encoded_chunk).view(
            batch_size, self.num_anchors, self.action_dim
        )

        raw_positions = self.anchor_position_head(encoded_chunk)
        anchor_positions, sort_indices = self._normalize_anchor_positions(raw_positions)
        anchor_values = anchor_values.gather(
            1, sort_indices.unsqueeze(-1).expand(-1, -1, self.action_dim)
        )
        coarse_chunk = self._piecewise_linear_interpolate(anchor_positions, anchor_values)
        return coarse_chunk, anchor_values, anchor_positions

    def spacing_loss(self, anchor_positions: Tensor) -> tuple[Tensor, Tensor]:
        anchor_gaps = anchor_positions[:, 1:] - anchor_positions[:, :-1]
        uniform_gap = float(self.chunk_size - 1) / float(self.num_anchors - 1)
        minimum_gap = self.min_anchor_spacing_ratio * uniform_gap
        loss_per_sample = F.relu(minimum_gap - anchor_gaps).pow(2).mean(dim=-1)
        return loss_per_sample, anchor_gaps

    def _normalize_anchor_positions(self, raw_positions: Tensor) -> tuple[Tensor, Tensor]:
        scaled_positions = torch.sigmoid(raw_positions) * float(self.chunk_size - 1)
        anchor_positions, sort_indices = torch.sort(scaled_positions, dim=-1)
        anchor_positions = anchor_positions.clone()
        anchor_positions[:, 0] = 0.0
        anchor_positions[:, -1] = float(self.chunk_size - 1)
        return anchor_positions, sort_indices

    def _piecewise_linear_interpolate(self, anchor_positions: Tensor, anchor_values: Tensor) -> Tensor:
        time_grid = torch.arange(
            self.chunk_size, device=anchor_positions.device, dtype=anchor_positions.dtype
        )
        coarse_chunks: list[Tensor] = []

        for positions, values in zip(anchor_positions.unbind(dim=0), anchor_values.unbind(dim=0), strict=True):
            right_indices = torch.searchsorted(positions, time_grid, right=True)
            right_indices = right_indices.clamp(min=1, max=self.num_anchors - 1)
            left_indices = right_indices - 1

            left_positions = positions.gather(0, left_indices)
            right_positions = positions.gather(0, right_indices)
            left_values = values.gather(0, left_indices.unsqueeze(-1).expand(-1, self.action_dim))
            right_values = values.gather(0, right_indices.unsqueeze(-1).expand(-1, self.action_dim))

            denom = (right_positions - left_positions).clamp_min(self.eps)
            alpha = ((time_grid - left_positions) / denom).unsqueeze(-1)
            coarse_chunks.append(left_values + alpha * (right_values - left_values))

        return torch.stack(coarse_chunks, dim=0)


class ResidualDecoder(nn.Module):
    """Decode a full residual chunk from the chunk representation."""

    def __init__(self, config: StageLabelerConfig) -> None:
        super().__init__()
        self.chunk_size = config.chunk_size
        self.action_dim = config.action_dim
        self.proj = nn.Sequential(
            nn.Linear(config.encoder_hidden_dim, config.encoder_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.encoder_dropout),
            nn.Linear(config.encoder_hidden_dim, config.chunk_size * config.action_dim),
        )

    def forward(self, encoded_chunk: Tensor) -> Tensor:
        batch_size = encoded_chunk.shape[0]
        residual = self.proj(encoded_chunk)
        return residual.view(batch_size, self.chunk_size, self.action_dim)


class FrozenConditionEncoder(nn.Module):
    """Encode image and language inputs with a frozen SmolVLM backbone."""

    def __init__(self, config: StageLabelerConfig) -> None:
        super().__init__()
        self.config = config
        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=config.vlm_model_name,
            load_vlm_weights=True,
            train_expert_only=True,
            freeze_vision_encoder=True,
        )
        vlm_hidden_dim = int(self.vlm_with_expert.config.text_config.hidden_size)
        self.image_proj = nn.Linear(vlm_hidden_dim, int(config.image_proj_dim))
        self.text_proj = nn.Linear(vlm_hidden_dim, int(config.text_proj_dim))
        self._configure_backbone_trainability()

    def encode_image(self, obs_image: Tensor) -> Tensor:
        if obs_image.ndim != 4:
            raise ValueError(
                f"`obs_image` must have shape [B, C, H, W], got {tuple(obs_image.shape)}."
            )

        with self._backbone_context():
            image_tokens = self.vlm_with_expert.embed_image(obs_image)
        image_feature = image_tokens.mean(dim=1)
        image_feature = image_feature.to(dtype=self.image_proj.weight.dtype)
        return self.image_proj(image_feature)

    def encode_text(self, task_tokens: Tensor, task_mask: Tensor) -> Tensor:
        if task_tokens.ndim != 2:
            raise ValueError(
                f"`task_tokens` must have shape [B, L], got {tuple(task_tokens.shape)}."
            )
        if task_mask.ndim != 2:
            raise ValueError(
                f"`task_mask` must have shape [B, L], got {tuple(task_mask.shape)}."
            )
        if task_tokens.shape != task_mask.shape:
            raise ValueError(
                "`task_tokens` and `task_mask` must have identical shapes, "
                f"got {tuple(task_tokens.shape)} and {tuple(task_mask.shape)}."
            )

        text_model = self._get_text_backbone()
        attention_mask = task_mask.to(device=task_tokens.device, dtype=torch.long)

        with self._backbone_context():
            lang_embeddings = self.vlm_with_expert.embed_language_tokens(task_tokens)
            text_outputs = text_model(
                inputs_embeds=lang_embeddings.to(dtype=self._get_text_dtype()),
                attention_mask=attention_mask,
                output_hidden_states=False,
                return_dict=True,
                use_cache=False,
            )

        hidden_states = text_outputs.last_hidden_state
        mask = task_mask.to(device=hidden_states.device, dtype=hidden_states.dtype).unsqueeze(-1)
        pooled_feature = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        pooled_feature = pooled_feature.to(dtype=self.text_proj.weight.dtype)
        return self.text_proj(pooled_feature)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.config.freeze_condition_encoders:
            self.vlm_with_expert.eval()
        return self

    def _configure_backbone_trainability(self) -> None:
        if self.config.freeze_condition_encoders:
            for parameter in self.vlm_with_expert.parameters():
                parameter.requires_grad = False
            self.vlm_with_expert.eval()
        else:
            self.vlm_with_expert.freeze_vision_encoder = False
            self.vlm_with_expert.train_expert_only = False
            for parameter in self.vlm_with_expert.parameters():
                parameter.requires_grad = True

    def _backbone_context(self):
        if self.config.freeze_condition_encoders:
            return torch.inference_mode()
        return nullcontext()

    def _get_text_backbone(self):
        text_model = self.vlm_with_expert.get_vlm_model().text_model
        return getattr(text_model, "model", text_model)

    def _get_text_dtype(self) -> torch.dtype:
        text_backbone = self._get_text_backbone()
        return next(text_backbone.parameters()).dtype


class StateEncoder(nn.Module):
    """Encode robot state with a small trainable MLP."""

    def __init__(self, config: StageLabelerConfig) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.LazyLinear(config.state_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.encoder_dropout),
            nn.Linear(config.state_hidden_dim, config.condition_hidden_dim),
        )

    def forward(self, obs_state: Tensor) -> Tensor:
        if obs_state.ndim == 3:
            obs_state = obs_state[:, -1, :]
        elif obs_state.ndim != 2:
            raise ValueError(
                f"`obs_state` must have shape [B, D] or [B, T, D], got {tuple(obs_state.shape)}."
            )
        return self.proj(obs_state.to(dtype=torch.float32))


class ConditionFusionMLP(nn.Module):
    """Fuse image/state/text features into a single condition embedding ``et``."""

    def __init__(self, config: StageLabelerConfig) -> None:
        super().__init__()
        input_dim = int(config.image_proj_dim) + config.condition_hidden_dim + int(config.text_proj_dim)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, config.condition_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.encoder_dropout),
            nn.Linear(config.condition_hidden_dim, config.condition_hidden_dim),
        )

    def forward(self, image_embedding: Tensor, state_embedding: Tensor, text_embedding: Tensor) -> Tensor:
        fused = torch.cat([image_embedding, state_embedding, text_embedding], dim=-1)
        return self.proj(fused)


class ConditionFiLM(nn.Module):
    """Apply FiLM modulation to the pooled action embedding."""

    def __init__(self, config: StageLabelerConfig) -> None:
        super().__init__()
        self.hidden = nn.Sequential(
            nn.Linear(config.condition_hidden_dim, config.film_hidden_dim),
            nn.GELU(),
            nn.Dropout(config.encoder_dropout),
        )
        self.gamma_head = nn.Linear(config.film_hidden_dim, config.encoder_hidden_dim)
        self.beta_head = nn.Linear(config.film_hidden_dim, config.encoder_hidden_dim)
        nn.init.normal_(self.gamma_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.gamma_head.bias)
        nn.init.normal_(self.beta_head.weight, mean=0.0, std=1e-3)
        nn.init.zeros_(self.beta_head.bias)

    def forward(self, encoded_chunk: Tensor, condition_embedding: Tensor) -> Tensor:
        if encoded_chunk.ndim != 2:
            raise ValueError(
                f"`encoded_chunk` must have shape [B, D], got {tuple(encoded_chunk.shape)}."
            )
        if condition_embedding.ndim != 2:
            raise ValueError(
                "`condition_embedding` must have shape [B, C], "
                f"got {tuple(condition_embedding.shape)}."
            )
        if encoded_chunk.shape[0] != condition_embedding.shape[0]:
            raise ValueError(
                "FiLM inputs must share the same batch size, "
                f"got {encoded_chunk.shape[0]} and {condition_embedding.shape[0]}."
            )

        hidden = self.hidden(condition_embedding)
        gamma_delta = self.gamma_head(hidden)
        beta = self.beta_head(hidden)
        gamma = 1.0 + gamma_delta
        return gamma * encoded_chunk + beta


class FutureChunkStageLabeler(nn.Module):
    """Infer a chunk-level option posterior from future actions.

    The action encoder / decoders remain unchanged. A separate multimodal
    branch computes image, state, and text features plus the fused condition
    embedding ``et`` and injects it into the pooled action representation via
    FiLM before the downstream heads.
    """

    def __init__(self, config: StageLabelerConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = ChunkEncoder1D(config)
        self.mixture_head = MixtureHead(config)
        self.coarse_decoder = CoarseAnchorDecoder(config)
        self.residual_decoder = ResidualDecoder(config)
        self.frozen_vlm_encoder = FrozenConditionEncoder(config)
        self.state_encoder = StateEncoder(config)
        self.condition_fusion = ConditionFusionMLP(config)
        self.condition_film = ConditionFiLM(config)
        self.register_buffer(
            "target_probs",
            torch.tensor(config.target_probs, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        action_chunk: torch.Tensor,
        obs_image: torch.Tensor | None = None,
        obs_state: torch.Tensor | None = None,
        task_tokens: torch.Tensor | None = None,
        task_mask: torch.Tensor | None = None,
        action_is_pad: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> StageLabelerOutput:
        """Run the stage labeler and compute the training losses."""

        action_chunk, action_is_pad = self._prepare_inputs(action_chunk, action_is_pad)
        image_embedding, state_embedding, text_embedding, condition_embedding = (
            self._encode_conditioning_inputs(
                obs_image=obs_image,
                obs_state=obs_state,
                task_tokens=task_tokens,
                task_mask=task_mask,
                batch_size=action_chunk.shape[0],
            )
        )

        encoded_chunk = self.encoder(action_chunk, action_is_pad=action_is_pad)
        conditioned_chunk = self._apply_condition_film(encoded_chunk, condition_embedding)
        mixture_logits, stage_probs = self.mixture_head(conditioned_chunk)
        coarse_chunk, anchor_values, anchor_positions = self.coarse_decoder(conditioned_chunk)
        residual_chunk = self.residual_decoder(conditioned_chunk)
        refined_chunk = coarse_chunk + residual_chunk

        p_global = stage_probs[:, 0]
        p_local = stage_probs[:, 1]
        mixed_chunk = (
            p_global.view(-1, 1, 1) * coarse_chunk + p_local.view(-1, 1, 1) * refined_chunk
        )

        valid_mask = self._make_valid_mask(action_chunk, action_is_pad)
        recon_l1_per_sample = self._masked_l1_per_sample(mixed_chunk, action_chunk, valid_mask)
        residual_l1_per_sample = self._masked_l1_per_sample(
            residual_chunk, torch.zeros_like(residual_chunk), valid_mask
        )
        entropy_per_sample = -(stage_probs * torch.log(stage_probs.clamp_min(self.config.eps))).sum(dim=-1)

        mean_probs = stage_probs.mean(dim=0)
        target_probs = self.target_probs.to(device=stage_probs.device, dtype=stage_probs.dtype)
        balance_loss = (mean_probs - target_probs).pow(2).sum()
        balance_loss_per_sample = balance_loss.expand(action_chunk.shape[0])

        spacing_loss_per_sample, anchor_gaps = self.coarse_decoder.spacing_loss(anchor_positions)

        loss_per_sample = (
            recon_l1_per_sample
            + self.config.residual_l1_weight * residual_l1_per_sample
            + self.config.entropy_weight * entropy_per_sample
            + self.config.balance_weight * balance_loss_per_sample
            + self.config.spacing_weight * spacing_loss_per_sample
        )

        recon_l1 = self._reduce_loss(recon_l1_per_sample, reduction)
        residual_l1 = self._reduce_loss(residual_l1_per_sample, reduction)
        entropy = self._reduce_loss(entropy_per_sample, reduction)
        spacing_loss = self._reduce_loss(spacing_loss_per_sample, reduction)
        loss = self._reduce_loss(loss_per_sample, reduction)

        return StageLabelerOutput(
            loss=loss,
            loss_per_sample=loss_per_sample,
            total_loss=loss,
            total_loss_per_sample=loss_per_sample,
            recon_l1=recon_l1,
            recon_l1_per_sample=recon_l1_per_sample,
            residual_l1=residual_l1,
            residual_l1_per_sample=residual_l1_per_sample,
            entropy=entropy,
            entropy_per_sample=entropy_per_sample,
            balance_loss=balance_loss,
            balance_loss_per_sample=balance_loss_per_sample,
            spacing_loss=spacing_loss,
            spacing_loss_per_sample=spacing_loss_per_sample,
            encoded_chunk=encoded_chunk,
            action_chunk=action_chunk,
            valid_mask=valid_mask,
            mixture_logits=mixture_logits,
            stage_probs=stage_probs,
            mean_probs=mean_probs,
            target_probs=target_probs,
            p_global=p_global,
            p_local=p_local,
            anchor_values=anchor_values,
            anchor_positions=anchor_positions,
            anchor_gaps=anchor_gaps,
            coarse_chunk=coarse_chunk,
            residual_chunk=residual_chunk,
            refined_chunk=refined_chunk,
            mixed_chunk=mixed_chunk,
            image_embedding=image_embedding,
            state_embedding=state_embedding,
            text_embedding=text_embedding,
            condition_embedding=condition_embedding,
            conditioned_chunk=conditioned_chunk,
        )

    def predict_stage_probs(
        self,
        action_chunk: torch.Tensor,
        obs_image: torch.Tensor | None = None,
        obs_state: torch.Tensor | None = None,
        task_tokens: torch.Tensor | None = None,
        task_mask: torch.Tensor | None = None,
        action_is_pad: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict stage posterior probabilities without computing losses."""

        action_chunk, action_is_pad = self._prepare_inputs(action_chunk, action_is_pad)
        _, _, _, condition_embedding = self._encode_conditioning_inputs(
            obs_image=obs_image,
            obs_state=obs_state,
            task_tokens=task_tokens,
            task_mask=task_mask,
            batch_size=action_chunk.shape[0],
        )
        encoded_chunk = self.encoder(action_chunk, action_is_pad=action_is_pad)
        conditioned_chunk = self._apply_condition_film(encoded_chunk, condition_embedding)
        _, stage_probs = self.mixture_head(conditioned_chunk)
        return stage_probs

    def _encode_conditioning_inputs(
        self,
        *,
        obs_image: Tensor | None,
        obs_state: Tensor | None,
        task_tokens: Tensor | None,
        task_mask: Tensor | None,
        batch_size: int,
    ) -> tuple[Tensor | None, Tensor | None, Tensor | None, Tensor | None]:
        provided = {
            "obs_image": obs_image,
            "obs_state": obs_state,
            "task_tokens": task_tokens,
            "task_mask": task_mask,
        }
        if all(value is None for value in provided.values()):
            return None, None, None, None
        if any(value is None for value in provided.values()):
            missing = [key for key, value in provided.items() if value is None]
            raise ValueError(
                "Conditioning inputs must either all be provided or all be omitted. "
                f"Missing: {missing}."
            )

        assert obs_image is not None
        assert obs_state is not None
        assert task_tokens is not None
        assert task_mask is not None

        if obs_image.shape[0] != batch_size:
            raise ValueError(
                f"`obs_image` batch size {obs_image.shape[0]} does not match action batch {batch_size}."
            )
        if obs_state.shape[0] != batch_size:
            raise ValueError(
                f"`obs_state` batch size {obs_state.shape[0]} does not match action batch {batch_size}."
            )
        if task_tokens.shape[0] != batch_size or task_mask.shape[0] != batch_size:
            raise ValueError(
                "Text conditioning batch size must match the action batch. "
                f"Got task_tokens={task_tokens.shape[0]} task_mask={task_mask.shape[0]} batch={batch_size}."
            )

        image_embedding = self.frozen_vlm_encoder.encode_image(obs_image)
        text_embedding = self.frozen_vlm_encoder.encode_text(task_tokens, task_mask)
        state_embedding = self.state_encoder(obs_state)
        condition_embedding = self.condition_fusion(
            image_embedding=image_embedding,
            state_embedding=state_embedding,
            text_embedding=text_embedding,
        )
        return image_embedding, state_embedding, text_embedding, condition_embedding

    def _apply_condition_film(
        self, encoded_chunk: Tensor, condition_embedding: Tensor | None
    ) -> Tensor:
        if condition_embedding is None or not self.config.use_condition_film:
            return encoded_chunk

        condition_embedding = condition_embedding.to(
            device=encoded_chunk.device,
            dtype=encoded_chunk.dtype,
        )
        return self.condition_film(encoded_chunk, condition_embedding)

    def _prepare_inputs(
        self, action_chunk: Tensor, action_is_pad: Tensor | None
    ) -> tuple[Tensor, Tensor | None]:
        if action_chunk.ndim == 2:
            action_chunk = action_chunk.unsqueeze(0)
        elif action_chunk.ndim != 3:
            raise ValueError(
                "`action_chunk` must have shape [B, H, D] or [H, D], "
                f"got {tuple(action_chunk.shape)}."
            )

        if action_chunk.shape[1] != self.config.chunk_size:
            raise ValueError(
                f"Expected chunk length {self.config.chunk_size}, got {action_chunk.shape[1]}."
            )
        if action_chunk.shape[2] != self.config.action_dim:
            raise ValueError(
                f"Expected action dimension {self.config.action_dim}, got {action_chunk.shape[2]}."
            )

        prepared_mask: Tensor | None = None
        if action_is_pad is not None:
            if action_is_pad.ndim == 1:
                action_is_pad = action_is_pad.unsqueeze(0)
            elif action_is_pad.ndim == 3 and action_is_pad.shape[-1] == 1:
                action_is_pad = action_is_pad.squeeze(-1)
            elif action_is_pad.ndim != 2:
                raise ValueError(
                    "`action_is_pad` must have shape [B, H], [H], or [B, H, 1], "
                    f"got {tuple(action_is_pad.shape)}."
                )

            if action_is_pad.shape != action_chunk.shape[:2]:
                raise ValueError(
                    "`action_is_pad` must match the leading dimensions of `action_chunk`. "
                    f"Got {tuple(action_is_pad.shape)} and {tuple(action_chunk.shape[:2])}."
                )
            prepared_mask = action_is_pad.to(device=action_chunk.device, dtype=torch.bool)

        return action_chunk, prepared_mask

    def _make_valid_mask(self, action_chunk: Tensor, action_is_pad: Tensor | None) -> Tensor:
        if action_is_pad is None:
            return torch.ones(
                action_chunk.shape[0],
                action_chunk.shape[1],
                device=action_chunk.device,
                dtype=torch.bool,
            )
        return ~action_is_pad

    def _masked_l1_per_sample(self, prediction: Tensor, target: Tensor, valid_mask: Tensor) -> Tensor:
        per_element_error = torch.abs(prediction - target)
        weighted_error = per_element_error * valid_mask.unsqueeze(-1).to(dtype=prediction.dtype)
        valid_counts = valid_mask.sum(dim=1).clamp_min(1).to(dtype=prediction.dtype)
        denom = valid_counts * prediction.shape[-1]
        return weighted_error.sum(dim=(1, 2)) / denom

    def _reduce_loss(self, loss_per_sample: Tensor, reduction: str) -> Tensor:
        if reduction == "mean":
            return loss_per_sample.mean()
        if reduction == "none":
            return loss_per_sample
        raise ValueError(f"Unsupported reduction '{reduction}'. Expected 'mean' or 'none'.")

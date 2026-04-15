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

This module models the latent stage variable ``z_t`` as a chunk-level option
posterior inferred only from the future action chunk ``A_t``. The model does
not consume images or robot state.

The implementation follows a two-expert design:

1. A coarse expert reconstructs the chunk from a small set of learned anchors
   expanded by fixed piecewise-linear interpolation.
2. A residual expert predicts a learned residual that refines the coarse chunk.

The final reconstruction is mixed by a two-way posterior over ``global`` and
``local`` options. The output dataclass exposes both reconstructions and the
individual diagnostic losses needed during training.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn


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

        normalized_probs = tuple(prob / target_probs_sum for prob in self.target_probs)
        object.__setattr__(self, "target_probs", normalized_probs)


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
        """Encode an action chunk into a single chunk-level representation.

        Args:
            action_chunk: Tensor of shape ``[B, H, D]``.
            action_is_pad: Optional padding mask of shape ``[B, H]`` where
                ``True`` marks padded action steps.

        Returns:
            Tensor of shape ``[B, hidden_dim]``.
        """

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
        """Return mixture logits and probabilities.

        Args:
            encoded_chunk: Chunk representation of shape ``[B, hidden_dim]``.

        Returns:
            Tuple ``(logits, probs)`` where both tensors have shape ``[B, 2]``.
            Index 0 is ``p_global`` and index 1 is ``p_local``.
        """

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
        """Decode anchors and interpolate them into a full coarse chunk.

        Args:
            encoded_chunk: Tensor of shape ``[B, hidden_dim]``.

        Returns:
            Tuple ``(coarse_chunk, anchor_values, anchor_positions)`` with
            shapes ``[B, H, D]``, ``[B, K, D]`` and ``[B, K]``.
        """

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
        """Compute a penalty that discourages anchors from collapsing together.

        Args:
            anchor_positions: Sorted anchor positions of shape ``[B, K]``.

        Returns:
            Tuple ``(loss_per_sample, anchor_gaps)`` where ``anchor_gaps`` has
            shape ``[B, K - 1]``.
        """

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
        """Decode the residual action chunk of shape ``[B, H, D]``."""

        batch_size = encoded_chunk.shape[0]
        residual = self.proj(encoded_chunk)
        return residual.view(batch_size, self.chunk_size, self.action_dim)


class FutureChunkStageLabeler(nn.Module):
    """Infer a chunk-level option posterior from future actions only.

    The model encodes a future action chunk, predicts a two-way posterior over
    global/local experts, reconstructs a coarse chunk from sparse anchors,
    refines it with a learned residual, and optimizes the masked reconstruction
    objective described in the module docstring.
    """

    def __init__(self, config: StageLabelerConfig) -> None:
        super().__init__()
        self.config = config
        self.encoder = ChunkEncoder1D(config)
        self.mixture_head = MixtureHead(config)
        self.coarse_decoder = CoarseAnchorDecoder(config)
        self.residual_decoder = ResidualDecoder(config)
        self.register_buffer(
            "target_probs",
            torch.tensor(config.target_probs, dtype=torch.float32),
            persistent=False,
        )

    def forward(
        self,
        action_chunk: torch.Tensor,
        action_is_pad: torch.Tensor | None = None,
        reduction: str = "mean",
    ) -> StageLabelerOutput:
        """Run the stage labeler and compute the training losses.

        Args:
            action_chunk: Future action chunk of shape ``[B, H, D]`` or
                ``[H, D]``.
            action_is_pad: Optional padding mask of shape ``[B, H]`` or ``[H]``
                where ``True`` marks padded time steps.
            reduction: Either ``"mean"`` or ``"none"``.

        Returns:
            A :class:`StageLabelerOutput` containing reconstructions, posterior
            probabilities, anchor diagnostics, and all loss terms.
        """

        action_chunk, action_is_pad = self._prepare_inputs(action_chunk, action_is_pad)

        encoded_chunk = self.encoder(action_chunk, action_is_pad=action_is_pad)
        mixture_logits, stage_probs = self.mixture_head(encoded_chunk)
        coarse_chunk, anchor_values, anchor_positions = self.coarse_decoder(encoded_chunk)
        residual_chunk = self.residual_decoder(encoded_chunk)
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
        )

    def predict_stage_probs(
        self, action_chunk: torch.Tensor, action_is_pad: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Predict stage posterior probabilities without computing losses.

        Args:
            action_chunk: Future action chunk of shape ``[B, H, D]`` or
                ``[H, D]``.
            action_is_pad: Optional padding mask of shape ``[B, H]`` or ``[H]``.

        Returns:
            Mixture probabilities of shape ``[B, 2]`` with index 0 equal to
            ``p_global`` and index 1 equal to ``p_local``.
        """

        action_chunk, action_is_pad = self._prepare_inputs(action_chunk, action_is_pad)
        encoded_chunk = self.encoder(action_chunk, action_is_pad=action_is_pad)
        _, stage_probs = self.mixture_head(encoded_chunk)
        return stage_probs

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

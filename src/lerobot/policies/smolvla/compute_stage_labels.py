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

"""Compute chunk-level stage labels for SmolVLA multimodal future chunks.

This script trains or loads a :class:`FutureChunkStageLabeler`, then runs it
across a full :class:`LeRobotDataset` to generate a
``smolvla_stage_labels.parquet`` file.

Each stage-label batch now includes:

1. future action chunks,
2. a current image observation ``obs_image``,
3. robot state ``obs_state``,
4. tokenized task text.
"""

from __future__ import annotations

import argparse
import logging
import random
from collections import defaultdict
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from lerobot.datasets import LeRobotDataset
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

try:
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
except ModuleNotFoundError:
    from configuration_smolvla import SmolVLAConfig

try:
    from lerobot.policies.smolvla.stage_labeler import (
        FutureChunkStageLabeler,
        StageLabelerConfig,
    )
except ModuleNotFoundError:
    from stage_labeler import FutureChunkStageLabeler, StageLabelerConfig


GENERATOR_VERSION = "smolvla_future_chunk_stage_labeler_multimodal_film_v1"
DEFAULT_VLM_MODEL_NAME = SmolVLAConfig.vlm_model_name
DEFAULT_TOKENIZER_MAX_LENGTH = SmolVLAConfig.tokenizer_max_length
DEFAULT_IMAGE_RESIZE_WITH_PADDING = SmolVLAConfig.resize_imgs_with_padding
DEFAULT_CONDITION_HIDDEN_DIM = StageLabelerConfig.__dataclass_fields__["condition_hidden_dim"].default
DEFAULT_STATE_HIDDEN_DIM = StageLabelerConfig.__dataclass_fields__["state_hidden_dim"].default
DEFAULT_IMAGE_PROJ_DIM = StageLabelerConfig.__dataclass_fields__["image_proj_dim"].default
DEFAULT_TEXT_PROJ_DIM = StageLabelerConfig.__dataclass_fields__["text_proj_dim"].default
DEFAULT_FILM_HIDDEN_DIM = StageLabelerConfig.__dataclass_fields__["film_hidden_dim"].default
DEFAULT_USE_CONDITION_FILM = StageLabelerConfig.__dataclass_fields__["use_condition_film"].default
DEFAULT_FREEZE_CONDITION_ENCODERS = StageLabelerConfig.__dataclass_fields__[
    "freeze_condition_encoders"
].default


def _resize_with_pad(img: Tensor, width: int, height: int, pad_value: float = 0.0) -> Tensor:
    """Resize images with aspect-ratio-preserving padding, matching SmolVLA preprocessing."""

    if img.ndim != 4:
        raise ValueError(f"(b, c, h, w) expected, but got {tuple(img.shape)}.")

    cur_height, cur_width = img.shape[2:]
    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)

    resized_img = F.interpolate(img, size=(resized_height, resized_width), mode="bilinear", align_corners=False)
    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))
    return F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)


@lru_cache(maxsize=4)
def _get_tokenizer(tokenizer_name: str):
    if AutoTokenizer is None:
        raise ImportError(
            "The 'transformers' library is required to tokenize task instructions. "
            "Install it with `uv sync --locked --extra all` or another environment that includes transformers."
        )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


class _MultimodalStageLabelDataset(Dataset[dict[str, Tensor]]):
    """Future-action-chunk view over :class:`LeRobotDataset`.

    The wrapper keeps the existing bulk future-action-chunk logic while also
    exposing the current image, state, and task text consumed by the current
    multimodal stage labeler. When auto-collation is enabled, it uses batched column
    reads for vector features and grouped video decoding for the selected image
    key.

    When auto-collation is enabled, :class:`torch.utils.data.DataLoader` can
    call :meth:`__getitems__` with a full batch of indices. This lets us fetch
    base columns and action rows in bulk instead of issuing dozens of Python-
    level random accesses per batch.
    """

    def __init__(
        self,
        dataset: LeRobotDataset,
        image_feature_key: str | None = None,
        state_feature_key: str = OBS_STATE,
        task_key: str = "task_text",
        include_multimodal_inputs: bool = False,
    ) -> None:
        self.dataset = dataset
        self.reader = dataset._ensure_reader()
        if self.reader.hf_dataset is None:
            self.reader.load_and_activate()

        self.state_feature_key = state_feature_key
        self.task_key = task_key
        self.include_multimodal_inputs = include_multimodal_inputs
        self.image_feature_key = image_feature_key or self._infer_default_image_key()

        if ACTION not in dataset.meta.features:
            raise KeyError(f"Dataset '{dataset.repo_id}' does not contain the '{ACTION}' feature.")

        if self.image_feature_key is not None and self.image_feature_key not in dataset.meta.camera_keys:
            raise KeyError(
                f"Requested image feature '{self.image_feature_key}' is not one of the dataset camera keys "
                f"{dataset.meta.camera_keys}."
            )
        if self.include_multimodal_inputs:
            if self.image_feature_key is None:
                raise KeyError(
                    f"Dataset '{dataset.repo_id}' does not expose any camera features, so no image conditioning "
                    "input can be produced."
                )
            if self.state_feature_key not in dataset.meta.features:
                raise KeyError(
                    f"Dataset '{dataset.repo_id}' does not contain the state feature '{self.state_feature_key}'."
                )
            if dataset.meta.tasks is None:
                raise KeyError(f"Dataset '{dataset.repo_id}' does not provide a task table.")

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        return self._get_single_item(idx)

    def __getitems__(self, indices: list[int]) -> list[dict[str, Tensor]]:
        if len(indices) == 0:
            return []

        hf_dataset = self._get_hf_dataset()
        batch_indices = [int(idx) for idx in indices]
        episode_indices = self._to_batched_tensor(hf_dataset["episode_index"][batch_indices])
        absolute_indices = self._to_batched_tensor(hf_dataset["index"][batch_indices])
        items = [
            {
                "episode_index": episode_indices[row_idx],
                "index": absolute_indices[row_idx],
            }
            for row_idx in range(len(batch_indices))
        ]

        if self.include_multimodal_inputs:
            state_batch = self._get_current_feature_batch(self.state_feature_key, batch_indices)
            image_batch = self._get_image_batch(batch_indices, episode_indices)
            task_values = self._get_task_text_batch(batch_indices)

            for row_idx, item in enumerate(items):
                item[self.state_feature_key] = state_batch[row_idx]
                item["obs_image"] = image_batch[row_idx]
                item[self.task_key] = task_values[row_idx]

        if self.reader.delta_indices is None:
            action_batch = self._to_batched_tensor(hf_dataset[ACTION][batch_indices])
            for row_idx, item in enumerate(items):
                item[ACTION] = action_batch[row_idx]
            return items

        if ACTION not in self.reader.delta_indices:
            raise KeyError(f"Delta indices are missing the '{ACTION}' key required for stage labeling.")

        action_pad_key = f"{ACTION}_is_pad"
        chunk_size = len(self.reader.delta_indices[ACTION])
        batched_relative_indices: list[int] = []
        batched_paddings: list[Tensor | None] = []

        for row_idx in range(len(batch_indices)):
            ep_idx = int(episode_indices[row_idx].item())
            abs_idx = int(absolute_indices[row_idx].item())
            query_indices, padding = self.reader._get_query_indices(abs_idx, ep_idx)
            batched_relative_indices.extend(self._to_relative_indices(query_indices[ACTION]))
            batched_paddings.append(padding.get(action_pad_key))

        action_batch = self._get_action_chunk_batch(
            relative_indices=batched_relative_indices,
            batch_size=len(batch_indices),
            chunk_size=chunk_size,
        )

        for row_idx, item in enumerate(items):
            item[ACTION] = action_batch[row_idx]
            action_is_pad = batched_paddings[row_idx]
            if action_is_pad is not None:
                item[action_pad_key] = action_is_pad

        return items

    def _get_single_item(self, idx: int) -> dict[str, Tensor]:
        episode_index = self._get_scalar_column("episode_index", idx)
        absolute_index = self._get_scalar_column("index", idx)

        item: dict[str, Tensor] = {
            "episode_index": episode_index,
            "index": absolute_index,
        }

        if self.include_multimodal_inputs:
            item[self.state_feature_key] = self._get_current_feature(self.state_feature_key, idx)
            item["obs_image"] = self._get_image(idx, int(episode_index.item()))
            item[self.task_key] = self._get_task_text(idx)

        if self.reader.delta_indices is None:
            item[ACTION] = self._get_action_row(idx)
            return item

        ep_idx = int(episode_index.item())
        abs_idx = int(absolute_index.item())
        query_indices, padding = self.reader._get_query_indices(abs_idx, ep_idx)
        action_indices = query_indices[ACTION]
        relative_indices = self._to_relative_indices(action_indices)
        item[ACTION] = self._get_action_chunk(relative_indices)

        action_pad_key = f"{ACTION}_is_pad"
        if action_pad_key in padding:
            item[action_pad_key] = padding[action_pad_key]

        return item

    def _get_hf_dataset(self):
        hf_dataset = self.reader.hf_dataset
        if hf_dataset is None:
            raise RuntimeError("The underlying HF dataset is not loaded.")
        return hf_dataset

    def _get_scalar_column(self, key: str, idx: int) -> Tensor:
        hf_dataset = self._get_hf_dataset()
        value = hf_dataset[key][idx]
        if isinstance(value, Tensor):
            return value
        return torch.as_tensor(value)

    def _get_column_batch(self, key: str, indices: list[int]) -> Any:
        hf_dataset = self._get_hf_dataset()
        try:
            return hf_dataset[key][indices]
        except (KeyError, TypeError, IndexError):
            return hf_dataset[indices][key]

    def _get_action_row(self, idx: int) -> Tensor:
        hf_dataset = self._get_hf_dataset()
        action_value = hf_dataset[ACTION][idx]
        if isinstance(action_value, Tensor):
            return action_value
        return torch.as_tensor(action_value)

    def _get_current_feature(self, key: str, idx: int) -> Tensor:
        value = self._get_column_batch(key, [idx])
        return self._to_batched_tensor(value)[0]

    def _get_current_feature_batch(self, key: str, indices: list[int]) -> Tensor:
        return self._to_batched_tensor(self._get_column_batch(key, indices))

    def _get_action_chunk(self, relative_indices: list[int]) -> Tensor:
        hf_dataset = self._get_hf_dataset()
        try:
            action_rows = hf_dataset[ACTION][relative_indices]
        except (TypeError, KeyError, IndexError):
            action_rows = hf_dataset[relative_indices][ACTION]

        return self._to_batched_tensor(action_rows)

    def _get_action_chunk_batch(
        self,
        relative_indices: list[int],
        batch_size: int,
        chunk_size: int,
    ) -> Tensor:
        action_rows = self._get_action_chunk(relative_indices)
        expected_rows = batch_size * chunk_size
        if action_rows.shape[0] != expected_rows:
            raise RuntimeError(
                "Fetched action rows do not match the expected batch shape: "
                f"got {action_rows.shape[0]} rows, expected {expected_rows}."
            )
        return action_rows.reshape(batch_size, chunk_size, *action_rows.shape[1:])

    def _to_batched_tensor(self, value: Any) -> Tensor:
        if isinstance(value, Tensor):
            return value
        if isinstance(value, list) and len(value) > 0 and isinstance(value[0], Tensor):
            return torch.stack(value)
        return torch.as_tensor(value)

    def _to_relative_indices(self, absolute_indices: list[int]) -> list[int]:
        if self.reader._absolute_to_relative_idx is None:
            return absolute_indices
        return [self.reader._absolute_to_relative_idx[idx] for idx in absolute_indices]

    def _infer_default_image_key(self) -> str | None:
        if len(self.dataset.meta.camera_keys) == 0:
            return None
        return self.dataset.meta.camera_keys[0]

    def _get_task_text(self, idx: int) -> str:
        return self._get_task_text_batch([idx])[0]

    def _get_task_text_batch(self, indices: list[int]) -> list[str]:
        task_indices = self._to_batched_tensor(self._get_column_batch("task_index", indices)).view(-1)
        if self.dataset.meta.tasks is None:
            raise KeyError(f"Dataset '{self.dataset.repo_id}' does not provide a task table.")
        return [self.dataset.meta.tasks.iloc[int(task_idx.item())].name for task_idx in task_indices]

    def _get_image(self, idx: int, ep_idx: int) -> Tensor:
        image_batch = self._get_image_batch([idx], torch.tensor([ep_idx], dtype=torch.int64))
        return image_batch[0]

    def _get_image_batch(self, indices: list[int], episode_indices: Tensor) -> Tensor:
        if self.image_feature_key is None:
            raise KeyError("No image feature key is configured for multimodal stage-label batches.")

        if self.image_feature_key not in self.dataset.meta.video_keys:
            image_values = self._get_column_batch(self.image_feature_key, indices)
            image_list = self._to_list_of_tensors(image_values, expected_batch_size=len(indices))
            return torch.stack([self._apply_image_transforms(img) for img in image_list])

        timestamps = self._to_batched_tensor(self._get_column_batch("timestamp", indices)).view(-1)
        row_indices_by_episode: dict[int, list[int]] = defaultdict(list)
        for row_idx, episode_index in enumerate(episode_indices.view(-1)):
            row_indices_by_episode[int(episode_index.item())].append(row_idx)

        images: list[Tensor | None] = [None] * len(indices)
        for ep_idx, row_indices in row_indices_by_episode.items():
            query_timestamps = {
                self.image_feature_key: [float(timestamps[row_idx].item()) for row_idx in row_indices]
            }
            decoded = self.reader._query_videos(query_timestamps, ep_idx)[self.image_feature_key]
            if decoded.ndim == 3:
                decoded = decoded.unsqueeze(0)

            for local_idx, row_idx in enumerate(row_indices):
                images[row_idx] = self._apply_image_transforms(decoded[local_idx])

        if any(img is None for img in images):
            raise RuntimeError("Failed to decode every requested image in the multimodal stage-label batch.")

        return torch.stack([img for img in images if img is not None])

    def _apply_image_transforms(self, image: Tensor) -> Tensor:
        if self.reader._image_transforms is None:
            return image
        return self.reader._image_transforms(image)

    def _to_list_of_tensors(self, value: Any, expected_batch_size: int | None = None) -> list[Tensor]:
        if isinstance(value, Tensor):
            if value.ndim == 0:
                return [value]
            if expected_batch_size == 1 and value.ndim >= 1:
                return [value[0] if value.shape[0] == 1 else value]
            return [value[i] for i in range(value.shape[0])]
        if isinstance(value, list):
            result = []
            for element in value:
                if isinstance(element, Tensor):
                    result.append(element)
                else:
                    result.append(torch.as_tensor(element))
            return result
        tensor = torch.as_tensor(value)
        if tensor.ndim == 0:
            return [tensor]
        if expected_batch_size == 1 and tensor.ndim >= 1:
            return [tensor[0] if tensor.shape[0] == 1 else tensor]
        return [tensor[i] for i in range(tensor.shape[0])]


class _StageLabelBatchCollator:
    """Collate multimodal stage-label samples into model-ready tensors."""

    def __init__(
        self,
        tokenizer_name: str,
        tokenizer_max_length: int,
        image_resize_with_padding: tuple[int, int] | None = DEFAULT_IMAGE_RESIZE_WITH_PADDING,
        state_feature_key: str = OBS_STATE,
        task_key: str = "task_text",
    ) -> None:
        self.tokenizer_name = tokenizer_name
        self.tokenizer_max_length = tokenizer_max_length
        self.image_resize_with_padding = image_resize_with_padding
        self.state_feature_key = state_feature_key
        self.task_key = task_key

    def __call__(self, items: list[dict[str, Any]]) -> dict[str, Any]:
        if len(items) == 0:
            raise ValueError("Cannot collate an empty batch.")

        batch: dict[str, Any] = {
            "index": torch.stack([self._to_tensor(item["index"], torch.int64) for item in items]),
            "episode_index": torch.stack(
                [self._to_tensor(item["episode_index"], torch.int64) for item in items]
            ),
            ACTION: torch.stack([self._to_tensor(item[ACTION], torch.float32) for item in items]),
            self.state_feature_key: torch.stack(
                [self._to_tensor(item[self.state_feature_key], torch.float32) for item in items]
            ),
            "obs_image": torch.stack([self._preprocess_image(item["obs_image"]) for item in items]),
        }

        action_pad_key = f"{ACTION}_is_pad"
        if action_pad_key in items[0]:
            batch[action_pad_key] = torch.stack(
                [self._to_tensor(item[action_pad_key], torch.bool) for item in items]
            )

        task_texts = [self._normalize_task_text(item[self.task_key]) for item in items]
        batch[self.task_key] = task_texts

        tokenizer = _get_tokenizer(self.tokenizer_name)
        tokenizer.padding_side = "right"
        tokenized = tokenizer(
            task_texts,
            max_length=self.tokenizer_max_length,
            truncation=True,
            padding="longest",
            return_tensors="pt",
        )
        batch[OBS_LANGUAGE_TOKENS] = tokenized["input_ids"]
        batch[OBS_LANGUAGE_ATTENTION_MASK] = tokenized["attention_mask"].to(dtype=torch.bool)
        return batch

    def _normalize_task_text(self, task_text: Any) -> str:
        if not isinstance(task_text, str):
            raise TypeError(f"Expected task text to be a string, but got {type(task_text).__name__}.")
        return task_text if task_text.endswith("\n") else f"{task_text}\n"

    def _preprocess_image(self, image: Any) -> Tensor:
        if not isinstance(image, Tensor):
            image = torch.as_tensor(image)

        if image.ndim != 3:
            raise ValueError(f"Expected a single image tensor with shape [C, H, W] or [H, W, C], got {image.shape}.")

        if image.shape[0] not in {1, 3} and image.shape[-1] in {1, 3}:
            image = image.permute(2, 0, 1)
        elif image.shape[0] not in {1, 3}:
            raise ValueError(f"Unsupported image shape for preprocessing: {tuple(image.shape)}.")

        if image.dtype == torch.uint8:
            image = image.to(dtype=torch.float32) / 255.0
        else:
            image = image.to(dtype=torch.float32)
        image = image.clamp(0.0, 1.0)

        if self.image_resize_with_padding is not None:
            image = _resize_with_pad(image.unsqueeze(0), *self.image_resize_with_padding, pad_value=0.0).squeeze(0)

        return image * 2.0 - 1.0

    def _to_tensor(self, value: Any, dtype: torch.dtype) -> Tensor:
        if isinstance(value, Tensor):
            return value.to(dtype=dtype)
        return torch.as_tensor(value, dtype=dtype)


def _extract_conditioning_inputs(batch: dict[str, Any]) -> dict[str, Any]:
    return {
        "obs_image": batch["obs_image"],
        OBS_STATE: batch[OBS_STATE],
        OBS_LANGUAGE_TOKENS: batch[OBS_LANGUAGE_TOKENS],
        OBS_LANGUAGE_ATTENTION_MASK: batch[OBS_LANGUAGE_ATTENTION_MASK],
        "task_text": batch["task_text"],
    }


def _validate_conditioning_inputs(
    conditioning_inputs: dict[str, Any],
    batch_size: int,
    context: str,
    image_feature_key: str | None,
    log_once: bool = False,
) -> None:
    missing_keys = [
        key
        for key in (
            "obs_image",
            OBS_STATE,
            OBS_LANGUAGE_TOKENS,
            OBS_LANGUAGE_ATTENTION_MASK,
            "task_text",
        )
        if key not in conditioning_inputs
    ]
    if missing_keys:
        raise KeyError(f"{context} batch is missing multimodal conditioning keys: {missing_keys}.")

    obs_image = conditioning_inputs["obs_image"]
    state = conditioning_inputs[OBS_STATE]
    lang_tokens = conditioning_inputs[OBS_LANGUAGE_TOKENS]
    lang_masks = conditioning_inputs[OBS_LANGUAGE_ATTENTION_MASK]
    task_text = conditioning_inputs["task_text"]

    if not isinstance(obs_image, Tensor):
        raise TypeError(f"{context} expected 'obs_image' to be a tensor, got {type(obs_image).__name__}.")
    if not isinstance(state, Tensor):
        raise TypeError(f"{context} expected '{OBS_STATE}' to be a tensor, got {type(state).__name__}.")
    if not isinstance(lang_tokens, Tensor):
        raise TypeError(
            f"{context} expected '{OBS_LANGUAGE_TOKENS}' to be a tensor, got {type(lang_tokens).__name__}."
        )
    if not isinstance(lang_masks, Tensor):
        raise TypeError(
            f"{context} expected '{OBS_LANGUAGE_ATTENTION_MASK}' to be a tensor, got {type(lang_masks).__name__}."
        )
    if not isinstance(task_text, list) or not all(isinstance(item, str) for item in task_text):
        raise TypeError(f"{context} expected 'task_text' to be a list of strings.")

    if obs_image.shape[0] != batch_size:
        raise ValueError(f"{context} obs_image batch size {obs_image.shape[0]} does not match action batch {batch_size}.")
    if state.shape[0] != batch_size:
        raise ValueError(f"{context} state batch size {state.shape[0]} does not match action batch {batch_size}.")
    if lang_tokens.shape[0] != batch_size:
        raise ValueError(
            f"{context} token batch size {lang_tokens.shape[0]} does not match action batch {batch_size}."
        )
    if lang_tokens.shape != lang_masks.shape:
        raise ValueError(
            f"{context} language token shape {tuple(lang_tokens.shape)} does not match mask shape "
            f"{tuple(lang_masks.shape)}."
        )
    if len(task_text) != batch_size:
        raise ValueError(
            f"{context} task_text batch length {len(task_text)} does not match action batch {batch_size}."
        )

    if log_once:
        logging.info(
            "%s multimodal batch ready: image_feature_key=%s obs_image=%s %s=%s %s=%s %s=%s task_text_example=%r",
            context,
            image_feature_key,
            tuple(obs_image.shape),
            OBS_STATE,
            tuple(state.shape),
            OBS_LANGUAGE_TOKENS,
            tuple(lang_tokens.shape),
            OBS_LANGUAGE_ATTENTION_MASK,
            tuple(lang_masks.shape),
            task_text[0] if len(task_text) > 0 else "",
        )


def build_argparser() -> argparse.ArgumentParser:
    """Build the CLI argument parser for stage-label generation."""

    parser = argparse.ArgumentParser(
        description="Train or load a SmolVLA FutureChunkStageLabeler and export stage labels to parquet."
    )
    parser.add_argument(
        "--dataset-repo-id",
        type=str,
        required=True,
        help="Dataset repo id, for example 'lerobot/aloha_sim_insertion_human'.",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Optional local dataset root. If omitted, LeRobot cache resolution is used.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional dataset revision (branch, tag, or commit).",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=None,
        help="Output parquet path. Defaults to '<dataset_root>/smolvla_stage_labels.parquet'.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=str,
        default=None,
        help="Checkpoint save path. Defaults to '<dataset_root>/smolvla_stage_labeler_checkpoint.pt'.",
    )
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Optional checkpoint to load before training or inference.",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training and use a loaded checkpoint for stage-label generation.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=50,
        help="Future action chunk size used to build delta_timestamps.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size used for both training and stage-label generation.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of dataloader workers. Set above 0 to enable background prefetch.",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="Number of prefetched batches per worker. Ignored when --num-workers=0.",
    )
    parser.add_argument(
        "--image-feature-key",
        type=str,
        default=None,
        help=(
            "Single camera feature key to sample for multimodal stage-label batches. "
            "If omitted, the first available dataset camera key is used."
        ),
    )
    parser.add_argument(
        "--tokenizer-max-length",
        type=int,
        default=DEFAULT_TOKENIZER_MAX_LENGTH,
        help="Maximum tokenized task length used when collating multimodal stage-label batches.",
    )
    parser.add_argument(
        "--vlm-model-name",
        type=str,
        default=DEFAULT_VLM_MODEL_NAME,
        help="SmolVLM checkpoint name shared by the tokenizer and frozen multimodal encoders.",
    )
    parser.add_argument(
        "--train-steps",
        type=int,
        default=2_000,
        help="Number of optimization steps for stage-labeler training.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="AdamW learning rate.",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="AdamW weight decay.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm.",
    )
    parser.add_argument(
        "--log-freq",
        type=int,
        default=50,
        help="Log training metrics every N steps.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: 'auto', 'cpu', 'cuda', or an explicit device like 'cuda:0'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    parser.add_argument(
        "--encoder-hidden-dim",
        type=int,
        default=256,
        help="Hidden width used by the temporal encoder and decoder heads.",
    )
    parser.add_argument(
        "--encoder-num-layers",
        type=int,
        default=3,
        help="Number of temporal convolution layers in the action encoder.",
    )
    parser.add_argument(
        "--encoder-kernel-size",
        type=int,
        default=5,
        help="Odd kernel size used by the temporal convolution encoder.",
    )
    parser.add_argument(
        "--encoder-dropout",
        type=float,
        default=0.1,
        help="Dropout used in the encoder and decoder heads.",
    )
    parser.add_argument(
        "--num-anchors",
        type=int,
        default=4,
        help="Number of coarse anchors for piecewise-linear interpolation.",
    )
    parser.add_argument(
        "--residual-l1-weight",
        type=float,
        default=0.05,
        help="Weight for the residual sparsity loss.",
    )
    parser.add_argument(
        "--entropy-weight",
        type=float,
        default=0.01,
        help="Weight for the posterior entropy term.",
    )
    parser.add_argument(
        "--balance-weight",
        type=float,
        default=0.1,
        help="Weight for the batch-level mixture balancing loss.",
    )
    parser.add_argument(
        "--spacing-weight",
        type=float,
        default=0.1,
        help="Weight for the anchor spacing regularizer.",
    )
    parser.add_argument(
        "--target-probs",
        type=float,
        nargs=2,
        default=(0.5, 0.5),
        metavar=("P_GLOBAL", "P_LOCAL"),
        help="Target batch-average mixture probabilities for global and local experts.",
    )
    parser.add_argument(
        "--min-anchor-spacing-ratio",
        type=float,
        default=0.5,
        help="Minimum anchor spacing as a fraction of the uniform anchor gap.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-8,
        help="Numerical epsilon used in logs and divisions.",
    )
    parser.add_argument(
        "--condition-hidden-dim",
        type=int,
        default=DEFAULT_CONDITION_HIDDEN_DIM,
        help="Output width of the fused multimodal condition embedding.",
    )
    parser.add_argument(
        "--state-hidden-dim",
        type=int,
        default=DEFAULT_STATE_HIDDEN_DIM,
        help="Hidden width of the robot-state encoder MLP.",
    )
    parser.add_argument(
        "--image-proj-dim",
        type=int,
        default=DEFAULT_IMAGE_PROJ_DIM,
        help="Optional projection width for pooled image features. Defaults to condition_hidden_dim.",
    )
    parser.add_argument(
        "--text-proj-dim",
        type=int,
        default=DEFAULT_TEXT_PROJ_DIM,
        help="Optional projection width for pooled text features. Defaults to condition_hidden_dim.",
    )
    parser.add_argument(
        "--film-hidden-dim",
        type=int,
        default=DEFAULT_FILM_HIDDEN_DIM,
        help="Hidden width of the FiLM MLP applied after the pooled action encoder.",
    )
    parser.add_argument(
        "--use-condition-film",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_CONDITION_FILM,
        help="Enable FiLM modulation between the pooled action embedding and downstream heads.",
    )
    parser.add_argument(
        "--freeze-condition-encoders",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FREEZE_CONDITION_ENCODERS,
        help="Keep the SmolVLM image/text encoders frozen and in eval mode.",
    )
    return parser


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested_device: str) -> torch.device:
    """Resolve a CLI device string to a valid :class:`torch.device`."""

    normalized = requested_device.strip().lower()
    if normalized == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available.")
    if device.type == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise ValueError("MPS was requested but is not available.")
    return device


def build_action_chunk_dataset(
    dataset_repo_id: str,
    root: str | Path | None,
    revision: str | None,
    chunk_size: int,
) -> LeRobotDataset:
    """Build a future-action-chunk dataset aligned with SmolVLA training.

    The function first loads a lightweight temporary dataset to read ``fps``.
    It then constructs the action ``delta_timestamps`` required for chunked
    future actions and re-instantiates :class:`LeRobotDataset`. Video assets are
    downloaded only when the dataset stores camera observations as videos so
    that later multimodal wrappers can fetch image observations.
    """

    logging.info("Loading dataset metadata to resolve fps: repo_id=%s", dataset_repo_id)
    temp_dataset = LeRobotDataset(
        repo_id=dataset_repo_id,
        root=root,
        revision=revision,
        download_videos=False,
    )
    fps = temp_dataset.fps
    delta_timestamps = {ACTION: [i / fps for i in range(chunk_size)]}
    download_videos = len(temp_dataset.meta.video_keys) > 0

    logging.info(
        "Building stage-label dataset with fps=%s, chunk_size=%s, camera_keys=%s, download_videos=%s",
        fps,
        chunk_size,
        temp_dataset.meta.camera_keys,
        download_videos,
    )
    dataset = LeRobotDataset(
        repo_id=dataset_repo_id,
        root=root,
        revision=revision,
        delta_timestamps=delta_timestamps,
        download_videos=download_videos,
    )
    logging.info(
        "Dataset ready: root=%s, revision=%s, episodes=%s, frames=%s",
        dataset.root,
        dataset.revision,
        dataset.num_episodes,
        dataset.num_frames,
    )
    return dataset


def infer_action_dim(dataset: LeRobotDataset) -> int:
    """Infer action dimensionality from dataset metadata or a sample item."""

    if ACTION in dataset.meta.shapes:
        shape = dataset.meta.shapes[ACTION]
        if len(shape) == 0:
            raise ValueError(f"Action feature '{ACTION}' has an empty shape in dataset metadata.")
        return int(shape[-1])

    stage_label_dataset = _MultimodalStageLabelDataset(dataset)
    sample = stage_label_dataset[0]
    action_tensor = sample[ACTION]
    if action_tensor.ndim == 1:
        return int(action_tensor.shape[0])
    if action_tensor.ndim >= 2:
        return int(action_tensor.shape[-1])
    raise ValueError(f"Could not infer action_dim from action tensor shape {tuple(action_tensor.shape)}.")


def move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor entries in a batch dictionary to the target device."""

    moved_batch: dict[str, Any] = {}
    non_blocking = device.type == "cuda"
    for key, value in batch.items():
        if isinstance(value, Tensor):
            moved_batch[key] = value.to(device=device, non_blocking=non_blocking)
        else:
            moved_batch[key] = value
    return moved_batch


def create_dataloader(
    dataset: LeRobotDataset | Dataset[dict[str, Tensor]],
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    shuffle: bool,
    device: torch.device,
    image_feature_key: str | None,
    tokenizer_name: str,
    tokenizer_max_length: int,
) -> DataLoader[dict[str, Any]]:
    """Create a dataloader for multimodal stage-label batches."""

    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}.")
    if prefetch_factor <= 0:
        raise ValueError(f"prefetch_factor must be > 0, got {prefetch_factor}.")
    if device.type == "cuda" and num_workers == 0:
        logging.info(
            "DataLoader is running with num_workers=0, so CPU reads stay on the main process. "
            "Increase --num-workers to overlap dataset prefetch with GPU compute."
        )

    stage_label_dataset: Dataset[dict[str, Tensor]]
    if isinstance(dataset, LeRobotDataset):
        stage_label_dataset = _MultimodalStageLabelDataset(
            dataset,
            image_feature_key=image_feature_key,
            include_multimodal_inputs=True,
        )
        logging.info(
            "Using multimodal stage-label dataset wrapper with image_feature_key=%s, state_feature_key=%s.",
            stage_label_dataset.image_feature_key,
            stage_label_dataset.state_feature_key,
        )
    else:
        stage_label_dataset = dataset

    dataloader_kwargs: dict[str, Any] = {
        "dataset": stage_label_dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
        "collate_fn": _StageLabelBatchCollator(
            tokenizer_name=tokenizer_name,
            tokenizer_max_length=tokenizer_max_length,
        ),
    }
    if num_workers > 0:
        dataloader_kwargs["prefetch_factor"] = prefetch_factor

    return DataLoader(
        **dataloader_kwargs,
    )


def train_stage_labeler(
    model: FutureChunkStageLabeler,
    dataset: LeRobotDataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    train_steps: int,
    learning_rate: float,
    weight_decay: float,
    grad_clip_norm: float,
    log_freq: int,
    image_feature_key: str | None,
    tokenizer_name: str,
    tokenizer_max_length: int,
) -> int:
    """Train the stage labeler with AdamW and gradient clipping."""

    if train_steps <= 0:
        logging.info("Skipping stage-labeler training because train_steps=%s.", train_steps)
        return 0

    dataloader = create_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        shuffle=True,
        device=device,
        image_feature_key=image_feature_key,
        tokenizer_name=tokenizer_name,
        tokenizer_max_length=tokenizer_max_length,
    )
    resolved_image_feature_key = getattr(dataloader.dataset, "image_feature_key", image_feature_key)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    model.to(device)
    model.train()

    running_metrics = {
        "loss": 0.0,
        "recon_l1": 0.0,
        "residual_l1": 0.0,
        "entropy": 0.0,
        "balance_loss": 0.0,
        "spacing_loss": 0.0,
    }
    steps_since_log = 0
    data_iter = iter(dataloader)
    logged_multimodal_debug = False

    progress_bar = tqdm(total=train_steps, desc="Training stage labeler")
    for step in range(1, train_steps + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = move_batch_to_device(batch, device)
        action_chunk = batch[ACTION].to(dtype=torch.float32)
        action_is_pad = batch.get("action_is_pad")
        batch_indices = batch["index"]
        batch_episode_indices = batch["episode_index"]
        conditioning_inputs = _extract_conditioning_inputs(batch)
        _validate_conditioning_inputs(
            conditioning_inputs=conditioning_inputs,
            batch_size=action_chunk.shape[0],
            context="train",
            image_feature_key=resolved_image_feature_key,
            log_once=not logged_multimodal_debug,
        )
        logged_multimodal_debug = True
        obs_image = conditioning_inputs["obs_image"]
        obs_state = conditioning_inputs[OBS_STATE]
        task_tokens = conditioning_inputs[OBS_LANGUAGE_TOKENS]
        task_mask = conditioning_inputs[OBS_LANGUAGE_ATTENTION_MASK]

        optimizer.zero_grad(set_to_none=True)
        output = model(
            action_chunk=action_chunk,
            obs_image=obs_image,
            obs_state=obs_state,
            task_tokens=task_tokens,
            task_mask=task_mask,
            action_is_pad=action_is_pad,
            reduction="mean",
        )
        output.loss.backward()
        grad_norm = float(clip_grad_norm_(model.parameters(), grad_clip_norm))
        optimizer.step()

        running_metrics["loss"] += float(output.loss.item())
        running_metrics["recon_l1"] += float(output.recon_l1.item())
        running_metrics["residual_l1"] += float(output.residual_l1.item())
        running_metrics["entropy"] += float(output.entropy.item())
        running_metrics["balance_loss"] += float(output.balance_loss.item())
        running_metrics["spacing_loss"] += float(output.spacing_loss.item())
        steps_since_log += 1

        progress_bar.update(1)
        progress_bar.set_postfix(loss=f"{output.loss.item():.4f}", grad=f"{grad_norm:.3f}")

        should_log = step == 1 or step % log_freq == 0 or step == train_steps
        if should_log:
            index_values = batch_indices.view(-1).detach().cpu().tolist()
            episode_values = batch_episode_indices.view(-1).detach().cpu().tolist()
            logging.info(
                "step=%s/%s loss=%.6f recon=%.6f residual=%.6f entropy=%.6f balance=%.6f "
                "spacing=%.6f grad_norm=%.3f lr=%.2e batch_index=[%s,%s] batch_episode=[%s,%s]",
                step,
                train_steps,
                running_metrics["loss"] / steps_since_log,
                running_metrics["recon_l1"] / steps_since_log,
                running_metrics["residual_l1"] / steps_since_log,
                running_metrics["entropy"] / steps_since_log,
                running_metrics["balance_loss"] / steps_since_log,
                running_metrics["spacing_loss"] / steps_since_log,
                grad_norm,
                optimizer.param_groups[0]["lr"],
                index_values[0],
                index_values[-1],
                episode_values[0],
                episode_values[-1],
            )
            running_metrics = {key: 0.0 for key in running_metrics}
            steps_since_log = 0

    progress_bar.close()
    return train_steps


def generate_stage_label_dataframe(
    model: FutureChunkStageLabeler,
    dataset: LeRobotDataset,
    batch_size: int,
    num_workers: int,
    prefetch_factor: int,
    device: torch.device,
    image_feature_key: str | None,
    tokenizer_name: str,
    tokenizer_max_length: int,
) -> pd.DataFrame:
    """Run the stage labeler over the dataset and return a stage-label dataframe."""

    dataloader = create_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        shuffle=False,
        device=device,
        image_feature_key=image_feature_key,
        tokenizer_name=tokenizer_name,
        tokenizer_max_length=tokenizer_max_length,
    )
    resolved_image_feature_key = getattr(dataloader.dataset, "image_feature_key", image_feature_key)

    rows: dict[str, list[Any]] = {
        "index": [],
        "episode_index": [],
        "p_global": [],
        "p_local": [],
        "entropy": [],
        "coarse_l1": [],
        "refined_l1": [],
        "dominant_stage": [],
        "generator_version": [],
    }

    model.to(device)
    model.eval()
    logged_multimodal_debug = False

    for batch in tqdm(dataloader, desc="Generating stage labels"):
        batch = move_batch_to_device(batch, device)
        action_chunk = batch[ACTION].to(dtype=torch.float32)
        action_is_pad = batch.get("action_is_pad")
        conditioning_inputs = _extract_conditioning_inputs(batch)
        _validate_conditioning_inputs(
            conditioning_inputs=conditioning_inputs,
            batch_size=action_chunk.shape[0],
            context="export",
            image_feature_key=resolved_image_feature_key,
            log_once=not logged_multimodal_debug,
        )
        logged_multimodal_debug = True
        obs_image = conditioning_inputs["obs_image"]
        obs_state = conditioning_inputs[OBS_STATE]
        task_tokens = conditioning_inputs[OBS_LANGUAGE_TOKENS]
        task_mask = conditioning_inputs[OBS_LANGUAGE_ATTENTION_MASK]

        with torch.inference_mode():
            stage_probs = model.predict_stage_probs(
                action_chunk=action_chunk,
                obs_image=obs_image,
                obs_state=obs_state,
                task_tokens=task_tokens,
                task_mask=task_mask,
                action_is_pad=action_is_pad,
            )
            output = model(
                action_chunk=action_chunk,
                obs_image=obs_image,
                obs_state=obs_state,
                task_tokens=task_tokens,
                task_mask=task_mask,
                action_is_pad=action_is_pad,
                reduction="none",
            )

        entropy = -(stage_probs * torch.log(stage_probs.clamp_min(model.config.eps))).sum(dim=-1)
        coarse_l1 = _masked_l1_per_sample(output.coarse_chunk, output.action_chunk, output.valid_mask)
        refined_l1 = _masked_l1_per_sample(output.refined_chunk, output.action_chunk, output.valid_mask)

        index_values = batch["index"].view(-1).detach().cpu().numpy().astype(np.int64)
        episode_values = batch["episode_index"].view(-1).detach().cpu().numpy().astype(np.int64)
        p_global = stage_probs[:, 0].detach().cpu().numpy().astype(np.float32)
        p_local = stage_probs[:, 1].detach().cpu().numpy().astype(np.float32)
        entropy_values = entropy.detach().cpu().numpy().astype(np.float32)
        coarse_values = coarse_l1.detach().cpu().numpy().astype(np.float32)
        refined_values = refined_l1.detach().cpu().numpy().astype(np.float32)
        dominant_stage = np.where(p_global >= p_local, "global", "local").tolist()

        rows["index"].extend(index_values.tolist())
        rows["episode_index"].extend(episode_values.tolist())
        rows["p_global"].extend(p_global.tolist())
        rows["p_local"].extend(p_local.tolist())
        rows["entropy"].extend(entropy_values.tolist())
        rows["coarse_l1"].extend(coarse_values.tolist())
        rows["refined_l1"].extend(refined_values.tolist())
        rows["dominant_stage"].extend(dominant_stage)
        rows["generator_version"].extend([GENERATOR_VERSION] * len(index_values))

    dataframe = pd.DataFrame(rows)
    dataframe = dataframe.sort_values("index").reset_index(drop=True)
    return dataframe


def save_stage_labels(
    dataframe: pd.DataFrame,
    output_path: str | Path,
    metadata: dict[str, Any],
) -> Path:
    """Save the stage-label dataframe to parquet with pyarrow metadata."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    table = pa.Table.from_pandas(dataframe, preserve_index=False)
    schema_metadata = dict(table.schema.metadata or {})
    schema_metadata.update({key.encode(): str(value).encode() for key, value in metadata.items()})
    table = table.replace_schema_metadata(schema_metadata)

    pq.write_table(table, output_path)
    logging.info("Saved %s stage labels to %s", len(dataframe), output_path)
    return output_path


def save_checkpoint(
    model: FutureChunkStageLabeler,
    checkpoint_path: str | Path,
    train_steps_completed: int,
) -> Path:
    """Save a stage-labeler checkpoint containing weights and config."""

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "model_state_dict": model.state_dict(),
        "stage_labeler_config": asdict(model.config),
        "train_steps_completed": int(train_steps_completed),
        "generator_version": GENERATOR_VERSION,
    }
    torch.save(payload, checkpoint_path)
    logging.info("Saved checkpoint to %s", checkpoint_path)
    return checkpoint_path


def load_checkpoint(
    checkpoint_path: str | Path,
    device: torch.device,
) -> tuple[FutureChunkStageLabeler, StageLabelerConfig, dict[str, Any]]:
    """Load a stage-labeler checkpoint from disk."""

    checkpoint_path = Path(checkpoint_path)
    payload = torch.load(checkpoint_path, map_location="cpu")

    if "model_state_dict" not in payload:
        raise KeyError(f"Checkpoint '{checkpoint_path}' is missing 'model_state_dict'.")
    if "stage_labeler_config" not in payload:
        raise KeyError(f"Checkpoint '{checkpoint_path}' is missing 'stage_labeler_config'.")

    config_dict = dict(payload["stage_labeler_config"])
    if "target_probs" in config_dict:
        config_dict["target_probs"] = tuple(config_dict["target_probs"])

    config = StageLabelerConfig(**config_dict)
    model = FutureChunkStageLabeler(config)
    load_result = model.load_state_dict(payload["model_state_dict"], strict=False)
    if len(load_result.missing_keys) > 0:
        logging.info("Checkpoint is missing keys that will be freshly initialized: %s", load_result.missing_keys)
    if len(load_result.unexpected_keys) > 0:
        logging.info("Checkpoint has unexpected keys that were ignored: %s", load_result.unexpected_keys)
    model.to(device)

    logging.info("Loaded checkpoint from %s", checkpoint_path)
    return model, config, payload


def main() -> None:
    """Entry point for training or loading the stage labeler and exporting labels."""

    parser = build_argparser()
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_seed(args.seed)
    device = resolve_device(args.device)
    logging.info("Using device: %s", device)

    dataset = build_action_chunk_dataset(
        dataset_repo_id=args.dataset_repo_id,
        root=args.root,
        revision=args.revision,
        chunk_size=args.chunk_size,
    )
    if dataset.num_frames == 0:
        raise ValueError(f"Dataset '{args.dataset_repo_id}' contains no frames to process.")

    action_dim = infer_action_dim(dataset)
    logging.info("Resolved action_dim=%s from dataset metadata.", action_dim)

    output_path = (
        Path(args.output_path)
        if args.output_path is not None
        else Path(dataset.root) / "smolvla_stage_labels.parquet"
    )
    checkpoint_path = (
        Path(args.checkpoint_path)
        if args.checkpoint_path is not None
        else Path(dataset.root) / "smolvla_stage_labeler_checkpoint.pt"
    )

    train_steps_completed = 0
    if args.resume_from_checkpoint is not None:
        model, stage_config, checkpoint_payload = load_checkpoint(args.resume_from_checkpoint, device)
        train_steps_completed = int(checkpoint_payload.get("train_steps_completed", 0))
        if stage_config.chunk_size != args.chunk_size:
            raise ValueError(
                "The loaded checkpoint uses chunk_size="
                f"{stage_config.chunk_size}, but the CLI requested chunk_size={args.chunk_size}."
            )
        if stage_config.action_dim != action_dim:
            raise ValueError(
                f"The loaded checkpoint expects action_dim={stage_config.action_dim}, "
                f"but the dataset provides action_dim={action_dim}."
            )
        if stage_config.vlm_model_name != args.vlm_model_name:
            logging.info(
                "Loaded checkpoint vlm_model_name=%s overrides CLI vlm_model_name=%s for tokenizer/encoder consistency.",
                stage_config.vlm_model_name,
                args.vlm_model_name,
            )
    else:
        stage_config = StageLabelerConfig(
            chunk_size=args.chunk_size,
            action_dim=action_dim,
            encoder_hidden_dim=args.encoder_hidden_dim,
            encoder_num_layers=args.encoder_num_layers,
            encoder_kernel_size=args.encoder_kernel_size,
            encoder_dropout=args.encoder_dropout,
            num_anchors=args.num_anchors,
            residual_l1_weight=args.residual_l1_weight,
            entropy_weight=args.entropy_weight,
            balance_weight=args.balance_weight,
            spacing_weight=args.spacing_weight,
            target_probs=tuple(args.target_probs),
            min_anchor_spacing_ratio=args.min_anchor_spacing_ratio,
            eps=args.eps,
            condition_hidden_dim=args.condition_hidden_dim,
            state_hidden_dim=args.state_hidden_dim,
            freeze_condition_encoders=args.freeze_condition_encoders,
            vlm_model_name=args.vlm_model_name,
            image_proj_dim=args.image_proj_dim,
            text_proj_dim=args.text_proj_dim,
            film_hidden_dim=args.film_hidden_dim,
            use_condition_film=args.use_condition_film,
        )
        model = FutureChunkStageLabeler(stage_config).to(device)

    if args.skip_train:
        if args.resume_from_checkpoint is None:
            raise ValueError("--skip-train requires --resume-from-checkpoint.")
        logging.info(
            "Skipping training and using checkpoint '%s' with train_steps_completed=%s.",
            args.resume_from_checkpoint,
            train_steps_completed,
        )
    else:
        additional_steps = train_stage_labeler(
            model=model,
            dataset=dataset,
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            train_steps=args.train_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            grad_clip_norm=args.grad_clip_norm,
            log_freq=args.log_freq,
            image_feature_key=args.image_feature_key,
            tokenizer_name=stage_config.vlm_model_name,
            tokenizer_max_length=args.tokenizer_max_length,
        )
        train_steps_completed += additional_steps
        save_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            train_steps_completed=train_steps_completed,
        )

    dataframe = generate_stage_label_dataframe(
        model=model,
        dataset=dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        device=device,
        image_feature_key=args.image_feature_key,
        tokenizer_name=stage_config.vlm_model_name,
        tokenizer_max_length=args.tokenizer_max_length,
    )

    metadata = {
        "dataset_repo_id": args.dataset_repo_id,
        "revision": dataset.revision,
        "chunk_size": stage_config.chunk_size,
        "action_dim": stage_config.action_dim,
        "num_anchors": stage_config.num_anchors,
        "train_steps": train_steps_completed,
        "generator_version": GENERATOR_VERSION,
        "vlm_model_name": stage_config.vlm_model_name,
        "condition_hidden_dim": stage_config.condition_hidden_dim,
        "state_hidden_dim": stage_config.state_hidden_dim,
        "image_proj_dim": stage_config.image_proj_dim,
        "text_proj_dim": stage_config.text_proj_dim,
        "film_hidden_dim": stage_config.film_hidden_dim,
        "use_condition_film": stage_config.use_condition_film,
        "freeze_condition_encoders": stage_config.freeze_condition_encoders,
    }
    save_stage_labels(dataframe=dataframe, output_path=output_path, metadata=metadata)

    logging.info("Finished stage-label generation.")
    logging.info("Parquet: %s", output_path)
    if not args.skip_train:
        logging.info("Checkpoint: %s", checkpoint_path)


def _masked_l1_per_sample(prediction: Tensor, target: Tensor, valid_mask: Tensor) -> Tensor:
    """Compute masked L1 per sample for tensors shaped ``[B, H, D]``."""

    error = (prediction - target).abs() * valid_mask.unsqueeze(-1).to(dtype=prediction.dtype)
    valid_steps = valid_mask.sum(dim=1).clamp_min(1).to(dtype=prediction.dtype)
    return error.sum(dim=(1, 2)) / (valid_steps * prediction.shape[-1])


if __name__ == "__main__":
    main()

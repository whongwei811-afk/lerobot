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

"""Compute chunk-level stage labels for SmolVLA future action chunks.

This script trains or loads a :class:`FutureChunkStageLabeler`, then runs it
across a full :class:`LeRobotDataset` to generate a
``smolvla_stage_labels.parquet`` file.

The stage labeler operates on future action chunks only. To keep the pipeline
lightweight, the script loads dataset metadata with ``download_videos=False``
and uses an action-only dataset wrapper so that images and videos are never
decoded during training or inference.
"""

from __future__ import annotations

import argparse
import logging
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from lerobot.datasets import LeRobotDataset
from lerobot.utils.constants import ACTION

try:
    from lerobot.policies.smolvla.stage_labeler import (
        FutureChunkStageLabeler,
        StageLabelerConfig,
    )
except ModuleNotFoundError:
    from stage_labeler import FutureChunkStageLabeler, StageLabelerConfig


GENERATOR_VERSION = "smolvla_future_chunk_stage_labeler_v1"


class _ActionChunkDataset(Dataset[dict[str, Tensor]]):
    """Action-only view over :class:`LeRobotDataset`.

    The standard :class:`LeRobotDataset` item path may decode video-backed
    observations when the dataset contains camera features. This wrapper reads
    only the columns needed for stage labeling and uses the dataset reader's
    delta-index utilities to build future action chunks without touching
    visual observations.

    When auto-collation is enabled, :class:`torch.utils.data.DataLoader` can
    call :meth:`__getitems__` with a full batch of indices. This lets us fetch
    base columns and action rows in bulk instead of issuing dozens of Python-
    level random accesses per batch.
    """

    def __init__(self, dataset: LeRobotDataset) -> None:
        self.dataset = dataset
        self.reader = dataset._ensure_reader()
        if self.reader.hf_dataset is None:
            self.reader.load_and_activate()

        if ACTION not in dataset.meta.features:
            raise KeyError(f"Dataset '{dataset.repo_id}' does not contain the '{ACTION}' feature.")

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

    def _get_action_row(self, idx: int) -> Tensor:
        hf_dataset = self._get_hf_dataset()
        action_value = hf_dataset[ACTION][idx]
        if isinstance(action_value, Tensor):
            return action_value
        return torch.as_tensor(action_value)

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
    future actions and re-instantiates :class:`LeRobotDataset` with
    ``download_videos=False``.
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

    logging.info(
        "Building action-chunk dataset with fps=%s, chunk_size=%s, download_videos=False",
        fps,
        chunk_size,
    )
    dataset = LeRobotDataset(
        repo_id=dataset_repo_id,
        root=root,
        revision=revision,
        delta_timestamps=delta_timestamps,
        download_videos=False,
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

    action_dataset = _ActionChunkDataset(dataset)
    sample = action_dataset[0]
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
) -> DataLoader[dict[str, Tensor]]:
    """Create a dataloader for action-only stage-labeler batches."""

    if num_workers < 0:
        raise ValueError(f"num_workers must be >= 0, got {num_workers}.")
    if prefetch_factor <= 0:
        raise ValueError(f"prefetch_factor must be > 0, got {prefetch_factor}.")
    if device.type == "cuda" and num_workers == 0:
        logging.info(
            "DataLoader is running with num_workers=0, so CPU reads stay on the main process. "
            "Increase --num-workers to overlap dataset prefetch with GPU compute."
        )

    action_dataset: Dataset[dict[str, Tensor]]
    if isinstance(dataset, LeRobotDataset):
        action_dataset = _ActionChunkDataset(dataset)
    else:
        action_dataset = dataset

    dataloader_kwargs: dict[str, Any] = {
        "dataset": action_dataset,
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
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
    )
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

        optimizer.zero_grad(set_to_none=True)
        output = model(action_chunk=action_chunk, action_is_pad=action_is_pad, reduction="mean")
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
) -> pd.DataFrame:
    """Run the stage labeler over the dataset and return a stage-label dataframe."""

    dataloader = create_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        shuffle=False,
        device=device,
    )

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

    for batch in tqdm(dataloader, desc="Generating stage labels"):
        batch = move_batch_to_device(batch, device)
        action_chunk = batch[ACTION].to(dtype=torch.float32)
        action_is_pad = batch.get("action_is_pad")

        with torch.inference_mode():
            stage_probs = model.predict_stage_probs(action_chunk=action_chunk, action_is_pad=action_is_pad)
            output = model(action_chunk=action_chunk, action_is_pad=action_is_pad, reduction="none")

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
    model.load_state_dict(payload["model_state_dict"])
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
    )

    metadata = {
        "dataset_repo_id": args.dataset_repo_id,
        "revision": dataset.revision,
        "chunk_size": stage_config.chunk_size,
        "action_dim": stage_config.action_dim,
        "num_anchors": stage_config.num_anchors,
        "train_steps": train_steps_completed,
        "generator_version": GENERATOR_VERSION,
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

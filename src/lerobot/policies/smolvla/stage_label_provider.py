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

"""Utilities to read chunk-level stage labels generated for SmolVLA training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch


class StageLabelProvider:
    """Load stage soft labels from parquet and serve them by dataset index.

    The parquet file is expected to contain at least the following columns:
    ``index``, ``p_global`` and ``p_local``.
    """

    _REQUIRED_COLUMNS = ("index", "p_global", "p_local")

    def __init__(
        self,
        stage_label_path: str | Path,
        fallback_probs: tuple[float, float] = (0.5, 0.5),
        cache_in_memory: bool = True,
    ) -> None:
        self.stage_label_path = Path(stage_label_path)
        if not self.stage_label_path.is_file():
            raise FileNotFoundError(
                "Stage-label parquet file was not found at "
                f"'{self.stage_label_path}'. Set `stage_label_path` to a valid parquet file."
            )

        self.fallback_probs = self._normalize_probs(fallback_probs)
        self.cache_in_memory = cache_in_memory
        self._cached_index_map: dict[int, tuple[float, float]] | None = None

        stage_label_frame = self._read_stage_label_frame()
        if self.cache_in_memory:
            self._cached_index_map = self._build_index_map(stage_label_frame)

    def get_batch_stage_probs(self, indices: torch.Tensor | list[int]) -> torch.Tensor:
        """Return stage probabilities aligned with the provided indices.

        Args:
            indices: Dataset indices as a CPU tensor, CUDA tensor, or Python list.

        Returns:
            A ``torch.float32`` tensor of shape ``[B, 2]`` containing
            ``[p_global, p_local]`` for each input index.
        """

        normalized_indices = self._normalize_indices(indices)
        if len(normalized_indices) == 0:
            return torch.empty((0, 2), dtype=torch.float32)

        index_map = self._cached_index_map
        if index_map is None:
            index_map = self._build_index_map(self._read_stage_label_frame())

        stage_probs = [index_map.get(index, self.fallback_probs) for index in normalized_indices]
        return torch.tensor(stage_probs, dtype=torch.float32)

    def _normalize_probs(self, probs: tuple[float, float]) -> tuple[float, float]:
        if len(probs) != 2:
            raise ValueError(f"`fallback_probs` must contain exactly 2 entries, got {len(probs)}.")

        if any(prob < 0.0 for prob in probs):
            raise ValueError("`fallback_probs` must be non-negative.")

        total = float(sum(probs))
        if total <= 0.0:
            raise ValueError("`fallback_probs` must sum to a positive value.")

        return tuple(float(prob) / total for prob in probs)

    def _normalize_indices(self, indices: torch.Tensor | list[int]) -> list[int]:
        if isinstance(indices, torch.Tensor):
            return [int(index) for index in indices.detach().reshape(-1).to(device="cpu").tolist()]

        if isinstance(indices, list):
            return [int(index) for index in indices]

        raise TypeError(
            "`indices` must be provided as a torch.Tensor or list[int], "
            f"got {type(indices).__name__}."
        )

    def _read_stage_label_frame(self) -> Any:
        try:
            import pandas as pd
        except ImportError:
            pd = None

        if pd is not None:
            stage_label_frame = pd.read_parquet(self.stage_label_path)
            self._validate_columns(stage_label_frame.columns)
            return stage_label_frame.loc[:, list(self._REQUIRED_COLUMNS)]

        try:
            import pyarrow.parquet as pq
        except ImportError as exc:
            raise ImportError(
                "Reading stage-label parquet files requires either `pandas` or `pyarrow` to be installed."
            ) from exc

        table = pq.read_table(self.stage_label_path)
        self._validate_columns(table.column_names)
        return table.select(list(self._REQUIRED_COLUMNS)).to_pandas()

    def _validate_columns(self, columns: Any) -> None:
        missing_columns = [column for column in self._REQUIRED_COLUMNS if column not in columns]
        if missing_columns:
            raise ValueError(
                f"Stage-label parquet '{self.stage_label_path}' is missing required columns: "
                f"{missing_columns}."
            )

    def _build_index_map(self, stage_label_frame: Any) -> dict[int, tuple[float, float]]:
        deduplicated = stage_label_frame.drop_duplicates(subset="index", keep="last")
        return {
            int(index): (float(p_global), float(p_local))
            for index, p_global, p_local in deduplicated.loc[
                :, list(self._REQUIRED_COLUMNS)
            ].itertuples(index=False, name=None)
        }

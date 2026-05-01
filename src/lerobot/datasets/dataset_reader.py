#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Private reader component for LeRobotDataset. Handles random-access reading (HF dataset, delta indices, video decoding)."""

import hashlib
import json
import logging
from collections.abc import Callable
from pathlib import Path
from urllib.parse import quote

import datasets
import numpy as np
import torch
from PIL import Image as PILImage

from .dataset_metadata import LeRobotDatasetMetadata
from .feature_utils import (
    check_delta_timestamps,
    get_delta_indices,
    get_hf_features_from_features,
)
from .io_utils import (
    hf_transform_to_torch,
    load_nested_dataset,
)
from .video_utils import decode_video_frames

logger = logging.getLogger(__name__)


class DatasetReader:
    """Encapsulates read-side state and methods for LeRobotDataset.

    Owns: hf_dataset, _absolute_to_relative_idx, delta_indices.
    """

    def __init__(
        self,
        meta: LeRobotDatasetMetadata,
        root: Path,
        episodes: list[int] | None,
        tolerance_s: float,
        video_backend: str,
        delta_timestamps: dict[str, list[float]] | None,
        image_transforms: Callable | None,
        use_image_cache: bool = False,
        image_cache_dir: str | Path | None = None,
        build_image_cache: bool = True,
    ):
        """Initialize the reader with metadata, filtering, and transform config.

        The HF dataset is not loaded here — call :meth:`try_load` or
        :meth:`load_and_activate` afterward.

        Args:
            meta: Dataset metadata instance.
            root: Local dataset root directory.
            episodes: Optional list of episode indices to select. ``None``
                means all episodes.
            tolerance_s: Timestamp synchronization tolerance in seconds.
            video_backend: Video decoding backend identifier.
            delta_timestamps: Optional dict mapping feature keys to lists of
                relative timestamp offsets for temporal context windows.
            image_transforms: Optional torchvision v2 transform applied to
                visual features.
            use_image_cache: If True, serve image-backed observations from a
                local uint8 memmap cache when available.
            image_cache_dir: Optional directory for image cache files. Defaults
                to ``root / "image_cache"``.
            build_image_cache: If True, build the cache when enabled and missing
                or stale. If False, silently falls back to HF image reads.
        """
        self._meta = meta
        self.root = root
        self.episodes = episodes
        self._tolerance_s = tolerance_s
        self._video_backend = video_backend
        self._image_transforms = image_transforms
        self._use_image_cache = use_image_cache
        self._image_cache_dir = Path(image_cache_dir) if image_cache_dir is not None else root / "image_cache"
        self._build_image_cache = build_image_cache

        self.hf_dataset: datasets.Dataset | None = None
        self._hf_dataset_without_cached_images: datasets.Dataset | None = None
        self._absolute_to_relative_idx: dict[int, int] | None = None
        self._delta_column_cache: dict[str, torch.Tensor] = {}
        self._image_cache: dict[str, np.memmap] = {}

        # Setup delta_indices (doesn't depend on hf_dataset)
        self.delta_indices = None
        if delta_timestamps is not None:
            check_delta_timestamps(delta_timestamps, meta.fps, tolerance_s)
            self.delta_indices = get_delta_indices(delta_timestamps, meta.fps)

    def try_load(self) -> bool:
        """Attempt to load from local cache. Returns True if data is sufficient."""
        try:
            self.hf_dataset = self._load_hf_dataset()
        except (FileNotFoundError, NotADirectoryError):
            self.hf_dataset = None
            return False
        if not self._check_cached_episodes_sufficient():
            self.hf_dataset = None
            return False
        self._build_index_mapping()
        self._build_delta_column_cache()
        self._activate_image_cache()
        return True

    def load_and_activate(self) -> None:
        """Load HF dataset from disk and build index mapping. Call after data is on disk."""
        self.hf_dataset = self._load_hf_dataset()
        self._build_index_mapping()
        self._build_delta_column_cache()
        self._activate_image_cache()

    def _build_index_mapping(self) -> None:
        """Build absolute-to-relative index mapping from loaded hf_dataset."""
        self._absolute_to_relative_idx = None
        if self.episodes is not None and self.hf_dataset is not None:
            self._absolute_to_relative_idx = {
                abs_idx.item() if isinstance(abs_idx, torch.Tensor) else abs_idx: rel_idx
                for rel_idx, abs_idx in enumerate(self.hf_dataset["index"])
            }

    def _build_delta_column_cache(self) -> None:
        """Cache numeric delta-timestamp columns for fast chunk queries."""
        self._delta_column_cache = {}
        if self.hf_dataset is None or self.delta_indices is None:
            return

        for key in self.delta_indices:
            feature = self._meta.features.get(key)
            if feature is None or feature["dtype"] in {"image", "video", "string"}:
                continue
            if key in self._meta.video_keys:
                continue

            column = self.hf_dataset.with_format("numpy", columns=[key])[:][key]
            if len(column) == 0:
                continue

            if not isinstance(column, np.ndarray):
                column = np.asarray(column)
            self._delta_column_cache[key] = torch.as_tensor(column)

    def _image_cache_keys(self) -> list[str]:
        return [
            key
            for key in self._meta.camera_keys
            if self._meta.features.get(key, {}).get("dtype") == "image"
        ]

    def _image_cache_filename(self, key: str) -> str:
        return f"{quote(key, safe='')}.npy"

    def _image_cache_index_hash(self) -> str:
        if self.hf_dataset is None:
            return ""
        indices = self.hf_dataset.with_format("numpy", columns=["index"])[:]["index"]
        indices = np.asarray(indices, dtype=np.int64)
        digest = hashlib.sha1()
        digest.update(indices.shape[0].to_bytes(8, "little", signed=False))
        digest.update(indices.tobytes())
        return digest.hexdigest()

    def _read_image_cache_meta(self) -> dict | None:
        meta_path = self._image_cache_dir / "meta.json"
        if not meta_path.exists():
            return None
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    def _image_cache_is_valid(self, meta: dict, keys: list[str], index_hash: str) -> bool:
        if self.hf_dataset is None:
            return False
        if meta.get("version") != 1:
            return False
        if meta.get("num_frames") != len(self.hf_dataset):
            return False
        if meta.get("index_hash") != index_hash:
            return False
        if sorted(meta.get("keys", {})) != sorted(keys):
            return False
        return all((self._image_cache_dir / meta["keys"][key]["filename"]).exists() for key in keys)

    def _activate_image_cache(self) -> None:
        """Open or build uint8 image cache for image-backed camera keys."""
        self._image_cache = {}
        self._hf_dataset_without_cached_images = None
        if not self._use_image_cache or self.hf_dataset is None:
            return

        keys = self._image_cache_keys()
        if not keys:
            return

        index_hash = self._image_cache_index_hash()
        meta = self._read_image_cache_meta()
        if meta is None or not self._image_cache_is_valid(meta, keys, index_hash):
            if not self._build_image_cache:
                logger.warning(
                    "Image cache is enabled but missing/stale at %s; using HF image reads.",
                    self._image_cache_dir,
                )
                return
            self._build_image_cache_files(keys, index_hash)
            meta = self._read_image_cache_meta()
            if meta is None or not self._image_cache_is_valid(meta, keys, index_hash):
                raise RuntimeError(f"Failed to build a valid image cache at {self._image_cache_dir}")

        for key in keys:
            key_meta = meta["keys"][key]
            self._image_cache[key] = np.load(
                self._image_cache_dir / key_meta["filename"],
                mmap_mode="r",
            )

        self._hf_dataset_without_cached_images = self.hf_dataset.remove_columns(list(self._image_cache))
        logger.info("Using image cache at %s for keys: %s", self._image_cache_dir, sorted(self._image_cache))

    def _build_image_cache_files(self, keys: list[str], index_hash: str) -> None:
        if self.hf_dataset is None:
            return

        self._image_cache_dir.mkdir(parents=True, exist_ok=True)
        raw_dataset = self.hf_dataset.with_format(None)
        cache_meta = {
            "version": 1,
            "num_frames": len(self.hf_dataset),
            "index_hash": index_hash,
            "keys": {},
        }

        for key in keys:
            first_frame = self._image_to_uint8_chw(raw_dataset[0][key])
            filename = self._image_cache_filename(key)
            cache_path = self._image_cache_dir / filename
            mmap = np.lib.format.open_memmap(
                cache_path,
                mode="w+",
                dtype=np.uint8,
                shape=(len(self.hf_dataset), *first_frame.shape),
            )
            mmap[0] = first_frame
            for idx in range(1, len(self.hf_dataset)):
                mmap[idx] = self._image_to_uint8_chw(raw_dataset[idx][key])
            mmap.flush()
            cache_meta["keys"][key] = {
                "filename": filename,
                "shape": list(mmap.shape),
                "dtype": "uint8",
            }

        with (self._image_cache_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump(cache_meta, f, indent=2, sort_keys=True)

    def _image_to_uint8_chw(self, image) -> np.ndarray:
        if isinstance(image, PILImage.Image):
            array = np.asarray(image.convert("RGB"))
        elif isinstance(image, torch.Tensor):
            array = image.detach().cpu().numpy()
        else:
            array = np.asarray(image)

        if array.ndim != 3:
            raise ValueError(f"Expected image with 3 dimensions, got shape {array.shape}")
        chw = array if array.shape[0] in {1, 3, 4} else np.transpose(array, (2, 0, 1))
        if chw.shape[0] == 4:
            chw = chw[:3]
        if np.issubdtype(chw.dtype, np.floating):
            chw = np.clip(chw * 255.0, 0, 255)
        return np.asarray(chw, dtype=np.uint8)

    @property
    def num_frames(self) -> int:
        """Number of frames in selected episodes."""
        if self.episodes is not None and self.hf_dataset is not None:
            return len(self.hf_dataset)
        return self._meta.total_frames

    @property
    def num_episodes(self) -> int:
        """Number of episodes selected."""
        return len(self.episodes) if self.episodes is not None else self._meta.total_episodes

    def _load_hf_dataset(self) -> datasets.Dataset:
        """hf_dataset contains all the observations, states, actions, rewards, etc."""
        features = get_hf_features_from_features(self._meta.features)
        hf_dataset = load_nested_dataset(self.root / "data", features=features, episodes=self.episodes)
        hf_dataset.set_transform(hf_transform_to_torch)
        return hf_dataset

    def _check_cached_episodes_sufficient(self) -> bool:
        """Check if the cached dataset contains all requested episodes and their video files."""
        if self.hf_dataset is None or len(self.hf_dataset) == 0:
            return False

        available_episodes = {
            ep_idx.item() if isinstance(ep_idx, torch.Tensor) else ep_idx
            for ep_idx in self.hf_dataset.unique("episode_index")
        }

        if self.episodes is None:
            requested_episodes = set(range(self._meta.total_episodes))
        else:
            requested_episodes = set(self.episodes)

        if not requested_episodes.issubset(available_episodes):
            return False

        if len(self._meta.video_keys) > 0:
            for ep_idx in requested_episodes:
                for vid_key in self._meta.video_keys:
                    video_path = self.root / self._meta.get_video_file_path(ep_idx, vid_key)
                    if not video_path.exists():
                        return False

        return True

    def get_episodes_file_paths(self) -> list[Path]:
        """Return deduplicated file paths (data + video) for selected episodes.

        Used to build the ``allow_patterns`` list for ``snapshot_download``.
        """
        episodes = self.episodes if self.episodes is not None else list(range(self._meta.total_episodes))
        fpaths = [str(self._meta.get_data_file_path(ep_idx)) for ep_idx in episodes]
        if len(self._meta.video_keys) > 0:
            video_files = [
                str(self._meta.get_video_file_path(ep_idx, vid_key))
                for vid_key in self._meta.video_keys
                for ep_idx in episodes
            ]
            fpaths += video_files
        # episodes are stored in the same files, so we return unique paths only
        fpaths = list(set(fpaths))
        return fpaths

    def _get_query_indices(
        self, abs_idx: int, ep_idx: int
    ) -> tuple[dict[str, list[int]], dict[str, torch.Tensor]]:
        """Compute query indices for delta timestamps."""
        ep = self._meta.episodes[ep_idx]
        ep_start = ep["dataset_from_index"]
        ep_end = ep["dataset_to_index"]
        query_indices = {
            key: [max(ep_start, min(ep_end - 1, abs_idx + delta)) for delta in delta_idx]
            for key, delta_idx in self.delta_indices.items()
        }
        padding = {
            f"{key}_is_pad": torch.BoolTensor(
                [(abs_idx + delta < ep_start) | (abs_idx + delta >= ep_end) for delta in delta_idx]
            )
            for key, delta_idx in self.delta_indices.items()
        }
        return query_indices, padding

    def _get_query_timestamps(
        self,
        current_ts: float,
        query_indices: dict[str, list[int]] | None = None,
    ) -> dict[str, list[float]]:
        query_timestamps = {}
        for key in self._meta.video_keys:
            if query_indices is not None and key in query_indices:
                if self._absolute_to_relative_idx is not None:
                    relative_indices = [self._absolute_to_relative_idx[idx] for idx in query_indices[key]]
                    timestamps = self.hf_dataset[relative_indices]["timestamp"]
                else:
                    timestamps = self.hf_dataset[query_indices[key]]["timestamp"]
                query_timestamps[key] = torch.stack(timestamps).tolist()
            else:
                query_timestamps[key] = [current_ts]

        return query_timestamps

    def _query_hf_dataset(self, query_indices: dict[str, list[int]]) -> dict:
        """Query dataset for indices across keys, skipping video keys."""
        result: dict = {}
        for key, q_idx in query_indices.items():
            if key in self._meta.video_keys:
                continue
            relative_indices = (
                q_idx
                if self._absolute_to_relative_idx is None
                else [self._absolute_to_relative_idx[idx] for idx in q_idx]
            )
            cached_images = self._image_cache.get(key)
            if cached_images is not None:
                result[key] = torch.from_numpy(np.asarray(cached_images[relative_indices]).copy()).float() / 255.0
                continue
            cached_column = self._delta_column_cache.get(key)
            if cached_column is not None:
                index_tensor = torch.as_tensor(relative_indices, dtype=torch.long)
                result[key] = cached_column.index_select(0, index_tensor)
                continue
            try:
                result[key] = torch.stack(self.hf_dataset[key][relative_indices])
            except (KeyError, TypeError, IndexError):
                result[key] = torch.stack(self.hf_dataset[relative_indices][key])
        return result

    def _query_videos(self, query_timestamps: dict[str, list[float]], ep_idx: int) -> dict[str, torch.Tensor]:
        """Note: When using data workers (e.g. DataLoader with num_workers>0), do not call this function
        in the main process (e.g. by using a second Dataloader with num_workers=0). It will result in a
        Segmentation Fault.
        """
        ep = self._meta.episodes[ep_idx]
        item = {}
        for vid_key, query_ts in query_timestamps.items():
            from_timestamp = ep[f"videos/{vid_key}/from_timestamp"]
            shifted_query_ts = [from_timestamp + ts for ts in query_ts]

            video_path = self.root / self._meta.get_video_file_path(ep_idx, vid_key)
            frames = decode_video_frames(video_path, shifted_query_ts, self._tolerance_s, self._video_backend)
            item[vid_key] = frames.squeeze(0)

        return item

    def get_item(self, idx) -> dict:
        """Core __getitem__ logic. Assumes hf_dataset is loaded.

        ``idx`` is a *relative* index into the (possibly episode-filtered)
        HF dataset, **not** the absolute frame index stored in the ``index``
        column.  The absolute index is retrieved from the row itself.
        """
        hf_dataset = self._hf_dataset_without_cached_images or self.hf_dataset
        item = hf_dataset[idx]
        for key, cache in self._image_cache.items():
            item[key] = torch.from_numpy(np.asarray(cache[idx]).copy()).float() / 255.0
        ep_idx = item["episode_index"].item()
        abs_idx = item["index"].item()

        query_indices = None
        if self.delta_indices is not None:
            query_indices, padding = self._get_query_indices(abs_idx, ep_idx)
            query_result = self._query_hf_dataset(query_indices)
            item = {**item, **padding}
            for key, val in query_result.items():
                item[key] = val

        if len(self._meta.video_keys) > 0:
            current_ts = item["timestamp"].item()
            query_timestamps = self._get_query_timestamps(current_ts, query_indices)
            video_frames = self._query_videos(query_timestamps, ep_idx)
            item = {**video_frames, **item}

        if self._image_transforms is not None:
            image_keys = self._meta.camera_keys
            for cam in image_keys:
                item[cam] = self._image_transforms(item[cam])

        # Add task as a string
        task_idx = item["task_index"].item()
        item["task"] = self._meta.tasks.iloc[task_idx].name

        # add subtask information if available
        if "subtask_index" in self._meta.features and self._meta.subtasks is not None:
            subtask_idx = item["subtask_index"].item()
            item["subtask"] = self._meta.subtasks.iloc[subtask_idx].name

        return item

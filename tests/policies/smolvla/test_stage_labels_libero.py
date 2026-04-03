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

"""Minimal smoke test:
raw LIBERO dataset -> stage-labeled LIBERO dataset.

Pytest:
    LEROBOT_RUN_STAGE_LABEL_LIBERO_TEST=1 \
    python -m pytest tests/policies/smolvla/test_stage_labels_libero.py -q -s

Direct:
    PYTHONPATH=src python tests/policies/smolvla/test_stage_labels_libero.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.smolvla.stage_labels.factory import build_stage_label_generator
from lerobot.policies.smolvla.stage_labels.wrapper import StageLabeledDataset

RUN_TEST = os.getenv("LEROBOT_RUN_STAGE_LABEL_LIBERO_TEST") == "1"
REPO_ID = os.getenv("LEROBOT_STAGE_LABEL_DATASET_REPO_ID", "HuggingFaceVLA/libero")
ROOT_ENV = os.getenv("LEROBOT_STAGE_LABEL_DATASET_ROOT")
EPISODE = int(os.getenv("LEROBOT_STAGE_LABEL_EPISODE", "0"))
MAX_PRINT_SAMPLES = int(os.getenv("LEROBOT_STAGE_LABEL_MAX_PRINT_SAMPLES", "3"))

pytestmark = pytest.mark.skipif(
    not RUN_TEST,
    reason="Set LEROBOT_RUN_STAGE_LABEL_LIBERO_TEST=1 to run this test with a real LIBERO dataset.",
)


def _run_smoke_test() -> None:
    dataset_root = Path(ROOT_ENV).expanduser() if ROOT_ENV else None

    raw_dataset = LeRobotDataset(
        repo_id=REPO_ID,
        root=dataset_root,
        episodes=[EPISODE],
        download_videos=False,
    )
    generator = build_stage_label_generator("hard")
    stage_labeled_dataset = StageLabeledDataset(raw_dataset, generator, include_debug=True)

    print("raw_dataset =", raw_dataset)
    print("stage_labeled_dataset =", stage_labeled_dataset)
    print("len(raw_dataset) =", len(raw_dataset))
    print("len(stage_labeled_dataset) =", len(stage_labeled_dataset))

    num_samples = min(len(stage_labeled_dataset), MAX_PRINT_SAMPLES)
    assert num_samples > 0

    for idx in range(num_samples):
        sample = stage_labeled_dataset[idx]
        product = {
            "stage_id": sample["stage_id"],
            "stage_prob_target": sample["stage_prob_target"],
            "stage_valid_mask": sample["stage_valid_mask"],
            "stage_source": sample["stage_source"],
            "stage_debug": sample["stage_debug"],
        }
        print(f"sample[{idx}] =", product)

        assert isinstance(sample["stage_id"], torch.Tensor)
        assert isinstance(sample["stage_prob_target"], torch.Tensor)
        assert isinstance(sample["stage_valid_mask"], torch.Tensor)
        assert isinstance(sample["stage_source"], str)
        assert isinstance(sample["stage_debug"], dict)

    assert len(stage_labeled_dataset) == len(raw_dataset)


def test_raw_libero_dataset_can_be_converted_to_stage_labeled_dataset() -> None:
    _run_smoke_test()


if __name__ == "__main__":
    _run_smoke_test()

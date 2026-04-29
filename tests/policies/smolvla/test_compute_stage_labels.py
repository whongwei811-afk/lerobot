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

from types import SimpleNamespace

import torch

from lerobot.policies.smolvla.compute_stage_labels import _ActionChunkDataset
from lerobot.utils.constants import ACTION


def test_action_chunk_dataset_uses_cached_action_column_when_available():
    dataset = object.__new__(_ActionChunkDataset)
    cached_actions = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    dataset.reader = SimpleNamespace(_delta_column_cache={ACTION: cached_actions})
    dataset._get_hf_dataset = lambda: (_ for _ in ()).throw(
        AssertionError("HF dataset should not be queried when action cache is available.")
    )

    action_chunk = dataset._get_action_chunk([2, 4, 7])

    torch.testing.assert_close(action_chunk, cached_actions.index_select(0, torch.tensor([2, 4, 7])))

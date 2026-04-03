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

from __future__ import annotations

from typing import Any

from .base import BaseStageLabelGenerator
from .hard_label_generator import HardStageLabelGenerator
from .soft_label_generator import SoftStageLabelGenerator

SUPPORTED_STAGE_LABEL_GENERATOR_MODES: tuple[str, ...] = (
    "hard",
    "soft",
)


def build_stage_label_generator(
    mode: str,
    cfg: Any | None = None,
    **kwargs: Any,
) -> BaseStageLabelGenerator:
    """Build a SmolVLA stage-label generator from a mode string.

    This factory is intentionally lightweight and model-agnostic. It only selects
    and instantiates the appropriate stage-label generator implementation.

    Args:
        mode: Label generator mode. Supported values are `"hard"` and `"soft"`.
        cfg: Optional config object, mapping, or dataclass instance passed to the
            selected generator constructor.
        **kwargs: Optional keyword arguments forwarded to the generator
            constructor. This is useful when the caller wants to override the
            default `cfg` argument name in a generic construction path.

    Returns:
        An instance of `BaseStageLabelGenerator`.

    Raises:
        ValueError: If `mode` is not one of the supported stage-label generator
            modes.

    Examples:
        ```python
        generator = build_stage_label_generator("hard")
        output = generator(sample)

        soft_generator = build_stage_label_generator(
            "soft",
            cfg={"temperature": 0.8, "default_gripper_width": 2},
        )
        soft_output = soft_generator(sample)
        ```
    """
    normalized_mode = mode.strip().lower()
    if "cfg" in kwargs:
        if cfg is not None:
            raise ValueError(
                "`cfg` was provided twice to `build_stage_label_generator`: "
                "once as the explicit `cfg=` argument and once inside `**kwargs`."
            )
        cfg = kwargs.pop("cfg")

    if normalized_mode == "hard":
        return HardStageLabelGenerator(cfg=cfg, **kwargs)
    if normalized_mode == "soft":
        return SoftStageLabelGenerator(cfg=cfg, **kwargs)

    supported_modes = ", ".join(SUPPORTED_STAGE_LABEL_GENERATOR_MODES)
    raise ValueError(
        f"Unsupported stage label generator mode: {mode!r}. "
        f"Supported modes are: {supported_modes}."
    )

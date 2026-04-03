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

import abc
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch


@dataclass
class StageLabelOutput:
    """Unified container for stage labels produced by SmolVLA stage-label generators.

    This structure is intentionally tensor-first so it can be passed through dataset,
    dataloader, and collation flows with minimal special handling. Different label
    generation modes can populate different supervision fields while keeping the same
    output schema:

    - Hard labels typically use `stage_id`.
    - Soft labels typically use `stage_prob_target`.

    Attributes:
        stage_id: Optional discrete stage label tensor. This is typically an integer
            tensor with shape like `(...)` depending on the granularity of stage
            supervision.
        stage_prob_target: Optional probabilistic stage target tensor. This is
            typically a float tensor containing per-stage probabilities or soft targets
            with a shape like `(..., num_stages)`.
        stage_valid_mask: Boolean tensor indicating where stage supervision is valid
            and should contribute to loss or metrics. This field is always present so
            downstream code can gate supervision without inspecting the label source.
        stage_source: String identifier describing where the stage labels came from,
            such as `"hard"`, `"soft"`, or `"none"`.
        stage_debug: Auxiliary debug metadata for logging or inspection. Keep values
            small and preferably collate-friendly when this structure is used before a
            dataloader collation step.
    """

    stage_id: torch.Tensor | None = field(
        default=None,
        metadata={
            "doc": (
                "Optional discrete stage label tensor used by hard-label supervision."
            )
        },
    )
    stage_prob_target: torch.Tensor | None = field(
        default=None,
        metadata={
            "doc": (
                "Optional soft stage target tensor, typically storing class "
                "probabilities or confidence scores."
            )
        },
    )
    stage_valid_mask: torch.Tensor = field(
        default_factory=lambda: torch.tensor(False, dtype=torch.bool),
        metadata={
            "doc": (
                "Boolean tensor that marks which items or time steps contain valid "
                "stage supervision."
            )
        },
    )
    stage_source: str = field(
        default="none",
        metadata={
            "doc": "String tag describing the origin or mode of the generated stage labels."
        },
    )
    stage_debug: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "doc": "Debug-only metadata associated with stage label generation."
        },
    )

    def __post_init__(self) -> None:
        """Validate the output structure and normalize the validity mask dtype."""
        if self.stage_id is not None and not isinstance(self.stage_id, torch.Tensor):
            raise TypeError(
                f"`stage_id` must be a torch.Tensor or None, got {type(self.stage_id).__name__}."
            )
        if self.stage_prob_target is not None and not isinstance(self.stage_prob_target, torch.Tensor):
            raise TypeError(
                "`stage_prob_target` must be a torch.Tensor or None, "
                f"got {type(self.stage_prob_target).__name__}."
            )
        if not isinstance(self.stage_valid_mask, torch.Tensor):
            raise TypeError(
                "`stage_valid_mask` must be a torch.Tensor, "
                f"got {type(self.stage_valid_mask).__name__}."
            )
        if not isinstance(self.stage_source, str):
            raise TypeError(
                f"`stage_source` must be a string, got {type(self.stage_source).__name__}."
            )
        if not isinstance(self.stage_debug, dict):
            raise TypeError(
                f"`stage_debug` must be a dict, got {type(self.stage_debug).__name__}."
            )

        if self.stage_valid_mask.dtype != torch.bool:
            self.stage_valid_mask = self.stage_valid_mask.to(dtype=torch.bool)

    @classmethod
    def empty(
        cls,
        stage_valid_mask: torch.Tensor,
        *,
        stage_source: str = "none",
        stage_debug: dict[str, Any] | None = None,
    ) -> StageLabelOutput:
        """Create an output with no labels but an explicit validity mask.

        Args:
            stage_valid_mask: Boolean-like tensor describing which positions would be
                valid if labels were available.
            stage_source: A source tag for the output.
            stage_debug: Optional debug metadata.

        Returns:
            A `StageLabelOutput` instance with missing labels and a normalized validity
            mask.
        """
        return cls(
            stage_id=None,
            stage_prob_target=None,
            stage_valid_mask=stage_valid_mask,
            stage_source=stage_source,
            stage_debug=stage_debug or {},
        )

    def tensor_dict(self) -> dict[str, torch.Tensor | None]:
        """Return only tensor-like supervision fields.

        This is useful when downstream code wants a collate-friendly view containing
        only tensors and optional tensors.

        Returns:
            A dictionary containing the tensor fields in the unified output schema.
        """
        return {
            "stage_id": self.stage_id,
            "stage_prob_target": self.stage_prob_target,
            "stage_valid_mask": self.stage_valid_mask,
        }

    def to_dict(self, include_debug: bool = True) -> dict[str, Any]:
        """Convert the output to a plain dictionary.

        Args:
            include_debug: Whether to include `stage_debug` in the returned mapping.

        Returns:
            A dictionary representation of the output.
        """
        output = {
            "stage_id": self.stage_id,
            "stage_prob_target": self.stage_prob_target,
            "stage_valid_mask": self.stage_valid_mask,
            "stage_source": self.stage_source,
        }
        if include_debug:
            output["stage_debug"] = self.stage_debug
        return output


class BaseStageLabelGenerator(abc.ABC):
    """Abstract interface for all SmolVLA stage-label generators.

    Concrete generators should implement `generate()` and return the same
    `StageLabelOutput` schema regardless of how labels are produced. This keeps
    downstream components decoupled from the label source and allows hard and
    soft label generation strategies to be swapped without changing model-facing
    code.
    """

    @abc.abstractmethod
    def generate(self, sample: Mapping[str, Any], **kwargs: Any) -> StageLabelOutput:
        """Generate unified stage labels from a single sample-like input.

        Args:
            sample: A mapping containing the single-sample context required to
                produce stage labels. The exact expected keys are defined by each
                concrete implementation.
            **kwargs: Optional implementation-specific generation arguments.

        Returns:
            A `StageLabelOutput` instance containing stage supervision in the unified
            schema.
        """

    def __call__(self, sample: Mapping[str, Any], **kwargs: Any) -> StageLabelOutput:
        """Forward callable usage to `generate()`."""
        return self.generate(sample=sample, **kwargs)

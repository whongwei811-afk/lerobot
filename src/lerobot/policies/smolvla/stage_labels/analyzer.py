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

"""Stage label quality analyzer for debugging and validation.

This module provides tools to:
1. Compute statistics about generated stage labels
2. Validate label consistency within batches
3. Compare different generators on the same data
4. Detect data quality issues

Example:
    ```python
    from lerobot.policies.smolvla.stage_labels.analyzer import StageLabelStatistics
    from lerobot.policies.smolvla.stage_labels.wrapper import StageLabeledDataset

    dataset = StageLabeledDataset(base_dataset, generator)
    stats = StageLabelStatistics(dataset, max_samples=1000)
    stats.compute()
    stats.report()
    ```
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Mapping, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class StageLabelStatistics:
    """Compute and report stage label statistics from a labeled dataset.

    CRITICAL FEATURE: Provides comprehensive validation of label quality including:
    - Valid label ratio (how many samples have stage_valid_mask=True)
    - Stage distribution (move/grasp/place counts)
    - Soft label statistics (probability distributions)
    - Invalid reason statistics
    - Device consistency checks
    """

    def __init__(
        self,
        dataset: Dataset,
        max_samples: Optional[int] = None,
        batch_size: int = 1,
        num_workers: int = 0,
        verbose: bool = True,
    ):
        """Initialize the stage label statistics analyzer.

        Args:
            dataset: StageLabeledDataset or compatible dataset.
            max_samples: Maximum number of samples to analyze. If None, analyze all.
            batch_size: Batch size for DataLoader.
            num_workers: Number of workers for DataLoader.
            verbose: Whether to print progress information.
        """
        self.dataset = dataset
        self.max_samples = max_samples or len(dataset)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.verbose = verbose

        # Statistics containers
        self.total_samples = 0
        self.valid_samples = 0
        self.invalid_samples = 0

        # Stage distribution (for discrete hard labels)
        self.stage_id_counts = Counter()
        self.stage_name_counts = Counter()

        # Soft label statistics
        self.soft_label_probs = []  # List of probability vectors
        self.soft_label_max_probs = []  # Maximum probability per sample
        self.soft_label_entropy_values = []  # Shannon entropy

        # Invalid reason statistics
        self.invalid_reasons = Counter()

        # Device statistics
        self.device_types = Counter()

        # Source type distribution
        self.source_counts = Counter()

        # Per-stage energy statistics (if available in debug)
        self.stage_energy_stats = defaultdict(list)

    def compute(self) -> None:
        """Compute all statistics by iterating through the dataset."""
        self._reset()

        dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._custom_collate,  # Handle variable-size tensors
        )

        samples_processed = 0
        for batch_idx, batch in enumerate(dataloader):
            if samples_processed >= self.max_samples:
                break

            for sample_idx in range(len(batch.get("stage_id", []))):
                if samples_processed >= self.max_samples:
                    break

                self._process_sample(
                    {key: value[sample_idx] for key, value in batch.items()}
                )
                samples_processed += 1

            if self.verbose and (batch_idx + 1) % max(1, len(dataloader) // 10) == 0:
                print(f"Processed {samples_processed}/{min(self.max_samples, len(self.dataset))} samples")

        if self.verbose:
            print(f"Analysis complete: {samples_processed} samples processed")

    def _process_sample(self, sample: dict[str, Any]) -> None:
        """Process a single sample and update statistics.

        Args:
            sample: A single sample dict with stage labels.
        """
        self.total_samples += 1

        stage_id = sample.get("stage_id")
        stage_prob_target = sample.get("stage_prob_target")
        stage_valid_mask = sample.get("stage_valid_mask")
        stage_source = sample.get("stage_source", "unknown")
        stage_debug = sample.get("stage_debug", {})

        # Track source type
        self.source_counts[stage_source] += 1

        # Track device
        if isinstance(stage_id, torch.Tensor):
            self.device_types[str(stage_id.device)] += 1
        elif isinstance(stage_prob_target, torch.Tensor):
            self.device_types[str(stage_prob_target.device)] += 1

        # Check validity
        is_valid = self._get_bool_value(stage_valid_mask)
        if is_valid:
            self.valid_samples += 1

            # Process discrete stage ids only for samples that truly came from the hard generator.
            if stage_source == "hard" and isinstance(stage_id, torch.Tensor):
                sid = int(stage_id.item()) if stage_id.numel() == 1 else int(stage_id)
                self.stage_id_counts[sid] += 1
                if "stage_name" in stage_debug:
                    self.stage_name_counts[stage_debug["stage_name"]] += 1

            # Process soft labels only for samples that truly came from the soft generator.
            if (
                stage_source == "soft"
                and isinstance(stage_prob_target, torch.Tensor)
                and stage_prob_target.numel() > 0
            ):
                prob_np = stage_prob_target.detach().cpu().numpy().astype(np.float32)
                self.soft_label_probs.append(prob_np)
                self.soft_label_max_probs.append(float(np.max(prob_np)))
                self.soft_label_entropy_values.append(self._compute_entropy(prob_np))

            # Collect energy statistics if available
            if "large_joint_energy" in stage_debug:
                self.stage_energy_stats["large_joint_energy"].append(
                    float(stage_debug.get("large_joint_energy", 0.0))
                )
                self.stage_energy_stats["small_joint_energy"].append(
                    float(stage_debug.get("small_joint_energy", 0.0))
                )
                self.stage_energy_stats["gripper_energy"].append(
                    float(stage_debug.get("gripper_energy", 0.0))
                )

        else:
            self.invalid_samples += 1
            if "decision_reason" in stage_debug:
                self.invalid_reasons[stage_debug["decision_reason"]] += 1

    def _get_bool_value(self, value: Any) -> bool:
        """Extract boolean value from various tensor/scalar types."""
        if isinstance(value, torch.Tensor):
            return bool(value.item())
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        return bool(value)

    @staticmethod
    def _compute_entropy(prob_vector: np.ndarray) -> float:
        """Compute Shannon entropy of a probability distribution."""
        # Clip to avoid log(0)
        p = np.clip(prob_vector, 1e-10, 1.0)
        return float(-np.sum(p * np.log(p)))

    def _reset(self) -> None:
        """Reset all statistics."""
        self.total_samples = 0
        self.valid_samples = 0
        self.invalid_samples = 0
        self.stage_id_counts.clear()
        self.stage_name_counts.clear()
        self.soft_label_probs.clear()
        self.soft_label_max_probs.clear()
        self.soft_label_entropy_values.clear()
        self.invalid_reasons.clear()
        self.device_types.clear()
        self.source_counts.clear()
        self.stage_energy_stats.clear()

    @staticmethod
    def _custom_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        """Collate a batch conservatively for stage-label analysis.

        Collation policy by field type:
        - If all values for a key are `torch.Tensor` and all tensor shapes match,
          stack them with `torch.stack(...)`.
        - If all values are tensors but shapes differ, keep them as
          `list[torch.Tensor]`.
        - If any non-tensor value is present, keep the full field as a Python
          list without attempting to stack.

        This is intentionally more conservative than training-time collation
        because stage-label analysis frequently inspects mixed field types such as
        strings (`stage_source`), dicts (`stage_debug`), empty tensors, or
        variable-shape tensors. Explicit type/shape checks make the behavior
        predictable and avoid relying on `RuntimeError` fallbacks from
        `torch.stack(...)`.

        Args:
            batch: List of sample dictionaries.

        Returns:
            A collated dictionary where same-shape tensor fields are stacked and
            all other fields remain per-sample lists.
        """
        if not batch:
            return {}

        collated: dict[str, Any] = {}
        all_keys = set().union(*(sample.keys() for sample in batch))
        for key in all_keys:
            values = [sample.get(key) for sample in batch]

            if all(isinstance(value, torch.Tensor) for value in values):
                reference_shape = values[0].shape
                if all(value.shape == reference_shape for value in values):
                    collated[key] = torch.stack(values)
                else:
                    collated[key] = values
            else:
                collated[key] = values

        return collated

    def report(self) -> str:
        """Generate a comprehensive statistics report.

        Returns:
            Formatted report string.
        """
        lines = []

        # Header
        lines.append("=" * 80)
        lines.append("STAGE LABEL STATISTICS REPORT")
        lines.append("=" * 80)

        # Summary
        lines.append(f"\nSummary:")
        lines.append(f"  Total samples: {self.total_samples}")
        lines.append(f"  Valid samples: {self.valid_samples} ({100*self.valid_samples/max(1, self.total_samples):.1f}%)")
        lines.append(f"  Invalid samples: {self.invalid_samples} ({100*self.invalid_samples/max(1, self.total_samples):.1f}%)")

        # Source distribution
        if self.source_counts:
            lines.append(f"\nLabel source distribution:")
            for source, count in sorted(self.source_counts.items(), key=lambda x: -x[1]):
                pct = 100 * count / self.total_samples
                lines.append(f"  {source}: {count} ({pct:.1f}%)")

        # Stage distribution
        if self.stage_name_counts:
            lines.append(f"\nStage distribution (by name):")
            total_named = sum(self.stage_name_counts.values())
            for stage, count in sorted(self.stage_name_counts.items(), key=lambda x: -x[1]):
                pct = 100 * count / total_named
                lines.append(f"  {stage}: {count} ({pct:.1f}%)")

        if self.stage_id_counts:
            lines.append(f"\nStage distribution (by ID):")
            total_id = sum(self.stage_id_counts.values())
            for stage_id, count in sorted(self.stage_id_counts.items()):
                pct = 100 * count / total_id
                lines.append(f"  Stage {stage_id}: {count} ({pct:.1f}%)")

        # Soft label statistics
        if self.soft_label_probs:
            lines.append(f"\nSoft label statistics:")
            max_probs = np.array(self.soft_label_max_probs)
            lines.append(f"  Max probability (per sample):")
            lines.append(f"    Mean: {np.mean(max_probs):.4f}")
            lines.append(f"    Std: {np.std(max_probs):.4f}")
            lines.append(f"    Min: {np.min(max_probs):.4f}")
            lines.append(f"    Max: {np.max(max_probs):.4f}")

            entropies = np.array(self.soft_label_entropy_values)
            lines.append(f"  Shannon entropy (distribution uncertainty):")
            lines.append(f"    Mean: {np.mean(entropies):.4f}")
            lines.append(f"    Std: {np.std(entropies):.4f}")
            lines.append(f"    Min: {np.min(entropies):.4f}")
            lines.append(f"    Max: {np.max(entropies):.4f}")

            # Average probability distribution
            probs_array = np.array(self.soft_label_probs)
            avg_probs = np.mean(probs_array, axis=0)
            lines.append(f"  Average probability distribution:")
            for idx, prob in enumerate(avg_probs):
                lines.append(f"    Stage {idx}: {prob:.4f}")

        # Invalid reason distribution
        if self.invalid_reasons:
            lines.append(f"\nInvalid label reasons:")
            total_invalid = sum(self.invalid_reasons.values())
            for reason, count in sorted(self.invalid_reasons.items(), key=lambda x: -x[1]):
                pct = 100 * count / total_invalid
                lines.append(f"  {reason}: {count} ({pct:.1f}%)")

        # Energy statistics
        if self.stage_energy_stats:
            lines.append(f"\nEnergy statistics (from debug data):")
            for energy_type in ["large_joint_energy", "small_joint_energy", "gripper_energy"]:
                if energy_type in self.stage_energy_stats:
                    values = np.array(self.stage_energy_stats[energy_type])
                    lines.append(f"  {energy_type}:")
                    lines.append(f"    Mean: {np.mean(values):.6f}")
                    lines.append(f"    Std: {np.std(values):.6f}")
                    lines.append(f"    Min: {np.min(values):.6f}")
                    lines.append(f"    Max: {np.max(values):.6f}")

        # Device information
        if self.device_types:
            lines.append(f"\nDevice types:")
            for device, count in sorted(self.device_types.items(), key=lambda x: -x[1]):
                pct = 100 * count / self.total_samples
                lines.append(f"  {device}: {count} ({pct:.1f}%)")

        lines.append("\n" + "=" * 80)

        return "\n".join(lines)

    def print_report(self) -> None:
        """Print the full statistics report to stdout."""
        print(self.report())

    def get_stats_dict(self) -> dict[str, Any]:
        """Return statistics as a dictionary for programmatic access.

        Returns:
            Dictionary with all computed statistics.
        """
        return {
            "total_samples": self.total_samples,
            "valid_samples": self.valid_samples,
            "invalid_samples": self.invalid_samples,
            "valid_ratio": self.valid_samples / max(1, self.total_samples),
            "stage_id_counts": dict(self.stage_id_counts),
            "stage_name_counts": dict(self.stage_name_counts),
            "invalid_reasons": dict(self.invalid_reasons),
            "source_counts": dict(self.source_counts),
            "device_types": dict(self.device_types),
            "soft_label_max_probs_stats": {
                "mean": float(np.mean(self.soft_label_max_probs)) if self.soft_label_max_probs else None,
                "std": float(np.std(self.soft_label_max_probs)) if self.soft_label_max_probs else None,
                "min": float(np.min(self.soft_label_max_probs)) if self.soft_label_max_probs else None,
                "max": float(np.max(self.soft_label_max_probs)) if self.soft_label_max_probs else None,
            },
            "soft_label_entropy_stats": {
                "mean": float(np.mean(self.soft_label_entropy_values)) if self.soft_label_entropy_values else None,
                "std": float(np.std(self.soft_label_entropy_values)) if self.soft_label_entropy_values else None,
                "min": float(np.min(self.soft_label_entropy_values)) if self.soft_label_entropy_values else None,
                "max": float(np.max(self.soft_label_entropy_values)) if self.soft_label_entropy_values else None,
            },
        }


def compare_generators(
    base_dataset: Dataset,
    generators: dict[str, Any],
    max_samples: int = 100,
    verbose: bool = True,
) -> dict[str, dict[str, Any]]:
    """Compare statistics across multiple stage label generators.

    CRITICAL FEATURE: Allows validation that different generators produce
    consistent labels on the same data.

    Args:
        base_dataset: Unlabeled dataset.
        generators: Dict mapping generator name to generator instance.
        max_samples: Maximum samples to analyze.
        verbose: Whether to print progress.

    Returns:
        Dictionary mapping generator name to statistics dict.
    """
    from .wrapper import StageLabeledDataset

    results = {}
    for name, generator in generators.items():
        if verbose:
            print(f"\nAnalyzing generator: {name}")

        labeled_dataset = StageLabeledDataset(base_dataset, generator)
        analyzer = StageLabelStatistics(labeled_dataset, max_samples=max_samples, verbose=verbose)
        analyzer.compute()
        results[name] = analyzer.get_stats_dict()

        if verbose:
            analyzer.print_report()

    return results

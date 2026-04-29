#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

"""
SmolVLA:

[Paper](https://huggingface.co/papers/2506.01844)

Designed by Hugging Face.

Install smolvla extra dependencies:
```bash
pip install -e ".[smolvla]"
```

Example of finetuning the smolvla pretrained model (`smolvla_base`):
```bash
lerobot-train \
--policy.path=lerobot/smolvla_base \
--dataset.repo_id=<USER>/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of finetuning a smolVLA. SmolVLA is composed of a pretrained VLM,
and an action expert.
```bash
lerobot-train \
--policy.type=smolvla \
--dataset.repo_id=<USER>/svla_so100_task1_v3 \
--batch_size=64 \
--steps=200000
```

Example of using the smolvla pretrained model outside LeRobot training framework:
```python
policy = SmolVLAPolicy.from_pretrained("lerobot/smolvla_base")
```

"""

import math
from collections import deque
from typing import TypedDict, Unpack

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE
from lerobot.utils.device_utils import get_safe_dtype

from ..pretrained import PreTrainedPolicy
from ..rtc.modeling_rtc import RTCProcessor
from ..utils import (
    populate_queues,
)
from .configuration_smolvla import SmolVLAConfig
from .smolvlm_with_expert import SmolVLMWithExpertModel
from .stage_label_provider import StageLabelProvider


class ActionSelectKwargs(TypedDict, total=False):
    inference_delay: int | None
    prev_chunk_left_over: Tensor | None
    execution_horizon: int | None


class StageHeadOutput(TypedDict, total=False):
    stage_context: Tensor
    stage_logits: Tensor
    stage_probs: Tensor


class ActionConditionedOutput(TypedDict, total=False):
    stage_outputs: StageHeadOutput | None
    action_hidden: Tensor
    conditioned_action_hidden: Tensor
    action_pred: Tensor
    action_target: Tensor
    z_cond: Tensor
    alpha: Tensor
    l_large_elementwise: Tensor
    l_small_elementwise: Tensor


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    # assume no-op when width height fits already
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """Can be (batch_size x sequence_length x features_dimension)
    or (batch_size x features_dimension)
    """
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def soft_target_cross_entropy(logits: Tensor, target_probs: Tensor) -> Tensor:
    """Compute cross-entropy against soft targets and return per-sample losses."""
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_probs * log_probs).sum(dim=-1)


def normalize(x, min_val, max_val):
    return (x - min_val) / (max_val - min_val)


def unnormalize(x, min_val, max_val):
    return x * (max_val - min_val) + min_val


def safe_arcsin(value):
    # This ensures that the input stays within
    # [−1,1] to avoid invalid values for arcsin
    return torch.arcsin(torch.clamp(value, -1.0, 1.0))


def aloha_gripper_to_angular(value):
    # Aloha transforms the gripper positions into a linear space. The following code
    # reverses this transformation to be consistent with smolvla which is pretrained in
    # angular space.
    #
    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_POSITION_OPEN, PUPPET_GRIPPER_POSITION_CLOSED
    value = unnormalize(value, min_val=0.01844, max_val=0.05800)

    # This is the inverse of the angular to linear transformation inside the Interbotix code.
    def linear_to_radian(linear_position, arm_length, horn_radius):
        value = (horn_radius**2 + linear_position**2 - arm_length**2) / (2 * horn_radius * linear_position)
        return safe_arcsin(value)

    # The constants are taken from the Interbotix code.
    value = linear_to_radian(value, arm_length=0.036, horn_radius=0.022)

    # Normalize to [0, 1].
    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    return normalize(value, min_val=0.4, max_val=1.5)


def aloha_gripper_from_angular(value):
    # Convert from the gripper position used by smolvla to the gripper position that is used by Aloha.
    # Note that the units are still angular but the range is different.

    # The values 0.4 and 1.5 were measured on an actual Trossen robot.
    value = unnormalize(value, min_val=0.4, max_val=1.5)

    # These values are coming from the Aloha code:
    # PUPPET_GRIPPER_JOINT_OPEN, PUPPET_GRIPPER_JOINT_CLOSE
    return normalize(value, min_val=-0.6213, max_val=1.4910)


def aloha_gripper_from_angular_inv(value):
    # Directly inverts the gripper_from_angular function.
    value = unnormalize(value, min_val=-0.6213, max_val=1.4910)
    return normalize(value, min_val=0.4, max_val=1.5)


class StageHead(nn.Module):
    """Small MLP that predicts the chunk-level stage distribution."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, hidden_state: Tensor) -> Tensor:
        first_linear = self.mlp[0]
        if isinstance(first_linear, nn.Linear):
            hidden_state = hidden_state.to(dtype=first_linear.weight.dtype)
        return self.mlp(hidden_state)


class StageFiLM(nn.Module):
    """Generate FiLM parameters from stage probabilities."""

    def __init__(self, condition_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(condition_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, output_dim * 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, z_cond: Tensor) -> tuple[Tensor, Tensor]:
        gamma_delta, beta = self.mlp(z_cond).chunk(2, dim=-1)
        gamma = 1.0 + gamma_delta
        return gamma, beta


class SmolVLAPolicy(PreTrainedPolicy):
    """Wrapper class around VLAFlowMatching model to train and run inference within LeRobot."""

    config_class = SmolVLAConfig
    name = "smolvla"
    _SMOOTHED_TRAIN_METRIC_KEYS = ("stage_loss", "stage_agreement", "l_stage")

    def __init__(
        self,
        config: SmolVLAConfig,
        **kwargs,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """

        super().__init__(config)
        config.validate_features()
        self.config = config
        self.register_buffer("num_updates", torch.zeros((), dtype=torch.long), persistent=False)
        self._stage_label_provider: StageLabelProvider | None = None
        self._last_stage_head_output: dict[str, Tensor | None] | None = None
        self._train_metric_windows = {
            key: deque(maxlen=self.config.train_metric_window_size)
            for key in self._SMOOTHED_TRAIN_METRIC_KEYS
        }
        self.init_rtc_processor()
        self.model = VLAFlowMatching(config, rtc_processor=self.rtc_processor)
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }
        self._stage_image_embedding_queues = {}
        self._stage_image_mask_queues = {}
        if self.config.n_obs_steps > 1:
            self._queues[OBS_STATE] = deque(maxlen=self.config.n_obs_steps)
            for key in self.config.image_features:
                self._queues[key] = deque(maxlen=self.config.n_obs_steps)
                if self.config.use_stage_head:
                    self._stage_image_embedding_queues[key] = deque(maxlen=self.config.n_obs_steps)
                    self._stage_image_mask_queues[key] = deque(maxlen=self.config.n_obs_steps)

    def init_rtc_processor(self):
        """Initialize RTC processor if RTC is enabled in config."""
        self.rtc_processor = None

        # Lets create processor if the config provided
        # If RTC is not enabled - we still can track the denoising data
        if self.config.rtc_config is not None:
            self.rtc_processor = RTCProcessor(self.config.rtc_config)

            # In case of calling init_rtc_processor after the model is created
            # We need to set the rtc_processor to the model
            # During the normal initialization process the model is not created yet
            model_value = getattr(self, "model", None)
            if model_value is not None:
                model_value.rtc_processor = self.rtc_processor

    def get_optim_params(self) -> dict:
        return self.parameters()

    def update(self) -> None:
        self.num_updates.add_(1)

    def _build_train_log_dict(
        self, alpha: float, stage_loss: float, stage_agreement: float, l_stage: float
    ) -> dict[str, float]:
        metrics = {
            "alpha": float(alpha),
            "stage_loss": float(stage_loss),
            "stage_agreement": float(stage_agreement),
            "l_stage": float(l_stage),
        }
        smoothed_metrics = {"alpha": metrics["alpha"]}
        for key in self._SMOOTHED_TRAIN_METRIC_KEYS:
            value = metrics[key]
            if not math.isfinite(value):
                continue
            metric_window = self._train_metric_windows[key]
            metric_window.append(value)
            smoothed_metrics[key] = sum(metric_window) / len(metric_window)
        return smoothed_metrics

    def compute_alpha(self) -> float:
        if not self.config.use_alpha_schedule:
            return float(self.config.alpha_start)

        if self.config.alpha_schedule not in {"linear", "cosine"}:
            raise ValueError(
                "`alpha_schedule` must be one of {'linear', 'cosine'} in this stage, "
                f"got '{self.config.alpha_schedule}'."
            )

        if self.config.alpha_warmup_steps <= 0:
            return float(self.config.alpha_end)

        t = int(self.num_updates.item())
        hold = self.config.alpha_hold_steps
        warm = self.config.alpha_warmup_steps

        if t < hold:
            alpha = self.config.alpha_start
        elif t >= hold + warm:
            alpha = self.config.alpha_end
        else:
            progress = float(t - hold) / float(warm)
            if self.config.alpha_schedule == "linear":
                mix = progress
            else:
                mix = 0.5 * (1.0 - math.cos(math.pi * progress))
            alpha = self.config.alpha_start + mix * (self.config.alpha_end - self.config.alpha_start)

        alpha_min = min(self.config.alpha_start, self.config.alpha_end)
        alpha_max = max(self.config.alpha_start, self.config.alpha_end)
        return float(max(alpha_min, min(alpha, alpha_max)))

    def blend_stage_condition(self, z: Tensor | None, z_hat: Tensor | None, alpha: float) -> Tensor | None:
        return self.model.blend_stage_condition(z, z_hat, alpha)

    def _get_action_chunk(
        self,
        batch: dict[str, Tensor],
        noise: Tensor | None = None,
        image_embs: list[Tensor] | None = None,
        current_img_masks: list[Tensor] | None = None,
        stage_image_embs: list[Tensor] | None = None,
        cached_stage_img_masks: list[Tensor] | None = None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        # TODO: Check if this for loop is needed.
        # Context: In fact, self.queues contains only ACTION field, and in inference, we don't have action in the batch
        # In the case of offline inference, we have the action in the batch
        # that why without the k != ACTION check, it will raise an error because we are trying to stack
        # on an empty container.
        for k in batch:
            if k in self._queues and k != ACTION:
                batch[k] = torch.stack(list(self._queues[k]), dim=1)

        images, img_masks = self.prepare_images(batch)
        if current_img_masks is not None:
            img_masks = current_img_masks
        stage_images = None
        stage_img_masks = None
        if self.config.use_stage_head and self.config.n_obs_steps > 1:
            if stage_image_embs is not None and cached_stage_img_masks is not None:
                stage_img_masks = cached_stage_img_masks
            else:
                stage_images, stage_img_masks = self.prepare_images(batch, use_history=True)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]

        actions = self.model.sample_actions(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            noise=noise,
            image_embs=image_embs,
            stage_images=stage_images,
            stage_img_masks=stage_img_masks,
            stage_image_embs=stage_image_embs,
            **kwargs,
        )
        self._cache_inference_stage_head_output()

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        if self.config.adapt_to_pi_aloha:
            actions = self._pi_aloha_encode_actions(actions)

        return actions

    def _prepare_batch(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])

        return batch

    @torch.no_grad()
    def predict_action_chunk(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        self.eval()

        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        image_embs, current_img_masks, stage_image_embs, cached_stage_img_masks = (
            self._maybe_update_inference_image_embedding_cache(batch)
        )

        actions = self._get_action_chunk(
            batch,
            noise,
            image_embs=image_embs,
            current_img_masks=current_img_masks,
            stage_image_embs=stage_image_embs,
            cached_stage_img_masks=cached_stage_img_masks,
            **kwargs,
        )
        return actions

    @torch.no_grad()
    def select_action(
        self, batch: dict[str, Tensor], noise: Tensor | None = None, **kwargs: Unpack[ActionSelectKwargs]
    ) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """

        assert not self._rtc_enabled(), (
            "RTC is not supported for select_action, use it with predict_action_chunk"
        )

        self.eval()
        batch = self._prepare_batch(batch)
        self._queues = populate_queues(self._queues, batch, exclude_keys=[ACTION])
        image_embs, current_img_masks, stage_image_embs, cached_stage_img_masks = (
            self._maybe_update_inference_image_embedding_cache(batch)
        )

        if self._check_get_actions_condition():
            actions = self._get_action_chunk(
                batch,
                noise,
                image_embs=image_embs,
                current_img_masks=current_img_masks,
                stage_image_embs=stage_image_embs,
                cached_stage_img_masks=cached_stage_img_masks,
            )

            # `self.predict_action_chunk` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            action_steps = self._resolve_inference_action_steps()
            self._queues[ACTION].extend(actions.transpose(0, 1)[:action_steps])
            self._cache_adaptive_action_steps(action_steps, device=actions.device)

        return self._queues[ACTION].popleft()

    def _check_get_actions_condition(self) -> bool:
        return len(self._queues[ACTION]) == 0

    def _rtc_enabled(self) -> bool:
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    def _resolve_inference_action_steps(self) -> int:
        if not self.config.use_adaptive_action_steps:
            return self.config.n_action_steps

        max_steps = self.config.adaptive_action_steps_max or self.config.n_action_steps
        max_steps = min(max_steps, self.config.n_action_steps, self.config.chunk_size)
        min_steps = min(self.config.adaptive_action_steps_min, max_steps)
        p_local = self._get_last_inference_p_local()
        if p_local is None:
            return max_steps

        p_local = max(0.0, min(1.0, p_local))
        action_steps = round(max_steps - p_local * (max_steps - min_steps))
        return int(max(min_steps, min(max_steps, action_steps)))

    def _get_last_inference_p_local(self) -> float | None:
        if self._last_stage_head_output is None:
            return None

        stage_probs = self._last_stage_head_output.get("stage_probs")
        if stage_probs is None or stage_probs.ndim != 2 or stage_probs.shape[-1] < 2:
            return None

        return float(stage_probs[:, 1].mean().item())

    def _cache_adaptive_action_steps(self, action_steps: int, device: torch.device) -> None:
        if self._last_stage_head_output is None:
            return
        self._last_stage_head_output["adaptive_action_steps"] = torch.tensor(
            action_steps, dtype=torch.long, device=device
        )

    def _should_cache_inference_image_embeddings(self) -> bool:
        if not (self.config.use_stage_head and self.config.n_obs_steps > 1):
            return False
        if not hasattr(self.model, "embed_images"):
            return False

        should_use_stage_condition = getattr(self.model, "should_use_stage_condition_at_inference", None)
        if callable(should_use_stage_condition):
            return bool(should_use_stage_condition())
        return bool(self.config.use_stage_condition_at_inference)

    def _maybe_update_inference_image_embedding_cache(
        self, batch: dict[str, Tensor]
    ) -> tuple[list[Tensor] | None, list[Tensor] | None, list[Tensor] | None, list[Tensor] | None]:
        if not self._should_cache_inference_image_embeddings():
            return None, None, None, None

        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]
        if len(present_img_keys) == 0 or len(missing_img_keys) > 0:
            return None, None, None, None

        current_images, current_img_masks = self.prepare_images(batch)
        if len(current_images) != len(present_img_keys):
            return None, None, None, None

        current_image_embs = self.model.embed_images(current_images)
        if len(current_image_embs) != len(present_img_keys):
            raise RuntimeError(
                "Image embedding cache expected one embedding tensor per present image feature, "
                f"got {len(current_image_embs)} embeddings for {len(present_img_keys)} features."
            )

        for key, image_emb, img_mask in zip(
            present_img_keys, current_image_embs, current_img_masks, strict=False
        ):
            embedding_queue = self._stage_image_embedding_queues.get(key)
            mask_queue = self._stage_image_mask_queues.get(key)
            if embedding_queue is None or mask_queue is None:
                return None, None, None, None

            image_emb = image_emb.detach()
            img_mask = img_mask.detach()
            if len(embedding_queue) != embedding_queue.maxlen:
                while len(embedding_queue) != embedding_queue.maxlen:
                    embedding_queue.append(image_emb)
                    mask_queue.append(img_mask)
            else:
                embedding_queue.append(image_emb)
                mask_queue.append(img_mask)

        stage_image_embs, stage_img_masks = self._get_cached_stage_image_embeddings(present_img_keys)
        return current_image_embs, current_img_masks, stage_image_embs, stage_img_masks

    def _get_cached_stage_image_embeddings(
        self, image_keys: list[str]
    ) -> tuple[list[Tensor] | None, list[Tensor] | None]:
        stage_image_embs = []
        stage_img_masks = []
        for key in image_keys:
            embedding_queue = self._stage_image_embedding_queues.get(key)
            mask_queue = self._stage_image_mask_queues.get(key)
            if (
                embedding_queue is None
                or mask_queue is None
                or len(embedding_queue) != self.config.n_obs_steps
                or len(mask_queue) != self.config.n_obs_steps
            ):
                return None, None
            stage_image_embs.extend(list(embedding_queue))
            stage_img_masks.extend(list(mask_queue))
        return stage_image_embs, stage_img_masks

    def _get_stage_label_provider(self) -> StageLabelProvider:
        if not self.config.use_stage_labels:
            raise RuntimeError("Stage-label provider is only available when `use_stage_labels=True`.")

        if self.config.stage_label_path is None or self.config.stage_label_path.strip() == "":
            raise ValueError(
                "`use_stage_labels=True` requires `stage_label_path` to point to a valid parquet file."
            )

        if self._stage_label_provider is None:
            self._stage_label_provider = StageLabelProvider(
                stage_label_path=self.config.stage_label_path,
                fallback_probs=self.config.stage_label_fallback_probs,
                cache_in_memory=self.config.stage_label_cache_in_memory,
            )
        return self._stage_label_provider

    def _maybe_get_batch_stage_probs(self, batch: dict[str, Tensor], device: torch.device) -> Tensor | None:
        if not self.config.use_stage_labels:
            return None

        if "index" not in batch:
            raise KeyError(
                "`use_stage_labels=True` requires `batch['index']` so SmolVLA can look up stage soft labels."
            )

        provider = self._get_stage_label_provider()
        return provider.get_batch_stage_probs(batch["index"]).to(device=device, dtype=torch.float32)

    @property
    def last_stage_head_output(self) -> dict[str, Tensor | None] | None:
        return self._last_stage_head_output

    def _empty_stage_head_output_cache(self) -> dict[str, Tensor | None]:
        return {
            "stage_logits": None,
            "stage_probs": None,
            "stage_context": None,
            "stage_label_probs": None,
            "stage_loss": None,
            "stage_loss_per_sample": None,
            "l_stage": None,
            "alpha": None,
            "z_cond": None,
            "action_hidden": None,
            "conditioned_action_hidden": None,
            "action_pred": None,
            "action_target": None,
            "l_fm": None,
            "l_fm_per_sample": None,
            "l_large": None,
            "l_large_per_sample": None,
            "l_small": None,
            "l_small_per_sample": None,
            "l_action": None,
            "l_action_per_sample": None,
            "action_loss": None,
            "action_loss_per_sample": None,
            "total_loss": None,
            "total_loss_per_sample": None,
            "inference_p_global_mean": None,
            "inference_p_local_mean": None,
            "adaptive_action_steps": None,
        }

    def _cache_inference_stage_head_output(self) -> None:
        inference_outputs = self.model.last_inference_stage_output
        if inference_outputs is None:
            self._last_stage_head_output = None
            return

        cached_outputs = self._empty_stage_head_output_cache()
        stage_outputs = inference_outputs.get("stage_outputs")
        if stage_outputs is not None:
            cached_outputs["stage_logits"] = stage_outputs["stage_logits"].detach()
            cached_outputs["stage_probs"] = stage_outputs["stage_probs"].detach()
            cached_outputs["stage_context"] = stage_outputs["stage_context"].detach()
            cached_outputs["inference_p_global_mean"] = stage_outputs["stage_probs"][:, 0].mean().detach()
            cached_outputs["inference_p_local_mean"] = stage_outputs["stage_probs"][:, 1].mean().detach()

        z_cond = inference_outputs.get("z_cond")
        if z_cond is not None:
            cached_outputs["z_cond"] = z_cond.detach()

        self._last_stage_head_output = cached_outputs

    def _cache_stage_head_output(
        self,
        model_outputs: ActionConditionedOutput | None,
        stage_label_probs: Tensor | None,
        stage_loss_per_sample: Tensor | None,
        fm_loss_per_sample: Tensor | None,
        l_large_per_sample: Tensor | None,
        l_small_per_sample: Tensor | None,
        l_action_per_sample: Tensor | None,
        action_loss_per_sample: Tensor | None,
        total_loss_per_sample: Tensor | None,
    ) -> None:
        stage_outputs = model_outputs.get("stage_outputs") if model_outputs is not None else None
        if (
            stage_outputs is None
            and model_outputs is None
            and stage_label_probs is None
            and stage_loss_per_sample is None
        ):
            self._last_stage_head_output = None
            return

        cached_outputs = self._empty_stage_head_output_cache()
        if stage_outputs is not None:
            cached_outputs["stage_logits"] = stage_outputs["stage_logits"].detach()
            cached_outputs["stage_probs"] = stage_outputs["stage_probs"].detach()
            cached_outputs["stage_context"] = stage_outputs["stage_context"].detach()
        if model_outputs is not None:
            if model_outputs.get("alpha") is not None:
                cached_outputs["alpha"] = model_outputs["alpha"].detach()
            if model_outputs.get("z_cond") is not None:
                cached_outputs["z_cond"] = model_outputs["z_cond"].detach()
            if model_outputs.get("action_hidden") is not None:
                cached_outputs["action_hidden"] = model_outputs["action_hidden"].detach()
            if model_outputs.get("conditioned_action_hidden") is not None:
                cached_outputs["conditioned_action_hidden"] = model_outputs[
                    "conditioned_action_hidden"
                ].detach()
            if model_outputs.get("action_pred") is not None:
                cached_outputs["action_pred"] = model_outputs["action_pred"].detach()
            if model_outputs.get("action_target") is not None:
                cached_outputs["action_target"] = model_outputs["action_target"].detach()
        if stage_label_probs is not None:
            cached_outputs["stage_label_probs"] = stage_label_probs.detach()
        if stage_loss_per_sample is not None:
            cached_outputs["stage_loss_per_sample"] = stage_loss_per_sample.detach()
            cached_outputs["stage_loss"] = stage_loss_per_sample.mean().detach()
            cached_outputs["l_stage"] = stage_loss_per_sample.mean().detach()
        if fm_loss_per_sample is not None:
            cached_outputs["l_fm_per_sample"] = fm_loss_per_sample.detach()
            cached_outputs["l_fm"] = fm_loss_per_sample.mean().detach()
        if l_large_per_sample is not None:
            cached_outputs["l_large_per_sample"] = l_large_per_sample.detach()
            cached_outputs["l_large"] = l_large_per_sample.mean().detach()
        if l_small_per_sample is not None:
            cached_outputs["l_small_per_sample"] = l_small_per_sample.detach()
            cached_outputs["l_small"] = l_small_per_sample.mean().detach()
        if l_action_per_sample is not None:
            cached_outputs["l_action_per_sample"] = l_action_per_sample.detach()
            cached_outputs["l_action"] = l_action_per_sample.mean().detach()
        if action_loss_per_sample is not None:
            cached_outputs["action_loss_per_sample"] = action_loss_per_sample.detach()
            cached_outputs["action_loss"] = action_loss_per_sample.mean().detach()
        if total_loss_per_sample is not None:
            cached_outputs["total_loss_per_sample"] = total_loss_per_sample.detach()
            cached_outputs["total_loss"] = total_loss_per_sample.mean().detach()
        self._last_stage_head_output = cached_outputs

    def forward(
        self, batch: dict[str, Tensor], noise=None, time=None, reduction: str = "mean"
    ) -> tuple[Tensor, dict]:
        """Do a full training forward pass to compute the loss.

        Args:
            batch: Training batch containing observations and actions.
            noise: Optional noise tensor for flow matching.
            time: Optional time tensor for flow matching.
            reduction: How to reduce the loss. Options:
                - "mean": Return scalar mean loss (default, backward compatible)
                - "none": Return per-sample losses of shape (batch_size,) for RA-BC weighting
        """
        if self.config.adapt_to_pi_aloha:
            batch[OBS_STATE] = self._pi_aloha_decode_state(batch[OBS_STATE])
            batch[ACTION] = self._pi_aloha_encode_actions_inv(batch[ACTION])

        images, img_masks = self.prepare_images(batch)
        stage_images = None
        stage_img_masks = None
        if self.config.use_stage_head and self.config.n_obs_steps > 1:
            stage_images, stage_img_masks = self.prepare_images(batch, use_history=True)
        state = self.prepare_state(batch)
        lang_tokens = batch[f"{OBS_LANGUAGE_TOKENS}"]
        lang_masks = batch[f"{OBS_LANGUAGE_ATTENTION_MASK}"]
        actions = self.prepare_action(batch)
        stage_label_probs = self._maybe_get_batch_stage_probs(batch, device=actions.device)
        actions_is_pad = batch.get("action_is_pad")
        alpha = self.compute_alpha()
        losses, model_outputs = self.model.forward(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            actions,
            noise,
            time,
            stage_condition_probs=stage_label_probs,
            stage_images=stage_images,
            stage_img_masks=stage_img_masks,
            alpha=alpha,
        )
        stage_outputs = model_outputs.get("stage_outputs") if model_outputs is not None else None
        stage_loss_per_sample = None
        stage_agreement_value = math.nan
        stage_loss_value = 0.0
        if stage_outputs is not None:
            stage_probs = stage_outputs["stage_probs"]
            if stage_label_probs is not None:
                stage_agreement = (stage_label_probs.argmax(dim=-1) == stage_probs.argmax(dim=-1)).to(
                    dtype=torch.float32
                )
                stage_agreement_value = stage_agreement.mean().item()
            if stage_label_probs is not None:
                stage_loss_per_sample = soft_target_cross_entropy(
                    stage_outputs["stage_logits"], stage_label_probs
                )
        original_action_dim = self.config.action_feature.shape[0]
        losses = losses[:, :, :original_action_dim]

        if actions_is_pad is not None:
            in_episode_bound = ~actions_is_pad
            losses = losses * in_episode_bound.unsqueeze(-1)

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        fm_loss_per_sample = losses.mean(dim=(1, 2))
        l_large_per_sample = None
        l_small_per_sample = None
        action_multiscale_per_sample = None

        if (
            self.config.use_stage_conditioned_action
            and self.config.use_multiscale_action_loss
            and model_outputs is not None
            and model_outputs.get("z_cond") is not None
            and model_outputs.get("l_large_elementwise") is not None
            and model_outputs.get("l_small_elementwise") is not None
        ):
            z_cond = model_outputs["z_cond"]
            l_large_elementwise = model_outputs["l_large_elementwise"][:, :, :original_action_dim]
            l_small_elementwise = model_outputs["l_small_elementwise"][:, :, :original_action_dim]
            if actions_is_pad is not None:
                l_large_elementwise = l_large_elementwise * in_episode_bound.unsqueeze(-1)
                l_small_elementwise = l_small_elementwise * in_episode_bound.unsqueeze(-1)

            l_large_elementwise = l_large_elementwise[:, :, : self.config.max_action_dim]
            l_small_elementwise = l_small_elementwise[:, :, : self.config.max_action_dim]
            l_large_per_sample = l_large_elementwise.mean(dim=(1, 2))
            l_small_per_sample = l_small_elementwise.mean(dim=(1, 2))
            action_multiscale_per_sample = (
                z_cond[:, 0] * l_large_per_sample + z_cond[:, 1] * l_small_per_sample
            )
            action_loss_per_sample = (
                fm_loss_per_sample + self.config.action_multiscale_loss_weight * action_multiscale_per_sample
            )
        else:
            action_loss_per_sample = fm_loss_per_sample

        if stage_loss_per_sample is not None:
            total_loss_per_sample = (
                action_loss_per_sample + self.config.stage_loss_weight * stage_loss_per_sample
            )
        else:
            total_loss_per_sample = action_loss_per_sample
        self._cache_stage_head_output(
            model_outputs,
            stage_label_probs,
            stage_loss_per_sample,
            fm_loss_per_sample,
            l_large_per_sample,
            l_small_per_sample,
            action_multiscale_per_sample,
            action_loss_per_sample,
            total_loss_per_sample,
        )
        if stage_loss_per_sample is not None:
            stage_loss_value = stage_loss_per_sample.mean().item()
        l_stage_value = stage_loss_value
        loss_dict = self._build_train_log_dict(
            alpha=alpha,
            stage_loss=stage_loss_value,
            stage_agreement=stage_agreement_value,
            l_stage=l_stage_value,
        )

        if reduction == "none":
            return total_loss_per_sample, loss_dict
        else:
            loss = total_loss_per_sample.mean()
            return loss, loss_dict

    def _get_image_valid_mask(
        self,
        batch: dict[str, Tensor],
        key: str,
        bsize: int,
        device: torch.device,
        num_frames: int | None = None,
    ) -> Tensor:
        if f"{key}_padding_mask" in batch:
            valid_mask = batch[f"{key}_padding_mask"].bool()
        elif f"{key}_is_pad" in batch:
            valid_mask = ~batch[f"{key}_is_pad"].bool()
        else:
            shape = (bsize, num_frames) if num_frames is not None else (bsize,)
            return torch.ones(shape, dtype=torch.bool, device=device)

        return valid_mask.to(device=device)

    def prepare_images(self, batch, use_history: bool = False):
        """Apply SmolVLA preprocessing to the images, like resizing to 224x224 and padding to keep aspect ratio, and
        convert pixel range from [0.0, 1.0] to [-1.0, 1.0] as requested by SigLIP.
        """
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]
        missing_img_keys = [key for key in self.config.image_features if key not in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. (batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )
        # Preprocess image features present in the batch
        max_frames_per_camera = 1
        for key in present_img_keys:
            raw_img = batch[key]
            bsize = raw_img.shape[0]
            device = raw_img.device
            if raw_img.ndim == 5:
                num_frames = raw_img.shape[1]
                valid_mask = self._get_image_valid_mask(batch, key, bsize, device, num_frames=num_frames)
                if use_history:
                    frame_imgs = list(raw_img.unbind(dim=1))
                    max_frames_per_camera = max(max_frames_per_camera, len(frame_imgs))
                    if valid_mask.ndim == 2:
                        frame_masks = list(valid_mask.unbind(dim=1))
                    else:
                        frame_masks = [valid_mask] * len(frame_imgs)
                else:
                    frame_imgs = [raw_img[:, -1, :, :, :]]
                    frame_masks = [valid_mask[:, -1]] if valid_mask.ndim == 2 else [valid_mask]
            else:
                valid_mask = self._get_image_valid_mask(batch, key, bsize, device)
                frame_imgs = [raw_img]
                frame_masks = [valid_mask]

            for img, mask in zip(frame_imgs, frame_masks, strict=False):
                if self.config.resize_imgs_with_padding is not None:
                    img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

                # Normalize from range [0,1] to [-1,1] as expected by SigLIP.
                img = img * 2.0 - 1.0

                images.append(img)
                img_masks.append(mask)

        # Create image features not present in the batch
        # as fully 0 padded images.
        for num_empty_cameras in range(len(missing_img_keys)):
            if num_empty_cameras >= self.config.empty_cameras:
                break
            for _ in range(max_frames_per_camera if use_history else 1):
                img = torch.ones_like(images[0]) * -1
                mask = torch.zeros_like(img_masks[0])
                images.append(img)
                img_masks.append(mask)
        return images, img_masks

    def _pi_aloha_decode_state(self, state):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            state[:, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            state[:, motor_idx] = aloha_gripper_to_angular(state[:, motor_idx])
        return state

    def _pi_aloha_encode_actions(self, actions):
        # Flip the joints.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular(actions[:, :, motor_idx])
        return actions

    def _pi_aloha_encode_actions_inv(self, actions):
        # Flip the joints again.
        for motor_idx in [1, 2, 8, 9]:
            actions[:, :, motor_idx] *= -1
        # Reverse the gripper transformation that is being applied by the Aloha runtime.
        for motor_idx in [6, 13]:
            actions[:, :, motor_idx] = aloha_gripper_from_angular_inv(actions[:, :, motor_idx])
        return actions

    def prepare_state(self, batch):
        """Pad state"""
        state = batch[OBS_STATE][:, -1, :] if batch[OBS_STATE].ndim > 2 else batch[OBS_STATE]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action"""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    def _get_default_peft_targets(self) -> dict[str, any]:
        """Return default PEFT target modules for SmolVLA fine-tuning."""
        common_projections = (
            "state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out"
        )
        head_projections = r"stage_head\.mlp\.(0|2)|stage_film\.mlp\.(0|2)"
        target_modules = (
            rf"(model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj|model\.({common_projections})|"
            rf"model\.({head_projections}))"
        )
        return {
            "target_modules": target_modules,
            "modules_to_save": [],
        }

    def _validate_peft_config(self, peft_config) -> None:
        """Validate PEFT configuration for SmolVLA."""
        super()._validate_peft_config(peft_config)
        if not self.config.load_vlm_weights:
            import logging

            logging.warning(
                "Training SmolVLA from scratch using PEFT. This is unlikely to yield good results. "
                "Set `load_vlm_weights=True` to fine-tune the existing policy."
            )


def pad_tensor(tensor, max_len, pad_value=0):
    """
    Efficiently pads a tensor along sequence dimension to match max_len.

    Args:
        tensor (torch.Tensor): Shape (B, L, ...) or (B, L).
        max_len (int): Fixed sequence length.
        pad_value (int/float): Value for padding.

    Returns:
        torch.Tensor: Shape (B, max_len, ...) or (B, max_len).
    """
    b, d = tensor.shape[:2]

    # Create a padded tensor of max_len and copy the existing values
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor  # Efficient in-place copy

    return padded_tensor


class VLAFlowMatching(nn.Module):
    """
    SmolVLA

    [Paper]()

    Designed by Hugging Face.
    ┌──────────────────────────────┐
    │                 actions      │
    │                    ▲         │
    │ ┌─────────┐      ┌─|────┐    │
    │ |         │────► │      │    │
    │ |         │ kv   │      │    │
    │ |         │────► │Action│    │
    │ |   VLM   │cache │Expert│    |
    │ │         │────► |      │    │
    │ │         │      │      │    │
    │ └▲──▲───▲─┘      └───▲──┘    |
    │  │  |   |            │       |
    │  |  |   |          noise     │
    │  │  │ state                  │
    │  │ language tokens           │
    │  image(s)                    │
    └──────────────────────────────┘
    """

    def __init__(self, config: SmolVLAConfig, rtc_processor: RTCProcessor | None = None):
        super().__init__()
        self.config = config

        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
            device=self.config.device if self.config.device is not None else "auto",
        )
        self.state_proj = nn.Linear(
            self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size
        )
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)

        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
        )
        self.stage_head = None
        if self.config.use_stage_head:
            self.stage_head = StageHead(
                self.vlm_with_expert.config.text_config.hidden_size,
                self.config.stage_head_hidden_dim,
            )
        self.stage_film = None
        if self.config.use_stage_conditioned_action:
            self.stage_film = StageFiLM(
                condition_dim=2,
                hidden_dim=self.config.stage_condition_hidden_dim,
                output_dim=self.vlm_with_expert.expert_hidden_size,
            )

        self.set_requires_grad()
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )

        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)
        self.prefix_length = self.config.prefix_length
        self.rtc_processor = rtc_processor
        self._last_inference_stage_output: dict[str, object] | None = None

        # Compile model if requested
        if config.compile_model:
            torch.set_float32_matmul_precision("high")
            self.sample_actions = torch.compile(self.sample_actions, mode=config.compile_mode)
            self.forward = torch.compile(self.forward, mode=config.compile_mode)

    def _rtc_enabled(self):
        return self.config.rtc_config is not None and self.config.rtc_config.enabled

    @property
    def last_inference_stage_output(self) -> dict[str, object] | None:
        return self._last_inference_stage_output

    def set_requires_grad(self):
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape, device):
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def embed_images(self, images) -> list[Tensor]:
        """Embed image tensors in shape-compatible batches while preserving prefix order."""
        if len(images) == 0:
            return []

        image_embs: list[Tensor | None] = [None] * len(images)
        image_groups: dict[tuple[torch.device, torch.dtype, tuple[int, ...]], list[int]] = {}
        for image_idx, image in enumerate(images):
            group_key = (image.device, image.dtype, tuple(image.shape[1:]))
            image_groups.setdefault(group_key, []).append(image_idx)

        for image_indices in image_groups.values():
            batched_images = torch.cat([images[image_idx] for image_idx in image_indices], dim=0)
            batched_embs = self.vlm_with_expert.embed_image(batched_images)

            # Normalize image embeddings.
            img_emb_dim = batched_embs.shape[-1]
            batched_embs = batched_embs * torch.tensor(
                img_emb_dim**0.5, dtype=batched_embs.dtype, device=batched_embs.device
            )

            split_sizes = [images[image_idx].shape[0] for image_idx in image_indices]
            for image_idx, image_emb in zip(
                image_indices, batched_embs.split(split_sizes, dim=0), strict=False
            ):
                image_embs[image_idx] = image_emb

        return [image_emb for image_emb in image_embs if image_emb is not None]

    def encode_prefix_context(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        *,
        use_cache: bool,
        image_embs: list[Tensor] | None = None,
    ) -> tuple[Tensor, Tensor, object]:
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state, image_embs=image_embs
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        prefix_outputs, past_key_values = self.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=use_cache,
            fill_kv_cache=True,
        )
        prefix_hidden_states = prefix_outputs[0]
        return prefix_hidden_states, prefix_pad_masks, past_key_values

    def embed_prefix(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state: torch.Tensor = None,
        image_embs: list[Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for SmolVLM transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []
        if image_embs is None:
            image_embs = self.embed_images(images)
        if len(image_embs) != len(img_masks):
            raise ValueError(
                "Image embeddings and image masks must have the same length, got "
                f"{len(image_embs)} embeddings and {len(img_masks)} masks."
            )

        for img_emb, img_mask in zip(image_embs, img_masks, strict=False):
            bsize = img_emb.shape[0]
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=img_emb.device)
                    )
                    .unsqueeze(0)
                    .expand(bsize, -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)

            num_img_embs = img_emb.shape[1]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)

            att_masks += [0] * (num_img_embs)
            if self.add_image_special_tokens:
                image_end_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.image_end_token.to(device=img_emb.device)
                    )
                    .unsqueeze(0)
                    .expand(bsize, -1, -1)
                )
                image_end_mask = torch.ones_like(
                    image_end_token[:, :, 0], dtype=torch.bool, device=image_end_token.device
                )
                embs.append(image_end_token)
                pad_masks.append(image_end_mask)
                att_masks += [0] * (image_end_mask.shape[1])
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        # Normalize language embeddings
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device

        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)

        # Set attention masks so that image and language inputs do not attend to state or actions
        att_masks += [1] * (states_seq_len)
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype
        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)  # swish == silu
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] * self.config.chunk_size
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))
        return embs, pad_masks, att_masks

    def pool_stage_context(self, prefix_hidden_states: Tensor, prefix_pad_masks: Tensor) -> Tensor:
        """Pool prefix hidden states into a single representation for stage prediction."""
        if self.config.stage_pooling == "last_state_token":
            # State tokens are appended at the end of the unpadded prefix sequence.
            last_valid_indices = prefix_pad_masks.sum(dim=1).clamp_min(1).to(dtype=torch.long) - 1
            batch_indices = torch.arange(prefix_hidden_states.shape[0], device=prefix_hidden_states.device)
            return prefix_hidden_states[batch_indices, last_valid_indices]

        if self.config.stage_pooling == "mean":
            valid_mask = prefix_pad_masks.unsqueeze(-1).to(dtype=prefix_hidden_states.dtype)
            valid_count = valid_mask.sum(dim=1).clamp_min(1.0)
            return (prefix_hidden_states * valid_mask).sum(dim=1) / valid_count

        raise ValueError(
            f"Unsupported stage pooling '{self.config.stage_pooling}'. Expected 'last_state_token' or 'mean'."
        )

    def predict_stage(self, prefix_hidden_states: Tensor, prefix_pad_masks: Tensor) -> StageHeadOutput | None:
        if not self.config.use_stage_head or self.stage_head is None:
            return None

        stage_context = self.pool_stage_context(prefix_hidden_states, prefix_pad_masks)
        stage_logits = self.stage_head(stage_context).to(dtype=torch.float32)
        stage_probs = torch.softmax(stage_logits, dim=-1)
        return {
            "stage_context": stage_context,
            "stage_logits": stage_logits,
            "stage_probs": stage_probs,
        }

    def should_use_stage_condition_at_inference(self) -> bool:
        if not self.config.use_stage_condition_at_inference:
            return False
        if self.config.disable_stage_condition_for_eval:
            return False
        if self.config.inference_stage_source != "predicted":
            raise ValueError(
                "`inference_stage_source` only supports 'predicted' in this stage, got "
                f"'{self.config.inference_stage_source}'."
            )
        return (
            self.config.use_stage_head
            and self.stage_head is not None
            and self.config.use_stage_conditioned_action
            and self.stage_film is not None
        )

    def get_inference_stage_condition(
        self, prefix_hidden_states: Tensor, prefix_pad_masks: Tensor
    ) -> tuple[Tensor | None, StageHeadOutput | None]:
        self._last_inference_stage_output = None
        if not self.should_use_stage_condition_at_inference():
            return None, None

        stage_outputs = self.predict_stage(prefix_hidden_states, prefix_pad_masks)
        if stage_outputs is None:
            return None, None

        z_hat = stage_outputs["stage_probs"]
        if z_hat.ndim != 2 or z_hat.shape[-1] != 2:
            raise ValueError(
                f"Predicted stage probabilities must have shape [batch_size, 2], got {tuple(z_hat.shape)}."
            )

        z_cond = z_hat.to(dtype=torch.float32)
        self._last_inference_stage_output = {
            "stage_outputs": stage_outputs,
            "z_cond": z_cond,
        }
        return z_cond, stage_outputs

    def blend_stage_condition(self, z: Tensor | None, z_hat: Tensor | None, alpha: float) -> Tensor | None:
        if z is None:
            return None

        z = z.to(dtype=torch.float32)
        if z_hat is None:
            return z

        z_hat_for_action = z_hat.detach() if self.config.detach_pred_stage_for_action else z_hat
        z_hat_for_action = z_hat_for_action.to(device=z.device, dtype=z.dtype)
        alpha_tensor = z.new_tensor(alpha)
        z_cond = (1.0 - alpha_tensor) * z + alpha_tensor * z_hat_for_action
        return z_cond

    def apply_stage_condition(
        self, action_hidden: Tensor, z_cond: Tensor | None
    ) -> tuple[Tensor, Tensor | None]:
        if not self.config.use_stage_conditioned_action or self.stage_film is None:
            return action_hidden, None

        if z_cond is None:
            z_cond = torch.full(
                (action_hidden.shape[0], 2), 0.5, dtype=torch.float32, device=action_hidden.device
            )
        else:
            z_cond = z_cond.to(device=action_hidden.device, dtype=torch.float32)

        gamma, beta = self.stage_film(z_cond)
        gamma = gamma.to(dtype=action_hidden.dtype).unsqueeze(1)
        beta = beta.to(dtype=action_hidden.dtype).unsqueeze(1)
        conditioned_hidden = gamma * action_hidden + beta
        return conditioned_hidden, z_cond

    def decompose_action_scales(self, action_chunk: Tensor) -> tuple[Tensor, Tensor]:
        if self.config.action_large_scale_mode != "fixed_lowpass":
            raise ValueError(f"Unsupported action large-scale mode '{self.config.action_large_scale_mode}'.")

        kernel = self.config.action_large_scale_kernel
        if kernel == 1:
            action_large = action_chunk
        else:
            action_chunk_t = action_chunk.transpose(1, 2)
            pad_left = (kernel - 1) // 2
            pad_right = kernel // 2
            padded_action = F.pad(action_chunk_t, (pad_left, pad_right), mode="replicate")
            action_large = F.avg_pool1d(padded_action, kernel_size=kernel, stride=1).transpose(1, 2)

        action_small = action_chunk - action_large
        return action_large, action_small

    def compute_multiscale_action_loss(
        self, action_pred: Tensor, action_target: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Compute large/small action-chunk losses after fixed low-pass decomposition."""
        action_pred_large, action_pred_small = self.decompose_action_scales(action_pred)
        action_target_large, action_target_small = self.decompose_action_scales(action_target)
        l_large = F.mse_loss(action_pred_large, action_target_large, reduction="none")
        l_small = F.mse_loss(action_pred_small, action_target_small, reduction="none")
        return l_large, l_small

    def forward(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        actions,
        noise=None,
        time=None,
        stage_condition_probs: Tensor | None = None,
        stage_images=None,
        stage_img_masks=None,
        alpha: float = 0.0,
    ) -> tuple[Tensor, ActionConditionedOutput | None]:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions
        prefix_out, prefix_pad_masks, past_key_values = self.encode_prefix_context(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            use_cache=True,
        )
        stage_outputs = None
        if self.config.use_stage_head and self.stage_head is not None:
            stage_prefix_out = prefix_out
            stage_prefix_pad_masks = prefix_pad_masks
            if stage_images is not None and stage_img_masks is not None:
                stage_prefix_out, stage_prefix_pad_masks, _ = self.encode_prefix_context(
                    stage_images,
                    stage_img_masks,
                    lang_tokens,
                    lang_masks,
                    state,
                    use_cache=False,
                )
            stage_outputs = self.predict_stage(stage_prefix_out, stage_prefix_pad_masks)
        z_hat = stage_outputs["stage_probs"] if stage_outputs is not None else None
        blended_stage_condition = self.blend_stage_condition(stage_condition_probs, z_hat, alpha)
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)
        conditioned_suffix_embs, z_cond = self.apply_stage_condition(suffix_embs, blended_stage_condition)
        model_outputs: ActionConditionedOutput | None = None
        if self.config.use_stage_conditioned_action:
            model_outputs = {
                "action_hidden": suffix_embs.to(dtype=torch.float32),
                "conditioned_action_hidden": conditioned_suffix_embs.to(dtype=torch.float32),
                # L_fm stays on the corrective field (u_t, v_t), while L_action is computed in action space.
                "action_target": actions,
                "alpha": u_t.new_tensor(alpha, dtype=torch.float32),
            }
            if z_cond is not None:
                model_outputs["z_cond"] = z_cond

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        suffix_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=suffix_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, conditioned_suffix_embs],
            use_cache=True,
            fill_kv_cache=False,
        )
        if stage_outputs is not None:
            if model_outputs is None:
                model_outputs = {"alpha": u_t.new_tensor(alpha, dtype=torch.float32)}
            model_outputs["stage_outputs"] = stage_outputs
        elif model_outputs is None:
            model_outputs = {"alpha": u_t.new_tensor(alpha, dtype=torch.float32)}
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        # Original openpi code, upcast attention output
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        action_pred = noise - v_t
        if model_outputs is not None and self.config.use_stage_conditioned_action:
            model_outputs["action_pred"] = action_pred
        if self.config.use_stage_conditioned_action and self.config.use_multiscale_action_loss:
            l_large_elementwise, l_small_elementwise = self.compute_multiscale_action_loss(
                action_pred, actions
            )
            if model_outputs is None:
                model_outputs = {}
            model_outputs["l_large_elementwise"] = l_large_elementwise
            model_outputs["l_small_elementwise"] = l_small_elementwise
        losses = F.mse_loss(u_t, v_t, reduction="none")
        return losses, model_outputs

    def sample_actions(
        self,
        images,
        img_masks,
        lang_tokens,
        lang_masks,
        state,
        noise=None,
        image_embs=None,
        stage_images=None,
        stage_img_masks=None,
        stage_image_embs=None,
        **kwargs: Unpack[ActionSelectKwargs],
    ) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        prefix_hidden_states, prefix_pad_masks, past_key_values = self.encode_prefix_context(
            images,
            img_masks,
            lang_tokens,
            lang_masks,
            state,
            use_cache=self.config.use_cache,
            image_embs=image_embs,
        )
        stage_prefix_hidden_states = prefix_hidden_states
        stage_prefix_pad_masks = prefix_pad_masks
        if self.should_use_stage_condition_at_inference() and stage_img_masks is not None and (
            stage_image_embs is not None or stage_images is not None
        ):
            stage_prefix_hidden_states, stage_prefix_pad_masks, _ = self.encode_prefix_context(
                stage_images,
                stage_img_masks,
                lang_tokens,
                lang_masks,
                state,
                use_cache=False,
                image_embs=stage_image_embs,
            )
        z_cond, _ = self.get_inference_stage_condition(stage_prefix_hidden_states, stage_prefix_pad_masks)
        num_steps = self.config.num_steps
        dt = -1.0 / num_steps

        x_t = noise
        for step in range(num_steps):
            time = 1.0 + step * dt
            time_tensor = torch.tensor(time, dtype=torch.float32, device=device).expand(bsize)

            def denoise_step_partial_call(input_x_t, current_timestep=time_tensor):
                return self.denoise_step(
                    x_t=input_x_t,
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    timestep=current_timestep,
                    z_cond=z_cond,
                )

            if self._rtc_enabled():
                inference_delay = kwargs.get("inference_delay")
                prev_chunk_left_over = kwargs.get("prev_chunk_left_over")
                execution_horizon = kwargs.get("execution_horizon")

                v_t = self.rtc_processor.denoise_step(
                    x_t=x_t,
                    prev_chunk_left_over=prev_chunk_left_over,
                    inference_delay=inference_delay,
                    time=time,
                    original_denoise_step_partial=denoise_step_partial_call,
                    execution_horizon=execution_horizon,
                )
            else:
                v_t = denoise_step_partial_call(x_t)

            x_t = x_t + dt * v_t

            if self.rtc_processor is not None and self.rtc_processor.is_debug_enabled():
                self.rtc_processor.track(time=time, x_t=x_t, v_t=v_t)

        return x_t

    def denoise_step(
        self,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
        z_cond: Tensor | None = None,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, timestep)
        conditioned_suffix_embs, _ = self.apply_stage_condition(suffix_embs, z_cond=z_cond)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]
        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        outputs_embeds, _ = self.vlm_with_expert.forward(
            attention_mask=full_att_2d_masks,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, conditioned_suffix_embs],
            use_cache=self.config.use_cache,
            fill_kv_cache=False,
        )
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        return v_t

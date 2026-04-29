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

from dataclasses import dataclass, field

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature, PreTrainedConfig
from lerobot.optim import AdamWConfig, CosineDecayWithWarmupSchedulerConfig
from lerobot.utils.constants import OBS_IMAGES

from ..rtc.configuration_rtc import RTCConfig


@PreTrainedConfig.register_subclass("smolvla")
@dataclass
class SmolVLAConfig(PreTrainedConfig):
    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Image preprocessing
    resize_imgs_with_padding: tuple[int, int] = (512, 512)

    # Add empty images. Used by smolvla_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False

    # Converts joint dimensions to relative values with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False

    # Tokenizer
    tokenizer_max_length: int = 48

    # Decoding
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"  # Select the VLM backbone.
    load_vlm_weights: bool = False  # Set to False in case of training the expert from scratch. True when init from pretrained SmolVLA weights

    add_image_special_tokens: bool = False  # Whether to use special image tokens around image features.

    attention_mode: str = "cross_attn"

    prefix_length: int = -1

    pad_language_to: str = "longest"  # "max_length"

    num_expert_layers: int = -1  # Less or equal to 0 is the default where the action expert has the same number of layers of VLM. Otherwise the expert have less layers.
    num_vlm_layers: int = 16  # Number of layers used in the VLM (first num_vlm_layers layers)
    self_attn_every_n_layers: int = 2  # Interleave SA layers each self_attn_every_n_layers
    expert_width_multiplier: float = 0.75  # The action expert hidden size (wrt to the VLM)

    min_period: float = 4e-3  # sensitivity range for the timestep used in sine-cosine positional encoding
    max_period: float = 4.0

    # Real-Time Chunking (RTC) configuration
    rtc_config: RTCConfig | None = None

    compile_model: bool = False  # Whether to use torch.compile for model optimization
    compile_mode: str = "max-autotune"  # Torch compile mode

    # Optional stage-label lookup during training.
    use_stage_labels: bool = False
    stage_label_path: str | None = None
    stage_label_fallback_probs: tuple[float, float] = (0.5, 0.5)
    stage_label_cache_in_memory: bool = True

    # Optional stage-head supervision on top of prefix hidden states.
    use_stage_head: bool = False
    stage_head_hidden_dim: int = 256
    stage_loss_weight: float = 0.1
    stage_pooling: str = "last_state_token"

    # Optional teacher stage conditioning on the action branch.
    use_stage_conditioned_action: bool = False
    stage_condition_mode: str = "film"
    stage_condition_hidden_dim: int = 128
    use_multiscale_action_loss: bool = True
    action_multiscale_loss_weight: float = 0.25
    action_large_scale_mode: str = "fixed_lowpass"
    action_large_scale_kernel: int = 5
    train_metric_window_size: int = 100
    use_alpha_schedule: bool = True
    alpha_schedule: str = "cosine"
    alpha_start: float = 0.0
    alpha_end: float = 1.0
    alpha_warmup_steps: int = 10_000
    alpha_hold_steps: int = 0
    detach_pred_stage_for_action: bool = True
    use_stage_condition_at_inference: bool = True
    inference_stage_source: str = "predicted"
    disable_stage_condition_for_eval: bool = False
    use_adaptive_action_steps: bool = False
    adaptive_action_steps_min: int = 2
    adaptive_action_steps_max: int | None = None
    adaptive_action_steps_source: str = "p_local"

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.n_obs_steps <= 0:
            raise ValueError(f"`n_obs_steps` must be positive, got {self.n_obs_steps}.")
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by smolvla for aloha real models. It is not ported yet in LeRobot."
            )

        if len(self.stage_label_fallback_probs) != 2:
            raise ValueError(
                "`stage_label_fallback_probs` must contain exactly 2 entries, got "
                f"{len(self.stage_label_fallback_probs)}."
            )

        fallback_sum = sum(self.stage_label_fallback_probs)
        if fallback_sum <= 0.0:
            raise ValueError(
                "`stage_label_fallback_probs` must sum to a positive value so it can represent a valid mixture."
            )
        if any(prob < 0.0 for prob in self.stage_label_fallback_probs):
            raise ValueError("`stage_label_fallback_probs` must be non-negative.")

        self.stage_label_fallback_probs = tuple(
            prob / fallback_sum for prob in self.stage_label_fallback_probs
        )

        if self.stage_head_hidden_dim <= 0:
            raise ValueError(f"`stage_head_hidden_dim` must be positive, got {self.stage_head_hidden_dim}.")
        if self.stage_loss_weight < 0.0:
            raise ValueError(f"`stage_loss_weight` must be non-negative, got {self.stage_loss_weight}.")
        if self.stage_pooling not in {"last_state_token", "mean"}:
            raise ValueError(
                f"`stage_pooling` must be one of {{'last_state_token', 'mean'}}, got '{self.stage_pooling}'."
            )
        if self.stage_condition_mode != "film":
            raise ValueError(
                f"`stage_condition_mode` only supports 'film' in this stage, got '{self.stage_condition_mode}'."
            )
        if self.stage_condition_hidden_dim <= 0:
            raise ValueError(
                f"`stage_condition_hidden_dim` must be positive, got {self.stage_condition_hidden_dim}."
            )
        if self.action_multiscale_loss_weight < 0.0:
            raise ValueError(
                "`action_multiscale_loss_weight` must be non-negative, got "
                f"{self.action_multiscale_loss_weight}."
            )
        if self.action_large_scale_mode != "fixed_lowpass":
            raise ValueError(
                "`action_large_scale_mode` only supports 'fixed_lowpass' in this stage, got "
                f"'{self.action_large_scale_mode}'."
            )
        if self.action_large_scale_kernel <= 0:
            raise ValueError(
                f"`action_large_scale_kernel` must be positive, got {self.action_large_scale_kernel}."
            )
        if self.train_metric_window_size <= 0:
            raise ValueError(
                f"`train_metric_window_size` must be positive, got {self.train_metric_window_size}."
            )
        if self.alpha_schedule not in {"linear", "cosine"}:
            raise ValueError(
                "`alpha_schedule` must be one of {'linear', 'cosine'} in this stage, "
                f"got '{self.alpha_schedule}'."
            )
        if self.alpha_hold_steps < 0:
            raise ValueError(f"`alpha_hold_steps` must be non-negative, got {self.alpha_hold_steps}.")
        if self.inference_stage_source != "predicted":
            raise ValueError(
                "`inference_stage_source` only supports 'predicted' in this stage, got "
                f"'{self.inference_stage_source}'."
            )
        if self.adaptive_action_steps_min <= 0:
            raise ValueError(
                f"`adaptive_action_steps_min` must be positive, got {self.adaptive_action_steps_min}."
            )
        if self.adaptive_action_steps_max is not None and self.adaptive_action_steps_max <= 0:
            raise ValueError(
                f"`adaptive_action_steps_max` must be positive, got {self.adaptive_action_steps_max}."
            )
        if self.use_adaptive_action_steps:
            adaptive_max = self.adaptive_action_steps_max or self.n_action_steps
            if adaptive_max > self.n_action_steps:
                raise ValueError(
                    "`adaptive_action_steps_max` cannot exceed `n_action_steps`, got "
                    f"{adaptive_max} > {self.n_action_steps}."
                )
            if self.adaptive_action_steps_min > adaptive_max:
                raise ValueError(
                    "`adaptive_action_steps_min` cannot exceed the adaptive action step maximum, got "
                    f"{self.adaptive_action_steps_min} > {adaptive_max}."
                )
        if self.adaptive_action_steps_source != "p_local":
            raise ValueError(
                "`adaptive_action_steps_source` only supports 'p_local' in this stage, got "
                f"'{self.adaptive_action_steps_source}'."
            )

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"{OBS_IMAGES}.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    def get_optimizer_preset(self) -> AdamWConfig:
        return AdamWConfig(
            lr=self.optimizer_lr,
            betas=self.optimizer_betas,
            eps=self.optimizer_eps,
            weight_decay=self.optimizer_weight_decay,
            grad_clip_norm=self.optimizer_grad_clip_norm,
        )

    def get_scheduler_preset(self):
        return CosineDecayWithWarmupSchedulerConfig(
            peak_lr=self.optimizer_lr,
            decay_lr=self.scheduler_decay_lr,
            num_warmup_steps=self.scheduler_warmup_steps,
            num_decay_steps=self.scheduler_decay_steps,
        )

    @property
    def observation_delta_indices(self) -> list:
        return list(range(1 - self.n_obs_steps, 1))

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

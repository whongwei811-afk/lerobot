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

    # Action-component branch
    enable_action_component_branch: bool = False
    enable_suffix_component_film: bool = False

    component_hidden_dim: int = 256
    component_conv_kernel_size: int = 3
    component_pooling: str = "mean"
    component_num_anchors: int = 4
    component_predictor_hidden_dim: int = 128
    suffix_component_film_hidden_dim: int = 128

    component_anchor_position_temperature: float = 0.25
    component_anchor_min_separation: float = 0.05

    component_teacher_min_weight: float = 0.1
    component_teacher_warmup_ratio: float = 0.1
    component_teacher_decay_ratio: float = 0.7
    component_teacher_schedule_steps: int | None = None

    loss_weight_recon: float = 0.25
    loss_weight_comp: float = 0.05
    loss_weight_reg_gate_entropy: float = 0.002
    loss_weight_reg_gate_balance: float = 0.02
    loss_weight_reg_trend_prior: float = 0.10
    loss_weight_reg_ref_mag: float = 0.0005
    loss_weight_reg_ref_center: float = 0.001

    def __post_init__(self):
        super().__post_init__()

        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by smolvla for aloha real models. It is not ported yet in LeRobot."
            )
        if self.enable_action_component_branch:
            if self.component_pooling not in {"mean", "max"}:
                raise ValueError("component_pooling must be one of {'mean', 'max'}")
            if self.component_conv_kernel_size < 1 or self.component_conv_kernel_size % 2 == 0:
                raise ValueError("component_conv_kernel_size must be a positive odd integer")
            if self.component_num_anchors < 2:
                raise ValueError("component_num_anchors must be >= 2")
            if not 0.0 <= self.component_teacher_min_weight <= 1.0:
                raise ValueError("component_teacher_min_weight must be in [0, 1]")
            if self.component_teacher_warmup_ratio < 0.0:
                raise ValueError("component_teacher_warmup_ratio must be >= 0")
            if self.component_teacher_decay_ratio < 0.0:
                raise ValueError("component_teacher_decay_ratio must be >= 0")
            if self.component_teacher_warmup_ratio + self.component_teacher_decay_ratio > 1.0:
                raise ValueError(
                    "component_teacher_warmup_ratio + component_teacher_decay_ratio must be <= 1"
                )
            if self.component_teacher_schedule_steps is not None and self.component_teacher_schedule_steps <= 0:
                raise ValueError("component_teacher_schedule_steps must be > 0 when set")
            if self.component_anchor_position_temperature <= 0.0:
                raise ValueError("component_anchor_position_temperature must be > 0")
            if self.component_anchor_min_separation < 0.0:
                raise ValueError("component_anchor_min_separation must be >= 0")
            if self.component_anchor_min_separation * (self.component_num_anchors - 1) > 1.0:
                raise ValueError("component_anchor_min_separation is too large for component_num_anchors")
            if self.chunk_size < self.component_num_anchors:
                raise ValueError("chunk_size must be >= component_num_anchors")

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
        return [0]

    @property
    def action_delta_indices(self) -> list:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

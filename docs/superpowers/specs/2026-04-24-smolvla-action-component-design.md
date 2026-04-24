# SmolVLA Action Component Branch Design

Date: 2026-04-24
Status: Draft for review

## Goal

Extend `lerobot/src/lerobot/policies/smolvla` with an auxiliary action-component decomposition path that improves suffix conditioning during training while preserving the existing SmolVLA main branch and inference flow as much as possible.

The design must:

- keep the original branch 1 semantics intact
- add branches 2, 3, and 4 during training
- keep only branches 1 and 3 during inference
- use chunk-level action-component labels `Z = (p_trend, p_refined)`
- derive multimodal context `e` from the final prefix-layer state token
- make the new behavior opt-in and backward compatible by default

## Confirmed Decisions

### Branch behavior

- Branch 1 remains the current SmolVLA path:
  - visual input `ob.i`
  - state input `ob.s`
  - instruction/task input
  - VLM prefix processing
  - per-layer KV transfer into the action expert
  - action prediction through the existing flow-matching denoiser
- Training uses branches 1, 2, 3, and 4.
- Inference uses only branches 1 and 3.

### Action-component representation

- `Z = (p_trend, p_refined)` is chunk-level, not timestep-level.
- Branch 2 consumes ground-truth future actions from the training batch, not noisy actions `x_t`.
- `A_trend` is produced from a fixed number `K` of anchors.
- Anchor values are predicted per sample.
- Anchor time positions are learnable parameters.
- `A_refined` is predicted across the full action chunk.
- The reconstruction is:
  - `A_hat = p_trend * A_trend + p_refined * A_refined`

### Multimodal context

- `e` is the final prefix-layer hidden state of the state token.
- `e` is not a pool over all prefix tokens.

### Training-vs-inference conditioning

- During training, branch 4 fuses teacher labels `Z` and predicted labels `z_hat` to modulate `suffix_embs`.
- During inference, branch 4 is removed and `z_hat` alone modulates `suffix_embs`.

### Fusion schedule

- Use cosine decay for teacher reliance.
- Keep a minimum teacher weight of `0.1`.
- Recommended schedule:
  - first `10%` of total steps: `w_Z = 1.0`, `w_z_hat = 0.0`
  - middle `70%`: cosine decay from `1.0` to `0.1`
  - final `20%`: `w_Z = 0.1`, `w_z_hat = 0.9`

## Architectural Design

## Section 1: Overall Architecture

The model keeps the current SmolVLA main branch untouched and adds an auxiliary decomposition-and-conditioning stack around the suffix path.

### Training path

1. Prefix branch:
   - `images + language + state -> VLM prefix`
   - compute prefix KV caches as in the current implementation
   - extract `e` from the final prefix-layer state token
2. Action decomposition branch:
   - `ground-truth actions -> temporal encoder`
   - modulate encoded action features with FiLM generated from `e`
   - decompose into:
     - chunk-level gate `Z = (p_trend, p_refined)`
     - trend anchors and interpolated `A_trend`
     - residual branch `A_refined`
     - reconstructed actions `A_hat`
3. Component prediction branch:
   - `e -> z_hat`
4. Fusion branch:
   - combine `Z` and `z_hat` using scheduled teacher/prediction weights
   - use the blended component signal to FiLM-modulate `suffix_embs`
5. Action expert:
   - keep the current denoising and action generation path
   - run flow matching with conditioned `suffix_embs`

### Inference path

1. Prefix branch remains unchanged and still provides the KV cache.
2. Extract `e` from the final prefix state token.
3. Predict `z_hat` from `e`.
4. FiLM-modulate `suffix_embs` with `z_hat`.
5. Run the existing denoising/action-expert path.

Inference does not use:

- branch 2 decomposition
- branch 4 teacher/prediction fusion
- any direct access to ground-truth future actions

## Section 2: Module Boundaries And Tensor Flow

### New outputs from the VLM wrapper

`SmolVLMWithExpertModel` should expose an optional way to return the final prefix state-token hidden state `e` without changing the current KV-cache semantics.

Recommended interface style:

- keep the current forward path as default
- add a flag or structured output that can optionally return:
  - `prefix_state_token: Tensor` with shape `[B, H_vlm]`

### New modules inside `VLAFlowMatching`

#### 1. Action chunk encoder

Purpose:

- encode real future actions into a compact chunk summary for decomposition

Suggested structure:

- linear projection from action dimension to auxiliary hidden dimension
- 1D temporal convolution
- temporal pooling

Input shape:

- `[B, T, A]`

Output shape:

- `[B, H_aux]`

#### 2. Action FiLM from `e`

Purpose:

- modulate the encoded action summary using multimodal context

Structure:

- MLP from `e` to FiLM scale and shift
- FiLM over the action chunk summary

Input:

- encoded action summary `[B, H_aux]`
- multimodal summary `e [B, H_vlm]`

Output:

- `h [B, H_aux]`

#### 3. Action decomposition head

Purpose:

- produce chunk-level decomposition labels and reconstruct actions

Outputs:

- gate logits -> `softmax` -> `Z = (p_trend, p_refined)` with shape `[B, 2]`
- trend anchor values with shape `[B, K, A]`
- learnable anchor positions with shape `[K]`
- interpolated trend `A_trend [B, T, A]`
- residual branch `A_refined [B, T, A]`
- reconstruction `A_hat [B, T, A]`

Notes:

- anchor positions are global learnable parameters, not sample-wise predictions
- `softmax` parameterization guarantees non-negative gate values summing to 1

#### 4. Component predictor

Purpose:

- predict the chunk-level component label directly from multimodal context

Input:

- `e [B, H_vlm]`

Output:

- `z_hat [B, 2]`

#### 5. Suffix FiLM

Purpose:

- modulate `suffix_embs` before the action expert consumes them

Input:

- training: blended component signal from `Z` and `z_hat`
- inference: `z_hat` only

Output:

- FiLM scale and shift broadcast over suffix tokens

Because the label is chunk-level, FiLM should be generated as `[B, 1, H_expert]` and then broadcast over suffix length `T`.

## Section 3: Losses And Training Schedule

The total training objective is grouped into four top-level terms:

- `L_flow`: original flow-matching objective
- `L_recon`: self-supervised reconstruction for the decomposition branch
- `L_comp`: branch 3 prediction-to-teacher fitting loss
- `L_reg`: all new regularization losses

### Total loss

`L_total = L_flow + lambda_recon * L_recon + lambda_comp * L_comp + L_reg`

### 1. `L_flow`

Unchanged from the current SmolVLA implementation:

- sample `x_t`
- predict denoising velocity `v_t`
- optimize the original flow-matching target

This remains the dominant optimization objective.

### 2. `L_recon`

Self-supervised reconstruction loss for the decomposition branch:

- `A_hat = p_trend * A_trend + p_refined * A_refined`
- optimize `MSE(A_hat, actions)`

### 3. `L_comp`

Fit branch 3 predictions to the chunk-level teacher labels:

- optimize `MSE(z_hat, stopgrad(Z))`

Teacher labels are detached for this loss so that branch 3 learns from branch 2 without destabilizing branch 2.

### 4. `L_reg`

All newly introduced constraints are grouped into a single regularization block:

`L_reg =`

- `mu_ent * L_gate_entropy`
- `+ mu_bal * L_gate_balance`
- `+ mu_trend * L_trend_prior`
- `+ mu_ref_mag * L_ref_mag`
- `+ mu_ref_center * L_ref_center`

#### `L_gate_entropy`

Purpose:

- prevent the gate from staying near the trivial `0.5 / 0.5` solution for all samples

Behavior:

- penalize high gate entropy

#### `L_gate_balance`

Purpose:

- prevent collapse to a single branch across the batch

Behavior:

- encourage the batch mean of `p = (p_trend, p_refined)` to stay near `[0.5, 0.5]`

#### `L_trend_prior`

Purpose:

- prevent the model from reconstructing actions almost entirely through `A_refined`

Behavior:

- make `A_trend` match a low-frequency target derived from the real action chunk
- the low-frequency target should be built from a fixed low-pass procedure aligned with the same anchor count `K`

#### `L_ref_mag`

Purpose:

- keep the residual branch from growing too large

Behavior:

- penalize the magnitude of `A_refined`

#### `L_ref_center`

Purpose:

- keep the residual branch from carrying the overall main motion trend

Behavior:

- penalize the temporal mean of `A_refined`

### Recommended default weights

These defaults prioritize stable training for the first implementation. All losses are assumed to use mean reduction.

- `lambda_recon = 0.25`
- `lambda_comp = 0.05`

Inside `L_reg`:

- `mu_ent = 0.002`
- `mu_bal = 0.02`
- `mu_trend = 0.10`
- `mu_ref_mag = 0.0005`
- `mu_ref_center = 0.001`

Rationale:

- `L_flow` must remain dominant
- `L_recon` must be strong enough to make branch 2 useful
- `L_comp` should stay small because branch 3 learns from an auxiliary teacher
- `L_trend_prior` is the most important regularizer against residual-only collapse
- gate and residual regularizers should shape the solution without overriding the main task

### Training schedule for branch-4 fusion

Use cosine teacher decay with a minimum teacher weight:

- warmup phase: `w_Z = 1.0`
- transition phase: cosine decay from `1.0` to `0.1`
- final phase: `w_Z = 0.1`

Then:

- `w_z_hat = 1.0 - w_Z`
- training modulation input is `z_mix = w_Z * stopgrad(Z) + w_z_hat * z_hat`

Inference uses:

- `z_mix = z_hat`

## Section 4: Config, Code Touch Points, And Tests

### Configuration additions

Add new SmolVLA config fields with defaults that preserve current behavior unless explicitly enabled.

#### Feature toggles

- `enable_action_component_branch: bool = False`
- `enable_suffix_component_film: bool = False`

#### Auxiliary branch dimensions

- `component_hidden_dim`
- `component_conv_kernel_size`
- `component_pooling`
- `component_num_anchors`
- `component_predictor_hidden_dim`
- `suffix_component_film_hidden_dim`

#### Anchor control

- `component_anchor_position_temperature`
- `component_anchor_min_separation` (optional)

#### Teacher schedule

- `component_teacher_min_weight: float = 0.1`
- `component_teacher_warmup_ratio: float = 0.1`
- `component_teacher_decay_ratio: float = 0.7`

#### Loss weights

- `loss_weight_recon: float = 0.25`
- `loss_weight_comp: float = 0.05`
- `loss_weight_reg_gate_entropy: float = 0.002`
- `loss_weight_reg_gate_balance: float = 0.02`
- `loss_weight_reg_trend_prior: float = 0.10`
- `loss_weight_reg_ref_mag: float = 0.0005`
- `loss_weight_reg_ref_center: float = 0.001`

### Code touch points

#### `src/lerobot/policies/smolvla/smolvlm_with_expert.py`

- add optional return of the final prefix state-token representation `e`
- preserve current forward behavior when the new output is not requested
- do not change KV-cache semantics

#### `src/lerobot/policies/smolvla/modeling_smolvla.py`

Inside `VLAFlowMatching`:

- register the new auxiliary modules
- compute `e`
- compute branch 2 decomposition outputs during training
- compute branch 3 `z_hat` during training and inference
- compute branch 4 fusion during training only
- modulate `suffix_embs` before the action expert path
- produce grouped loss values and detailed loss statistics

Inside `SmolVLAPolicy.forward(...)`:

- expose grouped training losses in `loss_dict`
- preserve backward-compatible behavior when the feature is disabled

Recommended `loss_dict` entries:

- `loss`
- `loss_flow`
- `loss_recon`
- `loss_comp`
- `loss_reg`
- `loss_reg_gate_entropy`
- `loss_reg_gate_balance`
- `loss_reg_trend_prior`
- `loss_reg_ref_mag`
- `loss_reg_ref_center`

### Validation and error handling

Add explicit validation for:

- `component_num_anchors >= 2`
- `component_teacher_min_weight` in `[0, 1]`
- `component_teacher_warmup_ratio + component_teacher_decay_ratio <= 1`
- `chunk_size >= component_num_anchors`

Losses that are time-based should respect `action_is_pad` when present.

### Test plan

Create a dedicated test file:

- `tests/policies/smolvla/test_smolvla_component_branch.py`

Minimum first-pass coverage:

1. feature disabled:
   - training forward still runs
   - inference still runs
   - behavior remains backward compatible
2. feature enabled:
   - training forward returns finite grouped losses
   - branch outputs have expected shapes
3. shape checks:
   - `Z [B, 2]`
   - `z_hat [B, 2]`
   - `A_trend [B, T, A]`
   - `A_refined [B, T, A]`
   - `A_hat [B, T, A]`
4. inference path:
   - uses `z_hat`
   - does not depend on branch 2 decomposition outputs
5. teacher schedule:
   - starts at full teacher weight
   - decays smoothly
   - never drops below `0.1`
6. regularization sanity checks:
   - entropy term rises for uniform gates
   - balance term rises for batch-level collapse
   - residual magnitude term rises when residuals are large

## Compatibility Requirements

- Default config must preserve current SmolVLA training and inference behavior.
- Existing checkpoints must still load when the new fields are absent.
- The new feature must be opt-in.
- The original branch 1 path is the source of truth for action generation quality.

## Out Of Scope For The First Implementation

- timestep-level component labels
- using noisy actions `x_t` for branch 2 supervision
- replacing the current action expert architecture
- introducing branch 2 into inference
- sample-specific anchor-time predictions
- large-scale refactors of the SmolVLA VLM/expert stack

## Review Checklist

- Branch 1 preserved unchanged in function
- Training uses branches 1/2/3/4
- Inference uses branches 1/3
- `Z` is chunk-level
- `e` is final prefix state-token output
- anchor count is fixed and anchor positions are learnable
- regularization explicitly addresses:
  - uniform `0.5 / 0.5` gates
  - single-branch collapse
  - residual-only reconstruction
  - oversized residuals
- new behavior is disabled by default

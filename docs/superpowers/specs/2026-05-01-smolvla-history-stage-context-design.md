# SmolVLA History Stage Context Design

## Goal

Update the SmolVLA stage-conditioning path so the stage head predicts from a sequence of per-timestep multimodal contexts. For each observation step `i`, the stage branch should encode `obs_i + instruction + state_i` with the existing frozen VLM prefix encoder, pool that prefix into `c_i`, and feed the ordered history `[c_0, ..., c_t]` to the stage head. The action branch remains current-frame SmolVLA flow matching.

The history length is `config.n_obs_steps`.

## Non-Goals

- Do not change the SmolVLA action diffusion flow for the current frame.
- Do not change the existing residual FiLM formula or predicted-stage detach behavior.
- Do not introduce a separate temporal model beyond the existing MLP stage head.
- Do not refactor unrelated dataset, processor, or training code.

## Current State

The current policy already has:

- `StageHead` producing `stage_logits` and `stage_probs`.
- predicted stage conditioning through `StageFiLM`.
- `z_hat.detach()` behavior before FiLM when configured.
- an inference image embedding cache for stage history.

The current stage-history path is image-centric: it can encode historical images, but it still passes the current state into the stage prefix. That does not implement per-timestep `obs_i + instruction + state_i`.

## Design

Add a stage-only history context path in `src/lerobot/policies/smolvla/modeling_smolvla.py`.

1. Keep the action path unchanged:
   - `prepare_images(batch)` selects the current frame.
   - `prepare_state(batch)` selects current `state_t`.
   - `encode_prefix_context(...)` for action caching remains the current-frame prefix.

2. Add a helper to encode stage history:
   - Input: history images, image masks, language tokens/masks, and state history.
   - For each timestep in `config.n_obs_steps`, build a prefix with that timestep's image(s), the same instruction, and that timestep's state.
   - Run the existing prefix encoder.
   - Pool the prefix with the existing `pool_stage_context` policy.
   - Concatenate pooled contexts in temporal order into shape `[batch_size, hidden_size * n_obs_steps]`.

3. Make `StageHead` consume the concatenated history context:
   - For `use_stage_head=True`, initialize `StageHead(input_dim=hidden_size * n_obs_steps, ...)`.
   - If `n_obs_steps == 1`, behavior remains equivalent to the current single-context head.

4. Training:
   - When `use_stage_head=True`, predict stage from the stage-history context.
   - Keep action flow prefix and action hidden generation on the current frame.
   - Keep supervised stage loss and multiscale action loss wiring unchanged.

5. Inference:
   - Use policy queues to maintain `config.n_obs_steps` historical observations.
   - Cache pooled stage contexts rather than only image embeddings, so repeated action chunk generation does not re-embed unchanged history.
   - The cache is stage-only and does not alter current-frame action prefix caching.

6. Conditioning:
   - Continue using predicted `stage_probs` as `z_pred`.
   - Continue detaching predicted stage probabilities before FiLM when `detach_pred_stage_for_action=True`.
   - Continue residual FiLM modulation of action hidden states.

## Edge Cases

- If history is shorter than `n_obs_steps` at episode start, reuse the existing queue fill behavior: repeat the first available observation until the history window is full.
- If historical image/state tensors are not present in a training batch, fall back to the current single-frame prefix only when `n_obs_steps == 1`; otherwise raise a clear error because the configured history cannot be constructed.
- Respect image padding masks for each timestep.

## Tests

Add focused tests under `tests/policies/smolvla/test_training_schedule.py` or a new SmolVLA test file:

- `StageHead` input dimension scales with `config.n_obs_steps`.
- stage history context is built from per-timestep states, not only current `state_t`.
- action branch still receives current-frame images/state while the stage branch receives history.
- predicted stage condition passed to FiLM remains detached.

Use light fake models/helpers where possible to avoid loading real VLM weights.

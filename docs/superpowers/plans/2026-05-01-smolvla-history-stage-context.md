# SmolVLA History Stage Context Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the SmolVLA stage head predict from `config.n_obs_steps` per-timestep multimodal contexts while keeping the action flow on the current frame.

**Architecture:** `SmolVLAPolicy` prepares time-major history images and state history for the stage path. `VLAFlowMatching` encodes each timestep with the existing prefix encoder, pools each prefix to `c_i`, concatenates `[c_i]`, and feeds the existing MLP stage head. Inference caches pooled stage contexts in a policy queue; current-frame image embeddings still feed the action prefix as before.

**Tech Stack:** Python 3.12, PyTorch, pytest, uv, existing LeRobot SmolVLA modules.

---

## File Structure

- Modify `src/lerobot/policies/smolvla/modeling_smolvla.py`
  - Add policy helpers for stage history images/state and stage context cache.
  - Add model helpers for per-timestep stage context encoding.
  - Change `StageHead` input dimension to `hidden_size * config.n_obs_steps`.
  - Thread optional `stage_state` and `stage_context` through training and inference.
- Modify `tests/policies/smolvla/test_training_schedule.py`
  - Add small fake-model tests for policy history preparation and routing.
  - Add lightweight `VLAFlowMatching` unit tests without loading real VLM weights.

---

### Task 1: Add Failing Tests For Stage Head History Shape

**Files:**
- Test: `tests/policies/smolvla/test_training_schedule.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_stage_head_input_dim_scales_with_observation_history():
    model = object.__new__(VLAFlowMatching)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(use_stage_head=True, stage_head_hidden_dim=5, n_obs_steps=4)
    model.vlm_with_expert = SimpleNamespace(
        config=SimpleNamespace(text_config=SimpleNamespace(hidden_size=3))
    )

    VLAFlowMatching._init_stage_head(model)

    first_layer = model.stage_head.mlp[0]
    assert first_layer.in_features == 12
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/policies/smolvla/test_training_schedule.py::test_stage_head_input_dim_scales_with_observation_history -q`

Expected: FAIL because `_init_stage_head` does not exist.

- [ ] **Step 3: Implement minimal code**

In `VLAFlowMatching.__init__`, replace direct stage-head initialization with:

```python
self.stage_head = None
self._init_stage_head()
```

Add:

```python
def _init_stage_head(self) -> None:
    self.stage_head = None
    if self.config.use_stage_head:
        hidden_size = self.vlm_with_expert.config.text_config.hidden_size
        self.stage_head = StageHead(
            hidden_size * self.config.n_obs_steps,
            self.config.stage_head_hidden_dim,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/policies/smolvla/test_training_schedule.py::test_stage_head_input_dim_scales_with_observation_history -q`

Expected: PASS.

---

### Task 2: Add Failing Tests For Policy Stage History Preparation

**Files:**
- Test: `tests/policies/smolvla/test_training_schedule.py`
- Modify: `src/lerobot/policies/smolvla/modeling_smolvla.py`

- [ ] **Step 1: Write the failing tests**

Append:

```python
def test_prepare_stage_state_preserves_history_order():
    policy = _make_policy(num_updates=0)
    policy.config.n_obs_steps = 4
    batch = {OBS_STATE: torch.arange(2 * 4 * 2, dtype=torch.float32).reshape(2, 4, 2)}

    stage_state = policy.prepare_stage_state(batch)

    assert stage_state.shape == (2, 4, policy.config.max_state_dim)
    torch.testing.assert_close(stage_state[:, :, :2], batch[OBS_STATE])


def test_prepare_stage_images_returns_time_major_history():
    policy = _make_policy(num_updates=0)
    policy.config.n_obs_steps = 3
    image_key = "observation.images.rgb"
    batch = {
        image_key: torch.stack(
            [
                torch.full((2, 3, 2, 2), 0.1),
                torch.full((2, 3, 2, 2), 0.2),
                torch.full((2, 3, 2, 2), 0.3),
            ],
            dim=1,
        )
    }

    images, masks = policy.prepare_stage_images(batch)

    assert len(images) == 3
    assert len(masks) == 3
    torch.testing.assert_close(images[0], torch.full_like(images[0], -0.8))
    torch.testing.assert_close(images[1], torch.full_like(images[1], -0.6))
    torch.testing.assert_close(images[2], torch.full_like(images[2], -0.4))
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
uv run pytest tests/policies/smolvla/test_training_schedule.py::test_prepare_stage_state_preserves_history_order tests/policies/smolvla/test_training_schedule.py::test_prepare_stage_images_returns_time_major_history -q
```

Expected: FAIL because `prepare_stage_state` and `prepare_stage_images` do not exist.

- [ ] **Step 3: Implement minimal code**

Add to `SmolVLAPolicy`:

```python
def prepare_stage_state(self, batch: dict[str, Tensor]) -> Tensor:
    raw_state = batch[OBS_STATE]
    if self.config.n_obs_steps == 1:
        state = raw_state[:, -1, :] if raw_state.ndim > 2 else raw_state
        return pad_vector(state[:, None, :], self.config.max_state_dim)
    if raw_state.ndim != 3:
        raise ValueError(
            "`n_obs_steps > 1` requires historical state with shape "
            f"[batch_size, {self.config.n_obs_steps}, state_dim], got {tuple(raw_state.shape)}."
        )
    if raw_state.shape[1] != self.config.n_obs_steps:
        raise ValueError(
            f"Expected {self.config.n_obs_steps} state history steps, got {raw_state.shape[1]}."
        )
    return pad_vector(raw_state, self.config.max_state_dim)

def prepare_stage_images(self, batch: dict[str, Tensor]) -> tuple[list[Tensor], list[Tensor]]:
    if self.config.n_obs_steps == 1:
        return self.prepare_images(batch)
    images_by_step: list[list[Tensor]] = [[] for _ in range(self.config.n_obs_steps)]
    masks_by_step: list[list[Tensor]] = [[] for _ in range(self.config.n_obs_steps)]
    present_img_keys = [key for key in self.config.image_features if key in batch]
    if len(present_img_keys) == 0:
        raise ValueError("All image features are missing from the batch.")
    for key in present_img_keys:
        raw_img = batch[key]
        if raw_img.ndim != 5:
            raise ValueError(
                "`n_obs_steps > 1` requires historical images with shape "
                f"[batch_size, {self.config.n_obs_steps}, channels, height, width], got {tuple(raw_img.shape)}."
            )
        if raw_img.shape[1] != self.config.n_obs_steps:
            raise ValueError(
                f"Expected {self.config.n_obs_steps} image history steps for '{key}', got {raw_img.shape[1]}."
            )
        bsize = raw_img.shape[0]
        valid_mask = self._get_image_valid_mask(batch, key, bsize, raw_img.device, num_frames=raw_img.shape[1])
        frame_masks = list(valid_mask.unbind(dim=1)) if valid_mask.ndim == 2 else [valid_mask] * raw_img.shape[1]
        for step_idx, (img, mask) in enumerate(zip(raw_img.unbind(dim=1), frame_masks, strict=False)):
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)
            images_by_step[step_idx].append(img * 2.0 - 1.0)
            masks_by_step[step_idx].append(mask)
    return (
        [img for step_images in images_by_step for img in step_images],
        [mask for step_masks in masks_by_step for mask in step_masks],
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run:

```bash
uv run pytest tests/policies/smolvla/test_training_schedule.py::test_prepare_stage_state_preserves_history_order tests/policies/smolvla/test_training_schedule.py::test_prepare_stage_images_returns_time_major_history -q
```

Expected: PASS.

---

### Task 3: Add Failing Tests For Per-Timestep Stage Context Encoding

**Files:**
- Test: `tests/policies/smolvla/test_training_schedule.py`
- Modify: `src/lerobot/policies/smolvla/modeling_smolvla.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_encode_stage_history_context_concatenates_per_timestep_contexts():
    model = object.__new__(VLAFlowMatching)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(n_obs_steps=3, stage_pooling="last_state_token")
    calls = []

    def fake_encode_prefix_context(images, img_masks, lang_tokens, lang_masks, state, *, use_cache, image_embs=None):
        calls.append(state.clone())
        hidden = state[:, None, :].repeat(1, 2, 1)
        pad_masks = torch.ones(state.shape[0], 2, dtype=torch.bool)
        return hidden, pad_masks, None

    model.encode_prefix_context = fake_encode_prefix_context
    model.pool_stage_context = lambda hidden, masks: hidden[:, -1, :]
    stage_state = torch.tensor([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]])

    context = model.encode_stage_history_context(
        images=[torch.empty(1, 3, 2, 2) for _ in range(3)],
        img_masks=[torch.ones(1, dtype=torch.bool) for _ in range(3)],
        lang_tokens=torch.ones(1, 4, dtype=torch.long),
        lang_masks=torch.ones(1, 4, dtype=torch.bool),
        stage_state=stage_state,
    )

    assert len(calls) == 3
    torch.testing.assert_close(context, torch.tensor([[1.0, 10.0, 2.0, 20.0, 3.0, 30.0]]))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/policies/smolvla/test_training_schedule.py::test_encode_stage_history_context_concatenates_per_timestep_contexts -q`

Expected: FAIL because `encode_stage_history_context` does not exist.

- [ ] **Step 3: Implement minimal code**

Add to `VLAFlowMatching`:

```python
def encode_stage_history_context(
    self,
    images,
    img_masks,
    lang_tokens,
    lang_masks,
    stage_state: Tensor,
    *,
    image_embs: list[Tensor] | None = None,
) -> Tensor:
    if stage_state.ndim != 3:
        raise ValueError(f"`stage_state` must have shape [B, T, D], got {tuple(stage_state.shape)}.")
    history_len = self.config.n_obs_steps
    if stage_state.shape[1] != history_len:
        raise ValueError(f"Expected {history_len} state history steps, got {stage_state.shape[1]}.")
    if len(img_masks) % history_len != 0:
        raise ValueError(
            f"Stage image masks length {len(img_masks)} is not divisible by history length {history_len}."
        )
    images_per_step = len(img_masks) // history_len
    contexts = []
    for step_idx in range(history_len):
        start = step_idx * images_per_step
        end = start + images_per_step
        step_image_embs = image_embs[start:end] if image_embs is not None else None
        prefix_hidden, prefix_masks, _ = self.encode_prefix_context(
            images[start:end],
            img_masks[start:end],
            lang_tokens,
            lang_masks,
            stage_state[:, step_idx, :],
            use_cache=False,
            image_embs=step_image_embs,
        )
        contexts.append(self.pool_stage_context(prefix_hidden, prefix_masks))
    return torch.cat(contexts, dim=-1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/policies/smolvla/test_training_schedule.py::test_encode_stage_history_context_concatenates_per_timestep_contexts -q`

Expected: PASS.

---

### Task 4: Route Training Stage Prediction Through History Context

**Files:**
- Test: `tests/policies/smolvla/test_training_schedule.py`
- Modify: `src/lerobot/policies/smolvla/modeling_smolvla.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_smolvla_forward_passes_stage_history_separately_from_current_action_state():
    policy = _make_policy(num_updates=0)
    policy.config.n_obs_steps = 4
    policy.config.use_stage_head = True
    batch = _batch()
    batch[OBS_STATE] = torch.arange(2 * 4 * 2, dtype=torch.float32).reshape(2, 4, 2)

    policy.prepare_stage_images = MethodType(
        lambda self, batch: ([torch.zeros(2, 3, 2, 2) for _ in range(4)], [torch.ones(2, dtype=torch.bool) for _ in range(4)]),
        policy,
    )
    policy.prepare_images = MethodType(
        lambda self, batch: ([torch.full((2, 3, 2, 2), 9.0)], [torch.ones(2, dtype=torch.bool)]),
        policy,
    )
    policy.prepare_state = MethodType(
        lambda self, batch: torch.full((2, self.config.max_state_dim), 7.0),
        policy,
    )

    loss, _ = policy.forward(batch)

    call = policy.model.calls[-1]
    assert loss.ndim == 0
    assert call["extra_kwargs"]["stage_state"].shape == (2, 4, policy.config.max_state_dim)
    assert torch.all(call["extra_kwargs"]["state"] == 7.0) if "state" in call["extra_kwargs"] else True
```

Update `_RecordingModel.forward` so it records `stage_state`, and records `state` from its positional argument.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/policies/smolvla/test_training_schedule.py::test_smolvla_forward_passes_stage_history_separately_from_current_action_state -q`

Expected: FAIL because `policy.forward` does not pass `stage_state`.

- [ ] **Step 3: Implement minimal code**

In `SmolVLAPolicy.forward`, replace stage image preparation with:

```python
stage_images = None
stage_img_masks = None
stage_state = None
if self.config.use_stage_head:
    stage_images, stage_img_masks = self.prepare_stage_images(batch)
    stage_state = self.prepare_stage_state(batch)
```

Pass `stage_state=stage_state` to `self.model(...)`.

In `VLAFlowMatching.forward`, add parameter:

```python
stage_state: Tensor | None = None,
stage_context: Tensor | None = None,
```

Replace stage prediction branch with:

```python
if self.config.use_stage_head and self.stage_head is not None:
    if stage_context is None and stage_images is not None and stage_img_masks is not None and stage_state is not None:
        stage_context = self.encode_stage_history_context(
            stage_images,
            stage_img_masks,
            lang_tokens,
            lang_masks,
            stage_state,
        )
    elif stage_context is None:
        if self.config.n_obs_steps != 1:
            raise ValueError("Stage history context requires `stage_state` and `stage_images` when `n_obs_steps > 1`.")
        stage_context = self.pool_stage_context(prefix_out, prefix_pad_masks)
    stage_outputs = self.predict_stage_from_context(stage_context)
```

Add:

```python
def predict_stage_from_context(self, stage_context: Tensor) -> StageHeadOutput | None:
    if not self.config.use_stage_head or self.stage_head is None:
        return None
    stage_logits = self.stage_head(stage_context).to(dtype=torch.float32)
    stage_probs = torch.softmax(stage_logits, dim=-1)
    return {"stage_context": stage_context, "stage_logits": stage_logits, "stage_probs": stage_probs}
```

Make `predict_stage` call `predict_stage_from_context` after pooling.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/policies/smolvla/test_training_schedule.py::test_smolvla_forward_passes_stage_history_separately_from_current_action_state -q`

Expected: PASS.

---

### Task 5: Route Inference Through Pooled Stage Context Cache

**Files:**
- Test: `tests/policies/smolvla/test_training_schedule.py`
- Modify: `src/lerobot/policies/smolvla/modeling_smolvla.py`

- [ ] **Step 1: Write the failing test**

Append:

```python
def test_inference_stage_context_cache_uses_pooled_context_history():
    policy = _make_policy(num_updates=0)
    policy.config.n_obs_steps = 3
    policy.config.use_stage_head = True
    policy.config.use_stage_condition_at_inference = True
    policy.model.embed_images = lambda images: [torch.ones(img.shape[0], 1, 2) * idx for idx, img in enumerate(images, start=1)]
    policy.model.should_use_stage_condition_at_inference = lambda: True
    policy.model.encode_stage_history_context = lambda images, masks, tokens, lang_masks, stage_state, image_embs=None: stage_state[:, :, :2].reshape(stage_state.shape[0], -1)
    batch = {
        OBS_STATE: torch.tensor([[1.0, 2.0]]),
        "observation.images.rgb": torch.ones(1, 3, 2, 2),
        OBS_LANGUAGE_TOKENS: torch.ones(1, 4, dtype=torch.long),
        OBS_LANGUAGE_ATTENTION_MASK: torch.ones(1, 4, dtype=torch.bool),
    }

    policy._queues = populate_queues(policy._queues, batch, exclude_keys=[ACTION])
    current_image_embs, current_img_masks, stage_context = policy._maybe_update_inference_stage_context_cache(batch)

    assert len(current_image_embs) == 1
    assert len(current_img_masks) == 1
    assert stage_context.shape == (1, 6)
```

Import `populate_queues` in the test file if needed.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/policies/smolvla/test_training_schedule.py::test_inference_stage_context_cache_uses_pooled_context_history -q`

Expected: FAIL because `_maybe_update_inference_stage_context_cache` does not exist.

- [ ] **Step 3: Implement minimal code**

In `reset`, add:

```python
self._stage_context_queue = deque(maxlen=self.config.n_obs_steps)
```

Replace `_maybe_update_inference_image_embedding_cache` with `_maybe_update_inference_stage_context_cache` returning:

```python
tuple[list[Tensor] | None, list[Tensor] | None, Tensor | None]
```

Compute current image embeddings as before, encode the current pooled context via `self.model.encode_stage_history_context(...)` over the stacked queue batch, append it to `_stage_context_queue`, and concatenate cached contexts only when the queue length equals `n_obs_steps`.

Thread `stage_context` through `_get_action_chunk` into `self.model.sample_actions(..., stage_context=stage_context)`.

In `VLAFlowMatching.sample_actions`, add optional `stage_context: Tensor | None = None` and use it before encoding `stage_images`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/policies/smolvla/test_training_schedule.py::test_inference_stage_context_cache_uses_pooled_context_history -q`

Expected: PASS.

---

### Task 6: Run Focused Regression Tests

**Files:**
- Verify: `tests/policies/smolvla/test_training_schedule.py`

- [ ] **Step 1: Run the focused SmolVLA tests**

Run: `uv run pytest tests/policies/smolvla/test_training_schedule.py -q`

Expected: all tests in the file pass.

- [ ] **Step 2: Run import/lint-sensitive subset if focused tests pass**

Run: `uv run pytest tests/policies/smolvla -q`

Expected: all SmolVLA policy tests pass or skipped for documented external-weight reasons.

- [ ] **Step 3: Inspect diff**

Run: `git diff -- src/lerobot/policies/smolvla/modeling_smolvla.py tests/policies/smolvla/test_training_schedule.py`

Expected: diff only touches stage-history context routing, tests, and cache names needed by the routing.

# Approach 2 (Revised): Score + Refine Head Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the A2 v1 score head to a 38-channel head that predicts a residual on all 37 raw Gaussian channels plus a score, cut by the score at a random budget, trained in fp16 mixed precision, then run the 1-scene and 5-scene overfits against A2 v1, random and opacity cuts.

**Architecture:** A new `head_mode: score | refine` switch in `Approach2Cfg`; `score` is A2 v1 unchanged. In `refine` mode, `src/approach2/refine.py` adds a forward pre-hook on `encoder.gaussian_regressor` (captures depth, channel 3 of its input), a forward hook on `encoder.gaussian_head` that runs the head and returns `raw + delta` in fp32, and an fp32 wrapper around `encoder.gaussian_adapter.forward`. DepthSplat's own code then builds the corrected Gaussians. A bypass flag gives frozen Gaussians for baselines. Under `16-mixed`, only the backbone and the head run in fp16; everything after the encoder runs with autocast disabled.

**Tech Stack:** Python 3.10, torch 2.4.0, pytorch-lightning 2.4 (`16-mixed`, `MixedPrecision` plugin), Hydra 1.3, `diff_gaussian_rasterization`, `mask_diff_gaussian_rasterization`, TensorBoard, pytest.

**Spec:** `docs/superpowers/specs/2026-11-17-approach-2-refine-head-design.md`

## Global Constraints

- Working directory: `/home/gbrezina/projects/depthsplat-a2`, branch `approach2-score-head` (not the main checkout at `/home/gbrezina/projects/depthsplat`). No new branch, no worktree per ticket. Tickets run strictly in order, one at a time; no GPU work runs in parallel.
- DepthSplat fully frozen. No edits to upstream files under `src/` or `config/` (only `src/approach2/`, `config/approach2/`, `scripts/`, `tests/approach2/`), none to `src/approach1/`. Trainable: `score_head.*` only.
- Checkpoint `/mnt/hdd/depthsplat/pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth`, 256x256, 2 context views, `supervise_intermediate_depth: false`, `return_depth: true` (`config/approach2/re10k_small.yaml`).
- Raw layout (37 ch, `encoder_depthsplat.py:242`, `gaussian_adapter.py:62`): opacity logit `[0]`, xy offset `[1:3]`, scale `[3:6]`, rotation `[6:10]`, SH `[10:37]` as `(xyz d_sh)` with `d_sh = 9`, so SH DC channels are `10, 19, 28`.
- Head input: frozen raw (37) + log depth (1) = 38 ch, built in fp32. Head output: 38 ch = 37 residual + 1 score (channel 37). Residual rows of the final conv zero-initialized; no bound on any residual.
- Training as A2 v1: `k ~ U[0.01, 1]`, Gumbel top-k on the score, `T = 1.0`, mask rasterizer, MSE + 0.05 LPIPS, AdamW 3e-4, OneCycle, `gradient_clip_val` from config.
- Budget: `N_k = floor(k * H * W * V)` via `src/approach1/budget.py:budget_count`, per scene. Every eval row asserts `kept == budget_count`.
- Precision: `approach2.precision` is `"32"` (default) or `"16-mixed"`. `"16-mixed"` → `Trainer(plugins=[MixedPrecision("16-mixed", "cuda", GradScaler(init_scale=2**10))])`. Backbone and head run in fp16. The Gaussian adapter, Gumbel mask, rasterizers, loss, PSNR, SSIM and LPIPS run in fp32 with `torch.autocast("cuda", enabled=False)`.
- GPU commands wrapped as `flock /tmp/depthsplat_gpu.lock <command>`. Run via `poetry run`. Long runs (A2R-08, A2R-09) run in the background; wait for them to exit.
- Training launches allowed in this plan: the memory probe, the 1-scene and the 5-scene overfit. The full RE10K run needs a separate user go-ahead.
- If a gate (G0, G1, G2) fails: stop, write up the result and diagnostics, report to the user. No fallback is built up front.
- `outputs/` is gitignored; only code, scripts and docs are committed.

## Review Focus

- Lightning runs val/test under `torch.inference_mode()`: the hooks must work there and build no graph (the subprocess eval test in A2R-05 goes through `Trainer.test`, so it is the inference-mode test).
- Bypass pass between two refine passes: the hook must leave no stale depth or score behind (test in A2R-02).
- A v1 checkpoint (no `a2_head_mode` key) loaded in `refine` mode must fail with a clear mode error before the shape mismatch (test in A2R-03).
- Autocast reaching a rasterizer or the stock decoder raises `expected scalar type Float but found Half` (`view_matrix @ projection_matrix`, `cuda_splatting.py`): every render path must run with autocast off (tests in A2R-04, A2R-06).
- Eval method names that do not belong to the mode must raise before the model is built (test in A2R-05).

## Shared Interfaces

| Name | Signature | Owner |
|---|---|---|
| `Approach2Cfg.head_mode` | `str = "score"`, allowed `"score"`, `"refine"`; `load_approach2_cfg` raises `ValueError` otherwise | A2R-01 |
| `Approach2Cfg.precision` | `str = "32"`, allowed `"32"`, `"16-mixed"`; `load_approach2_cfg` coerces with `str()` first (Hydra parses `precision=32` as int), `ValueError` otherwise | A2R-01 |
| `ScoreHead` | `(in_channels: int = 38, channels: list[int] = [32, 64, 96, 128], out_channels: int = 1, zero_init_channels: int = 0)`; `forward(x: [n, in, h, w]) -> [n, out, h, w]`; output rows `[0, zero_init_channels)` have zero weight and bias at init | A2R-01 |
| `RAW_GROUPS` | `dict[str, list[int]]`: `opacity [0]`, `offset [1, 2]`, `scale [3, 4, 5]`, `rotation [6, 7, 8, 9]`, `sh_dc [10, 19, 28]`, `sh_rest` (the other 24 of `10..36`) | A2R-02 |
| `DepthCapture` | `(regressor: nn.Module)`; `pop() -> Tensor [bv, 1, h, w]` linear depth (returns and clears, `RuntimeError` if empty); `remove() -> None` | A2R-02 |
| `RefineHook` | `(gaussian_head: nn.Module, head: ScoreHead, depth: DepthCapture)`; attributes `enabled: bool = True`, `sync_time: bool = False`, `last_time: float`; `pop() -> tuple[raw, delta, score]` (`[bv, 37, h, w]`, `[bv, 37, h, w]`, `[bv, 1, h, w]`, all fp32; returns and clears, `RuntimeError` if empty); `remove() -> None` | A2R-02 |
| `wrap_adapter_fp32` | `(adapter: nn.Module) -> None`; replaces `adapter.forward` (instance attribute) with a version that casts tensor args to fp32 and runs with autocast disabled; idempotent | A2R-02 |
| `refine_stats` | `(raw: [bv, 37, h, w], delta: [bv, 37, h, w], scale_min: float, scale_max: float) -> dict[str, Tensor]`, keys `refine/delta_abs/<group>` (mean of `|delta|` over the group's channels and all pixels) for every `RAW_GROUPS` key, `refine/scale_ratio/p50`, `.../p99`, `.../max`, `refine/opacity_mean`, `refine/opacity_mean_frozen`; fp32, no grad | A2R-02 |
| `gaussians_fp32` | `(g: Gaussians) -> Gaussians`, all four fields `.float()` | A2R-02 |
| `ModelWrapperA2.scores` | `(batch, timings: dict \| None = None) -> tuple[Gaussians, [b, g]]`; `refine` mode: corrected Gaussians, gradients reach the head when grad is enabled; sets `self.last_refine: tuple[raw, delta] \| None` (detached) | A2R-03 |
| `ModelWrapperA2.frozen_gaussians` | `(batch) -> Gaussians`, no grad; same encoder call as `scores`; `refine` mode: one bypassed pass | A2R-03 |
| `ModelWrapperA2._render` | override of `ModelWrapperA1._render` (same signature), runs the decoder on `gaussians_fp32(...)` with autocast disabled | A2R-03 |
| Checkpoint key | top-level `checkpoint["a2_head_mode"]: str`, written by `on_save_checkpoint`; missing key reads as `"score"` | A2R-03 |
| Eval methods | `score` mode: `score`, `random`, `opacity`; `refine` mode: `refine`, `refine_random`, `refine_opacity`, `random`, `opacity` | A2R-05 |

## Task Order

| Ticket | Content | Depends on |
|---|---|---|
| A2R-01 | Config keys, `ScoreHead` output channels, mask rasterizer parameter gradients | – |
| A2R-02 | `refine.py`: hooks, fp32 adapter wrapper, stats, fp32 cast (CPU tests) | A2R-01 |
| A2R-03 | Wrapper and builder in `refine` mode, checkpoint mode, zero-init check | A2R-02 |
| A2R-04 | Training step: fp32 boundary, refine logging, gradient reach | A2R-03 |
| A2R-05 | Eval methods, `evaluate.py` checks and precision | A2R-04 |
| A2R-06 | Validation, `train.py` precision plugin, `16-mixed` fit smoke | A2R-05 |
| A2R-07 | G0: full test suite, fp16 sanity | A2R-06 |
| A2R-08 | 1-scene overfit, v1 fp16 re-eval, gate G1 | A2R-07 |
| A2R-09 | Memory probe, 5-scene overfit, evals, gate G2 | A2R-08 |
| A2R-10 | Results doc | A2R-09 |
| A2R-11 | Full RE10K training (needs user go-ahead, not started by this plan) | A2R-10 |

---

### Task 1 (A2R-01): Config, head output channels, rasterizer parameter gradients

**Files:**
- Modify: `src/approach2/config.py`, `config/approach2/re10k_small.yaml`, `src/approach2/score_head.py`
- Test: `tests/approach2/test_config.py`, `tests/approach2/test_score_head.py`, `tests/approach2/test_mask_render.py`

**Interfaces:**
- Produces: `Approach2Cfg.head_mode`, `Approach2Cfg.precision`, `ScoreHead(out_channels, zero_init_channels)`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/approach2/test_config.py
def test_new_keys_defaults():
    cfg = load_approach2_cfg(OmegaConf.create({}))
    assert cfg.head_mode == "score" and cfg.precision == "32"

def test_precision_int_coerced():
    assert load_approach2_cfg(OmegaConf.create({"approach2": {"precision": 32}})).precision == "32"

@pytest.mark.parametrize("key,value", [("head_mode", "merge"), ("precision", "16")])
def test_rejects_unknown_values(key, value):
    with pytest.raises(ValueError):
        load_approach2_cfg(OmegaConf.create({"approach2": {key: value}}))

# tests/approach2/test_score_head.py
def test_refine_output_and_zero_init():
    head = ScoreHead(out_channels=38, zero_init_channels=37)
    out = head(torch.randn(2, 38, 64, 64))
    assert out.shape == (2, 38, 64, 64)
    assert torch.equal(out[:, :37], torch.zeros_like(out[:, :37]))
    assert out[:, 37].abs().sum() > 0

def test_v1_checkpoint_loads_strict():
    # skip if the file is missing; strip the "score_head." prefix as build.py does
    path = Path("outputs/approach2/overfit5_fix1/checkpoints/epoch_1499-step_3000.ckpt")
    ScoreHead().load_state_dict(<prefix-stripped state_dict>, strict=True)
```

New test in `tests/approach2/test_mask_render.py` (next to `test_mask_grad_reaches_masked_gaussians`, line 66, which only checks the mask gradient):

```python
def test_mask_param_grads():
    # _scene(d_sh=9) (G = 500) with requires_grad on means, cov, sh, opa;
    # masks: first 250 = 1, rest = 0; loss = render_cuda_masked(...).sum(); backward
    # grads of means, cov, sh, opa: finite everywhere, nonzero somewhere in [:250], exactly zero in [250:]
```

- [ ] **Step 2: Run, expect FAIL** on the config and head tests

Run: `flock /tmp/depthsplat_gpu.lock poetry run pytest tests/approach2/test_config.py tests/approach2/test_score_head.py tests/approach2/test_mask_render.py -v`
The rasterizer test may already pass (behavior verified in review); it pins the behavior.

- [ ] **Step 3: Implement**

`Approach2Cfg`: add `head_mode: str = "score"`, `precision: str = "32"`. In `load_approach2_cfg`, coerce `precision` with `str()` before `from_dict`, validate both after. YAML: add `head_mode: score`, `precision: "32"` under `approach2:`. `ScoreHead`: `self.out = nn.Conv2d(channels[0], out_channels, 1)`, then zero `weight[:zero_init_channels]` and `bias[:zero_init_channels]` under `torch.no_grad()`.

- [ ] **Step 4: Run `tests/approach2`, expect PASS (v1 tests included)**

Run: `flock /tmp/depthsplat_gpu.lock poetry run pytest tests/approach2 -v`

- [ ] **Step 5: Commit**

```bash
git add src/approach2/config.py config/approach2/re10k_small.yaml src/approach2/score_head.py tests/approach2/test_config.py tests/approach2/test_score_head.py tests/approach2/test_mask_render.py
git commit -m "feat(approach2): add refine head mode, precision and multi-channel score head"
```

---

### Task 2 (A2R-02): Refine hooks, fp32 adapter, stats

**Files:**
- Create: `src/approach2/refine.py`
- Test: `tests/approach2/test_refine_hooks.py` (CPU only)

**Interfaces:**
- Consumes: `ScoreHead` (A2R-01).
- Produces: `RAW_GROUPS`, `DepthCapture`, `RefineHook`, `wrap_adapter_fp32`, `refine_stats`, `gaussians_fp32`.

- [ ] **Step 1: Write the failing tests**

The toy model mimics only the hook mechanics, not the real encoder: `regressor = nn.Identity()` (so its pre-hook sees `x`), `gaussian_head = nn.Conv2d(42, 37, 1)` with `requires_grad=False`, head `ScoreHead(channels=[8, 8], out_channels=38, zero_init_channels=37)`. Toy forward: `gaussian_head(regressor(x))` with `x: [2, 42, 16, 16]`, `x[:, 3] = rand + 0.5` (positive depth).

```python
def test_depth_capture_is_channel_3(): ...      # DepthCapture alone: pop() equals x[:, 3:4]; second pop() raises RuntimeError
def test_hook_replaces_output_zero_init(): ...  # out == gaussian_head(x).float() (torch.equal); pop() shapes [2,37,16,16] x2, [2,1,16,16]; all fp32
def test_hook_output_uses_delta(): ...          # head.out.bias[:37] = 1 -> out == raw + 1
def test_bypass_returns_raw_and_clears(): ...   # enabled=False -> out == raw.float(), RefineHook.pop() raises, depth consumed;
                                                # re-enable with a new x2 -> captured depth is x2[:, 3:4]
def test_grad_reaches_head_only(): ...          # out.sum().backward(): head params have grads, gaussian_head params grad None
def test_inference_mode_works(): ...            # under torch.inference_mode(): forward and pop() work
def test_adapter_fp32_wrapper(): ...            # toy module whose forward returns x @ x.T; wrapped, called on half input
                                                # inside torch.autocast("cpu", dtype=torch.bfloat16) -> output dtype float32;
                                                # wrapping twice does not double-wrap
def test_refine_stats(): ...                    # delta = 0 -> scale_ratio p50 == p99 == max == 1, every delta_abs == 0,
                                                # opacity_mean == opacity_mean_frozen; delta[:, 3:6] = 1 -> p50 > 1
def test_raw_groups_cover_all_channels(): ...   # sorted(sum(RAW_GROUPS.values(), [])) == list(range(37)); sh_dc == [10, 19, 28]
def test_gaussians_fp32(): ...                  # half Gaussians -> all four fields float32
```

- [ ] **Step 2: Run, expect FAIL** (`ModuleNotFoundError: src.approach2.refine`)

Run: `poetry run pytest tests/approach2/test_refine_hooks.py -v`

- [ ] **Step 3: Implement `src/approach2/refine.py`**

- `DepthCapture`: `register_forward_pre_hook`, stores `inputs[0][:, 3:4]` (in the real encoder the `gaussian_regressor` input is `cat(image 3, depth 1, match_prob, features)`, `encoder_depthsplat.py:225-230`).
- `RefineHook`: `register_forward_hook` on `gaussian_head`. Enabled: pop the depth, `raw32 = raw.float()`, `x = cat([raw32, depth.float().clamp_min(1e-6).log()], 1)` (same op as `features.build_head_input`), `out = head(x).float()`, store `(raw32, out[:, :37], out[:, 37:])`, return `raw32 + out[:, :37]`. Time this path into `last_time` (`torch.cuda.synchronize()` before both clocks when `sync_time`). Disabled: pop and drop the depth, leave `last_time` unchanged, return `raw.float()`.
- `wrap_adapter_fp32`: mark the wrapped forward (e.g. attribute `_fp32_wrapped`) so a second call is a no-op.
- `refine_stats`: under `torch.no_grad()`, fp32. Scale ratio per element: `clamp(softplus(raw_s + delta_s - 4), scale_min, scale_max) / clamp(softplus(raw_s - 4), scale_min, scale_max)` (`gaussian_adapter.py:64-67`). Opacity means: `sigmoid` of channel 0 with and without delta. Quantiles via `torch.quantile` on the flattened ratio.

- [ ] **Step 4: Run, expect PASS**

Run: `poetry run pytest tests/approach2/test_refine_hooks.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/approach2/refine.py tests/approach2/test_refine_hooks.py
git commit -m "feat(approach2): add refine hooks, fp32 adapter wrapper and refine stats"
```

---

### Task 3 (A2R-03): Wrapper and builder in refine mode

**Files:**
- Modify: `src/approach2/model_wrapper_a2.py`, `src/approach2/build.py`
- Test: `tests/approach2/test_refine_wrapper.py` (GPU). Copy `_compose` and the `built` fixture from `tests/approach2/test_wrapper.py`, with `extra=["approach2.head_mode=refine"]`.

**Interfaces:**
- Consumes: `DepthCapture`, `RefineHook`, `wrap_adapter_fp32`, `gaussians_fp32` (A2R-02).
- Produces: `scores`, `frozen_gaussians`, `last_refine`, `_render` override, checkpoint key `a2_head_mode`.

- [ ] **Step 1: Write the failing tests**

```python
def test_only_head_trainable(built): ...            # as v1: {"score_head"}
def test_captured_depth_equals_returned(built, monkeypatch): ...
    # wrap wrapper.depth_capture.pop with monkeypatch to record its return value; run scores();
    # allclose(recorded, rearrange(encoder depths, "b v h w -> (b v) () h w"))
def test_zero_init_single_pass_exact(built): ...    # scores(); raw, delta = last_refine; torch.equal(raw + delta, raw)
def test_zero_init_matches_frozen(built): ...       # scores() vs frozen_gaussians() on the same batch (encoder in eval, deterministic):
                                                    # allclose(atol=1e-5) on means, covariances, harmonics, opacities; _render colors allclose(atol=1e-5)
def test_scores_grad(built): ...                    # s.shape == (1, 131072), s.requires_grad, gaussians.covariances.requires_grad
def test_render_under_autocast(built): ...          # with torch.autocast("cuda", dtype=torch.float16): _render(gaussians, batch) runs, color fp32
def test_checkpoint_has_mode_and_roundtrip(built, tmp_path): ...
    # ckpt = {"state_dict": wrapper.state_dict()}; wrapper.on_save_checkpoint(ckpt); ckpt["a2_head_mode"] == "refine";
    # only score_head.* keys; torch.save; fresh build with approach2.head_ckpt=<path> loads identical weights
def test_v1_checkpoint_rejected_in_refine_mode(tmp_path): ...
    # build a score-mode wrapper, on_save_checkpoint, delete "a2_head_mode", save;
    # build in refine mode with it -> ValueError mentioning "score" and "refine"
def test_refine_ckpt_rejected_in_score_mode(built, tmp_path): ...  # refine ckpt -> build in score mode raises ValueError
```

- [ ] **Step 2: Run, expect FAIL**

Run: `flock /tmp/depthsplat_gpu.lock poetry run pytest tests/approach2/test_refine_wrapper.py -v`

- [ ] **Step 3: Implement**

- `__init__`, `refine` mode: `ScoreHead(channels=a2_cfg.head_channels, out_channels=38, zero_init_channels=37)`, `self.depth_capture = DepthCapture(encoder.gaussian_regressor)`, `self.refine_hook = RefineHook(encoder.gaussian_head, self.score_head, self.depth_capture)`, `wrap_adapter_fp32(encoder.gaussian_adapter)`, `self.head_hook = None` (the v1 `HeadInputHook` is never registered in this mode). `score` mode: unchanged.
- `scores`, `refine` mode: same encoder call as v1, without `torch.no_grad()`: the caller's grad mode decides, and with frozen parameters autograd records only from the hook on. `refine_hook.sync_time = timings is not None`. `head_time = refine_hook.last_time + reorder time`, `encode_time = encoder time - refine_hook.last_time`. Pop raw/delta/score, set `last_refine = (raw.detach(), delta.detach())`, return `(gaussians, scores_to_gaussian_order(score, b, v))`.
- `frozen_gaussians`: under `torch.no_grad()`; `refine` mode sets `refine_hook.enabled = False` and restores it in `finally`.
- `_render`: override as in Shared Interfaces (covers validation, which calls `wrapper._render`).
- `on_save_checkpoint`: also set `checkpoint["a2_head_mode"] = self.a2_cfg.head_mode`.
- `build.py`: load the full checkpoint dict, compare `ckpt.get("a2_head_mode", "score")` with `a2_cfg.head_mode` **before** taking `ckpt["state_dict"]` and calling `load_state_dict(strict=True)`; mismatch → `ValueError` naming both modes and the path.

- [ ] **Step 4: Run new and v1 wrapper tests, expect PASS**

Run: `flock /tmp/depthsplat_gpu.lock poetry run pytest tests/approach2/test_refine_wrapper.py tests/approach2/test_wrapper.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/approach2/model_wrapper_a2.py src/approach2/build.py tests/approach2/test_refine_wrapper.py
git commit -m "feat(approach2): add refine mode to wrapper and builder"
```

---

### Task 4 (A2R-04): Training step

**Files:**
- Modify: `src/approach2/training.py`
- Test: `tests/approach2/test_refine_training.py` (GPU, refine fixture as in A2R-03)

**Interfaces:**
- Consumes: `scores`, `last_refine` (A2R-03), `refine_stats`, `gaussians_fp32` (A2R-02).
- Produces: `masked_training_step(wrapper, batch) -> Tensor` (v1 signature), logging the v1 scalars plus all `refine_stats` keys in `refine` mode.

- [ ] **Step 1: Write the failing tests**

Capture logs as `tests/approach2/test_training.py:50` does (`wrapper.log = lambda name, value, *a, **kw: logged.__setitem__(name, value)`).

```python
def test_refine_step_grad_reach(built): ...
    # zero_grad; loss = masked_training_step(...); loss.backward()
    # final conv: weight.grad[:37].abs().sum() > 0 (residual rows get gradient from the render even
    # though their weights are zero) and weight.grad[37].abs().sum() > 0
    # every non-score_head param: grad is None; logged ⊇ v1 keys ∪ refine_stats keys
def test_refine_step_under_autocast(built): ...
    # with torch.autocast("cuda", dtype=torch.float16): loss = masked_training_step(...)
    # loss.dtype == torch.float32, isfinite(loss); backward without scaler -> all score_head grads finite
    # (zeros allowed here; the scaled path is covered by the fit smoke in A2R-06)
```

The v1 `tests/approach2/test_training.py` must pass unchanged.

- [ ] **Step 2: Run, expect FAIL**

Run: `flock /tmp/depthsplat_gpu.lock poetry run pytest tests/approach2/test_refine_training.py -v`

- [ ] **Step 3: Implement**

After `wrapper.scores(batch)`, put the rest of `masked_training_step` (`gumbel_topk_mask` on `s.float()`, `render_masked` on `gaussians_fp32(gaussians)` and `m.float()`, `_photo_loss`, `_psnr`) under `torch.autocast("cuda", enabled=False)`. In `refine` mode, log every entry of `refine_stats(*wrapper.last_refine, cfg.gaussian_scale_min, cfg.gaussian_scale_max)` with `cfg = wrapper.encoder.gaussian_adapter.cfg`.

- [ ] **Step 4: Run, expect PASS**

Run: `flock /tmp/depthsplat_gpu.lock poetry run pytest tests/approach2/test_refine_training.py tests/approach2/test_training.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/approach2/training.py tests/approach2/test_refine_training.py
git commit -m "feat(approach2): train refine head with fp32 boundary and refine stats"
```

---

### Task 5 (A2R-05): Eval methods and `evaluate.py`

**Files:**
- Modify: `src/approach2/evaluation.py`, `src/approach2/evaluate.py`
- Test: `tests/approach2/test_refine_eval.py` (GPU, subprocess)

**Interfaces:**
- Consumes: `scores`, `frozen_gaussians` (A2R-03), `gaussians_fp32` (A2R-02).
- Produces: eval methods per mode; `budget_table.json` rows keyed by `method` with the strings of Shared Interfaces.

- [ ] **Step 1: Write the failing tests**

The test first saves a zero-init refine checkpoint (build a refine wrapper as in A2R-03, `ckpt = {"state_dict": wrapper.state_dict()}`, `wrapper.on_save_checkpoint(ckpt)`, `torch.save`). Then it runs, like `tests/approach2/test_eval_smoke.py`:

```python
["poetry", "run", "python", "-m", "src.approach2.evaluate",
 "+experiment=re10k", "+approach2=re10k_small", "mode=test",
 "dataset/view_sampler=evaluation",
 f"dataset.view_sampler.index_path={ROOT / 'evaluation_index.json'}",
 "dataset.view_sampler.num_context_views=2", f"dataset.roots=[{ROOT}]",
 "approach2.head_mode=refine", "approach2.precision=16-mixed",
 "approach2.eval_budgets=[0.1,1.0]",
 "approach2.eval_methods=[refine,refine_random,refine_opacity,random,opacity]",
 f"approach2.head_ckpt={ckpt}", f"output_dir={out}"]
```

```python
def test_refine_eval_table(tmp_path): ...
    # rows of out/metrics/budget_table.json: all 10 (method, k); kept == {0.1: 13107, 1.0: 131072}[k];
    # head_time > 0 and encode_time > 0; zero init: refine@1.0 psnr == random@1.0 psnr within 0.05 dB,
    # refine_random@0.1 psnr == random@0.1 psnr within 0.05 dB (same Gaussians, same seed)
def test_wrong_method_for_mode_raises(tmp_path): ...
    # head_mode=score with eval_methods=[refine] -> nonzero exit, no out/metrics dir
```

- [ ] **Step 2: Run, expect FAIL**

Run: `flock /tmp/depthsplat_gpu.lock poetry run pytest tests/approach2/test_refine_eval.py -v`

- [ ] **Step 3: Implement**

- `evaluate.py`: next to the existing `head_ckpt` check (top of `evaluate`), before `build_wrapper_a2`: reject methods not allowed for `head_mode` (`ValueError`), and require `head_ckpt` when any of `score`, `refine`, `refine_random`, `refine_opacity` is requested. Pass `precision` to `Trainer`: `"32"` → as today, `"16-mixed"` → `plugins=[MixedPrecision("16-mixed", "cuda", GradScaler(init_scale=2**10))]`.
- `evaluation.py`, `refine` mode: `refine` = (corrected, score); `refine_random` = (corrected, `random_scores(b, g, 0, device)`, the same seed as `random`); `refine_opacity` = (corrected, `gaussians.opacities`, fp32 since the adapter runs in fp32); `random`, `opacity` on `frozen_gaussians(batch)`, built once per scene and only if requested. Keep passing `timings` to `scores`. Everything after `scores` (selection, decoder, PSNR/SSIM/LPIPS, grids) runs under `torch.autocast("cuda", enabled=False)` on `gaussians_fp32(...)`. `score` mode unchanged apart from that fp32 block.

- [ ] **Step 4: Run, expect PASS**

Run: `flock /tmp/depthsplat_gpu.lock poetry run pytest tests/approach2/test_refine_eval.py tests/approach2/test_eval_smoke.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/approach2/evaluation.py src/approach2/evaluate.py tests/approach2/test_refine_eval.py
git commit -m "feat(approach2): evaluate refine head against frozen baselines"
```

---

### Task 6 (A2R-06): Validation, train precision, fit smoke

**Files:**
- Modify: `src/approach2/validation.py`, `src/approach2/train.py`, `src/approach2/model_wrapper_a2.py` (`on_before_optimizer_step`; scaler via `self.trainer.precision_plugin.scaler`)
- Test: `tests/approach2/test_refine_validation.py` (GPU), `tests/approach2/test_train_smoke.py`

**Interfaces:**
- Consumes: `scores`, `frozen_gaussians`, `_render` (A2R-03).
- Produces: `val/psnr@k`, `val/psnr_minus_random@k` and, in `refine` mode, `val/psnr_full_minus_frozen`; `train/amp_scale` under `16-mixed`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/approach2/test_refine_validation.py (refine fixture as in A2R-03, logs captured as in A2R-04)
def test_refine_validation_logs(built): ...
    # under torch.no_grad() and torch.autocast("cuda", dtype=torch.float16): budget_validation_step(wrapper, batch)
    # logged ⊇ {val/psnr@k, val/psnr_minus_random@k for every eval budget, val/psnr_full_minus_frozen};
    # zero init -> |val/psnr_full_minus_frozen| < 0.05

# tests/approach2/test_train_smoke.py
def test_train_smoke_refine_fp16(tmp_path): ...
    # scripts/a2_overfit.sh <out> trainer.max_steps=4 trainer.val_check_interval=2 checkpointing.every_n_train_steps=4
    #   data_loader.train.batch_size=1 approach2.head_mode=refine approach2.precision=16-mixed
    # checkpoint exists and torch.load(ckpt)["a2_head_mode"] == "refine";
    # tensorboard events contain train/amp_scale and refine/scale_ratio/p50 with finite values
    # (read with tensorboard.backend.event_processing.event_accumulator.EventAccumulator)
```

- [ ] **Step 2: Run, expect FAIL**

Run: `flock /tmp/depthsplat_gpu.lock poetry run pytest tests/approach2/test_refine_validation.py tests/approach2/test_train_smoke.py -v`

- [ ] **Step 3: Implement**

- `validation.py`: everything after `scores` runs under `torch.autocast("cuda", enabled=False)`. In `refine` mode the random reference uses `frozen_gaussians(batch)` (a second encoder pass per validation scene, intended), and `val/psnr_full_minus_frozen` = PSNR(corrected, all kept) − PSNR(frozen, all kept). `score` mode keeps v1 behavior.
- `train.py`: `precision` as in A2R-05's `evaluate.py`. Under `16-mixed`, log `train/amp_scale` (the plugin's `scaler.get_scale()`) from `ModelWrapperA2.on_before_optimizer_step` (call `super()` first; the A1 hook logs the unscaled grad norm, since Lightning unscales before this hook).
- Resume: `trainer.fit(ckpt_path=...)` with a checkpoint of the other mode must fail through the same mode check (reuse the `build.py` helper).

- [ ] **Step 4: Run, expect PASS**

Run: `flock /tmp/depthsplat_gpu.lock poetry run pytest tests/approach2/test_refine_validation.py tests/approach2/test_train_smoke.py tests/approach2/test_validation.py -v`

- [ ] **Step 5: Commit**

```bash
git add src/approach2/validation.py src/approach2/train.py src/approach2/model_wrapper_a2.py tests/approach2/test_refine_validation.py tests/approach2/test_train_smoke.py
git commit -m "feat(approach2): validate refine head and train in fp16 mixed precision"
```

---

### Task 7 (A2R-07): Gate G0

**Files:** none. Record all numbers in `outputs/approach2/refine_g0/summary.md` (used by A2R-10).

- [ ] **Step 1: Full test suite**

Run: `flock /tmp/depthsplat_gpu.lock poetry run pytest tests/approach2 -v`
Expected: all PASS.

- [ ] **Step 2: fp16 sanity (heldout10)**

```bash
H=/mnt/hdd/depthsplat/datasets/re10k_heldout10; I=assets/evaluation_index_a2_val10.json
flock /tmp/depthsplat_gpu.lock scripts/a2_eval.sh "" $H $I outputs/approach2/refine_g0/fp32 "approach2.eval_methods=[random,opacity]" "approach2.eval_budgets=[0.05,1.0]"
flock /tmp/depthsplat_gpu.lock scripts/a2_eval.sh <zero-init refine ckpt, saved as in A2R-05> $H $I outputs/approach2/refine_g0/fp16 approach2.head_mode=refine approach2.precision=16-mixed "approach2.eval_methods=[refine,refine_random,random,opacity]" "approach2.eval_budgets=[0.05,1.0]"
```

Read mean PSNR from `<out>/metrics/budget_table.json` (`rows[].method`, `.k`, `.psnr`).
Pass: fp32 `random@1.0` = 25.69 dB (`docs/experiments/2026-11-12-a2-overfit5.md`, heldout10 table); fp16 `refine@1.0`, `random@1.0` and `refine_random@0.05`, `random@0.05` each within 0.1 dB of fp32 `random` at the same k; all values finite. Otherwise stop and report.

---

### Task 8 (A2R-08): 1-scene overfit, gate G1

**Files:** none new. Record numbers in `outputs/approach2/refine_overfit1/summary.md`.

- [ ] **Step 1: Train (background, ~10-20 min)**

Run: `flock /tmp/depthsplat_gpu.lock scripts/a2_overfit1.sh outputs/approach2/refine_overfit1 approach2.head_mode=refine approach2.precision=16-mixed`
Expected: 3000 steps, finite loss, `outputs/approach2/refine_overfit1/checkpoints/epoch_0-step_3000.ckpt`.

- [ ] **Step 2: Eval the refine head and re-eval v1 in fp16 (same scene and views)**

```bash
IDX=assets/evaluation_index_a1_overfit1.json; R=/mnt/hdd/depthsplat/datasets/re10k_overfit5
flock /tmp/depthsplat_gpu.lock scripts/a2_eval.sh outputs/approach2/refine_overfit1/checkpoints/epoch_0-step_3000.ckpt $R $IDX outputs/approach2/refine_overfit1/eval approach2.head_mode=refine approach2.precision=16-mixed "approach2.eval_methods=[refine,refine_random,refine_opacity,random,opacity]" approach2.save_grids=true
flock /tmp/depthsplat_gpu.lock scripts/a2_eval.sh outputs/approach2/overfit1/checkpoints/epoch_0-step_3000.ckpt $R $IDX outputs/approach2/overfit1/eval_fp16 approach2.precision=16-mixed
```

This is an eval on the training scene, as intended for an overfit gate.

- [ ] **Step 3: Gate G1**

Pass if `refine` PSNR > v1 `score` PSNR (both `budget_table.json`, fp16) at k = 0.05, 0.1, 0.25. On fail, stop and report, including the last values of `refine/*`, `val/psnr@*` and `val/psnr_full_minus_frozen` from `outputs/approach2/refine_overfit1/tensorboard` (`EventAccumulator`).

---

### Task 9 (A2R-09): 5-scene overfit, gate G2

**Files:** none new. Record numbers in `outputs/approach2/refine_overfit5/summary.md`.

- [ ] **Step 1: Memory probe**

Run, with `nvidia-smi --query-gpu=memory.used --format=csv,noheader -lms 500` logging to a file in the background, and the max taken afterwards:
`flock /tmp/depthsplat_gpu.lock scripts/a2_overfit.sh outputs/approach2/refine_probe_bs4 trainer.max_steps=20 trainer.val_check_interval=10 data_loader.train.batch_size=4 approach2.head_mode=refine approach2.precision=16-mixed`
If it fails or exceeds 10.5 GB, repeat with batch 2.

- [ ] **Step 2: Train (background, ~45-90 min)**

Batch 4: `trainer.max_steps=3000`; batch 2: `trainer.max_steps=6000 checkpointing.every_n_train_steps=1000` (same samples as v1: batch 4 x 3000).
Run: `flock /tmp/depthsplat_gpu.lock scripts/a2_overfit.sh outputs/approach2/refine_overfit5 data_loader.train.batch_size=<N> trainer.max_steps=<S> approach2.head_mode=refine approach2.precision=16-mixed`
Last checkpoint: the one with `step_<S>` in `outputs/approach2/refine_overfit5/checkpoints/`.

- [ ] **Step 3: Evals**

| Head | Root, index | Extra overrides |
|---|---|---|
| refine, last ckpt | `re10k_overfit5`, `$ROOT/evaluation_index.json` | `approach2.head_mode=refine approach2.precision=16-mixed "approach2.eval_methods=[refine,refine_random,refine_opacity,random,opacity]" approach2.save_grids=true` |
| refine, last ckpt | `re10k_heldout10`, `assets/evaluation_index_a2_val10.json` | same |
| v1 `outputs/approach2/overfit5_fix1/checkpoints/epoch_1499-step_3000.ckpt` | both, as above | `approach2.precision=16-mixed` (default `head_mode=score`, methods `score, random, opacity`) |

Output dirs: `outputs/approach2/refine_overfit5/eval_{overfit5,heldout10}`, `outputs/approach2/overfit5_fix1/eval_{overfit5,heldout10}_fp16`.

- [ ] **Step 4: Gate G2**

Pass if on heldout10 `refine` PSNR > v1 `score` (fp16) and > `random` of the refine eval run (frozen, fp16) at k = 0.05, 0.1, 0.25. On fail, stop and report as in A2R-08.

---

### Task 10 (A2R-10): Results doc

**Files:**
- Create: `docs/experiments/2026-11-17-a2-refine-overfit.md`, `docs/experiments/img/2026-11-17-a2-refine/*.jpg`

- [ ] **Step 1: Write the doc**

Style of `docs/experiments/2026-11-12-a2-overfit5.md`. Content, from the three `summary.md` files and the eval outputs:
- Setup: head params, precision (`16-mixed`, fp32 adapter/render/metrics), batch and steps, the G0 fp16 sanity numbers.
- Tables (method x k: PSNR / SSIM / LPIPS, kept) for G1 and G2, with gate verdicts.
- `refine` vs `refine_random` (ranking vs. Gaussians), `refine_opacity` (sort by corrected opacity), `refine` at k=1 vs frozen.
- Final `refine/scale_ratio/{p50,p99,max}` and `refine/delta_abs/*`.
- Cost: `head_time` (fp16, includes the `raw + delta` add; not comparable 1:1 with v1's fp32 5.0 ms), peak memory.
- Notes: the budget is not a head input in this version; voxel merging and post-hoc pruning baselines are deferred until after full training.
- Grids: from `<eval out>/grids/*.png`, convert to JPG. Use the 3 views of the 1-scene run, plus target view 1 of 4 heldout10 scenes.

- [ ] **Step 2: Commit**

```bash
git add docs/experiments/2026-11-17-a2-refine-overfit.md docs/experiments/img/2026-11-17-a2-refine
git commit -m "docs(approach2): add refine head overfit results"
```

---

### Task 11 (A2R-11): Full RE10K training (needs user go-ahead)

Not started by this plan. After G2 passes: propose steps, batch size (from A2R-09), duration and output dir for `scripts/a2_train.sh ... approach2.head_mode=refine approach2.precision=16-mixed`, and launch only after the user says yes.

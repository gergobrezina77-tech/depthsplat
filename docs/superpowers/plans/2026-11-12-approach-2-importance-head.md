# Approach 2: Learned Importance Head Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a U-Net score head on a frozen DepthSplat that ranks per-pixel Gaussians, so a global top-`N_k` cut by score keeps rendering quality, with an overfit run, a full RE10K run and a budget eval against random and opacity cuts.

**Architecture:** New code in `src/approach2/`, `config/approach2/`, `scripts/a2_*.sh`, `tests/approach2/`. `ModelWrapperA2` subclasses `ModelWrapperA1` (reuses `_render`, `_photo_loss`, `_psnr`, `configure_optimizers`, `on_before_optimizer_step`) and owns a `ScoreHead`. A forward hook on `encoder.gaussian_head` provides the head input. Training renders with MaskGaussian's mask rasterizer; eval gathers the kept Gaussians and renders with the stock decoder.

**Tech Stack:** Python 3.10, torch 2.4.0, pytorch-lightning 2.4, Hydra 1.3, `diff_gaussian_rasterization` (stock), `mask_diff_gaussian_rasterization` (new), TensorBoard, pytest.

**Spec:** `docs/superpowers/specs/2026-11-12-approach-2-importance-head-design.md`

## Global Constraints

- DepthSplat fully frozen, incl. `gaussian_head`. Trainable: `score_head.*` only.
- Checkpoint: `/mnt/hdd/depthsplat/pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth`, 256x256, 2 context views, `upsample_factor=4`, `lowest_feature_resolution=4`, `supervise_intermediate_depth: false`.
- Head input: raw `gaussian_head` output (37 ch) + log depth (1 ch) = 38 ch, full resolution, per view.
- Budget: `N_k = floor(k * H * W * V)` via `src/approach1/budget.py:budget_count`, global top-`N_k` per scene (not over the batch).
- Training mask: Gumbel top-k on standardized scores, `p = 1 - exp(-exp((z - tau) / T))`, `m = m_hard + (p - p.detach())`, `k ~ U[k_min, 1]`, `k_min = 0.01`, `T = 1.0` default.
- Loss: upstream photometric (`mse` + `lpips` 0.05) on target views. No full-set term.
- Eval renders with the stock rasterizer only. Budgets `{0.05, 0.1, 0.25, 0.5, 1.0}`. Methods `score`, `random`, `opacity`. Kept count must equal `budget_count`.
- Mask rasterizer: `git+https://github.com/kaikai23/mask-diff-gaussian-rasterization@540dc11ef44c64f4c55fb894b3a4879de0cebfe7`.
- No edits to upstream files under `src/` or `config/`, and none to `src/approach1/`.
- GPU: RTX 2080 Ti, 11 GB, fp32. Run via `poetry run ...`. The full RE10K run (Task 7) needs explicit user go-ahead.

## Review Focus

- Batch with 2 scenes: mask and top-`N_k` are chosen per scene, never over the batch (test in Task 3).
- Lightning calls `train()` on the whole wrapper: the encoder must stay in `eval()` (`mv_unimatch.py` branches on `self.training`) (test in Task 4).
- Head-only checkpoint saved during training must load into a fresh wrapper for eval (test in Task 4).
- Extreme budgets: `k = 1e-6` keeps exactly 1 Gaussian, `k = 1.0` keeps all, both with finite gradients (test in Task 3).
- `supervise_intermediate_depth: true` repeats the batch after the hook fires and silently misaligns depth and features: building must raise (test in Task 4).

## File Map

| File | Responsibility | Task |
|---|---|---|
| `scripts/install_mask_rasterizer.sh` | build/install the mask rasterizer | 1 |
| `src/approach2/__init__.py` | package marker | 1 |
| `src/approach2/mask_render.py` | `render_cuda_masked`, `render_masked` | 1 |
| `config/approach2/re10k_small.yaml` | Hydra group `+approach2=re10k_small` | 2 |
| `src/approach2/config.py` | `Approach2Cfg`, `load_approach2_cfg` | 2 |
| `src/approach2/score_head.py` | `ScoreHead` U-Net | 2 |
| `src/approach2/features.py` | `HeadInputHook`, `build_head_input`, `scores_to_gaussian_order` | 3 |
| `src/approach2/select.py` | `gumbel_topk_mask`, `gather_gaussians`, `select_top_k`, `random_scores` | 3 |
| `src/approach2/build.py` | `build_wrapper_a2` | 4 |
| `src/approach2/model_wrapper_a2.py` | `ModelWrapperA2` | 4, 5, 6 |
| `src/approach2/evaluate.py`, `scripts/a2_eval.sh` | budget eval entry | 5 |
| `src/approach2/train.py`, `scripts/a2_overfit.sh` | train entry, fixed validation | 6 |
| `assets/evaluation_index_a2_val10.json` | upstream views for the 10 validation scenes | 6 |
| `scripts/a2_train.sh` | full RE10K run | 7 |
| `tests/approach2/*.py` | pytest tests | 1-6 |

Execution order: `1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7`. Tasks 2 and 3 can run in parallel with Task 1 after Task 1's install step. Task 7 waits for the user's go-ahead after Task 6's results.

---

### Task 1: Mask rasterizer install and masked render (A2-01)

**Files:**
- Create: `scripts/install_mask_rasterizer.sh`, `src/approach2/__init__.py`, `src/approach2/mask_render.py`
- Test: `tests/approach2/test_mask_render.py`

**Interfaces:**
- Consumes: `src/model/decoder/cuda_splatting.py:render_cuda` (structure to copy), `get_projection_matrix`.
- Produces:
  - `render_cuda_masked(extrinsics, intrinsics, near, far, image_shape, background_color, gaussian_means, gaussian_covariances, gaussian_sh_coefficients, gaussian_opacities, gaussian_masks: Float[Tensor, "batch gaussian"], scale_invariant: bool = True) -> Float[Tensor, "batch 3 height width"]`
  - `render_masked(decoder: DecoderSplattingCUDA, gaussians: Gaussians, masks: Float[Tensor, "batch gaussian"], target: dict) -> Float[Tensor, "batch view 3 height width"]` (mirrors `DecoderSplattingCUDA.forward`, repeating `masks` over target views like opacities).

- [ ] **Step 1: Write `scripts/install_mask_rasterizer.sh`**

Copy of `scripts/install_rasterizer.sh` with: idempotence check `import mask_diff_gaussian_rasterization`, install URL `git+https://github.com/kaikai23/mask-diff-gaussian-rasterization@540dc11ef44c64f4c55fb894b3a4879de0cebfe7`. Same `CC/CXX/CUDAHOSTCXX=gcc-13/g++-13`, `TORCH_CUDA_ARCH_LIST=7.5`, `NVCC_APPEND_FLAGS=-include cstdint`.

- [ ] **Step 2: Run it and verify import**

Run: `scripts/install_mask_rasterizer.sh && poetry run python -c "import mask_diff_gaussian_rasterization, diff_gaussian_rasterization; print('ok')"`
Expected: `ok` (both packages import side by side).

- [ ] **Step 3: Write the failing tests** (synthetic scene: one camera at the origin looking down +z, 500 random Gaussians at z in [2, 4], 64x64 image, CUDA only)

```python
def test_mask_ones_matches_stock():
    # render_cuda_masked(..., masks=ones) vs render_cuda(...), same inputs
    assert (img_masked - img_stock).abs().max() < 1e-4

def test_mask_zero_matches_removed():
    # masks zero on 100 Gaussians vs render_cuda on the other 400 only
    assert (img_masked - img_removed).abs().max() < 1e-4

def test_mask_grad_reaches_masked_gaussians():
    # masks = m_hard + (p - p.detach()) with 100 zeros; loss = img.mean()
    assert p.grad.shape == p.shape
    assert (p.grad[dropped_idx].abs() > 0).any()

def test_straight_through_forward_is_exact():
    m = m_hard + (p - p.detach())
    assert torch.equal(m, m_hard)
```

- [ ] **Step 4: Run tests, verify they fail**

Run: `poetry run pytest tests/approach2/test_mask_render.py -v`
Expected: FAIL (`ModuleNotFoundError: src.approach2.mask_render`).

- [ ] **Step 5: Implement `render_cuda_masked` and `render_masked` in `src/approach2/mask_render.py`**

Copy `render_cuda` (incl. `scale_invariant` block, `use_sh=True` path only), import the mask package's `GaussianRasterizationSettings`/`GaussianRasterizer`, call it with keyword arguments only and `masks=gaussian_masks[i][:, None].float().contiguous()` (shape `[P, 1]`, float32: the backward allocates `{P, 1}`, `rasterize_points.cu:158`). Raise `ValueError` if `gaussian_masks` is `None`.

- [ ] **Step 6: Run tests, verify they pass**

Run: `poetry run pytest tests/approach2/test_mask_render.py -v`
Expected: 4 PASS.

- [ ] **Step 7: Commit**

```bash
git add scripts/install_mask_rasterizer.sh src/approach2/__init__.py src/approach2/mask_render.py tests/approach2/test_mask_render.py
git commit -m "feat(approach2): add mask rasterizer install and masked render"
```

---

### Task 2: Config group and score head (A2-02)

**Files:**
- Create: `config/approach2/re10k_small.yaml`, `src/approach2/config.py`, `src/approach2/score_head.py`
- Test: `tests/approach2/test_score_head.py`, `tests/approach2/test_config.py`

**Interfaces:**
- Produces:
  - `@dataclass Approach2Cfg`: `k_min: float = 0.01`, `k_max: float = 1.0`, `temperature: float = 1.0`, `head_channels: list[int] = [32, 64, 96, 128]`, `eval_budgets: list[float] = [0.05, 0.1, 0.25, 0.5, 1.0]`, `eval_methods: list[str] = ["score", "random", "opacity"]`, `head_ckpt: str | None = None`, `val_root: str = "/mnt/hdd/depthsplat/datasets/re10k_heldout10"`, `val_index_path: str = "assets/evaluation_index_a2_val10.json"`.
  - `load_approach2_cfg(cfg_dict: DictConfig) -> Approach2Cfg` (reads key `approach2`, as `load_approach1_cfg`).
  - `ScoreHead(in_channels: int = 38, channels: list[int] = [32, 64, 96, 128])`, `forward(x: Float[Tensor, "n c h w"]) -> Float[Tensor, "n 1 h w"]`.

- [ ] **Step 1: Write the failing tests**

```python
def test_score_head_shape():
    head = ScoreHead()
    assert head(torch.randn(4, 38, 256, 256)).shape == (4, 1, 256, 256)

def test_score_head_param_budget():
    n = sum(p.numel() for p in ScoreHead().parameters())
    assert 200_000 < n < 1_500_000

def test_score_head_rejects_bad_resolution():
    with pytest.raises(ValueError):
        ScoreHead()(torch.randn(1, 38, 100, 100))  # not divisible by 8

def test_load_approach2_cfg_defaults():
    cfg = load_approach2_cfg(OmegaConf.create({}))
    assert cfg.k_min == 0.01 and cfg.temperature == 1.0
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `poetry run pytest tests/approach2/test_score_head.py tests/approach2/test_config.py -v`
Expected: FAIL (modules missing).

- [ ] **Step 3: Implement `ScoreHead` in `src/approach2/score_head.py`**

Stem `3x3 conv + GroupNorm(8) + GELU` (38 -> 32). Encoder: 4 levels at 1, 1/2, 1/4, 1/8, each two `3x3 conv + GELU`, downsample with stride-2 conv. Decoder: bilinear upsampling, concat skip, two `3x3 conv + GELU`. Final `1x1 conv` -> 1. `ValueError` if `h % 8` or `w % 8`.

- [ ] **Step 4: Implement `Approach2Cfg`, `load_approach2_cfg` and `config/approach2/re10k_small.yaml`**

The yaml mirrors `config/approach1/re10k_small.yaml` (same `model.encoder`, `checkpointing.pretrained_model`, `data_loader`, `trainer`, `train`, `dataset.view_sampler.warm_up_steps: 0`, `wandb.mode: disabled`) with `optimizer.lr: 3.e-4`, an `approach2:` block with the dataclass defaults, and `output_dir: outputs/approach2/default`.

- [ ] **Step 5: Run tests, verify they pass**

Run: `poetry run pytest tests/approach2/test_score_head.py tests/approach2/test_config.py -v`
Expected: 4 PASS.

- [ ] **Step 6: Commit**

```bash
git add config/approach2 src/approach2/config.py src/approach2/score_head.py tests/approach2/test_score_head.py tests/approach2/test_config.py
git commit -m "feat(approach2): add config group and U-Net score head"
```

---

### Task 3: Head input and selection (A2-03)

**Files:**
- Create: `src/approach2/features.py`, `src/approach2/select.py`
- Test: `tests/approach2/test_select.py`, `tests/approach2/test_features.py`

**Interfaces:**
- Consumes: `budget_count` (`src/approach1/budget.py`), `Gaussians` (`src/model/types.py`).
- Produces:
  - `class HeadInputHook`: `__init__(module: nn.Module)` registers a forward hook storing the module output; `pop() -> Tensor` returns and clears it (raises `RuntimeError` if empty); `remove()`.
  - `build_head_input(raw: Float[Tensor, "bv 37 h w"], depths: Float[Tensor, "b v h w"]) -> Float[Tensor, "bv 38 h w"]` (appends `log(depth.clamp_min(1e-6))`).
  - `scores_to_gaussian_order(s: Float[Tensor, "bv 1 h w"], b: int, v: int) -> Float[Tensor, "b g"]` with `g = v*h*w` in `(v h w)` order (`encoder_depthsplat.py:252`).
  - `gumbel_topk_mask(s: Float[Tensor, "b g"], k: float, temperature: float, generator: torch.Generator | None = None) -> tuple[Float[Tensor, "b g"], Bool[Tensor, "b g"]]` returns `(m, m_hard)`, per scene.
  - `gather_gaussians(gaussians: Gaussians, idx: Int64[Tensor, "b n"]) -> Gaussians`.
  - `select_top_k(gaussians: Gaussians, scores: Float[Tensor, "b g"], k: float) -> Gaussians`.
  - `random_scores(b: int, g: int, seed: int, device) -> Float[Tensor, "b g"]`.

Algorithm for `gumbel_topk_mask` (spec §4):

```python
z = (s - s.mean(1, keepdim=True).detach()) / s.std(1, keepdim=True).detach().clamp_min(1e-6)
g = z + temperature * gumbel_noise  # -log(-log(U)), U ~ U(0, 1) clamped to [1e-10, 1 - 1e-10]
n_k = budget_count(s.shape[1], k)
top = torch.topk(g, n_k, dim=1)
m_hard = zeros_like(g, bool).scatter(1, top.indices, True)
tau = top.values[:, -1:].detach()
p = 1 - torch.exp(-torch.exp((z - tau) / temperature))
m = m_hard.float() + (p - p.detach())
```

- [ ] **Step 1: Write the failing tests**

```python
def test_gumbel_mask_exact_budget_per_scene():
    s = torch.randn(2, 1000, requires_grad=True)
    m, m_hard = gumbel_topk_mask(s, 0.1, 1.0)
    assert m_hard.sum(1).tolist() == [100, 100]
    assert torch.equal(m, m_hard.float())

def test_gumbel_mask_grad_reaches_scores():
    s = torch.randn(1, 1000, requires_grad=True)
    m, _ = gumbel_topk_mask(s, 0.1, 1.0)
    (m * torch.randn(1, 1000)).sum().backward()
    assert torch.isfinite(s.grad).all() and (s.grad != 0).any()

@pytest.mark.parametrize("k, expected", [(1e-6, 1), (1.0, 1000)])
def test_gumbel_mask_extreme_budgets(k, expected):
    s = torch.randn(1, 1000, requires_grad=True)
    m, m_hard = gumbel_topk_mask(s, k, 1.0)
    m.sum().backward()
    assert int(m_hard.sum()) == expected and torch.isfinite(s.grad).all()

def test_select_top_k_per_scene():
    # scene 0 scores favor indices 0..9, scene 1 favors 990..999; k = 0.01 of g = 1000
    kept = select_top_k(gaussians, scores, 0.01)
    assert kept.opacities.shape == (2, 10)
    assert set(kept_idx[0]) == set(range(10)) and set(kept_idx[1]) == set(range(990, 1000))

def test_scores_to_gaussian_order():
    s = torch.arange(2 * 2 * 4 * 4.).reshape(4, 1, 4, 4)  # (b v) 1 h w, b=2, v=2
    out = scores_to_gaussian_order(s, b=2, v=2)
    assert out.shape == (2, 32) and out[1, 16] == s[3, 0, 0, 0]

def test_hook_pop_clears():
    hook = HeadInputHook(nn.Identity())
    hook.module(torch.ones(1))
    assert hook.pop().shape == (1,)
    with pytest.raises(RuntimeError):
        hook.pop()
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `poetry run pytest tests/approach2/test_select.py tests/approach2/test_features.py -v`
Expected: FAIL (modules missing).

- [ ] **Step 3: Implement `src/approach2/select.py` and `src/approach2/features.py`**

`gather_gaussians` follows the inner `gather` of `select_top_k_by_opacity` (`src/approach1/budget.py:17-27`). `random_scores` uses a `torch.Generator` seeded with `seed`.

- [ ] **Step 4: Run tests, verify they pass**

Run: `poetry run pytest tests/approach2/test_select.py tests/approach2/test_features.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/approach2/select.py src/approach2/features.py tests/approach2/test_select.py tests/approach2/test_features.py
git commit -m "feat(approach2): add head input hook and score selection"
```

---

### Task 4: Wrapper, builder and training step (A2-04)

**Files:**
- Create: `src/approach2/build.py`, `src/approach2/model_wrapper_a2.py`
- Test: `tests/approach2/test_train_step.py`

**Interfaces:**
- Consumes: Tasks 1-3, `ModelWrapperA1` (`src/approach1/model_wrapper_a1.py`), `Approach1Cfg`, `freeze_all_but`, `set_frozen_eval` (`src/approach1/freeze.py`).
- Produces:
  - `ModelWrapperA2(ModelWrapperA1)`, `__init__(..., a2_cfg: Approach2Cfg, eval_data_cfg=None)`: passes `Approach1Cfg(trainable_prefixes=["score_head."], eval_budgets=a2_cfg.eval_budgets)` to the parent, creates `self.score_head = ScoreHead(38, a2_cfg.head_channels)`, `self.head_hook = HeadInputHook(encoder.gaussian_head)`, sets `self.strict_loading = False`.
  - `ModelWrapperA2.scores(batch) -> tuple[Gaussians, Float[Tensor, "b g"]]`: encoder under `torch.no_grad()`, `depths` from the encoder's dict output, hook output, `build_head_input`, `score_head`, `scores_to_gaussian_order`.
  - `ModelWrapperA2.train(mode: bool = True) -> "ModelWrapperA2"`: `super().train(mode)`, then `set_frozen_eval(self, ["score_head."])`, return `self`.
  - `on_save_checkpoint(checkpoint)`: keep only `score_head.*` keys in `checkpoint["state_dict"]`.
  - `build_wrapper_a2(cfg_dict) -> tuple[ModelWrapperA2, DataModule, RootCfg, Approach2Cfg]`: mirrors `src/approach1/build.py`, loads the pretrained checkpoint with `strict=False`, raises `RuntimeError` unless every missing key starts with `score_head.` and there are no unexpected keys, loads `a2_cfg.head_ckpt` (Lightning ckpt, `score_head.*` keys) if set, raises `ValueError` if `cfg.model.encoder.supervise_intermediate_depth` or not `cfg.model.encoder.return_depth`, calls `freeze_all_but(wrapper, ["score_head."])`.

- [ ] **Step 1: Write the failing tests** (CUDA, `re10k_overfit5`, batch size 1, fixture as in `tests/approach1/test_train_step.py`)

```python
def test_train_step_grads_only_in_head(built):
    loss = wrapper.training_step(batch, 0); loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for n, p in wrapper.named_parameters() if n.startswith("score_head."))
    assert all(p.grad is None for n, p in wrapper.named_parameters() if not n.startswith("score_head."))

def test_encoder_stays_eval_after_train(built):
    wrapper.train()
    assert not wrapper.encoder.training and wrapper.score_head.training

def test_hook_depth_matches_gaussian_means(built):
    # z of gaussians.means in context camera v at pixel (v, y, x) equals depths[0, v, y, x]
    assert torch.allclose(z_cam, depths[0].flatten(), rtol=1e-3)

def test_head_only_checkpoint_roundtrip(built, tmp_path):
    ckpt = {"state_dict": wrapper.state_dict()}; wrapper.on_save_checkpoint(ckpt)
    assert all(k.startswith("score_head.") for k in ckpt["state_dict"])
    # fresh build with approach2.head_ckpt=<saved file> has equal score_head weights

def test_build_rejects_intermediate_depth():
    with pytest.raises(ValueError):
        build_wrapper_a2(compose(..., "model.encoder.supervise_intermediate_depth=true"))
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `poetry run pytest tests/approach2/test_train_step.py -v`
Expected: FAIL (modules missing).

- [ ] **Step 3: Implement `build_wrapper_a2` and `ModelWrapperA2` with `training_step`**

`training_step`: `data_shim`, `scores(batch)`, `k = k_min + (k_max - k_min) * rand`, `m, m_hard = gumbel_topk_mask(s, k, temperature)`, `render_masked(self.decoder, gaussians, m, batch["target"])`, wrap color in `DecoderOutput(color, None)` and use `self._photo_loss`. Logs: `loss/masked`, `train/psnr_masked`, `train/k`, `train/kept` (`m_hard.sum(1).float().mean()`), `train/score_std` (`s.std()`), `info/global_step`. Step tracker update as in A1.

- [ ] **Step 4: Run tests, verify they pass**

Run: `poetry run pytest tests/approach2/test_train_step.py -v`
Expected: 5 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/approach2/build.py src/approach2/model_wrapper_a2.py tests/approach2/test_train_step.py
git commit -m "feat(approach2): add wrapper, builder and masked training step"
```

---

### Task 5: Budget eval (A2-05)

**Files:**
- Modify: `src/approach2/model_wrapper_a2.py` (add `test_step`, `on_test_end`)
- Create: `src/approach2/evaluate.py`, `scripts/a2_eval.sh`
- Test: `tests/approach2/test_eval_smoke.py`

**Interfaces:**
- Consumes: `ModelWrapperA2.scores`, `select_top_k`, `random_scores`, `budget_count`, `compute_psnr/ssim/lpips`.
- Produces: `outputs/.../metrics/budget_table.json` = `{"scenes": [...], "rows": [{"method", "k", "psnr", "ssim", "lpips", "kept", "n_total", "scenes", "head_time"}]}` and `per_scene.json` (one row per scene, method, k).

- [ ] **Step 1: Write the failing test** (mirrors `tests/approach1/test_eval_smoke.py`, `src.approach2.evaluate`, `+approach2=re10k_small`, `approach2.eval_budgets=[0.1,1.0]`)

```python
data = json.loads((out / "metrics" / "budget_table.json").read_text())
assert len(data["scenes"]) == 5
assert {(r["method"], r["k"]) for r in data["rows"]} == {(m, k) for m in ["score", "random", "opacity"] for k in [0.1, 1.0]}
assert all(r["kept"] == {0.1: 13107, 1.0: 131072}[r["k"]] for r in data["rows"])
```

- [ ] **Step 2: Run test, verify it fails**

Run: `poetry run pytest tests/approach2/test_eval_smoke.py -v`
Expected: FAIL (`No module named src.approach2.evaluate`).

- [ ] **Step 3: Implement `test_step`, `on_test_end`, `evaluate.py`, `scripts/a2_eval.sh`**

`test_step`: one `scores(batch)` call per scene (timed, `torch.cuda.synchronize`, untimed warm-up once); per method the score tensor is `s` / `random_scores(b, g, seed=0)` / `gaussians.opacities`; per budget `select_top_k`, stock `self.decoder.forward`, assert kept `== budget_count`. `evaluate.py` mirrors `src/approach1/evaluate.py` with `build_wrapper_a2`. `scripts/a2_eval.sh` mirrors `scripts/a1_eval.sh`; args `[head_ckpt] [data_root] [index_json] [output_dir]`, passes `approach2.head_ckpt`. `on_test_end` prints a table (method, k, PSNR, SSIM, LPIPS, kept, scenes).

- [ ] **Step 4: Run test, verify it passes**

Run: `poetry run pytest tests/approach2/test_eval_smoke.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/approach2/model_wrapper_a2.py src/approach2/evaluate.py scripts/a2_eval.sh tests/approach2/test_eval_smoke.py
git commit -m "feat(approach2): add budget eval with score, random and opacity cuts"
```

---

### Task 6: Train entry, fixed validation, overfit run (A2-06)

**Files:**
- Modify: `src/approach2/model_wrapper_a2.py` (add `validation_step`)
- Create: `src/approach2/train.py`, `scripts/a2_overfit.sh`, `assets/evaluation_index_a2_val10.json`, `docs/experiments/2026-11-12-a2-overfit5.md` (results)
- Test: `tests/approach2/test_val_loader.py`

**Interfaces:**
- Consumes: `build_wrapper_a2`, `TensorBoardImageLogger` (`src/approach1/tb_logger.py`), `DataModule`, `load_typed_root_config`.
- Produces: `build_val_loader(cfg_dict: DictConfig, a2_cfg: Approach2Cfg) -> DataLoader` in `src/approach2/train.py`: copy of `cfg_dict` with `dataset.roots=[a2_cfg.val_root]`, `dataset.view_sampler={name: evaluation, index_path: a2_cfg.val_index_path, num_context_views: 2}`, returns `DataModule(...).test_dataloader()` (every scene, fixed views, each check).

- [ ] **Step 1: Create `assets/evaluation_index_a2_val10.json`**

For each scene key in `/mnt/hdd/depthsplat/datasets/re10k_heldout10/evaluation_index.json`, copy the entry from `assets/evaluation_index_re10k.json` (upstream views, gap ~66, not the self-picked gap-45 views). Assert all 10 keys exist upstream.

- [ ] **Step 2: Write the failing test**

```python
def test_val_loader_yields_all_10_scenes_with_upstream_views():
    batches = list(build_val_loader(cfg_dict, a2_cfg))
    assert len(batches) == 10
    assert batches[0]["context"]["index"][0].tolist() == upstream[batches[0]["scene"][0]]["context"]
```

- [ ] **Step 3: Run test, verify it fails**

Run: `poetry run pytest tests/approach2/test_val_loader.py -v`
Expected: FAIL (`build_val_loader` missing).

- [ ] **Step 4: Implement `train.py`, `validation_step`, `scripts/a2_overfit.sh`**

`train.py` mirrors `src/approach1/train.py` with `build_wrapper_a2` and `trainer.fit(wrapper, train_dataloaders=data_module.train_dataloader(), val_dataloaders=build_val_loader(...), ckpt_path=cfg.checkpointing.load)`; prints trainable params. `validation_step`: per budget, `select_top_k` by score and by `random_scores(seed=0)`, stock render, log `val/psnr@{k}` and `val/psnr_minus_random@{k}`; image grid GT | k=1 | smallest k by score for the first scene only. `scripts/a2_overfit.sh` mirrors `scripts/a1_overfit.sh` (`+approach2=re10k_small`, `re10k_overfit5`, 3000 steps, `val_check_interval=250`, `every_n_train_steps=500`, default `OUT=outputs/approach2/overfit5`).

- [ ] **Step 5: Run test, verify it passes**

Run: `poetry run pytest tests/approach2/test_val_loader.py -v`
Expected: PASS.

- [ ] **Step 6: Memory probe**

Run: `scripts/a2_overfit.sh outputs/approach2/probe trainer.max_steps=20 data_loader.train.batch_size=<1|2|4>`
Expected: finishes; note peak memory per batch size (`torch.cuda.max_memory_allocated`) in the script header comment; pick the largest that fits under ~10 GB.

- [ ] **Step 7: Overfit run and eval**

Run: `scripts/a2_overfit.sh` then `scripts/a2_eval.sh outputs/approach2/overfit5/checkpoints/<last>.ckpt /mnt/hdd/depthsplat/datasets/re10k_overfit5` and the same on `/mnt/hdd/depthsplat/datasets/re10k_heldout10` with `assets/evaluation_index_a2_val10.json`.
Expected: TensorBoard shows `loss/masked` falling; eval tables for both sets. Write the tables, trainable params, head runtime and peak memory to `docs/experiments/2026-11-12-a2-overfit5.md`. Goal from the spec: score cut beats random and opacity cut at small `k` on the overfit scenes.

- [ ] **Step 8: Commit**

```bash
git add src/approach2/train.py src/approach2/model_wrapper_a2.py scripts/a2_overfit.sh assets/evaluation_index_a2_val10.json tests/approach2/test_val_loader.py docs/experiments/2026-11-12-a2-overfit5.md
git commit -m "feat(approach2): add train entry, fixed validation and overfit results"
```

---

### Task 7: Full RE10K training (A2-07, needs user go-ahead)

**Files:**
- Create: `scripts/a2_train.sh`

- [ ] **Step 1: Write `scripts/a2_train.sh`**

Mirrors `scripts/a2_overfit.sh` with `dataset.roots=[/mnt/hdd/depthsplat/datasets/re10k]`, batch size from Task 6 Step 6, `trainer.max_steps` and `val_check_interval` chosen with the user, `OUT=outputs/approach2/re10k_full`.

- [ ] **Step 2: Ask the user for go-ahead with steps, expected duration and output dir. Launch only after a yes.**

- [ ] **Step 3: Eval the final checkpoint** with `scripts/a2_eval.sh` on a fixed RE10K test subset with `assets/evaluation_index_re10k.json` (subset agreed with the user), and write `docs/experiments/<date>-a2-re10k-full.md`.

- [ ] **Step 4: Commit**

```bash
git add scripts/a2_train.sh docs/experiments/*a2-re10k-full.md
git commit -m "feat(approach2): add full RE10K training run and results"
```

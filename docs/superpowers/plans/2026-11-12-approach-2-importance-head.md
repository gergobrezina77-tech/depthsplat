# Approach 2: Learned Importance Head Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Train a U-Net score head on a frozen DepthSplat that ranks per-pixel Gaussians, so a global top-`N_k` cut by score keeps rendering quality, with an overfit run, a full RE10K run and a budget eval against random and opacity cuts.

**Architecture:** New code in `src/approach2/`, `config/approach2/`, `scripts/a2_*.sh`, `tests/approach2/`. `ModelWrapperA2` subclasses `ModelWrapperA1` (reuses `_render`, `_photo_loss`, `_psnr`, `configure_optimizers`, `on_before_optimizer_step`), owns a `ScoreHead`, and delegates its Lightning steps to one module each (`training.py`, `validation.py`, `evaluation.py`) so those tickets can be built in parallel. A forward hook on `encoder.gaussian_head` provides the head input. Training renders with MaskGaussian's mask rasterizer; eval gathers the kept Gaussians and renders with the stock decoder.

**Tech Stack:** Python 3.10, torch 2.4.0, pytorch-lightning 2.4, Hydra 1.3, `diff_gaussian_rasterization` (stock), `mask_diff_gaussian_rasterization` (new), TensorBoard, pytest.

**Spec:** `docs/superpowers/specs/2026-11-12-approach-2-importance-head-design.md`

## Global Constraints

- DepthSplat fully frozen, incl. `gaussian_head`. Trainable: `score_head.*` only.
- Checkpoint: `/mnt/hdd/depthsplat/pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth`, 256x256, 2 context views, `upsample_factor=4`, `lowest_feature_resolution=4`, `supervise_intermediate_depth: false`, `return_depth: true`.
- Head input: raw `gaussian_head` output (37 ch) + log depth (1 ch) = 38 ch, full resolution, per view.
- Budget: `N_k = floor(k * H * W * V)` via `src/approach1/budget.py:budget_count`, global top-`N_k` per scene (not over the batch).
- Training mask: Gumbel top-k on standardized scores, `p = 1 - exp(-exp((z - tau) / T))`, `m = m_hard + (p - p.detach())`, `k ~ U[k_min, 1]`, `k_min = 0.01`, `T = 1.0` default.
- Loss: upstream photometric (`mse` + `lpips` 0.05) on target views. No full-set term.
- Eval renders with the stock rasterizer only. Budgets `{0.05, 0.1, 0.25, 0.5, 1.0}`. Methods `score`, `random`, `opacity`. Kept count must equal `budget_count`.
- Mask rasterizer: `git+https://github.com/kaikai23/mask-diff-gaussian-rasterization@540dc11ef44c64f4c55fb894b3a4879de0cebfe7`.
- No edits to upstream files under `src/` or `config/`, and none to `src/approach1/`.
- GPU: RTX 2080 Ti, 11 GB, fp32. One GPU is shared by all workers: every command that loads DepthSplat or runs CUDA tests is wrapped as `flock /tmp/depthsplat_gpu.lock <command>`, so GPU tests of parallel tickets run one after another.
- Run via `poetry run ...`. The full RE10K run (A2-14) needs explicit user go-ahead.

## Review Focus

- Batch with 2 scenes: mask and top-`N_k` are chosen per scene, never over the batch (test in A2-04).
- Lightning calls `train()` on the whole wrapper: the encoder must stay in `eval()` (`mv_unimatch.py` branches on `self.training`) (test in A2-08).
- Head-only checkpoint saved during training must load into a fresh wrapper for eval (test in A2-08).
- Extreme budgets: `k = 1e-6` keeps exactly 1 Gaussian, `k = 1.0` keeps all, both with finite gradients (test in A2-04).
- `supervise_intermediate_depth: true` repeats the batch after the hook fires and silently misaligns depth and features: building must raise (test in A2-08).

## Shared Interfaces

Defined once here; the owning ticket implements them, every other ticket uses these exact names.

| Name | Signature | Owner |
|---|---|---|
| `Approach2Cfg` | dataclass: `k_min: float = 0.01`, `k_max: float = 1.0`, `temperature: float = 1.0`, `head_channels: list[int] = [32, 64, 96, 128]`, `eval_budgets: list[float] = [0.05, 0.1, 0.25, 0.5, 1.0]`, `eval_methods: list[str] = ["score", "random", "opacity"]`, `head_ckpt: str \| None = None`, `val_root: str = "/mnt/hdd/depthsplat/datasets/re10k_heldout10"`, `val_index_path: str = "assets/evaluation_index_a2_val10.json"` | A2-02 |
| `load_approach2_cfg` | `(cfg_dict: DictConfig) -> Approach2Cfg`, reads key `approach2` | A2-02 |
| `ScoreHead` | `(in_channels: int = 38, channels: list[int] = [32, 64, 96, 128])`, `forward(x: [n, c, h, w]) -> [n, 1, h, w]` | A2-03 |
| `gumbel_topk_mask` | `(s: [b, g], k: float, temperature: float, generator: torch.Generator \| None = None) -> tuple[m: Float[b, g], m_hard: Bool[b, g]]` | A2-04 |
| `gather_gaussians` | `(gaussians: Gaussians, idx: Int64[b, n]) -> Gaussians` | A2-04 |
| `select_top_k` | `(gaussians: Gaussians, scores: [b, g], k: float) -> Gaussians` | A2-04 |
| `random_scores` | `(b: int, g: int, seed: int, device: torch.device) -> [b, g]` | A2-04 |
| `HeadInputHook` | `(module: nn.Module)`; `pop() -> Tensor` (returns and clears, `RuntimeError` if empty); `remove() -> None` | A2-05 |
| `build_head_input` | `(raw: [bv, 37, h, w], depths: [b, v, h, w]) -> [bv, 38, h, w]` | A2-05 |
| `scores_to_gaussian_order` | `(s: [bv, 1, h, w], b: int, v: int) -> [b, v*h*w]` in `(v h w)` order | A2-05 |
| `build_val_loader` | `(cfg_dict: DictConfig, root: str, index_path: str) -> DataLoader` | A2-06 |
| `render_cuda_masked` | as `render_cuda` (`src/model/decoder/cuda_splatting.py:46`) plus `gaussian_masks: [batch, gaussian]`, `use_sh=True` only | A2-07 |
| `render_masked` | `(decoder: DecoderSplattingCUDA, gaussians: Gaussians, masks: [b, g], target: dict) -> [b, v, 3, h, w]` | A2-07 |
| `ModelWrapperA2` | see A2-08; attributes `score_head`, `head_hook`, `a2_cfg`, `a2_rows: dict[tuple[str, float], list[dict]]`, `a2_scenes: set[str]`, `a2_warm: bool`, `a2_val_image_done: bool`; method `scores(batch) -> tuple[Gaussians, [b, g]]` | A2-08 |
| `build_wrapper_a2` | `(cfg_dict: DictConfig) -> tuple[ModelWrapperA2, DataModule, RootCfg, Approach2Cfg]` | A2-08 |
| `masked_training_step` | `(wrapper: ModelWrapperA2, batch: BatchedExample) -> Tensor` | A2-09 |
| `budget_test_step` | `(wrapper: ModelWrapperA2, batch: BatchedExample) -> None` | A2-10 |
| `write_budget_tables` | `(wrapper: ModelWrapperA2, out_dir: Path) -> list[dict]` | A2-10 |
| `budget_validation_step` | `(wrapper: ModelWrapperA2, batch: BatchedExample) -> None` | A2-11 |

## Dependency Graph

| Wave | Tickets (parallel within a wave) | Depends on |
|---|---|---|
| 1 | A2-01 install, A2-02 config, A2-03 head, A2-04 select, A2-05 features, A2-06 val loader | – |
| 1.5 | A2-07 masked render | A2-01 |
| 2 | A2-08 wrapper skeleton + builder | A2-02, A2-03, A2-05 |
| 3 | A2-09 training step, A2-10 eval, A2-11 validation step | A2-08, A2-04; A2-09 also A2-07 |
| 4 | A2-12 train entry | A2-06, A2-09, A2-11 |
| 5 | A2-13 memory probe + overfit run + results | A2-10, A2-12 |
| 6 | A2-14 full RE10K run (user go-ahead) | A2-13 |

---

### A2-01: Mask rasterizer install

**Files:** Create `scripts/install_mask_rasterizer.sh`, `src/approach2/__init__.py`

- [ ] **Step 1: Write `scripts/install_mask_rasterizer.sh`**

Copy of `scripts/install_rasterizer.sh` with idempotence check `import mask_diff_gaussian_rasterization` and the install URL from Global Constraints. Same `CC/CXX/CUDAHOSTCXX=gcc-13/g++-13`, `TORCH_CUDA_ARCH_LIST=7.5`, `NVCC_APPEND_FLAGS=-include cstdint`.

- [ ] **Step 2: Run it and verify**

Run: `scripts/install_mask_rasterizer.sh && poetry run python -c "import mask_diff_gaussian_rasterization, diff_gaussian_rasterization; print('ok')"`
Expected: `ok` (both packages import side by side). Second run of the script prints "already installed".

- [ ] **Step 3: Commit**

```bash
git add scripts/install_mask_rasterizer.sh src/approach2/__init__.py
git commit -m "build(approach2): add mask rasterizer install script"
```

---

### A2-02: Config group

**Files:** Create `config/approach2/re10k_small.yaml`, `src/approach2/config.py`; Test `tests/approach2/test_config.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_load_approach2_cfg_defaults():
    cfg = load_approach2_cfg(OmegaConf.create({}))
    assert cfg.k_min == 0.01 and cfg.temperature == 1.0 and cfg.head_ckpt is None

def test_yaml_composes():
    # compose(config_name="main", overrides=["+experiment=re10k", "+approach2=re10k_small"])
    assert cfg_dict.model.encoder.supervise_intermediate_depth is False
    assert cfg_dict.model.encoder.return_depth is True
    assert load_approach2_cfg(cfg_dict).eval_budgets == [0.05, 0.1, 0.25, 0.5, 1.0]
```

- [ ] **Step 2: Run, verify fail:** `poetry run pytest tests/approach2/test_config.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement `Approach2Cfg`, `load_approach2_cfg`, yaml**

Dataclass and loader as in Shared Interfaces, following `src/approach1/config.py`. The yaml mirrors `config/approach1/re10k_small.yaml` (same `model.encoder`, `checkpointing.pretrained_model`, `data_loader`, `trainer`, `train`, `dataset.view_sampler.warm_up_steps: 0`, `wandb.mode: disabled`) with `model.encoder.return_depth: true`, `optimizer.lr: 3.e-4`, an `approach2:` block with the dataclass defaults, `output_dir: outputs/approach2/default`.

- [ ] **Step 4: Run, verify pass:** same command → 2 PASS.

- [ ] **Step 5: Commit**

```bash
git add config/approach2 src/approach2/config.py tests/approach2/test_config.py
git commit -m "feat(approach2): add config group"
```

---

### A2-03: Score head

**Files:** Create `src/approach2/score_head.py`; Test `tests/approach2/test_score_head.py`

- [ ] **Step 1: Write the failing tests** (CPU)

```python
def test_score_head_shape():
    assert ScoreHead()(torch.randn(2, 38, 256, 256)).shape == (2, 1, 256, 256)

def test_score_head_param_budget():
    n = sum(p.numel() for p in ScoreHead().parameters())
    assert 200_000 < n < 1_500_000

def test_score_head_rejects_bad_resolution():
    with pytest.raises(ValueError):
        ScoreHead()(torch.randn(1, 38, 100, 100))
```

- [ ] **Step 2: Run, verify fail:** `poetry run pytest tests/approach2/test_score_head.py -v` → FAIL.

- [ ] **Step 3: Implement `ScoreHead`**

Stem `3x3 conv + GroupNorm(8) + GELU` (`in_channels` -> `channels[0]`). Encoder: 4 levels at 1, 1/2, 1/4, 1/8, each two `3x3 conv + GELU`, downsample with stride-2 conv. Decoder: bilinear upsampling, concat skip, two `3x3 conv + GELU`. Final `1x1 conv` -> 1. `ValueError` if `h % 8` or `w % 8`.

- [ ] **Step 4: Run, verify pass:** same command → 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/approach2/score_head.py tests/approach2/test_score_head.py
git commit -m "feat(approach2): add U-Net score head"
```

---

### A2-04: Selection

**Files:** Create `src/approach2/select.py`; Test `tests/approach2/test_select.py`

Consumes `budget_count` (`src/approach1/budget.py`), `Gaussians` (`src/model/types.py`).

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

- [ ] **Step 1: Write the failing tests** (CPU)

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
    # g = 1000; scene 0 scores favor indices 0..9, scene 1 favors 990..999; k = 0.01
    kept = select_top_k(gaussians, scores, 0.01)
    assert kept.opacities.shape == (2, 10)
    # kept means identify the indices: set == range(10) for scene 0, range(990, 1000) for scene 1

def test_random_scores_seeded():
    assert torch.equal(random_scores(1, 50, 0, "cpu"), random_scores(1, 50, 0, "cpu"))
```

- [ ] **Step 2: Run, verify fail:** `poetry run pytest tests/approach2/test_select.py -v` → FAIL.

- [ ] **Step 3: Implement `select.py`**

`gather_gaussians` follows the inner `gather` of `select_top_k_by_opacity` (`src/approach1/budget.py:17-27`); `select_top_k` = `topk` of `scores` with `budget_count`, then `gather_gaussians`; `random_scores` uses a seeded `torch.Generator`.

- [ ] **Step 4: Run, verify pass:** same command → all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/approach2/select.py tests/approach2/test_select.py
git commit -m "feat(approach2): add Gumbel top-k mask and score selection"
```

---

### A2-05: Head input features

**Files:** Create `src/approach2/features.py`; Test `tests/approach2/test_features.py`

- [ ] **Step 1: Write the failing tests** (CPU)

```python
def test_hook_pop_clears():
    module = nn.Identity(); hook = HeadInputHook(module)
    module(torch.ones(1))
    assert hook.pop().shape == (1,)
    with pytest.raises(RuntimeError):
        hook.pop()

def test_build_head_input():
    out = build_head_input(torch.zeros(4, 37, 8, 8), torch.full((2, 2, 8, 8), math.e))
    assert out.shape == (4, 38, 8, 8) and torch.allclose(out[:, 37], torch.ones(4, 8, 8))

def test_scores_to_gaussian_order():
    s = torch.arange(2 * 2 * 4 * 4.).reshape(4, 1, 4, 4)  # (b v) 1 h w, b=2, v=2
    out = scores_to_gaussian_order(s, b=2, v=2)
    assert out.shape == (2, 32) and out[1, 16] == s[3, 0, 0, 0]
```

- [ ] **Step 2: Run, verify fail:** `poetry run pytest tests/approach2/test_features.py -v` → FAIL.

- [ ] **Step 3: Implement `features.py`**

`HeadInputHook` uses `register_forward_hook` and stores the output; `build_head_input` appends `log(depth.clamp_min(1e-6))` rearranged `b v h w -> (b v) () h w`; `scores_to_gaussian_order` = `rearrange(s, "(b v) () h w -> b (v h w)")`.

- [ ] **Step 4: Run, verify pass:** same command → 3 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/approach2/features.py tests/approach2/test_features.py
git commit -m "feat(approach2): add head input hook and score ordering"
```

---

### A2-06: Validation index and loader

**Files:** Create `assets/evaluation_index_a2_val10.json`, `src/approach2/val_data.py`; Test `tests/approach2/test_val_data.py`

- [ ] **Step 1: Create `assets/evaluation_index_a2_val10.json`**

For each scene key in `/mnt/hdd/depthsplat/datasets/re10k_heldout10/evaluation_index.json`, copy the entry from `assets/evaluation_index_re10k.json` (upstream views, not the self-picked gap-45 views). Assert all 10 keys exist upstream and are non-null.

- [ ] **Step 2: Write the failing test** (CPU, reads `/mnt`)

```python
def test_val_loader_yields_10_scenes_with_upstream_views():
    batches = list(build_val_loader(cfg_dict, "/mnt/hdd/depthsplat/datasets/re10k_heldout10",
                                    "assets/evaluation_index_a2_val10.json"))
    assert len(batches) == 10
    b = batches[0]
    assert b["context"]["index"][0].tolist() == upstream[b["scene"][0]]["context"]
```

`cfg_dict` is composed with `+experiment=re10k` and `+approach1=re10k_small` (A2-02 not required).

- [ ] **Step 3: Run, verify fail:** `poetry run pytest tests/approach2/test_val_data.py -v` → FAIL.

- [ ] **Step 4: Implement `build_val_loader`**

Deep copy of `cfg_dict` with `dataset.roots=[root]`, `dataset.view_sampler={name: evaluation, index_path: index_path, num_context_views: 2}`, then `load_typed_root_config` and `DataModule(cfg.dataset, cfg.data_loader, StepTracker(), global_rank=0).test_dataloader()`.

- [ ] **Step 5: Run, verify pass:** same command → PASS.

- [ ] **Step 6: Commit**

```bash
git add assets/evaluation_index_a2_val10.json src/approach2/val_data.py tests/approach2/test_val_data.py
git commit -m "feat(approach2): add fixed validation set with upstream views"
```

---

### A2-07: Masked render

**Files:** Create `src/approach2/mask_render.py`; Test `tests/approach2/test_mask_render.py`

Depends on A2-01 (package installed).

- [ ] **Step 1: Write the failing tests** (CUDA; synthetic scene: camera at the origin looking down +z, 500 random Gaussians at z in [2, 4], 64x64 image)

```python
def test_mask_ones_matches_stock():
    assert (render_cuda_masked(..., masks=ones) - render_cuda(...)).abs().max() < 1e-4

def test_mask_zero_matches_removed():
    # masks zero on 100 Gaussians vs render_cuda on the other 400 only
    assert (img_masked - img_removed).abs().max() < 1e-4

def test_mask_grad_reaches_masked_gaussians():
    # masks = m_hard + (p - p.detach()) with 100 zeros; loss = img.mean(); backward
    assert p.grad.shape == p.shape and (p.grad[dropped_idx].abs() > 0).any()

def test_render_masked_shape():
    # Gaussians batch b=1, target with v=2 views, 64x64
    assert render_masked(decoder, gaussians, masks, target).shape == (1, 2, 3, 64, 64)
```

- [ ] **Step 2: Run, verify fail:** `flock /tmp/depthsplat_gpu.lock poetry run pytest tests/approach2/test_mask_render.py -v` → FAIL.

- [ ] **Step 3: Implement `render_cuda_masked` and `render_masked`**

Copy `render_cuda` (incl. the `scale_invariant` block), use the mask package's `GaussianRasterizationSettings`/`GaussianRasterizer`, call with keyword arguments only and `masks=gaussian_masks[i][:, None].float().contiguous()` (`[P, 1]` float32; the backward allocates `{P, 1}`, `rasterize_points.cu:158`, and has no dtype check). `ValueError` if masks is `None`. `render_masked` mirrors `DecoderSplattingCUDA.forward` (`src/model/decoder/decoder_splatting_cuda.py:34-58`) and repeats masks over target views like opacities.

- [ ] **Step 4: Run, verify pass:** same command → 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/approach2/mask_render.py tests/approach2/test_mask_render.py
git commit -m "feat(approach2): add masked render with mask rasterizer"
```

---

### A2-08: Wrapper skeleton and builder

**Files:** Create `src/approach2/build.py`, `src/approach2/model_wrapper_a2.py`; Test `tests/approach2/test_wrapper.py`

Consumes A2-02, A2-03, A2-05, `ModelWrapperA1`, `Approach1Cfg`, `freeze_all_but`, `set_frozen_eval`.

`ModelWrapperA2(ModelWrapperA1)`:
- `__init__(optimizer_cfg, test_cfg, train_cfg, encoder, encoder_visualizer, decoder, losses, step_tracker, a2_cfg: Approach2Cfg, eval_data_cfg=None)`: parent gets `Approach1Cfg(trainable_prefixes=["score_head."], eval_budgets=a2_cfg.eval_budgets)`; creates `score_head`, `head_hook = HeadInputHook(encoder.gaussian_head)`, `a2_rows = {}`, `a2_scenes = set()`, `a2_warm = False`; `self.strict_loading = False`.
- `scores(batch)`: encoder under `torch.no_grad()` returns `{"gaussians", "depths"}`; `build_head_input(head_hook.pop(), depths)`; `score_head`; `scores_to_gaussian_order`.
- `train(mode=True)`: `super().train(mode)`, `set_frozen_eval(self, ["score_head."])`, return `self`.
- `on_save_checkpoint(checkpoint)`: keep only `score_head.*` in `checkpoint["state_dict"]`.
- `on_validation_epoch_start()`: sets `self.a2_val_image_done = False` (used by A2-11).
- `training_step`, `validation_step`, `test_step`, `on_test_end`: one line each, importing inside the method and calling `masked_training_step`, `budget_validation_step`, `budget_test_step`, `write_budget_tables(self, Path(get_cfg()["output_dir"]) / "metrics")`.

`build_wrapper_a2`: mirrors `src/approach1/build.py`; `ValueError` if `supervise_intermediate_depth` or not `return_depth`; loads the pretrained checkpoint with `strict=False`, `RuntimeError` unless every missing key starts with `score_head.` and no key is unexpected; if `a2_cfg.head_ckpt` is set, loads its `score_head.*` keys strictly into `score_head`; calls `freeze_all_but(wrapper, ["score_head."])`.

- [ ] **Step 1: Write the failing tests** (CUDA, `re10k_overfit5`, batch size 1, module fixture as in `tests/approach1/test_train_step.py`)

```python
def test_only_head_trainable(built):
    assert {n.split(".")[0] for n, p in wrapper.named_parameters() if p.requires_grad} == {"score_head"}

def test_scores_shape_and_grad(built):
    gaussians, s = wrapper.scores(batch)
    assert s.shape == (1, 131072) and s.requires_grad and gaussians.opacities.shape == (1, 131072)

def test_encoder_stays_eval_after_train(built):
    wrapper.train()
    assert not wrapper.encoder.training and wrapper.score_head.training

def test_depth_matches_gaussian_means(built):
    # camera-space z of gaussians.means for context view v at index v*H*W + y*W + x == depths[0, v, y, x]
    assert torch.allclose(z_cam, depths[0].flatten(), rtol=1e-3)

def test_head_only_checkpoint_roundtrip(built, tmp_path):
    ckpt = {"state_dict": wrapper.state_dict()}; wrapper.on_save_checkpoint(ckpt)
    assert all(k.startswith("score_head.") for k in ckpt["state_dict"])
    # torch.save to tmp_path; fresh build_wrapper_a2 with approach2.head_ckpt=<file> has equal score_head weights

def test_build_rejects_intermediate_depth():
    with pytest.raises(ValueError):
        build_wrapper_a2(compose(..., "model.encoder.supervise_intermediate_depth=true"))
```

- [ ] **Step 2: Run, verify fail:** `flock /tmp/depthsplat_gpu.lock poetry run pytest tests/approach2/test_wrapper.py -v` → FAIL.

- [ ] **Step 3: Implement `build.py` and `model_wrapper_a2.py`**

- [ ] **Step 4: Run, verify pass:** same command → 6 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/approach2/build.py src/approach2/model_wrapper_a2.py tests/approach2/test_wrapper.py
git commit -m "feat(approach2): add wrapper skeleton and builder"
```

---

### A2-09: Masked training step

**Files:** Create `src/approach2/training.py`; Test `tests/approach2/test_training.py`

Consumes `ModelWrapperA2.scores`, `gumbel_topk_mask`, `render_masked`, `wrapper._photo_loss`, `wrapper._psnr`.

- [ ] **Step 1: Write the failing test** (CUDA, fixture as A2-08, `wrapper.log` replaced by a dict recorder)

```python
def test_masked_training_step(built):
    loss = masked_training_step(wrapper, batch); loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for n, p in wrapper.named_parameters() if n.startswith("score_head."))
    assert all(p.grad is None for n, p in wrapper.named_parameters() if not n.startswith("score_head."))
    assert {"loss/masked", "train/psnr_masked", "train/k", "train/kept", "train/score_std"} <= logged.keys()
```

- [ ] **Step 2: Run, verify fail:** `flock /tmp/depthsplat_gpu.lock poetry run pytest tests/approach2/test_training.py -v` → FAIL.

- [ ] **Step 3: Implement `masked_training_step`**

`data_shim`; `scores(batch)`; `k = k_min + (k_max - k_min) * rand`; `m, m_hard = gumbel_topk_mask(s, k, temperature)`; `color = render_masked(wrapper.decoder, gaussians, m, batch["target"])`; loss via `wrapper._photo_loss(DecoderOutput(color, None), batch, gaussians)`. Logs: `loss/masked`, `train/psnr_masked`, `train/k`, `train/kept` (`m_hard.sum(1).float().mean()`), `train/score_std`, `info/global_step`. Step tracker update and periodic print as in `ModelWrapperA1.training_step`.

- [ ] **Step 4: Run, verify pass:** same command → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/approach2/training.py tests/approach2/test_training.py
git commit -m "feat(approach2): add masked training step"
```

---

### A2-10: Budget eval

**Files:** Create `src/approach2/evaluation.py`, `src/approach2/evaluate.py`, `scripts/a2_eval.sh`; Test `tests/approach2/test_eval_smoke.py`

Consumes `ModelWrapperA2.scores`, `select_top_k`, `random_scores`, `budget_count`, `compute_psnr/ssim/lpips`, `build_wrapper_a2`.

Output: `<output_dir>/metrics/budget_table.json` = `{"scenes": [...], "rows": [{"method", "k", "psnr", "ssim", "lpips", "kept", "n_total", "scenes", "head_time"}]}` and `per_scene.json` (one row per scene, method, k).

- [ ] **Step 1: Write the failing test** (mirrors `tests/approach1/test_eval_smoke.py`: subprocess `src.approach2.evaluate`, `+approach2=re10k_small`, `approach2.eval_budgets=[0.1,1.0]`, `re10k_overfit5`)

```python
data = json.loads((out / "metrics" / "budget_table.json").read_text())
assert len(data["scenes"]) == 5
assert {(r["method"], r["k"]) for r in data["rows"]} == {(m, k) for m in ["score", "random", "opacity"] for k in [0.1, 1.0]}
assert all(r["kept"] == {0.1: 13107, 1.0: 131072}[r["k"]] for r in data["rows"])
```

- [ ] **Step 2: Run, verify fail:** `flock /tmp/depthsplat_gpu.lock poetry run pytest tests/approach2/test_eval_smoke.py -v` → FAIL.

- [ ] **Step 3: Implement**

`budget_test_step`: one timed `scores(batch)` per scene (`torch.cuda.synchronize`, untimed warm-up once via `a2_warm`); score tensor per method: `s` / `random_scores(b, g, 0, device)` / `gaussians.opacities`; per budget `select_top_k`, stock `wrapper.decoder.forward`, assert kept `== budget_count`, append to `a2_rows[(method, k)]`. `write_budget_tables` aggregates means, writes both JSON files and `peak_memory.json`, prints a table (method, k, PSNR, SSIM, LPIPS, kept, scenes), clears state. `evaluate.py` mirrors `src/approach1/evaluate.py` with `build_wrapper_a2`. `scripts/a2_eval.sh` mirrors `scripts/a1_eval.sh`, args `[head_ckpt] [data_root] [index_json] [output_dir]`, passes `approach2.head_ckpt`.

- [ ] **Step 4: Run, verify pass:** same command → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/approach2/evaluation.py src/approach2/evaluate.py scripts/a2_eval.sh tests/approach2/test_eval_smoke.py
git commit -m "feat(approach2): add budget eval with score, random and opacity cuts"
```

---

### A2-11: Validation step

**Files:** Create `src/approach2/validation.py`; Test `tests/approach2/test_validation.py`

Consumes `ModelWrapperA2.scores`, `select_top_k`, `random_scores`, `compute_psnr`, `wrapper.logger.log_image`.

- [ ] **Step 1: Write the failing test** (CUDA, fixture as A2-08, `wrapper.log` recorder, `wrapper.logger` stub with `log_image`)

```python
def test_budget_validation_step(built):
    budget_validation_step(wrapper, batch)
    for k in wrapper.a2_cfg.eval_budgets:
        assert f"val/psnr@{k}" in logged and f"val/psnr_minus_random@{k}" in logged
    assert len(images) == 1
```

- [ ] **Step 2: Run, verify fail:** `flock /tmp/depthsplat_gpu.lock poetry run pytest tests/approach2/test_validation.py -v` → FAIL.

- [ ] **Step 3: Implement `budget_validation_step`**

Under `torch.no_grad()`: `scores(batch)`; per budget, score cut and `random_scores(b, g, 0, device)` cut, stock render, log `val/psnr@{k}` and `val/psnr_minus_random@{k}`. Image grid GT | k=1 | smallest k by score (as `ModelWrapperA1.validation_step`) only while `wrapper.a2_val_image_done` is False, then set it True.

- [ ] **Step 4: Run, verify pass:** same command → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/approach2/validation.py tests/approach2/test_validation.py
git commit -m "feat(approach2): add budget validation step"
```

---

### A2-12: Train entry

**Files:** Create `src/approach2/train.py`, `scripts/a2_overfit.sh`; Test `tests/approach2/test_train_smoke.py`

Consumes `build_wrapper_a2`, `build_val_loader`, `TensorBoardImageLogger` (`src/approach1/tb_logger.py`).

- [ ] **Step 1: Write the failing test** (subprocess `scripts/a2_overfit.sh <tmp> trainer.max_steps=4 trainer.val_check_interval=2 checkpointing.every_n_train_steps=4 data_loader.train.batch_size=1`)

```python
assert list((out / "checkpoints").glob("*.ckpt"))
assert list((out / "tensorboard").rglob("events.out.tfevents.*"))
```

- [ ] **Step 2: Run, verify fail:** `flock /tmp/depthsplat_gpu.lock poetry run pytest tests/approach2/test_train_smoke.py -v` → FAIL.

- [ ] **Step 3: Implement `train.py` and `scripts/a2_overfit.sh`**

`train.py` mirrors `src/approach1/train.py` with `build_wrapper_a2`, prints trainable and total params, and calls `trainer.fit(wrapper, train_dataloaders=data_module.train_dataloader(), val_dataloaders=build_val_loader(cfg_dict, a2_cfg.val_root, a2_cfg.val_index_path), ckpt_path=cfg.checkpointing.load)`. `scripts/a2_overfit.sh` mirrors `scripts/a1_overfit.sh` (`src.approach2.train`, `+approach2=re10k_small`, `re10k_overfit5`, 3000 steps, `val_check_interval=250`, `every_n_train_steps=500`, default `OUT=outputs/approach2/overfit5`).

- [ ] **Step 4: Run, verify pass:** same command → PASS.

- [ ] **Step 5: Commit**

```bash
git add src/approach2/train.py scripts/a2_overfit.sh tests/approach2/test_train_smoke.py
git commit -m "feat(approach2): add train entry and overfit script"
```

---

### A2-13: Memory probe, overfit run, results

**Files:** Modify `scripts/a2_overfit.sh` (batch size, header comment); Create `docs/experiments/2026-11-12-a2-overfit5.md`

- [ ] **Step 1: Memory probe**

Run: `flock /tmp/depthsplat_gpu.lock scripts/a2_overfit.sh outputs/approach2/probe_bs<N> trainer.max_steps=20 data_loader.train.batch_size=<N>` for N in 1, 2, 4.
Expected: each finishes; record `torch.cuda.max_memory_allocated` per N in the script header; set the largest N under ~10 GB as default.

- [ ] **Step 2: Overfit run**

Run: `flock /tmp/depthsplat_gpu.lock scripts/a2_overfit.sh`
Expected: finishes 3000 steps; TensorBoard shows `loss/masked` falling and `val/psnr_minus_random@k`.

- [ ] **Step 3: Eval on overfit and validation scenes**

Run: `scripts/a2_eval.sh <last ckpt> /mnt/hdd/depthsplat/datasets/re10k_overfit5` and `scripts/a2_eval.sh <last ckpt> /mnt/hdd/depthsplat/datasets/re10k_heldout10 assets/evaluation_index_a2_val10.json outputs/approach2/eval_val10`
Expected: two budget tables.

- [ ] **Step 4: Write `docs/experiments/2026-11-12-a2-overfit5.md`**

Setup, command, both tables (method x k: PSNR, SSIM, LPIPS, kept), trainable params, head runtime, peak memory, and whether the score cut beats random and opacity at small `k` on the overfit scenes (spec goal).

- [ ] **Step 5: Commit**

```bash
git add scripts/a2_overfit.sh docs/experiments/2026-11-12-a2-overfit5.md
git commit -m "docs(approach2): add overfit5 results"
```

---

### A2-14: Full RE10K training (needs user go-ahead)

**Files:** Create `scripts/a2_train.sh`, `docs/experiments/<date>-a2-re10k-full.md`

- [ ] **Step 1: Write `scripts/a2_train.sh`**: mirrors `scripts/a2_overfit.sh` with `dataset.roots=[/mnt/hdd/depthsplat/datasets/re10k]`, batch size from A2-13, `OUT=outputs/approach2/re10k_full`; `trainer.max_steps` and `val_check_interval` agreed with the user.

- [ ] **Step 2: Ask the user for go-ahead** with steps, expected duration and output dir. Launch only after a yes.

- [ ] **Step 3: Eval the final checkpoint** with `scripts/a2_eval.sh` on a fixed RE10K test subset with `assets/evaluation_index_re10k.json` (subset agreed with the user); write the results doc.

- [ ] **Step 4: Commit**

```bash
git add scripts/a2_train.sh docs/experiments/*a2-re10k-full.md
git commit -m "feat(approach2): add full RE10K training run and results"
```

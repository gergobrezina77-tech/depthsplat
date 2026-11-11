# Approach 1: Opacity Fine-Tune Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fine-tune only `gaussian_head` of the small DepthSplat model so that a top-`k` cut by opacity keeps rendering quality at any budget `k`, with an overfit feasibility run, a long run with TensorBoard, and a budget eval.

**Architecture:** All new code lives in `src/approach1/` and `config/approach1/`; upstream files stay unchanged. `ModelWrapperA1` subclasses upstream `ModelWrapper` and overrides `training_step`, `validation_step`, `test_step` and `configure_optimizers`. Two Hydra entry points (`src.approach1.train`, `src.approach1.evaluate`) share one builder.

**Tech Stack:** Python 3.10, torch 2.4.0, pytorch-lightning 2.4.0, Hydra 1.3.2, `diff_gaussian_rasterization`, TensorBoard, pytest.

**Spec:** `docs/approach-1-opacity-finetune.md`

## Global Constraints

- Trainable: `encoder.gaussian_head.*` only. Every other parameter has `requires_grad=False` and its module stays in `eval()` mode.
- Budget: `N = R * I_size`, `R = H * W`, `I_size` = context views; kept count `N_k = floor(k * R * I_size)`, `k` in `(0, 1]`.
- Selection: global top-`N_k` by opacity per scene (over all views), not per view.
- Any budget at inference: training samples `k ~ U[k_min, 1]` (continuous); eval accepts any list of `k`.
- Loss: photometric only (upstream `mse` + `lpips`, weight 0.05). No depth or 3D labels.
- Every eval row logs the actual kept count; it must equal `N_k`.
- Checkpoint: `/mnt/hdd/depthsplat/pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth`, `image_shape=[256, 256]`, 2 context views, `upsample_factor=4`, `lowest_feature_resolution=4`.
- GPU: RTX 2080 Ti, 11 GB, fp32 only (no bf16).
- No edits to upstream files under `src/` or `config/`; only `pyproject.toml` / `poetry.lock` change outside new paths.
- Run everything via `poetry run ...`. Long training runs need explicit user go-ahead.

## Review Focus

- `k` tiny or invalid (`k=1e-6`, `k=0`, `k=1.5`): `budget_count` returns at least 1 for `k > 0` and raises `ValueError` outside `(0, 1]` (test in Task 2).
- Batch with >1 scene: top-`k` is chosen per scene, not over the batch (test in Task 2).
- Lightning calls `model.train()` every epoch: frozen modules must stay in `eval()`, because `mv_unimatch.py:534` branches on `self.training` (test in Task 5).
- Frozen weights drift: after training, every non-head tensor must equal the pretrained checkpoint (check in Task 9).
- Eval index scenes missing from the data are skipped silently by the RE10K loader: the eval table must print the number of evaluated scenes (Task 6).

## File Map

| File | Responsibility | Task |
|---|---|---|
| `pyproject.toml` | add `tensorboard`, `pytest` dev deps | 1 |
| `src/approach1/__init__.py` | package marker | 2 |
| `src/approach1/budget.py` | `budget_count`, `select_top_k_by_opacity` | 2 |
| `src/approach1/make_subset.py` | build 5-scene RE10K subset + eval index | 3 |
| `config/approach1/re10k_small.yaml` | Hydra group for this approach | 4 |
| `src/approach1/config.py` | `Approach1Cfg`, `load_approach1_cfg` | 4 |
| `src/approach1/freeze.py` | `freeze_all_but`, `set_frozen_eval` | 5 |
| `src/approach1/build.py` | `build_wrapper` shared by both entry points | 6 |
| `src/approach1/model_wrapper_a1.py` | `ModelWrapperA1` | 6, 7 |
| `src/approach1/evaluate.py` | Hydra eval entry (any budget) | 6 |
| `src/approach1/tb_logger.py` | `TensorBoardImageLogger` | 8 |
| `src/approach1/train.py` | Hydra train entry | 8 |
| `scripts/a1_overfit.sh`, `scripts/a1_train.sh`, `scripts/a1_eval.sh` | one-command runs | 9, 10, 6 |
| `tests/approach1/*.py` | pytest tests | 2-8 |

Execution order: `1 -> {2, 3, 4} -> 5 -> 6 -> 7 -> 8 -> 9 -> 10 -> 11`. Tasks 2, 3 and 4 can run in parallel, and Task 5 can run in parallel with Task 6. Task 12 runs only if Gate G1 (Task 9) fails.

---

### Task 1: Dev tooling (TensorBoard, pytest)

**Files:**
- Modify: `pyproject.toml`, `poetry.lock`

- [ ] **Step 1:** `poetry add --group dev tensorboard pytest`
- [ ] **Step 2:** Verify: `poetry run python -c "import tensorboard, pytest; print(tensorboard.__version__, pytest.__version__)"` prints two versions.
- [ ] **Step 3:** Verify torch is unchanged: `poetry run python -c "import torch; print(torch.__version__)"` prints `2.4.0+cu124`.
- [ ] **Step 4:** Commit `build: add tensorboard and pytest dev deps`.

### Task 2: Budget selection

**Files:**
- Create: `src/approach1/__init__.py`, `src/approach1/budget.py`
- Test: `tests/approach1/test_budget.py`

**Interfaces:**
- Consumes: `src.model.types.Gaussians` (`means [b g 3]`, `covariances [b g 3 3]`, `harmonics [b g 3 d_sh]`, `opacities [b g]`).
- Produces:
  - `budget_count(num_gaussians: int, k: float) -> int`
  - `select_top_k_by_opacity(gaussians: Gaussians, k: float) -> Gaussians`, which gathers along `g` per batch item and keeps gradients for the kept entries.

- [ ] **Step 1: Write failing tests**

```python
def test_budget_count_formula():
    assert budget_count(256 * 256 * 2, 0.1) == 13107
    assert budget_count(256 * 256 * 2, 1.0) == 131072

def test_budget_count_min_one_and_invalid():
    assert budget_count(100, 1e-6) == 1
    for k in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            budget_count(100, k)

def test_select_keeps_highest_opacity_per_scene():
    # batch of 2 with different opacity orders; k=0.25 of g=8 -> 2 kept each
    out = select_top_k_by_opacity(g, 0.25)
    assert out.opacities.shape == (2, 2)
    assert torch.equal(out.opacities.sort(dim=1).values, expected_top2_per_item)
    assert torch.equal(out.means, means_of_expected_indices)

def test_select_k1_is_identity_set():
    out = select_top_k_by_opacity(g, 1.0)
    assert out.opacities.shape == g.opacities.shape

def test_select_gradient_reaches_kept_opacities():
    out.opacities.sum().backward()
    assert (raw_opacities.grad != 0).sum() == kept_count
```

- [ ] **Step 2:** `poetry run pytest tests/approach1/test_budget.py -v` -> FAIL (import error).
- [ ] **Step 3:** Implement with `torch.topk(gaussians.opacities, n_k, dim=1)` and `torch.gather` on each field. `budget_count` = `max(1, math.floor(k * num_gaussians))`.
- [ ] **Step 4:** Re-run -> PASS.
- [ ] **Step 5:** Commit `feat(approach1): add opacity top-k budget selection`.

### Task 3: Overfit-5 subset builder

**Files:**
- Create: `src/approach1/make_subset.py`
- Test: `tests/approach1/test_make_subset.py`

**Interfaces:**
- Produces: CLI `poetry run python -m src.approach1.make_subset --src /mnt/hdd/depthsplat/datasets/re10k --split train --num-scenes 5 --seed 0 --out /mnt/hdd/depthsplat/datasets/re10k_overfit5`.
  It writes `<out>/train/000000.torch`, `<out>/test/000000.torch` (the same 5 scenes, because the upstream `val` stage reads `test/`), `index.json` in both, and `<out>/evaluation_index.json` with `{scene: {"context": [i, i+gap], "target": [3 indices inside]}}`.
- Function: `pick_eval_views(num_frames: int, gap: int = 45, num_targets: int = 3) -> dict[str, list[int]]`.

- [ ] **Step 1: Failing tests**: on a fake chunk of 7 scenes (each with >= 60 frames), check that the output has 5 scenes, that both `index.json` files map all 5 keys to `000000.torch`, and that the eval index contexts are 45 apart with targets strictly between them. Also check that a scene with fewer than 46 frames is skipped.
- [ ] **Step 2:** Run -> FAIL.
- [ ] **Step 3:** Implement: walk the chunks in sorted order and take the first 5 scenes with at least `gap + 1` frames. Pick the context start with `random.Random(seed)`.
- [ ] **Step 4:** Run -> PASS. Run the real CLI. Verify: `ls /mnt/hdd/depthsplat/datasets/re10k_overfit5/{train,test}` shows `000000.torch index.json`.
- [ ] **Step 5:** Commit `feat(approach1): add overfit subset builder`.

### Task 4: Approach-1 config

**Files:**
- Create: `config/approach1/re10k_small.yaml`, `src/approach1/config.py`
- Test: `tests/approach1/test_config.py`

**Interfaces:**
- Usage: `+experiment=re10k +approach1=re10k_small` (`# @package _global_`).
- Produces: `Approach1Cfg` dataclass and `load_approach1_cfg(cfg_dict: DictConfig) -> Approach1Cfg`, read from the `approach1:` key. Fields and defaults:
  - `trainable_prefixes: list[str] = ["encoder.gaussian_head."]`
  - `k_min: float = 0.01`, `k_max: float = 1.0`
  - `full_weight: float = 1.0`, `prefix_weight: float = 1.0`
  - `eval_budgets: list[float] = [0.05, 0.1, 0.25, 0.5, 1.0]`
  - `soft_mask: bool = False`, `soft_mask_temperature: float = 0.05`
- The yaml also sets:
  - `model.encoder.upsample_factor=4`, `lowest_feature_resolution=4`, `supervise_intermediate_depth=false`
  - `checkpointing.pretrained_model=<small ckpt>`
  - `optimizer.lr=1e-4`
  - `train.no_viz_video=true`, `train.eval_model_every_n_val=0`
  - `dataset.view_sampler.warm_up_steps=0`
  - `wandb.mode=disabled`, `output_dir=outputs/approach1/default`

- [ ] **Step 1: Failing test**: compose with `hydra.compose(config_name="main", overrides=["+experiment=re10k", "+approach1=re10k_small"])`. Assert that `load_approach1_cfg(cfg).eval_budgets == [0.05, 0.1, 0.25, 0.5, 1.0]`, that `load_typed_root_config(cfg)` succeeds, and that `cfg.model.encoder.upsample_factor == 4`.
- [ ] **Step 2:** Run -> FAIL. **Step 3:** implement. **Step 4:** Run -> PASS.
- [ ] **Step 5:** Commit `feat(approach1): add config group`.

### Task 5: Freeze utilities

**Files:**
- Create: `src/approach1/freeze.py`
- Test: `tests/approach1/test_freeze.py`

**Interfaces:**
- Produces:
  - `freeze_all_but(model: nn.Module, trainable_prefixes: list[str]) -> list[str]`: sets `requires_grad` and returns the trainable parameter names. Raises `ValueError` if no parameter matches.
  - `set_frozen_eval(model: nn.Module, trainable_prefixes: list[str]) -> None`: calls `eval()` on every submodule that has no trainable parameter.

- [ ] **Step 1: Failing tests**
  - On a toy `nn.Sequential`, check the trainable names and that `requires_grad` is set correctly.
  - On the real encoder built from the Task 4 config, check that all trainable names start with `encoder.gaussian_head.` and print the count.
  - Regression: after `model.train()` and then `set_frozen_eval(...)`, `model.encoder.depth_predictor.training is False` and `model.encoder.gaussian_head.training is True`.
- [ ] **Step 2-4:** FAIL -> implement -> PASS.
- [ ] **Step 5:** Commit `feat(approach1): add freeze utilities`.

### Task 6: Budget eval at any `k` (frozen baseline)

**Files:**
- Create: `src/approach1/build.py`, `src/approach1/model_wrapper_a1.py`, `src/approach1/evaluate.py`, `scripts/a1_eval.sh`
- Test: `tests/approach1/test_eval_smoke.py` (GPU; skipped if no CUDA)

**Interfaces:**
- Consumes: Task 2 `select_top_k_by_opacity`, `budget_count`; Task 4 `load_approach1_cfg`.
- Produces:
  - `build_wrapper(cfg_dict: DictConfig) -> tuple[ModelWrapperA1, DataModule, RootCfg, Approach1Cfg]`. It mirrors the model, data and checkpoint loading in `src/main.py:156-246` and loads `checkpointing.pretrained_model` (`state_dict` key optional) strictly.
  - `ModelWrapperA1(ModelWrapper)`, with `__init__(..., a1_cfg: Approach1Cfg)`.
  - `ModelWrapperA1.test_step`: encodes once, then for each `k` in `a1_cfg.eval_budgets` selects, renders, and records `psnr`, `ssim`, `lpips`, `kept`, `n_total`, decoder time. It asserts `kept == budget_count(n_total, k)`.
  - `ModelWrapperA1.on_test_end`: writes `<output_dir>/metrics/budget_table.json` and prints a `tabulate` table (rows = `k`; columns = PSNR, SSIM, LPIPS, kept, scenes).
  - CLI: `poetry run python -m src.approach1.evaluate +experiment=re10k +approach1=re10k_small mode=test dataset/view_sampler=evaluation dataset.view_sampler.index_path=<json> dataset.roots=[<root>] approach1.eval_budgets=[0.137] output_dir=<dir>`. Any `k` list is valid.

- [ ] **Step 1: Failing smoke test**: run `evaluate` on the overfit-5 subset with `eval_budgets=[0.1, 1.0]`. Assert that `budget_table.json` has 2 rows, that the `kept` values are `[13107, 131072]`, and that there are 5 scenes.
- [ ] **Step 2-4:** FAIL -> implement -> PASS.
- [ ] **Step 5: Frozen baseline**: run `scripts/a1_eval.sh` with the frozen checkpoint on the overfit-5 subset and on `/mnt/hdd/depthsplat/datasets/re10k` test (`assets/evaluation_index_re10k.json`, `dataset.test_chunk_interval=10`). Save the tables to `outputs/approach1/eval_frozen_{overfit5,re10k}/`.
- [ ] **Step 6:** Commit `feat(approach1): add budget eval and frozen baseline`.

### Task 7: Training step (full + prefix loss)

**Files:**
- Modify: `src/approach1/model_wrapper_a1.py`
- Test: `tests/approach1/test_train_step.py` (GPU)

**Interfaces:**
- Consumes: Task 5 `freeze_all_but`, `set_frozen_eval`; Task 2.
- Produces:
  - `ModelWrapperA1.training_step`, per step:
    1. Encode once, render all `N`.
    2. Sample `k ~ U[k_min, k_max]`, select the top-`N_k`, render.
    3. Compute `L = full_weight * L_photo(full) + prefix_weight * L_photo(prefix)`, where `L_photo` is the sum of the upstream `self.losses`.
    4. Log `loss/full`, `loss/prefix`, `loss/total`, `train/psnr_full`, `train/psnr_prefix`, `train/k`, `train/kept`.
  - `ModelWrapperA1.validation_step`: renders at each `eval_budgets` `k` and logs `val/psnr@{k}`. Logs one image grid (GT | k=1 | smallest k) via `self.logger.log_image`.
  - `ModelWrapperA1.configure_optimizers`: AdamW over `requires_grad` params only, using `OneCycleLR` with the upstream settings.
  - `ModelWrapperA1.on_train_epoch_start`: calls `set_frozen_eval`.

- [ ] **Step 1: Failing test**: with 1 batch from overfit-5, call `training_step` and then `backward()`. Assert that every param with a non-`None` grad starts with `encoder.gaussian_head.` and that the loss is finite.
- [ ] **Step 2-4:** FAIL -> implement -> PASS.
- [ ] **Step 5:** Commit `feat(approach1): add matryoshka-style prefix training step`.

### Task 8: Train entry + TensorBoard

**Files:**
- Create: `src/approach1/tb_logger.py`, `src/approach1/train.py`
- Test: `tests/approach1/test_tb_logger.py`

**Interfaces:**
- Produces:
  - `TensorBoardImageLogger(TensorBoardLogger)` with `log_image(key: str, images: list, step: int | None = None, caption: list[str] | None = None) -> None`, compatible with the upstream calls at `model_wrapper.py:690-736`.
  - CLI: `poetry run python -m src.approach1.train +experiment=re10k +approach1=re10k_small output_dir=<dir> ...`. Its `Trainer` uses `check_val_every_n_epoch=None` so that an int `val_check_interval` counts global steps. It also adds `LearningRateMonitor` and `ModelCheckpoint` (same settings as upstream), and prints the trainable param count.

- [ ] **Step 1: Failing test**: `log_image("val/x", [np.zeros((8, 8, 3), np.uint8)], step=0)` writes an event file under `save_dir`.
- [ ] **Step 2-4:** FAIL -> implement -> PASS.
- [ ] **Step 5: Smoke run**: 20 steps on overfit-5 with `trainer.max_steps=20 trainer.val_check_interval=10`. Verify: exit 0, a checkpoint in `<dir>/checkpoints/`, and that `tensorboard --logdir <dir>` lists `loss/total` and `val/psnr@0.05`.
- [ ] **Step 6:** Commit `feat(approach1): add train entry with tensorboard`.

### Task 9: Overfit feasibility run (Gate G1)

**Files:**
- Create: `scripts/a1_overfit.sh`

- [ ] **Step 1: Memory probe**: try `data_loader.train.batch_size` in `1, 2, 4` for 10 steps each and keep the largest that fits. Record the peak memory.
- [ ] **Step 2:** `scripts/a1_overfit.sh` runs:
  - `dataset.roots=[/mnt/hdd/depthsplat/datasets/re10k_overfit5] dataset.train_times_per_scene=1000`
  - `trainer.max_steps=3000 trainer.val_check_interval=250`
  - `output_dir=outputs/approach1/overfit5`
- [ ] **Step 3: Frozen check**: every tensor not under `encoder.gaussian_head.` in the final checkpoint equals the pretrained checkpoint (`torch.equal`).
- [ ] **Step 4:** Run the Task 6 eval with the fine-tuned checkpoint on overfit-5.
- [ ] **Gate G1 (pass):** `loss/total` falls, and on overfit-5 PSNR@`k=0.1` is at least 1.0 dB above the frozen baseline from Task 6 at the same `k`. If it fails, go to Task 12.
- [ ] **Step 5:** Commit the script and add the numbers to `docs/approach-1-opacity-finetune.md`.

### Task 10: Long run (needs user go-ahead)

**Files:**
- Create: `scripts/a1_train.sh`

- [ ] **Step 1:** Same config on `/mnt/hdd/depthsplat/datasets/re10k` train, with `trainer.max_steps` as agreed with the user (start at 20k) and `checkpointing.every_n_train_steps=2000`, `output_dir=outputs/approach1/re10k_long`.
- [ ] **Step 2:** Launch only after the user confirms. Monitor with `poetry run tensorboard --logdir outputs/approach1/re10k_long`.
- [ ] **Step 3:** Commit the script.

### Task 11: Final eval and report

- [ ] **Step 1:** Run the Task 6 eval on RE10K test (`assets/evaluation_index_re10k.json`) for the frozen model and the long-run checkpoint, at `k` in `{0.05, 0.1, 0.25, 0.5, 1.0}`, with the same scenes and view indices.
- [ ] **Step 2:** Add to the approach doc: the comparison table, trainable param count, added runtime (top-k select time per scene), and peak memory.
- [ ] **Step 3:** Tick the DoD boxes in `docs/approach-1-opacity-finetune.md`. Commit `docs(approach1): report results`.

### Task 12 (conditional): Soft opacity mask

Runs only if Gate G1 fails.

- [ ] **Step 1:** In `training_step`, when `a1_cfg.soft_mask` is on, multiply the opacities by `sigmoid((o - tau_k) / soft_mask_temperature)`, where `tau_k` is the `k`-quantile of `o` per scene. Do this before the prefix render, which then uses all `N` Gaussians. Eval keeps the hard top-`k`.
- [ ] **Step 2: Test**: the gradient reaches dropped Gaussians' opacities (`grad != 0` for some index outside the top-`N_k`).
- [ ] **Step 3:** Re-run Task 9 with `approach1.soft_mask=true` and re-check Gate G1.

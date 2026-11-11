# Approach 1: Fine-Tune Gaussian Head, Sort and Cut by Opacity

Status: planned (2026-11-11). Nothing below is implemented yet unless marked done.

## 1. Goal

Fine-tune only DepthSplat's `gaussian_head` so that the information in a scene moves into the
high-opacity Gaussians. At test time, sort Gaussians by opacity and keep the top `k` fraction.
Question: how much rendering quality does the fine-tuned model keep at budget `k`, compared with
the same opacity cut applied to the frozen model?

## 2. Deviation from Project Rules

- `CLAUDE.md` says DepthSplat is frozen. This approach breaks that rule on purpose: it trains
  `gaussian_head` only (`src/model/encoder/encoder_depthsplat.py:123`).
- Everything else stays frozen: ViT/CNN backbone, depth predictor (incl. its U-Net),
  `feature_upsampler`, `gaussian_regressor`, `gaussian_adapter`.
- Because of this, Approach 1 is not the final research method (the learned importance head on a
  frozen DepthSplat). It acts as a first feasibility study and a stronger baseline for it.

## 3. Locations

| What | Path |
|---|---|
| This repo (DepthSplat, upstream) | `~/projects/depthsplat` |
| Matryoshka GS (CVPR 2026, nested-prefix ordering) | `~/projects/matryoshka-gaussian-splatting` (see `mgs/sorting.py`, `mgs/subset_scheduler.py`) |
| MaskGaussian (CVPR 2025, probabilistic masks) | `~/projects/MaskGaussian` |
| RE10K train/test, preprocessed chunks (730 train / 460 test `.torch` + `index.json`) | `/mnt/hdd/depthsplat/datasets/re10k` |
| RE10K raw camera files | `/mnt/hdd/depthsplat/datasets/RealEstate10K` |
| RE10K / DL3DV small test subsets | `datasets/re10k_720p_test_subset`, `datasets/dl3dv_960p_test_subset` |
| Checkpoint used here: small, 256x256, 2 views | `/mnt/hdd/depthsplat/pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth` |
| Other checkpoints | `/mnt/hdd/depthsplat/pretrained/`, `pretrained/` |
| Run outputs | `outputs/approach1/<run-name>/` |

We use the small 256x256 checkpoint because it fits in the RTX 2080 Ti's 11 GB with gradients.

## 4. Budget Definition

- DepthSplat predicts one Gaussian per pixel per context view (`num_surfaces=1`).
- Per-pixel count: `N = R * I_size`, where `R = H * W` (resolution) and `I_size` = number of
  context views. Example: 256x256, 2 views gives `N = 131072`.
- Budget `k` is in `(0, 1]`. Kept count: `N_k = floor(k * R * I_size)`.
- Selection: global top-`N_k` by opacity over all views of a scene (not per view).
- Hard constraint: every eval logs the actual kept count, and it must equal `N_k`.

## 5. Method

- Trainable: `gaussian_head` parameters only. All other modules run under `torch.no_grad()`.
- Loss per step, following Matryoshka GS's two passes per iteration:
  `L = L_photo(all N) + L_photo(top-N_k by opacity)`, with `k ~ U[k_min, 1]` sampled
  continuously per step (default `k_min = 0.01`). Because of this, inference works at any budget,
  not only at the eval budgets.
- `L_photo` = upstream photometric loss (MSE + LPIPS) on the target views. No depth or 3D labels.
- Optional, MaskGaussian-style: a soft keep mask `sigmoid((o - tau_k) / T)` multiplied into the
  opacity, where `tau_k` is the `k`-quantile of opacity. This gives the ranking itself a gradient.
  The stock rasterizer skips Gaussians with alpha < 1/255, so a hard cut alone never sends a
  gradient to the dropped ones.

## 6. Pipeline

1. **Overfit run.** 5 fixed RE10K train scenes, a few thousand steps. Purpose: show that the
   loss falls and that PSNR at small `k` approaches PSNR at `k=1`.
2. **Long run.** Full RE10K train split, same config, longer schedule.
3. **Monitoring.** TensorBoard logger in new code (upstream uses wandb). Logs: total and per-budget
   loss, PSNR per budget, opacity histogram, and rendered images at `k=1` and the smallest `k`.
4. **Eval.** Top-`k` by opacity on the test split, for both the frozen and the fine-tuned
   checkpoint. Default budgets are `{0.05, 0.1, 0.25, 0.5, 1.0}`, and any list of `k` can be
   passed on the command line. Metrics: PSNR, SSIM, LPIPS, kept count, runtime.

New code goes into new modules and configs. Upstream files get only minimal hooks.

Implementation plan (tickets A1-01 to A1-12, dependencies, commands):
[docs/superpowers/plans/2026-11-11-approach-1-opacity-finetune.md](superpowers/plans/2026-11-11-approach-1-opacity-finetune.md)

## 7. Scope

- In scope: fine-tuning `gaussian_head`, opacity top-`k` selection, overfit and long training runs,
  TensorBoard, RE10K eval at 2 views (256x256).
- Out of scope for Approach 1: the learned importance head, merging, DL3DV and multi-view-count
  sweeps, voxel-merge baseline, rasterizer changes, unfreezing any other module.

## 8. Definition of Done

- [ ] Overfit script runs on command with one command line, on 3-5 scenes. Loss falls and
      PSNR@`k=0.1` clearly improves over the frozen model on the same scenes.
- [ ] Long-run script starts from the same config. TensorBoard shows all metrics in section 6.
- [ ] Eval script prints a table (budget x {PSNR, SSIM, LPIPS, kept count}) for frozen and
      fine-tuned models on the same scenes and view indices.
- [ ] Kept count equals `floor(k * R * I_size)` in every eval row.
- [ ] Trainable param count and added runtime are reported.
- [ ] No upstream files changed beyond minimal hooks.

## 9. Open Risks

- A hard top-`k` gives no gradient to the ranking. If the overfit test shows no gain, enable the
  soft mask from section 5.
- Opacity alone is a weak importance signal (it ignores scale and coverage). If cut quality stays
  poor, this motivates the learned head (the main method).

# Approach 2: Learned Importance Head (Design, Basic Version)

Status: superseded by [the refine-head design](2026-11-17-approach-2-refine-head-design.md)
(2026-11-17); implemented and evaluated as A2 v1. Originally: design, awaiting approval (2026-11-12). Branch: `approach2-importance-head` (from
`approach1-opacity-finetune`). Nothing below is implemented.

## 1. Idea

A new head on a fully frozen DepthSplat predicts one importance score per per-pixel Gaussian.
Gaussians are sorted by that score and the top `N_k` are kept. This is the basic version; more
specific variants come later (§6). Everything not stated here follows DepthSplat.

## 2. Decisions

| Topic | Decision |
|---|---|
| Backbone | DepthSplat fully frozen (incl. `gaussian_head`). Small 256x256 2-view checkpoint `/mnt/hdd/depthsplat/pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth` |
| Head input | Option 3: the predicted Gaussians at full resolution, i.e. the raw `gaussian_head` output (37 channels: opacity logit 1, xy offset 2, scale 3, rotation 4, SH 27, `encoder_depthsplat.py:242`) plus the depth map (1), per view, 256x256 |
| Head | U-Net, random init, weights shared across views, output 1 score per pixel |
| Selection | sort by score, keep global top `N_k = floor(k * H * W * V)` per scene (`src/approach1/budget.py`) |
| Training signal | MGS-style random budget per step + MaskGaussian-style Gumbel top-k mask, rendered with MaskGaussian's mask rasterizer so dropped Gaussians get gradient |
| Data, loss, optimizer, eval protocol | as DepthSplat / Approach 1: RE10K, MSE + LPIPS (0.05), AdamW, upstream eval index `assets/evaluation_index_re10k.json` |
| Logging | TensorBoard (`src/approach1/tb_logger.py`) |

## 3. Head

- Input: hook on `encoder.gaussian_head` captures its output `[B*V, 37, H, W]`; depth map from the
  encoder's `return_depth` output. Log depth is appended (my call). No upstream edit.
- U-Net: stem `3x3 conv + GroupNorm + GELU` (38 -> 32), 4 levels at 1, 1/2, 1/4, 1/8 resolution,
  channels 32/64/96/128, two `3x3 conv + GELU` per level, skip connections, bilinear upsampling,
  final `1x1 conv` -> 1 channel. About 0.5-1M params (report exact count and runtime).
- Per view; no explicit cross-view fusion in the basic version.

## 4. Training Step

1. Encoder forward under `torch.no_grad()`, frozen modules in `eval()`; hook yields the head input.
2. Head forward with grad -> score `s` per Gaussian. Standardize per scene: `z = (s - mean) / std`
   (mean/std not detached, so the gradient has no scale component; a detached std let the
   score scale grow until fp32 overflow in the first overfit run).
3. Sample `k ~ U[k_min, 1]`, `k_min = 0.01` (as Approach 1).
4. Gumbel top-k: `g = z + T * Gumbel(0, 1)`, `m_hard` = top-`N_k` of `g` (exact budget).
   `tau` = detached `N_k`-th largest `g`, keep probability `p = 1 - exp(-exp((z - tau) / T))`,
   straight-through `m = m_hard + (p - p.detach())`. `T` is a hyperparameter (start 1.0, tune in
   the overfit run).
5. Render the target views with the mask rasterizer, `masks = m`, all Gaussians.
6. Loss: DepthSplat photometric loss on the target views. (No full-set term: the frozen full set
   gives no gradient to the head.)

## 5. Pipeline

1. **Mask rasterizer.** Install `https://github.com/kaikai23/mask-diff-gaussian-rasterization` at
   `540dc11ef44c64f4c55fb894b3a4879de0cebfe7` with a script like `scripts/install_rasterizer.sh`.
   Check: with all masks = 1 it renders like the stock rasterizer; the mask gradient is nonzero for
   masked Gaussians.
2. **Overfit** on `re10k_overfit5` (as `scripts/a1_overfit.sh`). Goal: score cut beats random and
   opacity cut at small `k` on these scenes.
3. **Full training** on `/mnt/hdd/depthsplat/datasets/re10k` (needs go-ahead).
4. **Eval:** top-`N_k` by score, gather, render with the stock rasterizer. Budgets
   `{0.05, 0.1, 0.25, 0.5, 1.0}`. PSNR, SSIM, LPIPS, kept count. Baselines: random cut, opacity
   cut, same Gaussians and views.

Implementation notes from the reviews (for the tickets): mask tensor `[P, 1]` float32 per view;
pretrained load `strict=False` with only head keys missing, `strict_loading = False` for resume;
`supervise_intermediate_depth: false` in the config; keep frozen modules in `eval()` across
Lightning's `train()` calls; fixed validation scenes and views.

## 6. Later (not in the basic version)

- Other head inputs, tried after the predicted-Gaussians input: concat after the two branches
  (`mv_unimatch.py:512`, 768 ch at 1/4 res) and the depth U-Net output (`mv_unimatch.py:516`,
  128 ch at 1/4 res, cross-view fused).
- Cross-view fusion in the head.
- Coverage-aware selection: `docs/future-approaches/per-tile-quota-selection.md`.
- More baselines (stride, LightGaussian-style, voxel merge), more views, DL3DV.

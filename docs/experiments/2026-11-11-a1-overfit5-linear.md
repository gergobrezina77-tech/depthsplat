# Experiment: A1 Overfit, Linear Loss (`overfit5`)

Date: 2026-11-11. Ticket: A1-09 (Gate G1). Commit: `5dcb438`. Status: done, gate passed.

## Setup

| Item | Value |
|---|---|
| Command | `scripts/a1_overfit.sh` (output: `outputs/approach1/overfit5/`) |
| Data | 5 RE10K train scenes (`/mnt/hdd/depthsplat/datasets/re10k_overfit5`) |
| Checkpoint | small, 256x256, 2 views (`depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth`) |
| Trainable | `encoder.gaussian_head` only: 56,351 / 38,346,416 params |
| Loss | `L = L_photo(all N) + L_photo(top-N_k by opacity)`, `k ~ U[0.01, 1]` per step, `L_photo` = MSE + LPIPS |
| Optimizer | AdamW, lr 1e-4, OneCycleLR (cos, 1% warm-up), grad clip norm 0.5 |
| Schedule | 3000 steps, batch size 4 (peak ~9.4 GB), ckpt every 500 steps |
| Eval | `scripts/a1_eval.sh`, same 5 scenes, fixed view indices |

## Results (step 3000, eval)

| k | Frozen PSNR | Fine-tuned PSNR | Δ dB | Frozen LPIPS | Fine-tuned LPIPS | Kept |
|---|---|---|---|---|---|---|
| 0.05 | 5.41 | 11.83 | +6.42 | 0.735 | 0.626 | 6553 |
| 0.1 | 6.00 | 12.52 | +6.52 | 0.712 | 0.580 | 13107 |
| 0.25 | 8.35 | 14.87 | +6.51 | 0.598 | 0.460 | 32768 |
| 0.5 | 14.60 | 20.33 | +5.73 | 0.345 | 0.274 | 65536 |
| 1.0 | 29.73 | 27.00 | -2.73 | 0.090 | 0.147 | 131072 |

Frozen check: only the 4 `gaussian_head` tensors differ from the pretrained checkpoint (567 total).

## Training Curves (TensorBoard, every 10 steps)

| Steps | `loss/full` | `loss/prefix` | `train/psnr_full` |
|---|---|---|---|
| 9-499 | 0.0133 | 0.0666 | 25.49 |
| 509-999 | 0.0136 | 0.0478 | 25.36 |
| 1009-1499 | 0.0144 | 0.0527 | 24.83 |
| 1509-1999 | 0.0136 | 0.0449 | 25.24 |
| 2009-2499 | 0.0135 | 0.0413 | 25.34 |
| 2509-2999 | 0.0131 | 0.0516 | 25.51 |

`train/psnr_prefix` by sampled `k`, first vs. last third of the run:

| k range | Steps 0-1000 | Steps 2000-3000 |
|---|---|---|
| < 0.1 | 9.13 | 11.02 |
| 0.1-0.25 | 12.26 | 12.76 |
| 0.25-0.5 | 15.95 | 17.13 |
| 0.5-0.75 | 21.02 | 21.84 |
| >= 0.75 | 24.48 | 24.62 |

## Findings

- Gate G1 passes: PSNR@0.1 is +6.5 dB over the frozen model (needed +1.0 dB).
- Most of the gain comes in the first ~1000 steps. After that, prefix PSNR rises only 0.1-1.9 dB.
- Full-set PSNR stays flat at ~25.4 dB in training and loses 2.7 dB in eval. The prefix loss is
  up to ~20x the full loss at low `k`, so it dominates the gradient.
- The head cannot memorize 5 scenes: it only maps frozen features to Gaussian parameters.
  A plateau is expected.
- `loss/prefix` jumps step to step because `k` is resampled each step. It is not a sign of
  unstable training.
- Validation is not informative: upstream `ValidationWrapper` draws 1 random scene per check, and
  its iterator ran out after 5 checks (no `val/*` points after step 999).

## Next

- Variant `loss_mode: log`: `L = log L_full + log L_prefix`. Its gradient equals
  `grad L_full / L_full + grad L_prefix / L_prefix`, so both terms get a similar relative weight.
  Expected result: k=1 recovers without losing the low-`k` gains.
- Log `train/grad_norm` before clipping, to see how often the 0.5 clip is active.

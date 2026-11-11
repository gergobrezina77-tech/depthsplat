# Experiment: A1 Overfit, Log Loss (`overfit5_log`)

Date: 2026-11-11. Commit: `1ac229d`. Status: done (stopped at step 1000 of a 3000-step schedule).
Baseline: [2026-11-11-a1-overfit5-linear.md](2026-11-11-a1-overfit5-linear.md).

## Setup

Same as the linear run, except:

- `approach1.loss_mode=log`: `L = log L_full + log L_prefix`.
- The run uses the 3000-step OneCycle schedule but is stopped after the step-1000 checkpoint, so
  its LR curve matches the linear run's `epoch_499-step_1000.ckpt`.

Command: `scripts/a1_overfit.sh outputs/approach1/overfit5_log approach1.loss_mode=log`.

## Results (step 1000, eval, same 5 scenes and views)

| k | Frozen | Linear @1000 | Log @1000 | Log - Linear |
|---|---|---|---|---|
| 0.05 | 5.41 | 11.22 | 9.15 | -2.07 |
| 0.1 | 6.00 | 12.27 | 10.44 | -1.83 |
| 0.25 | 8.35 | 14.74 | 14.11 | -0.63 |
| 0.5 | 14.60 | 19.70 | 20.39 | +0.69 |
| 1.0 | 29.73 | 26.36 | 29.58 | +3.22 |

PSNR in dB. LPIPS at k=1: frozen 0.090, linear 0.153, log 0.095.

## Findings

- Log loss almost removes the full-set cost (-0.15 dB vs. frozen, linear: -3.4 dB at step 1000).
- The price is 1.8-2.1 dB at k <= 0.1. It is still +4.4 dB over the frozen model at k=0.1.
  The two modes are two points on one trade-off curve, not a strict improvement.
- `train/grad_norm` (pre-clip): median 5.2, p10 3.1, p90 9.0. The 0.5 clip is active on every
  step, so each update is rescaled to norm 0.5 before AdamW.
- Train PSNR is not above eval PSNR in either run (steps 500-1000: k < 0.15 gives 9.4-9.9 dB,
  k > 0.9 gives 24.3-26.3 dB, below the eval values). The model underfits: random view pairs per
  step make this a small-dataset run, not a memorization test.

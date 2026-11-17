# Approach 2 (Revised): Score + Refine Head (Design)

Status: design, reviewed by a builder check and a technical review (2026-11-17). Branch: `approach2-score-head` (worktree
`/home/gbrezina/projects/depthsplat-a2`), no new branch. Revises
[the basic A2 design](2026-11-12-approach-2-importance-head-design.md) ("A2 v1"), whose code and
results stay as the baseline. Everything not stated here follows A2 v1.

## 1. Intent

A2 v1 only ranks: its head predicts a score, the top `N_k` per-pixel Gaussians are kept unchanged.
Kept Gaussians stay ~1 pixel wide, so dropped regions render as holes at small `k`
(`docs/experiments/2026-11-12-a2-overfit5.md`, comparison grids). Approach 1's fixed-budget runs
(`docs/experiments/2026-11-12-a1-overfit3-k05.md`) show that the gain needs a learned ranking
**and** changed Gaussian parameters together (swap test).

The revised head does both: from the frozen `gaussian_head` output it predicts **corrected Gaussian
parameters for every Gaussian plus one importance score**. Gaussians are sorted and cut by the
score; the corrected parameters let the kept ones cover for the dropped ones. DepthSplat stays
frozen: the head is a new module, `gaussian_head` and `gaussian_adapter` are untouched.

## 2. Goals, Non-Goals, Anti-Goals

- **Goal:** at `k = 0.05, 0.1, 0.25` on unseen scenes, beat (a) the A2 v1 score cut and (b) the
  random cut on frozen DepthSplat, at matched Gaussian counts. One head serves all budgets.
- **Non-goals (later ablations, not in this version):** budget `k` as head input, richer head
  inputs (backbone features), depth correction (moving Gaussians along the ray), a post-cut merge
  step that sees dropped neighbors ("Option B"), cross-view fusion, DL3DV, more views.
- **Anti-goals:** any DepthSplat weight update or upstream file edit; output count above
  `N_k = floor(k * H * W * V)`; a k=1 quality drop vs. frozen DepthSplat that is not reported.

## 3. Approaches Considered

| Approach | Trade-off | Decision |
|---|---|---|
| A. Correct before the cut: one pass predicts score + corrections for all Gaussians, then cut | Small change to A2 v1, fast; a Gaussian does not know which neighbors get dropped | **Chosen** (user) |
| B. Correct after the cut: a second module sees each kept Gaussian and its dropped neighbors | Better hole filling, closer to merging; needs neighbor lookup, more runtime | Later, if holes remain |
| Predict full parameters from scratch vs. predict a residual on the frozen output | Residual with zero init starts exactly at frozen DepthSplat and makes the wiring testable | **Residual** (my call) |
| Budget-conditioned corrections (k as head input) | Avoids one correction having to serve all budgets (risk of blur at k=1) | **Not now** (user: naive first); first ablation if k=1 drops |

## 4. Decisions

| Topic | Decision |
|---|---|
| Backbone | Frozen DepthSplat small, 256x256, 2 views (as A2 v1) |
| Head input | As A2 v1: raw `gaussian_head` output `[BV, 37, H, W]` + log depth = 38 ch. The input is the **frozen** raw output (before correction) |
| Head output | 38 ch per pixel: 37-ch residual `delta` in the raw layout (opacity logit 1, xy offset 2, scale 3, rotation 4, SH 27; `encoder_depthsplat.py:242`, `gaussian_adapter.py:62`) + 1 score |
| Head architecture | A2 v1 U-Net (`src/approach2/score_head.py`) with a configurable number of output channels (1 or 38) |
| Init | Final 1x1 conv: weights and bias of the 37 `delta` rows = 0; score row keeps default init |
| Corrected Gaussians | `raw' = raw + delta`, converted by DepthSplat's own code after `gaussian_head` (opacity sigmoid, xy offset, `gaussian_adapter`) |
| Selection | Sort by score, keep global top `N_k` per scene (as A2 v1) |
| Training | As A2 v1: `k ~ U[0.01, 1]` per step, Gumbel top-k straight-through mask on the score (`T = 1.0`), mask rasterizer, MSE + 0.05 LPIPS, AdamW 3e-4 |
| Mode switch | Config key `approach2.head_mode: score | refine`, default `score`. `score` = A2 v1 unchanged (old checkpoints and scripts keep working); `refine` = this design |
| Residual bounds | None (user: fully naive). All 37 residuals unbounded; inflation watched in §6 |
| Precision | fp16 mixed precision (user) for training and eval of `refine`: Lightning `16-mixed` (autocast + GradScaler with `init_scale = 2**10`), config key `approach2.precision`, default `32` so v1 is unchanged. Backbone and head run in fp16; the Gaussian adapter, both rasterizers, the Gumbel mask, the loss and all metrics run in fp32 (see §5.9) |
| Logging | TensorBoard, as A2 v1, plus the refine stats in §6 |

## 5. Wiring (no upstream edit)

1. **Depth capture:** forward pre-hook on `encoder.gaussian_regressor`. Its input `concat` holds
   depth at channel 3 (`encoder_depthsplat.py:224`: image 3 ch, then depth). Store
   `concat[:, 3:4]` as `[BV, 1, H, W]`. A test checks it equals the encoder's returned depth.
2. **Replace hook:** forward hook on `encoder.gaussian_head`. It builds the head input from its
   output and the captured depth, runs the head, stores the score, and **returns `raw + delta`**.
   PyTorch uses a forward hook's return value as the module output, so the rest of the encoder
   forward builds Gaussians from the corrected tensor.
3. **Bypass flag:** when disabled, the hook returns `raw` unchanged and stores no score. Used for
   frozen baselines in the same process.
4. **Gradients:** in training the encoder runs under `torch.enable_grad()` (A2 v1 used
   `no_grad`). All DepthSplat parameters have `requires_grad=False` and stay in `eval()`, so
   autograd records only from the head output onward (head → opacity sigmoid → `gaussian_adapter`
   → mask rasterizer). Validation and eval run under `no_grad`.
5. **Score order:** `[BV, 1, H, W]` → `[B, V*H*W]` via `scores_to_gaussian_order`, as A2 v1.
6. **Timing:** `head_time` = time inside the replace hook plus the score reordering, as in v1
   (head input + head + reorder), CUDA-synchronized when timings are requested; `encode_time` =
   encoder total minus the in-hook time.
7. **Requirement:** `supervise_intermediate_depth=false` (already enforced by `build.py`), so the
   captured depth and the encoder's `depths` have the same batch layout.
8. **Checkpoints:** a refine checkpoint stores `head_mode`; loading it in the other mode fails
   with a clear error (head outputs differ: 1 vs. 38 channels).
9. **fp16 boundaries:** DepthSplat has no fp16 path upstream (no autocast anywhere in `src/`).
   Under autocast the backbone and the head run in fp16. The hook returns `raw + delta` in fp32
   (bypass: `raw` in fp32), and `encoder.gaussian_adapter.forward` is wrapped from
   `src/approach2/` (instance attribute, no upstream edit) to run in fp32 with autocast disabled.
   A technical review measured that an fp16 adapter costs 0.4-3.4% covariance error and shrinks
   the head's gradients. Everything after the encoder (Gumbel mask, rendering, loss, PSNR, SSIM,
   LPIPS) runs with autocast disabled.

## 6. Observability (TensorBoard)

On top of the A2 v1 scalars (`loss/masked`, `train/psnr_masked`, `train/k`, `train/kept`,
`train/score_std`):

- `refine/delta_abs/<group>`: mean |delta| per group (opacity, offset, scale, rotation, sh).
- `refine/delta_abs/sh` split into `sh_dc` and `sh_rest`: degrees 1-2 are damped by `sh_mask`
  (0.1 * 0.25^degree, `gaussian_adapter.py:76`), the DC term acts at full strength.
- `refine/scale_ratio/{p50,p99,max}`: ratio of corrected to frozen activated scale,
  `clamp(softplus(raw_s + delta_s - 4)) / clamp(softplus(raw_s - 4))` with the adapter's clamp
  (`gaussian_adapter.py:64`; the activated scale goes straight into the covariance, there is no
  depth multiplier). Tracks whether the head inflates Gaussians, the A1 `overfit5` failure; the
  median alone would hide a fat tail.
- `refine/opacity_mean`: mean corrected opacity vs. frozen.
- `val/psnr@k` for the refined score cut, `val/psnr_minus_random@k` against the random cut on
  **frozen** Gaussians, `val/psnr_full_minus_frozen` (k=1 refined vs. frozen full set).

## 7. Correctness Checks (tests, gate G0)

1. **Zero init = frozen:** with the fresh head, within one encoder pass the corrected tensor
   equals the captured `raw` exactly (`torch.equal`). Across passes (hook enabled vs. bypassed),
   Gaussians and the render match with `allclose` (atol ~1e-5), since backbone kernels may differ
   slightly between passes.
2. **Gradient reach:** after one masked training step's backward, the final conv's `delta` rows
   and score row both have nonzero gradients, every DepthSplat parameter has `grad is None`.
3. **Mask rasterizer parameter gradients:** `render_masked` passes finite, nonzero gradients to
   means, covariances, SH and opacities (current tests only cover the mask gradient).
4. **Depth capture:** captured depth equals the encoder's returned depth.
5. **Mode switch:** `head_mode=score` reproduces A2 v1 (existing tests stay green).
6. **fp16 sanity:** frozen DepthSplat (hook bypassed) at k=1 on heldout10 in fp16 vs. fp32:
   mean PSNR within 0.1 dB, no NaN/inf in Gaussians. One fp16 training step gives finite loss
   and finite head gradients.

## 8. Evaluation

Same scenes, views, budgets `{0.05, 0.1, 0.25, 0.5, 1.0}`, hard top-`N_k`, stock rasterizer as
A2 v1. PSNR, SSIM, LPIPS, kept count, head params, `head_time`.

| Method | Gaussians | Cut by |
|---|---|---|
| `refine` | corrected | predicted score |
| `refine_random` | corrected | random (seed 0) |
| `refine_opacity` | corrected | corrected opacity (the "sort by opacity" ablation) |
| `random`, `opacity` | frozen (hook bypassed) | random / frozen opacity |
| `score` (A2 v1) | frozen | A2 v1 score; from its existing eval outputs on the same index, or rerun with `head_mode=score` and the v1 checkpoint |

`refine` vs. `refine_random` separates "better ranking" from "better Gaussians"; `refine` at k=1
vs. frozen shows the cost of budget-independent corrections.

- All methods in one comparison run at the same precision. A2 v1 `score` is re-evaluated in
  fp16 (`head_mode=score`, v1 checkpoint `outputs/approach2/overfit5_fix1/`, final step) rather
  than taken from its fp32 tables.
- Frozen Gaussians come from one bypassed encoder pass per scene, shared by `random` and `opacity`
  (in validation also by `val/psnr_minus_random@k`). Cost: ~67 ms per scene.
- Voxel merging and post-hoc pruning (project baselines) are deferred to the comparison phase
  after full training; this version compares against random, opacity and A2 v1.
- The results doc states that the budget is not a head input in this version (a non-goal, §2).

## 9. Pipeline and Gates

1. **G0:** tests in §7 pass. Overfit runs (steps 2-3, incl. a short memory probe) are launched
   as part of the implementation (user go-ahead, 2026-11-17).
2. **1-scene overfit** (setup of `scripts/a2_overfit1.sh`, 3000 steps). Gate G1 (my call):
   `refine` beats the A2 v1 1-scene score cut (11.43 / 13.45 / 19.67 dB at k = 0.05 / 0.1 / 0.25)
   on its training scene. This is a wiring and capacity check; G2 on unseen scenes is the real
   test. A2 v1 values here and in G2 are its fp32 results; the gates use the fp16 re-eval of the
   v1 head (§8) once available.
3. **5-scene overfit** (setup of `scripts/a2_overfit.sh`, eval on overfit5 + heldout10). Gate G2:
   on heldout10, `refine` beats A2 v1 `score` and frozen `random` at k = 0.05 / 0.1 / 0.25, all
   taken from fp16 evals (fp32 references: v1 10.78 / 12.26 / 15.91, random 6.92 / 8.51 / 12.93).
   Report k=1 vs. frozen (fp32: 25.69 dB). The refine run sees the same number of samples as v1
   (batch 4 x 3000 steps, or batch 2 x 6000 steps if batch 4 does not fit).
4. **Full RE10K training:** only with the user's go-ahead.
5. Results doc under `docs/experiments/`, including grids (`save_grids=true`).

If a gate fails: stop, write up the result and the diagnostics (§6), and decide with the user
before trying a fallback (§10). No fallback is built up front.

## 10. Risks

- **Budget-independent compromise:** one correction must serve all `k`, which can inflate
  Gaussians and blur the k=1 render (A1 `overfit5`). Watched by `refine/scale_ratio` and
  `val/psnr_full_minus_frozen`; fallback is the k-conditioning ablation.
- **fp16:** DepthSplat was trained in fp32; fp16 may shift its depth and Gaussians (checked by
  §7.6). GradScaler handles small head gradients; overflow shows as skipped steps.
- **Unbounded scale growth (accepted, user: no cap):** the adapter's `softplus(s - 4)` is ~`exp(s - 4)` for typical raw
  values, so a scale residual of +3 grows a Gaussian ~20x. The clamp `gaussian_scale_max = 3.0`
  (normalized world units, per-pixel Gaussians ~0.01) does not bind in practice.
- **Residuals train on kept Gaussians only:** with mask 0 the mask rasterizer gives exactly zero
  gradient to a Gaussian's parameters (verified). So at small `k` a kept Gaussian's residual can
  only learn to cover its dropped neighbors by growing, which feeds the risk above.
- **Train/test selection gap (inherited from v1):** training cuts with Gumbel noise at `T = 1`,
  eval with a noiseless top-k; residuals co-adapt to noisy kept sets.
- **Memory:** gradients now flow through the adapter and rasterizer parameters (A2 v1 peak 6.9 GB
  at batch 4); probe before the 5-scene run, fall back to batch 2.
- **Learning-rate balance:** score and residual share one optimizer; if residuals stay near zero
  or explode, a separate lr for the final conv's `delta` rows is the first knob.

## 11. Left to the Builder

Exact TensorBoard tag spelling, test fixtures and tolerances, code layout inside
`src/approach2/`, and how `refine` / `score` modes share the wrapper.

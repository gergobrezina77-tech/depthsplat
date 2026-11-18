# Approach 1 Summary: Overfit Phase Results

Date: 2026-11-12. Covers the runs `overfit5` (linear loss) and `overfit5_log` (log loss), plus a new
check on unseen scenes. Detailed run logs: [linear](2026-11-11-a1-overfit5-linear.md),
[log](2026-11-11-a1-overfit5-log.md).

Notation: plain words first, exact names in parentheses. Names with a `/` are TensorBoard graphs
(`poetry run tensorboard --logdir outputs/approach1`); other names are config keys or files.

## 1. In One Paragraph

We retrained one small part of DepthSplat, the last layer that turns features into Gaussians
(`gaussian_head`, 56k of 38M weights). The goal was to teach it to put the important parts of a
scene into the most visible Gaussians, so that we can throw the rest away. On the scenes it trained
on and on new scenes, images made with only a fraction of the Gaussians score better than before
(+3 to +6 dB). The pictures and two extra checks show the reason: **the model did not learn which
Gaussians matter.** It learned to make every Gaussian about 3x bigger, so the holes left by
cutting get filled with blur. The full image gets worse because of this, and throwing away a
**random** set of Gaussians works better than our method in this 5-scene, random-budget setup.

**Update 2026-11-17: Approach 1 works.** The later fixed-budget runs
([1 scene](2026-11-12-a1-overfit1-k05.md), [3 scenes](2026-11-12-a1-overfit3-k05.md)) show that
tuning `gaussian_head` and cutting by opacity does learn a ranking: at k=0.5, PSNR rises from
13.53 to 28.81 dB (1 scene) and from 13.72 to 25.85 dB (3-scene mean), the full image holds
(+2.3 / +0.4 dB), and the gain needs both the tuned ranking and the tuned parameters (swap test).
The failure described here comes from the `overfit5` setup, not from the method. See §10.

## 2. Key Terms

| Simple term | Meaning | Exact name |
|---|---|---|
| Budget | Fraction of Gaussians we keep, 0 to 1. 0.25 = keep a quarter. | `k` |
| Kept count | Number of Gaussians after the cut. 256x256 x 2 views = 131,072 at k=1. | `kept`, `N_k` |
| Visibility | How opaque a Gaussian is, 0 (invisible) to 1 (solid). | `opacity` |
| Cut by visibility | Keep the `k` fraction with the highest opacity, drop the rest. | `select_top_k_by_opacity` |
| Image score | How close the render is to the real photo. Higher is better. +3 dB = about half the pixel error. | PSNR |
| Structure score | Similarity of edges and textures, 0 to 1. Higher is better. | SSIM |
| Looks-alike score | How different two images look to a neural net that mimics people. Lower is better. | LPIPS |
| Full image | Render with all Gaussians (k=1). | `loss/full`, `train/psnr_full` |
| Cut-down image | Render with only the kept Gaussians. | `loss/prefix`, `train/psnr_prefix` |

## 3. What We Did

1. **Picked 5 training scenes** from RE10K (`re10k_overfit5`) so the model could see them often.
2. **Froze everything except the last layer** (`gaussian_head`). Checked afterwards: only those 4
   weight tensors changed.
3. **Trained with two images per step.** One is the full image, one is cut down to a random budget
   between 0.01 and 1 (`train/k`). The model is scored on both (`loss/full` + `loss/prefix`).
4. **Tried two ways to add the two scores:**
   - Plain sum (`loss_mode=linear`), 3000 steps (`overfit5`).
   - Sum of logarithms (`loss_mode=log`), stopped at 1000 steps (`overfit5_log`). This keeps the
     big cut-down error from drowning out the small full-image error.
5. **Tested at fixed budgets** 1, 0.5, 0.25, 0.1, 0.05 on:
   - the same 5 training scenes, and
   - **10 new scenes** the model never saw, from the RE10K test split (`re10k_heldout10`).

   Every model uses the same views per scene.
6. **Ran two extra checks** to find out why the scores improved: a random cut instead of the cut
   by visibility, and Gaussian size and visibility statistics.

## 4. Training Graphs

**They look like noise. That is expected, not a sign of broken training.**

- **Cut-down image score** (`train/psnr_prefix`, `loss/prefix`): the budget is random each step
  (`train/k`). A step with k=0.05 scores around 10 dB and a step with k=1 around 25 dB, so the graph
  jumps with the budget. The budget alone explains 95% of the jumps.
- **Full image score** (`train/psnr_full`, `loss/full`): each step uses different scenes and view
  pairs, and some scenes are much harder than others. The graph jumps by about ±1.2 dB.
- **Total score** (`loss/total`): the sum of the two above, so it jumps too.

Once the budget effect is removed, the real trend is small:

| Part of run (`overfit5`) | Cut-down score vs. average, budget removed | Full image score (`train/psnr_full`) |
|---|---|---|
| Steps 0-1000 | -0.47 dB | 25.43 dB |
| Steps 1000-2000 | +0.15 dB | 25.03 dB |
| Steps 2000-3000 | +0.32 dB | 25.42 dB |

- The cut-down image improves by only ~0.8 dB over the whole run, mostly early.
- The full image does not improve at all.
- **Gradient size** (`train/grad_norm`, log run only): median 5.2. It is always above the clip
  limit of 0.5, so every update is cut down to the same size.
- **Fixed-budget scores** (`val/psnr@0.05` ... `val/psnr@1.0`) would be the clean graphs, but they
  stop after step 999 because of a bug in how validation picks scenes.

Tip: set the TensorBoard **Smoothing** slider to 0.95-0.99 to see the trend line.

## 5. Results: Image Score at Each Budget

Image score (PSNR, dB) / looks-alike score (LPIPS). Frozen = original DepthSplat, cut the same way.

**Training scenes (5):**

| Budget `k` | Frozen | Linear, step 3000 | Log, step 1000 |
|---|---|---|---|
| 1.0 (full) | 29.73 / 0.090 | 27.00 / 0.147 | 29.58 / 0.095 |
| 0.5 | 14.60 / 0.345 | 20.33 / 0.274 | 20.39 / 0.245 |
| 0.25 | 8.35 / 0.598 | 14.87 / 0.460 | 14.11 / 0.495 |
| 0.1 | 6.00 / 0.712 | 12.52 / 0.580 | 10.44 / 0.628 |
| 0.05 | 5.41 / 0.735 | 11.83 / 0.626 | 9.15 / 0.666 |

**New scenes (10), never seen in training:**

| Budget `k` | Frozen | Linear, step 3000 | Log, step 1000 |
|---|---|---|---|
| 1.0 (full) | 27.23 / 0.091 | 23.38 / 0.175 | 25.57 / 0.113 |
| 0.5 | 15.59 / 0.306 | 17.83 / 0.296 | 18.16 / 0.261 |
| 0.25 | 9.49 / 0.538 | 13.76 / 0.452 | 12.86 / 0.481 |
| 0.1 | 6.93 / 0.664 | 10.25 / 0.595 | 9.58 / 0.618 |
| 0.05 | 6.22 / 0.692 | 9.27 / 0.640 | 8.66 / 0.656 |

Kept count matches `floor(k x 131072)` in every row.

What this says:

- **Cut-down images improve on new scenes too:** +4.3 dB at k=0.25 for the linear run. The model
  did not only memorize the 5 scenes.
- **The full image gets worse on new scenes:** -3.9 dB (linear) and -1.7 dB (log). The log loss
  protects the full image on training scenes, but less on new ones.
- **All cut-down images are still bad:** 13-15 dB at k=0.25 is far from a usable picture
  (the full image is 27-30 dB).
- Small sample: 5 and 10 scenes, 3 views each. Scenes differ a lot (frozen full image: 18-34 dB),
  so treat differences under ~1 dB as noise.

## 6. What the Pictures Show

Image grids: `outputs/approach1/viz_train5/*.png` and `outputs/approach1/viz_heldout10/*.png`
(local only, `outputs/` is not in git). Top row: the two input views and the real target photo.
Then one row per model, budgets 1 to 0.05 from left to right.

- **Original model, cut down:** large **black holes**. Whole walls and doors disappear, because the
  model gives large flat areas low visibility.
- **Retrained model, cut down:** the holes are filled with **smooth, wall-colored blur**. Edges and
  textures (door panels, mirror, rug pattern) are gone at k=0.25 and below.
- **Most of the score gain is "gray instead of black"**, not kept detail. The image score punishes
  black pixels heavily, so filling them with any average color gains many dB.

## 7. Why: Two Extra Checks

**Check 1: did visibility move into fewer Gaussians? No.**
We measured the share of total visibility held by the most visible 10% (opacity mass in the top
10%). On new scenes: original 0.182, linear 0.192, log 0.182. Practically unchanged. The ranking
the cut depends on was not learned.

**Check 2: did the Gaussians get bigger? Yes.**
Median Gaussian size (square root of the largest covariance eigenvalue) on new scenes: original
0.0115, log 0.0176, **linear 0.0336 (about 3x)**. Big Gaussians cover holes after a cut but blur
the full image, which explains the full-image loss.

**Random cut vs. cut by visibility (image score, dB, new scenes):**

| Budget `k` | Original, by visibility | Original, random | Linear, by visibility | Linear, random |
|---|---|---|---|---|
| 0.5 | 15.59 | 18.79 | 17.83 | **22.64** |
| 0.25 | 9.49 | 12.36 | 13.76 | **19.36** |
| 0.1 | 6.93 | 8.30 | 10.25 | **14.88** |

- A **random cut beats the cut by visibility for every model**, by up to 5.6 dB.
- Reason: low-visibility Gaussians sit together (whole walls), so cutting by visibility removes
  whole regions. A random cut thins everything evenly and keeps every region covered.
- So "original model + cut by visibility" is a weak baseline. Beating it alone does not show that
  the method works.

## 8. How Well Did We Do on the Goal

Goal of Approach 1: after retraining, cutting by visibility should keep good image quality at a
small budget.

| Question | Answer |
|---|---|
| Does training run and lower the loss? | Yes. Gate G1 passed (+6.5 dB at k=0.1 on training scenes). |
| Does the gain carry over to new scenes? | Partly. +3 to +4 dB at low budgets. |
| Is the cut-down image usable? | No. 13-15 dB at k=0.25, blurry, no detail. |
| Did the model learn which Gaussians matter? | No. Visibility ranking is unchanged; Gaussians only got bigger. |
| Is the full image kept intact? | No. -1.7 to -3.9 dB on new scenes. |
| Does it beat a random cut? | No. A random cut on the same model is 5.6 dB better at k=0.25. |

**Verdict:** the pipeline works (training, budgets, evaluation, TensorBoard), but the method itself
does not reach the goal. The result still helps the main research question: **visibility alone is
a weak signal of importance**, and any learned importance score must beat a random cut.

Definition of Done status (from [the Approach 1 doc](../approach-1-opacity-finetune.md#8-definition-of-done)):
overfit run done; eval table, kept count and parameter count done; long run not started; added
runtime is measured (`select_render_time`) but not yet reported.

## 9. Next Steps

1. **Do not start the long run yet.** With the current loss, more data would most likely give a
   more general blur, not a better ranking.
2. **Add the random cut as a standard baseline** in the eval script (`src/approach1/evaluate.py`).
3. **Fix validation** so every check scores all scenes with fixed views. This gives clean
   fixed-budget graphs (`val/psnr@k`).
4. **Try the soft mask** (`approach1.soft_mask=true`). It is the only option that lets training
   change the visibility ranking itself.
5. **Main method:** use these results as motivation for a learned importance score that also
   looks at size and coverage, and compare it against the random cut.

## 10. Update 2026-11-17: Fixed-Budget Runs

The verdict in §8 covers only `overfit5` / `overfit5_log`. The fixed-budget runs change it:
**Approach 1 works.**

| Run | k=0.5 frozen → tuned | k=1 frozen → tuned | Opacity cut vs. random cut (tuned) |
|---|---|---|---|
| `overfit1_k05` (1 scene) | 13.53 → **28.81** dB | 30.32 → **32.57** dB | +3.1 dB |
| `overfit3_k05` (3 scenes, mean) | 13.72 → **25.85** dB | 28.56 → **28.93** dB | +1.7 / +2.8 / -0.3 dB per scene |

- The tuned model learns which Gaussians to keep: the gain needs the tuned ranking and the tuned
  parameters together (swap test, [3-scene doc](2026-11-12-a1-overfit3-k05.md) §3).
- Limits: the gain is specific to the trained budget (k ≤ 0.25 stays at 10-16 dB), and both runs
  are memorization (1-3 scenes, fixed views).

## Traceability

| Result | Source |
|---|---|
| Training graphs | `outputs/approach1/{overfit5,overfit5_log}/tensorboard` |
| Checkpoints | `outputs/approach1/overfit5/checkpoints/epoch_1499-step_3000.ckpt`, `outputs/approach1/overfit5_log/checkpoints/epoch_499-step_1000.ckpt` |
| Training-scene table | `scripts/a1_eval.sh`, `outputs/approach1/eval_*/metrics/budget_table.json` |
| New-scene data | `/mnt/hdd/depthsplat/datasets/re10k_heldout10` (`make_subset --split test --num-scenes 10 --seed 1`) |
| New-scene table, pictures, per-scene numbers | `outputs/approach1/viz_heldout10/`, `outputs/approach1/viz_train5/` (`per_scene.json` + PNGs) |
| Random cut, Gaussian size, noise analysis | Scratch scripts, not yet in the repo |

# Future Approach: Per-Tile Quota Selection

Status: idea, not scheduled (2026-11-12). Fallback for Approach 2
(`docs/superpowers/specs/2026-11-12-approach-2-importance-head-design.md`), which first tries the
naive global top-`N_k`.

## Problem

- Frozen per-pixel Gaussians are ~1 pixel wide. Any cut leaves holes where whole regions are dropped.
- A random cut beats the opacity cut for every model because it thins everything evenly and keeps
  every region covered (A1 summary §7: frozen, new scenes, `k=0.1`, 8.30 vs. 6.93 dB).
- A conv head gives spatially smooth scores. A global top-`N_k` of a smooth map keeps blobs and
  drops whole regions, the same failure as the opacity cut.
- Per-Gaussian gradients (`dL/dmask`) are marginal: two Gaussians covering the same hole both get
  "keep me". "Pick a spread-out set" is a joint property they cannot express.
- With a symmetric overlap cue, both views' copies of a shared surface score low and both are dropped.

## Idea

Split each view into tiles (e.g. 8x8 pixels). Each tile keeps its top `ceil(k * tile_size)`
Gaussians by score, and a final global trim or fill by score makes the total exactly `N_k`.
Coverage is guaranteed like the stride baseline; the learned score only decides which Gaussians
win inside a tile.

## Variants

1. **Fixed quota.** As above. Simplest, no new outputs.
2. **Learned tile budget.** The head also outputs one logit per tile. A softmax over tiles
   allocates `N_k` (rounded, then trimmed to exact). Detailed tiles get more, flat walls fewer,
   without dropping a tile to zero (a floor of 1 per tile, or per view).
3. **Cross-view symmetry breaking.** For Gaussians in an overlap region, keep one view's copy:
   rank by `c_ij` only against views `j < i`, or let the tile quota count overlapping tiles once.
   This removes redundancy instead of coverage.

## Training changes

- Gumbel top-k runs per tile instead of globally (`g` per tile, `tau_g` per tile). The straight-through
  surrogate and the mask rasterizer stay the same.
- The budget stays exact per step after the global trim/fill.
- Eval uses the same per-tile rule deterministically.

## When to try it

- The head does not beat the stride baseline at G1/G2, or
- the clumping diagnostics fire: mask autocorrelation high, coverage metric worse than random, or
  the kept overlap region at `k=0.5` is empty.

## Risks

- Fixed quotas waste budget on flat tiles. Variant 2 addresses it, at the cost of a second output.
- The tile size sets a coverage scale. Too small makes it equal to stride; too large brings back
  clumping. Sweep {4, 8, 16}.
- It changes the research claim from "learned global importance" to "learned importance within a
  coverage prior". Report both.

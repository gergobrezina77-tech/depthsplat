import torch

from ..evaluation.metrics import compute_psnr
from ..misc.image_io import prep_image
from ..visualization.layout import add_border, hcat
from .select import random_scores, select_top_k


@torch.no_grad()
def budget_validation_step(wrapper, batch) -> None:
    batch = wrapper.data_shim(batch)
    assert batch["target"]["image"].shape[0] == 1
    gaussians, s = wrapper.scores(batch)
    refine = wrapper.a2_cfg.head_mode == "refine"
    # Refine mode: the random reference and the full-set comparison use the uncorrected Gaussians.
    # Encoder pass runs under the caller's autocast, like `scores`.
    frozen = wrapper.frozen_gaussians(batch) if refine else gaussians
    with torch.autocast("cuda", enabled=False):
        b, g = s.shape
        rand = random_scores(b, g, 0, s.device)
        rgb_gt = batch["target"]["image"][0]

        budgets = wrapper.a2_cfg.eval_budgets
        renders = {}
        for k in budgets:
            rgb = wrapper._render(select_top_k(gaussians, s, k), batch).color[0]
            rgb_rand = wrapper._render(select_top_k(frozen, rand, k), batch).color[0]
            renders[k] = rgb
            psnr = compute_psnr(rgb_gt, rgb).mean()
            psnr_rand = compute_psnr(rgb_gt, rgb_rand).mean()
            wrapper.log(f"val/psnr@{k}", psnr)
            wrapper.log(f"val/psnr_minus_random@{k}", psnr - psnr_rand)

        k_full = renders[1.0] if 1.0 in renders else wrapper._render(gaussians, batch).color[0]
        if refine:
            rgb_frozen = wrapper._render(frozen, batch).color[0]
            wrapper.log(
                "val/psnr_full_minus_frozen",
                compute_psnr(rgb_gt, k_full).mean() - compute_psnr(rgb_gt, rgb_frozen).mean(),
            )

    if wrapper.a2_val_image_done:
        return
    # First target view: GT | k=1 | smallest k.
    grid = hcat(rgb_gt[0], k_full[0], renders[min(budgets)][0])
    wrapper.logger.log_image(
        "comparison", [prep_image(add_border(grid))], step=wrapper.global_step,
        caption=batch["scene"],
    )
    wrapper.a2_val_image_done = True

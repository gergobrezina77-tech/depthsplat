import json
from pathlib import Path

import torch
from tabulate import tabulate

from ..approach1.budget import budget_count
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from .select import random_scores, select_top_k


def budget_test_step(wrapper, batch) -> None:
    """Per scene: cut the Gaussians by each method at each budget, render with the stock decoder."""
    batch = wrapper.data_shim(batch)
    _, _, _, h, w = batch["target"]["image"].shape
    assert batch["target"]["image"].shape[0] == 1
    rgb_gt = batch["target"]["image"][0]
    scene = batch["scene"][0]
    wrapper.a2_scenes.update(batch["scene"])

    with torch.no_grad():
        if not wrapper.a2_warm:  # untimed warm-up, once per process
            wrapper.scores(batch)
            wrapper.a2_warm = True
        timings: dict = {}
        gaussians, s = wrapper.scores(batch, timings)

        b, g = s.shape
        method_scores = {
            "score": s,
            "random": random_scores(b, g, 0, s.device),
            "opacity": gaussians.opacities,
        }
        for method in wrapper.a2_cfg.eval_methods:
            for k in wrapper.a2_cfg.eval_budgets:
                kept_gaussians = select_top_k(gaussians, method_scores[method], k)
                output = wrapper.decoder.forward(
                    kept_gaussians,
                    batch["target"]["extrinsics"],
                    batch["target"]["intrinsics"],
                    batch["target"]["near"],
                    batch["target"]["far"],
                    (h, w),
                    depth_mode=None,
                )
                kept = kept_gaussians.opacities.shape[1]
                assert kept == budget_count(g, k), (kept, g, k)
                rgb = output.color[0]
                wrapper.a2_rows.setdefault((method, k), []).append(
                    {
                        "scene": scene,
                        "psnr": compute_psnr(rgb_gt, rgb).mean().item(),
                        "ssim": compute_ssim(rgb_gt, rgb).mean().item(),
                        "lpips": compute_lpips(rgb_gt, rgb).mean().item(),
                        "kept": kept,
                        "n_total": g,
                        "head_time": timings["head_time"],
                        "encode_time": timings["encode_time"],
                    }
                )


def write_budget_tables(wrapper, out_dir: Path) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    table, per_scene = [], []
    for (method, k), rows in wrapper.a2_rows.items():
        n = len(rows)
        table.append(
            {
                "method": method,
                "k": k,
                "psnr": sum(r["psnr"] for r in rows) / n,
                "ssim": sum(r["ssim"] for r in rows) / n,
                "lpips": sum(r["lpips"] for r in rows) / n,
                "kept": rows[0]["kept"],
                "n_total": rows[0]["n_total"],
                "scenes": n,
                # head_time: score head only (added runtime); encode_time: DepthSplat encoder.
                # Both are per scene, identical across methods/budgets (scored once per scene).
                "head_time": sum(r["head_time"] for r in rows) / n,
                "encode_time": sum(r["encode_time"] for r in rows) / n,
            }
        )
        per_scene += [{"method": method, "k": k, **r} for r in rows]
    with (out_dir / "budget_table.json").open("w") as f:
        json.dump({"scenes": sorted(wrapper.a2_scenes), "rows": table}, f, indent=2)
    with (out_dir / "per_scene.json").open("w") as f:
        json.dump(per_scene, f, indent=2)
    with (out_dir / "peak_memory.json").open("w") as f:
        json.dump(torch.cuda.max_memory_allocated(), f)
    print(
        tabulate(
            [
                [r["method"], r["k"], r["psnr"], r["ssim"], r["lpips"], r["kept"], r["scenes"]]
                for r in table
            ],
            headers=["method", "k", "PSNR", "SSIM", "LPIPS", "kept", "scenes"],
            floatfmt=".4f",
        )
    )
    wrapper.a2_rows.clear()
    wrapper.a2_scenes.clear()
    return table

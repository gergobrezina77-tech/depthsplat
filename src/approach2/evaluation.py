import json
from pathlib import Path

import torch
from tabulate import tabulate

from ..approach1.budget import budget_count
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..misc.image_io import save_image
from ..visualization.annotation import add_label
from ..visualization.layout import add_border, hcat, vcat
from .refine import gaussians_fp32
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

        renders = {}
        b, g = s.shape
        methods = wrapper.a2_cfg.eval_methods
        # method -> (Gaussians, scores). refine mode: the learned-head methods use the corrected
        # Gaussians, random/opacity the frozen ones (built once per scene, only if requested).
        method_inputs = {}
        if wrapper.a2_cfg.head_mode == "refine":
            method_inputs["refine"] = (gaussians, s)
            method_inputs["refine_random"] = (gaussians, random_scores(b, g, 0, s.device))
            method_inputs["refine_opacity"] = (gaussians, gaussians.opacities)
            if "random" in methods or "opacity" in methods:
                frozen = wrapper.frozen_gaussians(batch)
                method_inputs["random"] = (frozen, random_scores(b, g, 0, s.device))
                method_inputs["opacity"] = (frozen, frozen.opacities)
        else:
            method_inputs["score"] = (gaussians, s)
            method_inputs["random"] = (gaussians, random_scores(b, g, 0, s.device))
            method_inputs["opacity"] = (gaussians, gaussians.opacities)
        # Selection, rendering and metrics run in fp32 (autocast reaches the rasterizer otherwise).
        with torch.autocast("cuda", enabled=False):
            for method in methods:
                method_gaussians, method_scores = method_inputs[method]
                method_gaussians = gaussians_fp32(method_gaussians)
                method_scores = method_scores.float()
                for k in wrapper.a2_cfg.eval_budgets:
                    kept_gaussians = select_top_k(method_gaussians, method_scores, k)
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
                    renders[(method, k)] = rgb
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

    if wrapper.a2_cfg.save_grids:
        save_grids(rgb_gt, renders, scene, Path(get_cfg()["output_dir"]) / "grids")


def save_grids(rgb_gt, renders: dict, scene: str, out_dir: Path) -> None:
    # One PNG per target view: rows = methods, columns = GT then budgets, PSNR in each label.
    methods = list(dict.fromkeys(m for m, _ in renders))
    budgets = list(dict.fromkeys(k for _, k in renders))
    for v in range(rgb_gt.shape[0]):
        rows = []
        for method in methods:
            cells = [add_label(rgb_gt[v], f"{method}: GT")]
            for k in budgets:
                rgb = renders[(method, k)][v]
                psnr = compute_psnr(rgb_gt[v : v + 1], rgb[None]).item()
                cells.append(add_label(rgb, f"k={k}  {psnr:.2f} dB"))
            rows.append(hcat(*cells))
        save_image(add_border(vcat(*rows)), out_dir / f"{scene}_t{v}.png")


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

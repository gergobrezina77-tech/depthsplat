"""Compose a budget grid from src/approach2/dump_renders.py dumps: rows = k, columns = GT + methods.

Usage: poetry run python -m scripts.a2_budget_grid <out_png> <view> <scene> [<scene>...]
Columns are (label, dump dir, method); edit COLUMNS to change them. PSNR per cell is for that view.
"""
import sys
from pathlib import Path

import torch

from src.evaluation.metrics import compute_psnr
from src.misc.image_io import save_image
from src.visualization.annotation import add_label, draw_label
from src.visualization.layout import add_border, hcat, vcat

DUMPS = Path("outputs/approach2/refine_re10k_30k/grid_val200")
COLUMNS = [
    ("refine 30k", "dump_30k", "refine"),
    ("refine 20k", "dump_20k", "refine"),
    ("refine_random", "dump_30k", "refine_random"),
    ("refine_opacity", "dump_30k", "refine_opacity"),
    ("frozen random", "dump_30k", "random"),
    ("frozen opacity", "dump_30k", "opacity"),
]


def header(text: str, width: int) -> torch.Tensor:
    label = draw_label(text, Path("assets/Inter-Regular.otf"), 24, torch.device("cpu"))[:, :, :width]
    cell = torch.ones((3, label.shape[1], width))
    cell[:, :, : label.shape[2]] = label
    return cell


out_png, view, scenes = Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
blocks = []
for scene in scenes:
    dumps = {d: torch.load(DUMPS / d / "renders" / f"{scene}.pt") for d in {c[1] for c in COLUMNS}}
    gt = dumps["dump_30k"]["gt"][view]
    budgets = sorted({k for _, k in dumps["dump_30k"]["renders"]})
    width = gt.shape[-1]
    rows = [hcat(*[header(t, width) for t in [f"{scene} t{view}"] + [c[0] for c in COLUMNS]])]
    for k in budgets:
        cells = [add_label(gt, f"GT   k={k}")]
        for _, d, method in COLUMNS:
            rgb = dumps[d]["renders"][(method, k)][view]
            psnr = compute_psnr(gt[None], rgb[None]).item()
            cells.append(add_label(rgb.clamp(0, 1), f"{psnr:.2f} dB"))
        rows.append(hcat(*cells))
    blocks.append(vcat(*rows))
save_image(add_border(vcat(*blocks, gap=32)), out_png)
print(f"wrote {out_png}")

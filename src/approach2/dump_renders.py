"""A2 budget eval that also dumps the raw renders per scene to <output_dir>/renders/<scene>.pt.

Needs approach2.save_grids=true. Run via scripts/a2_eval.sh with MODULE=src.approach2.dump_renders.
"""
import warnings
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig

from . import evaluation
from .evaluate import evaluate


def dump_renders(rgb_gt, renders: dict, scene: str, out_dir: Path) -> None:
    out_dir = out_dir.parent / "renders"
    out_dir.mkdir(parents=True, exist_ok=True)
    data = {"gt": rgb_gt.cpu(), "renders": {key: rgb.cpu() for key, rgb in renders.items()}}
    torch.save(data, out_dir / f"{scene}.pt")


# Own hydra.main: calling the decorated `evaluate` from another module breaks its relative config_path.
@hydra.main(version_base=None, config_path="../../config", config_name="main")
def main(cfg_dict: DictConfig) -> None:
    evaluation.save_grids = dump_renders
    evaluate.__wrapped__(cfg_dict)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision("high")
    main()

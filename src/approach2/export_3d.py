"""Per scene: export frozen + refine top-k Gaussian sets as PLY and render a fly-through video.

Writes <output_dir>/ply/<scene>/<set>.ply and <output_dir>/videos/<scene>.mp4 (panels side by side,
camera interpolated between the two context views). Run via scripts/a2_eval.sh with
MODULE=src.approach2.export_3d (needs approach2.head_mode=refine and a head checkpoint).
"""
import warnings
from pathlib import Path

import hydra
import numpy as np
import torch
from einops import repeat
from omegaconf import DictConfig
from scipy.spatial.transform import Rotation

from ..global_cfg import get_cfg
from ..misc.image_io import save_video
from ..model.ply_export import export_ply
from ..visualization.annotation import add_label
from ..visualization.camera_trajectory.interpolation import interpolate_extrinsics, interpolate_intrinsics
from ..visualization.layout import add_border, hcat
from . import evaluation
from .evaluate import evaluate
from .refine import gaussians_fp32
from .select import select_top_k

BUDGETS = [1.0, 0.25, 0.1]
NUM_FRAMES = 60


def save_ply(gaussians, extrinsics, path: Path) -> None:
    # Gaussians only carry world covariances; recover scale + rotation by eigendecomposition.
    eigvals, eigvecs = torch.linalg.eigh(gaussians.covariances[0].double())
    flip = torch.det(eigvecs) < 0
    eigvecs[flip, :, 0] *= -1
    scales = eigvals.clamp_min(1e-12).sqrt().float()
    quats = torch.from_numpy(Rotation.from_matrix(eigvecs.cpu().numpy()).as_quat().astype(np.float32))
    export_ply(
        extrinsics, gaussians.means[0], scales, quats, gaussians.harmonics[0], gaussians.opacities[0], path
    )


def export_step(wrapper, batch) -> None:
    batch = wrapper.data_shim(batch)
    scene = batch["scene"][0]
    out = Path(get_cfg()["output_dir"])
    _, _, _, h, w = batch["context"]["image"].shape
    gaussians, s = wrapper.scores(batch)
    sets = {"frozen": wrapper.frozen_gaussians(batch)}
    sets |= {f"refine_k{k}": select_top_k(gaussians, s, k) for k in BUDGETS}

    with torch.autocast("cuda", enabled=False):
        ctx = batch["context"]
        t = torch.linspace(0, 1, NUM_FRAMES, device=s.device)
        t = (torch.cos(torch.pi * (t + 1)) + 1) / 2
        extrinsics = interpolate_extrinsics(ctx["extrinsics"][0, 0], ctx["extrinsics"][0, 1], t)[None]
        intrinsics = interpolate_intrinsics(ctx["intrinsics"][0, 0], ctx["intrinsics"][0, 1], t)[None]
        near = repeat(ctx["near"][:, 0], "b -> b v", v=NUM_FRAMES)
        far = repeat(ctx["far"][:, 0], "b -> b v", v=NUM_FRAMES)
        panels = []
        for name, g in sets.items():
            g = gaussians_fp32(g)
            save_ply(g, ctx["extrinsics"][0, 0], out / "ply" / scene / f"{name}.ply")
            color = wrapper.decoder.forward(g, extrinsics, intrinsics, near, far, (h, w), depth_mode=None).color[0]
            label = f"{name}  ({g.means.shape[1]:,} G)"
            panels.append([add_label(frame.clamp(0, 1), label) for frame in color])
    frames = [add_border(hcat(*row)) for row in zip(*panels)]

    # yuv420p needs even width and height: pad with white.
    fh, fw = frames[0].shape[-2:]
    frames = [torch.nn.functional.pad(f, (0, fw % 2, 0, fh % 2), value=1.0) for f in frames]
    save_video(frames + frames[::-1][1:-1], out / "videos" / f"{scene}.mp4", fps=30)


# Own hydra.main: calling the decorated `evaluate` from another module breaks its relative config_path.
@hydra.main(version_base=None, config_path="../../config", config_name="main")
def main(cfg_dict: DictConfig) -> None:
    evaluation.budget_test_step = export_step
    evaluate.__wrapped__(cfg_dict)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision("high")
    main()

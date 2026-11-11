import json
import time
from pathlib import Path

import torch
from tabulate import tabulate

from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..model.model_wrapper import ModelWrapper
from .budget import budget_count, select_top_k_by_opacity
from .config import Approach1Cfg


class ModelWrapperA1(ModelWrapper):
    def __init__(
        self,
        optimizer_cfg,
        test_cfg,
        train_cfg,
        encoder,
        encoder_visualizer,
        decoder,
        losses,
        step_tracker,
        a1_cfg: Approach1Cfg,
        eval_data_cfg=None,
    ) -> None:
        super().__init__(
            optimizer_cfg,
            test_cfg,
            train_cfg,
            encoder,
            encoder_visualizer,
            decoder,
            losses,
            step_tracker,
            eval_data_cfg=eval_data_cfg,
        )
        if len(set(a1_cfg.eval_budgets)) != len(a1_cfg.eval_budgets):
            raise ValueError(f"duplicate eval_budgets: {a1_cfg.eval_budgets}")
        self.a1_cfg = a1_cfg
        self.a1_scenes: set[str] = set()
        self.a1_warm = False
        # k -> list of per-scene dicts (psnr, ssim, lpips, kept, n_total, select_render_time).
        self.a1_rows: dict[float, list[dict]] = {}

    def test_step(self, batch, batch_idx):
        """Budget eval. Upstream save_image/save_gaussian/save_depth (rejected),
        stablize_camera and render_chunk_size are not supported."""
        tc = self.test_cfg
        if tc.save_image or tc.save_gaussian or tc.save_depth:
            raise NotImplementedError("save_image/save_gaussian/save_depth unsupported")
        batch: BatchedExample = self.data_shim(batch)
        _, _, _, h, w = batch["target"]["image"].shape
        assert batch["target"]["image"].shape[0] == 1

        # Encode once; every budget reuses the same full Gaussian set.
        gaussians = self.encoder(
            batch["context"], self.global_step, deterministic=False
        )
        if isinstance(gaussians, dict):
            gaussians = gaussians["gaussians"]
        n_total = gaussians.opacities.shape[1]
        rgb_gt = batch["target"]["image"][0]
        self.a1_scenes.update(batch["scene"])

        def select_render(k: float):
            kept_gaussians = select_top_k_by_opacity(gaussians, k)
            output = self.decoder.forward(
                kept_gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                depth_mode=None,
            )
            return kept_gaussians, output

        if not self.a1_warm:  # untimed warm-up, once per process
            select_render(self.a1_cfg.eval_budgets[0])
            self.a1_warm = True

        for k in self.a1_cfg.eval_budgets:
            torch.cuda.synchronize()
            start = time.time()
            kept_gaussians, output = select_render(k)
            torch.cuda.synchronize()
            elapsed = time.time() - start

            kept = kept_gaussians.opacities.shape[1]
            assert kept == budget_count(n_total, k), (kept, n_total, k)
            rgb = output.color[0]
            self.a1_rows.setdefault(k, []).append(
                {
                    "psnr": compute_psnr(rgb_gt, rgb).mean().item(),
                    "ssim": compute_ssim(rgb_gt, rgb).mean().item(),
                    "lpips": compute_lpips(rgb_gt, rgb).mean().item(),
                    "kept": kept,
                    "n_total": n_total,
                    "select_render_time": elapsed,
                }
            )

    def on_test_end(self) -> None:
        out_dir = Path(get_cfg()["output_dir"]) / "metrics"
        out_dir.mkdir(parents=True, exist_ok=True)
        table = []
        for k, rows in self.a1_rows.items():
            n = len(rows)
            table.append(
                {
                    "k": k,
                    "psnr": sum(r["psnr"] for r in rows) / n,
                    "ssim": sum(r["ssim"] for r in rows) / n,
                    "lpips": sum(r["lpips"] for r in rows) / n,
                    "kept": rows[0]["kept"],
                    "n_total": rows[0]["n_total"],
                    "scenes": n,
                    "select_render_time": sum(r["select_render_time"] for r in rows) / n,
                }
            )
        with (out_dir / "budget_table.json").open("w") as f:
            json.dump({"scenes": sorted(self.a1_scenes), "rows": table}, f, indent=2)
        with (out_dir / "peak_memory.json").open("w") as f:
            json.dump(torch.cuda.max_memory_allocated(), f)
        print(
            tabulate(
                [
                    [r["k"], r["psnr"], r["ssim"], r["lpips"], r["kept"], r["scenes"]]
                    for r in table
                ],
                headers=["k", "PSNR", "SSIM", "LPIPS", "kept", "scenes"],
                floatfmt=".4f",
            )
        )
        self.a1_rows.clear()
        self.a1_scenes.clear()

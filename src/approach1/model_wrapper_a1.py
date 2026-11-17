import json
import time
import weakref
from pathlib import Path

import torch
from einops import rearrange
from tabulate import tabulate

from ..dataset.types import BatchedExample
from ..evaluation.metrics import compute_lpips, compute_psnr, compute_ssim
from ..global_cfg import get_cfg
from ..misc.image_io import prep_image
from ..model.types import Gaussians
from ..model.model_wrapper import ModelWrapper
from ..visualization.layout import add_border, hcat
from .budget import budget_count, select_top_k_by_opacity
from .config import Approach1Cfg
from .freeze import set_frozen_eval


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

    def on_train_epoch_start(self) -> None:
        # Lightning calls train() on the whole model each epoch; re-freeze norm/dropout state.
        set_frozen_eval(self, self.a1_cfg.trainable_prefixes)

    def on_before_optimizer_step(self, optimizer) -> None:
        # Lightning calls this before gradient clipping, so this is the unclipped norm.
        grads = [p.grad for p in self.parameters() if p.grad is not None]
        self.log("train/grad_norm", torch.linalg.vector_norm(torch.stack([g.norm() for g in grads])))

    def _encode(self, batch, scene_names=None):
        gaussians = self.encoder(
            batch["context"], self.global_step, False, scene_names=scene_names
        )
        return gaussians["gaussians"] if isinstance(gaussians, dict) else gaussians

    def _render(self, gaussians, batch, depth_mode=None):
        _, _, _, h, w = batch["target"]["image"].shape

        # The CUDA rasterizer only takes fp32: under fp16 autocast, render in fp32.
        gaussians = Gaussians(
            gaussians.means.float(), gaussians.covariances.float(),
            gaussians.harmonics.float(), gaussians.opacities.float(),
        )
        with torch.autocast("cuda", enabled=False):
            return self.decoder.forward(
                gaussians,
                batch["target"]["extrinsics"],
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                depth_mode=depth_mode,
            )

    def _photo_loss(self, output, batch, gaussians):
        # Sum of upstream photometric losses (mse, lpips), as in upstream training_step.
        total = 0
        for loss_fn in self.losses:
            kwargs = {"valid_depth_mask": None}
            if loss_fn.name == "mse":
                kwargs["l1_loss"] = self.train_cfg.l1_loss
                kwargs["clamp_large_error"] = self.train_cfg.train_ignore_large_loss
            total = total + loss_fn.forward(
                output, batch, gaussians, self.global_step, **kwargs
            )
        return total

    @torch.no_grad()
    def _psnr(self, output, batch):
        return compute_psnr(
            rearrange(batch["target"]["image"], "b v c h w -> (b v) c h w"),
            rearrange(output.color, "b v c h w -> (b v) c h w"),
        ).mean()

    def training_step(self, batch, batch_idx):
        if not self.a1_cfg.fp16_saved_activations:
            return self._training_step(batch)

        with fp16_saved_tensors(self):
            return self._training_step(batch)

    def _training_step(self, batch):
        batch: BatchedExample = self.data_shim(batch)
        cfg = self.a1_cfg
        gaussians = self._encode(batch, scene_names=batch["scene"])
        n_total = gaussians.opacities.shape[1]

        depth_mode = self.train_cfg.depth_mode
        out_full = self._render(gaussians, batch, depth_mode)
        loss_full = self._photo_loss(out_full, batch, gaussians)

        # One continuous k per step; the prefix is the global top-N_k by opacity.
        k = cfg.k_min + (cfg.k_max - cfg.k_min) * torch.rand(()).item()
        kept_gaussians = select_top_k_by_opacity(gaussians, k)
        out_prefix = self._render(kept_gaussians, batch, depth_mode)
        loss_prefix = self._photo_loss(out_prefix, batch, kept_gaussians)

        if cfg.loss_mode == "log":
            # grad of log L = grad L / L: each term is normalized by its own scale.
            total = cfg.full_weight * torch.log(loss_full) + cfg.prefix_weight * torch.log(loss_prefix)
        else:
            total = cfg.full_weight * loss_full + cfg.prefix_weight * loss_prefix
        self.log("loss/full", loss_full)
        self.log("loss/prefix", loss_prefix)
        self.log("loss/total", total)
        self.log("train/psnr_full", self._psnr(out_full, batch))
        self.log("train/psnr_prefix", self._psnr(out_prefix, batch))
        self.log("train/k", k)
        self.log("train/kept", float(kept_gaussians.opacities.shape[1]))
        self.log("info/global_step", self.global_step)  # hack for ckpt monitor

        if self.global_rank == 0 and self.global_step % self.train_cfg.print_log_every_n_steps == 0:
            print(
                f"train step {self.global_step}; scene = {[x[:20] for x in batch['scene']]}; "
                f"k = {k:.3f} ({kept_gaussians.opacities.shape[1]}/{n_total}); loss = {total:.6f}"
            )
        # Tell the data loader processes about the current step.
        if self.step_tracker is not None:
            self.step_tracker.set_step(self.global_step)
        return total

    def validation_step(self, batch, batch_idx):
        batch: BatchedExample = self.data_shim(batch)
        assert batch["target"]["image"].shape[0] == 1
        gaussians = self._encode(batch)
        rgb_gt = batch["target"]["image"][0]

        budgets = self.a1_cfg.eval_budgets
        renders = {}
        for k in budgets:
            rgb = self._render(select_top_k_by_opacity(gaussians, k), batch).color[0]
            renders[k] = rgb
            self.log(f"val/psnr@{k}", compute_psnr(rgb_gt, rgb).mean())

        # First target view: GT | k=1 | smallest k.
        k_small = min(budgets)
        k_full = self._render(gaussians, batch).color[0] if 1.0 not in renders else renders[1.0]
        grid = hcat(rgb_gt[0], k_full[0], renders[k_small][0])
        self.logger.log_image(
            "comparison", [prep_image(add_border(grid))], step=self.global_step,
            caption=batch["scene"],
        )

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            params,
            lr=self.optimizer_cfg.lr,
            weight_decay=self.optimizer_cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            self.optimizer_cfg.lr,
            self.trainer.max_steps + 10,
            pct_start=0.01,
            cycle_momentum=False,
            anneal_strategy="cos",
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1},
        }

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


def fp16_saved_tensors(model: torch.nn.Module):
    """Context: float32 tensors that autograd keeps for backward are stored as fp16 and
    converted back in backward. Weights, buffers and non-float tensors stay as they are."""
    skip = {t.data_ptr() for t in [*model.parameters(), *model.buffers()]}

    # One fp16 copy per tensor that several ops save; the weakref guards against a freed
    # tensor's address being reused by a new one.
    cache = {}

    def pack(t):
        if t.dtype != torch.float32 or t.data_ptr() in skip:
            return (None, t)
        key = (t.data_ptr(), t._version, t.shape, t.stride())
        hit = cache.get(key)
        if hit is None or hit[0]() is not t:
            hit = (weakref.ref(t), t.half())
            cache[key] = hit
        return (t.dtype, hit[1])

    def unpack(packed):
        dtype, t = packed
        return t if dtype is None else t.to(dtype)

    return torch.autograd.graph.saved_tensors_hooks(pack, unpack)

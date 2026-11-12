import time
from pathlib import Path

import torch

from ..approach1.config import Approach1Cfg
from ..approach1.freeze import set_frozen_eval
from ..approach1.model_wrapper_a1 import ModelWrapperA1
from ..global_cfg import get_cfg
from .config import Approach2Cfg
from .features import HeadInputHook, build_head_input, scores_to_gaussian_order
from .score_head import ScoreHead

PREFIX = "score_head."


class ModelWrapperA2(ModelWrapperA1):
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
        a2_cfg: Approach2Cfg,
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
            Approach1Cfg(trainable_prefixes=[PREFIX], eval_budgets=a2_cfg.eval_budgets),
            eval_data_cfg=eval_data_cfg,
        )
        self.a2_cfg = a2_cfg
        self.score_head = ScoreHead(channels=a2_cfg.head_channels)
        self.head_hook = HeadInputHook(encoder.gaussian_head)
        self.a2_rows: dict = {}
        self.a2_scenes: set[str] = set()
        self.a2_warm = False
        self.strict_loading = False  # head-only checkpoints lack the frozen DepthSplat keys

    def scores(self, batch, timings: dict | None = None):
        # batch is already shimmed. DepthSplat runs without grad; only the head gets gradients.
        # If `timings` is given, fills "encode_time" (encoder) and "head_time" (head only:
        # head input + score head + reordering), both CUDA-synchronized.
        def tick():
            if timings is not None:
                torch.cuda.synchronize()
            return time.time()

        with torch.no_grad():
            t0 = tick()
            out = self.encoder(
                batch["context"], self.global_step, False, scene_names=batch.get("scene")
            )
            gaussians, depths = out["gaussians"], out["depths"]
            t1 = tick()
            head_in = build_head_input(self.head_hook.pop(), depths)
        b, v = depths.shape[:2]
        s = self.score_head(head_in)
        s = scores_to_gaussian_order(s, b, v)
        if timings is not None:
            t2 = tick()
            timings["encode_time"] = t1 - t0
            timings["head_time"] = t2 - t1
        return gaussians, s

    def train(self, mode: bool = True):
        super().train(mode)
        set_frozen_eval(self, [PREFIX])
        return self

    def on_save_checkpoint(self, checkpoint) -> None:
        checkpoint["state_dict"] = {
            k: v for k, v in checkpoint["state_dict"].items() if k.startswith(PREFIX)
        }

    def on_validation_epoch_start(self) -> None:
        self.a2_val_image_done = False

    def training_step(self, batch, batch_idx):
        from .training import masked_training_step

        return masked_training_step(self, batch)

    def validation_step(self, batch, batch_idx):
        from .validation import budget_validation_step

        return budget_validation_step(self, batch)

    def test_step(self, batch, batch_idx):
        from .evaluation import budget_test_step

        return budget_test_step(self, batch)

    def on_test_end(self) -> None:
        from .evaluation import write_budget_tables

        write_budget_tables(self, Path(get_cfg()["output_dir"]) / "metrics")

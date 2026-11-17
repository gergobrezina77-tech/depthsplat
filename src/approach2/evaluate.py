import warnings
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from ..global_cfg import set_cfg
from .build import build_wrapper_a2, trainer_precision_kwargs
from .config import load_approach2_cfg

METHODS_BY_MODE = {
    "score": ["score", "random", "opacity"],
    "refine": ["refine", "refine_random", "refine_opacity", "random", "opacity"],
}
HEAD_METHODS = {"score", "refine", "refine_random", "refine_opacity"}


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def evaluate(cfg_dict: DictConfig) -> None:
    if cfg_dict.checkpointing.pretrained_model is None:
        raise ValueError("evaluate needs checkpointing.pretrained_model (else random init)")
    a2_cfg = load_approach2_cfg(cfg_dict)
    allowed = METHODS_BY_MODE[a2_cfg.head_mode]
    bad = [m for m in a2_cfg.eval_methods if m not in allowed]
    if bad:
        raise ValueError(
            f"eval methods {bad} not allowed for head_mode={a2_cfg.head_mode!r}; use {allowed}"
        )
    needs_head = [m for m in a2_cfg.eval_methods if m in HEAD_METHODS]
    if a2_cfg.head_ckpt is None and needs_head:
        raise ValueError(
            f"eval methods {needs_head} need approach2.head_ckpt "
            "(else a random-init head is evaluated)"
        )
    set_cfg(cfg_dict)
    Path(cfg_dict.output_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg_dict.seed)
    wrapper, data_module, _, _ = build_wrapper_a2(cfg_dict)
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        logger=False,
        enable_progress_bar=True,
        enable_checkpointing=False,
        **trainer_precision_kwargs(a2_cfg.precision),
    )
    trainer.test(wrapper, datamodule=data_module)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision("high")
    evaluate()

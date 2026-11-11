import warnings
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from ..global_cfg import set_cfg
from .build import build_wrapper
from .freeze import freeze_all_but
from .tb_logger import TensorBoardImageLogger


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def train(cfg_dict: DictConfig) -> None:
    if cfg_dict.checkpointing.pretrained_model is None:
        raise ValueError("train needs checkpointing.pretrained_model (else random init)")
    set_cfg(cfg_dict)
    output_dir = Path(cfg_dict.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg_dict.seed)
    wrapper, data_module, cfg, a1_cfg = build_wrapper(cfg_dict)

    # build_wrapper does not freeze; only the prefixes in a1_cfg stay trainable.
    freeze_all_but(wrapper, a1_cfg.trainable_prefixes)
    n_train = sum(p.numel() for p in wrapper.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in wrapper.parameters())
    print(f"Trainable params: {n_train:,} / {n_total:,}")

    callbacks = [
        LearningRateMonitor("step", True),
        ModelCheckpoint(
            output_dir / "checkpoints",
            every_n_train_steps=cfg.checkpointing.every_n_train_steps,
            save_top_k=cfg.checkpointing.save_top_k,
            monitor="info/global_step",
            mode="max",
        ),
    ]
    for cb in callbacks:
        cb.CHECKPOINT_EQUALS_CHAR = "_"

    trainer = Trainer(
        max_epochs=-1,
        accelerator="gpu",
        devices=1,
        logger=TensorBoardImageLogger(save_dir=output_dir, name="tensorboard", version=""),
        callbacks=callbacks,
        val_check_interval=cfg.trainer.val_check_interval,
        check_val_every_n_epoch=None,  # int interval counts global steps
        enable_progress_bar=True,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        max_steps=cfg.trainer.max_steps,
        num_sanity_val_steps=cfg.trainer.num_sanity_val_steps,
    )
    # Resume (optimizer state included) from a Lightning checkpoint, as upstream.
    trainer.fit(wrapper, datamodule=data_module, ckpt_path=cfg.checkpointing.load)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision("high")
    train()

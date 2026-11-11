import warnings
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer

from ..global_cfg import set_cfg
from .build import build_wrapper


@hydra.main(version_base=None, config_path="../../config", config_name="main")
def evaluate(cfg_dict: DictConfig) -> None:
    set_cfg(cfg_dict)
    Path(cfg_dict.output_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(cfg_dict.seed)
    wrapper, data_module, _, _ = build_wrapper(cfg_dict)
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        logger=False,
        enable_progress_bar=True,
        enable_checkpointing=False,
    )
    trainer.test(wrapper, datamodule=data_module)


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    torch.set_float32_matmul_precision("high")
    evaluate()

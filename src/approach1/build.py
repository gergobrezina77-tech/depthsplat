import torch
from omegaconf import DictConfig

from ..config import RootCfg, load_typed_root_config
from ..dataset.data_module import DataModule
from ..loss import get_losses
from ..misc.step_tracker import StepTracker
from ..model.decoder import get_decoder
from ..model.encoder import get_encoder
from .config import Approach1Cfg, load_approach1_cfg
from .model_wrapper_a1 import ModelWrapperA1


def build_wrapper(
    cfg_dict: DictConfig,
) -> tuple[ModelWrapperA1, DataModule, RootCfg, Approach1Cfg]:
    # Mirrors src/main.py model/data/checkpoint setup, without a Trainer.
    cfg = load_typed_root_config(cfg_dict)
    a1_cfg = load_approach1_cfg(cfg_dict)
    step_tracker = StepTracker()

    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    wrapper = ModelWrapperA1(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        encoder_visualizer,
        get_decoder(cfg.model.decoder, cfg.dataset),
        get_losses(cfg.loss),
        step_tracker,
        a1_cfg,
    )
    data_module = DataModule(cfg.dataset, cfg.data_loader, step_tracker, global_rank=0)

    if cfg.checkpointing.pretrained_model is not None:
        state = torch.load(cfg.checkpointing.pretrained_model, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        wrapper.load_state_dict(state, strict=True)
        print(f"Loaded pretrained weights: {cfg.checkpointing.pretrained_model}")
    return wrapper, data_module, cfg, a1_cfg

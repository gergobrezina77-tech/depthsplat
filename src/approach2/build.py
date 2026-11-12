import torch
from omegaconf import DictConfig

from ..approach1.freeze import freeze_all_but
from ..config import RootCfg, load_typed_root_config
from ..dataset.data_module import DataModule
from ..loss import get_losses
from ..misc.step_tracker import StepTracker
from ..model.decoder import get_decoder
from ..model.encoder import get_encoder
from .config import Approach2Cfg, load_approach2_cfg
from .model_wrapper_a2 import PREFIX, ModelWrapperA2


def build_wrapper_a2(
    cfg_dict: DictConfig,
) -> tuple[ModelWrapperA2, DataModule, RootCfg, Approach2Cfg]:
    # Mirrors src/approach1/build.py; DepthSplat is loaded from the pretrained checkpoint.
    cfg = load_typed_root_config(cfg_dict)
    a2_cfg = load_approach2_cfg(cfg_dict)
    enc_cfg = cfg.model.encoder
    if enc_cfg.supervise_intermediate_depth or not enc_cfg.return_depth:
        raise ValueError(
            "Approach 2 needs supervise_intermediate_depth=false and return_depth=true"
        )
    step_tracker = StepTracker()

    encoder, encoder_visualizer = get_encoder(enc_cfg)
    wrapper = ModelWrapperA2(
        cfg.optimizer,
        cfg.test,
        cfg.train,
        encoder,
        encoder_visualizer,
        get_decoder(cfg.model.decoder, cfg.dataset),
        get_losses(cfg.loss),
        step_tracker,
        a2_cfg,
    )
    data_module = DataModule(cfg.dataset, cfg.data_loader, step_tracker, global_rank=0)

    state = torch.load(cfg.checkpointing.pretrained_model, map_location="cpu")
    if "state_dict" in state:
        state = state["state_dict"]
    result = wrapper.load_state_dict(state, strict=False)
    bad_missing = [k for k in result.missing_keys if not k.startswith(PREFIX)]
    if bad_missing or result.unexpected_keys:
        raise RuntimeError(
            f"Pretrained checkpoint mismatch: missing={bad_missing[:5]}, "
            f"unexpected={result.unexpected_keys[:5]}"
        )
    print(f"Loaded pretrained weights: {cfg.checkpointing.pretrained_model}")

    if a2_cfg.head_ckpt is not None:
        head_state = torch.load(a2_cfg.head_ckpt, map_location="cpu")["state_dict"]
        head_state = {
            k[len(PREFIX):]: v for k, v in head_state.items() if k.startswith(PREFIX)
        }
        wrapper.score_head.load_state_dict(head_state, strict=True)
        print(f"Loaded score head: {a2_cfg.head_ckpt}")

    freeze_all_but(wrapper, [PREFIX])
    return wrapper, data_module, cfg, a2_cfg

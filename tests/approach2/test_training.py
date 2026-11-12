from pathlib import Path

import pytest
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from src.approach2.build import build_wrapper_a2
from src.approach2.mask_render import render_masked
from src.approach2.training import masked_training_step
from src.global_cfg import set_cfg

ROOT = Path("/mnt/hdd/depthsplat/datasets/re10k_overfit5")
CONFIG_PATH = "../../config"  # relative to this file

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _compose(extra=()):
    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path=CONFIG_PATH):
        cfg_dict = compose(
            config_name="main",
            overrides=[
                "+experiment=re10k",
                "+approach2=re10k_small",
                "dataset.view_sampler.num_context_views=2",
                f"dataset.roots=[{ROOT}]",
                "data_loader.train.batch_size=1",
                "data_loader.train.num_workers=0",
                "data_loader.train.persistent_workers=false",
                *extra,
            ],
        )
    set_cfg(cfg_dict)
    return cfg_dict


@pytest.fixture(scope="module")
def built():
    wrapper, data_module, _, _ = build_wrapper_a2(_compose())
    wrapper.cuda().train()
    batch = next(iter(data_module.train_dataloader()))
    batch = wrapper.transfer_batch_to_device(batch, torch.device("cuda"), 0)
    batch = wrapper.data_shim(batch)
    yield wrapper, batch
    GlobalHydra.instance().clear()


def test_masked_training_step(built):
    wrapper, batch = built
    logged = {}
    wrapper.log = lambda name, value, *a, **kw: logged.__setitem__(name, value)
    wrapper.zero_grad()
    loss = masked_training_step(wrapper, batch)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is not None for n, p in wrapper.named_parameters() if n.startswith("score_head."))
    assert all(p.grad is None for n, p in wrapper.named_parameters() if not n.startswith("score_head."))
    assert {"loss/masked", "train/psnr_masked", "train/k", "train/kept", "train/score_std"} <= logged.keys()


def test_ones_mask_matches_stock_decoder_on_real_scene(built):
    wrapper, batch = built
    with torch.no_grad():
        gaussians, s = wrapper.scores(batch)
        t = batch["target"]
        h, w = t["image"].shape[-2:]
        stock = wrapper.decoder.forward(
            gaussians, t["extrinsics"], t["intrinsics"], t["near"], t["far"], (h, w)
        ).color
        masked = render_masked(wrapper.decoder, gaussians, torch.ones_like(s), t)
    diff = (stock - masked).abs().max().item()
    print(f"MAX_ABS_DIFF={diff:.3e}")
    assert diff < 1e-4

from pathlib import Path

import pytest
import torch
from einops import rearrange
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from src.approach2.build import build_wrapper_a2
from src.approach2.refine import RAW_GROUPS, refine_stats
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
    wrapper, data_module, _, _ = build_wrapper_a2(_compose(["approach2.head_mode=refine"]))
    wrapper.cuda().train()
    batch = next(iter(data_module.train_dataloader()))
    batch = wrapper.transfer_batch_to_device(batch, torch.device("cuda"), 0)
    batch = wrapper.data_shim(batch)
    yield wrapper, batch
    GlobalHydra.instance().clear()


V1_KEYS = {"loss/masked", "train/psnr_masked", "train/k", "train/kept", "train/score_std"}


def test_refine_step_grad_reach(built):
    wrapper, batch = built
    logged = {}
    wrapper.log = lambda name, value, *a, **kw: logged.__setitem__(name, value)
    wrapper.zero_grad()
    loss = masked_training_step(wrapper, batch)
    loss.backward()
    w = wrapper.score_head.out.weight
    assert w.grad[:37].abs().sum() > 0
    assert w.grad[37].abs().sum() > 0
    assert all(p.grad is None for n, p in wrapper.named_parameters() if not n.startswith("score_head."))
    cfg = wrapper.encoder.gaussian_adapter.cfg
    stat_keys = refine_stats(*wrapper.last_refine, cfg.gaussian_scale_min, cfg.gaussian_scale_max).keys()
    assert {f"refine/delta_abs/{g}" for g in RAW_GROUPS} <= set(stat_keys)
    assert V1_KEYS | set(stat_keys) <= logged.keys()


def test_refine_step_under_autocast(built):
    wrapper, batch = built
    wrapper.log = lambda *a, **kw: None
    wrapper.zero_grad()
    with torch.autocast("cuda", dtype=torch.float16):
        loss = masked_training_step(wrapper, batch)
    assert loss.dtype == torch.float32 and torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for n, p in wrapper.named_parameters() if n.startswith("score_head.")]
    assert grads and all(g is not None and torch.isfinite(g).all() for g in grads)

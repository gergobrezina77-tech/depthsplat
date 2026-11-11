from pathlib import Path

import pytest
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from src.approach1.build import build_wrapper
from src.approach1.freeze import freeze_all_but
from src.global_cfg import set_cfg

ROOT = Path("/mnt/hdd/depthsplat/datasets/re10k_overfit5")
CONFIG_PATH = "../../config"  # relative to this file

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def built():
    GlobalHydra.instance().clear()
    with initialize(version_base=None, config_path=CONFIG_PATH):
        cfg_dict = compose(
            config_name="main",
            overrides=[
                "+experiment=re10k",
                "+approach1=re10k_small",
                "dataset.view_sampler.num_context_views=2",
                f"dataset.roots=[{ROOT}]",
                "data_loader.train.batch_size=1",
                "data_loader.train.num_workers=0",
                "data_loader.train.persistent_workers=false",
            ],
        )
    set_cfg(cfg_dict)
    wrapper, data_module, _, a1_cfg = build_wrapper(cfg_dict)
    freeze_all_but(wrapper, a1_cfg.trainable_prefixes)
    yield wrapper, data_module, a1_cfg
    GlobalHydra.instance().clear()


def test_train_step_grads_only_in_head(built):
    wrapper, data_module, _ = built
    wrapper.cuda().train()
    logged = {}
    wrapper.log = lambda key, value, *a, **k: logged.__setitem__(key, value)  # no Trainer
    batch = next(iter(data_module.train_dataloader()))
    batch = wrapper.transfer_batch_to_device(batch, torch.device("cuda"), 0)

    loss = wrapper.training_step(batch, 0)
    assert torch.isfinite(loss)
    expected = {
        "loss/full", "loss/prefix", "loss/total", "train/psnr_full",
        "train/psnr_prefix", "train/k", "train/kept",
    }
    assert expected <= logged.keys()
    assert 0 < logged["train/kept"] <= 131072  # 2 views x 256x256 x 1 gaussian/px
    loss.backward()

    with_grad = [n for n, p in wrapper.named_parameters() if p.grad is not None]
    assert with_grad
    assert all(n.startswith("encoder.gaussian_head.") for n in with_grad)


def test_frozen_modules_eval_after_epoch_start(built):
    wrapper, _, _ = built
    wrapper.train()
    wrapper.on_train_epoch_start()
    head = wrapper.encoder.gaussian_head
    assert head.training
    frozen = [
        m
        for n, m in wrapper.named_modules()
        if n and not n.startswith("encoder.gaussian_head")
        and not any(p.requires_grad for p in m.parameters())
    ]
    assert frozen and not any(m.training for m in frozen)


def test_validation_step_logs_budgets_and_image(built):
    wrapper, data_module, a1_cfg = built
    wrapper.cuda().eval()
    logged, images = {}, []
    wrapper.log = lambda key, value, *a, **k: logged.__setitem__(key, value)
    fake = type("L", (), {"log_image": lambda self, key, imgs, **k: images.append((key, imgs))})()
    type(wrapper).logger = property(lambda self: fake)
    try:
        batch = next(iter(data_module.val_dataloader()))
        batch = wrapper.transfer_batch_to_device(batch, torch.device("cuda"), 0)
        with torch.no_grad():
            wrapper.validation_step(batch, 0)
    finally:
        del type(wrapper).logger
    assert {f"val/psnr@{k}" for k in a1_cfg.eval_budgets} <= logged.keys()
    assert len(images) == 1 and images[0][1][0].ndim == 3

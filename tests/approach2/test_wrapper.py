from pathlib import Path

import pytest
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from src.approach2.build import build_wrapper_a2
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


def test_only_head_trainable(built):
    wrapper, _ = built
    names = {n.split(".")[0] for n, p in wrapper.named_parameters() if p.requires_grad}
    assert names == {"score_head"}


def test_scores_shape_and_grad(built):
    wrapper, batch = built
    gaussians, s = wrapper.scores(batch)
    assert s.shape == (1, 131072) and s.requires_grad
    assert gaussians.opacities.shape == (1, 131072)


def test_encoder_stays_eval_after_train(built):
    wrapper, _ = built
    wrapper.train()
    assert not wrapper.encoder.training and wrapper.score_head.training


def test_depth_matches_gaussian_means(built):
    wrapper, batch = built
    with torch.no_grad():
        out = wrapper.encoder(batch["context"], 0, False, scene_names=batch.get("scene"))
    gaussians, depths = out["gaussians"], out["depths"]
    _, v, h, w = depths.shape
    c2w = batch["context"]["extrinsics"][0]  # [v, 4, 4]
    means = gaussians.means[0].reshape(v, h * w, 3)
    homo = torch.cat([means, torch.ones_like(means[..., :1])], dim=-1)
    cam = torch.einsum("vij,vnj->vni", torch.linalg.inv(c2w), homo)
    z_cam = cam[..., 2]
    assert torch.allclose(z_cam, depths[0].reshape(v, h * w), rtol=1e-3)


def test_head_only_checkpoint_roundtrip(built, tmp_path):
    wrapper, _ = built
    ckpt = {"state_dict": wrapper.state_dict()}
    wrapper.on_save_checkpoint(ckpt)
    assert ckpt["state_dict"] and all(k.startswith("score_head.") for k in ckpt["state_dict"])
    path = tmp_path / "head.ckpt"
    torch.save(ckpt, path)
    fresh, _, _, _ = build_wrapper_a2(_compose([f"approach2.head_ckpt={path}"]))
    for (n1, p1), (n2, p2) in zip(
        wrapper.score_head.state_dict().items(), fresh.score_head.state_dict().items()
    ):
        assert n1 == n2 and torch.equal(p1.cpu(), p2.cpu())


def test_build_rejects_intermediate_depth():
    cfg = _compose(["model.encoder.supervise_intermediate_depth=true"])
    with pytest.raises(ValueError):
        build_wrapper_a2(cfg)

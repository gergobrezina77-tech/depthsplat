from pathlib import Path

import pytest
import torch
from einops import rearrange
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
    wrapper, data_module, _, _ = build_wrapper_a2(_compose(["approach2.head_mode=refine"]))
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


def test_captured_depth_equals_returned(built, monkeypatch):
    wrapper, batch = built
    recorded = []
    orig = wrapper.depth_capture.pop

    def spy():
        d = orig()
        recorded.append(d)
        return d

    monkeypatch.setattr(wrapper.depth_capture, "pop", spy)
    wrapper.scores(batch)
    with torch.no_grad():
        out = wrapper.encoder(batch["context"], 0, False, scene_names=batch.get("scene"))
    expected = rearrange(out["depths"], "b v h w -> (b v) () h w")
    assert recorded and torch.allclose(recorded[0], expected)


def test_zero_init_single_pass_exact(built):
    wrapper, batch = built
    wrapper.scores(batch)
    raw, delta = wrapper.last_refine
    assert torch.equal(raw + delta, raw)


def test_zero_init_matches_frozen(built):
    wrapper, batch = built
    wrapper.eval()
    with torch.no_grad():
        g, _ = wrapper.scores(batch)
        f = wrapper.frozen_gaussians(batch)
        for name in ("means", "covariances", "harmonics", "opacities"):
            assert torch.allclose(getattr(g, name), getattr(f, name), atol=1e-5), name
        c1 = wrapper._render(g, batch).color
        c2 = wrapper._render(f, batch).color
    assert torch.allclose(c1, c2, atol=1e-5)
    wrapper.train()


def test_scores_grad(built):
    wrapper, batch = built
    gaussians, s = wrapper.scores(batch)
    assert s.shape == (1, 131072) and s.requires_grad
    assert gaussians.covariances.requires_grad


def test_render_under_autocast(built):
    wrapper, batch = built
    with torch.no_grad():
        gaussians, _ = wrapper.scores(batch)
        with torch.autocast("cuda", dtype=torch.float16):
            out = wrapper._render(gaussians, batch)
    assert out.color.dtype == torch.float32


def test_checkpoint_has_mode_and_roundtrip(built, tmp_path):
    wrapper, _ = built
    ckpt = {"state_dict": wrapper.state_dict()}
    wrapper.on_save_checkpoint(ckpt)
    assert ckpt["a2_head_mode"] == "refine"
    assert ckpt["state_dict"] and all(k.startswith("score_head.") for k in ckpt["state_dict"])
    path = tmp_path / "head.ckpt"
    torch.save(ckpt, path)
    fresh, _, _, _ = build_wrapper_a2(
        _compose(["approach2.head_mode=refine", f"approach2.head_ckpt={path}"])
    )
    for (n1, p1), (n2, p2) in zip(
        wrapper.score_head.state_dict().items(), fresh.score_head.state_dict().items()
    ):
        assert n1 == n2 and torch.equal(p1.cpu(), p2.cpu())


def test_v1_checkpoint_rejected_in_refine_mode(tmp_path):
    wrapper, _, _, _ = build_wrapper_a2(_compose())
    ckpt = {"state_dict": wrapper.state_dict()}
    wrapper.on_save_checkpoint(ckpt)
    del ckpt["a2_head_mode"]
    path = tmp_path / "score.ckpt"
    torch.save(ckpt, path)
    with pytest.raises(ValueError, match="score") as e:
        build_wrapper_a2(_compose(["approach2.head_mode=refine", f"approach2.head_ckpt={path}"]))
    assert "refine" in str(e.value)


def test_refine_ckpt_rejected_in_score_mode(built, tmp_path):
    wrapper, _ = built
    ckpt = {"state_dict": wrapper.state_dict()}
    wrapper.on_save_checkpoint(ckpt)
    path = tmp_path / "refine.ckpt"
    torch.save(ckpt, path)
    with pytest.raises(ValueError, match="refine"):
        build_wrapper_a2(_compose([f"approach2.head_ckpt={path}"]))

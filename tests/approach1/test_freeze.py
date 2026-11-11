import hydra
import pytest
import torch.nn as nn
from hydra.core.global_hydra import GlobalHydra

from src.approach1.freeze import freeze_all_but, set_frozen_eval
from src.config import load_typed_root_config
from src.model.encoder import get_encoder

CONFIG_PATH = "../../config"  # relative to this file


@pytest.fixture(autouse=True)
def clear_hydra():
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def test_toy_freeze():
    model = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    names = freeze_all_but(model, ["1."])
    assert names == ["1.weight", "1.bias"]
    assert not any(p.requires_grad for p in model[0].parameters())
    assert all(p.requires_grad for p in model[1].parameters())


def test_no_match_raises():
    with pytest.raises(ValueError):
        freeze_all_but(nn.Linear(2, 2), ["nope."])


@pytest.fixture
def wrapped_encoder():
    with hydra.initialize(version_base=None, config_path=CONFIG_PATH):
        cfg = hydra.compose(
            config_name="main",
            overrides=["+experiment=re10k", "+approach1=re10k_small"],
        )
    cfg = load_typed_root_config(cfg)
    encoder, _ = get_encoder(cfg.model.encoder)
    return nn.ModuleDict({"encoder": encoder})


def test_real_encoder_freeze(wrapped_encoder):
    names = freeze_all_but(wrapped_encoder, ["encoder.gaussian_head."])
    assert names and all(n.startswith("encoder.gaussian_head.") for n in names)
    n_train = sum(p.numel() for p in wrapped_encoder.parameters() if p.requires_grad)
    print(f"trainable params: {n_train}")
    assert n_train > 0


def test_frozen_stay_eval_after_train(wrapped_encoder):
    prefixes = ["encoder.gaussian_head."]
    freeze_all_but(wrapped_encoder, prefixes)
    wrapped_encoder.train()
    set_frozen_eval(wrapped_encoder, prefixes)
    assert wrapped_encoder.encoder.depth_predictor.training is False
    assert wrapped_encoder.encoder.gaussian_head.training is True

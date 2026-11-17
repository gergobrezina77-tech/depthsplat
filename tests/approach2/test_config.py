import hydra
import pytest
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from src.approach2.config import load_approach2_cfg

CONFIG_PATH = "../../config"  # relative to this file


@pytest.fixture(autouse=True)
def clear_hydra():
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def test_load_approach2_cfg_defaults():
    cfg = load_approach2_cfg(OmegaConf.create({}))
    assert cfg.k_min == 0.01 and cfg.temperature == 1.0 and cfg.head_ckpt is None


def test_yaml_composes():
    with hydra.initialize(version_base=None, config_path=CONFIG_PATH):
        cfg_dict = hydra.compose(
            config_name="main",
            overrides=["+experiment=re10k", "+approach2=re10k_small"],
        )
    assert cfg_dict.model.encoder.supervise_intermediate_depth is False
    assert cfg_dict.model.encoder.return_depth is True
    assert load_approach2_cfg(cfg_dict).eval_budgets == [0.05, 0.1, 0.25, 0.5, 1.0]


def test_new_keys_defaults():
    cfg = load_approach2_cfg(OmegaConf.create({}))
    assert cfg.head_mode == "score" and cfg.precision == "32"


def test_precision_int_coerced():
    assert load_approach2_cfg(OmegaConf.create({"approach2": {"precision": 32}})).precision == "32"


@pytest.mark.parametrize("key,value", [("head_mode", "merge"), ("precision", "16")])
def test_rejects_unknown_values(key, value):
    with pytest.raises(ValueError):
        load_approach2_cfg(OmegaConf.create({"approach2": {key: value}}))

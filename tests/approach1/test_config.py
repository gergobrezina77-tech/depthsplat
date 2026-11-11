import hydra
import pytest
from hydra.core.global_hydra import GlobalHydra

from src.approach1.config import load_approach1_cfg
from src.config import load_typed_root_config

CONFIG_PATH = "../../config"  # relative to this file


@pytest.fixture(autouse=True)
def clear_hydra():
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


@pytest.mark.parametrize("extra", [[], ["dataset/view_sampler=evaluation"]])
def test_config_loads(extra):
    with hydra.initialize(version_base=None, config_path=CONFIG_PATH):
        cfg = hydra.compose(
            config_name="main",
            overrides=["+experiment=re10k", "+approach1=re10k_small", *extra],
        )
    assert load_approach1_cfg(cfg).eval_budgets == [0.05, 0.1, 0.25, 0.5, 1.0]
    load_typed_root_config(cfg)
    assert cfg.model.encoder.upsample_factor == 4
    assert cfg.data_loader.train.batch_size == 2
    if not extra:  # default (bounded) view sampler composition
        assert cfg.dataset.view_sampler.warm_up_steps == 0
        assert cfg.dataset.view_sampler.max_distance_between_context_views == 135

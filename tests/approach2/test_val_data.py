import json

import hydra
import pytest
from hydra.core.global_hydra import GlobalHydra

from src.approach2.val_data import build_val_loader

CONFIG_PATH = "../../config"  # relative to this file
ROOT = "/mnt/hdd/depthsplat/datasets/re10k_heldout10"
INDEX = "assets/evaluation_index_a2_val10.json"


@pytest.fixture(autouse=True)
def clear_hydra():
    GlobalHydra.instance().clear()
    yield
    GlobalHydra.instance().clear()


def test_val_loader_yields_10_scenes_with_upstream_views():
    with open("assets/evaluation_index_re10k.json") as f:
        upstream = json.load(f)
    with hydra.initialize(version_base=None, config_path=CONFIG_PATH):
        cfg_dict = hydra.compose(
            config_name="main",
            overrides=["+experiment=re10k", "+approach1=re10k_small"],
        )
    batches = list(build_val_loader(cfg_dict, ROOT, INDEX))
    assert len(batches) == 10
    for b in batches:
        scene = b["scene"][0]
        assert b["context"]["index"][0].tolist() == upstream[scene]["context"]
        assert b["target"]["index"][0].tolist() == upstream[scene]["target"]

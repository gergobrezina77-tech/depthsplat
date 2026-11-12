import copy

from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from src.config import load_typed_root_config
from src.dataset.data_module import DataModule
from src.misc.step_tracker import StepTracker


def build_val_loader(cfg_dict: DictConfig, root: str, index_path: str) -> DataLoader:
    """Fixed validation loader: scenes under `root`, views from `index_path`."""
    cfg_dict = copy.deepcopy(cfg_dict)
    OmegaConf.set_struct(cfg_dict, False)
    cfg_dict.dataset.roots = [root]
    cfg_dict.dataset.view_sampler = {
        "name": "evaluation",
        "index_path": index_path,
        "num_context_views": 2,
    }
    cfg = load_typed_root_config(cfg_dict)
    data_module = DataModule(cfg.dataset, cfg.data_loader, StepTracker(), global_rank=0)
    return data_module.test_dataloader()

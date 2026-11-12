from dataclasses import dataclass, field

from dacite import from_dict
from omegaconf import DictConfig, OmegaConf


@dataclass
class Approach2Cfg:
    k_min: float = 0.01
    k_max: float = 1.0
    temperature: float = 1.0
    head_channels: list[int] = field(default_factory=lambda: [32, 64, 96, 128])
    eval_budgets: list[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.25, 0.5, 1.0]
    )
    eval_methods: list[str] = field(
        default_factory=lambda: ["score", "random", "opacity"]
    )
    head_ckpt: str | None = None
    val_root: str = "/mnt/hdd/depthsplat/datasets/re10k_heldout10"
    val_index_path: str = "assets/evaluation_index_a2_val10.json"


def load_approach2_cfg(cfg_dict: DictConfig) -> Approach2Cfg:
    # Reads the `approach2:` key; missing key falls back to defaults.
    sub = cfg_dict.get("approach2", None)
    data = {} if sub is None else OmegaConf.to_container(sub, resolve=True)
    return from_dict(Approach2Cfg, data)

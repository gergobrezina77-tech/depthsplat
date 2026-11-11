from dataclasses import dataclass, field

from dacite import from_dict
from omegaconf import DictConfig, OmegaConf


@dataclass
class Approach1Cfg:
    trainable_prefixes: list[str] = field(
        default_factory=lambda: ["encoder.gaussian_head."]
    )
    k_min: float = 0.01
    k_max: float = 1.0
    full_weight: float = 1.0
    prefix_weight: float = 1.0
    eval_budgets: list[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.25, 0.5, 1.0]
    )
    soft_mask: bool = False
    soft_mask_temperature: float = 0.05


def load_approach1_cfg(cfg_dict: DictConfig) -> Approach1Cfg:
    # Reads the `approach1:` key; missing key falls back to defaults.
    return from_dict(
        Approach1Cfg, OmegaConf.to_container(cfg_dict.get("approach1", {}), resolve=True)
    )

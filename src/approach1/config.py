from dataclasses import dataclass, field
from typing import Literal

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
    # linear: w_f*L_full + w_p*L_prefix; log: w_f*log L_full + w_p*log L_prefix (scale-normalized).
    loss_mode: Literal["linear", "log"] = "linear"
    eval_budgets: list[float] = field(
        default_factory=lambda: [0.05, 0.1, 0.25, 0.5, 1.0]
    )
    soft_mask: bool = False
    soft_mask_temperature: float = 0.05
    # Lightning Trainer precision: "32-true" (default) or "16-mixed" (fp16 autocast + grad scaling).
    precision: str = "32-true"
    # Store float32 tensors kept for backward as fp16 (converted back in backward).
    fp16_saved_activations: bool = False


def load_approach1_cfg(cfg_dict: DictConfig) -> Approach1Cfg:
    # Reads the `approach1:` key; missing key falls back to defaults.
    return from_dict(
        Approach1Cfg, OmegaConf.to_container(cfg_dict.get("approach1", {}), resolve=True)
    )

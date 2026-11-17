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
    save_grids: bool = False  # eval: write GT vs. method x budget image grids
    head_mode: str = "score"  # "score" or "refine"
    precision: str = "32"  # "32" or "16-mixed"


def load_approach2_cfg(cfg_dict: DictConfig) -> Approach2Cfg:
    # Reads the `approach2:` key; missing key falls back to defaults.
    sub = cfg_dict.get("approach2", None)
    data = {} if sub is None else OmegaConf.to_container(sub, resolve=True)
    if "precision" in data:
        data["precision"] = str(data["precision"])  # Hydra parses precision=32 as int
    cfg = from_dict(Approach2Cfg, data)
    if cfg.head_mode not in ("score", "refine"):
        raise ValueError(f"approach2.head_mode must be score or refine, got {cfg.head_mode!r}")
    if cfg.precision not in ("32", "16-mixed"):
        raise ValueError(f"approach2.precision must be 32 or 16-mixed, got {cfg.precision!r}")
    return cfg

import time

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..model.types import Gaussians
from .score_head import ScoreHead

# Raw Gaussian channel groups (37 ch): opacity logit, xy offset, scale, rotation, SH (xyz, d_sh=9).
_SH_DC = [10, 19, 28]
RAW_GROUPS: dict[str, list[int]] = {
    "opacity": [0],
    "offset": [1, 2],
    "scale": [3, 4, 5],
    "rotation": [6, 7, 8, 9],
    "sh_dc": _SH_DC,
    "sh_rest": [c for c in range(10, 37) if c not in _SH_DC],
}


class DepthCapture:
    """Pre-hook on the gaussian regressor: stores its input channel 3 (depth) until popped."""

    def __init__(self, regressor: nn.Module):
        self._depth: Tensor | None = None
        self._handle = regressor.register_forward_pre_hook(self._hook)

    def _hook(self, module: nn.Module, inputs: tuple) -> None:
        self._depth = inputs[0][:, 3:4]

    def pop(self) -> Tensor:
        if self._depth is None:
            raise RuntimeError("No depth captured.")
        depth, self._depth = self._depth, None
        return depth

    def remove(self) -> None:
        self._handle.remove()


class RefineHook:
    """Forward hook on gaussian_head: returns raw + head residual and stores (raw, delta, score)."""

    def __init__(self, gaussian_head: nn.Module, head: ScoreHead, depth: DepthCapture):
        self.head = head
        self.depth = depth
        self.enabled = True
        self.sync_time = False
        self.last_time = 0.0
        self._captured: tuple[Tensor, Tensor, Tensor] | None = None
        self._handle = gaussian_head.register_forward_hook(self._hook)

    def _sync(self) -> None:
        if self.sync_time:
            torch.cuda.synchronize()

    def _hook(self, module: nn.Module, inputs: tuple, output: Tensor) -> Tensor:
        # Always consume the depth so a bypass pass leaves nothing stale.
        depth = self.depth.pop()
        self._captured = None
        if not self.enabled:
            return output.float()
        self._sync()
        t0 = time.perf_counter()
        raw = output.float()
        x = torch.cat([raw, depth.float().clamp_min(1e-6).log()], dim=1)
        out = self.head(x).float()
        delta, score = out[:, :37], out[:, 37:]
        self._captured = (raw, delta, score)
        corrected = raw + delta
        self._sync()
        self.last_time = time.perf_counter() - t0
        return corrected

    def pop(self) -> tuple[Tensor, Tensor, Tensor]:
        if self._captured is None:
            raise RuntimeError("No refine pass captured.")
        captured, self._captured = self._captured, None
        return captured

    def remove(self) -> None:
        self._handle.remove()


def _cast_fp32(obj):
    if isinstance(obj, Tensor) and obj.is_floating_point():
        return obj.float()
    if isinstance(obj, (tuple, list)):
        return type(obj)(_cast_fp32(o) for o in obj)
    if isinstance(obj, dict):
        return {k: _cast_fp32(v) for k, v in obj.items()}
    return obj


def wrap_adapter_fp32(adapter: nn.Module) -> None:
    """Make adapter.forward cast float tensor args to fp32 and run with autocast disabled."""
    orig = adapter.forward
    if getattr(orig, "_fp32_wrapped", False):
        return

    def forward(*args, **kwargs):
        first = next((a for a in args if isinstance(a, Tensor)), None)
        device_type = first.device.type if first is not None else "cuda"
        with torch.autocast(device_type=device_type, enabled=False):
            return orig(*_cast_fp32(args), **_cast_fp32(kwargs))

    forward._fp32_wrapped = True
    adapter.forward = forward


@torch.no_grad()
def refine_stats(
    raw: Tensor, delta: Tensor, scale_min: float, scale_max: float
) -> dict[str, Tensor]:
    raw, delta = raw.float(), delta.float()
    stats = {
        f"refine/delta_abs/{g}": delta[:, ch].abs().mean() for g, ch in RAW_GROUPS.items()
    }
    s = RAW_GROUPS["scale"]
    new = F.softplus(raw[:, s] + delta[:, s] - 4.0).clamp(scale_min, scale_max)
    old = F.softplus(raw[:, s] - 4.0).clamp(scale_min, scale_max)
    ratio = (new / old).flatten()
    # torch.quantile has an input size limit; subsample deterministically if needed.
    q = ratio if ratio.numel() <= 2**24 else ratio[:: ratio.numel() // 2**24 + 1]
    stats["refine/scale_ratio/p50"] = torch.quantile(q, 0.5)
    stats["refine/scale_ratio/p99"] = torch.quantile(q, 0.99)
    stats["refine/scale_ratio/max"] = ratio.max()
    stats["refine/opacity_mean"] = torch.sigmoid(raw[:, 0] + delta[:, 0]).mean()
    stats["refine/opacity_mean_frozen"] = torch.sigmoid(raw[:, 0]).mean()
    return stats


def gaussians_fp32(g: Gaussians) -> Gaussians:
    return Gaussians(g.means.float(), g.covariances.float(), g.harmonics.float(), g.opacities.float())

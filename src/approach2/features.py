import torch
from einops import rearrange
from torch import nn


class HeadInputHook:
    # Captures the output of a module (gaussian_head) on every forward pass.
    def __init__(self, module: nn.Module):
        self._out = None
        self._handle = module.register_forward_hook(self._store)

    def _store(self, module, inputs, output):
        self._out = output

    def pop(self) -> torch.Tensor:
        if self._out is None:
            raise RuntimeError("No captured output; run the module's forward first")
        out, self._out = self._out, None
        return out

    def remove(self) -> None:
        self._handle.remove()


def build_head_input(raw: torch.Tensor, depths: torch.Tensor) -> torch.Tensor:
    # raw: [bv, 37, h, w], depths: [b, v, h, w] -> [bv, 38, h, w] (log-depth appended)
    log_depth = rearrange(depths.clamp_min(1e-6).log(), "b v h w -> (b v) () h w")
    return torch.cat([raw, log_depth], dim=1)


def scores_to_gaussian_order(s: torch.Tensor, b: int, v: int) -> torch.Tensor:
    # [bv, 1, h, w] -> [b, v*h*w] in (v h w) order, matching the encoder's Gaussian order
    return rearrange(s, "(b v) () h w -> b (v h w)", b=b, v=v)

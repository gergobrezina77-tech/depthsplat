import torch
import torch.nn.functional as F
from torch import nn


def _double_conv(c_in: int, c_out: int) -> nn.Sequential:
    # Two 3x3 convs with GELU.
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 3, padding=1), nn.GELU(),
        nn.Conv2d(c_out, c_out, 3, padding=1), nn.GELU(),
    )


class ScoreHead(nn.Module):
    """Small U-Net: per-pixel features [n, c, h, w] -> [n, out_channels, h, w] (score logits by default)."""

    def __init__(
        self,
        in_channels: int = 38,
        channels: list[int] = [32, 64, 96, 128],
        out_channels: int = 1,
        zero_init_channels: int = 0,
    ):
        super().__init__()
        self.num_levels = len(channels)
        self.divisor = 2 ** (self.num_levels - 1)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 3, padding=1),
            nn.GroupNorm(8, channels[0]),
            nn.GELU(),
        )
        # Encoder levels at 1, 1/2, 1/4, 1/8; stride-2 convs go between them.
        self.enc = nn.ModuleList(_double_conv(c, c) for c in channels)
        self.down = nn.ModuleList(
            nn.Conv2d(channels[i], channels[i + 1], 3, stride=2, padding=1)
            for i in range(self.num_levels - 1)
        )
        # Decoder: bilinear upsample, concat skip, two convs.
        self.dec = nn.ModuleList(
            _double_conv(channels[i + 1] + channels[i], channels[i])
            for i in range(self.num_levels - 1)
        )
        self.out = nn.Conv2d(channels[0], out_channels, 1)
        # First output rows start at exactly zero (residual channels).
        with torch.no_grad():
            self.out.weight[:zero_init_channels].zero_()
            self.out.bias[:zero_init_channels].zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[-2:]
        if h % self.divisor or w % self.divisor:
            raise ValueError(f"h and w must be divisible by {self.divisor}, got {h}x{w}")
        x = self.stem(x)
        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(x)
            if i < self.num_levels - 1:
                skips.append(x)
                x = self.down[i](x)
        for i in reversed(range(self.num_levels - 1)):
            x = F.interpolate(x, size=skips[i].shape[-2:], mode="bilinear", align_corners=False)
            x = self.dec[i](torch.cat([x, skips[i]], dim=1))
        return self.out(x)

import math

import torch

from ..model.types import Gaussians


def budget_count(num_gaussians: int, k: float) -> int:
    if not 0.0 < k <= 1.0:
        raise ValueError(f"k must be in (0, 1], got {k}")
    return max(1, math.floor(k * num_gaussians))


def select_top_k_by_opacity(gaussians: Gaussians, k: float) -> Gaussians:
    # Per-scene top-k on opacity; gather keeps gradients for kept entries.
    n_k = budget_count(gaussians.opacities.shape[1], k)
    idx = torch.topk(gaussians.opacities, n_k, dim=1).indices

    def gather(x: torch.Tensor) -> torch.Tensor:
        view = idx.reshape(*idx.shape, *([1] * (x.dim() - 2)))
        return torch.gather(x, 1, view.expand(-1, -1, *x.shape[2:]))

    return Gaussians(
        means=gather(gaussians.means),
        covariances=gather(gaussians.covariances),
        harmonics=gather(gaussians.harmonics),
        opacities=gather(gaussians.opacities),
    )

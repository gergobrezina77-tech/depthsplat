import torch
from jaxtyping import Bool, Float, Int64
from torch import Tensor

from ..approach1.budget import budget_count
from ..model.types import Gaussians


def gumbel_topk_mask(
    s: Float[Tensor, "b g"],
    k: float,
    temperature: float,
    generator: torch.Generator | None = None,
) -> tuple[Float[Tensor, "b g"], Bool[Tensor, "b g"]]:
    # Straight-through mask: forward value is the exact hard top-k, backward
    # uses the gradient of a soft keep-probability p.
    z = (s - s.mean(1, keepdim=True).detach()) / s.std(1, keepdim=True).detach().clamp_min(1e-6)
    u = torch.rand(s.shape, generator=generator, device=s.device, dtype=s.dtype)
    u = u.clamp(1e-10, 1 - 1e-10)
    g = z + temperature * -torch.log(-torch.log(u))
    n_k = budget_count(s.shape[1], k)
    top = torch.topk(g, n_k, dim=1)
    m_hard = torch.zeros_like(g, dtype=torch.bool).scatter(1, top.indices, True)
    tau = top.values[:, -1:].detach()
    # clamp avoids exp overflow (inf * 0 -> NaN grads); same value, expm1 form is stable
    x = ((z - tau) / temperature).clamp(max=30)
    p = -torch.expm1(-torch.exp(x))
    m = m_hard.float() + (p - p.detach())
    return m, m_hard


def gather_gaussians(gaussians: Gaussians, idx: Int64[Tensor, "b n"]) -> Gaussians:
    def gather(x: torch.Tensor) -> torch.Tensor:
        view = idx.reshape(*idx.shape, *([1] * (x.dim() - 2)))
        return torch.gather(x, 1, view.expand(-1, -1, *x.shape[2:]))

    return Gaussians(
        means=gather(gaussians.means),
        covariances=gather(gaussians.covariances),
        harmonics=gather(gaussians.harmonics),
        opacities=gather(gaussians.opacities),
    )


def select_top_k(
    gaussians: Gaussians, scores: Float[Tensor, "b g"], k: float
) -> Gaussians:
    n_k = budget_count(scores.shape[1], k)
    idx = torch.topk(scores, n_k, dim=1).indices
    return gather_gaussians(gaussians, idx)


def random_scores(
    b: int, g: int, seed: int, device: torch.device | str
) -> Float[Tensor, "b g"]:
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(b, g, generator=generator).to(device)

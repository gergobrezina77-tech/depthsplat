import pytest
import torch

from src.approach1.budget import budget_count, select_top_k_by_opacity
from src.model.types import Gaussians


def make_gaussians(opacities: torch.Tensor) -> Gaussians:
    b, g = opacities.shape
    means = torch.arange(b * g * 3, dtype=torch.float32).reshape(b, g, 3)
    return Gaussians(
        means=means,
        covariances=torch.eye(3).expand(b, g, 3, 3).clone(),
        harmonics=torch.randn(b, g, 3, 4),
        opacities=opacities,
    )


OPACITIES = torch.tensor(
    [
        [0.1, 0.9, 0.3, 0.8, 0.2, 0.4, 0.5, 0.6],
        [0.7, 0.2, 0.95, 0.1, 0.3, 0.85, 0.4, 0.5],
    ]
)


def test_budget_count_formula():
    assert budget_count(256 * 256 * 2, 0.1) == 13107
    assert budget_count(256 * 256 * 2, 1.0) == 131072


def test_budget_count_float_error():
    assert budget_count(100, 0.29) == 29
    assert budget_count(256 * 256 * 2, 0.1) == 13107


def test_budget_count_min_one_and_invalid():
    assert budget_count(100, 1e-6) == 1
    for k in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            budget_count(100, k)


def test_select_keeps_highest_opacity_per_scene():
    g = make_gaussians(OPACITIES)
    out = select_top_k_by_opacity(g, 0.25)
    assert out.opacities.shape == (2, 2)
    expected = torch.tensor([[0.8, 0.9], [0.85, 0.95]])
    assert torch.equal(out.opacities.sort(dim=1).values, expected)
    sort = out.opacities.argsort(dim=1)
    kept_means = torch.gather(out.means, 1, sort[..., None].expand(-1, -1, 3))
    expected_idx = torch.tensor([[3, 1], [5, 2]])
    expected_means = torch.gather(g.means, 1, expected_idx[..., None].expand(-1, -1, 3))
    assert torch.equal(kept_means, expected_means)
    assert out.covariances.shape == (2, 2, 3, 3)
    assert out.harmonics.shape == (2, 2, 3, 4)


def test_select_is_per_scene_not_cross_batch():
    # Item 0 is entirely above 0.5, item 1 entirely below: a cross-batch
    # top-k would take everything from item 0.
    ops = torch.tensor([[0.9, 0.6, 0.8, 0.7], [0.1, 0.4, 0.2, 0.3]])
    out = select_top_k_by_opacity(make_gaussians(ops), 0.5)
    assert out.opacities.shape == (2, 2)
    assert torch.equal(out.opacities[0].sort().values, torch.tensor([0.8, 0.9]))
    assert torch.equal(out.opacities[1].sort().values, torch.tensor([0.3, 0.4]))


def test_select_k1_is_identity_set():
    g = make_gaussians(OPACITIES)
    out = select_top_k_by_opacity(g, 1.0)
    assert out.opacities.shape == g.opacities.shape
    assert torch.equal(out.opacities.sort(dim=1).values, g.opacities.sort(dim=1).values)


def test_select_gradient_reaches_kept_opacities():
    raw = OPACITIES.clone().requires_grad_(True)
    out = select_top_k_by_opacity(make_gaussians(raw), 0.25)
    out.opacities.sum().backward()
    kept = torch.topk(OPACITIES, 2, dim=1).indices
    expected = torch.zeros_like(OPACITIES, dtype=torch.bool).scatter_(1, kept, True)
    assert torch.equal(raw.grad != 0, expected)

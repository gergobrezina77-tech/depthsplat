import pytest
import torch

from src.approach2.select import gumbel_topk_mask, random_scores, select_top_k
from src.model.types import Gaussians


def test_gumbel_mask_exact_budget_per_scene():
    s = torch.randn(2, 1000, requires_grad=True)
    m, m_hard = gumbel_topk_mask(s, 0.1, 1.0)
    assert m_hard.sum(1).tolist() == [100, 100]
    assert torch.equal(m, m_hard.float())


def test_gumbel_mask_grad_reaches_scores():
    s = torch.randn(1, 1000, requires_grad=True)
    m, _ = gumbel_topk_mask(s, 0.1, 1.0)
    (m * torch.randn(1, 1000)).sum().backward()
    assert torch.isfinite(s.grad).all() and (s.grad != 0).any()


@pytest.mark.parametrize("k, expected", [(1e-6, 1), (1.0, 1000)])
def test_gumbel_mask_extreme_budgets(k, expected):
    s = torch.randn(1, 1000, requires_grad=True)
    m, m_hard = gumbel_topk_mask(s, k, 1.0)
    m.sum().backward()
    assert int(m_hard.sum()) == expected and torch.isfinite(s.grad).all()


def test_select_top_k_per_scene():
    b, g = 2, 1000
    means = torch.zeros(b, g, 3)
    means[..., 0] = torch.arange(g).float()
    gaussians = Gaussians(
        means=means,
        covariances=torch.zeros(b, g, 3, 3),
        harmonics=torch.zeros(b, g, 3, 4),
        opacities=torch.rand(b, g),
    )
    scores = torch.zeros(b, g)
    scores[0, :10] = 1.0
    scores[1, 990:] = 1.0
    kept = select_top_k(gaussians, scores, 0.01)
    assert kept.opacities.shape == (2, 10)
    assert kept.means.shape == (2, 10, 3)
    assert kept.covariances.shape == (2, 10, 3, 3)
    assert kept.harmonics.shape == (2, 10, 3, 4)
    assert set(kept.means[0, :, 0].long().tolist()) == set(range(10))
    assert set(kept.means[1, :, 0].long().tolist()) == set(range(990, 1000))


def test_random_scores_seeded():
    assert torch.equal(random_scores(1, 50, 0, "cpu"), random_scores(1, 50, 0, "cpu"))


@pytest.mark.parametrize("k", [1.0, 0.1])
def test_gumbel_mask_outlier_score_finite_grad(k):
    s = torch.randn(1, 1000)
    s[0, 0] = 1000.0  # standardized outlier -> huge (z - tau) / T
    s.requires_grad_(True)
    m, _ = gumbel_topk_mask(s, k, 0.1)
    (m * torch.randn(1, 1000)).sum().backward()
    assert torch.isfinite(s.grad).all()


def test_gumbel_mask_grad_has_no_scale_or_shift_component():
    # The mask depends only on the standardized z, so the gradient must not push the
    # score scale or offset; with a detached std it did, and scores grew without bound.
    s = (0.01 * torch.randn(2, 1000)).requires_grad_()
    m, _ = gumbel_topk_mask(s, 0.3, 1.0, generator=torch.Generator().manual_seed(0))
    (m * torch.randn(2, 1000, generator=torch.Generator().manual_seed(1))).sum().backward()
    centered = (s - s.mean(1, keepdim=True)).detach()
    radial = (s.grad * centered).sum(1) / (s.grad.norm(dim=1) * centered.norm(dim=1))
    assert radial.abs().max() < 1e-4
    assert (s.grad.sum(1).abs() / s.grad.norm(dim=1)).max() < 1e-4

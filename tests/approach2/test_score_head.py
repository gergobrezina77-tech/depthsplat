import pytest
import torch

from src.approach2.score_head import ScoreHead


def test_score_head_shape():
    assert ScoreHead()(torch.randn(2, 38, 256, 256)).shape == (2, 1, 256, 256)


def test_score_head_param_budget():
    n = sum(p.numel() for p in ScoreHead().parameters())
    assert 200_000 < n < 1_500_000


def test_score_head_rejects_bad_resolution():
    with pytest.raises(ValueError):
        ScoreHead()(torch.randn(1, 38, 100, 100))

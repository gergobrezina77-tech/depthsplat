from pathlib import Path

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


def test_refine_output_and_zero_init():
    head = ScoreHead(out_channels=38, zero_init_channels=37)
    out = head(torch.randn(2, 38, 64, 64))
    assert out.shape == (2, 38, 64, 64)
    assert torch.equal(out[:, :37], torch.zeros_like(out[:, :37]))
    assert out[:, 37].abs().sum() > 0


def test_v1_checkpoint_loads_strict():
    path = Path("outputs/approach2/overfit5_fix1/checkpoints/epoch_1499-step_3000.ckpt")
    if not path.exists():
        pytest.skip(f"{path} not found")
    state = torch.load(path, map_location="cpu")["state_dict"]
    prefix = "score_head."
    state = {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}
    ScoreHead().load_state_dict(state, strict=True)

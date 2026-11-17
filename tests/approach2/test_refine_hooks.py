import pytest
import torch
from torch import nn

from src.approach2.refine import (
    RAW_GROUPS,
    DepthCapture,
    RefineHook,
    gaussians_fp32,
    refine_stats,
    wrap_adapter_fp32,
)
from src.approach2.score_head import ScoreHead
from src.model.types import Gaussians


def _toy():
    regressor = nn.Identity()
    gaussian_head = nn.Conv2d(42, 37, 1)
    for p in gaussian_head.parameters():
        p.requires_grad = False
    head = ScoreHead(channels=[8, 8], out_channels=38, zero_init_channels=37)
    depth = DepthCapture(regressor)
    hook = RefineHook(gaussian_head, head, depth)
    return regressor, gaussian_head, head, depth, hook


def _x():
    x = torch.randn(2, 42, 16, 16)
    x[:, 3] = torch.rand(2, 16, 16) + 0.5
    return x


def _ref(gh, x):
    # Reference raw output without triggering the forward hook.
    return nn.functional.conv2d(x, gh.weight, gh.bias).float()


def _fwd(regressor, gaussian_head, x):
    return gaussian_head(regressor(x))


def test_depth_capture_is_channel_3():
    regressor = nn.Identity()
    cap = DepthCapture(regressor)
    x = _x()
    regressor(x)
    assert torch.equal(cap.pop(), x[:, 3:4])
    with pytest.raises(RuntimeError):
        cap.pop()


def test_hook_replaces_output_zero_init():
    regressor, gh, head, depth, hook = _toy()
    x = _x()
    out = _fwd(regressor, gh, x)
    assert torch.equal(out, _ref(gh, x))
    raw, delta, score = hook.pop()
    assert raw.shape == delta.shape == (2, 37, 16, 16)
    assert score.shape == (2, 1, 16, 16)
    assert out.dtype == raw.dtype == delta.dtype == score.dtype == torch.float32


def test_hook_output_uses_delta():
    regressor, gh, head, depth, hook = _toy()
    with torch.no_grad():
        head.out.bias[:37] = 1.0
    x = _x()
    out = _fwd(regressor, gh, x)
    raw, delta, _ = hook.pop()
    assert torch.allclose(out, _ref(gh, x) + 1)
    assert torch.allclose(out, raw + 1)


def test_bypass_returns_raw_and_clears():
    regressor, gh, head, depth, hook = _toy()
    x = _x()
    _fwd(regressor, gh, x)
    hook.pop()
    hook.enabled = False
    out = _fwd(regressor, gh, x)
    assert torch.equal(out, _ref(gh, x))
    with pytest.raises(RuntimeError):
        hook.pop()
    with pytest.raises(RuntimeError):
        depth.pop()
    hook.enabled = True
    x2 = _x()
    _fwd(regressor, gh, x2)
    hook.pop()
    # depth was consumed by the refine pass; re-capture to check freshness
    regressor(x2)
    assert torch.equal(depth.pop(), x2[:, 3:4])


def test_grad_reaches_head_only():
    regressor, gh, head, depth, hook = _toy()
    with torch.no_grad():
        head.out.bias[:37] = 1.0
    out = _fwd(regressor, gh, _x())
    _, _, score = hook.pop()
    (out.sum() + score.sum()).backward()
    assert all(p.grad is not None for p in head.parameters())
    assert all(p.grad is None for p in gh.parameters())


def test_inference_mode_works():
    regressor, gh, head, depth, hook = _toy()
    with torch.inference_mode():
        out = _fwd(regressor, gh, _x())
        raw, delta, score = hook.pop()
    assert not out.requires_grad and score.shape == (2, 1, 16, 16)


def test_adapter_fp32_wrapper():
    class Toy(nn.Module):
        def forward(self, x):
            return x @ x.T

    m = Toy()
    wrap_adapter_fp32(m)
    first = m.forward
    wrap_adapter_fp32(m)
    assert m.forward is first
    x = torch.randn(4, 3).half()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        assert m(x).dtype == torch.float32


def test_refine_stats():
    raw = torch.randn(2, 37, 8, 8)
    delta = torch.zeros_like(raw)
    s = refine_stats(raw, delta, 1e-3, 1e-1)
    for k in ("p50", "p99", "max"):
        assert torch.allclose(s[f"refine/scale_ratio/{k}"], torch.tensor(1.0))
    for g in RAW_GROUPS:
        assert s[f"refine/delta_abs/{g}"] == 0
    assert torch.equal(s["refine/opacity_mean"], s["refine/opacity_mean_frozen"])
    assert s["refine/opacity_mean"].ndim == 0
    delta[:, 3:6] = 1.0
    s = refine_stats(raw, delta, 1e-6, 1e6)
    assert s["refine/scale_ratio/p50"] > 1


def test_raw_groups_cover_all_channels():
    assert sorted(sum(RAW_GROUPS.values(), [])) == list(range(37))
    assert RAW_GROUPS["sh_dc"] == [10, 19, 28]


def test_gaussians_fp32():
    g = Gaussians(
        torch.zeros(1, 2, 3).half(), torch.zeros(1, 2, 3, 3).half(),
        torch.zeros(1, 2, 3, 9).half(), torch.zeros(1, 2).half(),
    )
    g = gaussians_fp32(g)
    assert all(t.dtype == torch.float32 for t in (g.means, g.covariances, g.harmonics, g.opacities))

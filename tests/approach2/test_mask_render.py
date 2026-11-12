from types import SimpleNamespace

import pytest
import torch

from src.model.decoder.cuda_splatting import render_cuda
from src.model.decoder.decoder_splatting_cuda import DecoderSplattingCUDA
from src.model.types import Gaussians

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

from src.approach2.mask_render import render_cuda_masked, render_masked  # noqa: E402

G, H, W = 500, 64, 64
DEV = "cuda"


def _scene(d_sh=4, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    means = torch.rand(G, 3, device=DEV, generator=g)
    means[:, :2] = means[:, :2] - 0.5
    means[:, 2] = 2 + 2 * means[:, 2]
    cov = (0.05**2) * torch.eye(3, device=DEV).expand(G, 3, 3).contiguous()
    sh = 0.5 * torch.randn(G, 3, d_sh, device=DEV, generator=g)
    opa = 0.3 + 0.6 * torch.rand(G, device=DEV, generator=g)
    return means, cov, sh, opa


def _cam(v=1):
    ext = torch.eye(4, device=DEV).expand(v, 4, 4).contiguous()
    intr = torch.tensor(
        [[1.0, 0, 0.5], [0, 1.0, 0.5], [0, 0, 1.0]], device=DEV
    ).expand(v, 3, 3).contiguous()
    near = torch.full((v,), 0.5, device=DEV)
    far = torch.full((v,), 10.0, device=DEV)
    bg = torch.zeros(v, 3, device=DEV)
    return ext, intr, near, far, bg


def _render(fn, means, cov, sh, opa, **kw):
    ext, intr, near, far, bg = _cam()
    return fn(
        ext, intr, near, far, (H, W), bg,
        means[None], cov[None], sh[None], opa[None], **kw,
    )


@pytest.mark.parametrize("d_sh", [1, 4, 9])
def test_mask_ones_matches_stock(d_sh):
    means, cov, sh, opa = _scene(d_sh)
    ref = _render(render_cuda, means, cov, sh, opa)
    out = _render(render_cuda_masked, means, cov, sh, opa,
                  gaussian_masks=torch.ones(1, G, device=DEV))
    assert (out - ref).abs().max() < 1e-4


def test_mask_zero_matches_removed():
    means, cov, sh, opa = _scene(9)
    masks = torch.ones(1, G, device=DEV)
    masks[0, :100] = 0
    out = _render(render_cuda_masked, means, cov, sh, opa, gaussian_masks=masks)
    ref = _render(render_cuda, means[100:], cov[100:], sh[100:], opa[100:])
    assert (out - ref).abs().max() < 1e-4


def test_mask_grad_reaches_masked_gaussians():
    means, cov, sh, opa = _scene()
    p = torch.rand(1, G, device=DEV, requires_grad=True)
    m_hard = torch.ones(1, G, device=DEV)
    m_hard[0, :100] = 0
    masks = m_hard + (p - p.detach())
    img = _render(render_cuda_masked, means, cov, sh, opa, gaussian_masks=masks)
    img.mean().backward()
    assert p.grad.shape == p.shape
    assert (p.grad[0, :100].abs() > 0).any()


def test_mask_none_raises():
    means, cov, sh, opa = _scene()
    with pytest.raises(ValueError):
        _render(render_cuda_masked, means, cov, sh, opa, gaussian_masks=None)


def test_render_masked_shape():
    means, cov, sh, opa = _scene()
    gaussians = Gaussians(means[None], cov[None], sh[None], opa[None])
    decoder = DecoderSplattingCUDA(
        SimpleNamespace(name="splatting_cuda"),
        SimpleNamespace(background_color=[0.0, 0.0, 0.0]),
    ).to(DEV)
    ext, intr, near, far, _ = _cam(2)
    target = {
        "extrinsics": ext[None],
        "intrinsics": intr[None],
        "near": near[None],
        "far": far[None],
        "image": torch.zeros(1, 2, 3, H, W, device=DEV),
    }
    masks = torch.ones(1, G, device=DEV)
    assert render_masked(decoder, gaussians, masks, target).shape == (1, 2, 3, H, W)

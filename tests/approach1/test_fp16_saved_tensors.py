import torch
from torch import nn

from src.approach1.model_wrapper_a1 import fp16_saved_tensors


def test_fp16_saved_tensors_matches_fp32_grads():
    torch.manual_seed(0)
    net = nn.Sequential(nn.Conv2d(3, 8, 3, padding=1), nn.GELU(), nn.Conv2d(8, 1, 3, padding=1))
    x = torch.randn(2, 3, 16, 16)
    net(x).square().mean().backward()
    ref = [p.grad.clone() for p in net.parameters()]
    net.zero_grad()
    with fp16_saved_tensors(net):
        loss = net(x).square().mean()
    loss.backward()
    for g, r in zip((p.grad for p in net.parameters()), ref):
        assert g.dtype == torch.float32
        assert (g - r).norm() / r.norm() < 1e-2


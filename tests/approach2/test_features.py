import math

import pytest
import torch
from torch import nn

from src.approach2.features import HeadInputHook, build_head_input, scores_to_gaussian_order


def test_hook_pop_clears():
    module = nn.Identity()
    hook = HeadInputHook(module)
    module(torch.ones(1))
    assert hook.pop().shape == (1,)
    with pytest.raises(RuntimeError):
        hook.pop()


def test_build_head_input():
    out = build_head_input(torch.zeros(4, 37, 8, 8), torch.full((2, 2, 8, 8), math.e))
    assert out.shape == (4, 38, 8, 8) and torch.allclose(out[:, 37], torch.ones(4, 8, 8))


def test_scores_to_gaussian_order():
    s = torch.arange(2 * 2 * 4 * 4.0).reshape(4, 1, 4, 4)  # (b v) 1 h w, b=2, v=2
    out = scores_to_gaussian_order(s, b=2, v=2)
    assert out.shape == (2, 32) and out[1, 16] == s[3, 0, 0, 0]

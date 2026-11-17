from unittest.mock import PropertyMock, patch

import pytest
import torch

from src.approach2.validation import budget_validation_step
from tests.approach2.test_refine_wrapper import built  # noqa: F401

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class _LoggerStub:
    def log_image(self, key, images, step=None, caption=None):
        pass


def test_refine_validation_logs(built):  # noqa: F811
    wrapper, batch = built
    logged = {}
    wrapper.log = lambda name, value, *a, **kw: logged.__setitem__(name, value)
    wrapper.on_validation_epoch_start()
    with patch.object(type(wrapper), "logger", new_callable=PropertyMock, return_value=_LoggerStub()):
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            budget_validation_step(wrapper, batch)
    for k in wrapper.a2_cfg.eval_budgets:
        assert f"val/psnr@{k}" in logged and f"val/psnr_minus_random@{k}" in logged
    assert abs(float(logged["val/psnr_full_minus_frozen"])) < 0.05


def test_refine_validation_uses_frozen_for_reference(built, monkeypatch):  # noqa: F811
    import src.approach2.validation as val

    wrapper, batch = built
    logged = {}
    monkeypatch.setattr(wrapper, "log", lambda name, value, *a, **kw: logged.__setitem__(name, value))
    frozen_outs, cut_inputs = [], []
    orig_frozen, orig_cut = wrapper.frozen_gaussians, val.select_top_k

    def spy_frozen(b):
        out = orig_frozen(b)
        frozen_outs.append(out)
        return out

    def spy_cut(g, s, k):
        cut_inputs.append(g)
        return orig_cut(g, s, k)

    monkeypatch.setattr(wrapper, "frozen_gaussians", spy_frozen)
    monkeypatch.setattr(val, "select_top_k", spy_cut)
    bias = wrapper.score_head.out.bias
    saved = bias.detach().clone()
    with torch.no_grad():
        bias[0] += 3.0  # opacity residual: corrected != frozen
    try:
        wrapper.on_validation_epoch_start()
        wrapper.a2_val_image_done = True  # skip image logging
        with torch.no_grad():
            budget_validation_step(wrapper, batch)
    finally:
        with torch.no_grad():
            bias.copy_(saved)
    n = len(wrapper.a2_cfg.eval_budgets)
    assert len(frozen_outs) == 1
    assert len(cut_inputs) == 2 * n
    assert sum(g is frozen_outs[0] for g in cut_inputs) == n
    assert abs(float(logged["val/psnr_full_minus_frozen"])) > 0.05

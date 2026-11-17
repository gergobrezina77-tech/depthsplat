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

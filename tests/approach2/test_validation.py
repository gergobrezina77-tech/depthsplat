from unittest.mock import PropertyMock, patch

import pytest
import torch

from src.approach2.validation import budget_validation_step
from tests.approach2.test_wrapper import built  # noqa: F401

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


class _LoggerStub:
    def __init__(self):
        self.images = []

    def log_image(self, key, images, step=None, caption=None):
        self.images.append((key, images, caption))


def test_budget_validation_step(built):  # noqa: F811
    wrapper, batch = built
    logged = {}
    wrapper.log = lambda name, value, *a, **kw: logged.__setitem__(name, value)
    stub = _LoggerStub()
    wrapper.on_validation_epoch_start()
    with patch.object(type(wrapper), "logger", new_callable=PropertyMock, return_value=stub):
        # Unshimmed batch is not available here; the step shims itself and shim is idempotent.
        budget_validation_step(wrapper, batch)
        budget_validation_step(wrapper, batch)
    for k in wrapper.a2_cfg.eval_budgets:
        assert f"val/psnr@{k}" in logged and f"val/psnr_minus_random@{k}" in logged
    assert len(stub.images) == 1
    assert wrapper.a2_val_image_done

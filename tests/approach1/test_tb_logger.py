import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from src.approach1.tb_logger import TensorBoardImageLogger


def _images(save_dir) -> EventAccumulator:
    acc = EventAccumulator(str(save_dir), size_guidance={"images": 0})
    acc.Reload()
    return acc


def test_logs_uint8_hwc(tmp_path):
    logger = TensorBoardImageLogger(save_dir=str(tmp_path))
    logger.log_image("val/x", [np.zeros((8, 8, 3), np.uint8)], step=0)
    logger.experiment.flush()
    assert "val/x" in _images(logger.log_dir).Tags()["images"]


def test_logs_chw_float_tensor_and_multiple(tmp_path):
    logger = TensorBoardImageLogger(save_dir=str(tmp_path))
    imgs = [torch.rand(3, 8, 8), torch.rand(3, 8, 8)]
    logger.log_image("val/y", imgs, step=1, caption=["a", "b"])
    logger.experiment.flush()
    tags = _images(logger.log_dir).Tags()["images"]
    assert "val/y/a" in tags and "val/y/b" in tags

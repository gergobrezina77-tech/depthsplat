import subprocess
from pathlib import Path

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_train_smoke(tmp_path):
    out = tmp_path / "train"
    subprocess.run(
        [
            "scripts/a2_overfit.sh", str(out),
            "trainer.max_steps=4",
            "trainer.val_check_interval=2",
            "checkpointing.every_n_train_steps=4",
            "data_loader.train.batch_size=1",
        ],
        check=True,
        cwd=Path(__file__).parents[2],
    )
    assert list((out / "checkpoints").glob("*.ckpt"))
    assert list((out / "tensorboard").rglob("events.out.tfevents.*"))

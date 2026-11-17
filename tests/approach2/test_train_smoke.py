import json
import math
import subprocess
from pathlib import Path

import pytest
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture(scope="module")
def score_ckpt(tmp_path_factory):
    out = tmp_path_factory.mktemp("score") / "train"
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
    return out


def test_train_smoke(score_ckpt):
    assert list((score_ckpt / "checkpoints").glob("*.ckpt"))
    assert list((score_ckpt / "tensorboard").rglob("events.out.tfevents.*"))


def test_resume_mode_mismatch_fails(score_ckpt, tmp_path):
    ckpt = next((score_ckpt / "checkpoints").glob("*.ckpt"))
    proc = subprocess.run(
        [
            "scripts/a2_overfit.sh", str(tmp_path / "train"),
            "trainer.max_steps=4",
            "data_loader.train.batch_size=1",
            "approach2.head_mode=refine",
            f"checkpointing.load={ckpt}",
        ],
        cwd=Path(__file__).parents[2],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "head_mode='score'" in proc.stderr and "head_mode='refine' was requested" in proc.stderr


def test_train_smoke_refine_fp16(tmp_path):
    out = tmp_path / "train"
    subprocess.run(
        [
            "scripts/a2_overfit.sh", str(out),
            "trainer.max_steps=10",  # train.py logs every 10 steps; fewer logs no train scalars
            "trainer.val_check_interval=5",
            "checkpointing.every_n_train_steps=10",
            "data_loader.train.batch_size=1",
            "approach2.head_mode=refine",
            "approach2.precision=16-mixed",
        ],
        check=True,
        cwd=Path(__file__).parents[2],
    )
    ckpts = list((out / "checkpoints").glob("*.ckpt"))
    assert ckpts
    assert torch.load(ckpts[0], map_location="cpu", weights_only=True)["a2_head_mode"] == "refine"
    acc = EventAccumulator(str(out / "tensorboard"))
    acc.Reload()
    for tag in ("train/amp_scale", "refine/scale_ratio/p50"):
        values = [e.value for e in acc.Scalars(tag)]
        assert values and all(math.isfinite(v) for v in values), tag
    peak = json.loads((out / "peak_memory_train.json").read_text())
    assert peak["max_memory_allocated"] > 0 and peak["max_memory_reserved"] > 0

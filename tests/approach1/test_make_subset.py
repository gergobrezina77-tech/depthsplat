import json
from pathlib import Path

import torch

from src.approach1.make_subset import build_subset, pick_eval_views


def _scene(key: str, frames: int) -> dict:
    return {
        "key": key,
        "url": f"http://x/{key}",
        "timestamps": torch.arange(frames),
        "cameras": torch.zeros(frames, 18),
        "images": [torch.zeros(3, dtype=torch.uint8) for _ in range(frames)],
    }


def _make_src(root: Path) -> None:
    scenes = [_scene("short", 30)] + [_scene(f"s{i}", 60 + i) for i in range(7)]
    (root / "train").mkdir(parents=True)
    torch.save(scenes, root / "train" / "000000.torch")
    index = {s["key"]: "000000.torch" for s in scenes}
    (root / "train" / "index.json").write_text(json.dumps(index))


def test_pick_eval_views() -> None:
    v = pick_eval_views(100, gap=45, num_targets=3)
    a, b = v["context"]
    assert b - a == 45 and 0 <= a and b < 100
    assert len(v["target"]) == 3 and all(a < t < b for t in v["target"])


def test_build_subset(tmp_path: Path) -> None:
    src, out = tmp_path / "src", tmp_path / "out"
    _make_src(src)
    build_subset(src, "train", 5, 0, out)

    chunk = torch.load(out / "train" / "000000.torch")
    keys = [s["key"] for s in chunk]
    assert keys == [f"s{i}" for i in range(5)]  # "short" skipped
    assert len(torch.load(out / "test" / "000000.torch")) == 5
    for split in ("train", "test"):
        index = json.loads((out / split / "index.json").read_text())
        assert index == {k: "000000.torch" for k in keys}

    ev = json.loads((out / "evaluation_index.json").read_text())
    assert set(ev) == set(keys)
    for v in ev.values():
        a, b = v["context"]
        assert b - a == 45 and all(a < t < b for t in v["target"])

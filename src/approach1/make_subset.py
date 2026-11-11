import argparse
import json
import random
from pathlib import Path

import torch


def pick_eval_views(
    num_frames: int,
    gap: int = 45,
    num_targets: int = 3,
    rng: random.Random | None = None,
) -> dict[str, list[int]]:
    # Context = [start, start + gap]; targets are sampled strictly between them.
    rng = rng or random.Random(0)
    start = rng.randint(0, num_frames - gap - 1)
    targets = sorted(rng.sample(range(start + 1, start + gap), num_targets))
    return {"context": [start, start + gap], "target": targets}


def build_subset(
    src: Path, split: str, num_scenes: int, seed: int, out: Path, gap: int = 45
) -> list[str]:
    rng = random.Random(seed)
    scenes: list[dict] = []
    # Walk chunk files that exist on disk (the index may list missing ones).
    for chunk_path in sorted((src / split).glob("*.torch")):
        for scene in torch.load(chunk_path):
            if len(scene["images"]) >= gap + 1:
                scenes.append(scene)
            if len(scenes) == num_scenes:
                break
        if len(scenes) == num_scenes:
            break
    if len(scenes) < num_scenes:
        raise RuntimeError(f"found only {len(scenes)} usable scenes")

    keys = [s["key"] for s in scenes]
    index = {k: "000000.torch" for k in keys}
    # The upstream val stage reads test/, so write the same scenes to both.
    for name in ("train", "test"):
        (out / name).mkdir(parents=True, exist_ok=True)
        torch.save(scenes, out / name / "000000.torch")
        (out / name / "index.json").write_text(json.dumps(index))

    evaluation = {
        s["key"]: pick_eval_views(len(s["images"]), gap, rng=rng) for s in scenes
    }
    (out / "evaluation_index.json").write_text(json.dumps(evaluation))
    return keys


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num-scenes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    picked = build_subset(args.src, args.split, args.num_scenes, args.seed, args.out)
    print("picked scenes:", picked)

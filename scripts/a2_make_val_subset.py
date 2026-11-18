"""Build a fixed RE10K val subset from the local test split (superset of heldout10).

Writes <out>/{test,train}/ chunks + index.json, <out>/evaluation_index.json, the same index under
assets/, and a test index with the val scenes removed (for reporting on the rest of the test split).
Usage: poetry run python scripts/a2_make_val_subset.py [--n 200] [--seed 0] [--out <root>]
"""
import argparse
import json
import random
from pathlib import Path

import torch

SRC = Path("/mnt/hdd/depthsplat/datasets/re10k/test")
EVAL_INDEX = Path("assets/evaluation_index_re10k.json")
HELDOUT = Path("assets/evaluation_index_a2_val10.json")
SCENES_PER_CHUNK = 10

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=200)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=Path, default=Path("/mnt/hdd/depthsplat/datasets/re10k_val200"))
args = parser.parse_args()

test_index = json.loads((SRC / "index.json").read_text())
eval_index = json.loads(EVAL_INDEX.read_text())
heldout = json.loads(HELDOUT.read_text())

# Candidates: local test scenes with eval view indices; heldout10 is always included.
candidates = sorted(k for k in test_index if eval_index.get(k) and k not in heldout)
keys = sorted(list(heldout) + random.Random(args.seed).sample(candidates, args.n - len(heldout)))
assert all(k in test_index for k in keys)

wanted = set(keys)
examples = {}
for chunk in sorted({test_index[k] for k in keys}):
    for example in torch.load(SRC / chunk):
        if example["key"] in wanted:
            examples[example["key"]] = example
assert len(examples) == len(keys), f"found {len(examples)} of {len(keys)} scenes"

test_dir = args.out / "test"
test_dir.mkdir(parents=True, exist_ok=False)
chunk_index = {}
for i in range(0, len(keys), SCENES_PER_CHUNK):
    name = f"{i // SCENES_PER_CHUNK:06d}.torch"
    torch.save([examples[k] for k in keys[i:i + SCENES_PER_CHUNK]], test_dir / name)
    chunk_index.update({k: name for k in keys[i:i + SCENES_PER_CHUNK]})
(test_dir / "index.json").write_text(json.dumps(chunk_index))
(args.out / "train").symlink_to("test")

val_index = {k: heldout.get(k, eval_index.get(k)) for k in keys}
(args.out / "evaluation_index.json").write_text(json.dumps(val_index))
Path(f"assets/evaluation_index_a2_val{args.n}.json").write_text(json.dumps(val_index))
rest = {k: (None if k in wanted else v) for k, v in eval_index.items()}
Path(f"assets/evaluation_index_re10k_minus_val{args.n}.json").write_text(json.dumps(rest))
print(f"{len(keys)} scenes in {len(range(0, len(keys), SCENES_PER_CHUNK))} chunks -> {args.out}")
print(f"test minus val: {sum(1 for k, v in rest.items() if v and k in test_index)} local scenes")

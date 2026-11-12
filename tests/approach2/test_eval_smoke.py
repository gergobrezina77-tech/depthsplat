import json
import subprocess
from pathlib import Path

import pytest
import torch

ROOT = Path("/mnt/hdd/depthsplat/datasets/re10k_overfit5")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def test_eval_budget_table(tmp_path):
    out = tmp_path / "eval"
    subprocess.run(
        [
            "poetry", "run", "python", "-m", "src.approach2.evaluate",
            "+experiment=re10k", "+approach2=re10k_small", "mode=test",
            "dataset/view_sampler=evaluation",
            f"dataset.view_sampler.index_path={ROOT / 'evaluation_index.json'}",
            "dataset.view_sampler.num_context_views=2",
            f"dataset.roots=[{ROOT}]",
            "approach2.eval_budgets=[0.1,1.0]",
            "approach2.eval_methods=[random,opacity]",
            f"output_dir={out}",
        ],
        check=True,
        cwd=Path(__file__).parents[2],
    )
    data = json.loads((out / "metrics" / "budget_table.json").read_text())
    assert len(data["scenes"]) == 5
    rows = data["rows"]
    assert {(r["method"], r["k"]) for r in rows} == {
        (m, k) for m in ["random", "opacity"] for k in [0.1, 1.0]
    }
    assert all(r["kept"] == {0.1: 13107, 1.0: 131072}[r["k"]] for r in rows)
    assert all(r["scenes"] == 5 for r in rows)
    assert len(json.loads((out / "metrics" / "per_scene.json").read_text())) == 20
    assert all(r["head_time"] > 0 and r["encode_time"] > 0 for r in rows)

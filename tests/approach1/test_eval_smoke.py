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
            "poetry", "run", "python", "-m", "src.approach1.evaluate",
            "+experiment=re10k", "+approach1=re10k_small", "mode=test",
            "dataset/view_sampler=evaluation",
            f"dataset.view_sampler.index_path={ROOT / 'evaluation_index.json'}",
            "dataset.view_sampler.num_context_views=2",
            f"dataset.roots=[{ROOT}]",
            "approach1.eval_budgets=[0.1,1.0]",
            f"output_dir={out}",
        ],
        check=True,
        cwd=Path(__file__).parents[2],
    )
    table = json.loads((out / "metrics" / "budget_table.json").read_text())
    assert len(table) == 2
    assert [r["kept"] for r in table] == [13107, 131072]
    assert all(r["scenes"] == 5 for r in table)

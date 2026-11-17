import json
import subprocess
from pathlib import Path

import pytest
import torch

from src.approach2.build import build_wrapper_a2
from tests.approach2.test_refine_wrapper import ROOT, _compose

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

METHODS = ["refine", "refine_random", "refine_opacity", "random", "opacity"]


def _eval_cmd(out, *extra):
    return [
        "poetry", "run", "python", "-m", "src.approach2.evaluate",
        "+experiment=re10k", "+approach2=re10k_small", "mode=test",
        "dataset/view_sampler=evaluation",
        f"dataset.view_sampler.index_path={ROOT / 'evaluation_index.json'}",
        "dataset.view_sampler.num_context_views=2",
        f"dataset.roots=[{ROOT}]",
        "approach2.eval_budgets=[0.1,1.0]",
        f"output_dir={out}",
        *extra,
    ]


def _save_zero_init_ckpt(path):
    wrapper, _, _, _ = build_wrapper_a2(_compose(["approach2.head_mode=refine"]))
    ckpt = {"state_dict": wrapper.state_dict()}
    wrapper.on_save_checkpoint(ckpt)
    torch.save(ckpt, path)


def test_refine_eval_table(tmp_path):
    ckpt = tmp_path / "zero_refine.ckpt"
    _save_zero_init_ckpt(ckpt)
    out = tmp_path / "eval"
    subprocess.run(
        _eval_cmd(
            out,
            "approach2.head_mode=refine", "approach2.precision=16-mixed",
            f"approach2.eval_methods=[{','.join(METHODS)}]",
            f"approach2.head_ckpt={ckpt}",
        ),
        check=True,
        cwd=Path(__file__).parents[2],
    )
    rows = json.loads((out / "metrics" / "budget_table.json").read_text())["rows"]
    assert {(r["method"], r["k"]) for r in rows} == {(m, k) for m in METHODS for k in [0.1, 1.0]}
    assert all(r["kept"] == {0.1: 13107, 1.0: 131072}[r["k"]] for r in rows)
    assert all(r["head_time"] > 0 and r["encode_time"] > 0 for r in rows)
    psnr = {(r["method"], r["k"]): r["psnr"] for r in rows}
    assert abs(psnr[("refine", 1.0)] - psnr[("random", 1.0)]) < 0.05
    assert abs(psnr[("refine_random", 0.1)] - psnr[("random", 0.1)]) < 0.05


def test_wrong_method_for_mode_raises(tmp_path):
    out = tmp_path / "eval"
    res = subprocess.run(
        _eval_cmd(out, "approach2.head_mode=score", "approach2.eval_methods=[refine]"),
        cwd=Path(__file__).parents[2],
    )
    assert res.returncode != 0
    assert not (out / "metrics").exists()

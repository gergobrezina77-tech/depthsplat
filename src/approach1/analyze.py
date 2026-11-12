import argparse
import json
from pathlib import Path

import matplotlib
import torch
from hydra import compose, initialize_config_dir
from lightning_utilities.core.apply_func import apply_to_collection

from ..evaluation.metrics import compute_lpips, compute_psnr
from ..global_cfg import set_cfg
from ..model.types import Gaussians
from .budget import budget_count
from .build import build_wrapper

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FROZEN = "/mnt/hdd/depthsplat/pretrained/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth"


def load(ckpt: str, index: str, root: str):
    overrides = [
        "+experiment=re10k", "+approach1=re10k_small", "mode=test",
        "dataset/view_sampler=evaluation",
        f"dataset.view_sampler.index_path={index}",
        "dataset.view_sampler.num_context_views=2",
        f"dataset.roots=[{root}]",
        f"checkpointing.pretrained_model={ckpt}",
    ]
    with initialize_config_dir(config_dir=str(REPO / "config"), version_base=None):
        cfg = compose("main", overrides=overrides)
    set_cfg(cfg)
    wrapper, data_module, _, _ = build_wrapper(cfg)
    return wrapper.cuda().eval(), data_module


def gather(g: Gaussians, idx: torch.Tensor) -> Gaussians:
    def f(x: torch.Tensor) -> torch.Tensor:
        view = idx.reshape(*idx.shape, *([1] * (x.dim() - 2)))
        return torch.gather(x, 1, view.expand(-1, -1, *x.shape[2:]))
    return Gaussians(f(g.means), f(g.covariances), f(g.harmonics), f(g.opacities))


@torch.no_grad()
def analyze(tuned: str, frozen: str, index: str, root: str, out: Path) -> dict:
    # Compares frozen vs. tuned Gaussians on the first scene of the index (k=0.5 cut).
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    wf, data_module = load(frozen, index, root)
    wt, _ = load(tuned, index, root)
    batch = next(iter(data_module.test_dataloader()))
    batch = apply_to_collection(batch, torch.Tensor, lambda t: t.cuda())
    batch = wf.data_shim(batch)
    gt = batch["target"]["image"][0]
    _, v, _, h, w = batch["context"]["image"].shape

    G = {}
    for name, wrapper in (("frozen", wf), ("tuned", wt)):
        g = wrapper.encoder(batch["context"], 0, deterministic=False)
        G[name] = g["gaussians"] if isinstance(g, dict) else g
    n = G["frozen"].opacities.shape[1]

    def render(g: Gaussians) -> torch.Tensor:
        output = wf.decoder.forward(
            g, batch["target"]["extrinsics"], batch["target"]["intrinsics"],
            batch["target"]["near"], batch["target"]["far"], (h, w), depth_mode=None,
        )
        return output.color[0]

    def score(rgb: torch.Tensor) -> list[float]:
        return [round(compute_psnr(gt, rgb).mean().item(), 2), round(compute_lpips(gt, rgb).mean().item(), 3)]

    def topk_idx(g: Gaussians, k: float) -> torch.Tensor:
        return torch.topk(g.opacities, budget_count(n, k), dim=1).indices

    res: dict = {"scene": batch["scene"][0], "n": n}

    # 1. Per-Gaussian statistics; size = sqrt of the largest covariance eigenvalue.
    for name in ("frozen", "tuned"):
        o = G[name].opacities[0]
        size = torch.linalg.eigvalsh(G[name].covariances[0]).clamp_min(0).amax(-1).sqrt()
        keep = torch.zeros(n, dtype=torch.bool, device=o.device)
        keep[topk_idx(G[name], 0.5)[0]] = True
        q = torch.quantile(o, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=o.device))
        res[name] = {
            "opacity_mean": o.mean().item(),
            "opacity_quantiles_10_25_50_75_90": [round(x, 4) for x in q.tolist()],
            "frac_opacity_gt_0.5": (o > 0.5).float().mean().item(),
            "frac_opacity_gt_0.9": (o > 0.9).float().mean().item(),
            "frac_opacity_lt_0.05": (o < 0.05).float().mean().item(),
            "opacity_kept_mean": o[keep].mean().item(),
            "opacity_dropped_mean": o[~keep].mean().item(),
            "opacity_mass_top50": (o[keep].sum() / o.sum()).item(),
            "size_median": size.median().item(),
            "size_kept_median": size[keep].median().item(),
            "size_dropped_median": size[~keep].median().item(),
            "kept_per_view": [keep[i * h * w:(i + 1) * h * w].float().mean().item() for i in range(v)],
        }
        G[name + "_keep"] = keep
        G[name + "_size"] = size

    # 2. How much each parameter group moved (same Gaussian index in both models).
    f, t = G["frozen"], G["tuned"]
    shift = (t.means - f.means).norm(dim=-1)[0]
    ratio = G["tuned_size"] / G["frozen_size"]
    res["delta"] = {
        "mean_shift_median": shift.median().item(),
        "mean_shift_rel_to_size": (shift / G["frozen_size"]).median().item(),
        "size_ratio_median": ratio.median().item(),
        "size_ratio_kept_median": ratio[G["tuned_keep"]].median().item(),
        "dc_color_abs_diff_mean": (t.harmonics[..., 0] - f.harmonics[..., 0]).abs().mean().item(),
        "opacity_abs_diff_mean": (t.opacities - f.opacities).abs().mean().item(),
        "top50_overlap": (G["frozen_keep"] & G["tuned_keep"]).float().sum().item() / budget_count(n, 0.5),
        "corr_opacity_frozen_tuned": torch.corrcoef(torch.stack([f.opacities[0], t.opacities[0]]))[0, 1].item(),
    }

    # 3. Random cut (mean of 5 seeds) and swap test: one model's parameters, other model's ranking.
    rnd = {}
    for k in (0.5, 0.25):
        for name in ("frozen", "tuned"):
            vals = []
            for seed in range(5):
                gen = torch.Generator(device="cuda").manual_seed(seed)
                idx = torch.randperm(n, generator=gen, device="cuda")[: budget_count(n, k)][None]
                vals.append(compute_psnr(gt, render(gather(G[name], idx))).mean().item())
            rnd[f"{name}_random@{k}"] = round(sum(vals) / len(vals), 2)
    res["random_cut_psnr_mean5"] = rnd

    renders = {"GT": gt}
    table = {}
    for params in ("frozen", "tuned"):
        rgb = render(G[params])
        table[f"{params} params, all"] = score(rgb)
        renders[f"{params} k=1"] = rgb
        for rank in ("frozen", "tuned"):
            rgb = render(gather(G[params], topk_idx(G[rank], 0.5)))
            table[f"{params} params, {rank} ranking @0.5"] = score(rgb)
            renders[f"{params} params / {rank} rank @0.5"] = rgb
    res["swap_table_psnr_lpips"] = table
    (out / "stats.json").write_text(json.dumps(res, indent=2))

    # 4. Figures: renders of target view 0, per-pixel maps of the context views, histograms.
    fig, ax = plt.subplots(1, len(renders), figsize=(3 * len(renders), 3.3))
    for a, (key, rgb) in zip(ax, renders.items()):
        a.imshow(rgb[0].permute(1, 2, 0).clamp(0, 1).cpu())
        a.set_title(key, fontsize=8)
        a.axis("off")
    fig.tight_layout()
    fig.savefig(out / "renders_target0.png", dpi=120)

    def per_view(x: torch.Tensor, i: int) -> torch.Tensor:
        return x[i * h * w:(i + 1) * h * w].reshape(h, w).float().cpu()

    rows = [
        ("context image", lambda name, i: batch["context"]["image"][0, i].permute(1, 2, 0).cpu(), None),
        ("opacity", lambda name, i: per_view(G[name].opacities[0], i), (0, 1)),
        ("kept @0.5", lambda name, i: per_view(G[name + "_keep"], i), (0, 1)),
        ("log10 size", lambda name, i: per_view(G[name + "_size"].log10(), i), (-3, -0.5)),
    ]
    fig, ax = plt.subplots(len(rows), 2 * v, figsize=(3 * 2 * v, 3 * len(rows)))
    for r, (label, fn, lim) in enumerate(rows):
        for c, (name, i) in enumerate([(nm, i) for nm in ("frozen", "tuned") for i in range(v)]):
            im = ax[r, c].imshow(
                fn(name, i), vmin=lim[0] if lim else None, vmax=lim[1] if lim else None,
                cmap=None if lim is None else "viridis",
            )
            ax[r, c].set_title(f"{name} view{i}: {label}", fontsize=8)
            ax[r, c].axis("off")
            if lim is not None and c == 2 * v - 1:
                fig.colorbar(im, ax=ax[r, c], fraction=0.046)
    fig.tight_layout()
    fig.savefig(out / "maps_context.png", dpi=110)

    fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
    for name in ("frozen", "tuned"):
        ax[0].hist(G[name].opacities[0].cpu(), bins=100, alpha=0.5, label=name)
        ax[1].hist(G[name + "_size"].log10().cpu(), bins=100, alpha=0.5, label=name)
    ax[0].set_title("opacity")
    ax[1].set_title("log10 size (sqrt max eigval)")
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(out / "histograms.png", dpi=110)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tuned", required=True)
    parser.add_argument("--frozen", default=FROZEN)
    parser.add_argument("--index", default="assets/evaluation_index_a1_overfit1.json")
    parser.add_argument("--root", default="/mnt/hdd/depthsplat/datasets/re10k_overfit5")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.tuned, args.frozen, args.index, args.root, args.out), indent=2))

import torch

from ..model.decoder.decoder import DecoderOutput
from .mask_render import render_masked
from .refine import gaussians_fp32, refine_stats
from .select import gumbel_topk_mask


def masked_training_step(wrapper, batch):
    batch = wrapper.data_shim(batch)
    cfg = wrapper.a2_cfg
    gaussians, s = wrapper.scores(batch)
    n_total = s.shape[1]

    # Everything after the encoder and head runs in fp32 (autocast off).
    with torch.autocast("cuda", enabled=False):
        # One continuous k per step; Gumbel top-k mask with straight-through gradients.
        k = cfg.k_min + (cfg.k_max - cfg.k_min) * torch.rand(()).item()
        m, m_hard = gumbel_topk_mask(s.float(), k, cfg.temperature)
        color = render_masked(wrapper.decoder, gaussians_fp32(gaussians), m.float(), batch["target"])
        output = DecoderOutput(color, None)
        loss = wrapper._photo_loss(output, batch, gaussians)
        psnr = wrapper._psnr(output, batch)

    kept = m_hard.sum(1).float().mean()
    wrapper.log("loss/masked", loss)
    wrapper.log("train/psnr_masked", psnr)
    wrapper.log("train/k", k)
    wrapper.log("train/kept", kept)
    wrapper.log("train/score_std", s.detach().std(1).mean())
    if cfg.head_mode == "refine":
        adapter_cfg = wrapper.encoder.gaussian_adapter.cfg
        for name, value in refine_stats(
            *wrapper.last_refine, adapter_cfg.gaussian_scale_min, adapter_cfg.gaussian_scale_max
        ).items():
            wrapper.log(name, value)
    wrapper.log("info/global_step", wrapper.global_step)  # hack for ckpt monitor

    if wrapper.global_rank == 0 and wrapper.global_step % wrapper.train_cfg.print_log_every_n_steps == 0:
        print(
            f"train step {wrapper.global_step}; scene = {[x[:20] for x in batch['scene']]}; "
            f"k = {k:.3f} ({kept.item():.0f}/{n_total}); loss = {loss.item():.6f}"
        )
    # Tell the data loader processes about the current step.
    if wrapper.step_tracker is not None:
        wrapper.step_tracker.set_step(wrapper.global_step)
    return loss

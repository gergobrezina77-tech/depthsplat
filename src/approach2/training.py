import torch

from ..model.decoder.decoder import DecoderOutput
from .mask_render import render_masked
from .select import gumbel_topk_mask


def masked_training_step(wrapper, batch):
    batch = wrapper.data_shim(batch)
    cfg = wrapper.a2_cfg
    gaussians, s = wrapper.scores(batch)
    n_total = s.shape[1]

    # One continuous k per step; Gumbel top-k mask with straight-through gradients.
    k = cfg.k_min + (cfg.k_max - cfg.k_min) * torch.rand(()).item()
    m, m_hard = gumbel_topk_mask(s, k, cfg.temperature)
    color = render_masked(wrapper.decoder, gaussians, m, batch["target"])
    output = DecoderOutput(color, None)
    loss = wrapper._photo_loss(output, batch, gaussians)

    kept = m_hard.sum(1).float().mean()
    wrapper.log("loss/masked", loss)
    wrapper.log("train/psnr_masked", wrapper._psnr(output, batch))
    wrapper.log("train/k", k)
    wrapper.log("train/kept", kept)
    wrapper.log("train/score_std", s.detach().std(1).mean())
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

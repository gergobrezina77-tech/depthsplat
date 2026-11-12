from math import isqrt

import torch
from einops import rearrange, repeat
from jaxtyping import Float
from mask_diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
from torch import Tensor

from ..geometry.projection import get_fov
from ..model.decoder.cuda_splatting import get_projection_matrix
from ..model.decoder.decoder_splatting_cuda import DecoderSplattingCUDA
from ..model.types import Gaussians


# SH_C2[2] and SH_C2[4] of the rasterizers' SH constants.
SH_C2_2 = 0.31539156525252005
SH_C2_4 = 0.5462742152960396


def e3nn_to_standard_sh(
    sh: Float[Tensor, "batch gaussian d_sh 3"],
) -> Float[Tensor, "batch gaussian d_sh 3"]:
    """The stock DepthSplat rasterizer evaluates SH in e3nn's axis convention, while
    the mask rasterizer uses the original 3DGS convention. This linear map converts
    coefficients so that both rasterizers render the same colors (degrees 0 to 2).
    """
    d_sh = sh.shape[-2]
    if d_sh == 1:
        return sh
    if d_sh not in (4, 9):
        raise ValueError("Only SH degrees 0, 1 and 2 are supported.")
    s = sh.unbind(dim=-2)
    out = [s[0], -s[2], -s[3], s[1]]
    if d_sh == 9:
        out += [
            -s[5],
            s[7],
            -0.5 * s[6] + 0.5 * (SH_C2_4 / SH_C2_2) * s[8],
            -s[4],
            -1.5 * (SH_C2_2 / SH_C2_4) * s[6] - 0.5 * s[8],
        ]
    return torch.stack(out, dim=-2)


def render_cuda_masked(
    extrinsics: Float[Tensor, "batch 4 4"],
    intrinsics: Float[Tensor, "batch 3 3"],
    near: Float[Tensor, " batch"],
    far: Float[Tensor, " batch"],
    image_shape: tuple[int, int],
    background_color: Float[Tensor, "batch 3"],
    gaussian_means: Float[Tensor, "batch gaussian 3"],
    gaussian_covariances: Float[Tensor, "batch gaussian 3 3"],
    gaussian_sh_coefficients: Float[Tensor, "batch gaussian 3 d_sh"],
    gaussian_opacities: Float[Tensor, "batch gaussian"],
    gaussian_masks: Float[Tensor, "batch gaussian"],
    scale_invariant: bool = True,
) -> Float[Tensor, "batch 3 height width"]:
    """Same as render_cuda (use_sh=True only), but with the MaskGaussian rasterizer.
    The mask multiplies each Gaussian's contribution and receives gradients.
    """
    if gaussian_masks is None:
        raise ValueError("gaussian_masks must be provided.")

    # Make sure everything is in a range where numerical issues don't appear.
    if scale_invariant:
        scale = 1 / near
        extrinsics = extrinsics.clone()
        extrinsics[..., :3, 3] = extrinsics[..., :3, 3] * scale[:, None]
        gaussian_covariances = gaussian_covariances * (scale[:, None, None, None] ** 2)
        gaussian_means = gaussian_means * scale[:, None, None]
        near = near * scale
        far = far * scale

    _, _, _, n = gaussian_sh_coefficients.shape
    degree = isqrt(n) - 1
    shs = rearrange(gaussian_sh_coefficients, "b g xyz n -> b g n xyz")
    shs = e3nn_to_standard_sh(shs).contiguous()

    b, _, _ = extrinsics.shape
    h, w = image_shape

    fov_x, fov_y = get_fov(intrinsics).unbind(dim=-1)
    tan_fov_x = (0.5 * fov_x).tan()
    tan_fov_y = (0.5 * fov_y).tan()

    projection_matrix = get_projection_matrix(near, far, fov_x, fov_y)
    projection_matrix = rearrange(projection_matrix, "b i j -> b j i")
    view_matrix = rearrange(extrinsics.inverse(), "b i j -> b j i")
    full_projection = view_matrix @ projection_matrix

    all_images = []
    for i in range(b):
        # Set up a tensor for the gradients of the screen-space means.
        mean_gradients = torch.zeros_like(gaussian_means[i], requires_grad=True)
        try:
            mean_gradients.retain_grad()
        except Exception:
            pass

        settings = GaussianRasterizationSettings(
            image_height=h,
            image_width=w,
            tanfovx=tan_fov_x[i].item(),
            tanfovy=tan_fov_y[i].item(),
            bg=background_color[i],
            scale_modifier=1.0,
            viewmatrix=view_matrix[i],
            projmatrix=full_projection[i],
            sh_degree=degree,
            campos=extrinsics[i, :3, 3],
            prefiltered=False,  # This matches the original usage.
            debug=False,
        )
        rasterizer = GaussianRasterizer(settings)

        row, col = torch.triu_indices(3, 3)

        # The mask package takes masks as the 4th positional argument, so use keywords.
        image, _ = rasterizer(
            means3D=gaussian_means[i],
            means2D=mean_gradients,
            shs=shs[i],
            opacities=gaussian_opacities[i, ..., None],
            masks=gaussian_masks[i][:, None].float().contiguous(),
            cov3D_precomp=gaussian_covariances[i, :, row, col],
        )
        all_images.append(image)
    return torch.stack(all_images)


def render_masked(
    decoder: DecoderSplattingCUDA,
    gaussians: Gaussians,
    masks: Float[Tensor, "b g"],
    target: dict,
) -> Float[Tensor, "b v 3 h w"]:
    """Mirrors DecoderSplattingCUDA.forward, with a per-Gaussian mask."""
    extrinsics = target["extrinsics"]
    b, v, _, _ = extrinsics.shape
    _, _, _, h, w = target["image"].shape
    color = render_cuda_masked(
        rearrange(extrinsics, "b v i j -> (b v) i j"),
        rearrange(target["intrinsics"], "b v i j -> (b v) i j"),
        rearrange(target["near"], "b v -> (b v)"),
        rearrange(target["far"], "b v -> (b v)"),
        (h, w),
        repeat(decoder.background_color, "c -> (b v) c", b=b, v=v),
        repeat(gaussians.means, "b g xyz -> (b v) g xyz", v=v),
        repeat(gaussians.covariances, "b g i j -> (b v) g i j", v=v),
        repeat(gaussians.harmonics, "b g c d_sh -> (b v) g c d_sh", v=v),
        repeat(gaussians.opacities, "b g -> (b v) g", v=v),
        repeat(masks, "b g -> (b v) g", v=v),
    )
    return rearrange(color, "(b v) c h w -> b v c h w", b=b, v=v)

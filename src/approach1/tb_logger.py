import numpy as np
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only


class TensorBoardImageLogger(TensorBoardLogger):
    """TensorBoard logger exposing the wandb-style `log_image` that upstream calls."""

    @rank_zero_only
    def log_image(
        self,
        key: str,
        images: list,
        step: int | None = None,
        caption: list[str] | None = None,
        **kwargs,
    ) -> None:
        for i, image in enumerate(images):
            # Caption (if given) replaces the index in the tag.
            suffix = caption[i] if caption is not None else str(i)
            tag = key if len(images) == 1 and caption is None else f"{key}/{suffix}"
            if isinstance(image, torch.Tensor):
                image = image.detach().cpu().float().clip(0, 1).numpy()
                fmt = "HWC" if image.shape[-1] in (3, 4) else "CHW"
            else:
                # prep_image returns uint8 HWC numpy.
                image = np.asarray(image)
                fmt = "HWC"
            self.experiment.add_image(tag, image, global_step=step, dataformats=fmt)

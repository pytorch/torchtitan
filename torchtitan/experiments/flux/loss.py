from contextlib import nullcontext
from typing import Any, ContextManager, Dict

import torch
from torch import Tensor

from torchtitan.experiments.flux.model.model import FluxModel
from torchtitan.experiments.flux.utils import predict_noise


class FluxLossStep:
    # def __init__(self, guidance: float):
    #     self._guidance = guidance

    def __call__(
        self,
        pred: Tensor,
        target_imf: Tensor,  # target_
    ) -> Tensor:
        # latents = batch[
        #     "img_encoding"
        # ]  # This is the latent image after auto encoder. TODO(jianiw): setup auto encoder before this step
        # clip_encodings = batch["clip_text_encoding"]
        # t5_encodings = batch["t5_text_encoding"]

        # bsz = latents.shape[0]

        with torch.no_grad():
            noise = torch.randn_like(target_latents)
            # timesteps = torch.rand((bsz,)).to(latents)
            # sigmas = timesteps.view(-1, 1, 1, 1)
            # noisy_latents = (1 - sigmas) * latents + sigmas * noise
            # guidance = torch.full((bsz,), self._guidance).to(latents)

        # pred = predict_noise(
        #     model, noisy_latents, clip_encodings, t5_encodings, timesteps, guidance
        # )

        target = noise - target_latents
        loss = torch.nn.functional.mse_loss(pred.float(), target.float().detach())
        return loss

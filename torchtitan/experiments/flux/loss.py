# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch


class FluxLossStep:
    def __call__(
        self,
        pred: torch.Tensor,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        guidane: float,
    ) -> torch.Tensor:

        clip_encodings = input_dict["clip_encodings"]
        t5_encodings = input_dict["t5_encodings"]

        bsz = labels.shape[0]

        bsz = labels.shape[0]

        with torch.no_grad():
            noise = torch.randn_like(labels)
            timesteps = torch.rand((bsz,)).to(labels)
            sigmas = timesteps.view(-1, 1, 1, 1)
            noisy_latents = (1 - sigmas) * labels + sigmas * noise
            guidance_vec = torch.full((bsz,), guidance).to(labels)

        target = noise - labels
        # pred = predict_noise(
        #     model, noisy_latents, clip_encodings, t5_encodings, timesteps, guidance
        # )

        target = noise - target_latents
        loss = torch.nn.functional.mse_loss(pred.float(), target.float().detach())
        return loss

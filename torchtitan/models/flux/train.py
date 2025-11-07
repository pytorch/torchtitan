# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch

from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import utils as dist_utils

from torchtitan.models.flux.infra.parallelize import parallelize_encoders
from torchtitan.models.flux.model.autoencoder import load_ae
from torchtitan.models.flux.model.hf_embedder import FluxEmbedder
from torchtitan.models.flux.utils import (
    create_position_encoding_for_latents,
    pack_latents,
    preprocess_data,
)
from torchtitan.train import main, Trainer


class FluxTrainer(Trainer):
    def __init__(self, job_config: JobConfig):
        super().__init__(job_config)

        # Set random seed, and maybe enable deterministic mode
        # (mainly for debugging, expect perf loss).
        # For Flux model, we need distinct seed across FSDP ranks to ensure we randomly dropout prompts info in dataloader
        dist_utils.set_determinism(
            self.parallel_dims.world_mesh,
            self.device,
            job_config.debug,
            distinct_seed_mesh_dims=["dp_shard", "dp_replicate"],
        )

        # NOTE: self._dtype is the data type used for encoders (image encoder, T5 text encoder, CLIP text encoder).
        # We cast the encoders and it's input/output to this dtype.  If FSDP with mixed precision training is not used,
        # the dtype for encoders is torch.float32 (default dtype for Flux Model).
        # Otherwise, we use the same dtype as mixed precision training process.
        self._dtype = (
            TORCH_DTYPE_MAP[job_config.training.mixed_precision_param]
            if self.parallel_dims.dp_shard_enabled
            else torch.float32
        )

        # load components
        model_args = self.train_spec.model_args[job_config.model.flavor]

        self.autoencoder = load_ae(
            job_config.encoder.autoencoder_path,
            model_args.autoencoder_params,
            device=self.device,
            dtype=self._dtype,
            random_init=job_config.training.test_mode,
        )

        self.clip_encoder = FluxEmbedder(
            version=job_config.encoder.clip_encoder,
            random_init=job_config.training.test_mode,
        ).to(device=self.device, dtype=self._dtype)
        self.t5_encoder = FluxEmbedder(
            version=job_config.encoder.t5_encoder,
            random_init=job_config.training.test_mode,
        ).to(device=self.device, dtype=self._dtype)

        # Apply FSDP to the T5 model / CLIP model
        self.t5_encoder, self.clip_encoder = parallelize_encoders(
            t5_model=self.t5_encoder,
            clip_model=self.clip_encoder,
            parallel_dims=self.parallel_dims,
            job_config=job_config,
        )

        if job_config.validation.enable:
            self.validator.flux_init(
                device=self.device,
                _dtype=self._dtype,
                autoencoder=self.autoencoder,
                t5_encoder=self.t5_encoder,
                clip_encoder=self.clip_encoder,
            )

    def forward_backward_step(
        self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor
    ) -> torch.Tensor:
        # generate t5 and clip embeddings
        input_dict["image"] = labels
        input_dict = preprocess_data(
            device=self.device,
            dtype=self._dtype,
            autoencoder=self.autoencoder,
            clip_encoder=self.clip_encoder,
            t5_encoder=self.t5_encoder,
            batch=input_dict,
        )
        labels = input_dict["img_encodings"]

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        # explicitly convert flux model to be Bfloat16 no matter FSDP is applied or not
        model = self.model_parts[0]

        # image in latent space transformed by self.auto_encoder
        clip_encodings = input_dict["clip_encodings"]
        t5_encodings = input_dict["t5_encodings"]

        bsz = labels.shape[0]

        with torch.no_grad(), torch.device(self.device):
            noise = torch.randn_like(labels)
            timesteps = torch.rand((bsz,))
            sigmas = timesteps.view(-1, 1, 1, 1)
            latents = (1 - sigmas) * labels + sigmas * noise

        bsz, _, latent_height, latent_width = latents.shape

        POSITION_DIM = 3  # constant for Flux flow model
        with torch.no_grad(), torch.device(self.device):
            # Create positional encodings
            latent_pos_enc = create_position_encoding_for_latents(
                bsz, latent_height, latent_width, POSITION_DIM
            )
            text_pos_enc = torch.zeros(bsz, t5_encodings.shape[1], POSITION_DIM)

            # Patchify: Convert latent into a sequence of patches
            latents = pack_latents(latents)
            target = pack_latents(noise - labels)

        optional_context_parallel_ctx = (
            dist_utils.create_context_parallel_ctx(
                cp_mesh=self.parallel_dims.world_mesh["cp"],
                cp_buffers=[
                    latents,
                    latent_pos_enc,
                    t5_encodings,
                    text_pos_enc,
                    target,
                ],
                cp_seq_dims=[1, 1, 1, 1, 1],
                cp_no_restore_buffers={
                    latents,
                    latent_pos_enc,
                    t5_encodings,
                    text_pos_enc,
                    target,
                },
                cp_rotate_method=self.job_config.parallelism.context_parallel_rotate_method,
            )
            if self.parallel_dims.cp_enabled
            else None
        )
        with self.train_context(optional_context_parallel_ctx):
            with self.maybe_enable_amp:
                latent_noise_pred = model(
                    img=latents,
                    img_ids=latent_pos_enc,
                    txt=t5_encodings,
                    txt_ids=text_pos_enc,
                    y=clip_encodings,
                    timesteps=timesteps,
                )

                loss = self.loss_fn(latent_noise_pred, target)
            # latent_noise_pred.shape=(bs, seq_len, vocab_size)
            # need to free to before bwd to avoid peaking memory
            del (latent_noise_pred, noise, target)
            loss.backward()

        return loss


if __name__ == "__main__":
    main(FluxTrainer)

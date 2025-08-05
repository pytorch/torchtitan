# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional

import torch

from torchtitan.config import ConfigManager, JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import utils as dist_utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.train import Trainer

from .infra.parallelize import parallelize_encoders
from .model.autoencoder import load_ae
from .model.hf_embedder import FluxEmbedder
from .utils import (
    create_position_encoding_for_latents,
    pack_latents,
    preprocess_data,
    unpack_latents,
)


class FluxTrainer(Trainer):
    def __init__(self, job_config: JobConfig):
        super().__init__(job_config)

        # Set random seed, and maybe enable deterministic mode
        # (mainly for debugging, expect perf loss).
        # For Flux model, we need distinct seed across FSDP ranks to ensure we randomly dropout prompts info in dataloader
        dist_utils.set_determinism(
            self.parallel_dims.world_mesh,
            self.device,
            job_config.training.seed,
            job_config.training.deterministic,
            distinct_seed_mesh_dim="dp_shard",
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

        if job_config.validation.enabled:
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
        # explicitely convert flux model to be Bfloat16 no matter FSDP is applied or not
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

        with self.maybe_enable_amp:
            latent_noise_pred = model(
                img=latents,
                img_ids=latent_pos_enc,
                txt=t5_encodings,
                txt_ids=text_pos_enc,
                y=clip_encodings,
                timesteps=timesteps,
            )

            # Convert sequence of patches to latent shape
            pred = unpack_latents(latent_noise_pred, latent_height, latent_width)
            target = noise - labels
            loss = self.loss_fn(pred, target)
        # pred.shape=(bs, seq_len, vocab_size)
        # need to free to before bwd to avoid peaking memory
        del (pred, noise, target)
        loss.backward()

        return loss


if __name__ == "__main__":
    init_logger()
    config_manager = ConfigManager()
    config = config_manager.parse_args()
    trainer: Optional[FluxTrainer] = None

    try:
        trainer = FluxTrainer(config)
        if config.checkpoint.create_seed_checkpoint:
            assert (
                int(os.environ["WORLD_SIZE"]) == 1
            ), "Must create seed checkpoint using a single device, to disable sharding."
            assert (
                config.checkpoint.enable_checkpoint
            ), "Must enable checkpointing when creating a seed checkpoint."
            trainer.checkpointer.save(curr_step=0, last_step=True)
            logger.info("Created seed checkpoint")
        else:
            trainer.train()
    except Exception:
        if trainer:
            trainer.close()
        raise
    else:
        trainer.close()
        torch.distributed.destroy_process_group()
        logger.info("Process group destroyed.")

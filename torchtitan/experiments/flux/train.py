# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional

import torch

from torchtitan.config_manager import JobConfig
from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.flux.model.autoencoder import load_ae
from torchtitan.experiments.flux.model.hf_embedder import FluxEmbedder
from torchtitan.experiments.flux.model.model import FluxModel
from torchtitan.experiments.flux.parallelize_flux import parallelize_encoders
from torchtitan.experiments.flux.utils import (
    create_position_encoding_for_latents,
    pack_latents,
    preprocess_flux_data,
    unpack_latents,
)
from torchtitan.tools.logging import init_logger, logger
from torchtitan.train import Trainer


class FluxTrainer(Trainer):
    def __init__(self, job_config: JobConfig):
        super().__init__(job_config)

        self.preprocess_fn = preprocess_flux_data
        # NOTE: self._dtype is the data type used for encoders (image encoder, T5 text encoder, CLIP text encoder).
        # We cast the encoders and it's input/output to this dtype.
        # For Flux model, we use FSDP with mixed precision training.
        self._dtype = torch.bfloat16
        self._seed = job_config.training.seed
        self._guidance = job_config.training.guidance

        # load components
        model_config = self.train_spec.config[job_config.model.flavor]

        self.autoencoder = load_ae(
            job_config.encoder.auto_encoder_path,
            model_config.autoencoder_params,
            device=self.device,
            dtype=self._dtype,
        )
        self.clip_encoder = FluxEmbedder(version=job_config.encoder.clip_encoder).to(
            device=self.device, dtype=self._dtype
        )
        self.t5_encoder = FluxEmbedder(version=job_config.encoder.t5_encoder).to(
            device=self.device, dtype=self._dtype
        )

        # Apply FSDP to the T5 model / CLIP model
        self.t5_encoder, self.clip_encoder = parallelize_encoders(
            t5_model=self.t5_encoder,
            clip_model=self.clip_encoder,
            world_mesh=self.world_mesh,
            parallel_dims=self.parallel_dims,
            job_config=job_config,
        )

    def _predict_noise(
        self,
        model: FluxModel,
        latents: torch.Tensor,
        clip_encodings: torch.Tensor,
        t5_encodings: torch.Tensor,
        timesteps: torch.Tensor,
        guidance: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Use Flux's flow-matching model to predict the noise in image latents.
        Args:
            model (FluxFlowModel): The Flux flow model.
            latents (Tensor): Image encodings from the Flux autoencoder.
                Shape: [bsz, 16, latent height, latent width]
            clip_encodings (Tensor): CLIP text encodings.
                Shape: [bsz, 768]
            t5_encodings (Tensor): T5 text encodings.
                Shape: [bsz, sequence length, 256 or 512]
            timesteps (Tensor): The amount of noise (0 to 1).
                Shape: [bsz]
            guidance (Optional[Tensor]): The guidance value (1.5 to 4) if guidance-enabled model.
                Shape: [bsz]
                Default: None
            model_ctx (ContextManager): Optional context to wrap the model call (e.g. for activation offloading)
                Default: nullcontext
        Returns:
            Tensor: The noise prediction.
                Shape: [bsz, 16, latent height, latent width]
        """
        bsz, _, latent_height, latent_width = latents.shape

        POSITION_DIM = 3  # constant for Flux flow model
        with torch.no_grad():
            # Create positional encodings
            latent_pos_enc = create_position_encoding_for_latents(
                bsz, latent_height, latent_width, POSITION_DIM
            )
            text_pos_enc = torch.zeros(bsz, t5_encodings.shape[1], POSITION_DIM)

            # Convert latent into a sequence of patches
            latents = pack_latents(latents)

        # Predict noise
        latent_noise_pred = model(
            img=latents,
            img_ids=latent_pos_enc.to(latents),
            txt=t5_encodings.to(latents),
            txt_ids=text_pos_enc.to(latents),
            y=clip_encodings.to(latents),
            timesteps=timesteps.to(latents),
            guidance=guidance.to(latents) if guidance is not None else None,
        )

        # Convert sequence of patches to latent shape
        latent_noise_pred = unpack_latents(
            latent_noise_pred, latent_height, latent_width
        )

        return latent_noise_pred

    def train_step(self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor):
        # generate t5 and clip
        input_dict["image"] = labels
        input_dict = self.preprocess_fn(
            device=self.device,
            dtype=self._dtype,
            autoencoder=self.autoencoder,
            clip_encoder=self.clip_encoder,
            t5_encoder=self.t5_encoder,
            batch=input_dict,
        )
        labels = input_dict["img_encodings"]

        self.optimizers.zero_grad()

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        model_parts = self.model_parts
        world_mesh = self.world_mesh
        parallel_dims = self.parallel_dims

        # image in latent space transformed by self.auto_encoder
        clip_encodings = input_dict["clip_encodings"]
        t5_encodings = input_dict["t5_encodings"]

        bsz = labels.shape[0]

        with torch.no_grad():
            noise = torch.randn_like(labels)
            timesteps = torch.rand((bsz,)).to(labels)
            sigmas = timesteps.view(-1, 1, 1, 1)
            noisy_latents = (1 - sigmas) * labels + sigmas * noise
            guidance = torch.full((bsz,), self._guidance).to(labels)

        target = noise - labels

        assert len(model_parts) == 1

        pred = self._predict_noise(
            model_parts[0],
            noisy_latents,
            clip_encodings,
            t5_encodings,
            timesteps,
            guidance,
        )
        loss = self.loss_fn(pred, target)
        # pred.shape=(bs, seq_len, vocab_size)
        # need to free to before bwd to avoid peaking memory
        del (pred, noise, target)
        loss.backward()

        dist_utils.clip_grad_norm_(
            [p for m in model_parts for p in m.parameters()],
            self.job_config.training.max_norm,
            foreach=True,
            pp_mesh=self.world_mesh["pp"] if parallel_dims.pp_enabled else None,
        )
        self.checkpointer.maybe_wait_for_staging()
        self.optimizers.step()
        self.lr_schedulers.step()

        # log metrics
        if not self.metrics_processor.should_log(self.step):
            return

        if (
            parallel_dims.dp_replicate_enabled
            or parallel_dims.dp_shard_enabled
            or parallel_dims.cp_enabled
        ):
            loss = loss.detach()
            global_avg_loss, global_max_loss = (
                dist_utils.dist_mean(loss, world_mesh["dp_cp"]),
                dist_utils.dist_max(loss, world_mesh["dp_cp"]),
            )
        else:
            global_avg_loss = global_max_loss = loss.item()

        self.metrics_processor.log(self.step, global_avg_loss, global_max_loss)


if __name__ == "__main__":
    init_logger()
    config = JobConfig()
    config.maybe_add_custom_args()
    config.parse_args()
    trainer: Optional[FluxTrainer] = None

    try:
        trainer = FluxTrainer(config)
        if config.checkpoint.create_seed_checkpoint:
            assert int(
                os.environ["WORLD_SIZE"]
            ), "Must create seed checkpoint using a single device, to disable sharding."
            assert (
                config.checkpoint.enable_checkpoint
            ), "Must enable checkpointing when creating a seed checkpoint."
            trainer.checkpointer.save(curr_step=0, force=True)
            logger.info("Created seed checkpoint")
        else:
            trainer.train()
    finally:
        if trainer:
            trainer.close()

        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
            logger.info("Process group destroyed.")

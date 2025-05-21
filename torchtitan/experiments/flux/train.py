# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Optional

import torch

from torchtitan.config_manager import ConfigManager, JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.flux.dataset.tokenizer import FluxTokenizer
from torchtitan.experiments.flux.model.autoencoder import load_ae
from torchtitan.experiments.flux.model.hf_embedder import FluxEmbedder
from torchtitan.experiments.flux.parallelize_flux import parallelize_encoders
from torchtitan.experiments.flux.sampling import generate_image, save_image
from torchtitan.experiments.flux.utils import (
    create_position_encoding_for_latents,
    pack_latents,
    preprocess_data,
    unpack_latents,
)
from torchtitan.tools.logging import init_logger, logger
from torchtitan.train import Trainer


class FluxTrainer(Trainer):
    def __init__(self, job_config: JobConfig):
        super().__init__(job_config)

        # Set random seed, and maybe enable deterministic mode
        # (mainly for debugging, expect perf loss).
        # For Flux model, we need distinct seed across FSDP ranks to ensure we randomly dropout prompts info in dataloader
        dist_utils.set_determinism(
            self.world_mesh,
            self.device,
            job_config.training.seed,
            job_config.training.deterministic,
            distinct_seed_mesh_dim="dp_shard",
        )

        self.preprocess_fn = preprocess_data
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
        model_config = self.train_spec.config[job_config.model.flavor]

        self.autoencoder = load_ae(
            job_config.encoder.autoencoder_path,
            model_config.autoencoder_params,
            device=self.device,
            dtype=self._dtype,
        )

        self.clip_encoder = FluxEmbedder(
            version=job_config.encoder.clip_encoder,
        ).to(device=self.device, dtype=self._dtype)
        self.t5_encoder = FluxEmbedder(
            version=job_config.encoder.t5_encoder,
        ).to(device=self.device, dtype=self._dtype)

        # Apply FSDP to the T5 model / CLIP model
        self.t5_encoder, self.clip_encoder = parallelize_encoders(
            t5_model=self.t5_encoder,
            clip_model=self.clip_encoder,
            world_mesh=self.world_mesh,
            parallel_dims=self.parallel_dims,
            job_config=job_config,
        )

    def train_step(self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor):
        # generate t5 and clip embeddings
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
        assert len(self.model_parts) == 1
        # explicitely convert flux model to be Bfloat16 no matter FSDP is applied or not
        model = self.model_parts[0]

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
            latents = (1 - sigmas) * labels + sigmas * noise

        bsz, _, latent_height, latent_width = latents.shape

        POSITION_DIM = 3  # constant for Flux flow model
        with torch.no_grad():
            # Create positional encodings
            latent_pos_enc = create_position_encoding_for_latents(
                bsz, latent_height, latent_width, POSITION_DIM
            )
            text_pos_enc = torch.zeros(bsz, t5_encodings.shape[1], POSITION_DIM)

            # Patchify: Convert latent into a sequence of patches
            latents = pack_latents(latents)

        latent_noise_pred = model(
            img=latents,
            img_ids=latent_pos_enc.to(latents),
            txt=t5_encodings.to(latents),
            txt_ids=text_pos_enc.to(latents),
            y=clip_encodings.to(latents),
            timesteps=timesteps.to(latents),
        )

        # Convert sequence of patches to latent shape
        pred = unpack_latents(latent_noise_pred, latent_height, latent_width)
        target = noise - labels
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
            ft_pg = self.ft_manager.replicate_pg if self.ft_manager.enabled else None
            global_avg_loss, global_max_loss = (
                dist_utils.dist_mean(loss, world_mesh["dp_cp"], ft_pg),
                dist_utils.dist_max(loss, world_mesh["dp_cp"], ft_pg),
            )
        else:
            global_avg_loss = global_max_loss = loss.item()

        self.metrics_processor.log(self.step, global_avg_loss, global_max_loss)

        # Evaluate the model during training
        if (
            self.step % self.job_config.eval.eval_freq == 0
            or self.step == self.job_config.training.steps
        ):
            model.eval()
            # We need to set reshard_after_forward before last forward pass.
            # So the model wieghts are sharded the same way for checkpoint saving.
            model.final_layer.set_reshard_after_forward(True)
            self.eval_step()
            model.final_layer.set_reshard_after_forward(False)
            model.train()

    def eval_step(self, prompt: str = "A photo of a cat"):
        """
        Evaluate the Flux model.
        1) generate and save images every few steps. Currently, we run the eval and on the same
        prompts across all DP ranks. We will change this behavior to run on validation set prompts.
        Due to random noise generation, results could be different across DP ranks cause we assign
        different random seeds to each DP rank.
        2) [TODO] Calculate loss with fixed t value on validation set.
        """

        image = generate_image(
            device=self.device,
            dtype=self._dtype,
            job_config=self.job_config,
            model=self.model_parts[0],
            prompt=prompt,  # TODO(jianiw): change this to a prompt from validation set
            autoencoder=self.autoencoder,
            t5_tokenizer=FluxTokenizer(
                self.job_config.encoder.t5_encoder,
                max_length=self.job_config.encoder.max_t5_encoding_len,
            ),
            clip_tokenizer=FluxTokenizer(
                self.job_config.encoder.clip_encoder, max_length=77
            ),
            t5_encoder=self.t5_encoder,
            clip_encoder=self.clip_encoder,
        )

        save_image(
            name=f"image_rank{str(torch.distributed.get_rank())}_{self.step}.png",
            output_dir=os.path.join(
                self.job_config.job.dump_folder, self.job_config.eval.save_img_folder
            ),
            x=image,
            add_sampling_metadata=True,
            prompt=prompt,
        )


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

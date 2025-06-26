# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Iterable, Optional

import torch
from torch.distributed.fsdp import FSDPModule

from torchtitan.config_manager import ConfigManager, JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import utils as dist_utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.train import Trainer

from .dataset.tokenizer import build_flux_tokenizer
from .infra.parallelize import parallelize_encoders
from .model.autoencoder import load_ae
from .model.hf_embedder import FluxEmbedder
from .sampling import generate_image, save_image
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
            self.world_mesh,
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
        model_config = self.train_spec.config[job_config.model.flavor]

        self.autoencoder = load_ae(
            job_config.encoder.autoencoder_path,
            model_config.autoencoder_params,
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
            world_mesh=self.world_mesh,
            parallel_dims=self.parallel_dims,
            job_config=job_config,
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

    def eval_step(self, prompt: str = "A photo of a cat"):
        """
        Evaluate the Flux model.
        1) generate and save images every few steps. Currently, we run the eval and on the same
        prompts across all DP ranks. We will change this behavior to run on validation set prompts.
        Due to random noise generation, results could be different across DP ranks cause we assign
        different random seeds to each DP rank.
        2) [TODO] Calculate loss with fixed t value on validation set.
        """

        t5_tokenizer, clip_tokenizer = build_flux_tokenizer(self.job_config)

        image = generate_image(
            device=self.device,
            dtype=self._dtype,
            job_config=self.job_config,
            model=self.model_parts[0],
            prompt=prompt,  # TODO(jianiw): change this to a prompt from validation set
            autoencoder=self.autoencoder,
            t5_tokenizer=t5_tokenizer,
            clip_tokenizer=clip_tokenizer,
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

        # Reshard after run forward pass in eval_step.
        # This is to ensure the model weights are sharded the same way for checkpoint saving.
        for module in self.model_parts[0].modules():
            if isinstance(module, FSDPModule):
                module.reshard()

    def train_step(
        self, data_iterator: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):
        super().train_step(data_iterator)

        # Evaluate the model during training
        if (
            self.step % self.job_config.eval.eval_freq == 0
            or self.step == self.job_config.training.steps
        ):
            model = self.model_parts[0]
            model.eval()
            # We need to set reshard_after_forward before last forward pass.
            # So the model wieghts are sharded the same way for checkpoint saving.
            self.eval_step()
            model.train()


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

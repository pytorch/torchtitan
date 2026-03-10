# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from contextlib import AbstractContextManager
from dataclasses import dataclass, field, replace

import torch
import torch.nn as nn
from torch.distributed.pipelining.schedules import _PipelineSchedule

from torchtitan.components.dataloader import BaseDataLoader
from torchtitan.components.loss import LossFunction
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.components.validate import ValidationContext, Validator
from torchtitan.config import ParallelismConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.tools.logging import logger

from .flux_datasets import FluxDataLoader
from .inference.sampling import generate_image, save_image
from .model.autoencoder import AutoEncoder
from .model.hf_embedder import FluxEmbedder
from .tokenizer import FluxTokenizerContainer
from .utils import create_position_encoding_for_latents, pack_latents, preprocess_data


class FluxValidator(Validator):
    """
    Flux model validator focused on correctness and integration.

    Args:
        config: FluxValidator.Config configuration
        parallelism: Parallelism configuration
        dp_world_size: Data parallel world size
        dp_rank: Data parallel rank
        tokenizer: Tokenizer
        parallel_dims: Parallel dimensions
        loss_fn: Loss function to use for validation
        validation_context: Context manager for validation
        maybe_enable_amp: Context manager for AMP
        metrics_processor: Metrics processor
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Validator.Config):
        dataloader: BaseDataLoader.Config = field(
            default_factory=lambda: FluxDataLoader.Config(
                dataset="coco-validation",
                generate_timesteps=True,
            )
        )
        """DataLoader configuration for Flux validation"""

        all_timesteps: bool = False
        """Generate all 8 timesteps for each sample instead of round-robin"""

        save_img_count: int = -1
        """Number of images to save during validation (-1 for unlimited)"""

        save_img_folder: str = "validation_images"
        """Folder to save validation images"""

        enable_classifier_free_guidance: bool = False
        """Whether to use classifier-free guidance (CFG) during validation image generation.

        When enabled, the model runs two forward passes per denoising step — one with
        the text prompt and one without — then interpolates the results using
        `classifier_free_guidance_scale` to produce images that more closely follow
        the prompt. This typically yields higher-quality, more prompt-adherent images
        but doubles the compute cost per sampling step.

        Note: This only affects the image generation/sampling path (controlled by
        `save_img_count`), NOT the validation loss computation.
        """

        classifier_free_guidance_scale: float = 5.0
        """Interpolation weight for classifier-free guidance during sampling.

        Higher values steer the output more strongly toward the text prompt, producing
        sharper and more prompt-adherent images, but may reduce diversity or introduce
        artifacts. Typical values range from 1.0 (no guidance) to 10.0 (strong guidance).
        Only takes effect when `enable_classifier_free_guidance` is True.
        """

        denoising_steps: int = 50
        """How many denoising steps to sample when generating an image"""

    validation_dataloader: BaseDataLoader

    def __init__(
        self,
        config: Config,
        *,
        parallelism: ParallelismConfig,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        parallel_dims: ParallelDims,
        loss_fn: LossFunction,
        validation_context: ValidationContext,
        maybe_enable_amp: AbstractContextManager[None],
        local_batch_size: int,
        metrics_processor: MetricsProcessor | None = None,
        pp_schedule: _PipelineSchedule | None = None,
        pp_has_first_stage: bool | None = None,
        pp_has_last_stage: bool | None = None,
        **kwargs,
    ):
        self.config = config
        self.parallelism = parallelism
        self.tokenizer = tokenizer
        self.parallel_dims = parallel_dims
        self.loss_fn = loss_fn
        self.all_timesteps = config.all_timesteps

        assert isinstance(config.dataloader, FluxDataLoader.Config)
        assert isinstance(tokenizer, FluxTokenizerContainer)
        self.t5_tokenizer = tokenizer.t5_tokenizer
        self.clip_tokenizer = tokenizer.clip_tokenizer

        dl_config = replace(
            config.dataloader,
            infinite=config.steps != -1,
            generate_timesteps=not config.all_timesteps,
        )
        self.validation_dataloader = dl_config.build(
            dp_world_size=dp_world_size,
            dp_rank=dp_rank,
            local_batch_size=local_batch_size,
            tokenizer=tokenizer,
        )
        self.validation_context = validation_context
        self.maybe_enable_amp = maybe_enable_amp
        # pyrefly: ignore [bad-assignment]
        self.metrics_processor = metrics_processor

        if config.steps == -1:
            logger.warning(
                "Setting validation steps to -1 might cause hangs because of "
                "unequal sample counts across ranks when dataset is exhausted."
            )

    def flux_init(
        self,
        device: torch.device,
        _dtype: torch.dtype,
        autoencoder: AutoEncoder,
        t5_encoder: FluxEmbedder,
        clip_encoder: FluxEmbedder,
        dump_folder: str,
    ):
        # pyrefly: ignore [read-only]
        self.device = device
        self._dtype = _dtype
        self.autoencoder = autoencoder
        self.t5_encoder = t5_encoder
        self.clip_encoder = clip_encoder
        self.dump_folder = dump_folder

    @torch.no_grad()
    def validate(
        self,
        model_parts: list[nn.Module],
        step: int,
    ) -> None:
        # Set model to eval mode
        # TODO: currently does not support pipeline parallelism
        model = model_parts[0]
        model.eval()

        assert isinstance(self.config, FluxValidator.Config)
        save_img_count = self.config.save_img_count

        parallel_dims = self.parallel_dims

        accumulated_losses = []
        device_type = dist_utils.device_type
        num_steps = 0

        for input_dict, labels in self.validation_dataloader:
            if self.config.steps != -1 and num_steps >= self.config.steps:
                break

            prompt = input_dict.pop("prompt")
            if not isinstance(prompt, list):
                prompt = [prompt]
            for p in prompt:
                assert isinstance(p, str), f"prompt must be a string, got {type(p)}"
                if save_img_count != -1 and save_img_count <= 0:
                    break
                img_size = self.config.dataloader.img_size
                image = generate_image(
                    device=self.device,
                    dtype=self._dtype,
                    img_height=16 * (img_size // 16),
                    img_width=16 * (img_size // 16),
                    enable_classifier_free_guidance=self.config.enable_classifier_free_guidance,
                    denoising_steps=self.config.denoising_steps,
                    classifier_free_guidance_scale=self.config.classifier_free_guidance_scale,
                    model=model,
                    prompt=p,
                    autoencoder=self.autoencoder,
                    t5_tokenizer=self.t5_tokenizer,
                    clip_tokenizer=self.clip_tokenizer,
                    t5_encoder=self.t5_encoder,
                    clip_encoder=self.clip_encoder,
                )

                save_image(
                    name=f"image_rank{str(torch.distributed.get_rank())}_{step}.png",
                    output_dir=os.path.join(
                        self.dump_folder,
                        self.config.save_img_folder,
                    ),
                    x=image,
                    add_sampling_metadata=True,
                    prompt=p,
                )
                save_img_count -= 1

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
            labels = input_dict["img_encodings"].to(device_type)
            clip_encodings = input_dict["clip_encodings"]
            t5_encodings = input_dict["t5_encodings"]

            bsz = labels.shape[0]

            # If using all_timesteps we generate all 8 timesteps and expand our batch inputs here
            if self.all_timesteps:
                stratified_timesteps = torch.tensor(
                    [1 / 8 * (i + 0.5) for i in range(8)],
                    dtype=torch.float32,
                    device=self.device,
                ).repeat(bsz)
                clip_encodings = clip_encodings.repeat_interleave(8, dim=0)
                t5_encodings = t5_encodings.repeat_interleave(8, dim=0)
                labels = labels.repeat_interleave(8, dim=0)
            else:
                stratified_timesteps = input_dict.pop("timestep")

            # Note the tps may be inaccurate due to the generating image step not being counted
            self.metrics_processor.ntokens_since_last_log += labels.numel()

            # Apply timesteps here and update our bsz to efficiently compute all timesteps and samples in a single forward pass
            with torch.no_grad(), torch.device(self.device):
                noise = torch.randn_like(labels)
                timesteps = stratified_timesteps.to(labels)
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

            # Apply CP sharding if enabled
            if parallel_dims.cp_enabled:
                from torchtitan.distributed.context_parallel import cp_shard

                (
                    latents,
                    latent_pos_enc,
                    t5_encodings,
                    text_pos_enc,
                    target,
                ), _ = cp_shard(
                    parallel_dims.get_mesh("cp"),
                    (latents, latent_pos_enc, t5_encodings, text_pos_enc, target),
                    None,  # No attention masks for Flux
                    load_balancer_type=None,
                )

            with self.validation_context():
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

            del noise, target, latent_noise_pred, latents

            accumulated_losses.append(loss.detach())

            num_steps += 1

        # Compute average loss
        loss = torch.sum(torch.stack(accumulated_losses))
        loss /= num_steps
        if parallel_dims.dp_cp_enabled:
            global_avg_loss = dist_utils.dist_mean(
                loss, parallel_dims.get_optional_mesh("loss")
            )
        else:
            global_avg_loss = loss.item()

        self.metrics_processor.log_validation(loss=global_avg_loss, step=step)

        # Set model back to train mode
        model.train()

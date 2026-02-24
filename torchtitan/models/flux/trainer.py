# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
from typing import Iterable

import torch

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed import utils as dist_utils
from torchtitan.models.flux.configs import Encoder, Inference, Validation
from torchtitan.models.flux.model.autoencoder import load_ae
from torchtitan.models.flux.model.hf_embedder import FluxEmbedder
from torchtitan.models.flux.parallelize import parallelize_encoders
from torchtitan.models.flux.utils import (
    create_position_encoding_for_latents,
    pack_latents,
    preprocess_data,
)
from torchtitan.trainer import Trainer


class FluxTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        encoder: Encoder = field(default_factory=Encoder)
        validation: Validation = field(default_factory=Validation)
        inference: Inference = field(default_factory=Inference)

    def __init__(self, config: Config):
        super().__init__(config)

        # Set random seed, and maybe enable deterministic mode
        # (mainly for debugging, expect perf loss).
        # For Flux model, we need distinct seed across FSDP ranks to ensure we randomly dropout prompts info in dataloader
        dist_utils.set_determinism(
            self.parallel_dims,
            self.device,
            config.debug,
            distinct_seed_mesh_dims=["fsdp", "dp_replicate"],
        )

        # NOTE: self._dtype is the data type used for encoders (image encoder, T5 text encoder, CLIP text encoder).
        # We cast the encoders and it's input/output to this dtype.  If FSDP with mixed precision training is not used,
        # the dtype for encoders is torch.float32 (default dtype for Flux Model).
        # Otherwise, we use the same dtype as mixed precision training process.
        self._dtype = (
            TORCH_DTYPE_MAP[config.training.mixed_precision_param]
            if self.parallel_dims.dp_shard_enabled
            else torch.float32
        )

        # load components
        assert config.model_spec is not None
        model_args = config.model_spec.model

        self.autoencoder = load_ae(
            # pyrefly: ignore [missing-attribute]
            config.encoder.autoencoder_path,
            # pyrefly: ignore [missing-attribute]
            model_args.autoencoder_params,
            device=self.device,
            dtype=self._dtype,
            # pyrefly: ignore [missing-attribute]
            random_init=config.encoder.test_mode,
        )

        self.clip_encoder = FluxEmbedder(
            # pyrefly: ignore [missing-attribute]
            version=config.encoder.clip_encoder,
            # pyrefly: ignore [missing-attribute]
            random_init=config.encoder.test_mode,
        ).to(device=self.device, dtype=self._dtype)
        self.t5_encoder = FluxEmbedder(
            # pyrefly: ignore [missing-attribute]
            version=config.encoder.t5_encoder,
            # pyrefly: ignore [missing-attribute]
            random_init=config.encoder.test_mode,
        ).to(device=self.device, dtype=self._dtype)

        # Apply FSDP to the T5 model / CLIP model
        # pyrefly: ignore [bad-assignment]
        self.t5_encoder, self.clip_encoder = parallelize_encoders(
            t5_model=self.t5_encoder,
            clip_model=self.clip_encoder,
            parallel_dims=self.parallel_dims,
            training=config.training,
        )

        if config.validator.enable:
            # pyrefly: ignore [missing-attribute]
            self.validator.flux_init(
                device=self.device,
                _dtype=self._dtype,
                autoencoder=self.autoencoder,
                t5_encoder=self.t5_encoder,
                clip_encoder=self.clip_encoder,
                trainer_config=config,
            )

    def forward_backward_step(
        self,
        *,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor,
        global_valid_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Perform a single forward and backward pass through the model.

        Args:
            input_dict: Dictionary containing input data including prompts and other metadata
            labels: Target tensor containing the ground truth image data
            global_valid_tokens: Optional tensor tracking the total number of valid tokens across all processes.
                This field is a placeholder for now as we rescale the loss within forward_backward_step for FLUX.

        Returns:
            torch.Tensor: The computed loss value for this training step
        """

        assert (
            global_valid_tokens is None
        ), "FLUX model don't need to rescale loss by number of global valid tokens"

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

        # rewrite the global_valid_tokens because the `labels` are reset after image encoder.
        local_valid_tokens = torch.tensor(
            labels.numel(), dtype=torch.float32, device=self.device
        )

        if self.parallel_dims.dp_enabled:
            batch_mesh = self.parallel_dims.get_mesh("batch")
            # pyrefly: ignore [bad-assignment]
            global_valid_tokens = dist_utils.dist_sum(local_valid_tokens, batch_mesh)
        else:
            global_valid_tokens = local_valid_tokens.float()

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

        # Apply CP sharding if enabled
        if self.parallel_dims.cp_enabled:
            from torchtitan.distributed.context_parallel import cp_shard

            (
                latents,
                latent_pos_enc,
                t5_encodings,
                text_pos_enc,
                target,
            ), _ = cp_shard(
                self.parallel_dims.get_mesh("cp"),
                (latents, latent_pos_enc, t5_encodings, text_pos_enc, target),
                None,  # No attention masks for Flux
                load_balancer_type=None,
            )

        with self.train_context():
            with self.maybe_enable_amp:
                latent_noise_pred = model(
                    img=latents,
                    img_ids=latent_pos_enc,
                    txt=t5_encodings,
                    txt_ids=text_pos_enc,
                    y=clip_encodings,
                    timesteps=timesteps,
                )

                # Scale loss as we used SUM reduction for mse loss function
                # pyrefly: ignore [unsupported-operation]
                loss = self.loss_fn(latent_noise_pred, target) / global_valid_tokens
            # latent_noise_pred.shape=(bs, seq_len, vocab_size)
            # need to free to before bwd to avoid peaking memory
            # pyrefly: ignore[unsupported-delete]
            del (latent_noise_pred, noise, target)
            loss.backward()

        return loss

    def train_step(
        self, data_iterator: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):
        self.optimizers.zero_grad()
        # Save the current step learning rate for logging
        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        parallel_dims = self.parallel_dims

        if self.gradient_accumulation_steps > 1:
            raise ValueError("FLUX doesn't support gradient accumulation for now.")

        # pyrefly: ignore [no-matching-overload]
        input_dict, labels = next(data_iterator)

        loss = self.forward_backward_step(input_dict=input_dict, labels=labels)

        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.config.training.max_norm,
            foreach=True,
            pp_mesh=parallel_dims.get_optional_mesh("pp"),
            ep_enabled=parallel_dims.ep_enabled,
        )
        self.checkpointer.maybe_wait_for_staging()
        self.optimizers.step()
        self.lr_schedulers.step()

        # log metrics
        if not self.metrics_processor.should_log(self.step):
            return

        if parallel_dims.dp_cp_enabled:
            loss = loss.detach()
            loss_mesh = parallel_dims.get_optional_mesh("loss")

            # NOTE: the loss returned by train
            global_avg_loss, global_max_loss, global_ntokens_seen = (
                dist_utils.dist_sum(loss, loss_mesh),
                dist_utils.dist_max(loss, loss_mesh),
                dist_utils.dist_sum(
                    torch.tensor(
                        self.ntokens_seen, dtype=torch.int64, device=self.device
                    ),
                    loss_mesh,
                ),
            )
        else:
            global_avg_loss = global_max_loss = loss.detach().item()
            global_ntokens_seen = self.ntokens_seen

        extra_metrics = {
            "n_tokens_seen": global_ntokens_seen,
            "lr": lr,
        }
        self.metrics_processor.log(
            self.step,
            global_avg_loss,
            global_max_loss,
            grad_norm.item(),
            extra_metrics=extra_metrics,
        )

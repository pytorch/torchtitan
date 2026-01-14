# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Iterable

import torch

from torchtitan.components.dataloader import DataloaderExhaustedError
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
from torchtitan.tools import utils
from torchtitan.train import main, Trainer


class FluxTrainer(Trainer):
    def __init__(self, job_config: JobConfig):
        super().__init__(job_config)

        # Set random seed, and maybe enable deterministic mode
        # (mainly for debugging, expect perf loss).
        # For Flux model, we need distinct seed across FSDP ranks to ensure we randomly dropout prompts info in dataloader
        dist_utils.set_determinism(
            self.parallel_dims,
            self.device,
            job_config.debug,
            distinct_seed_mesh_dims=["fsdp", "dp_replicate"],
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
            # pyrefly: ignore [missing-attribute]
            job_config.encoder.autoencoder_path,
            # pyrefly: ignore [missing-attribute]
            model_args.autoencoder_params,
            device=self.device,
            dtype=self._dtype,
            # pyrefly: ignore [missing-attribute]
            random_init=job_config.training.test_mode,
        )

        self.clip_encoder = FluxEmbedder(
            # pyrefly: ignore [missing-attribute]
            version=job_config.encoder.clip_encoder,
            # pyrefly: ignore [missing-attribute]
            random_init=job_config.training.test_mode,
        ).to(device=self.device, dtype=self._dtype)
        self.t5_encoder = FluxEmbedder(
            # pyrefly: ignore [missing-attribute]
            version=job_config.encoder.t5_encoder,
            # pyrefly: ignore [missing-attribute]
            random_init=job_config.training.test_mode,
        ).to(device=self.device, dtype=self._dtype)

        # Apply FSDP to the T5 model / CLIP model
        # pyrefly: ignore [bad-assignment]
        self.t5_encoder, self.clip_encoder = parallelize_encoders(
            t5_model=self.t5_encoder,
            clip_model=self.clip_encoder,
            parallel_dims=self.parallel_dims,
            job_config=job_config,
        )

        if job_config.validation.enable:
            # pyrefly: ignore [missing-attribute]
            self.validator.flux_init(
                device=self.device,
                _dtype=self._dtype,
                autoencoder=self.autoencoder,
                t5_encoder=self.t5_encoder,
                clip_encoder=self.clip_encoder,
            )

    # pyrefly: ignore [bad-override]
    def batch_generator(
        self, data_iterable: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ) -> Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        """FluxTrainer uses a simple batch generator without prefetching."""
        device_type = utils.device_type
        data_iterator = iter(data_iterable)

        while True:
            try:
                batch = next(data_iterator)
            except StopIteration as ex:
                raise DataloaderExhaustedError() from ex
            input_dict, labels = batch
            ntokens_batch = labels.numel()
            self.ntokens_seen += ntokens_batch
            self.metrics_processor.ntokens_since_last_log += ntokens_batch

            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor):
                    input_dict[k] = v.to(device_type)
            labels = labels.to(device_type)

            yield input_dict, labels

    # pyrefly: ignore [bad-override]
    def train_step(
        self, data_iterator: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ):
        self.optimizers.zero_grad()
        lr = self.lr_schedulers.schedulers[0].get_last_lr()[0]

        parallel_dims = self.parallel_dims

        accumulated_losses = []
        for _microbatch in range(self.gradient_accumulation_steps):
            # pyrefly: ignore [no-matching-overload]
            input_dict, labels = next(data_iterator)
            loss = self.forward_backward_step(input_dict, labels)
            accumulated_losses.append(loss.detach())

        grad_norm = dist_utils.clip_grad_norm_(
            [p for m in self.model_parts for p in m.parameters()],
            self.job_config.training.max_norm,
            foreach=True,
            pp_mesh=parallel_dims.get_optional_mesh("pp"),
            ep_enabled=parallel_dims.ep_enabled,
        )
        self.checkpointer.maybe_wait_for_staging()
        self.optimizers.step()
        self.lr_schedulers.step()

        loss = torch.sum(torch.stack(accumulated_losses))

        if not self.metrics_processor.should_log(self.step):
            return

        if parallel_dims.dp_cp_enabled:
            loss = loss.detach()
            ft_pg = self.ft_manager.loss_sync_pg
            loss_mesh = parallel_dims.get_optional_mesh("loss")
            global_avg_loss, global_max_loss, global_ntokens_seen = (
                dist_utils.dist_mean(loss, loss_mesh, ft_pg),
                dist_utils.dist_max(loss, loss_mesh, ft_pg),
                dist_utils.dist_sum(
                    torch.tensor(
                        self.ntokens_seen, dtype=torch.int64, device=self.device
                    ),
                    loss_mesh,
                    ft_pg,
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

    # pyrefly: ignore [bad-param-name-override]
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

        optional_context_parallel_ctx = None
        if self.parallel_dims.cp_enabled:
            cp_mesh = self.parallel_dims.get_mesh("cp")
            optional_context_parallel_ctx = dist_utils.create_context_parallel_ctx(
                cp_mesh=cp_mesh,
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
            # pyrefly: ignore [unsupported-delete]
            del (latent_noise_pred, noise, target)
            loss.backward()

        return loss


if __name__ == "__main__":
    main(FluxTrainer)

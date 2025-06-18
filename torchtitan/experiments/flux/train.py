# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import os
import signal
import time
from datetime import timedelta
from typing import Iterable, Optional

import numpy as np
import torch
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.fsdp import FSDPModule

import torchtitan.components.ft as ft

from torchtitan.config_manager import ConfigManager, JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import utils as dist_utils
from torchtitan.experiments.flux.dataset.tokenizer import build_flux_tokenizer
from torchtitan.experiments.flux.mlperf_logging import MLPerfLogger
from torchtitan.experiments.flux.model.autoencoder import load_ae
from torchtitan.experiments.flux.model.autoencoder_utils import (
    generate_unscaled_latent_from_mean_logvar,
)
from torchtitan.experiments.flux.model.hf_embedder import FluxEmbedder
from torchtitan.experiments.flux.parallelize_flux import parallelize_encoders
from torchtitan.experiments.flux.sampling import (
    generate_and_save_images,
    generate_empty_batch,
)
from torchtitan.experiments.flux.utils import (
    create_position_encoding_for_latents,
    pack_latents,
    preprocess_data,
    unpack_latents,
)
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.profiling import (
    maybe_enable_memory_snapshot,
    maybe_enable_profiling,
)
from torchtitan.train import Trainer


class FluxTrainer(Trainer):
    def __init__(self, job_config: JobConfig):
        super().__init__(job_config)
        if job_config.metrics.mlperf_logging:
            self.mlperf_logger = MLPerfLogger(
                filename=os.path.join(
                    job_config.job.dump_folder, "mlperf_compliance.log"
                ),
                root_dir=os.path.normpath(os.path.dirname(os.path.realpath(__file__))),
                log_every_n_steps=job_config.metrics.log_freq,
            )
        else:
            self.mlperf_logger = None
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

        model_config = self.train_spec.config[job_config.model.flavor]

        self.val_dataloader = (
            self.train_spec.build_val_dataloader_fn(
                dp_world_size=self.dataloader.dp_world_size,
                dp_rank=self.dataloader.dp_rank,
                tokenizer=None,
                job_config=job_config,
                infinite=False,
            )
            if job_config.eval.dataset
            else None
        )

        if job_config.encoder.autoencoder_path:
            self.autoencoder = load_ae(
                job_config.encoder.autoencoder_path,
                model_config.autoencoder_params,
                device=self.device,
                dtype=self._dtype,
                random_init=job_config.training.test_mode,
            )
        else:
            self.autoencoder = None

        if (
            self.job_config.encoder.autoencoder_shift
            and self.job_config.encoder.autoencoder_scale
        ):
            self.autoencoder_shift_factor = self.job_config.encoder.autoencoder_shift
            self.autoencoder_scale_factor = self.job_config.encoder.autoencoder_scale
        else:
            if self.autoencoder is None:
                raise ValueError(
                    "Autoencoder shift and scale must be provided if autoencoder is not loaded"
                )
            else:
                self.autoencoder_shift_factor = self.autoencoder.shift_factor
                self.autoencoder_scale_factor = self.autoencoder.scale_factor

        if job_config.encoder.clip_encoder:
            self.clip_encoder = FluxEmbedder(
                version=job_config.encoder.clip_encoder,
                random_init=job_config.training.test_mode,
            ).to(device=self.device, dtype=self._dtype)
        else:
            self.clip_encoder = None

        if job_config.encoder.t5_encoder:
            self.t5_encoder = FluxEmbedder(
                version=job_config.encoder.t5_encoder,
                random_init=job_config.training.test_mode,
            ).to(device=self.device, dtype=self._dtype)
        else:
            self.t5_encoder = None

        # Apply FSDP to the T5 model / CLIP model
        self.t5_encoder, self.clip_encoder = parallelize_encoders(
            t5_model=self.t5_encoder,
            clip_model=self.clip_encoder,
            world_mesh=self.world_mesh,
            parallel_dims=self.parallel_dims,
            job_config=job_config,
        )

        if self.job_config.encoder.empty_encodings_path:
            self.empty_t5_encodings = torch.from_numpy(
                np.load(self.job_config.encoder.empty_encodings_path + "/t5_empty.npy")
            ).to(device=self.device, dtype=self._dtype)[0]
            self.empty_clip_encodings = torch.from_numpy(
                np.load(
                    self.job_config.encoder.empty_encodings_path + "/clip_empty.npy"
                )
            ).to(device=self.device, dtype=self._dtype)[0]
        else:
            t5_tokenizer, clip_tokenizer = build_flux_tokenizer(self.job_config)
            empty_encodings = generate_empty_batch(
                num_images=1,
                device=self.device,
                dtype=self._dtype,
                clip_tokenizer=clip_tokenizer,
                t5_tokenizer=t5_tokenizer,
                clip_encoder=self.clip_encoder,
                t5_encoder=self.t5_encoder,
            )
            self.empty_t5_encodings = empty_encodings["t5_encodings"][0]
            self.empty_clip_encodings = empty_encodings["clip_encodings"][0]

        self.cancelled_soon = False
        signal.signal(signal.SIGUSR1, self.cancelled_signal_handler)

    def cancelled_signal_handler(self, signum, frame):
        logger.info(
            "Received first stopping signal, saving checkpoint and exiting gracefully"
        )
        self.cancelled_soon = True

    def batch_generator(
        self, data_iterable: Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]
    ) -> Iterable[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        """Returns an iterator that processes batches from the data iterator.
        Modified from train as, in our case, labels is sometimes a tuple"""
        device_type = utils.device_type

        for batch in iter(data_iterable):
            data_load_start = time.perf_counter()
            input_dict, labels = batch
            self.metrics_processor.ntokens_since_last_log += (
                labels[0].numel() if isinstance(labels, list) else labels.numel()
            )
            self.metrics_processor.data_loading_times.append(
                time.perf_counter() - data_load_start
            )

            # Move tensors to the appropriate device
            for k, v in input_dict.items():
                if isinstance(v, torch.Tensor):
                    input_dict[k] = v.to(device_type)
            labels = (
                labels.to(device_type)
                if not isinstance(labels, list)
                else [label.to(device_type) for label in labels]
            )

            yield input_dict, labels

    def train_step(self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor):
        # generate t5 and clip embeddings
        if self.mlperf_logger:
            self.mlperf_logger.log_train_step_start(self.step)
        lrs = [scheduler.get_last_lr()[0] for scheduler in self.lr_schedulers]

        if isinstance(labels, list):  # if mean and logvar are provided
            mean, logvar = labels
            mean = mean.to(device=self.device, dtype=self._dtype)
            logvar = logvar.to(device=self.device, dtype=self._dtype)
            input_dict["t5_encodings"] = input_dict["t5_encodings"].to(
                device=self.device, dtype=self._dtype
            )
            input_dict["clip_encodings"] = input_dict["clip_encodings"].to(
                device=self.device, dtype=self._dtype
            )
            unscaled_latent = generate_unscaled_latent_from_mean_logvar(mean, logvar)
            labels = (
                unscaled_latent - self.autoencoder_shift_factor
            ) * self.autoencoder_scale_factor

            if "drop_encodings" in input_dict:
                input_dict["t5_encodings"][
                    input_dict["drop_encodings"]
                ] = self.empty_t5_encodings
                input_dict["clip_encodings"][
                    input_dict["drop_encodings"]
                ] = self.empty_clip_encodings
                input_dict.pop("drop_encodings")
        else:
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

        self.optimizers.zero_grad()

        # Keep these variables local to shorten the code as these are
        # the major variables that are used in the training loop.
        model_parts = self.model_parts
        assert len(self.model_parts) == 1
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
        if self.mlperf_logger:
            self.mlperf_logger.log_train_step_end(self.step, global_avg_loss, lrs[0])

        if self.metrics_processor.should_log(self.step):
            self.metrics_processor.log(
                self.step,
                global_avg_loss,
                global_max_loss,
                {f"hparams/lr_{i}": lr for i, lr in enumerate(lrs)},
            )

    def eval_step(
        self,
        input_dict: dict[str, torch.Tensor],
        labels: torch.Tensor | list[torch.Tensor],
        timesteps: torch.Tensor,
        save_imgs: bool = False,
    ):  # prompt: str = "A photo of a cat"):
        """
        Calculate the validation loss for the Flux model.

        This follows the original paper's evaluation protocol. For each sample, calculate the loss at 7 equally spaced
        values for t in [0, 1] (excluding 1) and average it. This will make each batch size 7x larger, which may require
        a different batch size.

        Returns: Average loss per timestep across all samples in the batch.
        """
        if isinstance(labels, list):  # if mean and logvar are provided
            mean, logvar = labels
            mean = mean.to(device=self.device, dtype=self._dtype)
            logvar = logvar.to(device=self.device, dtype=self._dtype)
            input_dict["t5_encodings"] = input_dict["t5_encodings"].to(
                device=self.device, dtype=self._dtype
            )
            input_dict["clip_encodings"] = input_dict["clip_encodings"].to(
                device=self.device, dtype=self._dtype
            )
            unscaled_latent = generate_unscaled_latent_from_mean_logvar(mean, logvar)
            labels = (
                unscaled_latent - self.autoencoder_shift_factor
            ) * self.autoencoder_scale_factor

            if "drop_encodings" in input_dict:
                input_dict["t5_encodings"][
                    input_dict["drop_encodings"]
                ] = self.empty_t5_encodings
                input_dict["clip_encodings"][
                    input_dict["drop_encodings"]
                ] = self.empty_clip_encodings
                input_dict.pop("drop_encodings")
        else:
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
        model_parts = self.model_parts
        assert len(self.model_parts) == 1
        model = model_parts[0]

        world_mesh = self.world_mesh
        parallel_dims = self.parallel_dims

        # image in latent space transformed by self.auto_encoder
        clip_encodings = input_dict["clip_encodings"]
        t5_encodings = input_dict["t5_encodings"]

        bsz = labels.shape[0]

        with torch.no_grad():
            noise = torch.randn_like(labels)
            timestep_values = (timesteps / 8.0).to(labels)
            sigmas = timestep_values.view(-1, 1, 1, 1)
            latents = (1 - sigmas) * labels + sigmas * noise

            bsz, _, latent_height, latent_width = latents.shape

            POSITION_DIM = 3  # constant for Flux flow model
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
                timesteps=timestep_values.to(latents),
            )

            # Convert sequence of patches to latent shape
            pred = unpack_latents(latent_noise_pred, latent_height, latent_width)
            target = noise - labels
            loss = self.loss_fn(pred, target, reduction="none")

            # Clean up large intermediate tensors immediately
            del pred, noise, target, latent_noise_pred, latents

            # average the loss across timesteps
            # might be useful to report this in the future, but currently not mechanism in torchtitan
            # for distributed averaging with numel > 1
            # loss_per_timestep = loss.view(7, -1).mean(dim=1)

            # Initialize a tensor to accumulate losses for each timestep (0-7)
            loss_per_timestep = torch.zeros(8, device=loss.device)
            # Reshape loss to have one value per sample
            loss_per_sample = loss.mean(dim=(1, 2, 3))

            # Get integer timestep values from the timestep_values
            timestep_indices = timesteps.long()

            # Use scatter_add_ for vectorized accumulation of losses by timestep
            loss_per_timestep.scatter_add_(0, timestep_indices, loss_per_sample)

            # Count samples per timestep for averaging (using bincount)
            timestep_counts = torch.bincount(timestep_indices, minlength=8)

            # Avoid division by zero
            timestep_counts = torch.maximum(
                timestep_counts, torch.ones_like(timestep_counts)
            )

            if (
                parallel_dims.dp_replicate_enabled
                or parallel_dims.dp_shard_enabled
                or parallel_dims.cp_enabled
            ):
                # Collect loss sums and counts from all devices
                ft_pg = (
                    self.ft_manager.replicate_pg if self.ft_manager.enabled else None
                )
                # Use the new dist_collect function to gather tensors across devices
                global_loss_per_timestep = dist_utils.dist_collect(
                    loss_per_timestep, world_mesh["dp_cp"], ft_pg
                )
                global_timestep_counts = dist_utils.dist_collect(
                    timestep_counts, world_mesh["dp_cp"], ft_pg
                )

            else:
                # For single device, just calculate locally
                global_loss_per_timestep = loss_per_timestep
                global_timestep_counts = timestep_counts

        # if encoders are not loaded we cannot save images
        if save_imgs:
            if not self.t5_encoder or not self.clip_encoder:
                logger.warning("Encoders are not loaded, cannot save images")
                return global_loss_per_timestep, global_timestep_counts

            t5_tokenizer, clip_tokenizer = build_flux_tokenizer(self.job_config)
            generate_and_save_images(
                input_dict,
                clip_tokenizer,
                t5_tokenizer,
                self.clip_encoder,
                self.t5_encoder,
                self.model_parts[0],
                self.autoencoder,
                self.job_config.training.img_size,
                self.step,
                save_img_folder=os.path.join(
                    self.job_config.job.dump_folder,
                    self.job_config.eval.save_img_folder,
                ),
                dtype=self._dtype,
            )

        # In the future, we could return avg_loss_per_timestep for more detailed reporting
        return global_loss_per_timestep, global_timestep_counts

    def train_success(self, eval_loss: float):
        return (
            eval_loss is not None
            and self.job_config.eval.target_eval_loss is not None
            and eval_loss <= self.job_config.eval.target_eval_loss
        )

    @record
    def train(self):
        job_config = self.job_config
        eval_loss = None
        gbs = job_config.training.batch_size * self.dataloader.dp_world_size
        eval_freq_steps = job_config.eval.eval_freq // gbs
        if self.mlperf_logger:
            self.mlperf_logger.log_run_start(
                gbs=gbs,
                seed=self.job_config.training.seed,
                lr=self.job_config.optimizer.lr,
                warmup_steps=self.job_config.lr_scheduler.warmup_steps,
                gradient_clip_norm=self.job_config.training.max_norm,
                optimizer_config=self.optimizers.optimizers[0].param_groups[0],
            )

        self.checkpointer.load(step=job_config.checkpoint.load_step)
        logger.info(f"Training starts at step {self.step + 1}.")

        with (
            maybe_enable_profiling(job_config, global_step=self.step) as torch_profiler,
            maybe_enable_memory_snapshot(
                job_config, global_step=self.step
            ) as memory_profiler,
            ft.maybe_semi_sync_training(
                job_config,
                ft_manager=self.ft_manager,
                model=self.model_parts[0],
                optimizer=self.optimizers,
                sync_every=job_config.fault_tolerance.sync_steps,
            ),
        ):
            for iteration, (inputs, labels) in enumerate(
                self.batch_generator(self.dataloader)
            ):
                if self.step >= job_config.training.steps:
                    break
                eval_loss = None
                self.step += 1
                if iteration == 0 and self.mlperf_logger:
                    self.mlperf_logger.log_train_start()
                self.gc_handler.run(self.step)
                self.train_step(inputs, labels)

                if self.step % eval_freq_steps == 0 and self.val_dataloader is not None:
                    eval_loss = self.eval()

                self.checkpointer.save(
                    self.step,
                    force=(
                        self.step == job_config.training.steps or self.cancelled_soon
                    ),
                )

                if self.train_success(eval_loss):
                    break

                # signal the profiler that the next profiling step has started
                if torch_profiler:
                    torch_profiler.step()
                if memory_profiler:
                    memory_profiler.step()

                # reduce timeout after first train step for faster signal
                # (assuming lazy init and compilation are finished)
                if self.step == 1:
                    dist_utils.set_pg_timeouts(
                        timeout=timedelta(
                            seconds=job_config.comm.train_timeout_seconds
                        ),
                        world_mesh=self.world_mesh,
                    )

                if self.cancelled_soon:
                    self.cancelled_soon = False
                    break

        if torch.distributed.get_rank() == 0:
            logger.info("Sleeping 2 seconds for other ranks to complete")
            time.sleep(2)

        self.metrics_processor.close()
        logger.info("Training completed")
        if self.mlperf_logger:
            self.mlperf_logger.log_train_end(
                success=eval_loss is not None
                and eval_loss <= job_config.eval.target_eval_loss
            )

    def eval(self) -> float:
        def generate_val_timesteps(cur_val_timestep, samples):
            """
            Generate timesteps for validation set

            This is a helper function to generate timesteps 0 through 7, repeating as necessary.
            """
            first_offset = torch.arange(cur_val_timestep, 8, device=self.device)[
                :samples
            ]
            samples_left = samples - first_offset.numel()
            val_timesteps = torch.arange(
                0, 8, dtype=torch.int8, device=self.device
            ).repeat_interleave(math.ceil(samples_left / 8))[:samples_left]
            val_timesteps = torch.cat([first_offset, val_timesteps])
            cur_val_timestep = (val_timesteps[-1].item() + 1) % 8
            return val_timesteps, cur_val_timestep

        if self.mlperf_logger:
            self.mlperf_logger.log_eval_start(self.step)
        model = self.model_parts[0]
        model.eval()

        # Follow procedure set out in Flux paper of stratified timestep sampling
        cur_val_timestep = 0
        eval_step = 0
        eval_samples = 0
        sum_loss_per_timestep = torch.zeros(8, device=self.device)
        sum_timestep_counts = torch.zeros(8, device=self.device)
        # Iterate through all validation batches
        # TODO: not sure how to handle profiling with validation
        for val_inputs, val_labels in self.batch_generator(self.val_dataloader):
            eval_step += 1
            samples = len(val_labels[0] if isinstance(val_labels, list) else val_labels)
            val_timesteps, cur_val_timestep = generate_val_timesteps(
                cur_val_timestep, samples
            )
            loss, counts = self.eval_step(
                val_inputs,
                val_labels,
                val_timesteps,
                save_imgs=eval_step == 1 and self.job_config.eval.save_img_folder,
            )
            eval_samples += samples
            sum_loss_per_timestep += loss
            sum_timestep_counts += counts

        # Different batches and timestepsmay have different number of samples, so we need to average the loss like this
        # rather than taking the mean of the mean batch losses.
        timestep_counts_proportions = sum_timestep_counts / sum_timestep_counts.sum()
        avg_loss_per_timestep = sum_loss_per_timestep / sum_timestep_counts
        avg_loss = (avg_loss_per_timestep * timestep_counts_proportions).sum()
        self.metrics_processor.val_log(self.step, avg_loss)
        model.train()

        # Reshard after run forward pass in eval_step.
        # This is to ensure the model weights are sharded the same way for checkpoint saving.
        for module in self.model_parts[0].modules():
            if isinstance(module, FSDPModule):
                module.reshard()
        if self.mlperf_logger:
            self.mlperf_logger.log_eval_end(self.step, avg_loss.item())

        return avg_loss.item()


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

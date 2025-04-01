import importlib
import os
import time
from datetime import timedelta
from typing import Any, Generator, Iterable, Optional, overload

import torch

import torchtitan.components.ft as ft
import torchtitan.protocols.train_spec as train_spec_module

from torch.distributed.elastic.multiprocessing.errors import record
from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.metrics import (
    build_metrics_processor,
    ensure_pp_loss_visible,
)

from torchtitan.config_manager import JobConfig
from torchtitan.distributed import ParallelDims, utils as dist_utils
from torchtitan.experiments.flux.model.model_builder import load_ae
from torchtitan.experiments.flux.model.modules.hf_embedder import FluxEmbedder
from torchtitan.experiments.flux.utils import predict_noise, preprocess_flux_data
from torchtitan.protocols.model_converter import build_model_converters
from torchtitan.tools import utils
from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.profiling import (
    maybe_enable_memory_snapshot,
    maybe_enable_profiling,
)
from torchtitan.train import Trainer
from typing_extensions import override


class FluxTrainer(Trainer):
    def __init__(self, job_config: JobConfig):
        super().__init__(self, job_config=job_config)

        self.preprocess_fn = preprocess_flux_data
        # self.dtype = job_config.encoder.dtype
        self._dtype = torch.bfloat16
        self._seed = job_config.training.seed
        self._guidence = job_config.training.guidence

        # load components
        self.autoencoder = load_ae(
            job_config.encoder.ae_name, device=self.torch_device
        ).to(dtype=self.dtype)
        self.clip_encoder = FluxEmbedder(version=job_config.encoder.clip_encoder).to(
            self.torch_device, dtype=self._dtype
        )
        self.t5_encoder = FluxEmbedder(version=job_config.encoder.t5_encoder).to(
            self.torch_device, dtype=self._dtype
        )

    def next_batch(
        self, data_iterator: Iterable
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        data_load_start = time.perf_counter()
        batch = next(data_iterator)

        # generate t5 and clip
        batch = self.preprocess_fn(
            device=self.torch_device,
            dtype=self.dtype,
            autoencoder=self.autoencoder,
            clip_encoder=self.clip_encoder,
            t5_encoder=self.t5_encoder,
            batch=batch,
            offload=True,
        )

        labels = batch["image_encodings"]
        input_dict = batch
        self.metrics_processor.ntokens_since_last_log += labels.numel()
        self.metrics_processor.data_loading_times.append(
            time.perf_counter() - data_load_start
        )

        device_type = utils.device_type
        for k, v in input_dict.items():
            input_dict[k] = input_dict[k].to(device_type)
        labels = labels.to(device_type)

        return input_dict, labels

    def train_step(self, input_dict: dict[str, torch.Tensor], labels: torch.Tensor):
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
        pred = predict_noise(
            model_parts[0],
            noisy_latents,
            clip_encodings,
            t5_encodings,
            timesteps,
            guidance,
        )
        loss = self.train_spec.loss_fn(pred, target)
        # pred.shape=(bs, seq_len, vocab_size)
        # need to free to before bwd to avoid peaking memory
        del pred
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

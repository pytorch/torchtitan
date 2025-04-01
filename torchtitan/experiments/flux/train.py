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
from torchtitan.experiments.flux.utils import preprocess_flux_data
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
        self.dtype = torch.bfloat16

        # load components
        self.autoencoder = load_ae(
            job_config.encoder.ae_name, device=self.torch_device
        ).to(dtype=self.dtype)
        self.clip_encoder = FluxEmbedder(version=job_config.encoder.clip_encoder).to(
            self.torch_device, dtype=self.dtype
        )
        self.t5_encoder = FluxEmbedder(version=job_config.encoder.t5_encoder).to(
            self.torch_device, dtype=self.dtype
        )

    def next_batch(self, data_iterator: Iterable) -> tuple[torch.Tensor, torch.Tensor]:
        data_load_start = time.perf_counter()
        batch = next(data_iterator)
        self.preprocess_fn(
            device=self.torch_device,
            dtype=self.dtype,
            autoencoder=self.autoencoder,
            clip_encoder=self.clip_encoder,
            t5_encoder=self.t5_encoder,
            batch=batch,
            offload=True,
        )
        input_dict = batch
        self.metrics_processor.ntokens_since_last_log += labels.numel()
        self.metrics_processor.data_loading_times.append(
            time.perf_counter() - data_load_start
        )

        device_type = utils.device_type
        for k, v in input_dict.items():
            input_dict[k] = input_dict[k].to(device_type)
        input_ids = input_ids.to(device_type)
        labels = labels.to(device_type)
        return input_ids, labels


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

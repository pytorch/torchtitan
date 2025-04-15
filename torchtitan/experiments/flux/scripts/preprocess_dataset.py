from typing import Callable

import torch
import torchtitan.protocols.train_spec as train_spec_module
from torchtitan.config_manager import JobConfig
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.flux.model.autoencoder import load_ae

from torchtitan.experiments.flux.model.hf_embedder import FluxEmbedder
from torchtitan.experiments.flux.model.model import FluxModel
from torchtitan.experiments.flux.parallelize_flux import parallelize_encoders
from torchtitan.experiments.flux.train import FluxTrainer

from torchtitan.experiments.flux.utils import (
    create_position_encoding_for_latents,
    pack_latents,
    preprocess_flux_data,
    unpack_latents,
)
from torchtitan.tools.logging import init_logger, logger
from torchtitan.train import Trainer


class FluxPreprocess(FluxTrainer):
    def __init__(self, job_config: JobConfig):
        """
        Reuse the
        """
        super().__init__(job_config)
        self.preprocess_fn = preprocess_flux_data

    def preprocess(self):
        data_iterator = iter(self.dataloader)
        for batch in data_iterator:
            input_dict, labels = batch
            for k, _ in input_dict.items():
                input_dict[k] = input_dict[k].to(self.device)
            labels = labels.to(self.device)

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

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field, replace
from typing import Annotated

import torch
import tyro

from torchtitan.config import TORCH_DTYPE_MAP
from torchtitan.distributed import utils as dist_utils
from torchtitan.models.flux.configs import FluxEncoderConfig, Inference
from torchtitan.models.flux.model.autoencoder import load_ae
from torchtitan.models.flux.model.model import FluxModel
from torchtitan.models.flux.parallelize import parallelize_encoders
from torchtitan.models.flux.tokenizer import FluxTokenizerContainer
from torchtitan.models.flux.validate import FluxValidator
from torchtitan.trainer import Trainer


class FluxTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        # Overwrite parent class tokenizer
        tokenizer: FluxTokenizerContainer.Config = (  # pyrefly: ignore [bad-override]
            field(default_factory=FluxTokenizerContainer.Config)
        )
        validator: Annotated[  # pyrefly: ignore [bad-override]
            FluxValidator.Config | None, tyro.conf.AvoidSubcommands
        ] = None
        encoder: FluxEncoderConfig = field(default_factory=FluxEncoderConfig)
        """Configuration for Flux encoders (T5 text encoder, CLIP text encoder, and autoencoder)."""
        inference: Inference = field(default_factory=Inference)

    def __init__(self, config: Config):
        super().__init__(config)

        # Flux samples diffusion noise and timesteps during each model step, so
        # data-parallel ranks need distinct model RNG streams. Dataset
        # transformations such as prompt dropout use Grain's separate RNG.
        distinct_seed_mesh_dims = ["cp", "dp_shard", "dp_replicate"]
        dist_utils.set_determinism(
            self.engine.parallel_dims,
            self.engine.device,
            config.debug,
            distinct_seed_mesh_dims=distinct_seed_mesh_dims,
        )

        # NOTE: self._dtype is the data type used for encoders (image encoder, T5 text encoder, CLIP text encoder).
        # We cast the encoders and it's input/output to this dtype.  If FSDP with mixed precision training is not used,
        # the dtype for encoders is torch.float32 (default dtype for Flux Model).
        # Otherwise, we use the same dtype as mixed precision training process.
        self._dtype = (
            TORCH_DTYPE_MAP[config.training.mixed_precision_param]
            if self.engine.parallel_dims.dp_shard_enabled
            else torch.float32
        )

        # load components
        assert config.model_spec is not None
        model_args = config.model_spec.model
        assert isinstance(model_args, FluxModel.Config)

        self.autoencoder = load_ae(
            config.encoder.autoencoder_path,
            model_args.autoencoder,
            device=self.engine.device,
            dtype=self._dtype,
            random_init=config.encoder.random_init,
        )

        # Use the encoder configs from the model registry, overriding version
        # and random_init from the trainer encoder config if set.
        clip_encoder_config = model_args.clip_encoder
        t5_encoder_config = model_args.t5_encoder
        self.clip_encoder = (
            replace(
                clip_encoder_config,
                version=config.encoder.clip_encoder or clip_encoder_config.version,
                random_init=config.encoder.random_init,
            )
            .build()
            .to(device=self.engine.device, dtype=self._dtype)
        )

        self.t5_encoder = (
            replace(
                t5_encoder_config,
                version=config.encoder.t5_encoder or t5_encoder_config.version,
                random_init=config.encoder.random_init,
            )
            .build()
            .to(device=self.engine.device, dtype=self._dtype)
        )

        # Apply FSDP to the T5 model / CLIP model
        self.t5_encoder, self.clip_encoder = parallelize_encoders(
            t5_model=self.t5_encoder,
            clip_model=self.clip_encoder,
            parallel_dims=self.engine.parallel_dims,
            training=config.training,
            symm_mem_scope=config.parallelism.fsdp_symm_mem_scope,
        )
        self.engine.preprocess_inputs_kwargs = {
            "autoencoder": self.autoencoder,
            "clip_encoder": self.clip_encoder,
            "t5_encoder": self.t5_encoder,
            "dtype": self._dtype,
        }

        if config.validator is not None:
            # pyrefly: ignore [missing-attribute]
            self.validator.flux_init(
                device=self.engine.device,
                _dtype=self._dtype,
                autoencoder=self.autoencoder,
                t5_encoder=self.t5_encoder,
                clip_encoder=self.clip_encoder,
                dump_folder=config.dump_folder,
            )

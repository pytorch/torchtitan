# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Base wrapper for TorchTitan models to work with vLLM V1 engine.

This module provides TorchTitanVLLMModel: Core model class that adapts
TorchTitan models for vLLM.
"""

from functools import partial

import torch
import torch.nn as nn
from torch.distributed._tensor import DTensor, Replicate
from torch.distributed.checkpoint.state_dict import (
    set_model_state_dict,
    StateDictOptions,
)

from torchtitan.experiments.rl.unified.models.utils import replace_with_vllm_attention
from torch.distributed.tensor import DTensor, Replicate
from torchtitan.models.qwen3.model.model import precompute_rope_cache
from torchtitan.protocols.model import BaseModelArgs, ModelProtocol
from torchtitan.protocols.state_dict_adapter import BaseStateDictAdapter
from torchtitan.protocols.train_spec import ParallelizeFunction

from vllm.config import VllmConfig
from vllm.logger import init_logger

from .parallelism_utils import create_parallel_dims_from_vllm_config


logger = init_logger(__name__)


class TorchTitanVLLMModelWrapper(nn.Module):
    """
    Generic vLLM-compatible model wrapper for TorchTitan models. Implemented
    required interface required by vLLM Engine.

    The wrapper handles:
    - Direct usage of TorchTitan model args (no HF config mapping needed)
    - Attention replacement with vLLM paged attention
    - Parallelism setup and DTensor conversion between torchtitan and vLLM
    - Weight loading from HF checkpoints
    - vLLM forward/compute_logits interface
    """

    is_text_generation_model = True  # Required for vLLM runner validation
    supports_pp = False  # Pipeline parallelism not supported yet
    supports_multimodal = False

    def __init__(
        self,
        *,
        model_cls: type[ModelProtocol],
        model_args: BaseModelArgs,
        state_dict_adapter: type[BaseStateDictAdapter],
        parallelize_fn: ParallelizeFunction,
        vllm_config: VllmConfig,
        prefix: str = "",
    ):
        super().__init__()

        assert vllm_config is not None, "vllm_config is required"

        # Store components
        self.model_cls = model_cls
        self.state_dict_adapter = state_dict_adapter
        self.parallelize_fn = parallelize_fn

        # Use TorchTitan model args directly (no HF config mapping)
        self.config = model_args
        logger.info(f"Creating {self.model_cls.__name__} with config: {model_args}")
        self.model = self.model_cls(model_args)

        # Setup RoPE cache extension function if provided
        self.rope_cache_extension_fn = partial(
            precompute_rope_cache,
            dim=self.config.head_dim,
            base=self.config.rope_theta,
        )

        # Replace attention with vLLM's attention
        replace_with_vllm_attention(self.model)

        # Create ParallelDims from vLLM config and apply parallelization
        # NOTE: We need to apply parallelize within model.__init__ because w
        parallel_dims = create_parallel_dims_from_vllm_config(vllm_config)
        if parallel_dims.tp_enabled:
            self.world_mesh = parallel_dims.world_mesh
            tp_mesh = self.world_mesh["tp"]
            parallelize_fn(
                model=self.model,
                tp_mesh=tp_mesh,
                loss_parallel=False,
                enable_float8_tensorwise_tp=False,
                enable_async_tp=False,
            )
            logger.info(
                f"Successfully initialized model with with TP={parallel_dims.tp}"
            )
        else:
            logger.info("Single GPU mode - no parallelization needed")
=======

        # Create ParallelDims and JobConfig from vLLM config at runtime
        # vLLM config contains the tensor_parallel_size from command-line args
        # and this will be consistent across all worker processes
        from torchtitan.experiments.rl.unified.infra.parallelism_utils import (
            create_job_config_from_vllm_config,
            create_parallel_dims_from_vllm_config,
        )

        self.parallel_dims = create_parallel_dims_from_vllm_config(vllm_config)
        self.parallel_config = create_job_config_from_vllm_config(
            vllm_config=vllm_config,
        )
        # Replace attention with vLLM paged attention
        tp_size = self.parallel_dims.tp
        if tp_size > 1:
            assert (
                model_args.n_heads % tp_size == 0
            ), "Only support when n_heads can be divided by tp_size"
        replace_with_vllm_attention(self.model, tp_degree=tp_size)

        # NOTE: We need to apply parallelize within model.__init__ because vllm
        # doesn't separate model creation and parallelism application and instead
        # requires parallelization to be done inside model constructor.
        self.model = parallelize_fn(
            model=self.model,
            parallel_dims=self.parallel_dims,
            job_config=self.parallel_config,
        )

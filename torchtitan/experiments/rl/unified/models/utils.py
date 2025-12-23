# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

<<<<<<< HEAD
<<<<<<< HEAD
=======
"""
Parallelization utilities for vLLM + TorchTitan models.

DEPRECATED: This module contains legacy functions that convert vLLM config -> TorchTitan config.
This is the WRONG direction!

RECOMMENDED: Use TorchTitanConfigAdapter from config_adapter.py instead, which converts
TorchTitan config -> vLLM config (making TorchTitan the single source of truth).

Legacy functions in this module are kept for backward compatibility but will be removed
in a future version.
"""

import warnings

import torch.distributed as dist

from torchtitan.config.job_config import JobConfig, Model, Parallelism, Training
from torchtitan.distributed.parallel_dims import ParallelDims
from vllm.config import VllmConfig
from vllm.logger import init_logger


logger = init_logger(__name__)


def create_parallel_dims_from_vllm_config(vllm_config: VllmConfig) -> ParallelDims:
    """
    DEPRECATED: Create ParallelDims from vLLM config.

    WARNING: This function converts vLLM config -> TorchTitan ParallelDims,
    which makes vLLM the source of truth. This is the WRONG direction!

    RECOMMENDED: Use TorchTitanConfigAdapter.to_parallel_dims() instead, which
    converts TorchTitan model args -> ParallelDims (making TorchTitan the
    single source of truth).

    Args:
        vllm_config: vLLM configuration object

    Returns:
        ParallelDims object with parallelism settings validated

    Note:
        vLLM doesn't use FSDP sharding (dp_shard=1) or expert parallelism (ep=1, etp=1)
        in inference. These are set to default values.
    """
    warnings.warn(
        "create_parallel_dims_from_vllm_config() is deprecated. "
        "Use TorchTitanConfigAdapter.to_parallel_dims() instead to make "
        "TorchTitan config the single source of truth.",
        DeprecationWarning,
        stacklevel=2,
    )
    world_size = dist.get_world_size()

    # Map vLLM config to TorchTitan ParallelDims
    parallel_dims = ParallelDims(
        dp_replicate=vllm_config.parallel_config.data_parallel_size,
        dp_shard=1,  # vLLM doesn't use FSDP sharding
        cp=vllm_config.parallel_config.decode_context_parallel_size,
        tp=vllm_config.parallel_config.tensor_parallel_size,
        pp=vllm_config.parallel_config.pipeline_parallel_size,
        ep=1,  # Expert parallelism not used in vLLM inference yet
        etp=1,  # Expert tensor parallelism not used in vLLM inference yet
        world_size=world_size,
    )

    logger.info(
        f"Created ParallelDims from vLLM config: "
        f"DP={parallel_dims.dp_replicate}, TP={parallel_dims.tp}, "
        f"CP={parallel_dims.cp}, PP={parallel_dims.pp}"
    )

    return parallel_dims


def create_job_config_from_vllm_config(
    vllm_config: VllmConfig,
    model_name: str = "qwen3",
    hf_assets_path: str = "/path/to/hf/assets",
) -> JobConfig:
    """
    Create TorchTitan JobConfig from vLLM configuration.

    Args:
        vllm_config: vLLM configuration object containing model, parallel, and cache configs
        model_name: Model name to use (default: "qwen3")
        hf_assets_path: Path to HuggingFace assets directory (default: "/path/to/hf/assets")

    Returns:
        JobConfig object with settings mapped from vLLM config
    """
    # Create JobConfig with defaults
    job_config = JobConfig()

    # --- Model Configuration ---
    model_config = vllm_config.model_config
    job_config.model = Model(
        name=model_name,
        hf_assets_path=hf_assets_path,
    )

    # --- Parallelism Configuration ---
    parallel_config = vllm_config.parallel_config
    job_config.parallelism = Parallelism(
        data_parallel_replicate_degree=parallel_config.data_parallel_size,
        data_parallel_shard_degree=1,  # vLLM doesn't use FSDP sharding in inference
        context_parallel_degree=parallel_config.decode_context_parallel_size,
        tensor_parallel_degree=parallel_config.tensor_parallel_size,
        pipeline_parallel_degree=parallel_config.pipeline_parallel_size,
        expert_parallel_degree=1,  # Not used in vLLM inference yet
        expert_tensor_parallel_degree=1,  # Not used in vLLM inference yet
    )

    # --- Training Configuration (minimal defaults for inference) ---
    # vLLM is primarily for inference, but we set reasonable defaults
    # in case the model is used for fine-tuning
    job_config.training = Training(
        local_batch_size=1,  # Inference typically processes one batch at a time
        steps=1,  # Single step for inference
    )

    return job_config


# ===== Merged from models/utils.py =====
=======
>>>>>>> e5d0a32d (clean up)

>>>>>>> 83047803 (refactor v3)
import logging
from enum import Enum

import torch
from safetensors.torch import load_file
<<<<<<< HEAD
<<<<<<< HEAD
from torchtitan.experiments.rl.unified.models.attention import VLLMAttention

from torchtitan.experiments.rl.vllm_compat.models.attention import (
    VLLMCompatibleFlashAttention,
)
=======
=======
from torchtitan.config.job_config import JobConfig, Model, Parallelism, Training
>>>>>>> e5d0a32d (clean up)

from torchtitan.experiments.rl.unified.models.attention import VLLMAttention
>>>>>>> 83047803 (refactor v3)

from torchtitan.experiments.rl.vllm_compat.weights_vllm_compat import (
    torchtitan_to_vllm_compat,
)
from torchtitan.models.qwen3.model.args import Qwen3ModelArgs
from transformers import AutoConfig
from vllm.config import VllmConfig

<<<<<<< HEAD
logger = logging.getLogger(__name__)
=======

logging_logger = logging.getLogger(__name__)
>>>>>>> 83047803 (refactor v3)


class ModelMode(str, Enum):
    """
    Enum defining which TorchTitan model to use.

    Attributes:
        UNIFIED: Standard TorchTitan model replaced with vLLM attention for unified
            training and inference.
        VLLM_COMPAT: vLLM-compatible TorchTitan model using vLLM's batch invariant kernels,
            ensuring bitwise determinism between training and inference.
        STANDARD: Plain TorchTitan model without any modifications.
    """

    UNIFIED = "unified"
    VLLM_COMPAT = "vllm_compat"
    STANDARD = "standard"


def replace_with_vllm_attention(model):
    """
<<<<<<< HEAD
    Replace TorchTitan attention with vLLM's Attention.
=======
    Replace TorchTitan attention with vLLM paged attention.
>>>>>>> 83047803 (refactor v3)

    Assumes model has .layers dict with .attention.inner_attention structure.
    """
    if not hasattr(model, "layers"):
        raise AttributeError(
            f"Model {type(model).__name__} must have .layers attribute"
        )

    model_args = model.model_args
    for layer_name, layer in model.layers.items():
        if not hasattr(layer, "attention"):
            raise ValueError(f"Layer {layer_name} must have .attention attribute")

        vllm_attn = VLLMAttention(
            hidden_size=model_args.dim,
            num_heads=model_args.n_heads,
            num_kv_heads=model_args.n_heads,  # Use n_heads (already replicated)
            head_dim=model_args.head_dim,
            layer_name=layer_name,
            scale=model_args.head_dim**-0.5,
        )

        layer.attention.inner_attention = vllm_attn

<<<<<<< HEAD
    logger.info(
        f"Successfully replaced TorchTitan attention with VLLMAttention "
        f"({len(model.layers)} layers)"
    )


def replace_with_vllm_compatible_flash_attention(model):
    """
    Replace TorchTitan attention with vLLM compatible flash attention.

    Assumes model has .layers dict with .attention.inner_attention structure.
    """
    if not hasattr(model, "layers"):
        raise AttributeError(
            f"Model {type(model).__name__} must have .layers attribute"
        )

    model_args = model.model_args
    for layer_name, layer in model.layers.items():
        if not hasattr(layer, "attention"):
            raise ValueError(f"Layer {layer_name} must have .attention attribute")

        vllm_attn = VLLMCompatibleFlashAttention()

        layer.attention.inner_attention = vllm_attn

    logger.info(
=======
    logging_logger.info(
>>>>>>> 83047803 (refactor v3)
        f"Successfully replaced TorchTitan attention with VLLMAttention "
        f"({len(model.layers)} layers)"
    )


def load_model(
    checkpoint_path: str, model_path: str, model_mode: str = ModelMode.VLLM_COMPAT
):
    """
    Load TorchTitan model from checkpoint.

    Args:
        checkpoint_path: Path to TorchTitan checkpoint
        model_path: Path to HuggingFace model (for config)
        model_mode: Indicates which model to use. Train inferece unified model, batch invariant Torchtitan model,
            or plain Torchtitan model

    Returns:
        model: Loaded TorchTitan model
    """
    # Load HuggingFace config
    # TODO: do not depend on transformers.AutoConfig, use qwen_args directly
    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    # Create model args
    model_args = Qwen3ModelArgs(
        dim=hf_config.hidden_size,
        n_layers=hf_config.num_hidden_layers,
        n_heads=hf_config.num_attention_heads,
        n_kv_heads=hf_config.num_key_value_heads,
        vocab_size=hf_config.vocab_size,
        head_dim=getattr(
            hf_config,
            "head_dim",
            hf_config.hidden_size // hf_config.num_attention_heads,
        ),
        hidden_dim=hf_config.intermediate_size,
        norm_eps=hf_config.rms_norm_eps,
        rope_theta=hf_config.rope_theta,
        max_seq_len=getattr(hf_config, "max_position_embeddings", 32768),
        qk_norm=True,
        depth_init=True,
        eos_id=getattr(hf_config, "eos_token_id", 151645),
    )

    # state_dict is in standard TorchTitan format (w1, w2, w3)
    state_dict = load_file(checkpoint_path)

    if model_mode == ModelMode.UNIFIED:
        from torchtitan.models.qwen3 import Qwen3Model

        model = Qwen3Model(model_args)
        # Set global default dtype to bfloat16. This is needed because vLLM's Attention
        # layer uses torch.get_default_dtype() and it doesn't support float32
        torch.set_default_dtype(torch.bfloat16)
<<<<<<< HEAD
        replace_with_vllm_compatible_flash_attention(model)
=======
        replace_with_vllm_attention(model)
>>>>>>> 83047803 (refactor v3)
        # Load standard TorchTitan format directly
        model.load_state_dict(state_dict, strict=True)
    elif model_mode == ModelMode.VLLM_COMPAT:
        # Create and load model that has bitwise determinism between training and inference
        from torchtitan.experiments.rl.vllm_compat.models.qwen3 import (
            Qwen3VLLMCompatModel,
        )

        model = Qwen3VLLMCompatModel(model_args)
        # Convert to vLLM-compat format (merged gate_up_proj, down_proj)
        vllm_compat_state = torchtitan_to_vllm_compat(state_dict)
        model.load_state_dict(vllm_compat_state, strict=False)
    else:
        # Use standard TorchTitan model
        from torchtitan.models.qwen3 import Qwen3Model

        model = Qwen3Model(model_args)
        # Load standard TorchTitan format directly
        model.load_state_dict(state_dict, strict=False)

    model.to(torch.bfloat16)

    return model


def create_job_config_from_vllm_config(
    vllm_config: VllmConfig,
    model_name: str = "qwen3",
    hf_assets_path: str = "/path/to/hf/assets",
) -> JobConfig:
    """
    Create TorchTitan JobConfig from vLLM configuration.

    Args:
        vllm_config: vLLM configuration object containing model, parallel, and cache configs
        model_name: Model name to use (default: "qwen3")
        hf_assets_path: Path to HuggingFace assets directory (default: "/path/to/hf/assets")

    Returns:
        JobConfig object with settings mapped from vLLM config
    """
    # Create JobConfig with defaults
    job_config = JobConfig()

    model_config = vllm_config.model_config
    job_config.model = Model(
        name=model_name,
        hf_assets_path=hf_assets_path,
    )

    parallel_config = vllm_config.parallel_config
    job_config.parallelism = Parallelism(
        data_parallel_replicate_degree=parallel_config.data_parallel_size,
        data_parallel_shard_degree=1,  # vLLM doesn't use FSDP sharding in inference
        context_parallel_degree=parallel_config.decode_context_parallel_size,
        tensor_parallel_degree=parallel_config.tensor_parallel_size,
        pipeline_parallel_degree=parallel_config.pipeline_parallel_size,
        expert_parallel_degree=1,  # Not used in vLLM inference yet
        expert_tensor_parallel_degree=1,  # Not used in vLLM inference yet
    )

    job_config.training = Training(
        local_batch_size=1,  # Inference typically processes one batch at a time
        steps=1,  # Single step for inference
    )

    return job_config

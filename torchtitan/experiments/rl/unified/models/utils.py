# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from enum import Enum

import torch
import torch.distributed as dist
from safetensors.torch import load_file

from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.experiments.rl.unified.models.attention import VLLMAttention

from torchtitan.experiments.rl.vllm_compat.weights_vllm_compat import (
    torchtitan_to_vllm_compat,
)
from torchtitan.models.qwen3.model.args import Qwen3ModelArgs
from transformers import AutoConfig
from vllm.config import VllmConfig

logger = logging.getLogger(__name__)


class ModelMode(str, Enum):
    UNIFIED = "unified"
    BATCH_INVARIANT = "batch_invariant"
    STANDARD = "standard"


def replace_with_vllm_attention(model):
    """
    Replace TorchTitan attention with vLLM paged attention.

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

    logger.info(
        f"Successfully replaced TorchTitan attention with VLLMAttention "
        f"({len(model.layers)} layers)"
    )


def load_model(
    checkpoint_path: str, model_path: str, model_mode: str = ModelMode.BATCH_INVARIANT
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
        replace_with_vllm_attention(model)
        # Load standard TorchTitan format directly
        model.load_state_dict(state_dict, strict=True)
    elif model_mode == ModelMode.BATCH_INVARIANT:
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


def create_parallel_dims_from_vllm_config(vllm_config: VllmConfig) -> ParallelDims:
    """
    Create ParallelDims from vLLM config and maps vLLM parallelism settings to TorchTitan's ParallelDims dataclass.

    This function is needed because vLLM doesn't separate model creation and
    parallelism application - it requires parallelization to be done inside
    the model constructor, so we are creating parallel_dims and apply parallelism
    in TorchTitanVLLMModelWrapper.__init__ function.

    Args:
        vllm_config: vLLM configuration object

    Returns:
        ParallelDims object with parallelism settings validated

    Note:
        vLLM doesn't use FSDP sharding (dp_shard=1) or expert parallelism (ep=1, etp=1)
        in inference. These are set to default values.
    """
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

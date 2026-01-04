# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.tools.logging import logger

from ..backend import get_compile_backend_with_passes
from ..simple_fsdp import data_parallel, MixedPrecisionPolicy


# Import LFM2-specific components for activation checkpointing
from lfm2.main import LFM2Block, LFM2ConvBlock
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)


# for selective op activation checkpointing
_op_sac_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    torch.ops.aten._scaled_dot_product_attention_math.default,
    torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
    # for low precision training, it's useful to always save
    # the result of max, since the absolute maximum is
    # used to compute the scaling factor for quantization.
    torch.ops.aten.max.default,
    torch._higher_order_ops.flex_attention,
    torch.ops.torch_attn._varlen_attn,
    torch._higher_order_ops.inductor_compiled_code,
}


class CompatibleLFM2ConvBlock(nn.Module):
    """Wrapper for LFM2ConvBlock that accepts and ignores attention-specific arguments.

    This is a thin wrapper that delegates to the original conv block but accepts
    the same signature as attention blocks for compatibility.
    """
    def __init__(self, conv_block):
        super().__init__()

        # Copy everything from the original conv block's __dict__
        # This includes all sub-modules, parameters, buffers, and attributes like config
        self.__dict__.update(conv_block.__dict__)

        # Store original forward for reference
        self._original_forward = conv_block.forward.__get__(self, type(self))

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        past_key_value=None,
        output_attentions=None,
        use_cache=None,
        **kwargs
    ):
        # Call the original LFM2ConvBlock forward with only hidden_states
        result = self._original_forward(hidden_states)
        # Return as tuple to match attention block signature
        return (result,)


def apply_lfm2_ac(
    model: nn.Module,
    ac_config,
    *,
    model_compile_enabled: bool = False,
    base_folder: str = "",
) -> None:
    """Apply activation checkpointing to LFM2 layers.

    Supports two modes:
    - "selective": Checkpoint only attention blocks (which have O(seq_len²) memory).
                   Conv blocks are skipped since they have much lower O(seq_len) memory.
    - "full": Checkpoint ALL layers (both attention and conv blocks) for maximum
              memory savings at the cost of more recomputation.

    Args:
        model: SimpleFSDPLFM2Model instance (wraps LFM2ForCausalLM)
        ac_config: Activation checkpointing configuration (mode: "selective" or "full")
        model_compile_enabled: Whether torch.compile is enabled
        base_folder: Base folder for saving checkpointing artifacts
    """
    # Access the inner LFM2Model: model.model is LFM2ForCausalLM, model.model.model is LFM2Model
    lfm2_model = model.model.model

    # First pass: Replace ALL conv blocks with compatible versions
    # This is needed because LFM2 calls all layers with the same signature
    for layer_id, layer in enumerate(lfm2_model.layers):
        if isinstance(layer, LFM2ConvBlock):
            lfm2_model.layers[layer_id] = CompatibleLFM2ConvBlock(layer)

    # Second pass: Apply checkpointing
    attn_block_count = 0
    conv_block_count = 0
    checkpointed_count = 0

    for layer_id, layer in enumerate(lfm2_model.layers):
        should_checkpoint = False
        is_conv_block = False

        if isinstance(layer, LFM2Block):
            # Attention block
            attn_block_count += 1
            should_checkpoint = True  # Always checkpoint attention in both modes
        elif isinstance(layer, CompatibleLFM2ConvBlock):
            # Conv block (now wrapped)
            conv_block_count += 1
            is_conv_block = True
            should_checkpoint = (ac_config.mode == "full")  # Only checkpoint conv in full mode

        if should_checkpoint:
            checkpointed_layer = ptd_checkpoint_wrapper(
                layer,
                preserve_rng_state=ac_config.preserve_rng_state,
                determinism_check=ac_config.determinism_check,
                early_stop=ac_config.early_stop,
                debug=ac_config.debug,
            )
            lfm2_model.layers[layer_id] = checkpointed_layer
            checkpointed_count += 1

    if ac_config.mode == "selective":
        logger.info(
            f"Applied selective AC to {checkpointed_count} attention blocks "
            f"({conv_block_count} conv blocks left untouched for efficiency)"
        )
    else:  # full mode
        logger.info(
            f"Applied full AC to all {checkpointed_count} layers "
            f"({attn_block_count} attention + {conv_block_count} conv blocks)"
        )



def get_transformer_block_buckets(model) -> list[list[str] | str]:
    """
    Get transformer block buckets for LFM2 model.

    LFM2 has a different structure:
    - model (SimpleFSDPLFM2Model wrapper)
      - model (LFM2ForCausalLM)
        - model (LFM2Model)
          - embed_tokens
          - layers (ModuleDict with conv and attention blocks)
          - norm
        - lm_head
    """
    # Access the inner LFM2Model: model.model is LFM2ForCausalLM, model.model.model is LFM2Model
    lfm2_model = model.model.model

    module_list = [
        lfm2_model.embed_tokens,
        [lfm2_model.norm, model.model.lm_head],
    ]

    # Add each layer (conv and attention blocks)
    # Note: layers is a ModuleList, not ModuleDict
    for layer_block in lfm2_model.layers:
        module_list.append(layer_block)

    def convert_modules_to_fqns(modules, module_to_fqn_mapping):
        """Convert a (possibly nested) list of modules to FQN strings."""
        result = []
        for m in modules:
            if isinstance(m, list):
                if fqn_list := convert_modules_to_fqns(m, module_to_fqn_mapping):
                    result.append(fqn_list)
            else:
                if fqn := module_to_fqn_mapping.get(m):
                    result.append(fqn)
        return result

    module_to_name = {m: n for n, m in model.named_modules()}
    module_fqns = convert_modules_to_fqns(module_list, module_to_name)
    return module_fqns


def parallelize_lfm2(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    """
    Apply activation checkpointing, torch.compile, and data parallelism to the LFM2
    model using SimpleFSDP.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.

    NOTE: Tensor parallelism is not yet supported for LFM2.
    """
    # Tensor parallelism not yet supported for LFM2
    assert (
        job_config.training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {job_config.training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    if parallel_dims.tp_enabled:
        raise NotImplementedError(
            "Tensor parallelism is not yet supported for LFM2 with SimpleFSDP. "
            "Please disable TP in your config."
        )

    # Apply data parallel FIRST (before AC)
    # This ensures SimpleFSDP can find all parameters before we wrap anything
    if (
        parallel_dims.dp_replicate_enabled
        or parallel_dims.dp_shard_enabled
        or parallel_dims.cp_enabled
    ):
        if parallel_dims.dp_replicate_enabled:
            if parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled:
                dp_mesh_dim_names = ["dp_replicate", "fsdp"]
                dp_mode = "hybrid_shard"
            else:
                dp_mesh_dim_names = ["dp_replicate"]
                dp_mode = "replicate"
        else:
            dp_mesh_dim_names = ["fsdp"]
            dp_mode = "fully_shard"

        mp_policy = MixedPrecisionPolicy(
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
        )

        model = data_parallel(
            model,
            parallel_dims.get_mesh(dp_mesh_dim_names),
            mode=dp_mode,
            mp_policy=mp_policy,
        )

        logger.info(
            "Applied Data Parallel (simple_fsdp) (dp mode=%s) to the model", dp_mode
        )

    # SKIP activation checkpointing for SimpleFSDP
    # SimpleFSDP with torch.compile handles memory efficiently without AC
    # Applying AC after SimpleFSDP breaks parametrization
    if job_config.activation_checkpoint.mode != "none":
        raise ValueError(
            f"Activation checkpointing mode '{job_config.activation_checkpoint.mode}' is not supported with SimpleFSDP for LFM2. "
            "Please set --activation_checkpoint.mode=none in your config or command line. "
            "SimpleFSDP with torch.compile provides efficient memory management without AC."
        )

    if job_config.compile.enable and "model" in job_config.compile.components:
        torch._inductor.config.reorder_for_peak_memory = False

        match job_config.parallelism.fsdp_reshard_after_forward:
            case "always":
                fsdp_reshard_after_forward = True
            case "never":
                fsdp_reshard_after_forward = False
            case "default":
                # For PP, by default do not reshard after forward to avoid per-microbatch
                # all-gathers, which can be expensive and non-overlapped
                fsdp_reshard_after_forward = not parallel_dims.pp_enabled
            case _:
                raise ValueError(
                    f"Invalid fsdp_reshard_after_forward_policy: {job_config.parallelism.fsdp_reshard_after_forward}."
                )

        backend = get_compile_backend_with_passes(
            job_config.compile,
            fsdp_reshard_after_forward,
            get_transformer_block_buckets(model),
        )
        model = torch.compile(
            model,
            backend=backend,
            fullgraph=True,
        )

    return model

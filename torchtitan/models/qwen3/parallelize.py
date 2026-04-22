# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This file applies the PT-D parallelisms (except pipeline parallelism) and various
# training techniques (e.g. activation checkpointing and compile) to the Llama model.

import torch
import torch._inductor.config
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    RowwiseParallel,
    SequenceParallel,
)

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.context_parallel import apply_cp_to_attention_module
from torchtitan.distributed.tensor_parallel import NoParallel
from torchtitan.models.common.attention import FusedQKVLinear
from torchtitan.models.llama4.parallelize import apply_fsdp, apply_moe_ep_tp
from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan.tools.logging import logger


def parallelize_qwen3(
    model: Qwen3Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )
    if parallel_dims.tp_enabled:
        if parallelism.enable_async_tensor_parallel and not model_compile_enabled:
            raise RuntimeError("Async TP requires torch.compile")

        enable_sp = parallelism.enable_sequence_parallel

        tp_mesh = parallel_dims.get_mesh("tp")
        apply_non_moe_tp(
            model,
            tp_mesh,
            enable_loss_parallel=not parallelism.disable_loss_parallel,
            enable_async_tp=parallelism.enable_async_tensor_parallel,
            enable_cp=parallel_dims.cp_enabled,
            enable_sp=enable_sp,
        )

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        apply_moe_ep_tp(
            model,
            tp_mesh=parallel_dims.get_optional_mesh("tp"),
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
            etp_mesh=parallel_dims.get_optional_mesh("etp"),
            ep_etp_mesh=parallel_dims.get_optional_mesh(["ep", "etp"]),
        )

    if parallel_dims.cp_enabled:
        apply_cp_to_attention_module(
            # pyrefly: ignore [missing-attribute, not-callable]
            [block.attention.inner_attention for block in model.layers.values()],
            parallel_dims.get_mesh("cp"),
        )

    if ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            base_folder=dump_folder,
        )

    # turn on per-TransformerBlock compile after AC wrapping and before FSDP
    if model_compile_enabled:
        apply_compile(model, compile_config)

    dp_mesh_names = (
        ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
    )
    dp_mesh = parallel_dims.get_mesh(dp_mesh_names)

    edp_mesh = None
    if parallel_dims.ep_enabled:
        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

    apply_fsdp(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=parallel_dims.ep,
        edp_mesh=edp_mesh,
    )

    logger.info("Applied fully_shard to the model")

    if training.enable_cpu_offload:
        logger.info("Applied CPU Offloading to the model")

    return model


def apply_non_moe_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    enable_loss_parallel: bool,
    enable_async_tp: bool,
    enable_cp: bool,
    enable_sp: bool = True,
):
    """Apply tensor parallelism."""
    # 1. Parallelize the embedding and shard its outputs (which are the first
    # transformer block's inputs)
    # 2. Parallelize the root norm layer over the sequence dim
    # 3. Parallelize the final linear output layer
    sp_layout = Shard(1) if enable_sp else Replicate()
    embed_plan = RowwiseParallel(
        input_layouts=Replicate(),
        output_layouts=sp_layout,
        use_local_output=enable_sp,
    )

    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": embed_plan,
            "norm": SequenceParallel() if enable_sp else NoParallel(),
            "output": ColwiseParallel(
                input_layouts=sp_layout,
                output_layouts=Shard(-1) if enable_loss_parallel else Replicate(),
                use_local_output=not enable_loss_parallel,
            ),
        },
    )

    # Apply tensor + sequence parallelism to every transformer block
    # NOTE: At the cost of model code change, we can accelerate Sequence Parallel
    #       by folding (and unfolding) the batch dimension and the sequence dimension.
    #       Examples can be found at https://github.com/pytorch/torchtitan/pull/437
    positions_sharding = Replicate() if enable_cp else None
    norm_plan = SequenceParallel() if enable_sp else NoParallel()
    qk_norm_plan = SequenceParallel(sequence_dim=2)
    rowwise_output_plan = RowwiseParallel(
        output_layouts=sp_layout, use_local_output=enable_sp
    )

    # Detect whether fused QKV is used by checking the first layer
    # pyrefly: ignore [not-callable]
    first_block = next(iter(model.layers.values()))
    use_fused_qkv = isinstance(
        first_block.attention.qkv_linear,  # pyrefly: ignore [missing-attribute]
        FusedQKVLinear,
    )

    # pyrefly: ignore [not-callable]
    for transformer_block in model.layers.values():
        if use_fused_qkv:
            qkv_plan = {
                "attention.qkv_linear.wqkv": ColwiseParallel(use_local_output=False),
                "attention.q_norm": qk_norm_plan,
                "attention.k_norm": qk_norm_plan,
            }
        else:
            qkv_plan = {
                "attention.qkv_linear.wq": ColwiseParallel(use_local_output=False),
                "attention.qkv_linear.wk": ColwiseParallel(use_local_output=False),
                "attention.qkv_linear.wv": ColwiseParallel(use_local_output=False),
                "attention.q_norm": qk_norm_plan,
                "attention.k_norm": qk_norm_plan,
            }
        layer_plan = {
            "attention_norm": norm_plan,
            "attention": PrepareModuleInput(
                input_layouts=(sp_layout, Replicate(), None, positions_sharding),
                desired_input_layouts=(
                    Replicate(),
                    Replicate(),
                    None,
                    positions_sharding,
                ),
            ),
            **qkv_plan,
            "attention.wo": rowwise_output_plan,
            "ffn_norm": norm_plan,
        }

        # pyrefly: ignore [missing-attribute]
        if not transformer_block.moe_enabled:
            layer_plan.update(
                {
                    "feed_forward": PrepareModuleInput(
                        input_layouts=(sp_layout,),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "feed_forward.w1": ColwiseParallel(),
                    "feed_forward.w2": rowwise_output_plan,
                    "feed_forward.w3": ColwiseParallel(),
                }
            )

        parallelize_module(
            # pyrefly: ignore [bad-argument-type]
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    if enable_async_tp:
        torch._inductor.config._micro_pipeline_tp = True

    logger.info(
        f"Applied {'Async ' if enable_async_tp else ''}"
        "Tensor Parallelism to the model"
    )

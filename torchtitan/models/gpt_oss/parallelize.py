# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch._inductor.config
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import distribute_tensor, Partial, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    PrepareModuleInputOutput,
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
from torchtitan.distributed.expert_parallel import ExpertParallel
from torchtitan.distributed.tensor_parallel import NoParallel
from torchtitan.models.common.attention import FusedQKVLinear
from torchtitan.models.common.token_dispatcher import (
    AllToAllTokenDispatcher,
    TorchAOTokenDispatcher,
)
from torchtitan.models.gpt_oss.model import GptOssModel
from torchtitan.models.llama4.parallelize import apply_fsdp
from torchtitan.tools.logging import logger

from .expert_parallel import GptossExpertTensorParallel, GptossTensorParallel


# Adapted from llama4/infra/parallelize.py
def parallelize_gptoss(
    model: GptOssModel,
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

        apply_non_moe_tp(
            model,
            parallel_dims.get_mesh("tp"),
            enable_loss_parallel=not parallelism.disable_loss_parallel,
            enable_async_tp=False,
            enable_sp=enable_sp,
        )

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        apply_moe_ep_tp(
            model,
            tp_mesh=parallel_dims.get_optional_mesh("tp"),
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
            ep_etp_mesh=parallel_dims.get_optional_mesh(["ep", "etp"]),
            etp_enabled=parallel_dims.etp_enabled,
            enable_sp=True,
        )

    if parallel_dims.cp_enabled:
        apply_cp_to_attention_module(
            # pyrefly: ignore [missing-attribute]
            [block.attention.inner_attention for block in model.layers.values()],
            parallel_dims.get_mesh("cp"),
        )

    if ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
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

    if parallel_dims.cp_enabled:
        logger.info("Applied Context Parallel to the model")

    if training.enable_cpu_offload:
        logger.info("Applied CPU Offloading to the model")

    return model


def apply_non_moe_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    enable_loss_parallel: bool,
    enable_async_tp: bool,
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
    norm_plan = SequenceParallel() if enable_sp else NoParallel()
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
            }
        else:
            qkv_plan = {
                "attention.qkv_linear.wq": ColwiseParallel(use_local_output=False),
                "attention.qkv_linear.wk": ColwiseParallel(use_local_output=False),
                "attention.qkv_linear.wv": ColwiseParallel(use_local_output=False),
            }
        layer_plan = {
            "attention_norm": norm_plan,
            "attention": PrepareModuleInput(
                input_layouts=(sp_layout, Replicate(), None, None),
                desired_input_layouts=(Replicate(), Replicate(), None, None),
            ),
            **qkv_plan,
            "attention.wo": rowwise_output_plan,
            "ffn_norm": norm_plan,
        }

        # shard attention.sinks across heads
        # pyrefly: ignore [missing-attribute]
        attn = transformer_block.attention
        attn.register_parameter(
            "sinks",
            nn.Parameter(distribute_tensor(attn.sinks, tp_mesh, [Shard(0)])),
        )

        parallelize_module(
            # pyrefly: ignore [bad-argument-type]
            module=transformer_block,
            device_mesh=tp_mesh,
            # pyrefly: ignore [bad-argument-type]
            parallelize_plan=layer_plan,
        )

    if enable_async_tp:
        torch._inductor.config._micro_pipeline_tp = True

    logger.info(
        f"Applied {'Async ' if enable_async_tp else ''}"
        "Tensor Parallelism to the model"
    )


def apply_moe_ep_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh | None,
    ep_mesh: DeviceMesh | None,
    ep_etp_mesh: DeviceMesh | None,
    etp_enabled: bool,
    enable_sp: bool = True,
):
    assert ep_mesh is not None or tp_mesh is not None

    sp_layout = Shard(1) if enable_sp else Replicate()

    # pyrefly: ignore [not-callable]
    for transformer_block in model.layers.values():
        # pyrefly: ignore [missing-attribute]
        if not transformer_block.moe_enabled:
            continue

        if tp_mesh is not None:
            moe_layer_plan = {
                # With SP: all-gather (Shard→Replicate) for input,
                # reduce-scatter (Partial→Shard) for output.
                # Without SP: input is already Replicate,
                # all-reduce (Partial→Replicate) for output.
                "moe": PrepareModuleInputOutput(
                    input_layouts=(sp_layout,),
                    desired_input_layouts=(Replicate(),),
                    # Keep input as a DTensor from SequenceParallel, do not wrap with to_local.
                    use_local_input=False,
                    output_layouts=(Partial(),),
                    desired_output_layouts=(sp_layout,),
                ),
                # replicate computation for the router
                "moe.router.gate": NoParallel(
                    local_output_grad_placements=(Partial(),),
                ),
            }
            parallelize_module(
                # pyrefly: ignore [bad-argument-type]
                module=transformer_block,
                device_mesh=tp_mesh,
                # pyrefly: ignore [bad-argument-type]
                parallelize_plan=moe_layer_plan,
            )

        experts_mesh, experts_plan = None, None
        if ep_mesh is None:
            experts_mesh = tp_mesh
            # input Replicate, output Partial
            experts_plan = GptossTensorParallel()
        elif tp_mesh is None or not etp_enabled:
            experts_mesh = ep_mesh
            # sp_size and sp_rank are set for sequence-parallel token splitting
            # when EP borrows from TP (ETP=1).
            experts_plan = ExpertParallel()
            # pyrefly: ignore [missing-attribute]
            dispatcher = transformer_block.moe.experts.token_dispatcher
            if tp_mesh is not None:
                if isinstance(dispatcher, AllToAllTokenDispatcher):
                    dispatcher.sp_size = tp_mesh.size()
                    dispatcher.sp_rank = tp_mesh.get_local_rank()
        else:
            # pyrefly: ignore [missing-attribute]
            dispatcher = transformer_block.moe.experts.token_dispatcher
            if isinstance(dispatcher, TorchAOTokenDispatcher):
                raise NotImplementedError(
                    "Quantized grouped GEMMs (FP8/MXFP8) with Expert Tensor "
                    "Parallelism (ETP) is not yet supported."
                )
            experts_mesh = ep_etp_mesh
            experts_plan = GptossExpertTensorParallel()

        parallelize_module(
            # pyrefly: ignore [missing-attribute]
            module=transformer_block.moe.experts,
            device_mesh=experts_mesh,
            parallelize_plan=experts_plan,
        )

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import Partial, Replicate, Shard
from torch.distributed.tensor.parallel import (
    parallelize_module,
    PrepareModuleInputOutput,
)

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.context_parallel import apply_cp_to_forward
from torchtitan.distributed.expert_parallel import ExpertParallel
from torchtitan.distributed.tensor_parallel import NoParallel
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher
from torchtitan.models.gpt_oss.model import GptOssModel
from torchtitan.models.llama4.parallelize import apply_fsdp
from torchtitan.tools.logging import logger

from .expert_parallel import GptossTensorParallel


# Adapted from llama4/infra/parallelize.py
def parallelize_gptoss(
    model: GptOssModel,
    *,
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig | None = None,
    ac_config: ActivationCheckpointConfig | None = None,
    dump_folder: str = "",
):

    model_compile_enabled = (
        compile_config is not None
        and compile_config.enable
        and "model" in compile_config.components
    )

    # CP: wrap inner attention forward BEFORE parallelize() so CP logic
    # runs inside the local_map boundary on local tensors.
    if parallel_dims.cp_enabled:
        apply_cp_to_forward(
            # pyrefly: ignore [missing-attribute]
            [block.attention.inner_attention for block in model.layers.values()],
            parallel_dims.get_mesh("cp"),
        )

    if parallel_dims.tp_enabled:
        tp_mesh = parallel_dims.get_mesh("tp")
        model.parallelize(tp_mesh)

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        apply_moe_ep_tp(
            model,
            tp_mesh=parallel_dims.get_optional_mesh("tp"),
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
            enable_sp=True,
        )

    if ac_config is not None:
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
        param_dtype=TORCH_DTYPE_MAP[parallelism.fsdp_mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[parallelism.fsdp_mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=parallelism.enable_fsdp_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=parallel_dims.ep,
        edp_mesh=edp_mesh,
    )

    logger.info("Applied fully_shard to the model")

    if parallel_dims.cp_enabled:
        logger.info("Applied Context Parallel to the model")

    if parallelism.enable_fsdp_cpu_offload:
        logger.info("Applied CPU Offloading to the model")

    return model


def apply_moe_ep_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh | None,
    ep_mesh: DeviceMesh | None,
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
                    use_local_input=False,
                    output_layouts=(Partial(),),
                    desired_output_layouts=(sp_layout,),
                    # Keep MoE output as DTensor so the residual add
                    # ``h + self.moe(...)`` composes with config-based
                    # attention (which flows DTensors).
                    use_local_output=False,
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
        # EP disabled: shard routed expert weights across TP mesh (input Replicate, produces Partial output reduced at MoE boundary)
        if ep_mesh is None:
            experts_mesh = tp_mesh
            experts_plan = GptossTensorParallel()
        else:
            experts_mesh = ep_mesh
            experts_plan = ExpertParallel()
            # pyrefly: ignore [missing-attribute]
            dispatcher = transformer_block.moe.experts.token_dispatcher
            if tp_mesh is not None:
                if isinstance(dispatcher, AllToAllTokenDispatcher):
                    dispatcher.sp_size = tp_mesh.size()
                    # Use _sym_get_coordinate so the rank is a SymInt
                    # under CooR precompile instead of a concrete int
                    # that gets baked into the FX graph.
                    dispatcher.sp_rank = tp_mesh._sym_get_coordinate(0)

        parallelize_module(
            # pyrefly: ignore [missing-attribute]
            module=transformer_block.moe.experts,
            device_mesh=experts_mesh,
            parallelize_plan=experts_plan,
        )

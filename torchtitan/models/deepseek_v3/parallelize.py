# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
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

from torchtitan.components.quantization.float8 import find_float8_linear_config
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import NoParallel, ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.context_parallel import apply_cp_to_attention_module
from torchtitan.distributed.dual_pipe_v import get_dual_pipe_v_flag
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.models.deepseek_v3 import DeepSeekV3Model
from torchtitan.models.llama3.parallelize import apply_ddp
from torchtitan.models.llama4.parallelize import (
    apply_compile,
    apply_fsdp,
    apply_moe_ep_tp,
)
from torchtitan.protocols import ModelConvertersContainer
from torchtitan.tools.logging import logger

# for selective op activation checkpointing
_op_sac_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    torch.ops.aten._scaled_dot_product_attention_math.default,
    torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
    torch.ops._c10d_functional.all_to_all_single.default,
    # for low precision training, it's useful to always save
    # the result of max, since the absolute maximum is
    # used to compute the scaling factor for quantization.
    torch.ops.aten.max.default,
    torch._higher_order_ops.flex_attention,
    torch._higher_order_ops.inductor_compiled_code,
}


# Adapted from llama4/infra/parallelize.py
def parallelize_deepseekv3(
    model: DeepSeekV3Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    # TODO: TP currently cannot handle uneven seq_len because we set
    #       `use_local_output=True` to use plain Tensors for legacy reasons.
    #       Need to revisit this.
    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    attn_backend = getattr(model.config.layer.attention, "attn_backend", "sdpa")
    if parallelism.context_parallel_degree > 1 and attn_backend != "sdpa":
        raise NotImplementedError(
            f"Context Parallel only supports SDPA attention. "
            f"Got attn_backend='{attn_backend}'. "
            f"FlexAttention and varlen attention are not supported with CP."
        )

    if parallel_dims.tp_enabled:
        float8_config = find_float8_linear_config(model_converters.converters)
        enable_float8_linear = float8_config is not None
        float8_is_rowwise = float8_config is not None and float8_config.recipe_name in (
            "rowwise",
            "rowwise_with_gw_hp",
        )

        enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise
        if enable_float8_tensorwise_tp:
            # TODO(jianiw): This branch needs to be tested and enabled
            raise NotImplementedError(
                "Currently, float8 tensorwise TP is not tested for deepseekv3"
            )

        tp_mesh = parallel_dims.get_mesh("tp")
        apply_non_moe_tp(
            model,
            tp_mesh,
            loss_parallel=not parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=False,
            cp_enabled=parallel_dims.cp_enabled,
        )
        maybe_enable_async_tp(parallelism, compile_config, tp_mesh)

    # Check if using DeepEP for MoE communication
    if parallelism.expert_parallel_comm_backend == "deepep":
        if not parallel_dims.ep_enabled:
            raise ValueError(
                "DeepEP requires expert parallelism (ep_degree > 1). "
                "The DeepEP MoE model code does not support EP=1. "
                "Please set expert_parallel_degree > 1 or use standard communication backend."
            )
        if parallel_dims.etp_enabled:
            raise NotImplementedError(
                "DeepEP with Expert Tensor Parallelism (ETP) is not supported yet. "
                "Please set expert_tensor_parallel_degree=1 or use standard communication backend."
            )

        use_deepep = True

        # Import deepep module to register custom ops before accessing them
        import torchtitan.distributed.deepep  # noqa: F401 - registers torch.ops.deepep

        _op_sac_save_list.add(torch.ops.deepep.dispatch.default)
        _op_sac_save_list.add(torch.ops.deepep.combine.default)
    else:
        use_deepep = False

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        dual_pipe_v = get_dual_pipe_v_flag(
            parallelism=parallelism, ac_config=ac_config, parallel_dims=parallel_dims
        )

        apply_moe_ep_tp(
            model,
            tp_mesh=parallel_dims.get_optional_mesh("tp"),
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
            etp_mesh=parallel_dims.get_optional_mesh("etp"),
            ep_etp_mesh=parallel_dims.get_optional_mesh(["ep", "etp"]),
            dual_pipe_v=dual_pipe_v,
            use_deepep=use_deepep,
        )

    if parallel_dims.cp_enabled:
        apply_cp_to_attention_module(
            # pyrefly: ignore [missing-attribute, not-callable]
            [block.attention.inner_attention for block in model.layers.values()],
            parallel_dims.get_mesh("cp"),
            attn_backend,
        )

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    if ac_config.mode != "none":
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            # pyrefly: ignore [bad-argument-type]
            op_sac_save_list=_op_sac_save_list,
            base_folder=dump_folder,
        )

    if model_compile_enabled:
        apply_compile(model, compile_config, parallel_dims.ep_enabled)

    dp_mesh: DeviceMesh | None = None
    if parallel_dims.fsdp_enabled or parallel_dims.ep_enabled:
        # apply FSDP or HSDP, potentially with Context Parallel
        dp_mesh_names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(dp_mesh_names)

        # the mesh dim names of which the MoE params are sharded on via FSDP/HSDP
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
            gradient_divide_factor=parallel_dims.fsdp_gradient_divide_factor,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        if training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to the model")
    elif parallel_dims.dp_replicate_enabled:
        dp_mesh = parallel_dims.get_mesh("dp_replicate")
        if dp_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        apply_ddp(
            model,
            dp_mesh,
            enable_compile=model_compile_enabled,
        )

    return model


def apply_non_moe_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    enable_float8_tensorwise_tp: bool,
    cp_enabled: bool,
):
    """Apply tensor parallelism."""
    # 1. Parallelize the embedding and shard its outputs (which are the first
    # transformer block's inputs)
    # 2. Parallelize the root norm layer over the sequence dim
    # 3. Parallelize the final linear output layer
    parallelize_module(
        model,
        tp_mesh,
        {
            "tok_embeddings": RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            ),
            "norm": SequenceParallel(),
            "output": ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            ),
        },
    )

    rowwise_parallel, colwise_parallel, prepare_module_input = (
        RowwiseParallel,
        ColwiseParallel,
        PrepareModuleInput,
    )

    attention_kernel_plan = prepare_module_input(
        input_layouts=(Shard(1), Shard(1), Shard(1)),
        desired_input_layouts=(Shard(1), Shard(1), Shard(1)),
        use_local_output=True,
    )
    # Apply tensor + sequence parallelism to every transformer block
    # NOTE: At the cost of model code change, we can accelerate Sequence Parallel
    #       by folding (and unfolding) the batch dimension and the sequence dimension.
    #       Examples can be found at https://github.com/pytorch/torchtitan/pull/437
    positions_sharding = Replicate() if cp_enabled else None
    # pyrefly: ignore [not-callable]
    for transformer_block in model.layers.values():
        layer_plan = {
            "attention_norm": SequenceParallel(),
            "attention": prepare_module_input(
                input_layouts=(Shard(1), Replicate(), None, positions_sharding),
                desired_input_layouts=(
                    Replicate(),
                    Replicate(),
                    None,
                    positions_sharding,
                ),
            ),
            # NOTE: use_local_output=False make the output to be a DTensor instead of a plain Tensor
            # so that the intermedidate results k is generated as a DTensor and its gradient is
            # correctly handled by the autograd engine.
            "attention.wkv_a": NoParallel(use_local_output=False),
            "attention.wkv_b": colwise_parallel(use_local_output=False),
            "attention.kv_norm": NoParallel(use_local_output=False),
            # NOTE: use_local_output=True so that the inputs to FlexAttention are plain Tensors
            "attention.inner_attention": attention_kernel_plan,
            "attention.wo": rowwise_parallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
        }

        # pyrefly: ignore [missing-attribute]
        if transformer_block.attention.q_lora_rank == 0:
            layer_plan.update(
                {
                    "attention.wq": colwise_parallel(
                        use_local_output=False
                    ),  # This is only used when q_lora_rank==0
                }
            )
        else:
            layer_plan.update(
                {
                    "attention.wq_a": NoParallel(use_local_output=False),
                    "attention.wq_b": colwise_parallel(use_local_output=False),
                    "attention.q_norm": NoParallel(use_local_output=False),
                }
            )

        # pyrefly: ignore [missing-attribute]
        if not transformer_block.moe_enabled:
            layer_plan.update(
                {
                    "feed_forward": prepare_module_input(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                    ),
                    "feed_forward.w1": colwise_parallel(),
                    "feed_forward.w2": rowwise_parallel(output_layouts=Shard(1)),
                    "feed_forward.w3": colwise_parallel(),
                }
            )

        parallelize_module(
            # pyrefly: ignore [bad-argument-type]
            module=transformer_block,
            device_mesh=tp_mesh,
            # pyrefly: ignore [bad-argument-type]
            parallelize_plan=layer_plan,
        )

    logger.info(
        f"Applied {'Float8 tensorwise ' if enable_float8_tensorwise_tp else ''}"
        "Tensor Parallelism to the model"
    )

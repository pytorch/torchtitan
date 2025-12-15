# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import Partial, Replicate, Shard
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    parallelize_module,
    PrepareModuleInput,
    PrepareModuleInputOutput,
    RowwiseParallel,
    SequenceParallel,
)
from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.config.job_config import Compile as CompileConfig
from torchtitan.distributed import NoParallel, ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac

from torchtitan.distributed.expert_parallel import (
    ExpertParallel,
    ExpertTensorParallel,
    ReordererSequenceParallel,
    TensorParallel,
)
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.models.llama3.infra.parallelize import apply_ddp
from torchtitan.models.moe import moe as moe_module
from torchtitan.tools.logging import logger


# for selective op activation checkpointing
_op_sac_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
    torch.ops._c10d_functional.all_to_all_single.default,
    # for low precision training, it's useful to always save
    # the result of max, since the absolute maximum is
    # used to compute the scaling factor for quantization.
    torch.ops.aten.max.default,
    torch._higher_order_ops.flex_attention,
}


def parallelize_llama(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    world_mesh = parallel_dims.world_mesh
    # TODO: TP currently cannot handle uneven seq_len because we set
    #       `use_local_output=True` to use plain Tensors for legacy reasons.
    #       Need to revisit this.
    assert (
        job_config.training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {job_config.training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    if parallel_dims.tp_enabled:
        enable_float8_linear = "float8" in job_config.model.converters
        float8_is_rowwise = job_config.quantize.linear.float8.recipe_name in (
            "rowwise",
            "rowwise_with_gw_hp",
        )

        # For now, float8 all-gather with TP is only supported for tensorwise
        # float8 scaling recipes. For rowwise recipes, we use regular TP and
        # all-gather happens in high precision.
        enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise

        apply_non_moe_tp(
            model,
            world_mesh["tp"],
            loss_parallel=not job_config.parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=enable_float8_tensorwise_tp,
        )
        maybe_enable_async_tp(job_config, world_mesh["tp"])

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        apply_moe_ep_tp(
            model,
            tp_mesh=world_mesh["tp"] if parallel_dims.tp_enabled else None,
            ep_mesh=world_mesh["ep"] if parallel_dims.ep_enabled else None,
            ep_tp_mesh=(
                world_mesh["ep", "tp"]
                if parallel_dims.tp_enabled
                and parallel_dims.ep_enabled
                and parallel_dims.etp_enabled
                else None
            ),
            etp_enabled=parallel_dims.etp_enabled,
        )

    model_compile_enabled = (
        job_config.compile.enable and "model" in job_config.compile.components
    )
    attn_type = getattr(model.model_args, "attn_type", "sdpa")
    use_flex_attn = attn_type == "flex"
    if job_config.activation_checkpoint.mode != "none":
        apply_ac(
            model,
            job_config.activation_checkpoint,
            model_compile_enabled=model_compile_enabled,
            use_flex_attn=use_flex_attn,
            op_sac_save_list=_op_sac_save_list,
            base_folder=job_config.job.dump_folder,
        )

    # turn on per-TransformerBlock compile after AC wrapping and before FSDP
    if model_compile_enabled:
        apply_compile(model, job_config.compile)

    dp_mesh: DeviceMesh | None = None
    if parallel_dims.fsdp_enabled or parallel_dims.ep_enabled:
        # apply FSDP or HSDP, potentially with Context Parallel
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "dp_shard_cp")
        else:
            dp_mesh_dim_names = ("dp_shard_cp",)
        dp_mesh = world_mesh[tuple(dp_mesh_dim_names)]

        # the mesh dim names of which the MoE params are sharded on via FSDP/HSDP
        dp_mod_ep_mesh_dim_names = []
        if parallel_dims.ep_enabled:
            if parallel_dims.dp_replicate_enabled:
                dp_mod_ep_mesh_dim_names.append("dp_replicate")
            dp_mod_ep_mesh_dim_names.append("dp_shard_mod_ep")

        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=job_config.training.enable_cpu_offload,
            reshard_after_forward_policy=job_config.parallelism.fsdp_reshard_after_forward,
            ep_degree=parallel_dims.ep,
            dp_mod_ep_mesh=(
                world_mesh[tuple(dp_mod_ep_mesh_dim_names)]
                if parallel_dims.ep_enabled
                else None
            ),
            gradient_divide_factor=parallel_dims.fsdp_gradient_divide_factor,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        if parallel_dims.cp_enabled:
            logger.info("Applied Context Parallel to the model")

        if job_config.training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to the model")
    elif parallel_dims.dp_replicate_enabled:
        if world_mesh.ndim > 1:
            raise RuntimeError("DDP has not supported > 1D parallelism")
        dp_mesh = world_mesh
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

    # Parallel styles used for transformer block linear weights and their
    # inputs may be different for float8 linears with tensorwise scaling.
    if enable_float8_tensorwise_tp:
        # TODO(vkuzo): add the items below to __init__.py of torchao.float8 and import from there
        from torchao.float8.float8_tensor_parallel import (
            Float8ColwiseParallel,
            Float8RowwiseParallel,
            PrepareFloat8ModuleInput,
        )

        rowwise_parallel, colwise_parallel, prepare_module_input = (
            Float8RowwiseParallel,
            Float8ColwiseParallel,
            PrepareFloat8ModuleInput,
        )
    else:
        rowwise_parallel, colwise_parallel, prepare_module_input = (
            RowwiseParallel,
            ColwiseParallel,
            PrepareModuleInput,
        )

    # Apply tensor + sequence parallelism to every transformer block
    for transformer_block in model.layers.values():
        layer_plan = {
            "attention_norm": SequenceParallel(),
            "attention": prepare_module_input(
                input_layouts=(Shard(1), None, None),
                desired_input_layouts=(Replicate(), None, None),
            ),
            "attention.wq": colwise_parallel(),
            "attention.wk": colwise_parallel(),
            "attention.wv": colwise_parallel(),
            "attention.wo": rowwise_parallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
        }
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
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    logger.info(
        f"Applied {'Float8 tensorwise ' if enable_float8_tensorwise_tp else ''}"
        "Tensor Parallelism to the model"
    )


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    ep_degree: int = 1,
    dp_mod_ep_mesh: DeviceMesh | None = None,
    gradient_divide_factor: int | None = None,
):
    """
    Apply data parallelism (via FSDP2) to the model.

    Args:
        model (nn.Module): The model to apply data parallelism to.
        dp_mesh (DeviceMesh): The device mesh to use for data parallelism.
        param_dtype (torch.dtype): The data type to use for model parameters.
        reduce_dtype (torch.dtype): The data type to use for reduction operations.
        pp_enabled (bool): Whether pipeline parallelism is enabled.
        cpu_offload (bool, optional): Whether to offload model parameters to CPU. Defaults to False.
        reshard_after_forward_policy (str, optional): The policy to use for resharding after forward pass. Defaults to "default".
            Other options: "never", "always".
            - "default" applies default resharding behavior, implementing "smart defaults" for known optimal scenarios.
            - "always" will enable `reshard_after_forward` for all forward passes.
            - "never" will disable `reshard_after_forward` for all forward passes.

    """
    mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    match reshard_after_forward_policy:
        case "always":
            reshard_after_forward = True
        case "never":
            reshard_after_forward = False
        case "default":
            # For PP, by default do not reshard after forward to avoid per-microbatch
            # all-gathers, which can be expensive and non-overlapped
            reshard_after_forward = not pp_enabled
        case _:
            raise ValueError(
                f"Invalid reshard_after_forward_policy: {reshard_after_forward_policy}."
            )

    if model.tok_embeddings is not None:
        fully_shard(
            model.tok_embeddings,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    for layer_id, transformer_block in model.layers.items():
        # NOTE: When EP is enabled, In an MoE layer, we use the following FSDP wrapping
        # - the router and the shared experts are sharded together with the TransformerBlock
        # - the routed experts are sharded with the remaining dp_mod_ep_mesh
        if transformer_block.moe_enabled and ep_degree > 1:
            fsdp_mod_ep_config = fsdp_config.copy()
            fsdp_mod_ep_config["mesh"] = dp_mod_ep_mesh

            # NOTE: EP alreadys shards the routed experts on dim 0 (num_experts).
            #       When dp_mod_ep * ep > num_experts, FSDP default dim-0 sharding
            #       causes inefficiency, so we choose to do FSDP sharding on dim-1.
            #       Even when EP is not used, we may still want to shard the experts
            #       on non-0 dim. For now it may not be worth the complexity to support
            #       shard_placement_fn on the outer TransformerBlock-level FSDP.
            _experts_shard_placement_fn = None
            assert dp_mod_ep_mesh is not None
            assert hasattr(transformer_block, "moe")
            if (
                dp_mod_ep_mesh.size() * ep_degree
                > transformer_block.moe.experts.num_experts
            ):
                _experts_shard_placement_fn = lambda param: Shard(1)

            fully_shard(
                transformer_block.moe.experts,
                **fsdp_mod_ep_config,
                reshard_after_forward=reshard_after_forward,
                shard_placement_fn=_experts_shard_placement_fn,
            )

            # NOTE: # Although the FSDP sharding of experts is done on a mesh of
            #       a different size than other parameters, the gradient division
            #       factor should be consistent with data.
            transformer_block.moe.experts.set_gradient_divide_factor(
                gradient_divide_factor,
            )

        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    # As an optimization, do not reshard_after_forward the last layers by default
    # since FSDP would prefetch them immediately after the forward pass
    if model.norm is not None and model.output is not None:
        fully_shard(
            [model.norm, model.output],
            **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )

    fully_shard(model, **fsdp_config)

    # NOTE: set up explicit prefetching when EP is enabled, as D2H syncs
    # in EP could interfere with implicit prefetching in FSDP
    if ep_degree == 1:
        return

    # forward
    transformer_blocks = list(model.layers.values())
    next_transformer_blocks = transformer_blocks[1:] + [None]

    if model.tok_embeddings is not None and len(model.layers) > 0:
        model.tok_embeddings.set_modules_to_forward_prefetch([transformer_blocks[0]])

    for transformer_block, next_transformer_block in zip(
        transformer_blocks, next_transformer_blocks
    ):
        if next_transformer_block is not None:
            if next_transformer_block.moe_enabled:
                transformer_block.set_modules_to_forward_prefetch(
                    [next_transformer_block, next_transformer_block.moe.experts]
                )
            else:
                transformer_block.set_modules_to_forward_prefetch(
                    [next_transformer_block]
                )
        elif model.norm is not None and model.output is not None:
            transformer_block.set_modules_to_forward_prefetch(
                [model.norm, model.output]
            )

    # backward
    reversed_transformer_blocks = list(reversed(model.layers.values()))
    prev_transformer_blocks = reversed_transformer_blocks[1:] + [None]

    if model.norm is not None and model.output is not None and len(model.layers) > 0:
        model.output.set_modules_to_backward_prefetch([reversed_transformer_blocks[0]])

    for transformer_block, prev_transformer_block in zip(
        reversed_transformer_blocks, prev_transformer_blocks
    ):
        if prev_transformer_block is not None:
            if prev_transformer_block.moe_enabled:
                transformer_block.set_modules_to_backward_prefetch(
                    [prev_transformer_block, prev_transformer_block.moe.experts]
                )
            else:
                transformer_block.set_modules_to_backward_prefetch(
                    [prev_transformer_block]
                )
        elif model.tok_embeddings is not None:
            transformer_block.set_modules_to_backward_prefetch([model.tok_embeddings])


def apply_moe_ep_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh | None,
    ep_mesh: DeviceMesh | None,
    ep_tp_mesh: DeviceMesh | None,
    etp_enabled: bool,
):
    assert ep_mesh is not None or tp_mesh is not None

    for transformer_block in model.layers.values():
        if not transformer_block.moe_enabled:
            continue

        if tp_mesh is not None:
            moe_layer_plan = {
                # input / output sharding on the seqlen dim
                # all-gather for input, reduce-scatter for output
                "moe": PrepareModuleInputOutput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                    use_local_input=True,
                    output_layouts=(Partial(),),
                    desired_output_layouts=(Shard(1),),
                ),
                # replicate computation for the router
                "moe.router.gate": NoParallel(),
            }
            if ep_mesh is not None and not etp_enabled:
                # If TP is borrowed for EP, then split the tokens across TP ranks so that
                # the reorderer, the all-to-all comms, and routed experts computation
                # are effectively running Sequence Parallel (split along the folded bs*slen dim)
                moe_layer_plan.update({"moe.reorderer": ReordererSequenceParallel()})
            if transformer_block.moe.shared_experts is not None:
                # input Replicate, output Partial
                moe_layer_plan.update(
                    {
                        "moe.shared_experts.w1": ColwiseParallel(),
                        "moe.shared_experts.w2": RowwiseParallel(
                            output_layouts=Partial()
                        ),
                        "moe.shared_experts.w3": ColwiseParallel(),
                    }
                )
            parallelize_module(
                module=transformer_block,
                device_mesh=tp_mesh,
                parallelize_plan=moe_layer_plan,
            )

        experts_mesh, experts_plan = None, None
        if ep_mesh is None:
            experts_mesh = tp_mesh
            # input Replicate, output Partial
            experts_plan = TensorParallel()
        elif tp_mesh is None or not etp_enabled:
            experts_mesh = ep_mesh
            # input / output sharding on the batch / tokens dim
            experts_plan = ExpertParallel()
        else:
            experts_mesh = ep_tp_mesh
            experts_plan = ExpertTensorParallel()

        parallelize_module(
            module=transformer_block.moe.experts,
            device_mesh=experts_mesh,
            parallelize_plan=experts_plan,
        )


def apply_compile(model: nn.Module, compile_config: CompileConfig):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    # NOTE: This flag is needed for torch.compile to avoid graph breaking on dynamic shapes in token-choice MoE
    # but it is experimental.
    torch._dynamo.config.capture_scalar_outputs = True
    # Workaround for https://github.com/pytorch/pytorch/issues/166926
    torch._C._dynamo.eval_frame._set_lru_cache(False)
    for layer_id, transformer_block in model.layers.named_children():
        if transformer_block.moe_enabled:
            # If it is a MoE layer, FSDP(GroupedExperts) will cause a graph break
            # So we must weave compile wrappers around those FSDP hooks to
            # prevent AC from falling back the whole graph to eager.
            # TODO: Fix Compile(AC(graph break))

            if isinstance(transformer_block, CheckpointWrapper):
                # TODO: Make CheckpointWrapper a transparent wrapper
                # unwrap so that .named_children() works
                block = transformer_block._checkpoint_wrapped_module
            else:
                block = transformer_block

            for attr_name, submod in block.named_children():
                assert getattr(block, attr_name) == getattr(
                    transformer_block, attr_name
                )

                if isinstance(submod, moe_module.MoE):
                    # avoid graph breaking on the GroupedExperts' FSDP hooks
                    # by wrapping each submod's forward instead of their __call__
                    moe = submod
                    for attr_name, submod in moe.named_children():
                        if attr_name == "experts":
                            # NOTE: We don't compile token dispatch and token combine due to an issue on B200:
                            # https://github.com/pytorch/torchtitan/issues/1940
                            continue
                        setattr(
                            moe,
                            attr_name,
                            torch.compile(
                                submod, backend=compile_config.backend, fullgraph=True
                            ),
                        )
                else:
                    setattr(
                        block,
                        attr_name,
                        torch.compile(
                            submod, backend=compile_config.backend, fullgraph=True
                        ),
                    )

        else:
            # If it's not a MoE layer, there is no FSDP(GroupedExperts)
            # So we can compile the whole block
            transformer_block = torch.compile(
                transformer_block,
                backend=compile_config.backend,
                fullgraph=True,
            )

        model.layers.register_module(layer_id, transformer_block)

    moe_module._run_experts_grouped_mm = torch.compile(
        moe_module._run_experts_grouped_mm,
        backend=compile_config.backend,
        fullgraph=True,
    )

    # NOTE: We don't compile for loop code path due to an issue with unbacked symints:
    # https://github.com/pytorch/pytorch/issues/166460

    logger.info("Compiling each TransformerBlock with torch.compile")

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

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
from torchtitan.distributed.dual_pipe_v import (
    DualPipeExpertParallel,
    get_dual_pipe_v_flag,
)
from torchtitan.distributed.expert_parallel import (
    BaseExpertParallel,
    DeepEPExpertParallel,
    ExpertParallel,
    ExpertTensorParallel,
    ReordererSequenceParallel,
    TensorParallel,
)
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.models.common.moe import moe as moe_module
from torchtitan.models.llama3.parallelize import (
    apply_ddp,
    disable_fsdp_gradient_division,
)
from torchtitan.models.llama4.model import Llama4Model
from torchtitan.protocols.model_converter import ModelConvertersContainer
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


def parallelize_llama(
    model: Llama4Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    # TODO: TP currently cannot handle uneven seq_len because we set
    #       `use_local_output=True` to use plain Tensors for legacy reasons.
    #       Need to revisit this.
    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    tp_mesh = None
    if parallel_dims.tp_enabled:
        float8_config = find_float8_linear_config(model_converters.converters)
        enable_float8_linear = float8_config is not None
        float8_is_rowwise = float8_config is not None and float8_config.recipe_name in (
            "rowwise",
            "rowwise_with_gw_hp",
        )

        # For now, float8 all-gather with TP is only supported for tensorwise
        # float8 scaling recipes. For rowwise recipes, we use regular TP and
        # all-gather happens in high precision.
        enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise

        tp_mesh = parallel_dims.get_mesh("tp")
        apply_non_moe_tp(
            model,
            tp_mesh,
            loss_parallel=not parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=enable_float8_tensorwise_tp,
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

    attn_backend = model.config.layer.attention.attn_backend
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
        if ac_config.selective_ac_option == "op":
            logger.info(
                f"SAC save list contains {len(_op_sac_save_list)} ops: "
                f"{sorted([str(op) for op in _op_sac_save_list])}"
            )
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            # pyrefly: ignore [bad-argument-type]
            op_sac_save_list=_op_sac_save_list,
            base_folder=dump_folder,
        )

    # turn on per-TransformerBlock compile after AC wrapping and before FSDP
    if model_compile_enabled:
        apply_compile(model, compile_config, parallel_dims.ep_enabled)

    if parallel_dims.fsdp_enabled or parallel_dims.ep_enabled:
        # dp_mesh is the mesh for FSDP/HSDP
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
        if parallel_dims.world_size != dp_mesh.size():
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
    # pyrefly: ignore [not-callable]
    for transformer_block in model.layers.values():
        layer_plan = {
            "attention_norm": SequenceParallel(),
            # NOTE: when the fourth argument (positions) is not None, its input layout
            # and desired input layout is still None as we don't convert freqs_cis to
            # a DTensor for llama4.
            "attention": prepare_module_input(
                input_layouts=(Shard(1), None, None, None),
                desired_input_layouts=(Replicate(), None, None, None),
            ),
            "attention.wq": colwise_parallel(),
            "attention.wk": colwise_parallel(),
            "attention.wv": colwise_parallel(),
            "attention.wo": rowwise_parallel(output_layouts=Shard(1)),
            "ffn_norm": SequenceParallel(),
        }
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


def apply_fsdp(
    model: nn.Module,
    dp_mesh: DeviceMesh,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    pp_enabled: bool,
    cpu_offload: bool = False,
    reshard_after_forward_policy: str = "default",
    ep_degree: int = 1,
    edp_mesh: DeviceMesh | None = None,
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
    fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}
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
        # pyrefly: ignore [no-matching-overload]
        fully_shard(
            model.tok_embeddings,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    # pyrefly: ignore [missing-attribute]
    for layer_id, transformer_block in model.layers.items():
        # NOTE: When EP is enabled, In an MoE layer, we use the following FSDP wrapping
        # - the router and the shared experts are sharded together with the TransformerBlock
        # - the routed experts are sharded with the remaining edp_mesh
        if transformer_block.moe_enabled and ep_degree > 1:
            fsdp_mod_ep_config = fsdp_config.copy()
            fsdp_mod_ep_config["mesh"] = edp_mesh

            # NOTE: EP alreadys shards the routed experts on dim 0 (num_experts).
            #       When dp_mod_ep * ep > num_experts, FSDP default dim-0 sharding
            #       causes inefficiency, so we choose to do FSDP sharding on dim-1.
            #       Even when EP is not used, we may still want to shard the experts
            #       on non-0 dim. For now it may not be worth the complexity to support
            #       shard_placement_fn on the outer TransformerBlock-level FSDP.
            _experts_shard_placement_fn = None
            assert edp_mesh is not None
            assert hasattr(transformer_block, "moe")
            if (
                edp_mesh["efsdp"].size() * ep_degree
                > transformer_block.moe.experts.num_experts
            ):
                _experts_shard_placement_fn = lambda param: Shard(1)

            fully_shard(
                transformer_block.moe.experts,
                **fsdp_mod_ep_config,
                reshard_after_forward=reshard_after_forward,
                shard_placement_fn=_experts_shard_placement_fn,
            )

        fully_shard(
            transformer_block,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    # As an optimization, do not reshard_after_forward the last layers by default
    # since FSDP would prefetch them immediately after the forward pass
    if model.norm is not None and model.output is not None:
        # pyrefly: ignore [no-matching-overload]
        fully_shard(
            [model.norm, model.output],
            **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )

    fully_shard(model, **fsdp_config)

    # Disable FSDP's automatic gradient division for all FSDP modules
    disable_fsdp_gradient_division(model)

    # NOTE: set up explicit prefetching when EP is enabled, as D2H syncs
    # in EP could interfere with implicit prefetching in FSDP
    if ep_degree == 1:
        return

    # forward
    # pyrefly: ignore [not-callable]
    transformer_blocks = list(model.layers.values())
    next_transformer_blocks = transformer_blocks[1:] + [None]

    # pyrefly: ignore [bad-argument-type]
    if model.tok_embeddings is not None and len(model.layers) > 0:
        # pyrefly: ignore [missing-attribute]
        model.tok_embeddings.set_modules_to_forward_prefetch([transformer_blocks[0]])

    for transformer_block, next_transformer_block in zip(
        transformer_blocks, next_transformer_blocks
    ):
        if next_transformer_block is not None:
            # pyrefly: ignore [missing-attribute]
            if next_transformer_block.moe_enabled:
                # pyrefly: ignore [missing-attribute]
                transformer_block.set_modules_to_forward_prefetch(
                    # pyrefly: ignore [missing-attribute]
                    [next_transformer_block, next_transformer_block.moe.experts]
                )
            else:
                # pyrefly: ignore [missing-attribute]
                transformer_block.set_modules_to_forward_prefetch(
                    [next_transformer_block]
                )
        elif model.norm is not None and model.output is not None:
            # pyrefly: ignore [missing-attribute]
            transformer_block.set_modules_to_forward_prefetch(
                [model.norm, model.output]
            )

    # backward
    # pyrefly: ignore [not-callable]
    reversed_transformer_blocks = list(reversed(model.layers.values()))
    prev_transformer_blocks = reversed_transformer_blocks[1:] + [None]

    # pyrefly: ignore [bad-argument-type]
    if model.norm is not None and model.output is not None and len(model.layers) > 0:
        # pyrefly: ignore [missing-attribute]
        model.output.set_modules_to_backward_prefetch([reversed_transformer_blocks[0]])

    for transformer_block, prev_transformer_block in zip(
        reversed_transformer_blocks, prev_transformer_blocks
    ):
        if prev_transformer_block is not None:
            # pyrefly: ignore [missing-attribute]
            if prev_transformer_block.moe_enabled:
                # pyrefly: ignore [missing-attribute]
                transformer_block.set_modules_to_backward_prefetch(
                    # pyrefly: ignore [missing-attribute]
                    [prev_transformer_block, prev_transformer_block.moe.experts]
                )
            else:
                # pyrefly: ignore [missing-attribute]
                transformer_block.set_modules_to_backward_prefetch(
                    [prev_transformer_block]
                )
        elif model.tok_embeddings is not None:
            # pyrefly: ignore [missing-attribute]
            transformer_block.set_modules_to_backward_prefetch([model.tok_embeddings])


def apply_moe_ep_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh | None,
    ep_mesh: DeviceMesh | None,
    etp_mesh: DeviceMesh | None,
    ep_etp_mesh: DeviceMesh | None,
    dual_pipe_v: bool = False,
    use_deepep: bool = False,
):
    assert ep_mesh is not None or tp_mesh is not None

    # pyrefly: ignore [not-callable]
    for transformer_block in model.layers.values():
        # pyrefly: ignore [missing-attribute]
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
            if ep_mesh is not None and etp_mesh is None:
                # If TP is borrowed for EP, then split the tokens across TP ranks so that
                # the reorderer, the all-to-all comms, and routed experts computation
                # are effectively running Sequence Parallel (split along the folded bs*slen dim)
                # pyrefly: ignore [no-matching-overload]
                moe_layer_plan.update({"moe.reorderer": ReordererSequenceParallel()})
            # pyrefly: ignore [missing-attribute]
            if transformer_block.moe.shared_experts is not None:
                # input Replicate, output Partial
                # pyrefly: ignore [no-matching-overload]
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
                # pyrefly: ignore [bad-argument-type]
                module=transformer_block,
                device_mesh=tp_mesh,
                # pyrefly: ignore [bad-argument-type]
                parallelize_plan=moe_layer_plan,
            )

        experts_mesh, experts_plan = None, None
        if ep_mesh is None:
            assert ep_etp_mesh is None
            experts_mesh = tp_mesh
            # input Replicate, output Partial
            experts_plan = TensorParallel()
        elif tp_mesh is None or etp_mesh is None:
            assert ep_etp_mesh is None
            experts_mesh = ep_mesh
            if use_deepep:
                # pyrefly: ignore [missing-attribute]
                score_before_experts = transformer_block.moe.score_before_experts

                experts_plan = DeepEPExpertParallel(
                    score_before_experts=score_before_experts,
                )
                logger.info("Applying DeepEP to MoE layer")
            else:
                # input / output sharding on the batch / tokens dim
                experts_plan = ExpertParallel()
        else:
            experts_mesh = ep_etp_mesh
            experts_plan = ExpertTensorParallel()

        if dual_pipe_v and isinstance(experts_plan, BaseExpertParallel):
            experts_plan = DualPipeExpertParallel(experts_plan)

        parallelize_module(
            # pyrefly: ignore [missing-attribute]
            module=transformer_block.moe.experts,
            device_mesh=experts_mesh,
            parallelize_plan=experts_plan,
        )


def apply_compile(model: nn.Module, compile_config: CompileConfig, ep_enabled: bool):
    """
    Apply torch.compile to each TransformerBlock, which makes compilation efficient due to
    repeated structure. Alternatively one can compile the whole model (after applying DP).
    """
    # NOTE: This flag is needed for torch.compile to avoid graph breaking on dynamic shapes in token-choice MoE
    # but it is experimental.
    torch._dynamo.config.capture_scalar_outputs = True
    # pyrefly: ignore [missing-attribute]
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

        # pyrefly: ignore [missing-attribute]
        model.layers.register_module(layer_id, transformer_block)

    # Patch some globals only once (apply_compile is called multiple times for PP setup)
    already_patched = (
        "_run_experts_grouped_mm_dynamic"
        in moe_module._run_experts_grouped_mm.__qualname__
    )
    if not already_patched:
        moe_module._run_experts_grouped_mm = torch.compile(
            moe_module._run_experts_grouped_mm,
            backend=compile_config.backend,
            fullgraph=True,
        )

        if ep_enabled:
            compiled_fn = moe_module._run_experts_grouped_mm

            # keep function logic in sync with `already_patched` above
            def _run_experts_grouped_mm_dynamic(
                w1: torch.Tensor,
                w2: torch.Tensor,
                w3: torch.Tensor,
                x: torch.Tensor,
                num_tokens_per_expert: torch.Tensor,
            ) -> torch.Tensor:
                # dynamic number of tokens in expert parallel
                torch._dynamo.mark_dynamic(x, 0)
                return compiled_fn(w1, w2, w3, x, num_tokens_per_expert)

            moe_module._run_experts_grouped_mm = _run_experts_grouped_mm_dynamic

    # NOTE: We don't compile for loop code path due to an issue with unbacked symints:
    # https://github.com/pytorch/pytorch/issues/166460

    logger.info("Compiling each TransformerBlock with torch.compile")

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
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

from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.models.llama3.parallelize import (
    apply_compile,
    apply_ddp,
    disable_fsdp_gradient_division,
)
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools.logging import logger


def parallelize_hf_transformers(
    model: nn.Module,
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

        apply_non_moe_tp(
            model,
            parallel_dims.get_mesh("tp"),
            loss_parallel=not parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=enable_float8_tensorwise_tp,
        )
        maybe_enable_async_tp(parallelism, compile_config, parallel_dims.get_mesh("tp"))

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    if ac_config.mode != "none":
        apply_ac(model, ac_config)

    # turn on per-TransformerBlock compile after AC wrapping and before FSDP
    if model_compile_enabled:
        apply_compile(model, compile_config)

    if parallel_dims.fsdp_enabled:
        # apply FSDP or HSDP, potentially with Context Parallel
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "fsdp")
        else:
            dp_mesh_dim_names = ("fsdp",)

        apply_fsdp(
            model,
            parallel_dims.get_mesh(list(dp_mesh_dim_names)),
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=training.enable_cpu_offload,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

        if parallel_dims.cp_enabled:
            model.set_cp_mesh(parallel_dims.get_mesh("cp"))
            logger.info("Applied Context Parallel to the model")

        if training.enable_cpu_offload:
            logger.info("Applied CPU Offloading to the model")
    elif parallel_dims.dp_replicate_enabled:
        dp_replicate_mesh = parallel_dims.get_mesh("dp_replicate")
        if parallel_dims.world_size != dp_replicate_mesh.size():
            raise RuntimeError("DDP has not supported > 1D parallelism")
        apply_ddp(
            model,
            dp_replicate_mesh,
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

    # skipping nn.Identity modules (which are added by pipeline parallelism for unused modules)
    root_plan = {}

    if hasattr(model, "tok_embeddings"):
        if isinstance(model.tok_embeddings, nn.Identity):
            root_plan["tok_embeddings"] = NoParallel()
        else:
            root_plan["tok_embeddings"] = RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            )

    if hasattr(model, "norm"):
        if isinstance(model.norm, nn.Identity):
            root_plan["norm"] = NoParallel()
        else:
            root_plan["norm"] = SequenceParallel()

    if hasattr(model, "output"):
        if isinstance(model.output, nn.Identity):
            root_plan["output"] = NoParallel()
        else:
            root_plan["output"] = ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if loss_parallel else Replicate(),
                use_local_output=not loss_parallel,
            )
    if root_plan:  # Only call if there's something to parallelize
        parallelize_module(model, tp_mesh, root_plan)

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
    for transformer_block in model.layers:
        layer_plan = {
            "input_layernorm": SequenceParallel(),
            "self_attn": prepare_module_input(
                input_kwarg_layouts={"hidden_states": Shard(1)},
                desired_input_kwarg_layouts={"hidden_states": Replicate()},
            ),
            "post_attention_layernorm": SequenceParallel(),
        }

        if getattr(transformer_block.self_attn, "q_lora_rank", None) is None:
            layer_plan.update(
                {
                    "self_attn.q_proj": colwise_parallel(),
                    "self_attn.k_proj": colwise_parallel(),
                    "self_attn.v_proj": colwise_parallel(),
                }
            )
        else:
            layer_plan.update(
                {
                    "self_attn.q_a_proj": NoParallel(),
                    "self_attn.q_a_layernorm": NoParallel(),
                    "self_attn.q_b_proj": colwise_parallel(),
                    "self_attn.kv_a_proj_with_mqa": NoParallel(),
                    "self_attn.kv_a_layernorm": NoParallel(),
                    "self_attn.kv_b_proj": colwise_parallel(),
                }
            )

        # Handle different names for the output projection layer, e.g. o_proj vs dense
        o_proj_name = (
            "o_proj" if hasattr(transformer_block.self_attn, "o_proj") else "dense"
        )
        layer_plan[f"self_attn.{o_proj_name}"] = rowwise_parallel(
            output_layouts=Shard(1)
        )
        # For model that uses RMSNorm on Q and K (i.e. Qwen3)
        if hasattr(transformer_block.self_attn, "q_norm") and hasattr(
            transformer_block.self_attn, "k_norm"
        ):
            layer_plan["self_attn.q_norm"] = SequenceParallel(
                sequence_dim=2, use_local_output=True
            )
            layer_plan["self_attn.k_norm"] = SequenceParallel(
                sequence_dim=2, use_local_output=True
            )

        if not transformer_block.moe_enabled:
            mlp_plan = {
                "mlp": prepare_module_input(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                ),
            }
            # Handle different names for MLP layers, e.g. gate_proj vs fc1
            gate_proj_name = (
                "gate_proj" if hasattr(transformer_block.mlp, "gate_proj") else "fc1"
            )
            mlp_plan[f"mlp.{gate_proj_name}"] = colwise_parallel()

            if hasattr(transformer_block.mlp, "up_proj"):
                mlp_plan["mlp.up_proj"] = colwise_parallel()

            down_proj_name = (
                "down_proj" if hasattr(transformer_block.mlp, "down_proj") else "fc2"
            )
            mlp_plan[f"mlp.{down_proj_name}"] = rowwise_parallel(
                output_layouts=Shard(1)
            )
            layer_plan.update(mlp_plan)

        # Some models like Phi-2 don't have post_attention_layernorm
        if not hasattr(transformer_block, "post_attention_layernorm"):
            layer_plan.pop("post_attention_layernorm")

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

    for transformer_block in model.layers:
        # NOTE: When EP is enabled, In an MoE layer, we use the following FSDP wrapping
        # - the router and the shared experts are sharded together with the TransformerBlock
        # - the routed experts are sharded with the remaining dp_mod_ep_mesh
        if (
            hasattr(transformer_block, "moe_enabled")
            and transformer_block.moe_enabled
            and ep_degree > 1
        ):
            fsdp_mod_ep_config = fsdp_config.copy()
            fsdp_mod_ep_config["mesh"] = dp_mod_ep_mesh
            moe_block = transformer_block.mlp
            # NOTE: EP alreadys shards the routed experts on dim 0 (num_experts).
            #       When dp_mod_ep * ep > num_experts, FSDP default dim-0 sharding
            #       causes inefficiency, so we choose to do FSDP sharding on dim-1.
            #       Even when EP is not used, we may still want to shard the experts
            #       on non-0 dim. For now it may not be worth the complexity to support
            #       shard_placement_fn on the outer TransformerBlock-level FSDP.
            _experts_shard_placement_fn = None
            assert dp_mod_ep_mesh is not None
            if dp_mod_ep_mesh.size() * ep_degree > moe_block.experts.num_experts:
                _experts_shard_placement_fn = lambda param: Shard(1)

            fully_shard(
                moe_block.experts,
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
    transformer_blocks = list(model.layers.values())
    next_transformer_blocks = transformer_blocks[1:] + [None]

    if model.tok_embeddings is not None and model.layers is not None:
        model.tok_embeddings.set_modules_to_forward_prefetch([transformer_blocks[0]])

    for transformer_block, next_transformer_block in zip(
        transformer_blocks, next_transformer_blocks
    ):
        if next_transformer_block is not None:
            if next_transformer_block.moe_enabled:
                transformer_block.set_modules_to_forward_prefetch(
                    [next_transformer_block, next_transformer_block.mlp.experts]
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

    if model.norm is not None and model.output is not None and model.layers is not None:
        model.output.set_modules_to_backward_prefetch([reversed_transformer_blocks[0]])

    for transformer_block, prev_transformer_block in zip(
        reversed_transformer_blocks, prev_transformer_blocks
    ):
        if prev_transformer_block is not None:
            if prev_transformer_block.moe_enabled:
                transformer_block.set_modules_to_backward_prefetch(
                    [prev_transformer_block, prev_transformer_block.mlp.experts]
                )
            else:
                transformer_block.set_modules_to_backward_prefetch(
                    [prev_transformer_block]
                )
        elif model.tok_embeddings is not None:
            transformer_block.set_modules_to_backward_prefetch([model.tok_embeddings])

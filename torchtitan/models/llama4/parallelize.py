# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Any

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import Partial, Replicate, Shard
from torch.distributed.tensor.parallel import (
    parallelize_module,
    PrepareModuleInputOutput,
    RowwiseParallel,
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
from torchtitan.distributed.context_parallel import apply_cp_to_forward
from torchtitan.distributed.expert_parallel import ExpertParallel, TensorParallel
from torchtitan.distributed.fsdp import (
    disable_fsdp_gradient_division,
    get_fsdp_reshard_after_forward_policy,
)
from torchtitan.distributed.tensor_parallel import (
    ColwiseParallelWithGradPlacement,
    maybe_enable_async_tp,
    NoParallel,
)
from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher
from torchtitan.models.llama4.model import Llama4Model
from torchtitan.tools.logging import logger


def parallelize_llama(
    model: Llama4Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
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
    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    # CP: wrap inner attention forward BEFORE parallelize() so CP logic
    # runs inside the local_map boundary on local tensors.
    if parallel_dims.cp_enabled:
        apply_cp_to_forward(
            # pyrefly: ignore [missing-attribute, not-callable]
            [block.attention.inner_attention for block in model.layers.values()],
            parallel_dims.get_mesh("cp"),
        )

    tp_mesh = None
    if parallel_dims.tp_enabled:
        tp_mesh = parallel_dims.get_mesh("tp")
        model.parallelize(tp_mesh)
        maybe_enable_async_tp(parallelism, compile_config, tp_mesh)

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        apply_moe_ep_tp(
            model,
            tp_mesh=parallel_dims.get_optional_mesh("tp"),
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
            enable_sp=True,
        )

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
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
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config: dict[str, Any] = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, pp_enabled
    )

    if getattr(model, "enable_weight_tying", False):
        modules = [
            m
            for m in (model.tok_embeddings, model.norm, model.lm_head)
            if m is not None
        ]
        # pyrefly: ignore [no-matching-overload]
        fully_shard(
            modules,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )
    else:
        if model.tok_embeddings is not None:
            # pyrefly: ignore [no-matching-overload]
            fully_shard(
                model.tok_embeddings,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
            )
        # As an optimization, do not reshard_after_forward the last layers
        # by default since FSDP would prefetch them immediately.
        if model.norm is not None and model.lm_head is not None:
            # pyrefly: ignore [no-matching-overload]
            fully_shard(
                [model.norm, model.lm_head],
                **fsdp_config,
                reshard_after_forward=reshard_after_forward_policy == "always",
            )

    # pyrefly: ignore [missing-attribute]
    for layer_id, transformer_block in model.layers.items():
        # NOTE: In an MoE layer, we use shard_placement_fn to apply different
        # FSDP mesh and shard placement to different parameters:
        # - When EP > 1: routed experts use edp_mesh, other params use dp_mesh
        # - When EP = 1: all params use the same FSDP mesh, but experts may
        #   use Shard(1) when FSDP degree > num_experts to avoid padding
        if transformer_block.moe_enabled:
            assert hasattr(transformer_block, "moe")
            expert_params = set(transformer_block.moe.experts.parameters())
            num_experts = transformer_block.moe.experts.num_experts

            if ep_degree > 1:
                assert edp_mesh is not None
                efsdp_ep_size = edp_mesh["efsdp"].size() * ep_degree
            else:
                efsdp_ep_size = fsdp_config["mesh"].size()

            if efsdp_ep_size > num_experts:
                expert_shard_placement = Shard(1)
            else:
                expert_shard_placement = Shard(0)

            # When ep_degree == 1 and no Shard(1) override needed, skip
            # shard_placement_fn entirely for simplicity
            if ep_degree == 1 and expert_shard_placement == Shard(0):
                fully_shard(
                    transformer_block,
                    **fsdp_config,
                    reshard_after_forward=reshard_after_forward,
                )
            elif ep_degree == 1:
                # ep_degree == 1 but need Shard(1) for experts to avoid padding
                def _experts_shard_placement_fn(
                    param: nn.Parameter,
                    _expert_params: set = expert_params,
                ) -> Shard | None:
                    if param in _expert_params:
                        return Shard(1)
                    return None

                fully_shard(
                    transformer_block,
                    **fsdp_config,
                    reshard_after_forward=reshard_after_forward,
                    shard_placement_fn=_experts_shard_placement_fn,
                )
            else:
                # ep_degree > 1: per-param mesh
                from torch.distributed.fsdp._fully_shard._fsdp_common import (
                    FSDPMeshInfo,
                    ShardPlacementResult,
                )

                assert edp_mesh is not None
                edp_mesh_info = FSDPMeshInfo(mesh=edp_mesh, shard_mesh_dim=0)
                dp_mesh_info = FSDPMeshInfo(mesh=dp_mesh, shard_mesh_dim=0)

                def _shard_placement_fn(
                    param: nn.Parameter,
                    _expert_params: set = expert_params,
                    _expert_placement: Shard = expert_shard_placement,
                    _edp_mesh_info: FSDPMeshInfo = edp_mesh_info,
                    _dp_mesh_info: FSDPMeshInfo = dp_mesh_info,
                ) -> ShardPlacementResult:
                    if param in _expert_params:
                        return ShardPlacementResult(
                            placement=_expert_placement, mesh_info=_edp_mesh_info
                        )
                    else:
                        return ShardPlacementResult(
                            placement=Shard(0), mesh_info=_dp_mesh_info
                        )

                fully_shard(
                    transformer_block,
                    **fsdp_config,
                    reshard_after_forward=reshard_after_forward,
                    shard_placement_fn=_shard_placement_fn,
                )
        else:
            fully_shard(
                transformer_block,
                **fsdp_config,
                reshard_after_forward=reshard_after_forward,
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
            transformer_block.set_modules_to_forward_prefetch([next_transformer_block])
        elif model.norm is not None and model.lm_head is not None:
            # pyrefly: ignore [missing-attribute]
            transformer_block.set_modules_to_forward_prefetch(
                [model.norm, model.lm_head]
            )

    # backward
    # pyrefly: ignore [not-callable]
    reversed_transformer_blocks = list(reversed(model.layers.values()))
    prev_transformer_blocks = reversed_transformer_blocks[1:] + [None]

    # pyrefly: ignore [bad-argument-type]
    if model.norm is not None and model.lm_head is not None and len(model.layers) > 0:
        # pyrefly: ignore [missing-attribute]
        model.lm_head.set_modules_to_backward_prefetch([reversed_transformer_blocks[0]])

    for transformer_block, prev_transformer_block in zip(
        reversed_transformer_blocks, prev_transformer_blocks
    ):
        if prev_transformer_block is not None:
            # pyrefly: ignore [missing-attribute]
            transformer_block.set_modules_to_backward_prefetch([prev_transformer_block])
        elif model.tok_embeddings is not None:
            # pyrefly: ignore [missing-attribute]
            transformer_block.set_modules_to_backward_prefetch([model.tok_embeddings])


def apply_moe_ep_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh | None,
    ep_mesh: DeviceMesh | None,
    enable_sp: bool = True,
):
    """
    Apply MoE expert/tensor parallelism plans to MoE-enabled transformer blocks.

    Args:
        model: Model containing transformer blocks.
        tp_mesh: Tensor-parallel device mesh.
        ep_mesh: Expert-parallel device mesh.
        enable_sp: Whether sequence parallelism is enabled.
    """
    assert ep_mesh is not None or tp_mesh is not None

    sp_layout = Shard(1) if enable_sp else Replicate()

    # pyrefly: ignore [not-callable]
    for transformer_block in model.layers.values():
        # pyrefly: ignore [missing-attribute]
        if not transformer_block.moe_enabled:
            continue

        if tp_mesh is not None:
            moe_layer_plan = {
                "moe": PrepareModuleInputOutput(
                    input_layouts=(sp_layout,),
                    desired_input_layouts=(Replicate(),),
                    use_local_input=False,
                    # MoE always runs SP-style token splitting on the TP mesh
                    # whenever a TP mesh exists (the dispatcher's
                    # ``_split_along_sp`` runs whenever ``sp_size > 1``,
                    # independent of ``enable_sp``). Each rank's combine output
                    # is therefore its SP chunk inside a full-sized buffer with
                    # zeros elsewhere — i.e. ``Partial`` — and the conversion
                    # to ``desired_output_layouts`` is reduce-scatter (when
                    # dense uses SP) or all-reduce (when dense doesn't).
                    # ``enable_sp`` only controls the dense part's layout.
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
            # pyrefly: ignore [missing-attribute]
            if transformer_block.moe.shared_experts is not None:
                # Use ColwiseParallelWithGradPlacement to keep d_x as Partial in
                # backward (avoids the all-reduce that from_local(Replicate)
                # would otherwise trigger). For w2, output_layouts=Partial()
                # skips the Partial→Replicate all-reduce in forward. The
                # reduction happens once at the MoE output boundary
                # (PrepareModuleInputOutput).
                # pyrefly: ignore [no-matching-overload]
                moe_layer_plan.update(
                    {
                        "moe.shared_experts.w1": ColwiseParallelWithGradPlacement(
                            local_input_grad_placements=(Partial(),)
                        ),
                        "moe.shared_experts.w2": RowwiseParallel(
                            output_layouts=Partial(),
                        ),
                        "moe.shared_experts.w3": ColwiseParallelWithGradPlacement(
                            local_input_grad_placements=(Partial(),)
                        ),
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
        # EP disabled: shard routed expert weights across TP mesh (input Replicate, produces Partial output reduced at MoE boundary)
        if ep_mesh is None:
            experts_mesh = tp_mesh
            experts_plan = TensorParallel()
        else:
            experts_mesh = ep_mesh
            experts_plan = ExpertParallel()
            # pyrefly: ignore [missing-attribute]
            dispatcher = transformer_block.moe.experts.token_dispatcher
            if tp_mesh is not None:
                if isinstance(dispatcher, AllToAllTokenDispatcher):
                    # sp_size and sp_rank are set for sequence-parallel token splitting
                    # when EP borrows from TP.
                    dispatcher.sp_size = tp_mesh.size()
                    # Use _sym_get_coordinate so the rank is a SymInt
                    # under CooR precompile instead of a concrete int
                    # that gets baked into the FX graph.
                    dispatcher.sp_rank = tp_mesh._sym_get_coordinate(0)

        if experts_mesh is not None and experts_plan is not None:
            parallelize_module(
                # pyrefly: ignore [missing-attribute]
                module=transformer_block.moe.experts,
                device_mesh=experts_mesh,
                parallelize_plan=experts_plan,
            )

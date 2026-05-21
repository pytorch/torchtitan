# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import DTensor, Replicate, Shard
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
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp, NoParallel
from torchtitan.experiments.transformers_modeling_backend.compile import (
    apply_compile_sparse,
)
from torchtitan.models.llama3.parallelize import disable_fsdp_gradient_division
from torchtitan.tools.logging import logger

# ---------------------------------------------------------------------------
# Main parallelization entry point
# ---------------------------------------------------------------------------


def parallelize_hf_transformers(
    model: nn.Module,
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
        apply_non_moe_tp(
            model,
            parallel_dims.get_mesh("tp"),
            enable_loss_parallel=not parallelism.disable_loss_parallel,
        )
        maybe_enable_async_tp(parallelism, compile_config, parallel_dims.get_mesh("tp"))

    # Build native MoE modules, apply sharding, and swap into model
    if any(getattr(b, "moe_enabled", False) for b in model.layers):
        from torchtitan.experiments.transformers_modeling_backend.moe_replacement import (
            build_and_swap_native_moe,
        )

        build_and_swap_native_moe(model, parallel_dims)

    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    if ac_config.mode != "none":
        apply_ac(model, ac_config)

    # turn on per-TransformerBlock compile after AC wrapping and before FSDP
    if model_compile_enabled:
        has_moe = any(getattr(b, "moe_enabled", False) for b in model.layers)
        if has_moe:
            apply_compile_sparse(model, compile_config)
        else:
            apply_compile(model, compile_config)

    if parallel_dims.fsdp_enabled or parallel_dims.ep_enabled:
        # apply FSDP or HSDP, potentially with Context Parallel
        if parallel_dims.dp_replicate_enabled:
            dp_mesh_dim_names = ("dp_replicate", "fsdp")
        else:
            dp_mesh_dim_names = ("fsdp",)

        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

        apply_fsdp(
            model,
            parallel_dims.get_mesh(list(dp_mesh_dim_names)),
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=training.enable_cpu_offload,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            ep_degree=parallel_dims.ep,
            dp_mod_ep_mesh=edp_mesh,
        )

    if training.enable_cpu_offload:
        logger.info("Applied CPU Offloading to the model")

    if parallel_dims.cp_enabled:
        model.set_cp_mesh(parallel_dims.get_mesh("cp"))
        logger.info("Applied Context Parallel to the model")

    return model


# ---------------------------------------------------------------------------
# TP for non-MoE layers + gate handling for MoE layers
# ---------------------------------------------------------------------------


def apply_non_moe_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    enable_loss_parallel: bool,
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
            root_plan["tok_embeddings"] = NoParallel(use_local_output=True)
        else:
            root_plan["tok_embeddings"] = RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            )

    if hasattr(model, "norm"):
        if isinstance(model.norm, nn.Identity):
            root_plan["norm"] = NoParallel(use_local_output=True)
        else:
            root_plan["norm"] = SequenceParallel()

    if hasattr(model, "lm_head"):
        if isinstance(model.lm_head, nn.Identity):
            root_plan["lm_head"] = NoParallel(use_local_output=True)
        else:
            root_plan["lm_head"] = ColwiseParallel(
                input_layouts=Shard(1),
                output_layouts=Shard(-1) if enable_loss_parallel else Replicate(),
                use_local_output=not enable_loss_parallel,
            )
    if root_plan:  # Only call if there's something to parallelize
        parallelize_module(model, tp_mesh, root_plan)

    # Apply tensor + sequence parallelism to every transformer block
    for transformer_block in model.layers:
        layer_plan = {
            "input_layernorm": SequenceParallel(),
            "self_attn": PrepareModuleInput(
                input_kwarg_layouts={"hidden_states": Shard(1)},
                desired_input_kwarg_layouts={"hidden_states": Replicate()},
            ),
            "post_attention_layernorm": SequenceParallel(),
        }

        if getattr(transformer_block.self_attn, "q_lora_rank", None) is None:
            layer_plan.update(
                {
                    "self_attn.q_proj": ColwiseParallel(),
                    "self_attn.k_proj": ColwiseParallel(),
                    "self_attn.v_proj": ColwiseParallel(),
                }
            )
        else:
            layer_plan.update(
                {
                    "self_attn.q_a_proj": NoParallel(use_local_output=True),
                    "self_attn.q_a_layernorm": NoParallel(use_local_output=True),
                    "self_attn.q_b_proj": ColwiseParallel(),
                    "self_attn.kv_a_proj_with_mqa": NoParallel(use_local_output=True),
                    "self_attn.kv_a_layernorm": NoParallel(use_local_output=True),
                    "self_attn.kv_b_proj": ColwiseParallel(),
                }
            )

        # Handle different names for the output projection layer, e.g. o_proj vs dense
        o_proj_name = (
            "o_proj" if hasattr(transformer_block.self_attn, "o_proj") else "dense"
        )
        layer_plan[f"self_attn.{o_proj_name}"] = RowwiseParallel(
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

        # GLM-5 DSA (Dense Sparse Attention) indexer: its inputs arrive as
        # plain tensors (from upstream NoParallel/to_local), but after FSDP
        # unshard its weights become DTensors — causing mixed tensor errors.
        # Shadow DTensor params with local copies via __dict__ on each
        # sub-module, same pattern as experts_to_local hooks.
        # The indexer is @torch.no_grad so no grad concerns.
        if hasattr(transformer_block.self_attn, "indexer"):
            indexer = transformer_block.self_attn.indexer

            # The indexer's inputs (hidden_states, q_resid) are
            # DTensor(Replicate) from PrepareModuleInput on self_attn,
            # but its weights are plain Parameters (FSDP manages them at
            # block level). Convert DTensor inputs to local so F.linear
            # doesn't hit mixed DTensor/Tensor errors.
            def _indexer_to_local_pre_hook(module, args):
                def _to_local(x):
                    if isinstance(x, DTensor):
                        return x.to_local()
                    if isinstance(x, tuple):
                        return tuple(_to_local(v) for v in x)
                    return x

                return tuple(_to_local(a) for a in args)

            indexer.register_forward_pre_hook(_indexer_to_local_pre_hook)

        if not transformer_block.moe_enabled:
            mlp_plan = {
                "mlp": PrepareModuleInput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                ),
            }
            # Handle different names for MLP layers, e.g. gate_proj vs fc1
            gate_proj_name = (
                "gate_proj" if hasattr(transformer_block.mlp, "gate_proj") else "fc1"
            )
            mlp_plan[f"mlp.{gate_proj_name}"] = ColwiseParallel()

            if hasattr(transformer_block.mlp, "up_proj"):
                mlp_plan["mlp.up_proj"] = ColwiseParallel()

            down_proj_name = (
                "down_proj" if hasattr(transformer_block.mlp, "down_proj") else "fc2"
            )
            mlp_plan[f"mlp.{down_proj_name}"] = RowwiseParallel(output_layouts=Shard(1))
            layer_plan.update(mlp_plan)
        # MoE layers: no mlp plan here. All MoE parallelism
        # (EP, TP, boundary handling) is in apply_moe_sharding.

        # Some models like Phi-2 don't have post_attention_layernorm
        if not hasattr(transformer_block, "post_attention_layernorm"):
            layer_plan.pop("post_attention_layernorm")

        parallelize_module(
            module=transformer_block,
            device_mesh=tp_mesh,
            parallelize_plan=layer_plan,
        )

    logger.info("Applied Tensor Parallelism to the model")


# ---------------------------------------------------------------------------
# FSDP
# ---------------------------------------------------------------------------


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
):
    """Apply data parallelism (via FSDP2) to the model.

    When EP is enabled (``ep_degree > 1``), uses flat FSDP with
    ``shard_placement_fn`` to route expert params to ``dp_mod_ep_mesh``
    and other params to ``dp_mesh`` within a single ``fully_shard`` call
    per transformer block — matching native titan's approach and avoiding
    nested FSDP hooks that cause SAC op-count mismatches during recompute.

    Args:
        model: The model to apply data parallelism to.
        dp_mesh: The device mesh for data parallelism (FSDP or HSDP).
        param_dtype: Data type for model parameters.
        reduce_dtype: Data type for gradient reduction.
        pp_enabled: Whether pipeline parallelism is enabled.
        cpu_offload: Whether to offload model parameters to CPU.
        reshard_after_forward_policy: Resharding policy after forward pass.
            "default", "always", or "never".
        ep_degree: Expert parallelism degree (1 = no EP).
        dp_mod_ep_mesh: DP mesh for expert params when EP is enabled.
            Required when ``ep_degree > 1``.
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config = {"mesh": dp_mesh, "mp_policy": mp_policy}
    if cpu_offload:
        fsdp_config["offload_policy"] = CPUOffloadPolicy()

    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        reshard_after_forward_policy, pp_enabled
    )

    if model.tok_embeddings is not None:
        fully_shard(
            model.tok_embeddings,
            **fsdp_config,
            reshard_after_forward=reshard_after_forward,
        )

    for transformer_block in model.layers:
        # NOTE: When EP is enabled, we use shard_placement_fn to route
        # expert params to dp_mod_ep_mesh and other params to dp_mesh,
        # all within a single fully_shard call (flat FSDP, no nesting).
        # This avoids nested FSDP hooks on the experts module which
        # cause SAC op-count mismatches during recompute.
        if (
            hasattr(transformer_block, "moe_enabled")
            and transformer_block.moe_enabled
            and ep_degree > 1
        ):
            from torch.distributed.fsdp._fully_shard._fsdp_common import (
                FSDPMeshInfo,
                HSDPMeshInfo,
                ShardPlacementResult,
            )

            assert dp_mod_ep_mesh is not None
            experts = transformer_block.mlp.experts
            expert_params = set(experts.parameters())
            num_local_experts = experts.num_experts // ep_degree

            if dp_mod_ep_mesh.size() > num_local_experts:
                expert_shard_placement = Shard(1)
            else:
                expert_shard_placement = Shard(0)

            def _get_fsdp_mesh_info(mesh: DeviceMesh) -> FSDPMeshInfo:
                if mesh.ndim == 1:
                    return FSDPMeshInfo(mesh=mesh, shard_mesh_dim=0)
                if mesh.ndim == 2:
                    return HSDPMeshInfo(
                        mesh=mesh, replicate_mesh_dim=0, shard_mesh_dim=1
                    )
                raise ValueError(f"Expected 1D or 2D FSDP mesh, got {mesh.ndim}D mesh.")

            edp_mesh_info = _get_fsdp_mesh_info(dp_mod_ep_mesh)
            dp_mesh_info = _get_fsdp_mesh_info(dp_mesh)

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
                return ShardPlacementResult(placement=Shard(0), mesh_info=_dp_mesh_info)

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

    # As an optimization, do not reshard_after_forward the last layers by default
    # since FSDP would prefetch them immediately after the forward pass
    if model.norm is not None and model.lm_head is not None:
        fully_shard(
            [model.norm, model.lm_head],
            **fsdp_config,
            reshard_after_forward=reshard_after_forward_policy == "always",
        )

    fully_shard(model, **fsdp_config)

    # Disable FSDP's automatic gradient division for all FSDP modules
    disable_fsdp_gradient_division(model)

    # NOTE: set up explicit prefetching when EP is enabled, as D2H syncs
    # in EP could interfere with implicit prefetching in FSDP.
    # With flat FSDP (no nested experts group), prefetching targets
    # transformer blocks only — expert params are part of the block's
    # FSDPState and prefetched together with the block.
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
            transformer_block.set_modules_to_forward_prefetch([next_transformer_block])
        elif model.norm is not None and model.lm_head is not None:
            transformer_block.set_modules_to_forward_prefetch(
                [model.norm, model.lm_head]
            )

    # backward
    reversed_transformer_blocks = list(reversed(model.layers.values()))
    prev_transformer_blocks = reversed_transformer_blocks[1:] + [None]

    if (
        model.norm is not None
        and model.lm_head is not None
        and model.layers is not None
    ):
        model.lm_head.set_modules_to_backward_prefetch([reversed_transformer_blocks[0]])

    for transformer_block, prev_transformer_block in zip(
        reversed_transformer_blocks, prev_transformer_blocks
    ):
        if prev_transformer_block is not None:
            transformer_block.set_modules_to_backward_prefetch([prev_transformer_block])
        elif model.tok_embeddings is not None:
            transformer_block.set_modules_to_backward_prefetch([model.tok_embeddings])

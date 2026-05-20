# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import (
    distribute_tensor,
    DTensor,
    Partial,
    Replicate,
    Shard,
)
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
from torchtitan.protocols.sharding import NamedPlacement, resolve_placements
from torchtitan.protocols.types import MeshAxisName

EP = MeshAxisName.EP
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp, NoParallel
from torchtitan.experiments.transformers_modeling_backend.compile import (
    apply_compile_sparse,
)
from torchtitan.experiments.transformers_modeling_backend.token_dispatcher import (
    HFTokenDispatcher,
)
from torchtitan.models.llama3.parallelize import disable_fsdp_gradient_division
from torchtitan.tools.logging import logger

# ---------------------------------------------------------------------------
# Config-based sharding for HF modules
# ---------------------------------------------------------------------------


def shard_module_states(
    module: nn.Module,
    state_shardings: dict[str, NamedPlacement],
    parallel_dims: ParallelDims,
) -> None:
    """Apply declarative state shardings to any ``nn.Module``.

    Mirrors ``Module._shard_states`` from the titan protocol but works on
    plain ``nn.Module`` subclasses (e.g. HF transformers modules) that
    don't implement the ``Module`` protocol.

    Unlike ``Module._shard_states``, undeclared params are left unsharded
    rather than raising — HF modules may carry params we don't need to
    distribute.
    """
    for mod in [module, *module.children()]:
        for name, param in list(mod.named_parameters(recurse=False)):
            named_placements = state_shardings.get(name)
            if named_placements is None:
                continue
            mesh = parallel_dims.resolve_mesh(named_placements.keys())
            if mesh is None:
                continue
            placements = resolve_placements(named_placements, mesh)
            if isinstance(param, DTensor):
                continue
            mod.register_parameter(
                name,
                nn.Parameter(
                    distribute_tensor(param, mesh, list(placements)),
                    requires_grad=param.requires_grad,
                ),
            )



# ---------------------------------------------------------------------------
# Gate and expert hooks for DTensor → local conversion
# ---------------------------------------------------------------------------


def _make_moe_to_local_pre_hook(grad_placements):
    """Create a pre-hook that converts DTensor input to local.

    Args:
        grad_placements: Gradient placements for to_local(). Use
            (Partial(),) for TP-only (output is partial sum from
            TP-sharded experts), (Shard(1),) for EP+TP (each TP rank
            processes its local shard independently through EP).
    """

    def hook(module, args):
        hidden_states = args[0]
        if isinstance(hidden_states, DTensor):
            # clone() is needed because to_local() returns a view from a
            # custom autograd Function (_ToTorchTensorBackward). Some HF
            # models apply in-place ops which autograd forbids on custom
            # Function views.
            return (hidden_states.to_local(grad_placements=grad_placements).clone(),)
        return None

    return hook


def _experts_to_local_pre_hook(module, args):
    """Convert DTensor expert params to local for the HF for-loop.

    FSDP (and EP via distribute_tensor) makes expert params DTensors.
    The original HF Experts.forward uses these params in a for-loop
    with F.linear and index_add_ which can't handle DTensors. Native
    titan solves this with to_local() inside GroupedExperts.forward;
    we use __dict__ shadowing since we don't modify the HF forward.

    Python checks instance __dict__ before nn.Module's __getattr__
    (which accesses _parameters), so self.gate_up_proj in the forward
    finds the local tensor instead of the DTensor parameter.
    """
    gate_up = module.gate_up_proj
    down = module.down_proj
    if isinstance(gate_up, DTensor):
        module.__dict__["gate_up_proj"] = gate_up.to_local()
        module.__dict__["down_proj"] = down.to_local()
        module._saved_num_experts = module.num_experts
        module.num_experts = module.__dict__["gate_up_proj"].shape[0]
    return None


def _experts_restore_post_hook(module, args, output):
    """Restore DTensor expert params and num_experts after the HF forward."""
    for key in ("gate_up_proj", "down_proj"):
        module.__dict__.pop(key, None)
    if hasattr(module, "_saved_num_experts"):
        module.num_experts = module._saved_num_experts
        del module._saved_num_experts
    return output


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
    assert training.seq_len % parallel_dims.seq_len_divisor == 0, f"""
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

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        apply_moe_sharding(
            model,
            parallel_dims=parallel_dims,
            tp_mesh=parallel_dims.get_optional_mesh("tp"),
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
        )

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

    # Register experts_to_local hooks AFTER apply_fsdp so they fire after
    # FSDP unshard. Native titan does to_local() inside GroupedExperts.forward;
    # we use __dict__ shadowing hooks since we don't modify the HF forward.
    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        for transformer_block in model.layers:
            if getattr(transformer_block, "moe_enabled", False):
                experts = transformer_block.mlp.experts
                experts.register_forward_pre_hook(_experts_to_local_pre_hook)
                experts.register_forward_hook(_experts_restore_post_hook, prepend=True)

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
# MoE parallelism: EP and TP-only
# ---------------------------------------------------------------------------


def apply_moe_sharding(
    model: nn.Module,
    parallel_dims: ParallelDims,
    tp_mesh: DeviceMesh | None = None,
    ep_mesh: DeviceMesh | None = None,
):
    """Apply Expert Parallelism and/or Tensor Parallelism to MoE layers.

    Shards MoE sub-module states using declarative ``NamedPlacement`` specs
    (from ``sharding.py``) via ``shard_module_states``, and registers
    HF-specific hooks for dispatch/combine and DTensor↔local conversion.

    Args:
        model: The model with MoE layers.
        parallel_dims: Parallel dimensions for mesh resolution.
        tp_mesh: TP device mesh for gate replication and expert TP sharding.
        ep_mesh: EP device mesh for expert EP sharding and dispatch/combine.
    """
    from torchtitan.experiments.transformers_modeling_backend.sharding import (
        expert_ep_shardings,
        expert_tp_shardings,
        gate_shardings,
        shared_expert_shardings,
    )

    assert tp_mesh is not None or ep_mesh is not None

    for transformer_block in model.layers:
        if not getattr(transformer_block, "moe_enabled", False):
            continue

        moe_block = transformer_block.mlp
        experts = moe_block.experts
        gate = getattr(moe_block, "gate", None) or getattr(moe_block, "router", None)

        # --- EP: shard experts on dim 0, register dispatch/combine ---
        if ep_mesh is not None:
            top_k = getattr(moe_block, "top_k", None) or getattr(
                moe_block, "num_experts_per_tok", 1
            )
            experts.token_dispatcher = HFTokenDispatcher.Config(
                num_experts=experts.num_experts,
                top_k=top_k,
            ).build()

            shard_module_states(experts, expert_ep_shardings(), parallel_dims)

            ep_size = ep_mesh.size()
            experts.token_dispatcher.ep_group = ep_mesh.get_group()
            experts.token_dispatcher.ep_size = ep_size
            experts.token_dispatcher.num_local_experts = (
                experts.num_experts // ep_size
            )
            experts.register_forward_pre_hook(
                lambda mod, args: mod.token_dispatcher.dispatch(*args)
            )
            experts.register_forward_hook(
                lambda mod, args, output: mod.token_dispatcher.combine(output)
            )

        # --- TP: shard states, replicate gate, register boundary hooks ---
        if tp_mesh is not None:
            # Expert TP sharding (TP-only, not EP+TP)
            if ep_mesh is None:
                shard_module_states(experts, expert_tp_shardings(), parallel_dims)

            # Gate: replicate weights, wrap input as DTensor, output to local
            shard_module_states(gate, gate_shardings(), parallel_dims)
            parallelize_module(gate, tp_mesh, NoParallel(use_local_output=True))

            # Shadow DTensor gate buffers with local copies so
            # route_tokens_to_experts (which accesses buffers outside the
            # gate forward) doesn't hit mixed DTensor/Tensor errors.
            def _gate_buffers_to_local(gate_mod):
                def pre_hook(module, args):
                    for name, buf in gate_mod.named_buffers(recurse=False):
                        if isinstance(buf, DTensor):
                            gate_mod.__dict__[name] = buf.to_local()

                return pre_hook

            moe_block.register_forward_pre_hook(_gate_buffers_to_local(gate))

            # Shared experts: ColwiseParallel/RowwiseParallel TP sharding
            for shared_name in ("shared_expert", "shared_experts"):
                shared = getattr(moe_block, shared_name, None)
                if shared is not None:
                    shared_plan = {}
                    for name in ("gate_proj", "up_proj"):
                        if hasattr(shared, name):
                            shared_plan[name] = ColwiseParallel()
                    if hasattr(shared, "down_proj"):
                        shared_plan["down_proj"] = RowwiseParallel(
                            output_layouts=Partial()
                        )
                    if shared_plan:
                        parallelize_module(shared, tp_mesh, shared_plan)

            # Shared expert gate: replicate on TP mesh
            shared_gate = getattr(moe_block, "shared_expert_gate", None)
            if shared_gate is not None:
                parallelize_module(
                    shared_gate, tp_mesh, NoParallel(use_local_output=True)
                )

            # MoE block TP boundary
            if ep_mesh is None:
                # TP-only: all-gather input, reduce-scatter output
                parallelize_module(
                    moe_block,
                    tp_mesh,
                    PrepareModuleInputOutput(
                        input_layouts=(Shard(1),),
                        desired_input_layouts=(Replicate(),),
                        use_local_input=False,
                        output_layouts=(Partial(),),
                        desired_output_layouts=(Shard(1),),
                    ),
                )
                moe_block.register_forward_pre_hook(
                    _make_moe_to_local_pre_hook((Partial(),))
                )
            else:
                # EP+TP: each TP rank processes its SP shard through EP
                moe_block.register_forward_pre_hook(
                    _make_moe_to_local_pre_hook((Shard(1),))
                )

        # NOTE: experts_to_local hooks are registered in
        # parallelize_hf_transformers AFTER apply_fsdp, to ensure they
        # fire after FSDP unshard. See _experts_to_local_pre_hook docstring.

    logger.info("Applied MoE parallelism to the model")


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

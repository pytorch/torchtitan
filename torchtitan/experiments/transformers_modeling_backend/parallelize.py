# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from torch.distributed._functional_collectives import all_to_all_single_autograd
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointWrapper,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import CPUOffloadPolicy, fully_shard, MixedPrecisionPolicy
from torch.distributed.tensor import (
    distribute_module,
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

from torchtitan.components.quantization.float8 import find_float8_linear_config
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)

from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.compile import apply_compile_dense
from torchtitan.distributed.expert_parallel import BaseExpertParallel
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp, NoParallel
from torchtitan.models.llama3.parallelize import (
    apply_replicate,
    disable_fsdp_gradient_division,
)
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools.logging import logger


# ---------------------------------------------------------------------------
# HFExpertParallel: ParallelStyle for HF MoE experts
# ---------------------------------------------------------------------------


class HFExpertParallel(BaseExpertParallel):
    """Expert Parallelism for HF Transformers MoE models.

    Adapts torchtitan's BaseExpertParallel for the HF experts interface
    (hidden_states, top_k_index, top_k_weights). Shards expert params
    on dim 0, dispatches tokens via all-to-all based on routing decisions,
    and combines results after expert computation.

    Applied to the experts module via parallelize_module. The MoE block
    forward and experts forward run unchanged — dispatch/combine happen
    in input/output hooks registered by distribute_module.
    """

    def __init__(self):
        super().__init__()
        self._ep_group = None
        self._ep_size = None
        self._num_local_experts = None
        self._global_num_experts = None

    def _partition_fn(self, name, mod, device_mesh):
        """Shard expert params on dim 0 across EP ranks."""
        for param_name, param in list(mod.named_parameters(recurse=False)):
            mod.register_parameter(
                param_name,
                nn.Parameter(distribute_tensor(param, device_mesh, [Shard(0)])),
            )
        # Only set EP metadata on the top-level experts module
        # (distribute_module calls _partition_fn on children too)
        if hasattr(mod, "num_experts"):
            ep_size = device_mesh.size()
            self._ep_group = device_mesh.get_group()
            self._ep_size = ep_size
            self._global_num_experts = mod.num_experts
            self._num_local_experts = mod.num_experts // ep_size

    def _token_dispatch(self, mod, inputs, device_mesh):
        """Sort tokens by expert, all-to-all dispatch, build local routing."""
        hidden_states, top_k_index, top_k_weights = inputs
        ep_size = self._ep_size
        num_local = self._num_local_experts
        global_num_experts = self._global_num_experts
        ep_group = self._ep_group

        num_tokens = hidden_states.size(0)
        top_k = top_k_index.size(-1)

        # Flatten token-expert pairs
        token_idx = (
            torch.arange(num_tokens, device=hidden_states.device)
            .unsqueeze(1)
            .expand(-1, top_k)
            .reshape(-1)
        )
        expert_ids = top_k_index.reshape(-1)
        sample_weights = top_k_weights.reshape(-1)

        # Sort by expert
        perm = torch.argsort(expert_ids, stable=True)
        inv_perm = torch.empty_like(perm)
        inv_perm[perm] = torch.arange(perm.size(0), device=hidden_states.device)

        routed_input = hidden_states[token_idx[perm]]
        sorted_weights = sample_weights[perm]

        # Compute num_tokens_per_expert
        num_tokens_per_expert = torch.histc(
            expert_ids[perm].int(),
            bins=global_num_experts,
            min=0,
            max=global_num_experts - 1,
        )

        # Exchange per-expert token counts
        with torch.no_grad():
            input_splits_t = num_tokens_per_expert.view(ep_size, num_local).sum(dim=1)
            output_splits_t = torch.empty_like(input_splits_t)
            torch.distributed.all_to_all_single(
                output_splits_t, input_splits_t, group=ep_group
            )
            self._input_splits = input_splits_t.int().tolist()
            self._output_splits = output_splits_t.int().tolist()

        # Dispatch tokens
        dispatched = all_to_all_single_autograd(
            routed_input, self._output_splits, self._input_splits, ep_group
        )

        # Dispatch routing weights alongside tokens
        dispatched_weights = all_to_all_single_autograd(
            sorted_weights.unsqueeze(-1),
            self._output_splits,
            self._input_splits,
            ep_group,
        ).squeeze(-1)

        # Exchange per-expert counts for local routing
        with torch.no_grad():
            local_ntpe_tensor = torch.empty_like(num_tokens_per_expert)
            torch.distributed.all_to_all_single(
                local_ntpe_tensor, num_tokens_per_expert, group=ep_group
            )

        # Build local mock routing (matches by-source-rank token ordering)
        local_ntpe_per_source = local_ntpe_tensor.view(ep_size, num_local)
        local_expert_indices = torch.repeat_interleave(
            torch.arange(num_local, device=hidden_states.device).repeat(ep_size),
            local_ntpe_per_source.reshape(-1).long(),
        )
        mock_top_k_index = local_expert_indices.unsqueeze(1)
        mock_top_k_weights = dispatched_weights.unsqueeze(1)

        # Save state for combine
        self._inv_perm = inv_perm
        self._num_tokens = num_tokens
        self._top_k = top_k

        return dispatched, mock_top_k_index, mock_top_k_weights

    def _token_combine(self, mod, output, device_mesh):
        """All-to-all combine, unsort, sum across top_k."""
        combined = all_to_all_single_autograd(
            output,
            self._input_splits,  # reversed
            self._output_splits,  # reversed
            self._ep_group,
        )
        combined = combined[self._inv_perm]
        return combined.view(self._num_tokens, self._top_k, -1).sum(dim=1)

    def _apply(self, module, device_mesh):
        return distribute_module(
            module,
            device_mesh,
            partition_fn=self._partition_fn,
            input_fn=self._token_dispatch,
            output_fn=self._token_combine,
        )


# ---------------------------------------------------------------------------
# Gate and expert hooks for DTensor → local conversion
# ---------------------------------------------------------------------------


def _replicate_gate_params(moe_block: nn.Module, tp_mesh: DeviceMesh):
    """Replicate MoE router gate parameters on the TP mesh.

    Ensures gate params become DTensors with Replicate placement on the
    TP mesh, so that after FSDP wrapping they end up on the same
    (fsdp, tp) mesh as other non-EP params.
    """
    if not hasattr(moe_block, "gate"):
        return
    for submod in moe_block.gate.modules():
        for param_name, param in list(submod.named_parameters(recurse=False)):
            submod.register_parameter(
                param_name,
                nn.Parameter(distribute_tensor(param, tp_mesh, [Replicate()])),
            )


def _gate_to_local_pre_hook(module, args):
    """Shadow gate weight with local tensor for the HF router forward.

    The gate weight is a Replicate DTensor on the TP mesh (for FSDP mesh
    alignment). Shadowing with to_local() ensures F.linear produces a
    plain tensor output, so topk never sees DTensors (avoiding the
    aten.scatter.src backward crash).
    """
    weight = module.weight
    if isinstance(weight, DTensor):
        module.__dict__["weight"] = weight.to_local()
    return None


def _gate_restore_post_hook(module, args, output):
    """Restore gate DTensor weight after the forward."""
    module.__dict__.pop("weight", None)
    return output


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
            return (hidden_states.to_local(grad_placements=grad_placements),)
        return None

    return hook


def _experts_to_local_pre_hook(module, args):
    """Convert DTensor expert params to local for the HF forward.

    Uses __dict__ shadowing for params and saves/restores num_experts
    to match the local expert count. Python checks instance __dict__
    before nn.Module's __getattr__ (which accesses _parameters), so
    self.gate_up_proj in the forward finds the local tensor.
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
            ep_enabled=parallel_dims.ep_enabled,
        )
        maybe_enable_async_tp(parallelism, compile_config, parallel_dims.get_mesh("tp"))

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        apply_moe_ep_tp(
            model,
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
            _apply_compile_moe(model, compile_config)
        else:
            apply_compile_dense(model, compile_config)

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
        apply_replicate(
            model,
            parallel_dims.get_mesh("dp_replicate"),
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        )

    return model


# ---------------------------------------------------------------------------
# TP for non-MoE layers + gate handling for MoE layers
# ---------------------------------------------------------------------------


def apply_non_moe_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh,
    loss_parallel: bool,
    enable_float8_tensorwise_tp: bool,
    ep_enabled: bool = False,
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
            root_plan["tok_embeddings"] = NoParallel(
                local_output_grad_placements=(Replicate(),),
            )
        else:
            root_plan["tok_embeddings"] = RowwiseParallel(
                input_layouts=Replicate(),
                output_layouts=Shard(1),
            )

    if hasattr(model, "norm"):
        if isinstance(model.norm, nn.Identity):
            root_plan["norm"] = NoParallel(
                local_output_grad_placements=(Replicate(),),
            )
        else:
            root_plan["norm"] = SequenceParallel()

    if hasattr(model, "output"):
        if isinstance(model.output, nn.Identity):
            root_plan["output"] = NoParallel(
                local_output_grad_placements=(Replicate(),),
            )
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
                    "self_attn.q_a_proj": NoParallel(
                        local_output_grad_placements=(Replicate(),),
                    ),
                    "self_attn.q_a_layernorm": NoParallel(
                        local_output_grad_placements=(Replicate(),),
                    ),
                    "self_attn.q_b_proj": colwise_parallel(),
                    "self_attn.kv_a_proj_with_mqa": NoParallel(
                        local_output_grad_placements=(Replicate(),),
                    ),
                    "self_attn.kv_a_layernorm": NoParallel(
                        local_output_grad_placements=(Replicate(),),
                    ),
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
        elif not ep_enabled:
            # MoE with TP only (no EP): all-gather input, reduce-scatter
            # output. Expert weights and gate handled in apply_moe_ep_tp.
            mlp_plan = {
                "mlp": PrepareModuleInputOutput(
                    input_layouts=(Shard(1),),
                    desired_input_layouts=(Replicate(),),
                    use_local_input=False,
                    output_layouts=(Partial(),),
                    desired_output_layouts=(Shard(1),),
                ),
            }
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


# ---------------------------------------------------------------------------
# MoE parallelism: EP and TP-only
# ---------------------------------------------------------------------------


def apply_moe_ep_tp(
    model: nn.Module,
    tp_mesh: DeviceMesh | None = None,
    ep_mesh: DeviceMesh | None = None,
):
    """Apply Expert Parallelism and/or Tensor Parallelism to MoE layers.

    Mirrors native torchtitan's apply_moe_ep_tp pattern. Handles all
    combinations: EP-only, TP-only, EP+TP.

    Args:
        model: The model with MoE layers.
        tp_mesh: TP device mesh. If provided, expert weights are TP-sharded
            and gate params are replicated on this mesh.
        ep_mesh: EP device mesh. If provided, expert params are sharded on
            dim 0 and dispatch/combine hooks are registered.
    """
    assert tp_mesh is not None or ep_mesh is not None

    for transformer_block in model.layers:
        if not getattr(transformer_block, "moe_enabled", False):
            continue

        moe_block = transformer_block.mlp
        experts = moe_block.experts

        # --- EP: shard experts on dim 0, register dispatch/combine ---
        if ep_mesh is not None:
            parallelize_module(experts, ep_mesh, HFExpertParallel())

        # --- TP: shard expert weights (TP-only), replicate gate, hooks ---
        if tp_mesh is not None:
            if ep_mesh is None:
                # TP-only: shard expert weights (column-wise gate_up,
                # row-wise down). Not done for EP+TP — experts are
                # EP-sharded and each TP rank processes independently.
                experts.gate_up_proj = nn.Parameter(
                    distribute_tensor(experts.gate_up_proj, tp_mesh, [Shard(1)])
                )
                experts.down_proj = nn.Parameter(
                    distribute_tensor(experts.down_proj, tp_mesh, [Shard(2)])
                )

            # Replicate gate params on TP mesh (FSDP mesh alignment)
            _replicate_gate_params(moe_block, tp_mesh)

            # Shadow gate weight with to_local so F.linear produces plain
            # output and topk never sees DTensors
            moe_block.gate.register_forward_pre_hook(_gate_to_local_pre_hook)
            moe_block.gate.register_forward_hook(_gate_restore_post_hook)

            # Convert MoE block DTensor input to local
            if ep_mesh is not None:
                # EP+TP: input is Shard(1) from SP, each TP rank processes
                # its shard independently through EP
                moe_block.register_forward_pre_hook(
                    _make_moe_to_local_pre_hook((Shard(1),))
                )
            else:
                # TP-only: input is Replicate from PrepareModuleInputOutput,
                # output is Partial (row-parallel)
                moe_block.register_forward_pre_hook(
                    _make_moe_to_local_pre_hook((Partial(),))
                )

        # Pre/post hooks on experts: convert DTensor params to local.
        # Store handles so apply_fsdp can re-register after fully_shard.
        h1 = experts.register_forward_pre_hook(_experts_to_local_pre_hook)
        h2 = experts.register_forward_hook(_experts_restore_post_hook, prepend=True)
        experts._to_local_hook_handles = (h1, h2)

    logger.info("Applied MoE parallelism to the model")


# ---------------------------------------------------------------------------
# Compile
# ---------------------------------------------------------------------------


def _apply_compile_moe(model: nn.Module, compile_config: CompileConfig):
    """Apply torch.compile to each TransformerBlock in a MoE-aware manner.

    For non-MoE layers, compiles the whole block. For MoE layers, only
    compiles the gate (when it has no TP hooks). The experts module is
    skipped (FSDP hooks cause graph breaks) and runs eagerly.
    """
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True

    for layer_id, transformer_block in model.layers.named_children():
        if getattr(transformer_block, "moe_enabled", False):
            # MoE layer: compile gate only, skip experts and other
            # sub-modules — see NOTE below.
            # Unwrap CheckpointWrapper (added by apply_ac) to access
            # the actual decoder layer's children.
            if isinstance(transformer_block, CheckpointWrapper):
                block = transformer_block._checkpoint_wrapped_module
            else:
                block = transformer_block

            for attr_name, submod in block.named_children():
                if attr_name == "mlp":
                    for mlp_attr, mlp_submod in submod.named_children():
                        if mlp_attr == "experts":
                            # Skip experts — FSDP hooks cause graph breaks,
                            # and the HF for-loop runs eagerly.
                            continue
                        if mlp_attr == "gate" and mlp_submod._forward_hooks:
                            # Skip compiling the gate when it has TP hooks
                            # (gate to_local). The __dict__ shadowing in
                            # hooks is incompatible with fullgraph=True.
                            continue
                        setattr(
                            submod,
                            mlp_attr,
                            torch.compile(
                                mlp_submod,
                                backend=compile_config.backend,
                                fullgraph=True,
                            ),
                        )
                # NOTE: Don't compile other sub-modules (self_attn,
                # layernorms) individually for MoE layers. When TP is
                # enabled, SequenceParallel and RowwiseParallel produce
                # AsyncCollectiveTensor outputs via async redistribute.
                # These cross into the non-compiled MoE block forward
                # which materializes them via to_local(). In backward,
                # the compiled module expects AsyncCollectiveTensor
                # tangents but gets plain tensors — causing a metadata
                # mismatch crash.
                # See: https://github.com/pytorch/pytorch/issues/172556
        else:
            # Non-MoE layer: compile the whole block
            transformer_block = torch.compile(
                transformer_block,
                backend=compile_config.backend,
                fullgraph=True,
            )

        model.layers.register_module(layer_id, transformer_block)

    logger.info("Compiling each TransformerBlock with torch.compile")


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
            experts = moe_block.experts
            # Expert params are DTensor Shard(0) on EP mesh. After FSDP,
            # local dim-0 has num_experts/ep_degree experts. When
            # efsdp_size > num_local, dim-0 can't be sharded further.
            _experts_shard_placement_fn = None
            assert dp_mod_ep_mesh is not None
            num_local_experts = experts.num_experts // ep_degree
            if dp_mod_ep_mesh.size() > num_local_experts:
                _experts_shard_placement_fn = lambda param: Shard(1)

            # Remove to_local hooks registered in apply_moe_ep_tp (they'd
            # fire before FSDP unshard, getting sharded params).
            if hasattr(experts, "_to_local_hook_handles"):
                for h in experts._to_local_hook_handles:
                    h.remove()
                del experts._to_local_hook_handles

            fully_shard(
                experts,
                **fsdp_mod_ep_config,
                reshard_after_forward=reshard_after_forward,
                shard_placement_fn=_experts_shard_placement_fn,
            )

            # Re-register AFTER fully_shard: pre-hook fires AFTER FSDP
            # unshard, post-hook (prepend) fires BEFORE FSDP reshard.
            experts.register_forward_pre_hook(_experts_to_local_pre_hook)
            experts.register_forward_hook(_experts_restore_post_hook, prepend=True)

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

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

# Patch torch.topk to compute values via gather instead of native topk values.
# torch.topk backward uses aten.scatter.src which crashes on DTensor.
# gather backward uses aten.scatter_add which works on DTensor.
# This makes topk safe for DTensor without changing any model code.
# Native titan avoids this by using topk for indices only + gather for scores
# (moe.py:274,281). HF routers use topk for both, so we patch topk itself.
_orig_topk = torch.topk


def _topk_with_gather(input, k, dim=-1, largest=True, sorted=True, *, out=None):
    with torch.no_grad():
        _, indices = _orig_topk(input, k, dim=dim, largest=largest, sorted=sorted)
    values = input.gather(dim, indices)
    return torch.return_types.topk((values, indices))


torch.topk = _topk_with_gather

from torch.distributed._functional_collectives import all_to_all_single_autograd
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
    ParallelStyle,
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
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp, NoParallel
from torchtitan.experiments.transformers_modeling_backend.compile import (
    apply_compile_sparse,
)
from torchtitan.models.llama3.parallelize import disable_fsdp_gradient_division
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools.logging import logger

# ---------------------------------------------------------------------------
# ParallelStyle classes for HF MoE
# ---------------------------------------------------------------------------


class HFExpertParallel(ParallelStyle):
    """Expert Parallelism for HF Transformers MoE models.

    Adapts torchtitan's ``ExpertParallel`` for the HF experts interface.
    The key difference is the input/output contract:

    - **Native ``ExpertParallel``** expects pre-sorted tokens:
      input ``(routed_input, num_tokens_per_expert)``, output is sorted
      expert results. The ``Reorderer`` in ``MoE.forward`` handles sorting
      before and unsorting after the experts call.

    - **``HFExpertParallel``** receives unsorted tokens from the HF
      ``SparseMoeBlock``: input ``(hidden_states, top_k_index, top_k_weights)``.
      Sorting, routing weight transport, and local routing construction
      happen inside ``_token_dispatch``. ``_token_combine`` handles
      unsorting and top_k accumulation, returning the final output
      directly (HF expects experts to return the accumulated result).

    Partition is the same: both shard expert params on dim 0 via
    ``distribute_tensor(..., [Shard(0)])``.

    Applied to the experts module via ``parallelize_module``. The MoE block
    forward and experts forward run unchanged — dispatch/combine happen
    in input/output hooks registered by ``distribute_module``.
    """

    def __init__(self):
        super().__init__()
        self._ep_group = None
        self._ep_size = None
        self._num_local_experts = None
        self._global_num_experts = None

    def _partition_fn(self, name, mod, device_mesh):
        """Shard expert params on dim 0 across EP ranks.

        Same as native ``ExpertParallel``. Also captures EP metadata
        (group, sizes) for use in dispatch/combine.
        """
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
        """Sort tokens by expert, all-to-all dispatch, build local routing.

        Unlike native ``ExpertParallel`` which receives pre-sorted
        ``(routed_input, num_tokens_per_expert)``, this receives the HF
        interface and performs sorting internally.

        Args:
            inputs: ``(hidden_states, top_k_index, top_k_weights)``
                - hidden_states: ``(num_tokens, hidden_dim)``
                - top_k_index: ``(num_tokens, top_k)`` global expert indices
                - top_k_weights: ``(num_tokens, top_k)`` routing weights

        Returns:
            ``(dispatched_tokens, local_top_k_index, local_top_k_weights)``
                - dispatched_tokens: ``(num_received, hidden_dim)``
                - local_top_k_index: ``(num_received, 1)`` local expert indices
                - local_top_k_weights: ``(num_received, 1)`` routing weights
        """
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
        """All-to-all combine, unsort, sum across top_k.

        Unlike native ``ExpertParallel`` which returns sorted output
        (``MoE.forward`` handles unsorting), this returns the final
        accumulated result — shape ``(num_tokens, hidden_dim)``.
        """
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
            # models (e.g. PhiMoE) apply in-place ops (input_jitter_noise
            # *= ...) which autograd forbids on custom Function views.
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
    assert training.seq_len % parallel_dims.seq_len_divisor == 0, f"""
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
            enable_loss_parallel=not parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=enable_float8_tensorwise_tp,
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
                output_layouts=Shard(-1) if enable_loss_parallel else Replicate(),
                use_local_output=not enable_loss_parallel,
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
        # MoE layers: no mlp plan here. All MoE parallelism
        # (EP, TP, boundary handling) is in apply_moe_ep_tp.

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

    Mirrors native torchtitan's ``apply_moe_ep_tp`` (llama4/parallelize.py).
    Handles all combinations: EP-only, TP-only, EP+TP.

    Key differences from native:

    - **EP dispatch/combine:** Uses ``HFExpertParallel`` which adapts the
      HF experts interface (unsorted tokens) instead of native's
      ``ExpertParallel`` (pre-sorted tokens).
    - **Gate TP:** Uses ``NoParallel`` (same as native) with tuple
      return support. HF routers return tuples (logits, scores, indices)
      while native's ``moe.router.gate`` is an ``nn.Linear`` returning
      a single tensor. Relies on the topk gather patch for DTensor safety.
    - **Expert params to_local:** Done via ``__dict__`` shadowing hooks
      (registered in ``parallelize_hf_transformers`` after ``apply_fsdp``
      to ensure correct hook ordering) instead of ``to_local()`` inside
      the forward — we don't modify the HF forward.

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
        # HF models use either "gate" or "router" for the routing module
        gate = getattr(moe_block, "gate", None) or getattr(moe_block, "router", None)

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

            # Replicate gate params on TP mesh. Most routers (Mixtral,
            # Qwen3, DeepSeek V3, etc.) use NoParallel which wraps
            # inputs as DTensor for the gate's nn.Linear forward.
            #
            # PhiMoE's PhimoeTopKRouter overrides forward to call a
            # custom autograd Function (sparsemixer) whose backward
            # uses scatter_add_, incompatible with DTensor placement
            # changes. For these routers, use distribute_module (bare)
            # + __dict__ param shadowing so the entire forward operates
            # on local tensors. Same pattern as _experts_to_local hooks.
            if type(gate).forward is not nn.Linear.forward:
                distribute_module(gate, tp_mesh)

                def _router_params_to_local(gate_mod):
                    # Use direct attribute access (not named_parameters).
                    # FSDP2 manages params at block level — after unshard,
                    # module.weight works but named_parameters may not.
                    # The local tensor stays in __dict__ between forwards;
                    # FSDP manages the DTensor via its own FSDPParamGroup
                    # references, not module attribute access, so this is
                    # safe. The pre-hook overwrites it each forward with
                    # a fresh to_local() from the newly-unsharded DTensor.
                    def pre_hook(module, args):
                        weight = gate_mod.weight
                        if isinstance(weight, DTensor):
                            gate_mod.__dict__["weight"] = weight.to_local()

                    return pre_hook

                gate.register_forward_pre_hook(_router_params_to_local(gate))
            else:
                parallelize_module(
                    gate,
                    tp_mesh,
                    NoParallel(local_output_grad_placements=(Partial(),)),
                )

            # FSDP converts gate buffers (e.g. e_score_correction_bias in
            # DeepSeek V3/GLM-4.7 routers) to DTensors for mesh alignment.
            # The gate output is local (F.linear with FSDP-unsharded weight
            # returns a local tensor), but the MoE block's route_tokens_to
            # _experts accesses gate buffers directly (outside gate forward)
            # and adds them to the local output, triggering a mixed
            # DTensor/Tensor error.
            # Fix: shadow DTensor buffers with local copies for the entire
            # MoE block forward, not just the gate forward.
            def _gate_buffers_to_local(gate_mod):
                # Local tensors stay in __dict__ between forwards;
                # pre-hook overwrites each forward with fresh to_local().
                def pre_hook(module, args):
                    for name, buf in gate_mod.named_buffers(recurse=False):
                        if isinstance(buf, DTensor):
                            gate_mod.__dict__[name] = buf.to_local()

                return pre_hook

            moe_block.register_forward_pre_hook(_gate_buffers_to_local(gate))

            # TP-shard shared experts if present (e.g., Qwen2/3.5 MoE,
            # DeepSeek V3). The shared expert is a dense MLP — apply
            # ColwiseParallel/RowwiseParallel so its output is Partial,
            # matching the routed expert output.
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

            # Replicate shared_expert_gate on TP mesh (FSDP mesh alignment)
            shared_gate = getattr(moe_block, "shared_expert_gate", None)
            if shared_gate is not None:
                parallelize_module(
                    shared_gate,
                    tp_mesh,
                    NoParallel(local_output_grad_placements=(Partial(),)),
                )

            # MoE block TP boundary (TP-only): all-gather input, reduce-
            # scatter output. Must be registered BEFORE to_local hook so
            # hook order is: PrepareModuleInputOutput → to_local.
            if ep_mesh is None:
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

            fully_shard(
                experts,
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

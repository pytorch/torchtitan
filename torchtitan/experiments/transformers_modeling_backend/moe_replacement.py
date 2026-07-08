# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Replace HF MoE blocks with Titan MoE modules.

Two-phase replacement:
  Phase 1 (init time): ``prepare_native_moe_configs`` probes the HF MoE block
      and builds a ``MoE.Config``, stored on each layer as ``_native_moe_config``.
  Phase 2 (parallelize time): ``build_and_swap_native_moe`` calls
      ``set_moe_sharding_config`` on each stored config, builds the Titan MoE,
      initializes it, and swaps it into the layer. Actual parallelization
      happens later via ``model.parallelize(parallel_dims)``.
"""

from functools import partial

import spmd_types as spmd
import torch
import torch.nn as nn

from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models.common.config_utils import (
    make_experts_config,
    make_ffn_config,
    make_moe_config,
    make_router_config,
    make_token_dispatcher_config,
)
from torchtitan.models.common.decoder_sharding import (
    dense_activation_placement,
    dense_param_placement,
)
from torchtitan.models.common.feed_forward import SharedExperts
from torchtitan.models.common.moe import GroupedExperts, MoE
from torchtitan.models.common.moe_sharding import set_moe_sharding_config
from torchtitan.models.common.nn_modules import Linear
from torchtitan.protocols.sharding import ShardingConfig
from torchtitan.tools.logging import logger


# ---------------------------------------------------------------------------
# Phase 1 — config preparation (called from HFTransformerModel.__init__)
# ---------------------------------------------------------------------------


def prepare_native_moe_configs(model: nn.Module, config) -> None:
    """Probe each MoE layer and store a ``MoE.Config`` for later build.

    Called during ``HFTransformerModel.__init__`` on meta device.
    Does NOT instantiate the Titan MoE modules yet -- that happens in Phase 2.
    """
    for layer in model.layers.values():
        if not getattr(layer, "moe_enabled", False):
            continue

        is_layer_level = getattr(layer, "_layer_level_moe", False)
        if is_layer_level:
            # Layer-level MoE (Gemma4): router/experts are siblings of dense
            # MLP. Probe from the layer itself, treating the dense MLP as
            # the shared expert.
            moe_params = _probe_layer_level_moe(layer, config)
        else:
            moe_block = _get_moe_block(layer)
            moe_params = _probe_hf_moe_block(moe_block, config)
        moe_config = _build_moe_config(moe_params, config)
        layer._native_moe_config = moe_config

    logger.info("Prepared Titan MoE configs for all MoE layers")


# ---------------------------------------------------------------------------
# Phase 2 — build, init, swap (called from parallelize_hf_transformers)
# ---------------------------------------------------------------------------


def build_and_swap_native_moe(
    model: nn.Module,
    parallel_dims: ParallelDims,
) -> None:
    """Build Titan MoE modules and swap them into the model.

    For each MoE layer with a stored ``_native_moe_config``:
    1. Set sharding config on the MoE.Config (now that EP/TP is known)
    2. Build the Titan MoE module
    3. Initialize parameters and buffers
    4. Swap into the layer's MoE attribute, set ``layer.moe`` for load-balancing hook

    Args:
        model: The HFTransformerModel with ``_native_moe_config`` stored on
            each MoE-enabled layer (from ``prepare_native_moe_configs``).
        parallel_dims: Parallel dimensions for EP/TP mesh resolution.
    """
    enable_ep = parallel_dims.ep_enabled
    enable_sp = parallel_dims.tp_enabled

    for layer in model.layers.values():
        moe_config = getattr(layer, "_native_moe_config", None)
        if moe_config is None:
            continue

        _, expert_layout = _get_expert_param_info()
        set_moe_sharding_config(
            moe_config,
            enable_ep=enable_ep,
            enable_sp=enable_sp,
            expert_param_layout=expert_layout,
        )

        # set_moe_sharding_config shards the shared FFN (w1/w2/w3) but leaves the
        # SharedExperts sigmoid gate to model-specific code. Replicate it (weight
        # and output), matching qwen3_5's _set_shared_expert_gate_sharding.
        shared = moe_config.shared_experts
        if isinstance(shared, SharedExperts.Config):
            shared.gate.sharding_config = ShardingConfig(
                state_shardings={
                    "weight": dense_param_placement(tp=spmd.R),
                    "bias": dense_param_placement(tp=spmd.R),
                },
                out_dst_shardings=dense_activation_placement(tp=spmd.R),
            )

        with torch.device("meta"):
            native_moe = moe_config.build()

        # Materialize meta params to real tensors, then initialize values.
        # This mirrors the trainer's flow: to_empty → init_states.
        native_moe.to_empty(device=torch.device("cpu"))
        native_moe.init_states(buffer_device=torch.device("cpu"))

        # Swap the Titan MoE into the layer's original attribute
        # (``mlp`` for most models, ``feed_forward`` for Llama4).
        moe_attr = _get_moe_attr_name(layer)
        setattr(layer, moe_attr, native_moe)
        object.__setattr__(layer, "moe", native_moe)

        # For layer-level MoE (Gemma4), the original router and experts
        # are layer-level siblings of the dense MLP. After swapping in
        # the Titan MoE at ``layer.mlp``, disable the HF forward's
        # separate MoE path (which references ``self.router`` and
        # ``self.experts``) and delete the original modules to prevent
        # duplicate parameter registration.
        if getattr(layer, "_layer_level_moe", False):
            if hasattr(layer, "enable_moe_block"):
                layer.enable_moe_block = False
            if hasattr(layer, "router"):
                delattr(layer, "router")
            if hasattr(layer, "experts"):
                delattr(layer, "experts")
            # Remove unused norms from the separate MoE path to keep
            # the parameter count clean.
            for norm_name in (
                "pre_feedforward_layernorm_2",
                "post_feedforward_layernorm_1",
                "post_feedforward_layernorm_2",
            ):
                if hasattr(layer, norm_name):
                    delattr(layer, norm_name)

        # Llama4 decoder forward unpacks ``(output, router_logits)`` when
        # ``is_moe_layer`` is True.  The Titan MoE returns a plain tensor,
        # so disable the unpacking.
        if getattr(layer, "is_moe_layer", False):
            layer.is_moe_layer = False

        del layer._native_moe_config

    logger.info("Built and swapped Titan MoE modules into the model")


# ---------------------------------------------------------------------------
# MoE attribute helpers
# ---------------------------------------------------------------------------


def _get_moe_attr_name(layer: nn.Module) -> str:
    """Return the attribute name holding the MoE block on a decoder layer.

    Most models use ``mlp``; Llama4 uses ``feed_forward``.
    For layer-level MoE (Gemma4), the Titan MoE replaces ``mlp``.
    """
    for name in ("mlp", "feed_forward"):
        if hasattr(layer, name):
            return name
    raise AttributeError(
        f"Layer {type(layer).__name__} has neither 'mlp' nor 'feed_forward'"
    )


def _get_moe_block(layer: nn.Module) -> nn.Module:
    """Return the MoE block module from a decoder layer."""
    return getattr(layer, _get_moe_attr_name(layer))


# ---------------------------------------------------------------------------
# HF MoE block probing
# ---------------------------------------------------------------------------


def _probe_hf_moe_block(moe_block: nn.Module, config) -> dict:
    """Extract MoE configuration from an HF MoE block.

    Args:
        moe_block: The HF MoE block (e.g., ``Qwen3MoeSparseMoeBlock``).
        config: The HF model config with MoE-related attributes.

    Returns:
        Dict with all parameters needed to build a Titan ``MoE.Config``.
    """
    gate = getattr(moe_block, "gate", None) or getattr(moe_block, "router", None)
    experts = moe_block.experts

    num_experts = _resolve_num_experts(experts, gate, moe_block, config)
    dim = config.hidden_size

    # Intermediate size: HF MoE models use fused gate_up_proj.
    # Standard layout: (E, 2*I, H) — dim 1 is 2*I.
    # Llama4 layout: (E, H, 2*I) — dim 2 is 2*I (transposed).
    if hasattr(experts, "gate_up_proj"):
        shape = experts.gate_up_proj.shape
        # Standard: shape[1] == 2*I, shape[2] == H == dim
        # Transposed: shape[1] == H == dim, shape[2] == 2*I
        if shape[2] == dim and shape[1] != dim:
            # Standard layout
            moe_intermediate_size = shape[1] // 2
        elif shape[1] == dim and shape[2] != dim:
            # Transposed layout (Llama4)
            moe_intermediate_size = shape[2] // 2
        else:
            # Both dims match H (e.g. 2*I == H), fall back to config
            if (
                hasattr(config, "moe_intermediate_size")
                and config.moe_intermediate_size
            ):
                moe_intermediate_size = config.moe_intermediate_size
            else:
                moe_intermediate_size = shape[1] // 2
    elif hasattr(config, "moe_intermediate_size") and config.moe_intermediate_size:
        moe_intermediate_size = config.moe_intermediate_size
    else:
        moe_intermediate_size = getattr(config, "intermediate_size", dim * 4)

    top_k = _resolve_top_k(moe_block, gate, config)

    # Router scoring function
    score_func = _resolve_score_func(gate, config)

    # Route normalization: some models (Mixtral, Qwen3.5) always normalize
    # but don't have a config flag. Detect by checking if norm_topk_prob is
    # absent (meaning the router hardcodes normalization).
    # Sigmoid models (Llama4) without explicit norm_topk_prob default to
    # False — sigmoid outputs are used directly as scores without normalization.
    if hasattr(config, "norm_topk_prob"):
        route_norm = config.norm_topk_prob
    elif hasattr(gate, "norm_topk_prob"):
        route_norm = gate.norm_topk_prob
    else:
        route_norm = score_func != "sigmoid"

    # Route scaling factor (DeepSeek V2/V3)
    route_scale = getattr(config, "routed_scaling_factor", 1.0)

    # Group-limited routing (DeepSeek V2/V3)
    num_expert_groups = getattr(config, "n_group", None)
    num_limited_groups = getattr(config, "topk_group", None)

    # Load balance coefficient
    load_balance_coeff = getattr(config, "load_balance_coeff", 1e-3)

    # Comm backend
    comm_backend = getattr(config, "comm_backend", "standard")

    # Shared experts
    shared_expert_info = _probe_shared_experts(moe_block, config)

    return {
        "num_experts": num_experts,
        "dim": dim,
        "moe_intermediate_size": moe_intermediate_size,
        "top_k": top_k,
        "score_func": score_func,
        "route_norm": route_norm,
        "route_scale": route_scale,
        "num_expert_groups": num_expert_groups,
        "num_limited_groups": num_limited_groups,
        "load_balance_coeff": load_balance_coeff,
        "comm_backend": comm_backend,
        "shared_expert_info": shared_expert_info,
    }


def _resolve_num_experts(
    experts: nn.Module, gate: nn.Module | None, moe_block: nn.Module, config
) -> int:
    """Infer the total expert count from the HF MoE block or config."""
    for owner in (experts, gate, moe_block):
        if owner is None:
            continue
        for attr in ("num_experts", "n_routed_experts", "num_local_experts"):
            val = getattr(owner, attr, None)
            if val is not None:
                return int(val)
    for attr in ("num_experts", "n_routed_experts", "num_local_experts"):
        val = getattr(config, attr, None)
        if val is not None:
            return int(val)
    if gate is not None and hasattr(gate, "weight"):
        return gate.weight.shape[0]
    raise ValueError("Could not determine number of experts from HF MoE block")


def _resolve_top_k(moe_block: nn.Module, gate: nn.Module | None, config) -> int:
    """Infer top-k routing from the HF MoE block or config."""
    for owner in (moe_block, gate):
        if owner is None:
            continue
        for attr in ("top_k", "num_experts_per_tok", "top_k_experts"):
            val = getattr(owner, attr, None)
            if val is not None:
                return int(val)
    for attr in ("num_experts_per_tok", "top_k_experts"):
        val = getattr(config, attr, None)
        if val is not None:
            return int(val)
    raise ValueError("Could not determine top_k from HF MoE block")


def _resolve_score_func(gate: nn.Module | None, config) -> str:
    """Determine the router scoring function (softmax or sigmoid)."""
    # DeepSeek-V3 / GLM use sigmoid with e_score_correction_bias.
    if gate is not None and "e_score_correction_bias" in getattr(gate, "_buffers", {}):
        return "sigmoid"

    scoring_func = getattr(config, "scoring_func", None)
    if scoring_func is not None:
        if scoring_func in ("softmax", "sigmoid"):
            return scoring_func
        raise ValueError(
            f"Unsupported scoring function '{scoring_func}'. "
            "Titan MoE router supports 'softmax' and 'sigmoid'."
        )

    return "softmax"


def _probe_shared_experts(moe_block: nn.Module, config) -> dict | None:
    """Detect shared expert configuration from the HF MoE block."""
    shared = None
    for name in ("shared_expert", "shared_experts", "shared_mlp"):
        shared = getattr(moe_block, name, None)
        if shared is not None:
            break

    if shared is None:
        return None

    # Determine shared expert intermediate size
    for attr in ("intermediate_size", "hidden_size"):
        if hasattr(shared, attr):
            shared_hidden_dim = getattr(shared, attr)
            break
    else:
        # Try to infer from weight shapes
        gate_proj = getattr(shared, "gate_proj", None)
        if gate_proj is not None and hasattr(gate_proj, "weight"):
            shared_hidden_dim = gate_proj.weight.shape[0]
        else:
            shared_hidden_dim = getattr(config, "shared_expert_intermediate_size", None)
            if shared_hidden_dim is None:
                n_shared = getattr(config, "n_shared_experts", 1)
                shared_hidden_dim = (
                    getattr(config, "moe_intermediate_size", config.hidden_size)
                    * n_shared
                )

    # Check for sigmoid-gated shared expert (Qwen3.5 pattern)
    shared_expert_gate = getattr(moe_block, "shared_expert_gate", None)
    has_sigmoid_gate = shared_expert_gate is not None

    return {
        "hidden_dim": shared_hidden_dim,
        "dim": config.hidden_size,
        "has_sigmoid_gate": has_sigmoid_gate,
    }


def _probe_layer_level_moe(layer: nn.Module, config) -> dict:
    """Probe layer-level MoE (Gemma4) where router/experts are layer siblings.

    The dense MLP is treated as a shared expert: its output is summed with
    the routed expert output in the original HF forward. The Titan MoE
    replaces ``layer.mlp`` and contains all three components (router,
    experts, shared_experts=dense MLP).
    """
    gate = getattr(layer, "gate", None) or getattr(layer, "router", None)
    experts = layer.experts

    num_experts = _resolve_num_experts(experts, gate, layer, config)
    dim = config.hidden_size
    top_k = _resolve_top_k(layer, gate, config)
    score_func = _resolve_score_func(gate, config)

    # Expert intermediate size from fused gate_up_proj
    if hasattr(experts, "gate_up_proj"):
        shape = experts.gate_up_proj.shape
        if shape[2] == dim and shape[1] != dim:
            moe_intermediate_size = shape[1] // 2
        elif shape[1] == dim and shape[2] != dim:
            moe_intermediate_size = shape[2] // 2
        else:
            moe_intermediate_size = getattr(
                config, "moe_intermediate_size", shape[1] // 2
            )
    else:
        moe_intermediate_size = getattr(config, "moe_intermediate_size", dim * 4)

    # Route normalization
    if hasattr(config, "norm_topk_prob"):
        route_norm = config.norm_topk_prob
    elif hasattr(gate, "norm_topk_prob") if gate else False:
        route_norm = gate.norm_topk_prob
    else:
        route_norm = score_func != "sigmoid"

    route_scale = getattr(config, "routed_scaling_factor", 1.0)
    num_expert_groups = getattr(config, "n_group", None)
    num_limited_groups = getattr(config, "topk_group", None)
    load_balance_coeff = getattr(config, "load_balance_coeff", 1e-3)
    comm_backend = getattr(config, "comm_backend", "standard")

    # Dense MLP is the shared expert
    mlp = getattr(layer, "mlp", None)
    shared_expert_info = None
    if mlp is not None:
        gate_proj = getattr(mlp, "gate_proj", None)
        if gate_proj is not None and hasattr(gate_proj, "weight"):
            shared_hidden_dim = gate_proj.weight.shape[0]
        else:
            shared_hidden_dim = getattr(config, "intermediate_size", dim * 4)
        shared_expert_info = {
            "hidden_dim": shared_hidden_dim,
            "dim": dim,
            "has_sigmoid_gate": False,
        }

    return {
        "num_experts": num_experts,
        "dim": dim,
        "moe_intermediate_size": moe_intermediate_size,
        "top_k": top_k,
        "score_func": score_func,
        "route_norm": route_norm,
        "route_scale": route_scale,
        "num_expert_groups": num_expert_groups,
        "num_limited_groups": num_limited_groups,
        "load_balance_coeff": load_balance_coeff,
        "comm_backend": comm_backend,
        "shared_expert_info": shared_expert_info,
    }


# ---------------------------------------------------------------------------
# Sigmoid-gated shared expert wrapper (Qwen3.5 pattern)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Config building
# ---------------------------------------------------------------------------

_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}

_expert_param_info_cache: tuple[dict, dict] | None = None


def _get_expert_param_info() -> tuple[dict, dict[str, spmd.PerMeshAxisSpmdType]]:
    """Discover GroupedExperts parameter names and their TP shard placements.

    Builds a tiny throwaway instance on meta device to introspect actual
    parameter names (which may carry dimension suffixes like ``_EFD``).
    Returns ``(param_init, param_layout)`` where ``param_init`` maps each
    name to ``trunc_normal_`` and ``param_layout`` maps each name to the
    correct ``Shard`` placement for TP.
    """
    global _expert_param_info_cache
    if _expert_param_info_cache is not None:
        return _expert_param_info_cache

    with torch.device("meta"):
        temp = GroupedExperts.Config(
            dim=2,
            hidden_dim=4,
            num_experts=2,
            token_dispatcher=make_token_dispatcher_config(
                num_experts=2, top_k=1, comm_backend="standard"
            ),
        ).build()

    init_fn = partial(nn.init.trunc_normal_, std=0.02)
    param_init: dict = {}
    param_layout: dict[str, spmd.PerMeshAxisSpmdType] = {}

    for name, param in temp.named_parameters(recurse=False):
        param_init[name] = init_fn
        # (E, hidden_dim, dim) → colwise S(1)  [w1/w3 pattern]
        # (E, dim, hidden_dim) → rowwise S(2)  [w2 pattern]
        if param.shape[1] >= param.shape[2]:
            param_layout[name] = spmd.S(1)
        else:
            param_layout[name] = spmd.S(2)

    _expert_param_info_cache = (param_init, param_layout)
    del temp
    return _expert_param_info_cache


def _build_moe_config(params: dict, config) -> MoE.Config:
    """Build a fully-specified MoE.Config from probed parameters."""
    router = make_router_config(
        dim=params["dim"],
        num_experts=params["num_experts"],
        gate_param_init=_LINEAR_INIT,
        top_k=params["top_k"],
        score_func=params["score_func"],
        route_norm=params["route_norm"],
        route_scale=params["route_scale"],
        num_expert_groups=params["num_expert_groups"],
        num_limited_groups=params["num_limited_groups"],
    )

    expert_init, _ = _get_expert_param_info()
    experts = make_experts_config(
        dim=params["dim"],
        hidden_dim=params["moe_intermediate_size"],
        num_experts=params["num_experts"],
        top_k=params["top_k"],
        param_init=expert_init,
        comm_backend=params["comm_backend"],
    )

    shared_experts = None
    shared_info = params["shared_expert_info"]
    if shared_info is not None:
        ffn_config = make_ffn_config(
            dim=shared_info["dim"],
            hidden_dim=shared_info["hidden_dim"],
            w1_param_init=_LINEAR_INIT,
            w2w3_param_init=_LINEAR_INIT,
        )
        if shared_info["has_sigmoid_gate"]:
            # SharedExperts is a FeedForward subclass, so w1/w2/w3 stay flat
            # (no nested ``ffn.`` level) and are directly shardable by
            # set_moe_sharding_config -- no temporary-swap workaround needed.
            shared_experts = SharedExperts.Config(
                w1=ffn_config.w1,
                w2=ffn_config.w2,
                w3=ffn_config.w3,
                gate=Linear.Config(
                    in_features=shared_info["dim"],
                    out_features=1,
                    bias=False,
                    param_init=_LINEAR_INIT,
                ),
            )
        else:
            shared_experts = ffn_config

    return make_moe_config(
        num_experts=params["num_experts"],
        router=router,
        experts=experts,
        shared_experts=shared_experts,
        load_balance_coeff=params["load_balance_coeff"],
    )

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Replace HF MoE blocks with native TorchTitan MoE modules.

Two-phase replacement:
  Phase 1 (init time): ``prepare_native_moe_configs`` probes the HF MoE block
      and builds a ``MoE.Config``, stored on each layer as ``_native_moe_config``.
  Phase 2 (parallelize time): ``build_and_swap_native_moe`` calls
      ``set_moe_sharding_config`` on each stored config, builds the native MoE,
      initializes it, swaps it into the layer, and calls ``parallelize()``.
"""

from dataclasses import dataclass
from functools import partial

import torch
import torch.nn as nn
from torch.distributed.tensor import Shard

from torchtitan.distributed.parallel_dims import ParallelDims
from torchtitan.models.common.config_utils import (
    make_experts_config,
    make_ffn_config,
    make_moe_config,
    make_router_config,
)
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.moe_sharding import set_moe_sharding_config
from torchtitan.protocols.module import Module
from torchtitan.tools.logging import logger


# ---------------------------------------------------------------------------
# Phase 1 — config preparation (called from HFTransformerModel.__init__)
# ---------------------------------------------------------------------------


def prepare_native_moe_configs(model: nn.Module, config) -> None:
    """Probe each MoE layer and store a ``MoE.Config`` for later build.

    Called during ``HFTransformerModel.__init__`` on meta device.
    Does NOT instantiate native modules yet — that happens in Phase 2.
    """
    for layer in model.layers.values():
        if not getattr(layer, "moe_enabled", False):
            continue

        moe_params = _probe_hf_moe_block(layer.mlp, config)
        moe_config = _build_moe_config(moe_params, config)
        layer._native_moe_config = moe_config

    logger.info("Prepared native MoE configs for all MoE layers")


# ---------------------------------------------------------------------------
# Phase 2 — build, init, swap, parallelize (called from parallelize_hf_transformers)
# ---------------------------------------------------------------------------


def build_and_swap_native_moe(
    model: nn.Module,
    parallel_dims: ParallelDims,
) -> None:
    """Build native MoE modules and swap them into the model.

    For each MoE layer with a stored ``_native_moe_config``:
    1. Set sharding config on the MoE.Config (now that EP/TP is known)
    2. Build the native MoE module
    3. Initialize parameters and buffers
    4. Swap into layer.mlp, set layer.moe for load-balancing hook
    5. Call parallelize() to shard states and wire token dispatchers
    """
    enable_ep = parallel_dims.ep_enabled
    enable_sp = parallel_dims.tp_enabled

    for layer in model.layers.values():
        moe_config = getattr(layer, "_native_moe_config", None)
        if moe_config is None:
            continue

        set_moe_sharding_config(
            moe_config,
            enable_ep=enable_ep,
            enable_sp=enable_sp,
            expert_param_layout={"w1": Shard(1), "w2": Shard(2), "w3": Shard(1)},
        )

        with torch.device("meta"):
            native_moe = moe_config.build()

        # Materialize meta params to real tensors, then initialize values.
        # This mirrors the trainer's flow: to_empty → init_states.
        native_moe.to_empty(device=torch.device("cpu"))
        native_moe.init_states(buffer_device=torch.device("cpu"))

        layer.mlp = native_moe
        object.__setattr__(layer, "moe", native_moe)

        if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
            native_moe.parallelize(parallel_dims)

        del layer._native_moe_config

    logger.info("Built and swapped native MoE modules into the model")


# ---------------------------------------------------------------------------
# HF MoE block probing
# ---------------------------------------------------------------------------


def _probe_hf_moe_block(moe_block: nn.Module, config) -> dict:
    """Extract MoE configuration from an HF MoE block.

    Returns a dict with all parameters needed to build a native MoE.Config.
    """
    gate = getattr(moe_block, "gate", None) or getattr(moe_block, "router", None)
    experts = moe_block.experts

    num_experts = _resolve_num_experts(experts, gate, moe_block, config)
    dim = config.hidden_size

    # Intermediate size: all HF MoE models use fused gate_up_proj (E, 2*I, H)
    if hasattr(experts, "gate_up_proj"):
        moe_intermediate_size = experts.gate_up_proj.shape[1] // 2
    elif hasattr(config, "moe_intermediate_size") and config.moe_intermediate_size:
        moe_intermediate_size = config.moe_intermediate_size
    else:
        moe_intermediate_size = getattr(config, "intermediate_size", dim * 4)

    top_k = _resolve_top_k(moe_block, gate, config)

    # Router scoring function
    score_func = _resolve_score_func(gate, config)

    # Route normalization
    route_norm = getattr(config, "norm_topk_prob", False)

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
    for owner in (moe_block, gate):
        if owner is None:
            continue
        for attr in ("top_k", "num_experts_per_tok"):
            val = getattr(owner, attr, None)
            if val is not None:
                return int(val)
    val = getattr(config, "num_experts_per_tok", None)
    if val is not None:
        return int(val)
    raise ValueError("Could not determine top_k from HF MoE block")


def _resolve_score_func(gate: nn.Module | None, config) -> str:
    # DeepSeek V3/V4 uses sigmoid with e_score_correction_bias
    if gate is not None and "e_score_correction_bias" in getattr(gate, "_buffers", {}):
        return "sigmoid"

    scoring_func = getattr(config, "scoring_func", None)
    if scoring_func is not None:
        if scoring_func in ("softmax", "sigmoid"):
            return scoring_func
        raise ValueError(
            f"Unsupported scoring function '{scoring_func}'. "
            "Native MoE router supports 'softmax' and 'sigmoid'."
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


# ---------------------------------------------------------------------------
# Sigmoid-gated shared expert wrapper (Qwen3.5 pattern)
# ---------------------------------------------------------------------------


class GatedSharedExpert(Module):
    """Wraps a FeedForward shared expert with a learned sigmoid gate.

    Matches the Qwen3.5 pattern: ``sigmoid(gate(x)) * shared_expert(x)``.
    The gate is a single-output linear: ``nn.Linear(dim, 1, bias=False)``.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        ffn: FeedForward.Config
        gate: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.ffn = config.ffn.build()
        self.gate = config.gate.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.gate(x)) * self.ffn(x)


# ---------------------------------------------------------------------------
# Config building
# ---------------------------------------------------------------------------

_EXPERT_INIT = {
    "w1": partial(nn.init.trunc_normal_, std=0.02),
    "w2": partial(nn.init.trunc_normal_, std=0.02),
    "w3": partial(nn.init.trunc_normal_, std=0.02),
}

_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}


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

    experts = make_experts_config(
        dim=params["dim"],
        hidden_dim=params["moe_intermediate_size"],
        num_experts=params["num_experts"],
        top_k=params["top_k"],
        param_init=_EXPERT_INIT,
        score_before_experts=False,
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
            shared_experts = GatedSharedExpert.Config(
                ffn=ffn_config,
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

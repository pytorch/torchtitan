# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ShardingConfig-based TP setup for HF model modules.

Sets ``_sharding_config`` on every HF sub-module so that a single
``model.parallelize(parallel_dims)`` call handles all TP distribution
and forward wrapping via the Module protocol.

Unlike native titan models (which use ``local_map`` on inner attention
to convert DTensors to local tensors before SDPA), HF attention
internals (view, RoPE, SDPA) operate directly on DTensors. This works
because DTensor dispatch handles these ops transparently. The rotary
embedding's buffers are also distributed so its computed cos/sin are
DTensors, avoiding mixed plain-Tensor / DTensor errors in RoPE.

MoE layers are already native Module instances with ShardingConfig and
are handled by ``model.parallelize()`` directly.
"""

import inspect

import torch.nn as nn
from torch.distributed.tensor import Placement, Replicate, Shard

from torchtitan.models.common.decoder_sharding import (
    colwise_config,
    dense_activation_placement,
    dense_param_placement,
    rowwise_config,
)
from torchtitan.protocols.sharding import ShardingConfig
from torchtitan.tools.logging import logger


def set_hf_sharding_configs(
    model: nn.Module,
    *,
    enable_sp: bool,
    enable_loss_parallel: bool,
) -> None:
    """Set ``_sharding_config`` on all HF modules for TP/EP parallelization.

    Root-level and per-layer modules all use ``_sharding_config``.
    MoE layers are skipped (already have ShardingConfig from native MoE).
    Actual DTensor distribution happens later in ``model.parallelize()``.

    Args:
        model: The HFTransformerModel with Module-converted children.
        enable_sp: Whether sequence parallelism is enabled (TP > 1).
        enable_loss_parallel: Whether to shard lm_head output for loss parallel.
    """
    activation_tp: Placement = Shard(1) if enable_sp else Replicate()
    loss_tp: Placement = Shard(-1) if enable_loss_parallel else Replicate()

    # Root-level modules — nn.Identity modules are skipped (no params).
    if model.tok_embeddings is not None and not isinstance(
        model.tok_embeddings, nn.Identity
    ):
        emb_state: dict = {"weight": dense_param_placement(tp=Shard(0))}
        for buf_name, _ in model.tok_embeddings.named_buffers(recurse=False):
            emb_state[buf_name] = dense_param_placement(tp=Replicate())
        model.tok_embeddings._sharding_config = ShardingConfig(
            state_shardings=emb_state,
            in_src_shardings={"input": dense_activation_placement(tp=Replicate())},
            in_dst_shardings={"input": dense_activation_placement(tp=Replicate())},
            out_dst_shardings=dense_activation_placement(tp=activation_tp),
        )

    if model.norm is not None and not isinstance(model.norm, nn.Identity):
        model.norm._sharding_config = _hf_norm_config(enable_sp=enable_sp)

    if model.lm_head is not None and not isinstance(model.lm_head, nn.Identity):
        model.lm_head._sharding_config = ShardingConfig(
            state_shardings={
                "weight": dense_param_placement(tp=Shard(0)),
                "bias": dense_param_placement(tp=Shard(0)),
            },
            in_src_shardings={
                "input": dense_activation_placement(tp=activation_tp),
            },
            in_dst_shardings={
                "input": dense_activation_placement(tp=Replicate()),
            },
            out_dst_shardings=dense_activation_placement(tp=loss_tp),
        )

    # Rotary embedding — distribute buffers (inv_freq) and wrap inputs
    # as DTensors so computed cos/sin are DTensors, avoiding mixed
    # plain-Tensor / DTensor ops in apply_rotary_pos_emb.
    if hasattr(model, "rotary_emb") and not isinstance(model.rotary_emb, nn.Identity):
        rope = model.rotary_emb
        rope._sharding_config = _replicate_config(rope, wrap_inputs=True)

    # Per-layer modules
    for transformer_block in model.layers:
        _set_layer_sharding_configs(transformer_block, enable_sp=enable_sp)

    logger.info("Set sharding configs on all HF modules")


def _set_layer_sharding_configs(layer: nn.Module, *, enable_sp: bool) -> None:
    """Set ``_sharding_config`` on each sub-module of a decoder layer.

    Covers norms, attention projections, and (for non-MoE layers) dense MLP.
    MoE layers have their own ShardingConfig from the native MoE swap.
    """
    # --- Norms ---
    if hasattr(layer, "input_layernorm"):
        layer.input_layernorm._sharding_config = _hf_norm_config(enable_sp=enable_sp)
    if hasattr(layer, "post_attention_layernorm"):
        layer.post_attention_layernorm._sharding_config = _hf_norm_config(
            enable_sp=enable_sp
        )
    if hasattr(layer, "pre_feedforward_layernorm"):
        layer.pre_feedforward_layernorm._sharding_config = _hf_norm_config(
            enable_sp=enable_sp
        )
    if hasattr(layer, "post_feedforward_layernorm"):
        layer.post_feedforward_layernorm._sharding_config = _hf_norm_config(
            enable_sp=enable_sp
        )

    # --- Attention ---
    attn = layer.self_attn

    # Gather input from SP layout to Replicate at the attention boundary.
    attn_x: Placement = Shard(1) if enable_sp else Replicate()
    attn._sharding_config = ShardingConfig(
        in_src_shardings={
            "hidden_states": dense_activation_placement(tp=attn_x),
        },
        in_dst_shardings={
            "hidden_states": dense_activation_placement(tp=Replicate()),
        },
    )

    # Q/K/V projections
    if getattr(attn, "q_lora_rank", None) is None:
        for proj_name in ("q_proj", "k_proj", "v_proj"):
            if hasattr(attn, proj_name):
                getattr(attn, proj_name)._sharding_config = colwise_config()
    else:
        # MLA — low-rank projections replicated, up-projections colwise
        for name in (
            "q_a_proj",
            "q_a_layernorm",
            "kv_a_proj_with_mqa",
            "kv_a_layernorm",
        ):
            mod = getattr(attn, name, None)
            if mod is not None:
                mod._sharding_config = _replicate_config(mod)
        for name in ("q_b_proj", "kv_b_proj"):
            if hasattr(attn, name):
                getattr(attn, name)._sharding_config = colwise_config()

    # O projection
    o_proj_name = "o_proj" if hasattr(attn, "o_proj") else "dense"
    getattr(attn, o_proj_name)._sharding_config = rowwise_config(output_sp=enable_sp)

    # Q/K norms (Qwen3) — weight replicated, activations stay heads-sharded
    if hasattr(attn, "q_norm") and hasattr(attn, "k_norm"):
        attn.q_norm._sharding_config = _replicate_config(attn.q_norm)
        attn.k_norm._sharding_config = _replicate_config(attn.k_norm)

    # GLM-5 DSA indexer
    if hasattr(attn, "indexer"):
        attn.indexer._sharding_config = _replicate_config(attn.indexer)

    # V-norm (Gemma4) — parameter-free RMSNorm applied per-head
    if hasattr(attn, "v_norm"):
        attn.v_norm._sharding_config = _replicate_config(attn.v_norm)

    # --- Dense MLP (non-MoE layers only) ---
    if not getattr(layer, "moe_enabled", False):
        mlp_attr = "mlp" if hasattr(layer, "mlp") else "feed_forward"
        mlp = getattr(layer, mlp_attr)

        mlp_arg = _first_forward_arg(mlp)
        mlp_x: Placement = Shard(1) if enable_sp else Replicate()
        mlp._sharding_config = ShardingConfig(
            in_src_shardings={mlp_arg: dense_activation_placement(tp=mlp_x)},
            in_dst_shardings={mlp_arg: dense_activation_placement(tp=Replicate())},
        )

        gate_name = "gate_proj" if hasattr(mlp, "gate_proj") else "fc1"
        getattr(mlp, gate_name)._sharding_config = colwise_config()

        if hasattr(mlp, "up_proj"):
            mlp.up_proj._sharding_config = colwise_config()

        down_name = "down_proj" if hasattr(mlp, "down_proj") else "fc2"
        getattr(mlp, down_name)._sharding_config = rowwise_config(output_sp=enable_sp)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _hf_norm_config(*, enable_sp: bool) -> ShardingConfig:
    """Norm sharding using HF's ``hidden_states`` arg name."""
    state = {
        "weight": dense_param_placement(tp=Replicate()),
        "bias": dense_param_placement(tp=Replicate()),
    }
    if not enable_sp:
        return ShardingConfig(state_shardings=state)
    return ShardingConfig(
        state_shardings=state,
        in_src_shardings={
            "hidden_states": dense_activation_placement(tp=Shard(1)),
        },
        in_dst_shardings={
            "hidden_states": dense_activation_placement(tp=Shard(1)),
        },
        out_dst_shardings=dense_activation_placement(tp=Shard(1)),
    )


def _replicate_config(
    module: nn.Module, *, wrap_inputs: bool = False
) -> ShardingConfig:
    """Replicate all params and buffers — ShardingConfig equivalent of NoParallel.

    Dynamically enumerates the module's own parameters and buffers to avoid
    ``_shard_states`` raising on undeclared entries.

    When ``wrap_inputs=True``, also wraps all positional forward args as
    DTensor(Replicate). This is needed for modules (like rotary embedding)
    whose forward receives plain tensors that must be DTensors to avoid
    mixed-type errors in downstream ops.
    """
    state_shardings: dict = {}
    for name, _ in module.named_parameters(recurse=False):
        state_shardings[name] = dense_param_placement(tp=Replicate())
    for name, _ in module.named_buffers(recurse=False):
        state_shardings[name] = dense_param_placement(tp=Replicate())

    if not wrap_inputs:
        return ShardingConfig(state_shardings=state_shardings)

    sig = inspect.signature(type(module).forward)
    arg_names = [p.name for p in sig.parameters.values() if p.name != "self"]
    in_shardings = {
        name: dense_activation_placement(tp=Replicate()) for name in arg_names
    }
    return ShardingConfig(
        state_shardings=state_shardings,
        in_src_shardings=in_shardings,
        in_dst_shardings=in_shardings,
    )


def _first_forward_arg(module: nn.Module) -> str:
    """Return the first positional arg name of ``module.forward``."""
    sig = inspect.signature(type(module).forward)
    for p in sig.parameters.values():
        if p.name != "self":
            return p.name
    raise ValueError(f"No positional args in {type(module).__name__}.forward")

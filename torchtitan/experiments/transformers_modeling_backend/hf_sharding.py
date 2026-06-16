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

import spmd_types as spmd
import torch.nn as nn

from torchtitan.models.common.decoder_sharding import (
    colwise_config,
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
    rowwise_config,
)
from torchtitan.protocols.sharding import ShardingConfig, SpmdLayout
from torchtitan.tools.logging import logger


def _sp_activation(*, enable_sp: bool) -> SpmdLayout:
    """Activation layout for the sequence-parallel region.

    When SP is enabled, the sequence dim is sharded across both CP and TP
    (``partition_spec=(DP, (CP, TP), None)``) — use the canonical
    ``dense_sequence_parallel_placement`` so the CP/TP shard ordering on tensor
    dim 1 is explicit. When SP is disabled, TP replicates (CP still seq-shards).
    """
    return (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.R)
    )


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
    loss_tp: spmd.PerMeshAxisSpmdType = spmd.S(-1) if enable_loss_parallel else spmd.R

    # Root-level modules — nn.Identity modules are skipped (no params).
    if model.tok_embeddings is not None and not isinstance(
        model.tok_embeddings, nn.Identity
    ):
        emb_state: dict = {"weight": dense_param_placement(tp=spmd.S(0))}
        for buf_name, _ in model.tok_embeddings.named_buffers(recurse=False):
            emb_state[buf_name] = dense_param_placement(tp=spmd.R)
        model.tok_embeddings._sharding_config = ShardingConfig(
            state_shardings=emb_state,
            in_src_shardings={"input": dense_activation_placement(tp=spmd.R)},
            in_dst_shardings={"input": dense_activation_placement(tp=spmd.R)},
            out_dst_shardings=_sp_activation(enable_sp=enable_sp),
        )

    if model.norm is not None and not isinstance(model.norm, nn.Identity):
        model.norm._sharding_config = _hf_norm_config(enable_sp=enable_sp)

    if model.lm_head is not None and not isinstance(model.lm_head, nn.Identity):
        model.lm_head._sharding_config = ShardingConfig(
            state_shardings={
                "weight": dense_param_placement(tp=spmd.S(0)),
                "bias": dense_param_placement(tp=spmd.S(0)),
            },
            in_src_shardings={
                "input": _sp_activation(enable_sp=enable_sp),
            },
            in_dst_shardings={
                "input": dense_activation_placement(tp=spmd.R),
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

    # Completeness backstop: every parameter/buffer-bearing module this function
    # is responsible for must have a sharding config, or it silently mixes a
    # plain tensor with a DTensor under TP and crashes deep in forward. Covers
    # root modules and every decoder layer; the native MoE subtree is configured
    # separately by set_moe_sharding_config, so it is excluded per layer.
    for name in ("tok_embeddings", "norm", "lm_head", "rotary_emb"):
        mod = getattr(model, name, None)
        if mod is not None and not isinstance(mod, nn.Identity):
            _assert_all_states_sharded(mod, path=name)
    for transformer_block in model.layers:
        moe = _moe_subtree(transformer_block)
        _assert_all_states_sharded(
            transformer_block,
            path=type(transformer_block).__name__,
            skip=(moe,) if moe is not None else (),
        )

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
    attn._sharding_config = ShardingConfig(
        in_src_shardings={
            "hidden_states": _sp_activation(enable_sp=enable_sp),
        },
        in_dst_shardings={
            "hidden_states": dense_activation_placement(tp=spmd.R),
        },
    )

    # Query projection. Detected independently of the KV path: a model may
    # use low-rank Q (q_a_proj/q_b_proj, e.g. DeepSeek-V3, GLM-4.7) or
    # full-rank Q (q_proj, e.g. GQA models and DeepSeek-V2-Lite).
    if hasattr(attn, "q_a_proj"):
        for name in ("q_a_proj", "q_a_layernorm"):
            mod = getattr(attn, name, None)
            if mod is not None:
                mod._sharding_config = _replicate_config(mod)
        if hasattr(attn, "q_b_proj"):
            attn.q_b_proj._sharding_config = colwise_config()
    elif hasattr(attn, "q_proj"):
        attn.q_proj._sharding_config = colwise_config()
    else:
        raise ValueError(
            f"{type(attn).__name__}: no recognized query projection "
            "(expected 'q_a_proj'/'q_b_proj' or 'q_proj'). "
            "Add a case in _set_layer_sharding_configs (hf_sharding.py)."
        )

    # Key/Value projection. Detected independently of Q: DeepSeek-V2-Lite has
    # full-rank Q but low-rank KV (kv_a_proj_with_mqa/kv_b_proj), so keying KV
    # off q_lora_rank misses it. MLA models replicate the low-rank down-proj
    # and column-shard the up-proj; GQA models column-shard k_proj/v_proj.
    if hasattr(attn, "kv_a_proj_with_mqa"):
        for name in ("kv_a_proj_with_mqa", "kv_a_layernorm"):
            mod = getattr(attn, name, None)
            if mod is not None:
                mod._sharding_config = _replicate_config(mod)
        if hasattr(attn, "kv_b_proj"):
            attn.kv_b_proj._sharding_config = colwise_config()
    elif hasattr(attn, "k_proj") and hasattr(attn, "v_proj"):
        attn.k_proj._sharding_config = colwise_config()
        attn.v_proj._sharding_config = colwise_config()
    else:
        raise ValueError(
            f"{type(attn).__name__}: no recognized key/value projection "
            "(expected 'kv_a_proj_with_mqa'/'kv_b_proj' or 'k_proj'/'v_proj'). "
            "Add a case in _set_layer_sharding_configs (hf_sharding.py)."
        )

    # O projection
    o_proj_name = "o_proj" if hasattr(attn, "o_proj") else "dense"
    getattr(attn, o_proj_name)._sharding_config = rowwise_config(output_sp=enable_sp)

    # Q/K norms (Qwen3) — weight replicated, activations stay heads-sharded.
    for norm_name in ("q_norm", "k_norm"):
        norm = getattr(attn, norm_name, None)
        if norm is not None:
            norm._sharding_config = _replicate_config(norm)

    # GLM-5 DSA indexer — a small auxiliary subtree with its own nested
    # projections (e.g. wq_b/wk). Replicating its weights is necessary but not
    # sufficient under TP: its @torch.no_grad() forward uses scatter_/index ops
    # (and the surrounding attention does index_mask.scatter_(topk_indices)),
    # which DTensor's eager dispatch cannot handle, so it crashes with a
    # "mixed Tensor and DTensor" error.
    #
    # TODO: this is a pure DTensor artifact. The fix is NOT a DTensor-era
    # workaround (NoParallel / local_map are both deprecated and being removed);
    # it is the ``spmd_backend="spmd_types"`` execution path, where state and
    # activations are plain local shards (no tensor subclass) and these
    # index/scatter ops run natively. The declarative ShardingConfig/SpmdLayout
    # this file emits is already backend-agnostic, so once spmd_types is wired
    # into the eager Module path this whole branch (and the fail-loud below) can
    # be deleted — the indexer needs no special handling on local tensors.
    # Until then, fail loud under TP; replicate (a no-op that resolves to no
    # mesh) otherwise so FSDP/EP keep working.
    if hasattr(attn, "indexer"):
        if enable_sp:
            raise NotImplementedError(
                f"{type(attn).__name__}: the DSA indexer is not supported under "
                "tensor parallelism with the DTensor sharding backend (its no_grad "
                "forward uses scatter_/index ops that DTensor cannot dispatch). "
                "This resolves under spmd_backend='spmd_types' (local tensors); "
                "until that is wired into the eager path, run this model with "
                "FSDP/EP and no TP."
            )
        for sub in attn.indexer.modules():
            sub._sharding_config = _replicate_config(sub)

    # V-norm (Gemma4) — parameter-free RMSNorm applied per-head
    if hasattr(attn, "v_norm"):
        attn.v_norm._sharding_config = _replicate_config(attn.v_norm)

    # --- Dense MLP (non-MoE layers only) ---
    if not getattr(layer, "moe_enabled", False):
        mlp_attr = "mlp" if hasattr(layer, "mlp") else "feed_forward"
        mlp = getattr(layer, mlp_attr)

        mlp_arg = _first_forward_arg(mlp)
        mlp._sharding_config = ShardingConfig(
            in_src_shardings={mlp_arg: _sp_activation(enable_sp=enable_sp)},
            in_dst_shardings={mlp_arg: dense_activation_placement(tp=spmd.R)},
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
        "weight": dense_param_placement(tp=spmd.R),
        "bias": dense_param_placement(tp=spmd.R),
    }
    if not enable_sp:
        return ShardingConfig(state_shardings=state)
    sp_layout = dense_sequence_parallel_placement()
    return ShardingConfig(
        state_shardings=state,
        in_src_shardings={"hidden_states": sp_layout},
        in_dst_shardings={"hidden_states": sp_layout},
        out_dst_shardings=sp_layout,
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
        state_shardings[name] = dense_param_placement(tp=spmd.R)
    for name, _ in module.named_buffers(recurse=False):
        state_shardings[name] = dense_param_placement(tp=spmd.R)

    if not wrap_inputs:
        return ShardingConfig(state_shardings=state_shardings)

    sig = inspect.signature(type(module).forward)
    arg_names = [p.name for p in sig.parameters.values() if p.name != "self"]
    in_shardings = {name: dense_activation_placement(tp=spmd.R) for name in arg_names}
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


def _moe_subtree(layer: nn.Module) -> nn.Module | None:
    """Return the native MoE module on a layer, or None for dense layers.

    The MoE subtree is configured by ``set_moe_sharding_config`` (not this
    file), so it is excluded from the completeness backstop below.
    """
    if not getattr(layer, "moe_enabled", False):
        return None
    for moe_attr in ("mlp", "feed_forward"):
        moe = getattr(layer, moe_attr, None)
        if moe is not None:
            return moe
    return None


def _assert_all_states_sharded(
    root: nn.Module, *, path: str, skip: tuple[nn.Module, ...] = ()
) -> None:
    """Raise if any param/buffer-bearing submodule of ``root`` lacks a config.

    A module that owns parameters or buffers but has no ``_sharding_config``
    keeps them as plain tensors; under TP they then meet DTensor activations
    and fail with a cryptic "mixed Tensor and DTensor" error deep in forward.
    This backstop turns that into a precise, setup-time error naming the
    undeclared module, so unhandled modules fail loud rather than silently
    slipping through the cases above.

    ``skip`` lists module subtrees configured elsewhere (e.g. the native MoE,
    handled by ``set_moe_sharding_config``); they and their descendants are not
    checked here.
    """
    skip_ids = {id(sub) for m in skip for sub in m.modules()}
    missing = []
    for name, sub in root.named_modules():
        if id(sub) in skip_ids:
            continue
        has_own_state = (
            next(sub.named_parameters(recurse=False), None) is not None
            or next(sub.named_buffers(recurse=False), None) is not None
        )
        if has_own_state and getattr(sub, "_sharding_config", None) is None:
            missing.append(f"{path}.{name}" if name else path)
    if missing:
        raise ValueError(
            f"Unsharded modules with no _sharding_config: {missing}. "
            "Every parameter/buffer-bearing module on the dense path needs a "
            "sharding config (shard it, or use _replicate_config). Add a case in "
            "set_hf_sharding_configs / _set_layer_sharding_configs (hf_sharding.py)."
        )

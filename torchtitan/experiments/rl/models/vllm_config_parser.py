# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Helpers for the torchtitan custom vLLM ``ConfigParser``.

The parser class itself is defined inside ``vllm_registry.registry_to_vllm``
so it can capture ``model_spec`` via closure, using the same pattern as the
dynamic model class.
"""

from __future__ import annotations

from typing import Any

from torchtitan.protocols.model_spec import ModelSpec


# Model-agnostic name used for vLLM model registration.
VLLM_MODEL_NAME = "TorchTitanCausalLM"

# Allowlist of keys the torchtitan ConfigParser accepts via
# ``EngineArgs(hf_overrides={...})``. Any other key raises ValueError at
# engine init.
_ALLOWED_TORCHTITAN_CONFIG_OVERRIDES = frozenset({"compile_config", "debug_config"})
TORCHTITAN_CONFIG_FORMAT = "torchtitan"


def model_spec_to_hf_config_dict(spec: ModelSpec) -> dict[str, Any]:
    """Build the HF-shaped config dict that vLLM's engine init reads.

    Field names match HF conventions because vLLM's engine reads them by
    hardcoded name (``vocab_size``, ``hidden_size``, ``num_attention_heads``,
    …) before any model class is constructed. ``PretrainedConfig`` stores any
    extra kwargs as attributes, so callers can pass torchtitan-specific
    runtime flags via ``EngineArgs(hf_overrides={...})`` and read them off
    ``hf_config`` later (e.g. ``compile_config``).
    """
    cfg = spec.model
    if not cfg.layers:
        raise ValueError(f"ModelSpec {spec.name!r} has no layers")
    layer0 = cfg.layers[0]
    attn = layer0.attention

    n_heads = attn.n_heads
    n_kv_heads = attn.n_kv_heads or n_heads
    head_dim = attn.head_dim if attn.head_dim is not None else cfg.dim // n_heads

    hf: dict[str, Any] = {
        # All torchtitan-backed models register under VLLM_MODEL_NAME.
        "architectures": [VLLM_MODEL_NAME],
        "model_type": "torchtitan",
        "vocab_size": cfg.vocab_size,
        "hidden_size": cfg.dim,
        "num_hidden_layers": len(cfg.layers),
        "num_attention_heads": n_heads,
        "num_key_value_heads": n_kv_heads,
        "head_dim": head_dim,
        "max_position_embeddings": cfg.rope.max_seq_len,
        "rope_theta": cfg.rope.theta,
        "rms_norm_eps": cfg.norm.eps,
        "tie_word_embeddings": getattr(cfg, "enable_weight_tying", False),
        # Best-effort special tokens: tokenizer_config.json overrides at
        # request-processing time.
        "bos_token_id": 0,
        "eos_token_id": 1,
    }

    ffn = getattr(layer0, "feed_forward", None)
    if ffn is not None:
        # FeedForward.Config holds w1/w2/w3 Linear.Config objects; the SwiGLU
        # hidden dim is w1.out_features (== w3.out_features == w2.in_features).
        hf["intermediate_size"] = ffn.w1.out_features

    moe = getattr(layer0, "moe", None)
    if moe is not None:
        hf["num_experts"] = moe.experts.num_experts
        # top_k lives on the router config, not the experts config.
        hf["num_experts_per_tok"] = moe.router.top_k
        hf["moe_intermediate_size"] = moe.experts.hidden_dim
        # vLLM's qwen3_moe model loader checks this for sparse layer placement.
        hf["decoder_sparse_step"] = 1
        hf.setdefault("norm_topk_prob", True)

    return hf


def apply_hf_overrides_to_config_dict(
    config_dict: dict[str, Any],
    hf_overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    """Validate and stamp ``EngineArgs(hf_overrides=...)`` onto the config dict.

    Rejects any key not in ``_ALLOWED_TORCHTITAN_CONFIG_OVERRIDES`` so the writer
    side (call site) and the reader side (wrapper) can't drift via typos or
    rogue literals. Each allowed key carries a ``dataclasses.asdict()`` of a
    torchtitan config section (e.g. ``CompileConfig``, ``DebugConfig``);
    downstream code reconstructs the typed object via ``Cls(**d)``.

    """
    if not hf_overrides:
        return config_dict
    unknown = set(hf_overrides) - _ALLOWED_TORCHTITAN_CONFIG_OVERRIDES
    if unknown:
        raise ValueError(
            f"Unknown torchtitan hf_overrides keys: {sorted(unknown)}. "
            f"Allowed: {sorted(_ALLOWED_TORCHTITAN_CONFIG_OVERRIDES)}."
        )
    config_dict.update(hf_overrides)
    return config_dict

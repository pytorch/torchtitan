# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Helpers for the torchtitan custom vLLM ``ConfigParser``.

The parser class itself is defined inside ``vllm_registry.registry`` so it
can capture ``model_spec`` via closure, using the same pattern as the
dynamic model class.
"""

from __future__ import annotations

from typing import Any

from torchtitan.protocols.model_spec import ModelSpec


TORCHTITAN_CONFIG_FORMAT = "torchtitan"

# Model-agnostic name used for vLLM model registration. The custom config
# parser writes this as ``architectures=[VLLM_MODEL_NAME]`` on the resulting
# PretrainedConfig, so callers don't need to pass
# hf_overrides={"architectures": [VLLM_MODEL_NAME]} to EngineArgs.
VLLM_MODEL_NAME = "TorchTitanCausalLM"


def model_spec_to_hf_config_dict(spec: ModelSpec) -> dict[str, Any]:
    """Build the HF-shaped config dict that vLLM's engine init reads.

    Field names match HF conventions because vLLM's engine reads them by
    hardcoded name (``vocab_size``, ``hidden_size``, ``num_attention_heads``,
    …) before any model class is constructed. ``PretrainedConfig`` stores any
    extra kwargs as attributes, so callers can pass torchtitan-specific
    runtime flags via ``EngineArgs(hf_overrides={...})`` and read them off
    ``hf_config`` later (e.g. ``skip_init_weights_load``).
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
        # torch_dtype is intentionally omitted: EngineArgs(dtype=...) always
        # sets it explicitly at engine init, and PretrainedConfig defaults to
        # None which is fine for that override path.
        # Best-effort special tokens; tokenizer_config.json overrides at
        # request-processing time.
        "bos_token_id": 0,
        "eos_token_id": 1,
    }

    ffn = getattr(layer0, "feed_forward", None)
    if ffn is not None:
        hf["intermediate_size"] = ffn.hidden_dim

    moe = getattr(layer0, "moe", None)
    if moe is not None:
        hf["num_experts"] = moe.experts.num_experts
        hf["num_experts_per_tok"] = moe.experts.top_k
        hf["moe_intermediate_size"] = moe.experts.hidden_dim
        # vLLM's qwen3_moe model loader checks this for sparse layer placement.
        hf["decoder_sparse_step"] = 1
        hf.setdefault("norm_topk_prob", True)

    return hf


def add_custom_fields_to_config_dict(
    config_dict: dict[str, Any],
    **custom_fields: Any,
) -> dict[str, Any]:
    """Layer caller-provided custom fields on top of the spec-derived dict.

    Use this to extend the HF config with torchtitan-specific or runtime-only
    fields the spec doesn't carry — anything ``PretrainedConfig`` should store
    as an attribute that downstream code (the wrapper, vLLM internals) reads
    via ``getattr(hf_config, "<key>", default)``.

    Mutates and returns the same dict for chaining. Existing keys are
    overwritten; pass with care if you might collide with an HF-required
    field set by ``model_spec_to_hf_config_dict``.

    Example::

        config_dict = model_spec_to_hf_config_dict(spec)
        config_dict = add_custom_fields_to_hf_config_dict(
            config_dict,
            torchtitan_build_tag="2026-05-06",
            extra_meta={"answer": 42},
        )
    """
    config_dict.update(custom_fields)
    return config_dict

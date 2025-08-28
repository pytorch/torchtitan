# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Key mappings between HuggingFace and TorchTitan formats for DeepSeek V3.

This module provides centralized key mappings that can be used by various components
including the state dict adapter, storage reader, and planner.
"""

# Mapping from HuggingFace checkpoint keys to TorchTitan keys
# {} placeholders are used for layer numbers and expert indices
HF_TO_TT_KEY_MAP = {
    "model.embed_tokens.weight": "tok_embeddings.weight",
    # Attention Module
    "model.layers.{}.self_attn.q_a_proj.weight": "layers.{}.attention.wq_a.weight",
    "model.layers.{}.self_attn.q_a_layernorm.weight": "layers.{}.attention.q_norm.weight",
    "model.layers.{}.self_attn.q_b_proj.weight": "layers.{}.attention.wq_b.weight",
    "model.layers.{}.self_attn.kv_a_proj_with_mqa.weight": "layers.{}.attention.wkv_a.weight",
    "model.layers.{}.self_attn.kv_a_layernorm.weight": "layers.{}.attention.kv_norm.weight",
    "model.layers.{}.self_attn.kv_b_proj.weight": "layers.{}.attention.wkv_b.weight",
    "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
    # MLP Module
    "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
    "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
    "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
    # Transformer Layer
    "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
    "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
    # MoE Module
    "model.layers.{}.mlp.experts.{}.gate_proj.weight": "layers.{}.moe.experts.w1",
    "model.layers.{}.mlp.experts.{}.up_proj.weight": "layers.{}.moe.experts.w3",
    "model.layers.{}.mlp.experts.{}.down_proj.weight": "layers.{}.moe.experts.w2",
    "model.layers.{}.mlp.gate.weight": "layers.{}.moe.router.gate.weight",
    "model.layers.{}.mlp.shared_experts.gate_proj.weight": "layers.{}.moe.shared_experts.w1.weight",
    "model.layers.{}.mlp.shared_experts.up_proj.weight": "layers.{}.moe.shared_experts.w3.weight",
    "model.layers.{}.mlp.shared_experts.down_proj.weight": "layers.{}.moe.shared_experts.w2.weight",
    "model.layers.{}.mlp.gate.e_score_correction_bias": "layers.{}.moe.expert_bias",
    "model.norm.weight": "norm.weight",
    "lm_head.weight": "output.weight",
}

# Reverse mapping from TorchTitan keys to HuggingFace keys
TT_TO_HF_KEY_MAP = {v: k for k, v in HF_TO_TT_KEY_MAP.items()}


def get_hf_to_tt_map():
    """Get the HuggingFace to TorchTitan key mapping."""
    return HF_TO_TT_KEY_MAP.copy()


def get_tt_to_hf_map():
    """Get the TorchTitan to HuggingFace key mapping."""
    return TT_TO_HF_KEY_MAP.copy()


def convert_hf_key_to_tt_key(hf_key: str) -> str | None:
    """
    Convert a HuggingFace key to TorchTitan key.

    Args:
        hf_key: HuggingFace format key

    Returns:
        Corresponding TorchTitan format key, or None if the key should be ignored
    """
    import re

    if "mlp.experts" in hf_key:
        abstract_key = re.sub(r"(\d+)", "{}", hf_key, count=2)
        layer_num, expert_num = re.findall(r"\d+", hf_key)

        if abstract_key in HF_TO_TT_KEY_MAP:
            new_key = HF_TO_TT_KEY_MAP[abstract_key]
            return new_key.format(layer_num)
    elif "layers" in hf_key:
        abstract_key = re.sub(r"(\d+)", "{}", hf_key, count=1)
        layer_num = re.search(r"\d+", hf_key).group(0)

        if abstract_key in HF_TO_TT_KEY_MAP:
            new_key = HF_TO_TT_KEY_MAP[abstract_key]
            return new_key.format(layer_num)
    elif hf_key in HF_TO_TT_KEY_MAP:
        new_key = HF_TO_TT_KEY_MAP[hf_key]
        return new_key
    return None

def convert_tt_key_to_hf_key(tt_key: str) -> str:
    """
    Convert a TorchTitan key to HuggingFace key.

    Args:
        tt_key: TorchTitan format key

    Returns:
        Corresponding HuggingFace format key, or None if it's a grouped expert tensor
        that doesn't have a direct HF equivalent
    """
    import re

    # Special handling for grouped expert tensors - these don't have direct HF equivalents
    # since they are stacked/grouped in TT but individual in HF
    if "moe.experts" in tt_key:
        # These are grouped expert tensors that need to be split into individual experts
        # Return None to indicate this key doesn't have a direct HF mapping
        return None
    elif "layers" in tt_key:
        abstract_key = re.sub(r"(\d+)", "{}", tt_key, count=1)
        layer_num = re.search(r"\d+", tt_key).group(0)
        new_key = TT_TO_HF_KEY_MAP[abstract_key]
        return new_key.format(layer_num)
    else:
        new_key = TT_TO_HF_KEY_MAP[tt_key]
        return new_key


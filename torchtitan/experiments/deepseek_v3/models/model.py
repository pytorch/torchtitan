# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This code is based on model definition of `deepseek-ai/DeepSeek-V3-Base` on
# Hugging Face Model Hub. Url:
# https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/blob/main/modeling_deepseek.py
# https://huggingface.co/deepseek-ai/DeepSeek-V3-Base/resolve/main/configuration_deepseek.py
#
# It has been modified from its original forms to accommodate naming convention
# and usage patterns of the TorchTitan project.

""" PyTorch DeepSeek model."""

# Define all exported symbols
__all__ = [
    # Normalization
    "RMSNorm",
    # Rotary Embeddings
    "RotaryEmbedding",
    "LinearScalingRotaryEmbedding",
    "DynamicNTKScalingRotaryEmbedding",
    "YarnRotaryEmbedding",
    "rotate_half",
    "apply_rotary_pos_emb",
    "yarn_find_correction_dim",
    "yarn_find_correction_range",
    "yarn_get_mscale",
    "yarn_linear_ramp_mask",
    # MLP and MoE
    "MLP",
    "MoE",
    "MoEGate",
    "get_group",
    # Attention
    "Attention",
    "_prepare_4d_causal_attention_mask",
    # Model Architecture
    "DecoderLayer",
    "DeepseekModel",
    "DeepseekForCausalLM",
    # Configuration
    "ModelArgs",
    "deepseek_config_registry",
    "deepseek_v2_lite_config",
    # Backward compatibility
    "model",
    "model_for_causal_lm",
]

# Re-export all components from their respective modules

from .attention import Attention
from .attn_mask_utils import _prepare_4d_causal_attention_mask
from .mlp import MLP
from .model_architecture import DecoderLayer, DeepseekForCausalLM, DeepseekModel
from .model_config import deepseek_config_registry, deepseek_v2_lite_config, ModelArgs
from .moe import get_group, MoE, MoEGate

# Re-export all components from their respective modules
from .normalization import RMSNorm
from .rope_embeddings import (
    apply_rotary_pos_emb,
    DynamicNTKScalingRotaryEmbedding,
    LinearScalingRotaryEmbedding,
    RotaryEmbedding,
    rotate_half,
    yarn_find_correction_dim,
    yarn_find_correction_range,
    yarn_get_mscale,
    yarn_linear_ramp_mask,
    YarnRotaryEmbedding,
)

# For backward compatibility
model = DeepseekModel
model_for_causal_lm = DeepseekForCausalLM

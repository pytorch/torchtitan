# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Nemotron-3 Nano State Dict Adapter

import logging
import re
from typing import Any

logger = logging.getLogger()

from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.protocols.state_dict_adapter import StateDictAdapter
from .model import Nemotron3NanoModel


class NemotronStateDictAdapter(StateDictAdapter):
    """Adapter for converting Nemotron-3 Nano checkpoints between HF and TorchTitan formats.
    
    Handles both Mamba layers (simplified) and Transformer layers with MoE routing.
    """
    
    def __init__(
        self,
        model_config: Nemotron3NanoModel.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)
        self.model_config = model_config
        self.hf_assets_path = hf_assets_path

        # Mapping from HuggingFace keys to TorchTitan keys
        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            # Transformer layers
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.qkv_linear.wq.weight",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.qkv_linear.wk.weight",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.qkv_linear.wv.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            # MoE routing
            "model.layers.{}.block_sparse_moe.gate.weight": "layers.{}.moe.router.gate.weight",
            # MoE experts
            "model.layers.{}.block_sparse_moe.experts.{}.w1.weight": "layers.{}.moe.routed_experts.inner_experts.w1_EFD",
            "model.layers.{}.block_sparse_moe.experts.{}.w2.weight": "layers.{}.moe.routed_experts.inner_experts.w2_EDF",
            "model.layers.{}.block_sparse_moe.experts.{}.w3.weight": "layers.{}.moe.routed_experts.inner_experts.w3_EFD",
            # Norms
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
        }

    def _permute(self, w, n_heads_arg, dim1=None, dim2=None):
        if dim1 is None:
            dim1 = w.shape[0]
        if dim2 is None:
            dim2 = w.shape[1]
        return (
            w.view(n_heads_arg, dim1 // n_heads_arg // 2, 2, dim2)
            .transpose(1, 2)
            .reshape(dim1, dim2)
            .clone()
        )

    def _reverse_permute(self, w, n_heads_arg, dim1=None, dim2=None):
        if dim1 is None:
            dim1 = w.shape[0]
        if dim2 is None:
            dim2 = w.shape[1]
        return (
            w.view(n_heads_arg, 2, dim1 // n_heads_arg // 2, dim2)
            .transpose(1, 2)
            .reshape(dim1, dim2)
        )

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert TorchTitan checkpoint to HuggingFace format."""
        attn = self.model_config.layers[0].attention
        n_heads = attn.n_heads
        n_kv_heads = attn.n_kv_heads if attn.n_kv_heads is not None else n_heads
        dim = self.model_config.dim
        head_dim = attn.head_dim or dim // n_heads
        hf_state_dict = {}

        to_hf_map = {v: k for k, v in self.from_hf_map.items() if v is not None}

        for key, value in state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)

                new_key = to_hf_map.get(abstract_key)
                if new_key is None:
                    continue

                # Apply HF permutation for Q/K weights
                if abstract_key == "layers.{}.attention.qkv_linear.wq.weight":
                    value = self._permute(value, n_heads)
                if abstract_key == "layers.{}.attention.qkv_linear.wk.weight":
                    key_value_dim = head_dim * n_kv_heads
                    value = self._permute(value, n_kv_heads, key_value_dim, dim)

                new_key = new_key.format(layer_num)
            else:
                new_key = to_hf_map.get(key)
                if new_key is None:
                    continue

            hf_state_dict[new_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert HuggingFace checkpoint to TorchTitan format."""
        self._validate_hf_rope_config(ComplexRoPE.Config)

        attn = self.model_config.layers[0].attention
        n_heads = attn.n_heads
        n_kv_heads = attn.n_kv_heads if attn.n_kv_heads is not None else n_heads
        dim = self.model_config.dim
        head_dim = attn.head_dim or dim // n_heads
        state_dict = {}

        for key, value in hf_state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)

                # Reverse-permute Q and K for RoPE compatibility
                if abstract_key == "model.layers.{}.self_attn.q_proj.weight":
                    value = self._reverse_permute(value, n_heads)
                if abstract_key == "model.layers.{}.self_attn.k_proj.weight":
                    key_value_dim = head_dim * n_kv_heads
                    value = self._reverse_permute(value, n_kv_heads, key_value_dim, dim)

                new_key = self.from_hf_map.get(abstract_key)
                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
            else:
                new_key = self.from_hf_map.get(key)
                if new_key is None:
                    continue

            state_dict[new_key] = value

        return state_dict

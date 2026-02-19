# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
from typing import Any

logger = logging.getLogger()

from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .model import Llama3Model


class Llama3StateDictAdapter(StateDictAdapter):
    def __init__(
        self,
        model_config: Llama3Model.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)

        self.model_config = model_config
        self.hf_assets_path = hf_assets_path
        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }

    # HuggingFace permutation function (exact copy from their conversion script)
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
        to_hf_map = {v: k for k, v in self.from_hf_map.items()}

        n_heads = self.model_config.layer.attention.n_heads
        n_kv_heads = (
            self.model_config.layer.attention.n_kv_heads
            # pyrefly: ignore [missing-attribute]
            if self.model_config.layer.attention.n_kv_heads is not None
            else n_heads
        )
        dim = self.model_config.dim
        head_dim = dim // n_heads
        hf_state_dict = {}

        for key, value in state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)
                new_key = to_hf_map[abstract_key]
                # We need to permute the weights in wq and wk layer in order to account for the difference between
                # the native Llama and huggingface RoPE implementation.
                if abstract_key == "layers.{}.attention.wq.weight":
                    value = self._permute(value, n_heads)
                if abstract_key == "layers.{}.attention.wk.weight":
                    # pyrefly: ignore [unsupported-operation]
                    key_value_dim = head_dim * n_kv_heads
                    value = self._permute(value, n_kv_heads, key_value_dim, dim)

                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
            else:
                new_key = to_hf_map[key]

            hf_state_dict[new_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        n_heads = self.model_config.layer.attention.n_heads
        n_kv_heads = (
            self.model_config.layer.attention.n_kv_heads
            # pyrefly: ignore [missing-attribute]
            if self.model_config.layer.attention.n_kv_heads is not None
            else n_heads
        )
        dim = self.model_config.dim
        head_dim = dim // n_heads
        state_dict = {}

        for key, value in hf_state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)
                new_key = self.from_hf_map[abstract_key]

                # We need to permute the weights in wq and wk layer in order to account for the difference between
                # the native Llama and huggingface RoPE implementation.
                if abstract_key == "model.layers.{}.self_attn.q_proj.weight":
                    value = self._reverse_permute(value, n_heads)
                if abstract_key == "model.layers.{}.self_attn.k_proj.weight":
                    # pyrefly: ignore [unsupported-operation]
                    key_value_dim = head_dim * n_kv_heads
                    value = self._reverse_permute(value, n_kv_heads, key_value_dim, dim)

                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
            else:
                new_key = self.from_hf_map[key]

            # pyrefly: ignore [unsupported-operation]
            state_dict[new_key] = value
        # pyrefly: ignore [bad-return]
        return state_dict

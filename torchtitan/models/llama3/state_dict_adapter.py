# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
import re
from typing import Any

import torch

logger = logging.getLogger()

from torchtitan.models.common.attention import FusedQKVLinear

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
        self.fuse_qkv = isinstance(
            model_config.layers[0].attention.qkv_linear, FusedQKVLinear.Config
        )

        if self.fuse_qkv:
            self.from_hf_map = {
                "model.embed_tokens.weight": "tok_embeddings.weight",
                "model.layers.{}.self_attn.q_proj.weight": None,
                "model.layers.{}.self_attn.k_proj.weight": None,
                "model.layers.{}.self_attn.v_proj.weight": None,
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
        else:
            self.from_hf_map = {
                "model.embed_tokens.weight": "tok_embeddings.weight",
                "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.qkv_linear.wq.weight",
                "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.qkv_linear.wk.weight",
                "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.qkv_linear.wv.weight",
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
        n_heads = self.model_config.layers[0].attention.n_heads
        n_kv_heads = (
            self.model_config.layers[0].attention.n_kv_heads
            # pyrefly: ignore [missing-attribute]
            if self.model_config.layers[0].attention.n_kv_heads is not None
            else n_heads
        )
        dim = self.model_config.dim
        head_dim = dim // n_heads
        hf_state_dict = {}

        if self.fuse_qkv:
            to_hf_map = {v: k for k, v in self.from_hf_map.items() if v is not None}
        else:
            to_hf_map = {v: k for k, v in self.from_hf_map.items()}

        for key, value in state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)

                if (
                    self.fuse_qkv
                    and abstract_key == "layers.{}.attention.qkv_linear.wqkv.weight"
                ):
                    # Split fused weight into separate Q, K, V for HF format
                    wq, wk, wv = self.fused_to_separate_qkv(
                        value, n_heads, n_kv_heads, head_dim
                    )
                    # Apply HF permutation
                    wq = self._permute(wq, n_heads)
                    key_value_dim = head_dim * n_kv_heads
                    wk = self._permute(wk, n_kv_heads, key_value_dim, dim)
                    hf_state_dict[
                        f"model.layers.{layer_num}.self_attn.q_proj.weight"
                    ] = wq
                    hf_state_dict[
                        f"model.layers.{layer_num}.self_attn.k_proj.weight"
                    ] = wk
                    hf_state_dict[
                        f"model.layers.{layer_num}.self_attn.v_proj.weight"
                    ] = wv
                    continue

                new_key = to_hf_map.get(abstract_key)
                if new_key is None:
                    continue

                if not self.fuse_qkv:
                    # Apply HF permutation for unfused Q/K weights
                    if abstract_key == "layers.{}.attention.qkv_linear.wq.weight":
                        value = self._permute(value, n_heads)
                    if abstract_key == "layers.{}.attention.qkv_linear.wk.weight":
                        # pyrefly: ignore [unsupported-operation]
                        key_value_dim = head_dim * n_kv_heads
                        value = self._permute(value, n_kv_heads, key_value_dim, dim)

                new_key = new_key.format(layer_num)
            else:
                if self.model_config.enable_weight_tying and key == "output.weight":
                    continue
                new_key = to_hf_map[key]

            hf_state_dict[new_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        if (
            self.model_config.enable_weight_tying
            and "lm_head.weight" not in hf_state_dict
        ):
            assert "model.embed_tokens.weight" in hf_state_dict
            hf_state_dict["lm_head.weight"] = hf_state_dict["model.embed_tokens.weight"]

        n_heads = self.model_config.layers[0].attention.n_heads
        n_kv_heads = (
            self.model_config.layers[0].attention.n_kv_heads
            # pyrefly: ignore [missing-attribute]
            if self.model_config.layers[0].attention.n_kv_heads is not None
            else n_heads
        )
        dim = self.model_config.dim
        head_dim = dim // n_heads
        state_dict = {}

        # Collect Q/K/V per layer, then fuse (only used when fuse_qkv=True)
        pending_qkv: dict[str, dict[str, torch.Tensor]] = {}

        for key, value in hf_state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)

                # Reverse-permute Q and K for RoPE compatibility
                if abstract_key == "model.layers.{}.self_attn.q_proj.weight":
                    value = self._reverse_permute(value, n_heads)
                if abstract_key == "model.layers.{}.self_attn.k_proj.weight":
                    # pyrefly: ignore [unsupported-operation]
                    key_value_dim = head_dim * n_kv_heads
                    value = self._reverse_permute(value, n_kv_heads, key_value_dim, dim)

                if self.fuse_qkv and abstract_key in (
                    "model.layers.{}.self_attn.q_proj.weight",
                    "model.layers.{}.self_attn.k_proj.weight",
                    "model.layers.{}.self_attn.v_proj.weight",
                ):
                    # Collect Q/K/V; fuse once all three are available
                    if layer_num not in pending_qkv:
                        pending_qkv[layer_num] = {}
                    proj = abstract_key.split(".")[-2]  # q_proj, k_proj, v_proj
                    pending_qkv[layer_num][proj] = value
                    if len(pending_qkv[layer_num]) == 3:
                        fused = self.separate_to_fused_qkv(
                            pending_qkv[layer_num]["q_proj"],
                            pending_qkv[layer_num]["k_proj"],
                            pending_qkv[layer_num]["v_proj"],
                            n_heads,
                            n_kv_heads,
                            head_dim,
                        )
                        state_dict[
                            f"layers.{layer_num}.attention.qkv_linear.wqkv.weight"
                        ] = fused
                        del pending_qkv[layer_num]
                    continue

                new_key = self.from_hf_map[abstract_key]
                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
            else:
                new_key = self.from_hf_map[key]

            # pyrefly: ignore [unsupported-operation]
            state_dict[new_key] = value

        if self.fuse_qkv and pending_qkv:
            raise ValueError(
                f"Incomplete Q/K/V projections for layers: {list(pending_qkv.keys())}"
            )

        # pyrefly: ignore [bad-return]
        return state_dict

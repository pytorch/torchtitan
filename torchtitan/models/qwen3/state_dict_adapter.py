# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This script is adapted from torchtitan/models/llama3/model/state_dict_adapter.py.

We can use this script to adapt the checkpoint from HF to the format that we can load into the torchtitan model and vice versa.
This can enable us to do a parity test with the HF implementation and make sure that our results are
aligned with the HF implementation.

"""
import re
from typing import Any

import torch
from torch.distributed.tensor import DTensor

from torchtitan.models.common.attention import FusedQKVLinear
from torchtitan.models.utils import MoEStateDictAdapter
from .model import Qwen3Model


class Qwen3StateDictAdapter(MoEStateDictAdapter):
    def __init__(self, model_config: Qwen3Model.Config, hf_assets_path: str | None):
        super().__init__(model_config, hf_assets_path)
        self.fuse_qkv = isinstance(
            model_config.layers[0].attention.qkv_linear, FusedQKVLinear.Config
        )

        if self.fuse_qkv:
            qkv_map = {
                "model.layers.{}.self_attn.q_proj.weight": None,
                "model.layers.{}.self_attn.k_proj.weight": None,
                "model.layers.{}.self_attn.v_proj.weight": None,
            }
        else:
            qkv_map = {
                "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.qkv_linear.wq.weight",
                "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.qkv_linear.wk.weight",
                "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.qkv_linear.wv.weight",
            }

        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            # Attention module
            **qkv_map,  # pyrefly: ignore [invalid-argument]
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
            "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
            "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            # MLP module for non-MoE
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            # Transformer layer
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            # MoE
            "model.layers.{}.mlp.experts.{}.gate_proj.weight": "layers.{}.moe.experts.w1",
            "model.layers.{}.mlp.experts.{}.up_proj.weight": "layers.{}.moe.experts.w3",
            "model.layers.{}.mlp.experts.{}.down_proj.weight": "layers.{}.moe.experts.w2",
            "model.layers.{}.mlp.gate.weight": "layers.{}.moe.router.gate.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }

    def _get_attention_dims(self) -> tuple[int, int, int]:
        """Return (n_heads, n_kv_heads, head_dim) from model config."""
        attn = self.model_config.layers[0].attention
        n_heads = attn.n_heads
        n_kv_heads = attn.n_kv_heads if attn.n_kv_heads is not None else n_heads
        head_dim = (
            attn.head_dim
            if attn.head_dim is not None
            else self.model_config.dim // n_heads
        )
        return n_heads, n_kv_heads, head_dim

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        1. Convert between the HF shape and the torchtitan shape.
        2. Split the GroupedExperts' weight into separate expert's wegiht.
        """
        if self.fuse_qkv:
            to_hf_map = {v: k for k, v in self.from_hf_map.items() if v is not None}
            n_heads, n_kv_heads, head_dim = self._get_attention_dims()
        else:
            to_hf_map = {v: k for k, v in self.from_hf_map.items()}
        hf_state_dict = {}

        for key, value in state_dict.items():
            if "moe.experts" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                if abstract_key not in to_hf_map:
                    continue
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)
                new_abstract_key = to_hf_map[abstract_key]

                # Store the GroupedExperts Weight metadata for from_hf()
                if isinstance(value, DTensor):
                    self.grouped_expert_weight_placements[
                        abstract_key
                    ] = value.placements
                    self.grouped_expert_weight_shape[abstract_key] = value.shape
                    self.grouped_expert_weight_mesh[abstract_key] = value.device_mesh

                    # Split GroupedExperts weight to local individual expert weights
                    local_expert_fqn = self._get_local_experts_weights(
                        new_abstract_key,
                        abstract_key,
                        layer_num,
                        value,
                    )
                    hf_state_dict.update(local_expert_fqn)

                else:
                    # keep this path for offline conversion
                    moe_layer = next(
                        l for l in self.model_config.layers if l.moe is not None
                    )
                    split_values = self._split_experts_weights(
                        value,
                        moe_layer.moe.num_experts,
                    )

                    for expert_num in range(moe_layer.moe.num_experts):
                        new_key = new_abstract_key.format(layer_num, expert_num)
                        hf_state_dict[new_key] = split_values[expert_num].squeeze()

            elif "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)

                if (
                    self.fuse_qkv
                    and abstract_key == "layers.{}.attention.qkv_linear.wqkv.weight"
                ):
                    wq, wk, wv = self.fused_to_separate_qkv(
                        value,
                        n_heads,  # pyrefly: ignore [unbound-name]
                        n_kv_heads,  # pyrefly: ignore [unbound-name]
                        head_dim,  # pyrefly: ignore [unbound-name]
                    )
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

                if abstract_key not in to_hf_map:
                    continue
                new_key = to_hf_map[abstract_key]
                new_key = new_key.format(layer_num)
                hf_state_dict[new_key] = value

            else:
                if key not in to_hf_map:
                    continue
                # pyrefly: ignore [missing-attribute]
                if self.model_config.enable_weight_tying and key == "output.weight":
                    continue
                new_key = to_hf_map[key]
                hf_state_dict[new_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        1. Convert between the HF shape and the torchtitan shape.
        2. Concate separate expert's wegiht into GroupedExperts' weight.
        """

        state_dict = {}
        expert_weights_by_layer = {}  # {layer: {abstract_key: {expert_id: tensor}}}
        # Collect Q/K/V per layer for fusing (only used when fuse_qkv=True)
        pending_qkv: dict[str, dict[str, torch.Tensor]] = {}

        if self.fuse_qkv:
            n_heads, n_kv_heads, head_dim = self._get_attention_dims()

        if (
            # pyrefly: ignore [missing-attribute]
            self.model_config.enable_weight_tying
            and "lm_head.weight" not in hf_state_dict
        ):
            assert "model.embed_tokens.weight" in hf_state_dict
            hf_state_dict["lm_head.weight"] = hf_state_dict["model.embed_tokens.weight"]

        for key, value in hf_state_dict.items():
            if "mlp.experts" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=2)
                layer_num, expert_num = re.findall(r"\d+", key)
                titan_abstract_key = self.from_hf_map[abstract_key]
                assert titan_abstract_key is not None
                new_key = titan_abstract_key.format(layer_num)

                # Store the expert's weight in expert_weights_by_layer for concatenating later.
                if layer_num not in expert_weights_by_layer:
                    expert_weights_by_layer[layer_num] = {}
                if titan_abstract_key not in expert_weights_by_layer[layer_num]:
                    expert_weights_by_layer[layer_num][titan_abstract_key] = {}
                expert_weights_by_layer[layer_num][titan_abstract_key][
                    int(expert_num)
                ] = value

                # Use stored metadata to decide path (online vs offline)
                # Online mode: local_experts_indices was populated during to_hf()
                if titan_abstract_key in self.local_experts_indices:
                    stacked_value = self._concatenate_expert_weights_dtensor(
                        expert_weights_by_layer,
                        titan_abstract_key,
                        layer_num,
                    )
                else:  # keep this path to be compatible with offline conversion
                    stacked_value = self._concatenate_expert_weights(
                        expert_weights_by_layer,
                        titan_abstract_key,
                        layer_num,
                        next(
                            l for l in self.model_config.layers if l.moe is not None
                        ).moe.num_experts,
                    )

                if stacked_value is not None:
                    state_dict[new_key] = stacked_value

            elif "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)

                if self.fuse_qkv and abstract_key in (
                    "model.layers.{}.self_attn.q_proj.weight",
                    "model.layers.{}.self_attn.k_proj.weight",
                    "model.layers.{}.self_attn.v_proj.weight",
                ):
                    if layer_num not in pending_qkv:
                        pending_qkv[layer_num] = {}
                    proj = abstract_key.split(".")[-2]  # q_proj, k_proj, v_proj
                    pending_qkv[layer_num][proj] = value
                    if len(pending_qkv[layer_num]) == 3:
                        fused = self.separate_to_fused_qkv(
                            pending_qkv[layer_num]["q_proj"],
                            pending_qkv[layer_num]["k_proj"],
                            pending_qkv[layer_num]["v_proj"],
                            n_heads,  # pyrefly: ignore [unbound-name]
                            n_kv_heads,  # pyrefly: ignore [unbound-name]
                            head_dim,  # pyrefly: ignore [unbound-name]
                        )
                        state_dict[
                            f"layers.{layer_num}.attention.qkv_linear.wqkv.weight"
                        ] = fused
                        del pending_qkv[layer_num]
                    continue

                new_key = self.from_hf_map[abstract_key]
                if new_key is None:
                    continue
                # pyrefly: ignore [missing-attribute]
                new_key = new_key.format(layer_num)
                state_dict[new_key] = value

            else:
                new_key = self.from_hf_map[key]
                # pyrefly: ignore [unsupported-operation]
                state_dict[new_key] = value

        if self.fuse_qkv and pending_qkv:
            raise ValueError(
                f"Incomplete Q/K/V projections for layers: {list(pending_qkv.keys())}"
            )

        return state_dict

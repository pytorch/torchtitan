# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections import defaultdict
import re
from typing import Any
import torch
from einops import rearrange

from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .args import Qwen3TransformerModelArgs


class Qwen3StateDictAdapter(StateDictAdapter):
    """
    Convert HuggingFace Qwen3 weights <-> TorchTitan Qwen3.

    Weights expected from HF model classes under keys like:
    - model.embed_tokens.weight
    - model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
    - model.layers.{i}.mlp.{gate,up,down}_proj.weight
    - model.layers.{i}.input_layernorm.weight
    - model.layers.{i}.post_attention_layernorm.weight
    - model.norm.weight
    - lm_head.weight
    """

    def __init__(self, model_args: Qwen3TransformerModelArgs, hf_assets_path: str | None):
        super().__init__(model_args, hf_assets_path)
        self.model_args = model_args
        # Mappings with a single layer placeholder: {...} -> {...}
        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            # Qwen-specific q/k RMSNorm within attention
            "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
            "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
            # rotary_emb.inv_freq exists in HF but not used in TT cis precompute
            "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            # Dense MLP (non-MoE) layers
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.mlp.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.mlp.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.mlp.w2.weight",
            # MoE gating (token router)
            "model.layers.{}.mlp.gate.weight": "layers.{}.mlp.router.gate.weight",
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }

        # Mappings with layer and expert placeholders
        self.from_hf_map_expert = {
            "model.layers.{}.mlp.experts.{}.gate_proj.weight": "layers.{}.mlp.experts.w1",
            "model.layers.{}.mlp.experts.{}.up_proj.weight": "layers.{}.mlp.experts.w3",
            "model.layers.{}.mlp.experts.{}.down_proj.weight": "layers.{}.mlp.experts.w2",
        }
        self.from_hf_map |= self.from_hf_map_expert

    def to_hf(self, state_dict: dict[str, Any], tie_word_embeddings: bool = False) -> dict[str, Any]:
        to_hf_map = {v: k for k, v in self.from_hf_map.items() if v is not None}
        experts_to_hf_map = {v: k for k, v in self.from_hf_map_expert.items() if v is not None}

        n_heads = self.model_args.n_heads
        n_kv_heads = (
            self.model_args.n_kv_heads
            if self.model_args.n_kv_heads is not None
            else n_heads
        )
        hf_state_dict = {}

        for key, value in state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                new_key = to_hf_map[abstract_key]
                # We need to permute the weights in wq and wk layer in order to account for the difference between
                # the native Llama and huggingface RoPE implementation.
                if abstract_key == "layers.{}.attention.wq.weight":
                    value = rearrange(value, "(nheads ropedim two) indim -> (nheads two ropedim) indim", nheads=n_heads, two=2)
                elif abstract_key == "layers.{}.attention.wk.weight":
                    value = rearrange(value, "(nheads ropedim two) indim -> (nheads two ropedim) indim", nheads=n_kv_heads, two=2)
                elif abstract_key == "layers.{}.attention.q_norm.weight":
                    value = rearrange(value, "(ropedim two) -> (two ropedim)", two=2)
                elif abstract_key == "layers.{}.attention.k_norm.weight":
                    value = rearrange(value, "(ropedim two) -> (two ropedim)", two=2)
                if abstract_key in experts_to_hf_map:
                    # Unstack the experts
                    for i in range(value.shape[0]):
                        # expert_value = value[i].permute(1,0).contiguous()
                        expert_value = value[i].contiguous()
                        expert_key = experts_to_hf_map[abstract_key].format(layer_num, i)
                        hf_state_dict[expert_key] = expert_value
                    continue

                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
            else:
                new_key = to_hf_map[key]

            hf_state_dict[new_key] = value

        if tie_word_embeddings:
            hf_state_dict["lm_head.weight"] = hf_state_dict["model.embed_tokens.weight"]

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any], tie_word_embeddings: bool = False) -> dict[str, Any]:
        state_dict: dict[str, Any] = {}
        # Pull shape parameters for permutation logic
        n_heads = self.model_args.n_heads
        n_kv_heads = self.model_args.n_kv_heads if self.model_args.n_kv_heads is not None else n_heads

        # Temporary storage for per-layer expert weights to stack later
        # layer_num -> expert_idx -> tensor
        layer_expert_buckets: dict[str, dict[int, torch.Tensor]] = defaultdict(lambda: defaultdict(list))

        for key, value in hf_state_dict.items():
            # Ensure we operate on CPU real tensors
            if isinstance(value, torch.Tensor):
                value = value.detach().to("cpu").contiguous()

            if key.startswith("model.layers."):
                layer_num = re.search(r"model\.layers\.(\d+)", key).group(1)
            else:
                layer_num = None

            if re.search(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)", key) is not None:
                expert_num = re.search(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)", key).group(2)
                expert_num = int(expert_num)
            else:
                expert_num = None

            # replace the number after model.layers. with {}, replace the number after experts. with {}
            abstract_key = re.sub(r"model\.layers\.(\d+)", r"model.layers.{}", key, count=1)
            abstract_key = re.sub(r".mlp\.experts\.(\d+)", r".mlp.experts.{}", abstract_key, count=1)

            if abstract_key in self.from_hf_map_expert.keys():
                assert expert_num is not None
                mapped_key = self.from_hf_map_expert[abstract_key]
                layer_expert_buckets[mapped_key.format(layer_num)][expert_num] = value
                # We will stack after this loop.
                continue
            assert expert_num is None
            
            if abstract_key == "model.layers.{}.self_attn.q_proj.weight":
                value = rearrange(value, "(nheads two ropedim) indim -> (nheads ropedim two) indim", nheads=n_heads, two=2)
            elif abstract_key == "model.layers.{}.self_attn.k_proj.weight":
                value = rearrange(value, "(nheads two ropedim) indim -> (nheads ropedim two) indim", nheads=n_kv_heads, two=2)
            elif abstract_key == "model.layers.{}.self_attn.q_norm.weight":
                value = rearrange(value, "(two ropedim) -> (ropedim two)", two=2)
            elif abstract_key == "model.layers.{}.self_attn.k_norm.weight":
                value = rearrange(value, "(two ropedim) -> (ropedim two)", two=2)

            mapped_key = self.from_hf_map[abstract_key]
            assert layer_num is not None or '{}' not in mapped_key
            state_dict[mapped_key.format(layer_num)] = value

        for mapped_key, proj_dicts in layer_expert_buckets.items():
            experts = [proj_dicts[i] for i in sorted(proj_dicts.keys())]
            stacked = torch.stack(experts, dim=0)
            # state_dict[mapped_key] = stacked.permute(0, 2, 1).contiguous()
            state_dict[mapped_key] = stacked.contiguous()

        if tie_word_embeddings:
            state_dict["output.weight"] = state_dict["tok_embeddings.weight"]

        return state_dict


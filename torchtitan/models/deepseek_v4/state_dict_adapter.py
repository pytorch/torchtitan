# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Any

import torch

from torchtitan.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter

from .model import DeepSeekV4Model


class DeepSeekV4StateDictAdapter(DeepSeekV3StateDictAdapter):
    hf_experts_key_fragment = "ffn.experts"

    def __init__(
        self,
        model_config: DeepSeekV4Model.Config,
        hf_assets_path: str | None,
    ):
        # V4 configs reuse the V3 decoder layout, so the V3 adapter's init
        # accepts them; the narrower subclass type trips pyrefly here.
        # pyrefly: ignore [bad-argument-type]
        super().__init__(model_config, hf_assets_path)

        self.from_hf_map = {
            "embed.weight": "tok_embeddings.weight",
            # Attention
            "layers.{}.attn.attn_sink": "layers.{}.attention.attn_sink.weight",
            "layers.{}.attn.kv_norm.weight": "layers.{}.attention.kv_norm.weight",
            "layers.{}.attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
            "layers.{}.attn.wo_a.weight": "layers.{}.attention.wo_a.weight",
            "layers.{}.attn.wo_b.weight": "layers.{}.attention.wo_b.weight",
            "layers.{}.attn.wkv.weight": "layers.{}.attention.wkv.weight",
            "layers.{}.attn.wq_a.weight": "layers.{}.attention.wq_a.weight",
            "layers.{}.attn.wq_b.weight": "layers.{}.attention.wq_b.weight",
            # Norms
            "layers.{}.attn_norm.weight": "layers.{}.attention_norm.weight",
            "layers.{}.ffn_norm.weight": "layers.{}.ffn_norm.weight",
            # MoE
            "layers.{}.ffn.experts.{}.w1.weight": "layers.{}.moe.routed_experts.inner_experts.w1_EFD",
            "layers.{}.ffn.experts.{}.w3.weight": "layers.{}.moe.routed_experts.inner_experts.w3_EFD",
            "layers.{}.ffn.experts.{}.w2.weight": "layers.{}.moe.routed_experts.inner_experts.w2_EDF",
            "layers.{}.ffn.gate.weight": "layers.{}.moe.router.gate.weight",
            "layers.{}.ffn.gate.bias": "layers.{}.moe.expert_bias_E",
            "layers.{}.ffn.shared_experts.w1.weight": "layers.{}.moe.shared_experts.w1.weight",
            "layers.{}.ffn.shared_experts.w3.weight": "layers.{}.moe.shared_experts.w3.weight",
            "layers.{}.ffn.shared_experts.w2.weight": "layers.{}.moe.shared_experts.w2.weight",
            # mHC
            "layers.{}.hc_attn_base": "layers.{}.hc_attn_pre.hc_base",
            "layers.{}.hc_attn_fn": "layers.{}.hc_attn_pre.hc_fn",
            "layers.{}.hc_attn_scale": "layers.{}.hc_attn_pre.hc_scale",
            "layers.{}.hc_ffn_base": "layers.{}.hc_ffn_pre.hc_base",
            "layers.{}.hc_ffn_fn": "layers.{}.hc_ffn_pre.hc_fn",
            "layers.{}.hc_ffn_scale": "layers.{}.hc_ffn_pre.hc_scale",
            # MTP
            "layers.{}.enorm.weight": "layers.{}.enorm.weight",
            "layers.{}.hnorm.weight": "layers.{}.hnorm.weight",
            "layers.{}.e_proj.weight": "layers.{}.e_proj.weight",
            "layers.{}.h_proj.weight": "layers.{}.h_proj.weight",
            "layers.{}.norm.weight": "layers.{}.mtp_norm.weight",
            "layers.{}.hc_head_base": "layers.{}.hc_head.hc_base",
            "layers.{}.hc_head_fn": "layers.{}.hc_head.hc_fn",
            "layers.{}.hc_head_scale": "layers.{}.hc_head.hc_scale",
            "hc_head_base": "hc_head.hc_base",
            "hc_head_fn": "hc_head.hc_fn",
            "hc_head_scale": "hc_head.hc_scale",
            "norm.weight": "norm.weight",
            "head.weight": "lm_head.weight",
        }

        self.compress_ratios = model_config.compress_ratios
        for layer_id in range(model_config.n_layers):
            cr = self.compress_ratios[layer_id]
            if cr != 1:
                comp = "compressor" if cr == 4 else "compressor_128"
                self.from_hf_map.update(
                    {
                        f"layers.{layer_id}.attn.compressor.ape": (
                            f"layers.{layer_id}.attention.{comp}.ape"
                        ),
                        f"layers.{layer_id}.attn.compressor.norm.weight": (
                            f"layers.{layer_id}.attention.{comp}.norm.weight"
                        ),
                        f"layers.{layer_id}.attn.compressor.wgate.weight": (
                            f"layers.{layer_id}.attention.{comp}.wgate.weight"
                        ),
                        f"layers.{layer_id}.attn.compressor.wkv.weight": (
                            f"layers.{layer_id}.attention.{comp}.wkv.weight"
                        ),
                    }
                )
            if cr == 4:
                self.from_hf_map.update(
                    {
                        f"layers.{layer_id}.attn.indexer.compressor.ape": (
                            f"layers.{layer_id}.attention.indexer.compressor.ape"
                        ),
                        f"layers.{layer_id}.attn.indexer.compressor.norm.weight": (
                            f"layers.{layer_id}.attention.indexer.compressor.norm.weight"
                        ),
                        f"layers.{layer_id}.attn.indexer.compressor.wgate.weight": (
                            f"layers.{layer_id}.attention.indexer.compressor.wgate.weight"
                        ),
                        f"layers.{layer_id}.attn.indexer.compressor.wkv.weight": (
                            f"layers.{layer_id}.attention.indexer.compressor.wkv.weight"
                        ),
                        f"layers.{layer_id}.attn.indexer.wq_b.weight": (
                            f"layers.{layer_id}.attention.indexer.wq_b.weight"
                        ),
                        f"layers.{layer_id}.attn.indexer.weights_proj.weight": (
                            f"layers.{layer_id}.attention.indexer.weights_proj.weight"
                        ),
                    }
                )
            if layer_id < model_config.layers[0].moe.router.n_hash_layers:
                self.from_hf_map.update(
                    {
                        f"layers.{layer_id}.ffn.gate.tid2eid": f"layers.{layer_id}.moe.router.tid2eid",
                    }
                )

    @staticmethod
    def _abstract_key(key: str, count: int) -> str:
        return re.sub(r"(\d+)", "{}", key, count=count)

    @staticmethod
    def _first_number(key: str) -> str:
        match = re.search(r"\d+", key)
        if match is None:
            raise ValueError(f"Expected a layer number in key: {key}")
        return match.group(0)

    def _map_layer(self, key: str, mapping: dict[str, str]) -> str:
        return mapping[self._abstract_key(key, count=1)].format(self._first_number(key))

    @staticmethod
    def _is_v4_special_titan_key(key: str) -> bool:
        return any(t in key for t in ("compressor", "indexer", "tid2eid"))

    @staticmethod
    def _is_v4_special_hf_key(key: str) -> bool:
        return any(t in key for t in ("compressor", "indexer", "tid2eid"))

    def _can_delegate_titan_key(self, key: str, to_hf_map: dict[str, str]) -> bool:
        if key in to_hf_map or "moe.routed_experts.inner_experts" in key:
            return True
        if key.startswith("mtp_layers."):
            abstract_key = self._abstract_key(key, count=1).replace(
                "mtp_layers.{}.",
                "layers.{}.",
                1,
            )
            return abstract_key in to_hf_map
        if "layers" in key:
            return self._abstract_key(key, count=1) in to_hf_map
        return False

    def _can_delegate_hf_key(self, key: str) -> bool:
        if key in self.from_hf_map or self.hf_experts_key_fragment in key:
            return True
        if "layers" in key:
            count = 2 if self.hf_experts_key_fragment in key else 1
            return self._abstract_key(key, count=count) in self.from_hf_map
        return False

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        to_hf_map = {v: k for k, v in self.from_hf_map.items()}
        hf_state_dict = {}
        delegated_state_dict = {}

        for key, value in state_dict.items():
            if self._is_v4_special_titan_key(key) and key in to_hf_map:
                new_key = to_hf_map[key]
                if "tid2eid" in key:
                    value = value.to(torch.float32)
                hf_state_dict[new_key] = value
            elif "attention.attn_sink.weight" in key:
                hf_state_dict[self._map_layer(key, to_hf_map)] = value.squeeze(-1)
            elif self._can_delegate_titan_key(key, to_hf_map):
                delegated_state_dict[key] = value
            else:
                hf_state_dict[key] = value

        if delegated_state_dict:
            hf_state_dict.update(super().to_hf(delegated_state_dict))
        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict = {}
        delegated_hf_state_dict = {}

        for key, value in hf_state_dict.items():
            if self._is_v4_special_hf_key(key) and key in self.from_hf_map:
                new_key = self.from_hf_map[key]
                if "tid2eid" in key:
                    value = value.to(torch.int64)
                state_dict[new_key] = value
            elif "attn.attn_sink" in key:
                state_dict[self._map_layer(key, self.from_hf_map)] = value.unsqueeze(-1)
            elif self._can_delegate_hf_key(key):
                delegated_hf_state_dict[key] = value
            else:
                state_dict[key] = value

        if delegated_hf_state_dict:
            state_dict.update(super().from_hf(delegated_hf_state_dict))
        return state_dict

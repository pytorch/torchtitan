# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Any

import torch
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.tensor import DTensor

from torchtitan.models.utils import MoEStateDictAdapter

from .model import DeepSeekV4Model


class DeepSeekV4StateDictAdapter(MoEStateDictAdapter):
    def __init__(
        self,
        model_config: DeepSeekV4Model.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)

        self.from_hf_map = {
            "embed.weight": "tok_embeddings.weight",
            # Attention (monolithic)
            "layers.{}.attn.attn_sink": "layers.{}.attention.attn_sink.weight",
            "layers.{}.attn.kv_norm.weight": "layers.{}.attention.kv_norm.weight",
            "layers.{}.attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
            "layers.{}.attn.wo_a": "layers.{}.attention.wo_a.weight",
            "layers.{}.attn.wo_b.weight": "layers.{}.attention.wo_b.weight",
            "layers.{}.attn.wkv.weight": "layers.{}.attention.wkv.weight",
            "layers.{}.attn.wq_a.weight": "layers.{}.attention.wq_a.weight",
            "layers.{}.attn.wq_b.weight": "layers.{}.attention.wq_b.weight",
            # Norms
            "layers.{}.attn_norm.weight": "layers.{}.attention_norm.weight",
            "layers.{}.ffn_norm.weight": "layers.{}.ffn_norm.weight",
            # MoE
            "layers.{}.ffn.experts.{}.w1.weight": "layers.{}.moe.experts.w1_EFD",
            "layers.{}.ffn.experts.{}.w3.weight": "layers.{}.moe.experts.w3_EFD",
            "layers.{}.ffn.experts.{}.w2.weight": "layers.{}.moe.experts.w2_EDF",
            "layers.{}.ffn.gate.weight": "layers.{}.moe.router.gate.weight",
            "layers.{}.ffn.gate.bias": "layers.{}.moe.expert_bias_E",
            "layers.{}.ffn.shared_experts.w1.weight": "layers.{}.moe.shared_experts.w1.weight",
            "layers.{}.ffn.shared_experts.w3.weight": "layers.{}.moe.shared_experts.w3.weight",
            "layers.{}.ffn.shared_experts.w2.weight": "layers.{}.moe.shared_experts.w2.weight",
            # mHC
            "layers.{}.hc_attn_base": "layers.{}.hc_attn_base",
            "layers.{}.hc_attn_fn": "layers.{}.hc_attn_fn",
            "layers.{}.hc_attn_scale": "layers.{}.hc_attn_scale",
            "layers.{}.hc_ffn_base": "layers.{}.hc_ffn_base",
            "layers.{}.hc_ffn_fn": "layers.{}.hc_ffn_fn",
            "layers.{}.hc_ffn_scale": "layers.{}.hc_ffn_scale",
            "hc_head_base": "hc_head_base",
            "hc_head_fn": "hc_head_fn",
            "hc_head_scale": "hc_head_scale",
            "norm.weight": "norm.weight",
            "head.weight": "lm_head.weight",
        }

        self.compress_ratios = model_config.compress_ratios
        for layer_id in range(model_config.n_layers):
            cr = self.compress_ratios[layer_id]
            if cr != 1:
                comp = "compressor" if cr == 4 else "compressor_128"
                self.from_hf_map.update({
                    f"layers.{layer_id}.attn.compressor.ape": (
                        f"layers.{layer_id}.attention.{comp}.ape.weight"
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
                })
            if cr == 4:
                self.from_hf_map.update({
                    f"layers.{layer_id}.attn.indexer.compressor.ape": (
                        f"layers.{layer_id}.attention.indexer.compressor.ape.weight"
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
                })
            if layer_id < model_config.layers[0].moe.router.n_hash_layers:
                self.from_hf_map.update(
                    {
                        f"layers.{layer_id}.ffn.gate.tid2eid": f"layers.{layer_id}.moe.router.tid2eid",
                    }
                )

    def get_hf_storage_reader(
        self, path: str, from_quantized: bool = False
    ) -> HuggingFaceStorageReader:
        if from_quantized:
            from torch.distributed.checkpoint.quantized_hf_storage import (
                QuantizedHuggingFaceStorageReader,
            )
            return QuantizedHuggingFaceStorageReader(
                path=path,
                target_dtype=torch.float32,
                block_size=128,
                thread_count=4,
            )
        return HuggingFaceStorageReader(path)

    @staticmethod
    def _abstract_key(key: str, count: int) -> str:
        return re.sub(r"(\d+)", "{}", key, count=count)

    @staticmethod
    def _first_number(key: str) -> str:
        return re.search(r"\d+", key).group(0)

    def _map_layer(self, key: str, mapping: dict[str, str]) -> str:
        return mapping[self._abstract_key(key, count=1)].format(
            self._first_number(key)
        )

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        to_hf_map = {v: k for k, v in self.from_hf_map.items()}
        hf_state_dict = {}

        for key, value in state_dict.items():
            if any(t in key for t in ("compressor", "indexer", "tid2eid")):
                new_key = to_hf_map[key]
                if "tid2eid" in key:
                    value = value.to(torch.float32)
                hf_state_dict[new_key] = value

            elif "moe.experts" in key:
                abstract_key = self._abstract_key(key, count=1)
                layer_num = self._first_number(key)
                new_abstract = to_hf_map[abstract_key]

                if isinstance(value, DTensor):
                    self.grouped_expert_weight_placements[abstract_key] = value.placements
                    self.grouped_expert_weight_shape[abstract_key] = value.shape
                    self.grouped_expert_weight_mesh[abstract_key] = value.device_mesh
                    local_fqn = self._get_local_experts_weights(
                        new_abstract, abstract_key, layer_num, value,
                    )
                    hf_state_dict.update(local_fqn)
                else:
                    num_experts = next(
                        l for l in self.model_config.layers if l.moe is not None
                    ).moe.num_experts
                    split_values = self._split_experts_weights(value, num_experts)
                    for e in range(num_experts):
                        hf_state_dict[new_abstract.format(layer_num, e)] = split_values[e].squeeze()

            elif "layers" in key:
                hf_state_dict[self._map_layer(key, to_hf_map)] = value

            else:
                if key in to_hf_map:
                    hf_state_dict[to_hf_map[key]] = value
                else:
                    hf_state_dict[key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict = {}
        expert_weights = {}

        for key, value in hf_state_dict.items():
            if any(t in key for t in ("compressor", "indexer", "tid2eid")):
                new_key = self.from_hf_map[key]
                if "tid2eid" in key:
                    value = value.to(torch.int64)
                state_dict[new_key] = value

            elif "ffn.experts" in key:
                abstract_key = self._abstract_key(key, count=2)
                layer_num, expert_num, _ = re.findall(r"\d+", key)
                titan_abstract = self.from_hf_map[abstract_key]
                new_key = titan_abstract.format(layer_num)

                if layer_num not in expert_weights:
                    expert_weights[layer_num] = {}
                if titan_abstract not in expert_weights[layer_num]:
                    expert_weights[layer_num][titan_abstract] = {}
                expert_weights[layer_num][titan_abstract][int(expert_num)] = value

                if titan_abstract in self.local_experts_indices:
                    stacked = self._concatenate_expert_weights_dtensor(
                        expert_weights, titan_abstract, layer_num,
                    )
                else:
                    num_experts = next(
                        l for l in self.model_config.layers if l.moe is not None
                    ).moe.num_experts
                    stacked = self._concatenate_expert_weights(
                        expert_weights, titan_abstract, layer_num, num_experts,
                    )
                if stacked is not None:
                    state_dict[new_key] = stacked

            elif "layers" in key:
                state_dict[self._map_layer(key, self.from_hf_map)] = value

            else:
                if key in self.from_hf_map:
                    state_dict[self.from_hf_map[key]] = value
                else:
                    state_dict[key] = value

        return state_dict

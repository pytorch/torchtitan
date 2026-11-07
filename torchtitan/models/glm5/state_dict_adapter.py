# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GLM checkpoint names on the shared DeepSeek and fused-linear adapters.

GLM stores routed gate/up weights in the same fused layout as native W13.
Keep that tensor intact, including its DTensor placements, rather than splitting
and reassembling TP shards with model-specific all-to-all collectives.
"""

from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter

if TYPE_CHECKING:
    from .model import Glm5Model


class Glm5StateDictAdapter(DeepSeekV3StateDictAdapter):
    """Bidirectional adapter for Transformers GLM-MoE-DSA checkpoints."""

    HF_TO_TITAN_TOP_LEVEL = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    HF_TO_TITAN_INDEXER = {
        "self_attn.indexer.wq_b.weight": "attention.indexer.wq_b.weight",
        "self_attn.indexer.wk.weight": "attention.indexer.wk.weight",
        "self_attn.indexer.k_norm.weight": "attention.indexer.k_norm.weight",
        "self_attn.indexer.k_norm.bias": "attention.indexer.k_norm.bias",
        "self_attn.indexer.weights_proj.weight": "attention.indexer.weights_proj.weight",
        "mlp.experts.gate_up_proj": "feed_forward.routed_experts.w13.weight",
        "mlp.experts.down_proj": "feed_forward.routed_experts.w2.weight",
    }
    HF_TO_TITAN_MTP = {
        "enorm.weight": "enorm.weight",
        "hnorm.weight": "hnorm.weight",
        "eh_proj.weight": "eh_proj.weight",
        "shared_head.norm.weight": "mtp_norm.weight",
    }
    _IGNORED_HF_KEYS = {
        "model.rotary_emb.inv_freq",
        "model.rotary_emb.original_inv_freq",
    }
    _HF_LAYER_KEY = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
    _TITAN_LAYER_KEY = re.compile(r"^layers\.(\d+)\.(.+)$")
    _TITAN_MTP_KEY = re.compile(r"^mtp_layers\.(\d+)\.(.+)$")
    _TITAN_GATE_UP = "feed_forward.routed_experts.w13.weight"
    _TITAN_DOWN = "feed_forward.routed_experts.w2.weight"

    def __init__(
        self,
        model_config: Glm5Model.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)
        self.glm_config = model_config
        self.HF_TO_TITAN_LAYER = {
            hf.removeprefix("model.layers.{}."): native.removeprefix("layers.{}.")
            for hf, native in self.from_hf_map.items()
            if hf.startswith("model.layers.{}.") and "mlp.experts.{}." not in hf
        }
        self.HF_TO_TITAN_LAYER.update(self.HF_TO_TITAN_INDEXER)
        self.from_hf_map.update(
            {
                f"model.layers.{{}}.{hf}": f"layers.{{}}.{native}"
                for hf, native in self.HF_TO_TITAN_INDEXER.items()
            }
        )
        self._to_hf_top_level = {v: k for k, v in self.HF_TO_TITAN_TOP_LEVEL.items()}
        self._to_hf_layer = {v: k for k, v in self.HF_TO_TITAN_LAYER.items()}
        self._to_hf_mtp = {v: k for k, v in self.HF_TO_TITAN_MTP.items()}

    def _layer_config(self, layer_index: int):
        if 0 <= layer_index < len(self.glm_config.layers):
            return self.glm_config.layers[layer_index]
        mtp_index = layer_index - len(self.glm_config.layers)
        if 0 <= mtp_index < len(self.glm_config.mtp_layers):
            return self.glm_config.mtp_layers[mtp_index]
        raise KeyError(f"unmapped HF key: layer index {layer_index} is unsupported")

    def from_hf(
        self, hf_state_dict: dict[str, Any], *, is_quantized_layout: bool = False
    ) -> dict[str, Any]:
        if is_quantized_layout:
            raise ValueError(
                "GLM supports BF16/FP32 checkpoints, not quantized layouts"
            )
        self._validate_hf_rope_config(ComplexRoPE.Config)
        result = {}
        for key, value in hf_state_dict.items():
            if key in self._IGNORED_HF_KEYS:
                continue
            native = self.HF_TO_TITAN_TOP_LEVEL.get(key)
            if native is None:
                match = self._HF_LAYER_KEY.fullmatch(key)
                if match is None:
                    raise KeyError(f"unmapped HF key: {key}")
                index, suffix = int(match.group(1)), match.group(2)
                self._layer_config(index)
                is_mtp = index >= len(self.glm_config.layers)
                mapping = (
                    {**self.HF_TO_TITAN_LAYER, **self.HF_TO_TITAN_MTP}
                    if is_mtp
                    else self.HF_TO_TITAN_LAYER
                )
                if suffix not in mapping:
                    raise KeyError(f"unmapped HF key: {key}")
                prefix = (
                    f"mtp_layers.{index - len(self.glm_config.layers)}"
                    if is_mtp
                    else f"layers.{index}"
                )
                native = f"{prefix}.{mapping[suffix]}"
            if native.endswith("routed_experts.w13.weight"):
                value = value.unflatten(1, (2, -1))
            result[native] = value
        return self._native_fused_linears_from_hf(result)

    def to_hf(
        self, state_dict: dict[str, Any], *, to_quantized_layout: bool = False
    ) -> dict[str, Any]:
        if to_quantized_layout:
            raise ValueError(
                "GLM supports BF16/FP32 checkpoints, not quantized layouts"
            )
        result = {}
        for key, value in self._native_fused_linears_to_hf(state_dict).items():
            hf = self._to_hf_top_level.get(key)
            if hf is None:
                match = self._TITAN_LAYER_KEY.fullmatch(key)
                mtp_match = self._TITAN_MTP_KEY.fullmatch(key)
                match = mtp_match or match
                if match is None:
                    raise KeyError(f"unmapped TorchTitan key: {key}")
                index, suffix = int(match.group(1)), match.group(2)
                if mtp_match is not None:
                    index += len(self.glm_config.layers)
                self._layer_config(index)
                mapping = (
                    {**self._to_hf_layer, **self._to_hf_mtp}
                    if mtp_match
                    else self._to_hf_layer
                )
                if suffix not in mapping:
                    raise KeyError(f"unmapped TorchTitan key: {key}")
                hf = f"model.layers.{index}.{mapping[suffix]}"
            result[hf] = (
                value.flatten(1, 2)
                if key.endswith("routed_experts.w13.weight")
                else value
            )
        return result

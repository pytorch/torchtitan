# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Flash-Next checkpoint mapping, including its 128 row-wise n-gram shards."""

# N = embedding rows, D = embedding features.

from __future__ import annotations

import logging
import re
from typing import Any, TYPE_CHECKING

import torch

from torchtitan.models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter

if TYPE_CHECKING:
    from .model import Qwen38FlashNextModel

logger = logging.getLogger(__name__)


class Qwen38FlashNextStateDictAdapter(Qwen35StateDictAdapter):
    def __init__(
        self, model_config: Qwen38FlashNextModel.Config, hf_assets_path: str | None
    ):
        super().__init__(model_config, hf_assets_path)
        self.flash_config = model_config
        prefix = self.hf_language_model_prefix
        for norm in ("input_layernorm", "post_attention_layernorm"):
            self.from_hf_map.pop(f"{prefix}.layers.{{}}.{norm}.weight")
        self.from_hf_map.pop(f"{prefix}.norm.weight")
        for connection in ("attn_hyper_connection", "mlp_hyper_connection"):
            for field in (
                "hc_norm",
                "input_mix_weight_down",
                "input_mix_weight_up",
                "block_inject_weight",
            ):
                self.from_hf_map[
                    f"{prefix}.layers.{{}}.{connection}.{field}.weight"
                ] = f"layers.{{}}.{connection}.{field}.weight"
        for field in ("hc_norm", "input_mix_weight_down", "input_mix_weight_up"):
            self.from_hf_map[
                f"{prefix}.hyper_connection_mixer.{field}.weight"
            ] = f"hyper_connection_mixer.{field}.weight"
        for field in ("index_qk_proj", "q_layernorm", "k_layernorm"):
            self.from_hf_map[
                f"{prefix}.layers.{{}}.self_attn.indexer.{field}.weight"
            ] = f"layers.{{}}.attention.indexer.{field}.weight"
        for field in (
            "key_proj",
            "value_proj",
            "norm_key",
            "norm_query",
            "norm_conv",
            "conv1d",
        ):
            self.from_hf_map[
                f"{prefix}.layers.{{}}.ple.{field}.weight"
            ] = f"layers.{{}}.ple.{field}.weight"
        for field in (
            "layer_multipliers",
            "ngram_heads_vocab_sizes",
            "ngram_heads_offsets",
            "ngram_embedding.weight",
        ):
            self.from_hf_map[
                f"{prefix}.layers.{{}}.ple.ple_embedding.{field}"
            ] = f"layers.{{}}.ple.ple_embedding.{field}"

    def to_hf(
        self, state_dict: dict[str, Any], *, to_quantized_layout: bool = False
    ) -> dict[str, Any]:
        result = super().to_hf(state_dict, to_quantized_layout=to_quantized_layout)
        for layer_id, layer in enumerate(self.flash_config.layers):
            if layer.ple is None:
                continue
            base = f"{self.hf_language_model_prefix}.layers.{layer_id}.ple.ple_embedding.ngram_embedding"
            key = base + ".weight"
            if key not in result:
                continue
            weight_ND = result.pop(key)
            parts = layer.ple.ple_embedding.split_ngram_parts
            for part, shard_ND in enumerate(weight_ND.chunk(parts, 0)):
                result[f"{base}.shard_{part}.weight"] = shard_ND
        return result

    def from_hf(
        self, hf_state_dict: dict[str, Any], *, is_quantized_layout: bool = False
    ) -> dict[str, Any]:
        state_dict = dict(hf_state_dict)
        shards_by_layer: dict[int, dict[int, Any]] = {}
        pattern = re.compile(
            re.escape(self.hf_language_model_prefix)
            + r"\.layers\.(\d+)\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$"
        )
        for key in tuple(state_dict):
            match = pattern.fullmatch(key)
            if match is not None:
                layer_id, part = map(int, match.groups())
                shards_by_layer.setdefault(layer_id, {})[part] = state_dict.pop(key)
        for layer_id, shards in shards_by_layer.items():
            ple = self.flash_config.layers[layer_id].ple
            parts = ple.ple_embedding.split_ngram_parts
            if set(shards) != set(range(parts)):
                raise ValueError(
                    f"PLE layer {layer_id} requires all {parts} n-gram shards"
                )
            base = f"{self.hf_language_model_prefix}.layers.{layer_id}.ple.ple_embedding.ngram_embedding.weight"
            if base in state_dict:
                raise ValueError(
                    "Checkpoint contains both consolidated and split n-gram weights"
                )
            state_dict[base] = torch.cat([shards[part] for part in range(parts)], 0)
        if any(key.startswith("mtp.") for key in state_dict):
            logger.warning(
                "The Flash-Next Transformers reference excludes the MTP draft head; loading the backbone and vision weights only."
            )
        return super().from_hf(state_dict, is_quantized_layout=is_quantized_layout)

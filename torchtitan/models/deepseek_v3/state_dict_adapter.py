# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Any

import torch
from torch.distributed.checkpoint import HuggingFaceStorageReader

from torchtitan.models.utils import MoEStateDictAdapter

from .model import DeepSeekV3Model

_LAYER_RE = re.compile(r"^layers\.(\d+)\.")
_HF_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


class DeepSeekV3StateDictAdapter(MoEStateDictAdapter):
    _HF_EXPERT_RE = re.compile(
        r"""
        ^model\.layers\.(\d+)           # layer index
        \.mlp\.experts\.(\d+)            # expert index
        \.(gate_proj|up_proj|down_proj)  # projection type
        \.weight$
        """,
        re.VERBOSE,
    )
    _PROJ_TO_HF = {
        "moe.experts.w1": "gate_proj",
        "moe.experts.w3": "up_proj",
        "moe.experts.w2": "down_proj",
    }
    _PROJ_FROM_HF = {
        "gate_proj": "moe.experts.w1",
        "up_proj": "moe.experts.w3",
        "down_proj": "moe.experts.w2",
    }

    def __init__(
        self,
        model_config: DeepSeekV3Model.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)
        # pyrefly: ignore [missing-attribute]
        self._has_q_lora = model_config.layers[0].attention.q_lora_rank != 0

    def get_hf_storage_reader(
        self, path: str, from_quantized: bool = False
    ) -> HuggingFaceStorageReader:
        # NOTE: Now we use Quantized HF storage reader to read DeepSeek-V3 671B model.
        # If loading checkpoints without quantization, use HuggingFaceStorageReader instead
        if from_quantized:
            from torch.distributed.checkpoint.quantized_hf_storage import (
                QuantizedHuggingFaceStorageReader,
            )

            BLOCK_SIZE = 128
            return QuantizedHuggingFaceStorageReader(
                path=path,
                target_dtype=torch.float32,
                block_size=BLOCK_SIZE,
                thread_count=4,
            )
        else:
            return HuggingFaceStorageReader(path)

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert torchtitan state dict to HF format.

        Renames FQNs and splits 3D GroupedExperts weights into individual
        2D per-expert weights.
        """
        RENAME = {
            "tok_embeddings.weight": "model.embed_tokens.weight",
            "norm.weight": "model.norm.weight",
            "output.weight": "lm_head.weight",
        }
        LAYER_RENAME = {
            "attention.wkv_a.weight": "self_attn.kv_a_proj_with_mqa.weight",
            "attention.kv_norm.weight": "self_attn.kv_a_layernorm.weight",
            "attention.wkv_b.weight": "self_attn.kv_b_proj.weight",
            "attention.wo.weight": "self_attn.o_proj.weight",
            "feed_forward.w1.weight": "mlp.gate_proj.weight",
            "feed_forward.w3.weight": "mlp.up_proj.weight",
            "feed_forward.w2.weight": "mlp.down_proj.weight",
            "attention_norm.weight": "input_layernorm.weight",
            "ffn_norm.weight": "post_attention_layernorm.weight",
            "moe.router.gate.weight": "mlp.gate.weight",
            "moe.expert_bias": "mlp.gate.e_score_correction_bias",
            "moe.shared_experts.w1.weight": "mlp.shared_experts.gate_proj.weight",
            "moe.shared_experts.w3.weight": "mlp.shared_experts.up_proj.weight",
            "moe.shared_experts.w2.weight": "mlp.shared_experts.down_proj.weight",
        }

        hf: dict[str, Any] = {}
        unmapped_keys: list[str] = []

        for key, value in state_dict.items():
            if key in RENAME:
                hf[RENAME[key]] = value
            elif m := _LAYER_RE.match(key):
                layer = m.group(1)
                suffix = key[m.end() :]

                if suffix in LAYER_RENAME:
                    hf[f"model.layers.{layer}.{LAYER_RENAME[suffix]}"] = value
                # Attention — Q projection (depends on q_lora_rank)
                elif suffix == "attention.wq_a.weight" and self._has_q_lora:
                    hf[f"model.layers.{layer}.self_attn.q_a_proj.weight"] = value
                elif suffix == "attention.q_norm.weight" and self._has_q_lora:
                    hf[f"model.layers.{layer}.self_attn.q_a_layernorm.weight"] = value
                elif suffix == "attention.wq_b.weight" and self._has_q_lora:
                    hf[f"model.layers.{layer}.self_attn.q_b_proj.weight"] = value
                elif suffix == "attention.wq.weight" and not self._has_q_lora:
                    hf[f"model.layers.{layer}.self_attn.q_proj.weight"] = value
                # MoE experts (3D GroupedExperts → individual 2D experts)
                elif suffix in self._PROJ_TO_HF:
                    self._experts_to_hf(suffix, layer, value, hf)
                else:
                    unmapped_keys.append(key)
            else:
                unmapped_keys.append(key)

        if unmapped_keys:
            raise ValueError(
                f"{type(self).__name__}.to_hf: unmapped keys: {unmapped_keys}"
            )
        return hf

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert HF state dict to torchtitan format.

        Renames FQNs and concatenates individual 2D per-expert weights
        back into 3D GroupedExperts weights. Dequantization (if needed) is
        handled by QuantizedHuggingFaceStorageReader during load.
        """
        RENAME = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }
        LAYER_RENAME = {
            "self_attn.kv_a_proj_with_mqa.weight": "attention.wkv_a.weight",
            "self_attn.kv_a_layernorm.weight": "attention.kv_norm.weight",
            "self_attn.kv_b_proj.weight": "attention.wkv_b.weight",
            "self_attn.o_proj.weight": "attention.wo.weight",
            "mlp.gate_proj.weight": "feed_forward.w1.weight",
            "mlp.up_proj.weight": "feed_forward.w3.weight",
            "mlp.down_proj.weight": "feed_forward.w2.weight",
            "input_layernorm.weight": "attention_norm.weight",
            "post_attention_layernorm.weight": "ffn_norm.weight",
            "mlp.gate.weight": "moe.router.gate.weight",
            "mlp.gate.e_score_correction_bias": "moe.expert_bias",
            "mlp.shared_experts.gate_proj.weight": "moe.shared_experts.w1.weight",
            "mlp.shared_experts.up_proj.weight": "moe.shared_experts.w3.weight",
            "mlp.shared_experts.down_proj.weight": "moe.shared_experts.w2.weight",
        }

        sd: dict[str, Any] = {}
        unmapped_keys: list[str] = []
        expert_weights_by_layer: dict[str, dict[str, dict[int, Any]]] = {}

        for key, value in hf_state_dict.items():
            # Check for MoE expert keys first
            if self._experts_from_hf(key, value, sd, expert_weights_by_layer):
                continue

            if key in RENAME:
                sd[RENAME[key]] = value
            elif m := _HF_LAYER_RE.match(key):
                layer = m.group(1)
                suffix = key[m.end() :]

                if suffix in LAYER_RENAME:
                    sd[f"layers.{layer}.{LAYER_RENAME[suffix]}"] = value
                # Attention — Q projection (depends on q_lora_rank)
                elif suffix == "self_attn.q_a_proj.weight" and self._has_q_lora:
                    sd[f"layers.{layer}.attention.wq_a.weight"] = value
                elif suffix == "self_attn.q_a_layernorm.weight" and self._has_q_lora:
                    sd[f"layers.{layer}.attention.q_norm.weight"] = value
                elif suffix == "self_attn.q_b_proj.weight" and self._has_q_lora:
                    sd[f"layers.{layer}.attention.wq_b.weight"] = value
                elif suffix == "self_attn.q_proj.weight" and not self._has_q_lora:
                    sd[f"layers.{layer}.attention.wq.weight"] = value
                else:
                    unmapped_keys.append(key)
            else:
                unmapped_keys.append(key)

        if unmapped_keys:
            raise ValueError(
                f"{type(self).__name__}.from_hf: unmapped keys: {unmapped_keys}"
            )
        return sd

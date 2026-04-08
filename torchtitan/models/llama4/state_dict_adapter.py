# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from collections import defaultdict
from typing import Any

import torch

from torchtitan.models.utils import MoEStateDictAdapter

from .model import Llama4Model

_LAYER_RE = re.compile(r"^layers\.(\d+)\.")
_HF_LAYER_RE = re.compile(r"^language_model\.model\.layers\.(\d+)\.")


class Llama4StateDictAdapter(MoEStateDictAdapter):
    def __init__(self, model_config: Llama4Model.Config, hf_assets_path: str | None):
        super().__init__(model_config, hf_assets_path)

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        RENAME = {
            "tok_embeddings.weight": "language_model.model.embed_tokens.weight",
            "norm.weight": "language_model.model.norm.weight",
            "output.weight": "language_model.lm_head.weight",
        }
        LAYER_RENAME = {
            "attention.wq.weight": "self_attn.q_proj.weight",
            "attention.wk.weight": "self_attn.k_proj.weight",
            "attention.wv.weight": "self_attn.v_proj.weight",
            "attention.wo.weight": "self_attn.o_proj.weight",
            "attention_norm.weight": "input_layernorm.weight",
            "ffn_norm.weight": "post_attention_layernorm.weight",
            "moe.router.gate.weight": "feed_forward.router.weight",
            "moe.shared_experts.w1.weight": "feed_forward.shared_expert.gate_proj.weight",
            "moe.shared_experts.w2.weight": "feed_forward.shared_expert.down_proj.weight",
            "moe.shared_experts.w3.weight": "feed_forward.shared_expert.up_proj.weight",
        }

        hf: dict[str, Any] = {}
        # Collect w1/w3 per layer for gate_up_proj combination
        to_combine: dict[str, dict[str, torch.Tensor]] = defaultdict(dict)

        for key, value in state_dict.items():
            if key in RENAME:
                hf[RENAME[key]] = value
            elif m := _LAYER_RE.match(key):
                layer = m.group(1)
                suffix = key[m.end() :]

                if suffix in LAYER_RENAME:
                    hf[
                        f"language_model.model.layers.{layer}.{LAYER_RENAME[suffix]}"
                    ] = value
                # MoE expert bias — no HF equivalent, drop
                elif suffix == "moe.expert_bias":
                    pass
                # MoE experts: w2 (down_proj) — transpose for HF format
                elif suffix == "moe.experts.w2":
                    hf[
                        f"language_model.model.layers.{layer}.feed_forward.experts.down_proj"
                    ] = value.transpose(-1, -2)
                # MoE experts: w1, w3 — collect for gate_up_proj combination
                elif suffix in ("moe.experts.w1", "moe.experts.w3"):
                    hf_fqn = f"language_model.model.layers.{layer}.feed_forward.experts.gate_up_proj"
                    to_combine[hf_fqn][suffix] = value

        # Combine w1 + w3 → gate_up_proj (transposed, then concatenated)
        for hf_fqn, parts in to_combine.items():
            w1 = parts["moe.experts.w1"].transpose(-1, -2)
            w3 = parts["moe.experts.w3"].transpose(-1, -2)
            hf[hf_fqn] = torch.cat([w1, w3], dim=-1)

        return hf

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        RENAME = {
            "language_model.model.embed_tokens.weight": "tok_embeddings.weight",
            "language_model.model.norm.weight": "norm.weight",
            "language_model.lm_head.weight": "output.weight",
        }
        LAYER_RENAME = {
            "self_attn.q_proj.weight": "attention.wq.weight",
            "self_attn.k_proj.weight": "attention.wk.weight",
            "self_attn.v_proj.weight": "attention.wv.weight",
            "self_attn.o_proj.weight": "attention.wo.weight",
            "input_layernorm.weight": "attention_norm.weight",
            "post_attention_layernorm.weight": "ffn_norm.weight",
            "feed_forward.router.weight": "moe.router.gate.weight",
            "feed_forward.shared_expert.gate_proj.weight": "moe.shared_experts.w1.weight",
            "feed_forward.shared_expert.down_proj.weight": "moe.shared_experts.w2.weight",
            "feed_forward.shared_expert.up_proj.weight": "moe.shared_experts.w3.weight",
        }

        sd: dict[str, Any] = {}

        for key, value in hf_state_dict.items():
            if key in RENAME:
                sd[RENAME[key]] = value
            elif m := _HF_LAYER_RE.match(key):
                layer = m.group(1)
                suffix = key[m.end() :]

                if suffix in LAYER_RENAME:
                    sd[f"layers.{layer}.{LAYER_RENAME[suffix]}"] = value
                # MoE experts: down_proj → w2 (transpose back)
                elif suffix == "feed_forward.experts.down_proj":
                    sd[f"layers.{layer}.moe.experts.w2"] = value.transpose(-1, -2)
                # MoE experts: gate_up_proj → split into w1 + w3 (transpose back)
                elif suffix == "feed_forward.experts.gate_up_proj":
                    w1, w3 = value.chunk(2, dim=-1)
                    sd[f"layers.{layer}.moe.experts.w1"] = w1.transpose(-1, -2)
                    sd[f"layers.{layer}.moe.experts.w3"] = w3.transpose(-1, -2)

        return sd

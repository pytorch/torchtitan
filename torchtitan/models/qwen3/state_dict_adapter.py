# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Any

from torchtitan.models.utils import MoEStateDictAdapter

from .model import Qwen3Model

_LAYER_RE = re.compile(r"^layers\.(\d+)\.")
_HF_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


class Qwen3StateDictAdapter(MoEStateDictAdapter):
    _HF_EXPERT_RE = re.compile(
        r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
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

    def __init__(self, model_config: Qwen3Model.Config, hf_assets_path: str | None):
        super().__init__(model_config, hf_assets_path)

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
            "attention.wq.weight": "self_attn.q_proj.weight",
            "attention.wk.weight": "self_attn.k_proj.weight",
            "attention.wv.weight": "self_attn.v_proj.weight",
            "attention.wo.weight": "self_attn.o_proj.weight",
            "attention.q_norm.weight": "self_attn.q_norm.weight",
            "attention.k_norm.weight": "self_attn.k_norm.weight",
            "feed_forward.w1.weight": "mlp.gate_proj.weight",
            "feed_forward.w3.weight": "mlp.up_proj.weight",
            "feed_forward.w2.weight": "mlp.down_proj.weight",
            "attention_norm.weight": "input_layernorm.weight",
            "ffn_norm.weight": "post_attention_layernorm.weight",
            "moe.router.gate.weight": "mlp.gate.weight",
        }

        hf: dict[str, Any] = {}
        unmapped_keys: list[str] = []

        for key, value in state_dict.items():
            # Skip output.weight when weight tying is enabled
            # pyrefly: ignore [missing-attribute]
            if self.model_config.enable_weight_tying and key == "output.weight":
                continue

            if key in RENAME:
                hf[RENAME[key]] = value
            elif m := _LAYER_RE.match(key):
                layer = m.group(1)
                suffix = key[m.end() :]

                if suffix in LAYER_RENAME:
                    hf[f"model.layers.{layer}.{LAYER_RENAME[suffix]}"] = value
                # MoE experts (3D GroupedExperts → individual 2D experts)
                elif suffix in self._PROJ_TO_HF:
                    self._experts_to_hf(suffix, layer, value, hf)
                # MoE expert bias — no HF equivalent, drop
                elif suffix == "moe.expert_bias":
                    pass
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
        back into 3D GroupedExperts weights.
        """
        RENAME = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }
        LAYER_RENAME = {
            "self_attn.q_proj.weight": "attention.wq.weight",
            "self_attn.k_proj.weight": "attention.wk.weight",
            "self_attn.v_proj.weight": "attention.wv.weight",
            "self_attn.o_proj.weight": "attention.wo.weight",
            "self_attn.q_norm.weight": "attention.q_norm.weight",
            "self_attn.k_norm.weight": "attention.k_norm.weight",
            "self_attn.rotary_emb.inv_freq": None,  # drop
            "mlp.gate_proj.weight": "feed_forward.w1.weight",
            "mlp.up_proj.weight": "feed_forward.w3.weight",
            "mlp.down_proj.weight": "feed_forward.w2.weight",
            "input_layernorm.weight": "attention_norm.weight",
            "post_attention_layernorm.weight": "ffn_norm.weight",
            "mlp.gate.weight": "moe.router.gate.weight",
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
                    target = LAYER_RENAME[suffix]
                    if target is not None:
                        sd[f"layers.{layer}.{target}"] = value
                else:
                    unmapped_keys.append(key)
            else:
                unmapped_keys.append(key)

        if unmapped_keys:
            raise ValueError(
                f"{type(self).__name__}.from_hf: unmapped keys: {unmapped_keys}"
            )

        # Weight tying: copy embedding as output if lm_head absent
        if (
            # pyrefly: ignore [missing-attribute]
            self.model_config.enable_weight_tying
            and "output.weight" not in sd
            and "tok_embeddings.weight" in sd
        ):
            sd["output.weight"] = sd["tok_embeddings.weight"]

        return sd

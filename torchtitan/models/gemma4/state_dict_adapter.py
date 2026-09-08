# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Gemma-4 State Dict Adapter
# Converts between HuggingFace Transformers and TorchTitan checkpoint formats

import logging
import re
from typing import Any

logger = logging.getLogger()

from torchtitan.models.common.rope import CosSinRoPE
from torchtitan.protocols.state_dict_adapter import StateDictAdapter
from .model import Gemma4Model


class Gemma4StateDictAdapter(StateDictAdapter):
    """Adapter for converting Gemma-4 checkpoints between HF and TorchTitan formats.

    Handles the mapping between HuggingFace Transformers model keys and TorchTitan's
    expected parameter names, including all four layer norms, Q/K norms, and CosSinRoPE.
    """

    def __init__(
        self,
        model_config: Gemma4Model.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)

        self.model_config = model_config
        self.hf_assets_path = hf_assets_path

        # Mapping from HuggingFace keys to TorchTitan keys
        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.qkv_linear.wq.weight",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.qkv_linear.wk.weight",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.qkv_linear.wv.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
            "model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
            "model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.post_attention_norm.weight",
            "model.layers.{}.pre_feedforward_layernorm.weight": "layers.{}.ffn_norm.weight",
            "model.layers.{}.post_feedforward_layernorm.weight": "layers.{}.post_ffn_norm.weight",
            "model.layers.{}.layer_scalar": "layers.{}.layer_scalar",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
        }

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert TorchTitan checkpoint to HuggingFace format."""
        hf_state_dict = {}
        to_hf_map = {v: k for k, v in self.from_hf_map.items() if v is not None}

        for key, value in state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                m = re.search(r"\d+", key)
                if m is None:
                    raise ValueError(f"Expected layer index in key '{key}'")
                layer_num = m.group(0)

                new_key = to_hf_map.get(abstract_key)
                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
            else:
                if (
                    self.model_config.enable_weight_tying  # pyrefly: ignore [missing-attribute]
                    and key == "lm_head.weight"
                ):
                    if self.fqn_to_index_mapping:
                        self.fqn_to_index_mapping.pop("lm_head.weight", None)
                    continue
                new_key = to_hf_map.get(key)
                if new_key is None:
                    continue

            hf_state_dict[new_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert HuggingFace checkpoint to TorchTitan format."""
        self._validate_hf_rope_config(CosSinRoPE.Config)
        if (
            self.model_config.enable_weight_tying  # pyrefly: ignore [missing-attribute]
            and "lm_head.weight" not in hf_state_dict
        ):
            if "model.embed_tokens.weight" not in hf_state_dict:
                raise ValueError(
                    "Weight tying enabled but 'model.embed_tokens.weight' is missing from HF state dict."
                )
            hf_state_dict["lm_head.weight"] = hf_state_dict["model.embed_tokens.weight"]

        state_dict = {}

        for key, value in hf_state_dict.items():
            if key.startswith("model.language_model."):
                key = "model." + key[len("model.language_model."):]
            elif key.startswith("language_model."):
                key = "model." + key[len("language_model."):]

            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                m = re.search(r"\d+", key)
                if m is None:
                    raise ValueError(f"Expected layer index in HF key '{key}'")
                layer_num = m.group(0)

                new_key = self.from_hf_map.get(abstract_key)
                if new_key is None:
                    continue
                new_key = new_key.format(layer_num)
            else:
                new_key = self.from_hf_map.get(key)
                if new_key is None:
                    continue

            state_dict[new_key] = value

        return state_dict

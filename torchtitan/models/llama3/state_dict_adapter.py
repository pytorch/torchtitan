# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Any

from torchtitan.protocols.state_dict_adapter import BaseStateDictAdapter

from .model import Llama3Model

_LAYER_RE = re.compile(r"^layers\.(\d+)\.")
_HF_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


class Llama3StateDictAdapter(BaseStateDictAdapter):
    def __init__(
        self,
        model_config: Llama3Model.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)

        self.model_config = model_config

        n_heads = model_config.layers[0].attention.n_heads
        n_kv_heads = (
            model_config.layers[0].attention.n_kv_heads
            # pyrefly: ignore [missing-attribute]
            if model_config.layers[0].attention.n_kv_heads is not None
            else n_heads
        )
        dim = model_config.dim
        head_dim = dim // n_heads
        self._n_heads = n_heads
        self._n_kv_heads = n_kv_heads
        # pyrefly: ignore [unsupported-operation]
        self._key_value_dim = head_dim * n_kv_heads
        self._dim = dim

    # -- RoPE permutation helpers --

    def _permute(self, w, n_heads_arg, dim1=None, dim2=None):
        if dim1 is None:
            dim1 = w.shape[0]
        if dim2 is None:
            dim2 = w.shape[1]
        return (
            w.view(n_heads_arg, dim1 // n_heads_arg // 2, 2, dim2)
            .transpose(1, 2)
            .reshape(dim1, dim2)
            .clone()
        )

    def _reverse_permute(self, w, n_heads_arg, dim1=None, dim2=None):
        if dim1 is None:
            dim1 = w.shape[0]
        if dim2 is None:
            dim2 = w.shape[1]
        return (
            w.view(n_heads_arg, 2, dim1 // n_heads_arg // 2, dim2)
            .transpose(1, 2)
            .reshape(dim1, dim2)
        )

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        RENAME = {
            "tok_embeddings.weight": "model.embed_tokens.weight",
            "norm.weight": "model.norm.weight",
            "output.weight": "lm_head.weight",
        }
        LAYER_RENAME = {
            "attention.wv.weight": "self_attn.v_proj.weight",
            "attention.wo.weight": "self_attn.o_proj.weight",
            "feed_forward.w1.weight": "mlp.gate_proj.weight",
            "feed_forward.w3.weight": "mlp.up_proj.weight",
            "feed_forward.w2.weight": "mlp.down_proj.weight",
            "attention_norm.weight": "input_layernorm.weight",
            "ffn_norm.weight": "post_attention_layernorm.weight",
        }

        hf: dict[str, Any] = {}
        for key, value in state_dict.items():
            if self.model_config.enable_weight_tying and key == "output.weight":
                continue

            if key in RENAME:
                hf[RENAME[key]] = value
            elif m := _LAYER_RE.match(key):
                layer, suffix = m.group(1), key[m.end() :]
                if suffix in LAYER_RENAME:
                    hf[f"model.layers.{layer}.{LAYER_RENAME[suffix]}"] = value
                # Permute wq and wk to account for the difference between
                # the native Llama and HuggingFace RoPE implementations.
                elif suffix == "attention.wq.weight":
                    hf[f"model.layers.{layer}.self_attn.q_proj.weight"] = self._permute(
                        value, self._n_heads
                    )
                elif suffix == "attention.wk.weight":
                    hf[f"model.layers.{layer}.self_attn.k_proj.weight"] = self._permute(
                        value,
                        self._n_kv_heads,
                        self._key_value_dim,
                        self._dim,
                    )

        return hf

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        RENAME = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }
        LAYER_RENAME = {
            "self_attn.v_proj.weight": "attention.wv.weight",
            "self_attn.o_proj.weight": "attention.wo.weight",
            "self_attn.rotary_emb.inv_freq": None,  # drop
            "mlp.gate_proj.weight": "feed_forward.w1.weight",
            "mlp.up_proj.weight": "feed_forward.w3.weight",
            "mlp.down_proj.weight": "feed_forward.w2.weight",
            "input_layernorm.weight": "attention_norm.weight",
            "post_attention_layernorm.weight": "ffn_norm.weight",
        }

        sd: dict[str, Any] = {}
        for key, value in hf_state_dict.items():
            if key in RENAME:
                sd[RENAME[key]] = value
            elif m := _HF_LAYER_RE.match(key):
                layer, suffix = m.group(1), key[m.end() :]
                if suffix in LAYER_RENAME:
                    target = LAYER_RENAME[suffix]
                    if target is not None:
                        sd[f"layers.{layer}.{target}"] = value
                # Reverse-permute wq and wk to account for the difference
                # between the native Llama and HuggingFace RoPE implementations.
                elif suffix == "self_attn.q_proj.weight":
                    sd[f"layers.{layer}.attention.wq.weight"] = self._reverse_permute(
                        value, self._n_heads
                    )
                elif suffix == "self_attn.k_proj.weight":
                    sd[f"layers.{layer}.attention.wk.weight"] = self._reverse_permute(
                        value,
                        self._n_kv_heads,
                        self._key_value_dim,
                        self._dim,
                    )

        # Weight tying: copy embedding as output if lm_head absent
        if (
            self.model_config.enable_weight_tying
            and "output.weight" not in sd
            and "tok_embeddings.weight" in sd
        ):
            sd["output.weight"] = sd["tok_embeddings.weight"]

        return sd

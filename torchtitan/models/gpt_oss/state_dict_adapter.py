# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Any

from torch.distributed.checkpoint import HuggingFaceStorageReader

from torchtitan.models.utils import MoEStateDictAdapter

from .model import GptOssModel

_LAYER_RE = re.compile(r"^layers\.(\d+)\.")
_HF_LAYER_RE = re.compile(r"^model\.layers\.(\d+)\.")


class GptOssStateDictAdapter(MoEStateDictAdapter):
    """Pure FQN rename — no value transforms needed.

    Warning: Conversion does not support saving to MXFP4 quantization format.
    One can save into unquantized HF checkpoints with ``last_save_in_hf = true``.
    For loading from quantized checkpoints, the QuantizedHuggingFaceStorageReader
    handles dequantization during load.
    """

    def __init__(self, model_config: GptOssModel.Config, hf_assets_path: str | None):
        super().__init__(model_config, hf_assets_path)

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        RENAME = {
            "tok_embeddings.weight": "model.embed_tokens.weight",
            "norm.weight": "model.norm.weight",
            "output.weight": "lm_head.weight",
        }
        LAYER_RENAME = {
            "attention.wq.weight": "self_attn.q_proj.weight",
            "attention.wq.bias": "self_attn.q_proj.bias",
            "attention.wk.weight": "self_attn.k_proj.weight",
            "attention.wk.bias": "self_attn.k_proj.bias",
            "attention.wv.weight": "self_attn.v_proj.weight",
            "attention.wv.bias": "self_attn.v_proj.bias",
            "attention.wo.weight": "self_attn.o_proj.weight",
            "attention.wo.bias": "self_attn.o_proj.bias",
            "attention.sinks": "self_attn.sinks",
            "attention_norm.weight": "input_layernorm.weight",
            "ffn_norm.weight": "post_attention_layernorm.weight",
            "moe.experts.mlp1_weight": "mlp.experts.gate_up_proj_blocks",
            "moe.experts.mlp1_bias": "mlp.experts.gate_up_proj_bias",
            "moe.experts.mlp2_weight": "mlp.experts.down_proj_blocks",
            "moe.experts.mlp2_bias": "mlp.experts.down_proj_bias",
            "moe.router.gate.weight": "mlp.router.weight",
            "moe.router.gate.bias": "mlp.router.bias",
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
        """Convert from hf format state dict to tt model state dict."""
        RENAME = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }
        LAYER_RENAME = {
            "self_attn.q_proj.weight": "attention.wq.weight",
            "self_attn.q_proj.bias": "attention.wq.bias",
            "self_attn.k_proj.weight": "attention.wk.weight",
            "self_attn.k_proj.bias": "attention.wk.bias",
            "self_attn.v_proj.weight": "attention.wv.weight",
            "self_attn.v_proj.bias": "attention.wv.bias",
            "self_attn.o_proj.weight": "attention.wo.weight",
            "self_attn.o_proj.bias": "attention.wo.bias",
            "self_attn.sinks": "attention.sinks",
            "input_layernorm.weight": "attention_norm.weight",
            "post_attention_layernorm.weight": "ffn_norm.weight",
            "mlp.experts.gate_up_proj_blocks": "moe.experts.mlp1_weight",
            "mlp.experts.gate_up_proj_bias": "moe.experts.mlp1_bias",
            "mlp.experts.down_proj_blocks": "moe.experts.mlp2_weight",
            "mlp.experts.down_proj_bias": "moe.experts.mlp2_bias",
            "mlp.router.weight": "moe.router.gate.weight",
            "mlp.router.bias": "moe.router.gate.bias",
        }

        sd: dict[str, Any] = {}
        unmapped_keys: list[str] = []

        for key, value in hf_state_dict.items():
            if key in RENAME:
                sd[RENAME[key]] = value
            elif m := _HF_LAYER_RE.match(key):
                layer = m.group(1)
                suffix = key[m.end() :]

                if suffix in LAYER_RENAME:
                    sd[f"layers.{layer}.{LAYER_RENAME[suffix]}"] = value
                else:
                    unmapped_keys.append(key)
            else:
                unmapped_keys.append(key)

        if unmapped_keys:
            raise ValueError(
                f"{type(self).__name__}.from_hf: unmapped keys: {unmapped_keys}"
            )
        return sd

    def get_hf_storage_reader(
        self, path: str, from_quantized: bool = False
    ) -> HuggingFaceStorageReader:
        # NOTE: Now we use Quantized HF storage reader to read GPT-OSS model where
        # expert weights are saved in MXFP4 format.
        # If loading checkpoints without quantization, use HuggingFaceStorageReader instead
        if from_quantized:
            from torch.distributed.checkpoint.quantized_hf_storage import (
                QuantizedHuggingFaceStorageReader,
            )

            return QuantizedHuggingFaceStorageReader(
                path=path,
                thread_count=4,
            )
        else:
            return HuggingFaceStorageReader(path)

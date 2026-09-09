# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Any

import torch
from torch.distributed.checkpoint import HuggingFaceStorageReader

from torchtitan.models.common.rope import CosSinRoPE
from torchtitan.models.utils import MoEStateDictAdapter
from .model import GptOssModel


class GptOssStateDictAdapter(MoEStateDictAdapter):
    _EXPERT_BIAS_KEY = "layers.{}.moe.expert_bias_E"

    def __init__(self, model_config: GptOssModel.Config, hf_assets_path: str | None):
        super().__init__(model_config, hf_assets_path)

        # HF GPT-OSS checkpoints do not have the auxiliary load-balancing bias.
        # Keep its source tensors so from_hf() can recreate zero buffers with the
        # same device and distributed layout during checkpoint loading.
        self._expert_bias_templates: dict[str, Any] = {}

        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            # Attention module
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.qkv_linear.wq.weight",
            "model.layers.{}.self_attn.q_proj.bias": "layers.{}.attention.qkv_linear.wq.bias",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.qkv_linear.wk.weight",
            "model.layers.{}.self_attn.k_proj.bias": "layers.{}.attention.qkv_linear.wk.bias",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.qkv_linear.wv.weight",
            "model.layers.{}.self_attn.v_proj.bias": "layers.{}.attention.qkv_linear.wv.bias",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.self_attn.o_proj.bias": "layers.{}.attention.wo.bias",
            "model.layers.{}.self_attn.sinks": "layers.{}.attention.sinks",
            # Transformer layer
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            # MoE
            "model.layers.{}.mlp.experts.gate_up_proj_blocks": "layers.{}.moe.routed_experts.inner_experts.mlp1_weight_EGD",
            "model.layers.{}.mlp.experts.gate_up_proj_bias": "layers.{}.moe.routed_experts.inner_experts.mlp1_bias_EG",
            "model.layers.{}.mlp.experts.down_proj_blocks": "layers.{}.moe.routed_experts.inner_experts.mlp2_weight_EDF",
            "model.layers.{}.mlp.experts.down_proj_bias": "layers.{}.moe.routed_experts.inner_experts.mlp2_bias_ED",
            "model.layers.{}.mlp.router.weight": "layers.{}.moe.router.gate.weight",
            "model.layers.{}.mlp.router.bias": "layers.{}.moe.router.gate.bias",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
        }

    def get_hf_storage_reader(
        self, path: str, from_quantized: bool = False
    ) -> HuggingFaceStorageReader:
        """
        Override default get_hf_storage_reader function to return QuantizedHFStorageReader.
        """
        if from_quantized:
            from torch.distributed.checkpoint.quantized_hf_storage import (
                QuantizedHuggingFaceStorageReader,
            )

            # NOTE: Now we use Quantized HF storage reader to read GPT-OSS model where
            # expert weights are saved in MXFP4 format.
            # If loading checkpoints without quantization, use HuggingFaceStorageReader instead
            return QuantizedHuggingFaceStorageReader(
                path=path,
                thread_count=4,
            )
        else:
            return HuggingFaceStorageReader(path)

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Convert from a tt model state dict to a hf format state dict.

        Only map keys without changing shapes to the same as MXFP4 checkpoint.
        For loading from quantized checkpoints, the QuantizedHuggingFaceStorageReader
            will handle dequantization during load.

        Warning: Conversion does not support saving to mxfp4 quantization format.
                 One can save into unquantized hf checkpoints with last_save_in_hf = true.
        """

        to_hf_map = {v: k for k, v in self.from_hf_map.items()}
        hf_state_dict = {}
        self._expert_bias_templates = {}

        for key, value in state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore
                layer_num = re.search(r"\d+", key).group(0)

                if abstract_key == self._EXPERT_BIAS_KEY:
                    self._expert_bias_templates[key] = value
                    continue
                if abstract_key not in to_hf_map:
                    continue
                hf_key = to_hf_map[abstract_key]
                hf_key = hf_key.format(layer_num)
                hf_state_dict[hf_key] = value
            else:
                if key not in to_hf_map:
                    continue
                hf_key = to_hf_map[key]
                hf_state_dict[hf_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Convert from hf format state dict to tt model state dict.
        """
        self._validate_hf_rope_config(CosSinRoPE.Config)

        state_dict = {}
        layer_nums = set()

        for key, value in hf_state_dict.items():
            if "layers" in key:
                # pyrefly: ignore
                layer_num = re.search(r"\d+", key).group(0)
                layer_nums.add(int(layer_num))
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)

                tt_key = self.from_hf_map.get(abstract_key)
                if tt_key is None:
                    continue
                tt_key = tt_key.format(layer_num)
                state_dict[tt_key] = value
            else:
                tt_key = self.from_hf_map[key]
                if tt_key is None:
                    continue
                state_dict[tt_key] = value

        # expert_bias_E is TorchTitan training state with no HF equivalent.
        # Reset it when loading an HF checkpoint instead of retaining stale
        # load-balancing history. Preserve DTensor metadata when to_hf() supplied
        # a target-state template; direct offline conversion uses a CPU tensor.
        for layer_num in layer_nums:
            # pyrefly: ignore [missing-attribute]
            moe_config = self.model_config.layers[layer_num].moe
            if moe_config is None or moe_config.load_balance_coeff is None:
                continue
            tt_key = self._EXPERT_BIAS_KEY.format(layer_num)
            template = self._expert_bias_templates.get(tt_key)
            state_dict[tt_key] = (
                torch.zeros_like(template)
                if template is not None
                else torch.zeros(moe_config.num_experts, dtype=torch.float32)
            )

        return state_dict

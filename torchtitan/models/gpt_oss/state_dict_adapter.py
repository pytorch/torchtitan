# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

import torch
from torch.distributed.checkpoint import HuggingFaceStorageReader

from torchtitan.models.common.rope import CosSinRoPE
from torchtitan.models.utils import MoEStateDictAdapter

if TYPE_CHECKING:
    from .model import GptOssModel


class GptOssStateDictAdapter(MoEStateDictAdapter):
    # Routed expert parameters. E = num experts, D = hidden dim, F = expert hidden dim.
    # w13 holds the gate (w1) and up (w3) projections, D -> F each; w2 is the down
    # projection, F -> D. HF fuses gate and up into one gate_up_proj and interleaves
    # their rows (gate row 0, up row 0, gate row 1, ...).
    #
    #   TorchTitan               HF bf16 (quantized=False)   HF MXFP4 (quantized=True)
    #   w13.weight [E, 2, F, D]  gate_up_proj [E, D, 2F]     gate_up_proj_blocks [E, 2F, D/32, 16]
    #   w13.bias   [E, 2, F]     gate_up_proj_bias [E, 2F]   gate_up_proj_bias [E, 2F]
    #   w2.weight  [E, D, F]     down_proj [E, F, D]         down_proj_blocks [E, D, F/32, 16]
    #   w2.bias    [E, D]        down_proj_bias [E, D]       down_proj_bias [E, D]
    #
    # Weights: TorchTitan stores each projection like nn.Linear, output size first:
    # gate and up (D -> F) as [F, D], down (F -> D) as [D, F]. HF bf16 stores the
    # input size first, [D, F] and [F, D], because transformers computes x @ W, so
    # converting transposes the last two dims. HF MXFP4 is output size first like
    # TorchTitan. It packs the last (input) dim in groups of 32 values with one scale
    # each (gate_up_proj_scales / down_proj_scales), and
    # QuantizedHuggingFaceStorageReader dequantizes the blocks to [E, 2F, D] and
    # [E, D, F], so only the name changes.
    # Biases: one value per output feature, so there is no input/output order to
    # swap, and MXFP4 leaves them unquantized. Both formats use the same names and shapes.
    _EXPERT_BIAS_KEY = "layers.{}.moe.expert_bias_E"
    _W13_WEIGHT_KEY = "layers.{}.moe.routed_experts.w13.weight"
    _W13_BIAS_KEY = "layers.{}.moe.routed_experts.w13.bias"
    _W2_WEIGHT_KEY = "layers.{}.moe.routed_experts.w2.weight"

    def __init__(self, model_config: GptOssModel.Config, hf_assets_path: str | None):
        super().__init__(model_config, hf_assets_path)
        # fqn_to_index_mapping always uses unquantized names: only HF saves read
        # it, and they write unquantized experts. hf_assets_path may hold the
        # released MXFP4 index, so X_blocks is saved as X and X_scales is not saved.
        if self.fqn_to_index_mapping is not None:
            self.fqn_to_index_mapping = {
                key.removesuffix("_blocks"): index
                for key, index in self.fqn_to_index_mapping.items()
                if not key.endswith("_scales")
            }

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
            "model.layers.{}.mlp.experts.gate_up_proj": self._W13_WEIGHT_KEY,
            "model.layers.{}.mlp.experts.gate_up_proj_bias": self._W13_BIAS_KEY,
            "model.layers.{}.mlp.experts.down_proj": self._W2_WEIGHT_KEY,
            "model.layers.{}.mlp.experts.down_proj_bias": "layers.{}.moe.routed_experts.w2.bias",
            "model.layers.{}.mlp.router.weight": "layers.{}.moe.router.gate.weight",
            "model.layers.{}.mlp.router.bias": "layers.{}.moe.router.gate.bias",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
        }
        expert_weight_keys = (self._W13_WEIGHT_KEY, self._W2_WEIGHT_KEY)
        self.from_hf_map_quantized = {
            hf_key: tt_key
            for hf_key, tt_key in self.from_hf_map.items()
            if tt_key not in expert_weight_keys
        } | {
            "model.layers.{}.mlp.experts.gate_up_proj_blocks": self._W13_WEIGHT_KEY,
            "model.layers.{}.mlp.experts.down_proj_blocks": self._W2_WEIGHT_KEY,
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

    def to_hf(
        self, state_dict: dict[str, Any], quantized: bool = False
    ) -> dict[str, Any]:
        """
        Convert from a tt model state dict to a hf format state dict.

        By default, expert weights use the HF bf16 names and layout. With
        ``quantized``, they use the MXFP4 ``_blocks`` names and keep the TorchTitan
        layout, which QuantizedHuggingFaceStorageReader dequantizes into. See the
        table above the class constants.

        Warning: Conversion does not support saving to mxfp4 quantization format.
        """
        state_dict = self._native_fused_linears_to_hf(state_dict)

        from_hf_map = self.from_hf_map_quantized if quantized else self.from_hf_map
        to_hf_map = {v: k for k, v in from_hf_map.items()}
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
                if abstract_key in (self._W13_WEIGHT_KEY, self._W13_BIAS_KEY):
                    # [E, 2, F, ...] -> [E, 2F, ...], gate and up rows interleaved.
                    value = value.transpose(1, 2).flatten(1, 2)
                if (
                    abstract_key in (self._W13_WEIGHT_KEY, self._W2_WEIGHT_KEY)
                    and not quantized
                ):
                    # TorchTitan output size first -> HF bf16 input size first.
                    # Dequantized MXFP4 is output size first already.
                    value = value.transpose(1, 2)
                hf_state_dict[hf_key] = value
            else:
                if key not in to_hf_map:
                    continue
                hf_key = to_hf_map[key]
                hf_state_dict[hf_key] = value

        return hf_state_dict

    def from_hf(
        self, hf_state_dict: dict[str, Any], quantized: bool = False
    ) -> dict[str, Any]:
        """
        Convert from hf format state dict to tt model state dict.
        """
        self._validate_hf_rope_config(CosSinRoPE.Config)

        from_hf_map = self.from_hf_map_quantized if quantized else self.from_hf_map
        state_dict = {}
        layer_nums = set()

        for key, value in hf_state_dict.items():
            if "layers" in key:
                # pyrefly: ignore
                layer_num = re.search(r"\d+", key).group(0)
                layer_nums.add(int(layer_num))
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)

                tt_abstract_key = from_hf_map.get(abstract_key)
                if tt_abstract_key is None:
                    continue
                tt_key = tt_abstract_key.format(layer_num)
                if (
                    tt_abstract_key in (self._W13_WEIGHT_KEY, self._W2_WEIGHT_KEY)
                    and not quantized
                ):
                    # HF bf16 input size first -> TorchTitan output size first.
                    value = value.transpose(1, 2)
                if tt_abstract_key in (self._W13_WEIGHT_KEY, self._W13_BIAS_KEY):
                    # [E, 2F, ...] with gate and up interleaved -> [E, 2, F, ...].
                    value = value.unflatten(1, (-1, 2)).transpose(1, 2).contiguous()
                state_dict[tt_key] = value
            else:
                tt_key = from_hf_map[key]
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

        return self._native_fused_linears_from_hf(state_dict)

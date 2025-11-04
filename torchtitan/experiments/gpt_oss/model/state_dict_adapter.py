# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import math
import re
from typing import Any

import torch
from torch.distributed.tensor import DTensor
from torchtitan.models.utils import MoEStateDictAdapter

from .args import GptOssModelArgs


FP4_VALUES = [
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def get_mxfp4_tensor(
    blocks,
    scales,
    *,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 16384 * 512,
) -> torch.Tensor:
    """
    Adapted from openai's implementation of mxfp4 dequantization:
    https://github.com/openai/gpt-oss/blob/8890e95919f975a490fc0ba09ffb10890ec7319d/gpt_oss/torch/weights.py#L68
    """

    is_dtensor = isinstance(blocks, DTensor)
    if is_dtensor:
        device_mesh = blocks.device_mesh
        placements = blocks.placements
        blocks = blocks.to_local()
        scales = scales.to_local()

    scales = scales.to(torch.int32) - 127

    assert (
        blocks.shape[:-1] == scales.shape
    ), f"{blocks.shape=} does not match {scales.shape=}"

    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)

    *prefix_shape, G, B = blocks.shape
    rows_total = math.prod(prefix_shape) * G

    blocks = blocks.reshape(rows_total, B)
    scales = scales.reshape(rows_total, 1)

    out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)

        blk = blocks[r0:r1]
        exp = scales[r0:r1]

        # nibble indices -> int64
        idx_lo = (blk & 0x0F).to(torch.long)
        idx_hi = (blk >> 4).to(torch.long)

        sub = out[r0:r1]
        sub[:, 0::2] = lut[idx_lo]
        sub[:, 1::2] = lut[idx_hi]

        torch.ldexp(sub, exp, out=sub)
        del idx_lo, idx_hi, blk, exp

    result = out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)

    if is_dtensor:
        result = DTensor.from_local(
            result, device_mesh=device_mesh, placements=placements
        )

    return result


class GptOssStateDictAdapter(MoEStateDictAdapter):
    def __init__(self, model_args: GptOssModelArgs, hf_assets_path: str | None):
        super().__init__(model_args, hf_assets_path)
        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            # Attention module
            "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
            "model.layers.{}.self_attn.q_proj.bias": "layers.{}.attention.wq.bias",
            "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
            "model.layers.{}.self_attn.k_proj.bias": "layers.{}.attention.wk.bias",
            "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
            "model.layers.{}.self_attn.v_proj.bias": "layers.{}.attention.wv.bias",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.self_attn.o_proj.bias": "layers.{}.attention.wo.bias",
            "model.layers.{}.self_attn.sinks": "layers.{}.attention.sinks",
            # Transformer layer
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            # MoE
            (
                "model.layers.{}.mlp.experts.gate_up_proj_blocks",
                "model.layers.{}.mlp.experts.gate_up_proj_scales",
            ): "layers.{}.moe.experts.mlp1_weight",
            "model.layers.{}.mlp.experts.gate_up_proj_bias": "layers.{}.moe.experts.mlp1_bias",
            (
                "model.layers.{}.mlp.experts.down_proj_blocks",
                "model.layers.{}.mlp.experts.down_proj_scales",
            ): "layers.{}.moe.experts.mlp2_weight",
            "model.layers.{}.mlp.experts.down_proj_bias": "layers.{}.moe.experts.mlp2_bias",
            "model.layers.{}.mlp.router.weight": "layers.{}.moe.router.gate.weight",
            "model.layers.{}.mlp.router.bias": "layers.{}.moe.router.gate.bias",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Convert from a tt model state dict to a hf format state dict.
        Warning: Conversion does not support mxfp4 quantization,
                 and the function is only for the purpose of loading from hf checkpoints.
                 TODO: Add support for exact conversion of mxfp4 quantized tensors,
                 then one can save into hf checkpoints with last_save_in_hf = true.
        """
        to_hf_map = {v: k for k, v in self.from_hf_map.items()}
        hf_state_dict = {}

        for key, value in state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                if abstract_key not in to_hf_map:
                    continue
                layer_num = re.search(r"\d+", key).group(0)
                hf_key = to_hf_map[abstract_key]
                match hf_key:
                    case (blocks, scales):
                        blocks = blocks.format(layer_num)
                        scales = scales.format(layer_num)
                        hf_state_dict[blocks] = value.new_empty(
                            (*value.shape[:2], value.shape[2] // 32, 16),
                            dtype=torch.uint8,
                        )
                        hf_state_dict[scales] = value.new_empty(
                            (*value.shape[:2], value.shape[2] // 32),
                            dtype=torch.uint8,
                        )
                    case tensor_name:
                        tensor_name = tensor_name.format(layer_num)
                        hf_state_dict[tensor_name] = value
            else:
                hf_key = to_hf_map[key]
                hf_state_dict[hf_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Convert from quantized hf format state dict to tt model state dict.
        """

        state_dict = {}

        subtract_key = lambda key: re.sub(r"(\d+)", "{}", key, count=1)

        for key, value in hf_state_dict.items():
            if "layers" in key:
                layer_num = re.search(r"\d+", key).group(0)
                if "_blocks" in key:
                    value_scale = hf_state_dict[key.replace("_blocks", "_scales")]
                    abstract_key = (
                        subtract_key(key),
                        subtract_key(key.replace("_blocks", "_scales")),
                    )
                    tt_key = self.from_hf_map[abstract_key]
                    tt_key = tt_key.format(layer_num)
                    dequantized_values = get_mxfp4_tensor(value, value_scale)
                    state_dict[tt_key] = dequantized_values
                elif "_scales" not in key:
                    abstract_key = subtract_key(key)
                    tt_key = self.from_hf_map[abstract_key]
                    tt_key = tt_key.format(layer_num)
                    state_dict[tt_key] = value
            else:
                tt_key = self.from_hf_map[key]
                state_dict[tt_key] = value

        return state_dict

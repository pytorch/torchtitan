# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Any

import torch
from torch.distributed.checkpoint import HuggingFaceStorageReader

from torchtitan.models.common.attention import FusedGQAttention
from torchtitan.models.utils import MoEStateDictAdapter

from .model import GptOssModel


class GptOssStateDictAdapter(MoEStateDictAdapter):
    def __init__(self, model_config: GptOssModel.Config, hf_assets_path: str | None):
        super().__init__(model_config, hf_assets_path)
        self.fuse_qkv = isinstance(
            model_config.layers[0].attention, FusedGQAttention.Config
        )

        if self.fuse_qkv:
            qkv_map = {
                "model.layers.{}.self_attn.q_proj.weight": None,
                "model.layers.{}.self_attn.q_proj.bias": None,
                "model.layers.{}.self_attn.k_proj.weight": None,
                "model.layers.{}.self_attn.k_proj.bias": None,
                "model.layers.{}.self_attn.v_proj.weight": None,
                "model.layers.{}.self_attn.v_proj.bias": None,
            }
        else:
            qkv_map = {
                "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
                "model.layers.{}.self_attn.q_proj.bias": "layers.{}.attention.wq.bias",
                "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
                "model.layers.{}.self_attn.k_proj.bias": "layers.{}.attention.wk.bias",
                "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
                "model.layers.{}.self_attn.v_proj.bias": "layers.{}.attention.wv.bias",
            }

        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            # Attention module
            **qkv_map,  # pyrefly: ignore [invalid-argument]
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.self_attn.o_proj.bias": "layers.{}.attention.wo.bias",
            "model.layers.{}.self_attn.sinks": "layers.{}.attention.sinks",
            # Transformer layer
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            # MoE
            "model.layers.{}.mlp.experts.gate_up_proj_blocks": "layers.{}.moe.experts.mlp1_weight",
            "model.layers.{}.mlp.experts.gate_up_proj_bias": "layers.{}.moe.experts.mlp1_bias",
            "model.layers.{}.mlp.experts.down_proj_blocks": "layers.{}.moe.experts.mlp2_weight",
            "model.layers.{}.mlp.experts.down_proj_bias": "layers.{}.moe.experts.mlp2_bias",
            "model.layers.{}.mlp.router.weight": "layers.{}.moe.router.gate.weight",
            "model.layers.{}.mlp.router.bias": "layers.{}.moe.router.gate.bias",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
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

    def _get_attention_dims(self) -> tuple[int, int, int]:
        """Return (n_heads, n_kv_heads, head_dim) from model config."""
        attn = self.model_config.layers[0].attention
        n_heads = attn.n_heads
        n_kv_heads = attn.n_kv_heads if attn.n_kv_heads is not None else n_heads
        head_dim = (
            attn.head_dim
            if attn.head_dim is not None
            else self.model_config.dim // n_heads
        )
        return n_heads, n_kv_heads, head_dim

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        Convert from a tt model state dict to a hf format state dict.

        Only map keys without changing shapes to the same as MXFP4 checkpoint.
        For loading from quantized checkpoints, the QuantizedHuggingFaceStorageReader
            will handle dequantization during load.

        Warning: Conversion does not support saving to mxfp4 quantization format.
                 One can save into unquantized hf checkpoints with last_save_in_hf = true.
        """
        if self.fuse_qkv:
            to_hf_map = {v: k for k, v in self.from_hf_map.items() if v is not None}
            n_heads, n_kv_heads, head_dim = self._get_attention_dims()
        else:
            to_hf_map = {v: k for k, v in self.from_hf_map.items()}
        hf_state_dict = {}

        for key, value in state_dict.items():
            if "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore
                layer_num = re.search(r"\d+", key).group(0)

                if self.fuse_qkv and abstract_key == "layers.{}.attention.wqkv.weight":
                    wq, wk, wv = self.fused_to_separate_qkv(
                        # pyrefly: ignore [unbound-name]
                        value, n_heads, n_kv_heads, head_dim
                    )
                    hf_state_dict[
                        f"model.layers.{layer_num}.self_attn.q_proj.weight"
                    ] = wq
                    hf_state_dict[
                        f"model.layers.{layer_num}.self_attn.k_proj.weight"
                    ] = wk
                    hf_state_dict[
                        f"model.layers.{layer_num}.self_attn.v_proj.weight"
                    ] = wv
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

        state_dict = {}
        # Collect Q/K/V per layer for fusing (only used when fuse_qkv=True)
        pending_qkv: dict[str, dict[str, torch.Tensor]] = {}

        if self.fuse_qkv:
            n_heads, n_kv_heads, head_dim = self._get_attention_dims()

        for key, value in hf_state_dict.items():
            if "layers" in key:
                # pyrefly: ignore
                layer_num = re.search(r"\d+", key).group(0)
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)

                if self.fuse_qkv and abstract_key in (
                    "model.layers.{}.self_attn.q_proj.weight",
                    "model.layers.{}.self_attn.k_proj.weight",
                    "model.layers.{}.self_attn.v_proj.weight",
                ):
                    if layer_num not in pending_qkv:
                        pending_qkv[layer_num] = {}
                    proj = abstract_key.split(".")[-2]  # q_proj, k_proj, v_proj
                    pending_qkv[layer_num][proj] = value
                    if len(pending_qkv[layer_num]) == 3:
                        fused = self.separate_to_fused_qkv(
                            pending_qkv[layer_num]["q_proj"],
                            pending_qkv[layer_num]["k_proj"],
                            pending_qkv[layer_num]["v_proj"],
                            n_heads,  # pyrefly: ignore [unbound-name]
                            n_kv_heads,  # pyrefly: ignore [unbound-name]
                            head_dim,  # pyrefly: ignore [unbound-name]
                        )
                        state_dict[f"layers.{layer_num}.attention.wqkv.weight"] = fused
                        del pending_qkv[layer_num]
                    continue

                tt_key = self.from_hf_map.get(abstract_key)
                if tt_key is None:
                    continue
                tt_key = tt_key.format(layer_num)
                state_dict[tt_key] = value
            else:
                tt_key = self.from_hf_map[key]
                state_dict[tt_key] = value

        if self.fuse_qkv and pending_qkv:
            raise ValueError(
                f"Incomplete Q/K/V projections for layers: {list(pending_qkv.keys())}"
            )

        return state_dict

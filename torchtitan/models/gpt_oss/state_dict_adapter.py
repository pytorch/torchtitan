# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
from typing import Any

import torch
from torch.distributed.checkpoint import HuggingFaceStorageReader

from torchtitan.models.common.attention import FusedQKVLinear
from torchtitan.models.utils import MoEStateDictAdapter
from .model import GptOssModel


class GptOssStateDictAdapter(MoEStateDictAdapter):
    def __init__(self, model_config: GptOssModel.Config, hf_assets_path: str | None):
        super().__init__(model_config, hf_assets_path)
        self.fuse_qkv = isinstance(
            model_config.layers[0].attention.qkv_linear, FusedQKVLinear.Config
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
                "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.qkv_linear.wq.weight",
                "model.layers.{}.self_attn.q_proj.bias": "layers.{}.attention.qkv_linear.wq.bias",
                "model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.qkv_linear.wk.weight",
                "model.layers.{}.self_attn.k_proj.bias": "layers.{}.attention.qkv_linear.wk.bias",
                "model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.qkv_linear.wv.weight",
                "model.layers.{}.self_attn.v_proj.bias": "layers.{}.attention.qkv_linear.wv.bias",
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

    @staticmethod
    def _fused_to_separate_qkv_bias(
        fused_bias: torch.Tensor,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Split fused wqkv bias [n_kv * R * hd] into separate Q, K, V biases."""
        heads_per_kv = n_heads // n_kv_heads
        r_dim = heads_per_kv + 2
        b = fused_bias.view(n_kv_heads, r_dim, head_dim)
        bq = b[:, :heads_per_kv, :].reshape(n_heads * head_dim)
        bk = b[:, heads_per_kv, :].reshape(n_kv_heads * head_dim)
        bv = b[:, heads_per_kv + 1, :].reshape(n_kv_heads * head_dim)
        return bq, bk, bv

    @staticmethod
    def _separate_to_fused_qkv_bias(
        bq: torch.Tensor,
        bk: torch.Tensor,
        bv: torch.Tensor,
        n_heads: int,
        n_kv_heads: int,
        head_dim: int,
    ) -> torch.Tensor:
        """Combine separate Q, K, V biases into fused wqkv layout."""
        heads_per_kv = n_heads // n_kv_heads
        r_dim = heads_per_kv + 2
        q = bq.view(n_kv_heads, heads_per_kv, head_dim)
        k = bk.view(n_kv_heads, 1, head_dim)
        v = bv.view(n_kv_heads, 1, head_dim)
        fused = torch.cat([q, k, v], dim=1)
        return fused.reshape(n_kv_heads * r_dim * head_dim)

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

                if self.fuse_qkv and abstract_key in (
                    "layers.{}.attention.qkv_linear.wqkv.weight",
                    "layers.{}.attention.qkv_linear.wqkv.bias",
                ):
                    is_bias = abstract_key.endswith("bias")
                    if is_bias:
                        bq, bk, bv = self._fused_to_separate_qkv_bias(
                            value,
                            n_heads,  # pyrefly: ignore [unbound-name]
                            n_kv_heads,  # pyrefly: ignore [unbound-name]
                            head_dim,  # pyrefly: ignore [unbound-name]
                        )
                    else:
                        bq, bk, bv = self.fused_to_separate_qkv(
                            value,
                            n_heads,  # pyrefly: ignore [unbound-name]
                            n_kv_heads,  # pyrefly: ignore [unbound-name]
                            head_dim,  # pyrefly: ignore [unbound-name]
                        )
                    suffix = "bias" if is_bias else "weight"
                    hf_state_dict[
                        f"model.layers.{layer_num}.self_attn.q_proj.{suffix}"
                    ] = bq
                    hf_state_dict[
                        f"model.layers.{layer_num}.self_attn.k_proj.{suffix}"
                    ] = bk
                    hf_state_dict[
                        f"model.layers.{layer_num}.self_attn.v_proj.{suffix}"
                    ] = bv
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
        # Separate dicts for weight and bias since they arrive independently
        pending_qkv_weight: dict[str, dict[str, torch.Tensor]] = {}
        pending_qkv_bias: dict[str, dict[str, torch.Tensor]] = {}

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
                    "model.layers.{}.self_attn.q_proj.bias",
                    "model.layers.{}.self_attn.k_proj.bias",
                    "model.layers.{}.self_attn.v_proj.bias",
                ):
                    is_bias = abstract_key.endswith("bias")
                    pending = pending_qkv_bias if is_bias else pending_qkv_weight
                    if layer_num not in pending:
                        pending[layer_num] = {}
                    proj = abstract_key.split(".")[-2]  # q_proj, k_proj, v_proj
                    pending[layer_num][proj] = value
                    if len(pending[layer_num]) == 3:
                        if is_bias:
                            fused = self._separate_to_fused_qkv_bias(
                                pending[layer_num]["q_proj"],
                                pending[layer_num]["k_proj"],
                                pending[layer_num]["v_proj"],
                                n_heads,  # pyrefly: ignore [unbound-name]
                                n_kv_heads,  # pyrefly: ignore [unbound-name]
                                head_dim,  # pyrefly: ignore [unbound-name]
                            )
                        else:
                            fused = self.separate_to_fused_qkv(
                                pending[layer_num]["q_proj"],
                                pending[layer_num]["k_proj"],
                                pending[layer_num]["v_proj"],
                                n_heads,  # pyrefly: ignore [unbound-name]
                                n_kv_heads,  # pyrefly: ignore [unbound-name]
                                head_dim,  # pyrefly: ignore [unbound-name]
                            )
                        suffix = "bias" if is_bias else "weight"
                        state_dict[
                            f"layers.{layer_num}.attention.qkv_linear.wqkv.{suffix}"
                        ] = fused
                        del pending[layer_num]
                    continue

                tt_key = self.from_hf_map.get(abstract_key)
                if tt_key is None:
                    continue
                tt_key = tt_key.format(layer_num)
                state_dict[tt_key] = value
            else:
                tt_key = self.from_hf_map[key]
                state_dict[tt_key] = value

        if self.fuse_qkv and pending_qkv_weight:
            raise ValueError(
                f"Incomplete Q/K/V weight projections for layers: {list(pending_qkv_weight.keys())}"
            )
        if self.fuse_qkv and pending_qkv_bias:
            raise ValueError(
                f"Incomplete Q/K/V bias projections for layers: {list(pending_qkv_bias.keys())}"
            )

        return state_dict

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import re

from typing import Any

import torch
from torchtitan.protocols.state_dict_adapter import BaseStateDictAdapter

from .model import FluxModel

logger = logging.getLogger()

_SINGLE_BLOCK_RE = re.compile(r"^single_blocks\.(\d+)\.")
_DOUBLE_BLOCK_RE = re.compile(r"^double_blocks\.(\d+)\.")
_HF_SINGLE_BLOCK_RE = re.compile(r"^single_transformer_blocks\.(\d+)\.")
_HF_DOUBLE_BLOCK_RE = re.compile(r"^transformer_blocks\.(\d+)\.")


class FluxStateDictAdapter(BaseStateDictAdapter):
    """
    State dict adapter for Flux model to convert between HuggingFace safetensors format
    and torchtitan DCP format.

    This state dict adapter handles only the state dict of transformer from Flux HF model repo.
    """

    def __init__(self, model_config: FluxModel.Config, hf_assets_path: str | None):
        # Flux needs custom index file resolution (multimodal subdirs,
        # diffusion_pytorch_model.safetensors.index.json), so skip the
        # base class index loading by passing hf_assets_path=None.
        super().__init__(model_config)
        self.model_config = model_config
        self.hf_assets_path = hf_assets_path

        if not hf_assets_path:
            self.fqn_to_index_mapping = None
            return

        # Multimodal repos store transformer weights in a subdirectory
        if os.path.exists(os.path.join(hf_assets_path, "model_index.json")):
            hf_assets_path = os.path.join(hf_assets_path, "transformers")

        index_files = [
            "model.safetensors.index.json",
            "diffusion_pytorch_model.safetensors.index.json",
        ]

        hf_safetensors_indx = None
        for index_file in index_files:
            mapping_path = os.path.join(hf_assets_path, index_file)
            if os.path.exists(mapping_path):
                with open(mapping_path, "r") as f:
                    hf_safetensors_indx = json.load(f)
                break
        if hf_safetensors_indx is None:
            logger.warning(
                f"no safetensors index file found at hf_assets_path: {hf_assets_path}. "
                "Defaulting to saving a single safetensors file if checkpoint is saved in HF format.",
            )

        if hf_safetensors_indx:
            self.fqn_to_index_mapping = {}
            for hf_key, raw_indx in hf_safetensors_indx["weight_map"].items():
                # pyrefly: ignore [missing-attribute]
                indx = re.search(r"\d+", raw_indx).group(0)
                self.fqn_to_index_mapping[hf_key] = int(indx)
        else:
            self.fqn_to_index_mapping = None

    # Original flux implementation and HF swap shift and scale
    # https://github.com/huggingface/diffusers/blob/main/scripts/convert_flux_to_diffusers.py#L63-L68
    def _swap_scale_shift(self, weight):
        shift, scale = weight.chunk(2, dim=0)
        new_weight = torch.cat([scale, shift], dim=0)
        return new_weight

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert TorchTitan DCP state dict to HuggingFace safetensors format."""
        RENAME = {
            "img_in.bias": "x_embedder.bias",
            "img_in.weight": "x_embedder.weight",
            "txt_in.bias": "context_embedder.bias",
            "txt_in.weight": "context_embedder.weight",
            "final_layer.linear.bias": "proj_out.bias",
            "final_layer.linear.weight": "proj_out.weight",
            "vector_in.in_layer.bias": "time_text_embed.text_embedder.linear_1.bias",
            "vector_in.in_layer.weight": "time_text_embed.text_embedder.linear_1.weight",
            "time_in.in_layer.bias": "time_text_embed.timestep_embedder.linear_1.bias",
            "time_in.in_layer.weight": "time_text_embed.timestep_embedder.linear_1.weight",
            "vector_in.out_layer.bias": "time_text_embed.text_embedder.linear_2.bias",
            "vector_in.out_layer.weight": "time_text_embed.text_embedder.linear_2.weight",
            "time_in.out_layer.bias": "time_text_embed.timestep_embedder.linear_2.bias",
            "time_in.out_layer.weight": "time_text_embed.timestep_embedder.linear_2.weight",
        }
        SINGLE_BLOCK_RENAME = {
            "norm.key_norm.weight": "attn.norm_k.weight",
            "norm.query_norm.weight": "attn.norm_q.weight",
            "modulation.lin.bias": "norm.linear.bias",
            "modulation.lin.weight": "norm.linear.weight",
            "linear2.bias": "proj_out.bias",
            "linear2.weight": "proj_out.weight",
        }
        DOUBLE_BLOCK_RENAME = {
            "txt_attn.norm.key_norm.weight": "attn.norm_added_k.weight",
            "txt_attn.norm.query_norm.weight": "attn.norm_added_q.weight",
            "img_attn.norm.key_norm.weight": "attn.norm_k.weight",
            "img_attn.norm.query_norm.weight": "attn.norm_q.weight",
            "txt_attn.proj.bias": "attn.to_add_out.bias",
            "txt_attn.proj.weight": "attn.to_add_out.weight",
            "img_attn.proj.bias": "attn.to_out.0.bias",
            "img_attn.proj.weight": "attn.to_out.0.weight",
            "img_mlp.0.bias": "ff.net.0.proj.bias",
            "img_mlp.0.weight": "ff.net.0.proj.weight",
            "img_mlp.2.bias": "ff.net.2.bias",
            "img_mlp.2.weight": "ff.net.2.weight",
            "txt_mlp.0.bias": "ff_context.net.0.proj.bias",
            "txt_mlp.0.weight": "ff_context.net.0.proj.weight",
            "txt_mlp.2.bias": "ff_context.net.2.bias",
            "txt_mlp.2.weight": "ff_context.net.2.weight",
            "img_mod.lin.bias": "norm1.linear.bias",
            "img_mod.lin.weight": "norm1.linear.weight",
            "txt_mod.lin.bias": "norm1_context.linear.bias",
            "txt_mod.lin.weight": "norm1_context.linear.weight",
        }

        hf: dict[str, Any] = {}

        for key, value in state_dict.items():
            if key in RENAME:
                hf[RENAME[key]] = value
            # adaLN modulation — swap scale/shift
            elif key == "final_layer.adaLN_modulation.1.bias":
                hf["norm_out.linear.bias"] = self._swap_scale_shift(value)
            elif key == "final_layer.adaLN_modulation.1.weight":
                hf["norm_out.linear.weight"] = self._swap_scale_shift(value)

            # -- Single blocks --
            elif m := _SINGLE_BLOCK_RE.match(key):
                layer = m.group(1)
                suffix = key[m.end() :]

                if suffix in SINGLE_BLOCK_RENAME:
                    hf[
                        f"single_transformer_blocks.{layer}.{SINGLE_BLOCK_RENAME[suffix]}"
                    ] = value
                # Combined linear1 → split into q, k, v, mlp
                elif suffix in ("linear1.bias", "linear1.weight"):
                    mlp_hidden_dim = int(
                        self.model_config.hidden_size * self.model_config.mlp_ratio
                    )
                    split_plan = [
                        self.model_config.hidden_size,
                        self.model_config.hidden_size,
                        self.model_config.hidden_size,
                        mlp_hidden_dim,
                    ]
                    q, k, v, mlp = torch.split(value, split_plan, dim=0)
                    param = suffix.split(".")[1]  # "bias" or "weight"
                    hf[f"single_transformer_blocks.{layer}.attn.to_q.{param}"] = q
                    hf[f"single_transformer_blocks.{layer}.attn.to_k.{param}"] = k
                    hf[f"single_transformer_blocks.{layer}.attn.to_v.{param}"] = v
                    hf[f"single_transformer_blocks.{layer}.proj_mlp.{param}"] = mlp

            # -- Double blocks --
            elif m := _DOUBLE_BLOCK_RE.match(key):
                layer = m.group(1)
                suffix = key[m.end() :]

                if suffix in DOUBLE_BLOCK_RENAME:
                    hf[
                        f"transformer_blocks.{layer}.{DOUBLE_BLOCK_RENAME[suffix]}"
                    ] = value
                # Combined qkv → split into q, k, v
                elif suffix in (
                    "txt_attn.qkv.bias",
                    "txt_attn.qkv.weight",
                    "img_attn.qkv.bias",
                    "img_attn.qkv.weight",
                ):
                    q, k, v = torch.split(value, self.model_config.hidden_size, dim=0)
                    param = suffix.split(".")[-1]  # "bias" or "weight"
                    if suffix.startswith("txt_attn"):
                        hf[f"transformer_blocks.{layer}.attn.add_q_proj.{param}"] = q
                        hf[f"transformer_blocks.{layer}.attn.add_k_proj.{param}"] = k
                        hf[f"transformer_blocks.{layer}.attn.add_v_proj.{param}"] = v
                    else:
                        hf[f"transformer_blocks.{layer}.attn.to_q.{param}"] = q
                        hf[f"transformer_blocks.{layer}.attn.to_k.{param}"] = k
                        hf[f"transformer_blocks.{layer}.attn.to_v.{param}"] = v

        return hf

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert HuggingFace safetensors state dict to TorchTitan DCP format."""
        RENAME = {
            "x_embedder.bias": "img_in.bias",
            "x_embedder.weight": "img_in.weight",
            "context_embedder.bias": "txt_in.bias",
            "context_embedder.weight": "txt_in.weight",
            "proj_out.bias": "final_layer.linear.bias",
            "proj_out.weight": "final_layer.linear.weight",
            "time_text_embed.text_embedder.linear_1.bias": "vector_in.in_layer.bias",
            "time_text_embed.text_embedder.linear_1.weight": "vector_in.in_layer.weight",
            "time_text_embed.timestep_embedder.linear_1.bias": "time_in.in_layer.bias",
            "time_text_embed.timestep_embedder.linear_1.weight": "time_in.in_layer.weight",
            "time_text_embed.text_embedder.linear_2.bias": "vector_in.out_layer.bias",
            "time_text_embed.text_embedder.linear_2.weight": "vector_in.out_layer.weight",
            "time_text_embed.timestep_embedder.linear_2.bias": "time_in.out_layer.bias",
            "time_text_embed.timestep_embedder.linear_2.weight": "time_in.out_layer.weight",
        }
        SINGLE_BLOCK_RENAME = {
            "attn.norm_k.weight": "norm.key_norm.weight",
            "attn.norm_q.weight": "norm.query_norm.weight",
            "norm.linear.bias": "modulation.lin.bias",
            "norm.linear.weight": "modulation.lin.weight",
            "proj_out.bias": "linear2.bias",
            "proj_out.weight": "linear2.weight",
        }
        DOUBLE_BLOCK_RENAME = {
            "attn.norm_added_k.weight": "txt_attn.norm.key_norm.weight",
            "attn.norm_added_q.weight": "txt_attn.norm.query_norm.weight",
            "attn.norm_k.weight": "img_attn.norm.key_norm.weight",
            "attn.norm_q.weight": "img_attn.norm.query_norm.weight",
            "attn.to_add_out.bias": "txt_attn.proj.bias",
            "attn.to_add_out.weight": "txt_attn.proj.weight",
            "attn.to_out.0.bias": "img_attn.proj.bias",
            "attn.to_out.0.weight": "img_attn.proj.weight",
            "ff.net.0.proj.bias": "img_mlp.0.bias",
            "ff.net.0.proj.weight": "img_mlp.0.weight",
            "ff.net.2.bias": "img_mlp.2.bias",
            "ff.net.2.weight": "img_mlp.2.weight",
            "ff_context.net.0.proj.bias": "txt_mlp.0.bias",
            "ff_context.net.0.proj.weight": "txt_mlp.0.weight",
            "ff_context.net.2.bias": "txt_mlp.2.bias",
            "ff_context.net.2.weight": "txt_mlp.2.weight",
            "norm1.linear.bias": "img_mod.lin.bias",
            "norm1.linear.weight": "img_mod.lin.weight",
            "norm1_context.linear.bias": "txt_mod.lin.bias",
            "norm1_context.linear.weight": "txt_mod.lin.weight",
        }

        sd: dict[str, Any] = {}
        # Collect combination keys: {tt_fqn: list of (order_index, value)}
        to_combine: dict[str, list[tuple[int, torch.Tensor]]] = {}

        def _collect(tt_fqn: str, order_index: int, val: torch.Tensor) -> None:
            if tt_fqn not in to_combine:
                to_combine[tt_fqn] = []
            to_combine[tt_fqn].append((order_index, val))

        for key, value in hf_state_dict.items():
            if key in RENAME:
                sd[RENAME[key]] = value
            # adaLN modulation — swap scale/shift
            elif key == "norm_out.linear.bias":
                sd["final_layer.adaLN_modulation.1.bias"] = self._swap_scale_shift(
                    value
                )
            elif key == "norm_out.linear.weight":
                sd["final_layer.adaLN_modulation.1.weight"] = self._swap_scale_shift(
                    value
                )

            # -- Single blocks --
            elif m := _HF_SINGLE_BLOCK_RE.match(key):
                layer = m.group(1)
                suffix = key[m.end() :]

                if suffix in SINGLE_BLOCK_RENAME:
                    sd[f"single_blocks.{layer}.{SINGLE_BLOCK_RENAME[suffix]}"] = value
                # Combination keys: q(0), k(1), v(2), mlp(3) → linear1
                elif suffix == "attn.to_q.bias":
                    _collect(f"single_blocks.{layer}.linear1.bias", 0, value)
                elif suffix == "attn.to_k.bias":
                    _collect(f"single_blocks.{layer}.linear1.bias", 1, value)
                elif suffix == "attn.to_v.bias":
                    _collect(f"single_blocks.{layer}.linear1.bias", 2, value)
                elif suffix == "proj_mlp.bias":
                    _collect(f"single_blocks.{layer}.linear1.bias", 3, value)
                elif suffix == "attn.to_q.weight":
                    _collect(f"single_blocks.{layer}.linear1.weight", 0, value)
                elif suffix == "attn.to_k.weight":
                    _collect(f"single_blocks.{layer}.linear1.weight", 1, value)
                elif suffix == "attn.to_v.weight":
                    _collect(f"single_blocks.{layer}.linear1.weight", 2, value)
                elif suffix == "proj_mlp.weight":
                    _collect(f"single_blocks.{layer}.linear1.weight", 3, value)

            # -- Double blocks --
            elif m := _HF_DOUBLE_BLOCK_RE.match(key):
                layer = m.group(1)
                suffix = key[m.end() :]

                if suffix in DOUBLE_BLOCK_RENAME:
                    sd[f"double_blocks.{layer}.{DOUBLE_BLOCK_RENAME[suffix]}"] = value
                # Combination keys — txt_attn qkv: q(0), k(1), v(2)
                elif suffix == "attn.add_q_proj.bias":
                    _collect(f"double_blocks.{layer}.txt_attn.qkv.bias", 0, value)
                elif suffix == "attn.add_k_proj.bias":
                    _collect(f"double_blocks.{layer}.txt_attn.qkv.bias", 1, value)
                elif suffix == "attn.add_v_proj.bias":
                    _collect(f"double_blocks.{layer}.txt_attn.qkv.bias", 2, value)
                elif suffix == "attn.add_q_proj.weight":
                    _collect(f"double_blocks.{layer}.txt_attn.qkv.weight", 0, value)
                elif suffix == "attn.add_k_proj.weight":
                    _collect(f"double_blocks.{layer}.txt_attn.qkv.weight", 1, value)
                elif suffix == "attn.add_v_proj.weight":
                    _collect(f"double_blocks.{layer}.txt_attn.qkv.weight", 2, value)
                # Combination keys — img_attn qkv: q(0), k(1), v(2)
                elif suffix == "attn.to_q.bias":
                    _collect(f"double_blocks.{layer}.img_attn.qkv.bias", 0, value)
                elif suffix == "attn.to_k.bias":
                    _collect(f"double_blocks.{layer}.img_attn.qkv.bias", 1, value)
                elif suffix == "attn.to_v.bias":
                    _collect(f"double_blocks.{layer}.img_attn.qkv.bias", 2, value)
                elif suffix == "attn.to_q.weight":
                    _collect(f"double_blocks.{layer}.img_attn.qkv.weight", 0, value)
                elif suffix == "attn.to_k.weight":
                    _collect(f"double_blocks.{layer}.img_attn.qkv.weight", 1, value)
                elif suffix == "attn.to_v.weight":
                    _collect(f"double_blocks.{layer}.img_attn.qkv.weight", 2, value)

        # Concatenate combination groups in sorted order
        for tt_fqn, parts in to_combine.items():
            parts.sort(key=lambda x: x[0])
            sd[tt_fqn] = torch.cat([v for _, v in parts], dim=0)

        return sd

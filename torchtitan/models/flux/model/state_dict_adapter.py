# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import logging
import os
import re

from collections import defaultdict
from typing import Any

import torch
from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .model import FluxModel

logger = logging.getLogger()


class FluxStateDictAdapter(StateDictAdapter):
    """
    State dict adapter for Flux model to convert between HuggingFace safetensors format
    and torchtitan DCP format.

    This state dict adapter handles only the state dict of transformer from Flux HF model repo.
    """

    def __init__(self, model_config: FluxModel.Config, hf_assets_path: str | None):
        super().__init__(model_config, hf_assets_path)
        # Build fqn to index mapping if hf_assets_path
        if hf_assets_path:
            # If directory is multimodal ensure that hf_assets_path is to the folder containing transformer's safetensors
            if os.path.exists(os.path.join(hf_assets_path, "model_index.json")):
                hf_assets_path = os.path.join(hf_assets_path, "transformers")

            # Check if safetensors index file exists
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
                    f"no safetensors index file found at hf_assets_path: {hf_assets_path}. \
                    Defaulting to saving a single safetensors file if checkpoint is saved in HF format.",
                )

            if hf_safetensors_indx:
                self.fqn_to_index_mapping = {}
                for hf_key, raw_indx in hf_safetensors_indx["weight_map"].items():
                    # pyrefly: ignore [missing-attribute]
                    indx = re.search(r"\d+", raw_indx).group(0)
                    self.fqn_to_index_mapping[hf_key] = indx
            else:
                self.fqn_to_index_mapping = None

        self.model_config = model_config
        self.hf_assets_path = hf_assets_path

        # mapping containing direct 1 to 1 mappings from HF to torchtitan
        self.from_hf_map_direct = {
            "x_embedder.bias": "img_in.bias",
            "x_embedder.weight": "img_in.weight",
            "context_embedder.bias": "txt_in.bias",
            "context_embedder.weight": "txt_in.weight",
            "norm_out.linear.bias": "final_layer.adaLN_modulation.1.bias",
            "norm_out.linear.weight": "final_layer.adaLN_modulation.1.weight",
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
            "single_transformer_blocks.{}.attn.norm_k.weight": "single_blocks.{}.norm.key_norm.weight",
            "single_transformer_blocks.{}.attn.norm_q.weight": "single_blocks.{}.norm.query_norm.weight",
            "single_transformer_blocks.{}.norm.linear.bias": "single_blocks.{}.modulation.lin.bias",
            "single_transformer_blocks.{}.norm.linear.weight": "single_blocks.{}.modulation.lin.weight",
            "single_transformer_blocks.{}.proj_out.bias": "single_blocks.{}.linear2.bias",
            "single_transformer_blocks.{}.proj_out.weight": "single_blocks.{}.linear2.weight",
            "transformer_blocks.{}.attn.norm_added_k.weight": "double_blocks.{}.txt_attn.norm.key_norm.weight",
            "transformer_blocks.{}.attn.norm_added_q.weight": "double_blocks.{}.txt_attn.norm.query_norm.weight",
            "transformer_blocks.{}.attn.norm_k.weight": "double_blocks.{}.img_attn.norm.key_norm.weight",
            "transformer_blocks.{}.attn.norm_q.weight": "double_blocks.{}.img_attn.norm.query_norm.weight",
            "transformer_blocks.{}.attn.to_add_out.bias": "double_blocks.{}.txt_attn.proj.bias",
            "transformer_blocks.{}.attn.to_add_out.weight": "double_blocks.{}.txt_attn.proj.weight",
            "transformer_blocks.{}.attn.to_out.0.bias": "double_blocks.{}.img_attn.proj.bias",
            "transformer_blocks.{}.attn.to_out.0.weight": "double_blocks.{}.img_attn.proj.weight",
            "transformer_blocks.{}.ff.net.0.proj.bias": "double_blocks.{}.img_mlp.0.bias",
            "transformer_blocks.{}.ff.net.0.proj.weight": "double_blocks.{}.img_mlp.0.weight",
            "transformer_blocks.{}.ff.net.2.bias": "double_blocks.{}.img_mlp.2.bias",
            "transformer_blocks.{}.ff.net.2.weight": "double_blocks.{}.img_mlp.2.weight",
            "transformer_blocks.{}.ff_context.net.0.proj.bias": "double_blocks.{}.txt_mlp.0.bias",
            "transformer_blocks.{}.ff_context.net.0.proj.weight": "double_blocks.{}.txt_mlp.0.weight",
            "transformer_blocks.{}.ff_context.net.2.bias": "double_blocks.{}.txt_mlp.2.bias",
            "transformer_blocks.{}.ff_context.net.2.weight": "double_blocks.{}.txt_mlp.2.weight",
            "transformer_blocks.{}.norm1.linear.bias": "double_blocks.{}.img_mod.lin.bias",
            "transformer_blocks.{}.norm1.linear.weight": "double_blocks.{}.img_mod.lin.weight",
            "transformer_blocks.{}.norm1_context.linear.bias": "double_blocks.{}.txt_mod.lin.bias",
            "transformer_blocks.{}.norm1_context.linear.weight": "double_blocks.{}.txt_mod.lin.weight",
        }

        # combination plan to keep track of the order of layers to be combined
        self.combination_plan = {
            "single_blocks.{}.linear1.bias": [
                "single_transformer_blocks.{}.attn.to_q.bias",
                "single_transformer_blocks.{}.attn.to_k.bias",
                "single_transformer_blocks.{}.attn.to_v.bias",
                "single_transformer_blocks.{}.proj_mlp.bias",
            ],
            "single_blocks.{}.linear1.weight": [
                "single_transformer_blocks.{}.attn.to_q.weight",
                "single_transformer_blocks.{}.attn.to_k.weight",
                "single_transformer_blocks.{}.attn.to_v.weight",
                "single_transformer_blocks.{}.proj_mlp.weight",
            ],
            "double_blocks.{}.txt_attn.qkv.bias": [
                "transformer_blocks.{}.attn.add_q_proj.bias",
                "transformer_blocks.{}.attn.add_k_proj.bias",
                "transformer_blocks.{}.attn.add_v_proj.bias",
            ],
            "double_blocks.{}.txt_attn.qkv.weight": [
                "transformer_blocks.{}.attn.add_q_proj.weight",
                "transformer_blocks.{}.attn.add_k_proj.weight",
                "transformer_blocks.{}.attn.add_v_proj.weight",
            ],
            "double_blocks.{}.img_attn.qkv.bias": [
                "transformer_blocks.{}.attn.to_q.bias",
                "transformer_blocks.{}.attn.to_k.bias",
                "transformer_blocks.{}.attn.to_v.bias",
            ],
            "double_blocks.{}.img_attn.qkv.weight": [
                "transformer_blocks.{}.attn.to_q.weight",
                "transformer_blocks.{}.attn.to_k.weight",
                "transformer_blocks.{}.attn.to_v.weight",
            ],
        }

        # reverse of combination plan: maps fqns to the fqn they are combined into
        self.reverse_combination_plan = {
            value: key
            for key, value_list in self.combination_plan.items()
            for value in value_list
        }

    # original flux implementation and HF swap shift and scale
    # https://github.com/huggingface/diffusers/blob/main/scripts/convert_flux_to_diffusers.py#L63-L68
    def _swap_scale_shift(self, weight):
        shift, scale = weight.chunk(2, dim=0)
        new_weight = torch.cat([scale, shift], dim=0)
        return new_weight

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert TorchTitan DCP state dict to HuggingFace safetensors format."""

        to_hf_map_direct = {
            v: k for k, v in self.from_hf_map_direct.items() if v is not None
        }
        hf_state_dict = {}

        for key, value in state_dict.items():
            # Extract layer_num and abstract key if necessary
            if "blocks" in key:
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)
                key = re.sub(r"(\d+)", "{}", key, count=1)
            else:
                layer_num = None

            if key in to_hf_map_direct:
                # handle direct mapping
                new_key = to_hf_map_direct[key]

                # perform swap to be compatible with HF
                if key in [
                    "final_layer.adaLN_modulation.1.weight",
                    "final_layer.adaLN_modulation.1.bias",
                ]:
                    value = self._swap_scale_shift(value)

                if new_key is None:
                    continue
                if layer_num:
                    new_key = new_key.format(layer_num)

                hf_state_dict[new_key] = value

            elif key in self.combination_plan:
                # handle splitting layers
                if key in [
                    "single_blocks.{}.linear1.bias",
                    "single_blocks.{}.linear1.weight",
                ]:
                    mlp_hidden_dim = int(
                        self.model_config.hidden_size * self.model_config.mlp_ratio
                    )
                    split_plan = [
                        self.model_config.hidden_size,
                        self.model_config.hidden_size,
                        self.model_config.hidden_size,
                        mlp_hidden_dim,
                    ]
                    # split into q, k, v, mlp
                    split_vals = torch.split(
                        value,
                        split_plan,
                        dim=0,
                    )
                else:
                    # split into q, k, v
                    split_vals = torch.split(
                        value, self.model_config.hidden_size, dim=0
                    )

                new_keys = (
                    abstract_key.format(layer_num)
                    for abstract_key in self.combination_plan[key]
                )

                for new_key, value in zip(new_keys, split_vals):
                    hf_state_dict[new_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert HuggingFace safetensors state dict to TorchTitan DCP format."""
        state_dict = {}

        # Keeps track of HF fqn values to combine into one TT fqn later
        # {tt_fqn : {hf_fqn1 : value}, {hf_fqn2 : value}, ...}
        to_combine = defaultdict(dict)

        for key, value in hf_state_dict.items():
            # extract layer_num and abstract key if necessary
            if "blocks" in key:
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)
                key = re.sub(r"(\d+)", "{}", key, count=1)
            else:
                layer_num = None

            if key in self.from_hf_map_direct:
                new_key = self.from_hf_map_direct[key]

                # perform swap to be compatible with HF
                if key in [
                    "norm_out.linear.weight",
                    "norm_out.linear.bias",
                ]:
                    value = self._swap_scale_shift(value)
                if new_key is None:
                    continue
                if layer_num:
                    new_key = new_key.format(layer_num)

                state_dict[new_key] = value
            elif key in self.reverse_combination_plan:
                # collect the layers that need to be combined
                tt_abstract_key = self.reverse_combination_plan[key]
                if tt_abstract_key is None:
                    continue
                to_combine[tt_abstract_key.format(layer_num)][
                    key.format(layer_num)
                ] = value

        # combine collected values
        for tt_fqn, hf_fqn_map in to_combine.items():
            # pyrefly: ignore [missing-attribute]
            layer_num = re.search(r"\d+", tt_fqn).group(0)
            tt_abstract_key = re.sub(r"(\d+)", "{}", tt_fqn, count=1)
            combine_values = []
            # use combination_plan to ensure correct order before concatenation
            for hf_abstract_key in self.combination_plan[tt_abstract_key]:
                hf_key = hf_abstract_key.format(layer_num)
                combine_values.append(hf_fqn_map[hf_key])

            value = torch.cat(combine_values, dim=0)
            state_dict[tt_fqn] = value

        return state_dict

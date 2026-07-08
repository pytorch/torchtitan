# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


import re
from typing import Any

import torch
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.tensor import DTensor

from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.models.utils import MoEStateDictAdapter
from .model import DeepSeekV3Model


class DeepSeekV3StateDictAdapter(MoEStateDictAdapter):
    """
    StateDictAdapter for DeepSeekV3 model.
    """

    def __init__(
        self,
        model_config: DeepSeekV3Model.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)
        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            # Attention Module
            "model.layers.{}.self_attn.kv_a_proj_with_mqa.weight": "layers.{}.attention.wkv_a.weight",
            "model.layers.{}.self_attn.kv_a_layernorm.weight": "layers.{}.attention.kv_norm.weight",
            "model.layers.{}.self_attn.kv_b_proj.weight": "layers.{}.attention.wkv_b.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            # MLP Module
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            # Transformer Layer
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            # MoE Module
            "model.layers.{}.mlp.experts.{}.gate_proj.weight": "layers.{}.moe.experts.w1_EFD",
            "model.layers.{}.mlp.experts.{}.up_proj.weight": "layers.{}.moe.experts.w3_EFD",
            "model.layers.{}.mlp.experts.{}.down_proj.weight": "layers.{}.moe.experts.w2_EDF",
            "model.layers.{}.mlp.gate.weight": "layers.{}.moe.router.gate.weight",
            "model.layers.{}.mlp.shared_experts.gate_proj.weight": "layers.{}.moe.shared_experts.w1.weight",
            "model.layers.{}.mlp.shared_experts.up_proj.weight": "layers.{}.moe.shared_experts.w3.weight",
            "model.layers.{}.mlp.shared_experts.down_proj.weight": "layers.{}.moe.shared_experts.w2.weight",
            "model.layers.{}.mlp.gate.e_score_correction_bias": "layers.{}.moe.expert_bias_E",
            # MTP Module
            "model.layers.{}.enorm.weight": "layers.{}.enorm.weight",
            "model.layers.{}.hnorm.weight": "layers.{}.hnorm.weight",
            "model.layers.{}.eh_proj.weight": "layers.{}.eh_proj.weight",
            "model.layers.{}.shared_head.norm.weight": "layers.{}.final_norm.weight",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
        }

        # Adjustments for from_hf_map based on model architecture
        if model_config.layers[0].attention.q_lora_rank != 0:
            self.from_hf_map.update(
                {
                    "model.layers.{}.self_attn.q_a_proj.weight": "layers.{}.attention.wq_a.weight",
                    "model.layers.{}.self_attn.q_a_layernorm.weight": "layers.{}.attention.q_norm.weight",
                    "model.layers.{}.self_attn.q_b_proj.weight": "layers.{}.attention.wq_b.weight",
                }
            )
        else:
            self.from_hf_map.update(
                {
                    "model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
                }
            )

    def _map_from_hf_layer_key(
        self,
        abstract_key: str,
        layer_num: str,
    ) -> tuple[str, str]:
        new_key = self.from_hf_map[abstract_key]
        mtp_cfg = self.model_config.mtp
        if mtp_cfg is not None and mtp_cfg.num_mtp_layers > 0:
            num_main_layers = len(self.model_config.layers)
            layer_idx = int(layer_num)
            if layer_idx >= num_main_layers:
                if not any(
                    new_key.startswith(f"layers.{{}}.{name}.")
                    for name in ("enorm", "hnorm", "eh_proj", "final_norm")
                ):
                    new_key = new_key.replace(
                        "layers.{}.",
                        "mtp_block.layers.{}.inner.",
                        1,
                    )
                else:
                    new_key = new_key.replace(
                        "layers.{}.", "mtp_block.layers.{}.", 1
                    )
                layer_num = str(layer_idx - num_main_layers)
        return new_key, layer_num

    def _map_to_hf_layer_key(
        self,
        key: str,
        to_hf_map: dict[str, str],
    ) -> tuple[str, str]:
        if key.startswith("mtp_block.layers."):
            abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
            # pyrefly: ignore [missing-attribute]
            layer_num = re.search(r"\d+", key).group(0)
            main_abstract_key = abstract_key.replace(
                "mtp_block.layers.{}.inner.",
                "layers.{}.",
                1,
            ).replace("mtp_block.layers.{}.", "layers.{}.", 1)
            hf_layer_num = str(len(self.model_config.layers) + int(layer_num))
            return to_hf_map[main_abstract_key], hf_layer_num

        abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
        # pyrefly: ignore [missing-attribute]
        layer_num = re.search(r"\d+", key).group(0)
        return to_hf_map[abstract_key], layer_num

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

            # NOTE: Now we use Quantized HF storage reader to read DeepSeek-V3 671B model.
            # If loading checkpoints without quantization, use HuggingFaceStorageReader instead
            BLOCK_SIZE = 128
            return QuantizedHuggingFaceStorageReader(
                path=path,
                target_dtype=torch.float32,
                block_size=BLOCK_SIZE,
                thread_count=4,
            )
        else:
            return HuggingFaceStorageReader(path)

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        1. Convert between the HF shape and the torchtitan shape.
        2. Split the GroupedExperts' weight into separate expert's weight.
        """

        to_hf_map = {v: k for k, v in self.from_hf_map.items()}

        hf_state_dict = {}

        for key, value in state_dict.items():
            if "moe.experts" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)
                if key.startswith("mtp_block.layers."):
                    abstract_key = abstract_key.replace(
                        "mtp_block.layers.{}.inner.",
                        "layers.{}.",
                        1,
                    ).replace("mtp_block.layers.{}.", "layers.{}.", 1)
                    layer_num = str(len(self.model_config.layers) + int(layer_num))
                new_abstract_key = to_hf_map[abstract_key]

                # Store the GroupedExperts Weight metadata for from_hf()
                if isinstance(value, DTensor):
                    self.grouped_expert_weight_placements[
                        abstract_key
                    ] = value.placements
                    self.grouped_expert_weight_shape[abstract_key] = value.shape
                    self.grouped_expert_weight_mesh[abstract_key] = value.device_mesh

                    # Split GroupedExperts weight to local individual expert weights
                    local_expert_fqn = self._get_local_experts_weights(
                        new_abstract_key,
                        abstract_key,
                        layer_num,
                        value,
                    )
                    hf_state_dict.update(local_expert_fqn)

                else:
                    # keep this path for offline conversion
                    moe_layer = next(
                        l
                        for l in self.model_config.layers  # pyrefly: ignore [missing-attribute]
                        if l.moe is not None
                    )
                    split_values = self._split_experts_weights(
                        value,
                        moe_layer.moe.num_experts,
                    )

                    for expert_num in range(0, moe_layer.moe.num_experts):
                        new_key = new_abstract_key.format(layer_num, expert_num)
                        hf_state_dict[new_key] = split_values[expert_num].squeeze()

            elif "layers" in key:
                new_key, layer_num = self._map_to_hf_layer_key(key, to_hf_map)
                new_key = new_key.format(layer_num)
                hf_state_dict[new_key] = value

            else:
                new_key = to_hf_map[key]
                hf_state_dict[new_key] = value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """
        1. When loading from HF checkpoint, dequantize the weights from float8 to float32.
        2. Convert between the HF shape and the torchtitan shape.
        3. Concat separate expert's weight into GroupedExperts' weight.
        """
        self._validate_hf_rope_config(ComplexRoPE.Config)

        state_dict = {}
        expert_weights_by_layer = {}  # {layer: {abstract_key: {expert_id: tensor}}}

        for key, value in hf_state_dict.items():
            if "mlp.experts" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=2)
                layer_num, expert_num = re.findall(r"\d+", key)
                titan_abstract_key, mapped_layer_num = self._map_from_hf_layer_key(
                    abstract_key,
                    layer_num,
                )
                if mapped_layer_num != layer_num:
                    layer_num = mapped_layer_num
                new_key = titan_abstract_key.format(layer_num)

                # Store the expert's weight in expert_weights_by_layer for concatenating later.
                if layer_num not in expert_weights_by_layer:
                    expert_weights_by_layer[layer_num] = {}
                if titan_abstract_key not in expert_weights_by_layer[layer_num]:
                    expert_weights_by_layer[layer_num][titan_abstract_key] = {}
                expert_weights_by_layer[layer_num][titan_abstract_key][
                    int(expert_num)
                ] = value

                # Use stored metadata to decide path (online vs offline)
                # Online mode: local_experts_indices was populated during to_hf()
                if titan_abstract_key in self.local_experts_indices:
                    stacked_value = self._concatenate_expert_weights_dtensor(
                        expert_weights_by_layer,
                        titan_abstract_key,
                        layer_num,
                    )
                else:  # keep this path to be compatible with offline conversion
                    stacked_value = self._concatenate_expert_weights(
                        expert_weights_by_layer,
                        titan_abstract_key,
                        layer_num,
                        next(
                            l
                            for l in self.model_config.layers  # pyrefly: ignore [missing-attribute]
                            if l.moe is not None
                        ).moe.num_experts,
                    )

                if stacked_value is not None:
                    state_dict[new_key] = stacked_value

            elif "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)
                new_key, layer_num = self._map_from_hf_layer_key(
                    abstract_key,
                    layer_num,
                )
                new_key = new_key.format(layer_num)
                state_dict[new_key] = value

            else:
                new_key = self.from_hf_map[key]
                state_dict[new_key] = value

        return state_dict

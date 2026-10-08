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
from torch.distributed.tensor import DTensor

from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.models.utils import MoEStateDictAdapter

if TYPE_CHECKING:
    from .model import DeepSeekV3Model


class DeepSeekV3StateDictAdapter(MoEStateDictAdapter):
    """
    StateDictAdapter for DeepSeekV3 model.
    """

    hf_experts_key_fragment = "mlp.experts"

    # HF checkpoints store a copy of the shared embedding and output head under
    # each MTP layer. TorchTitan has one tensor for each, saved under both names.
    _MTP_SHARED_COPIES: dict[str, str] = {
        "model.layers.{}.embed_tokens.weight": "tok_embeddings.weight",
        "model.layers.{}.shared_head.head.weight": "lm_head.weight",
    }

    def __init__(
        self,
        model_config: DeepSeekV3Model.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)
        # fqn_to_index_mapping lists the tensors an HF save writes. hf_assets_path
        # may hold the released FP8 index, whose X.weight_scale_inv tensors are not
        # saved, and whose MTP layer (61) is not saved by a config without MTP.
        if self.fqn_to_index_mapping is not None:
            num_hf_layers = len(model_config.layers) + len(
                getattr(model_config, "mtp_layers", None) or []
            )
            fqn_to_index_for_save = {}
            for key, index in self.fqn_to_index_mapping.items():
                is_fp8_scale = key.endswith(".weight_scale_inv")
                layer_match = re.match(r"model\.layers\.(\d+)\.", key)
                is_layer_not_in_config = (
                    layer_match is not None
                    and int(layer_match.group(1)) >= num_hf_layers
                )
                if is_fp8_scale or is_layer_not_in_config:
                    continue
                fqn_to_index_for_save[key] = index
            self.fqn_to_index_mapping = fqn_to_index_for_save
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
            "model.layers.{}.mlp.experts.{}.gate_proj.weight": "layers.{}.moe.routed_experts.w1_EFD",
            "model.layers.{}.mlp.experts.{}.up_proj.weight": "layers.{}.moe.routed_experts.w3_EFD",
            "model.layers.{}.mlp.experts.{}.down_proj.weight": "layers.{}.moe.routed_experts.w2.weight",
            "model.layers.{}.mlp.gate.weight": "layers.{}.moe.router.gate.weight",
            "model.layers.{}.mlp.shared_experts.gate_proj.weight": "layers.{}.moe.shared_experts.w1.weight",
            "model.layers.{}.mlp.shared_experts.up_proj.weight": "layers.{}.moe.shared_experts.w3.weight",
            "model.layers.{}.mlp.shared_experts.down_proj.weight": "layers.{}.moe.shared_experts.w2.weight",
            "model.layers.{}.mlp.gate.e_score_correction_bias": "layers.{}.moe.expert_bias_E",
            # MTP Module
            "model.layers.{}.enorm.weight": "layers.{}.enorm.weight",
            "model.layers.{}.hnorm.weight": "layers.{}.hnorm.weight",
            "model.layers.{}.eh_proj.weight": "layers.{}.eh_proj.weight",
            "model.layers.{}.shared_head.norm.weight": "layers.{}.mtp_norm.weight",
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
        if getattr(self.model_config, "mtp_layers", None):
            # pyrefly: ignore [missing-attribute]
            num_main_layers = len(self.model_config.layers)
            layer_idx = int(layer_num)
            if layer_idx >= num_main_layers:
                if not any(
                    new_key.startswith(f"layers.{{}}.{name}.")
                    for name in ("enorm", "hnorm", "eh_proj", "mtp_norm")
                ):
                    new_key = new_key.replace(
                        "layers.{}.",
                        "mtp_layers.{}.",
                        1,
                    )
                else:
                    new_key = new_key.replace("layers.{}.", "mtp_layers.{}.", 1)
                layer_num = str(layer_idx - num_main_layers)
        return new_key, layer_num

    def _map_to_hf_layer_key(
        self,
        key: str,
        to_hf_map: dict[str, str],
    ) -> tuple[str, str]:
        if key.startswith("mtp_layers."):
            abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
            # pyrefly: ignore [missing-attribute]
            layer_num = re.search(r"\d+", key).group(0)
            main_abstract_key = abstract_key.replace(
                "mtp_layers.{}.",
                "layers.{}.",
                1,
            ).replace("mtp_layers.{}.", "layers.{}.", 1)
            # pyrefly: ignore [missing-attribute]
            hf_layer_num = str(len(self.model_config.layers) + int(layer_num))
            return to_hf_map[main_abstract_key], hf_layer_num

        abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
        # pyrefly: ignore [missing-attribute]
        layer_num = re.search(r"\d+", key).group(0)
        return to_hf_map[abstract_key], layer_num

    def get_hf_storage_reader(
        self,
        path: str,
        from_quantized: bool = False,
        *,
        thread_count: int | None = None,
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
                thread_count=4 if thread_count is None else thread_count,
            )
        return super().get_hf_storage_reader(path, thread_count=thread_count)

    def to_hf(
        self, state_dict: dict[str, Any], quantized: bool = False
    ) -> dict[str, Any]:
        """
        1. Convert between the HF shape and the torchtitan shape.
        2. Split grouped-linear weights into individual expert weights.
        """
        state_dict = self._native_fused_linears_to_hf(
            state_dict,
            split_routed_experts=True,
        )

        to_hf_map = {v: k for k, v in self.from_hf_map.items()}

        hf_state_dict = {}

        for key, value in state_dict.items():
            if self._is_expert_weight_key(key):
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                new_abstract_key, layer_num = self._map_to_hf_layer_key(key, to_hf_map)

                # Store grouped-weight metadata for from_hf().
                if isinstance(value, DTensor):
                    self.grouped_expert_weight_placements[
                        abstract_key
                    ] = value.placements
                    self.grouped_expert_weight_shape[abstract_key] = value.shape
                    self.grouped_expert_weight_mesh[abstract_key] = value.device_mesh

                    # Split the grouped weight into local individual experts.
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

        # pyrefly: ignore [missing-attribute]
        num_main_layers = len(self.model_config.layers)
        num_mtp_layers = len(getattr(self.model_config, "mtp_layers", None) or [])
        for mtp_layer in range(num_mtp_layers):
            for hf_abstract_key, tt_key in self._MTP_SHARED_COPIES.items():
                if tt_key in state_dict:
                    hf_key = hf_abstract_key.format(num_main_layers + mtp_layer)
                    hf_state_dict[hf_key] = state_dict[tt_key]

        return hf_state_dict

    def from_hf(
        self, hf_state_dict: dict[str, Any], quantized: bool = False
    ) -> dict[str, Any]:
        """
        1. When loading from HF checkpoint, dequantize the weights from float8 to float32.
        2. Convert between the HF shape and the torchtitan shape.
        3. Concatenate individual expert weights into grouped-linear weights.
        """
        self._validate_hf_rope_config(ComplexRoPE.Config)

        state_dict = {}
        expert_weights_by_layer = {}  # {layer: {abstract_key: {expert_id: tensor}}}
        # pyrefly: ignore [missing-attribute]
        num_main_layers = len(self.model_config.layers)
        num_mtp_layers = len(getattr(self.model_config, "mtp_layers", None) or [])
        mtp_shared_copy_keys = {
            hf_abstract_key.format(num_main_layers + mtp_layer)
            for mtp_layer in range(num_mtp_layers)
            for hf_abstract_key in self._MTP_SHARED_COPIES
        }

        for key, value in hf_state_dict.items():
            if key in mtp_shared_copy_keys:
                # HF checkpoints store a copy of the shared embedding and output
                # head under each MTP layer, and this key is one of those copies.
                # TorchTitan keeps only one tok_embeddings and one lm_head, which
                # are set from model.embed_tokens.weight and lm_head.weight below.
                # During an HF load, to_hf pointed this key and the main key at the
                # same tensor, so DCP has already filled it and we can skip the copy.
                continue
            if self.hf_experts_key_fragment in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=2)
                layer_num, expert_num = re.findall(r"\d+", key)[:2]
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

        return self._native_fused_linears_from_hf(
            state_dict,
            fuse_routed_experts=True,
        )

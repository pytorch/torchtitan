# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HuggingFace checkpoint adapter for unquantized Kimi K3 weights."""

import re
from typing import Any

import torch
from torch.distributed.tensor import DTensor

from torchtitan.models.utils import MoEStateDictAdapter

from .model import KimiK3Model


_UNUSED_HF_LAYER_ZERO_ATTN_RES_KEYS = {
    "language_model.model.layers.0.self_attention_res_norm.weight",
    "language_model.model.layers.0.self_attention_res_proj.weight",
}


class KimiK3StateDictAdapter(MoEStateDictAdapter):
    def __init__(
        self,
        model_config: KimiK3Model.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)
        self.kimi_config = model_config

        self.from_hf_map = {
            # Language model.
            "language_model.model.embed_tokens.weight": "tok_embeddings.weight",
            "language_model.model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "language_model.model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            "language_model.model.layers.{}.self_attention_res_norm.weight": "layers.{}.attention_res_norm.weight",
            "language_model.model.layers.{}.self_attention_res_proj.weight": "layers.{}.attention_res_proj.weight",
            "language_model.model.layers.{}.mlp_res_norm.weight": "layers.{}.ffn_res_norm.weight",
            "language_model.model.layers.{}.mlp_res_proj.weight": "layers.{}.ffn_res_proj.weight",
            "language_model.model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "language_model.model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "language_model.model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            # MoE.
            "language_model.model.layers.{}.block_sparse_moe.experts.{}.w1.weight": (
                "layers.{}.moe.routed_experts.inner_experts.w1_EFD"
            ),
            "language_model.model.layers.{}.block_sparse_moe.experts.{}.w2.weight": (
                "layers.{}.moe.routed_experts.inner_experts.w2_EDF"
            ),
            "language_model.model.layers.{}.block_sparse_moe.experts.{}.w3.weight": (
                "layers.{}.moe.routed_experts.inner_experts.w3_EFD"
            ),
            "language_model.model.layers.{}.block_sparse_moe.gate.weight": "layers.{}.moe.router.gate.weight",
            "language_model.model.layers.{}.block_sparse_moe.gate.e_score_correction_bias": "layers.{}.moe.expert_bias_E",
            "language_model.model.layers.{}.block_sparse_moe.routed_expert_down_proj.weight": "layers.{}.moe.routed_down.weight",
            "language_model.model.layers.{}.block_sparse_moe.routed_expert_up_proj.weight": "layers.{}.moe.routed_up.weight",
            "language_model.model.layers.{}.block_sparse_moe.routed_expert_norm.weight": "layers.{}.moe.routed_norm.weight",
            "language_model.model.layers.{}.block_sparse_moe.shared_experts.gate_proj.weight": (
                "layers.{}.moe.shared_experts.w1.weight"
            ),
            "language_model.model.layers.{}.block_sparse_moe.shared_experts.up_proj.weight": (
                "layers.{}.moe.shared_experts.w3.weight"
            ),
            "language_model.model.layers.{}.block_sparse_moe.shared_experts.down_proj.weight": (
                "layers.{}.moe.shared_experts.w2.weight"
            ),
            "language_model.model.output_attn_res_norm.weight": "output_res_norm.weight",
            "language_model.model.output_attn_res_proj.weight": "output_res_proj.weight",
            "language_model.model.norm.weight": "norm.weight",
            "language_model.lm_head.weight": "lm_head.weight",
            # Vision encoder.
            "vision_tower.patch_embed.proj.weight": "vision_encoder.patch_embed.weight",
            "vision_tower.patch_embed.pos_emb.weight": "vision_encoder.pos_embed",
            "vision_tower.encoder.blocks.{}.norm0.weight": "vision_encoder.layers.{}.norm1.weight",
            "vision_tower.encoder.blocks.{}.norm1.weight": "vision_encoder.layers.{}.norm2.weight",
            "vision_tower.encoder.blocks.{}.wo.weight": "vision_encoder.layers.{}.attn.proj.weight",
            "vision_tower.encoder.blocks.{}.mlp.fc0.weight": "vision_encoder.layers.{}.mlp.linear_fc1.weight",
            "vision_tower.encoder.blocks.{}.mlp.fc1.weight": "vision_encoder.layers.{}.mlp.linear_fc2.weight",
            "vision_tower.encoder.final_layernorm.weight": "vision_encoder.final_norm.weight",
            "mm_projector.proj.0.weight": "vision_encoder.projector.linear_1.weight",
            "mm_projector.proj.2.weight": "vision_encoder.projector.linear_2.weight",
            "mm_projector.post_norm.weight": "vision_encoder.projector.post_norm.weight",
        }
        self.mla_from_hf_map = {
            "language_model.model.layers.{}.self_attn.q_a_proj.weight": "layers.{}.attention.wq_a.weight",
            "language_model.model.layers.{}.self_attn.q_a_layernorm.weight": "layers.{}.attention.q_norm.weight",
            "language_model.model.layers.{}.self_attn.q_b_proj.weight": "layers.{}.attention.wq_b.weight",
            "language_model.model.layers.{}.self_attn.kv_a_proj_with_mqa.weight": "layers.{}.attention.wkv_a.weight",
            "language_model.model.layers.{}.self_attn.kv_a_layernorm.weight": "layers.{}.attention.kv_norm.weight",
            "language_model.model.layers.{}.self_attn.kv_b_proj.weight": "layers.{}.attention.wkv_b.weight",
            "language_model.model.layers.{}.self_attn.g_proj.weight": "layers.{}.attention.gate.weight",
            "language_model.model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
        }
        self.kda_from_hf_map = {
            "language_model.model.layers.{}.self_attn.q_proj.weight": "layers.{}.delta_attention.q_proj.weight",
            "language_model.model.layers.{}.self_attn.k_proj.weight": "layers.{}.delta_attention.k_proj.weight",
            "language_model.model.layers.{}.self_attn.v_proj.weight": "layers.{}.delta_attention.v_proj.weight",
            "language_model.model.layers.{}.self_attn.q_conv1d.weight": "layers.{}.delta_attention.q_conv.weight",
            "language_model.model.layers.{}.self_attn.k_conv1d.weight": "layers.{}.delta_attention.k_conv.weight",
            "language_model.model.layers.{}.self_attn.v_conv1d.weight": "layers.{}.delta_attention.v_conv.weight",
            "language_model.model.layers.{}.self_attn.f_a_proj.weight": "layers.{}.delta_attention.forget_a.weight",
            "language_model.model.layers.{}.self_attn.f_b_proj.weight": "layers.{}.delta_attention.forget_b.weight",
            "language_model.model.layers.{}.self_attn.b_proj.weight": "layers.{}.delta_attention.beta.weight",
            "language_model.model.layers.{}.self_attn.g_proj.weight": "layers.{}.delta_attention.output_gate.weight",
            "language_model.model.layers.{}.self_attn.o_norm.weight": "layers.{}.delta_attention.output_norm.weight",
            "language_model.model.layers.{}.self_attn.o_proj.weight": "layers.{}.delta_attention.output_proj.weight",
            "language_model.model.layers.{}.self_attn.A_log": "layers.{}.delta_attention.A_log",
            "language_model.model.layers.{}.self_attn.dt_bias": "layers.{}.delta_attention.dt_bias",
        }

        # The released index contains MXFP4 packed/scale FQNs, while this
        # adapter exports unquantized weights.
        self.fqn_to_index_mapping = None

    def _map_from_hf_layer_key(
        self,
        abstract_key: str,
        layer_num: str,
    ) -> str | None:
        new_key = self.from_hf_map.get(abstract_key)
        if new_key is not None:
            return new_key

        layer_config = self.kimi_config.layers[int(layer_num)]
        attention_map = (
            self.mla_from_hf_map
            if layer_config.attention is not None
            else self.kda_from_hf_map
        )
        return attention_map.get(abstract_key)

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert a TorchTitan state dict to unquantized HuggingFace format."""
        to_hf_map = {
            tt_key: hf_key
            for mapping in (
                self.from_hf_map,
                self.mla_from_hf_map,
                self.kda_from_hf_map,
            )
            for hf_key, tt_key in mapping.items()
        }
        hf_state_dict: dict[str, Any] = {}
        vision_qkv_by_layer: dict[str, dict[str, torch.Tensor]] = {}
        unmapped: list[str] = []

        for key, value in state_dict.items():
            if "moe.routed_experts.inner_experts" in key:
                abstract_key = re.sub(r"(?<=\.)\d+(?=\.)", "{}", key, count=1)
                layer_num_match = re.search(r"layers\.(\d+)\.", key)
                assert layer_num_match is not None
                layer_num = layer_num_match.group(1)
                hf_abstract_key = to_hf_map.get(abstract_key)
                if hf_abstract_key is None:
                    unmapped.append(key)
                    continue

                if isinstance(value, DTensor):
                    self.grouped_expert_weight_placements[
                        abstract_key
                    ] = value.placements
                    self.grouped_expert_weight_shape[abstract_key] = value.shape
                    self.grouped_expert_weight_mesh[abstract_key] = value.device_mesh
                    hf_state_dict.update(
                        self._get_local_experts_weights(
                            hf_abstract_key,
                            abstract_key,
                            layer_num,
                            value,
                        )
                    )
                else:
                    moe_config = self.kimi_config.layers[int(layer_num)].moe
                    assert moe_config is not None
                    split_values = self._split_experts_weights(
                        value,
                        moe_config.num_experts,
                    )
                    for expert_num, expert_weight in enumerate(split_values):
                        hf_state_dict[
                            hf_abstract_key.format(layer_num, expert_num)
                        ] = expert_weight.squeeze(0)
                continue

            vision_qkv_match = re.fullmatch(
                r"vision_encoder\.layers\.(\d+)\.attn\.w(q|k|v)\.weight",
                key,
            )
            if vision_qkv_match is not None:
                layer_num, projection = vision_qkv_match.groups()
                vision_qkv_by_layer.setdefault(layer_num, {})[projection] = value
                continue

            layer_num_match = re.search(r"(?<=\.)\d+(?=\.)", key)
            if layer_num_match is not None:
                layer_num = layer_num_match.group(0)
                abstract_key = re.sub(
                    r"(?<=\.)\d+(?=\.)",
                    "{}",
                    key,
                    count=1,
                )
                hf_abstract_key = to_hf_map.get(abstract_key)
                if hf_abstract_key is None:
                    unmapped.append(key)
                    continue
                if abstract_key == "layers.{}.delta_attention.dt_bias":
                    value = value.reshape(-1)
                hf_state_dict[hf_abstract_key.format(layer_num)] = value
                continue

            hf_key = to_hf_map.get(key)
            if hf_key is None:
                unmapped.append(key)
                continue
            if key == "vision_encoder.patch_embed.weight":
                vision_config = self.kimi_config.vision_encoder
                if vision_config is None:
                    raise ValueError(
                        "Vision state was provided for a text-only config."
                    )
                value = value.reshape(
                    value.shape[0],
                    vision_config.in_channels,
                    vision_config.patch_size,
                    vision_config.patch_size,
                )
            hf_state_dict[hf_key] = value

        for layer_num, qkv in vision_qkv_by_layer.items():
            missing = {"q", "k", "v"} - qkv.keys()
            if missing:
                raise ValueError(
                    f"Vision layer {layer_num} is missing QKV parts: {sorted(missing)}."
                )
            hf_state_dict[
                f"vision_tower.encoder.blocks.{layer_num}.wqkv.weight"
            ] = torch.cat((qkv["q"], qkv["k"], qkv["v"]), dim=0)

        # The released HF model contain these unused layer-0 attn res parameters.
        # TT omits them, so synthesize deterministic, placeholders to preserve strict HF state-dict loading.
        if self.kimi_config.layers[0].attention_res_norm is None:
            norm_template_key = (
                "language_model.model.layers.1.self_attention_res_norm.weight"
            )
            proj_template_key = (
                "language_model.model.layers.1.self_attention_res_proj.weight"
            )
            hf_state_dict[
                "language_model.model.layers.0.self_attention_res_norm.weight"
            ] = torch.ones_like(hf_state_dict[norm_template_key])
            hf_state_dict[
                "language_model.model.layers.0.self_attention_res_proj.weight"
            ] = torch.zeros_like(hf_state_dict[proj_template_key])

        if unmapped:
            raise ValueError(
                "KimiK3StateDictAdapter found TorchTitan keys without a "
                f"mapping: {unmapped}."
            )
        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert an unquantized HuggingFace state dict to TorchTitan."""
        state_dict: dict[str, Any] = {}
        expert_weights_by_layer: dict[str, dict[str, dict[int, torch.Tensor]]] = {}
        unmapped: list[str] = []

        for key, value in hf_state_dict.items():
            if key in _UNUSED_HF_LAYER_ZERO_ATTN_RES_KEYS:
                continue
            if key.endswith("rotary_emb.inv_freq"):
                continue

            new_key = self.from_hf_map.get(key)
            if new_key is not None:
                if key == "vision_tower.patch_embed.proj.weight":
                    value = value.reshape(value.shape[0], -1)
                state_dict[new_key] = value
                continue

            if "block_sparse_moe.experts" in key:
                abstract_key = re.sub(
                    r"(?<=\.)\d+(?=\.)",
                    "{}",
                    key,
                    count=2,
                )
                indices = re.findall(r"(?<=\.)\d+(?=\.)", key)
                if len(indices) != 2:
                    unmapped.append(key)
                    continue
                layer_num, expert_num = indices
                titan_abstract_key = self.from_hf_map.get(abstract_key)
                if titan_abstract_key is None:
                    unmapped.append(key)
                    continue
                new_key = titan_abstract_key.format(layer_num)

                experts = expert_weights_by_layer.setdefault(layer_num, {}).setdefault(
                    titan_abstract_key, {}
                )
                experts[int(expert_num)] = value

                if titan_abstract_key in self.local_experts_indices:
                    stacked_value = self._concatenate_expert_weights_dtensor(
                        expert_weights_by_layer,
                        titan_abstract_key,
                        layer_num,
                    )
                else:
                    moe_config = self.kimi_config.layers[int(layer_num)].moe
                    assert moe_config is not None
                    stacked_value = self._concatenate_expert_weights(
                        expert_weights_by_layer,
                        titan_abstract_key,
                        layer_num,
                        moe_config.num_experts,
                    )
                if stacked_value is not None:
                    state_dict[new_key] = stacked_value
                continue

            layer_num_match = re.search(r"(?<=\.)\d+(?=\.)", key)
            if layer_num_match is not None:
                layer_num = layer_num_match.group(0)
                abstract_key = re.sub(
                    r"(?<=\.)\d+(?=\.)",
                    "{}",
                    key,
                    count=1,
                )

                if abstract_key == "vision_tower.encoder.blocks.{}.wqkv.weight":
                    q, k, v = torch.chunk(value, 3, dim=0)
                    base = f"vision_encoder.layers.{layer_num}.attn"
                    state_dict[f"{base}.wq.weight"] = q
                    state_dict[f"{base}.wk.weight"] = k
                    state_dict[f"{base}.wv.weight"] = v
                    continue

                new_abstract_key = (
                    self._map_from_hf_layer_key(abstract_key, layer_num)
                    if key.startswith("language_model.model.layers.")
                    else self.from_hf_map.get(abstract_key)
                )
                if new_abstract_key is None:
                    unmapped.append(key)
                    continue
                if new_abstract_key == "layers.{}.delta_attention.dt_bias":
                    delta_config = self.kimi_config.layers[
                        int(layer_num)
                    ].delta_attention
                    if delta_config is None:
                        raise ValueError(f"HF key '{key}' targets a non-KDA layer.")
                    value = value.reshape(
                        delta_config.num_heads,
                        delta_config.head_dim,
                    )
                state_dict[new_abstract_key.format(layer_num)] = value
                continue

            unmapped.append(key)

        if unmapped:
            raise ValueError(
                "KimiK3StateDictAdapter found HuggingFace keys without a "
                f"mapping: {unmapped}."
            )
        if expert_weights_by_layer:
            raise ValueError(
                "KimiK3StateDictAdapter received an incomplete set of "
                f"routed-expert weights: {expert_weights_by_layer.keys()}."
            )
        return state_dict

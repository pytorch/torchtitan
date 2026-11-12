# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

"""HuggingFace checkpoint adapter for Kimi K3 weights."""

import re
from typing import Any, TYPE_CHECKING

import torch
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Placement, Replicate

from torchtitan.models.common.attention.kda import KDA
from torchtitan.models.common.moe.state_dict_adapter import MoEStateDictAdapter

from .moe import KimiLatentMoE

if TYPE_CHECKING:
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
                "layers.{}.feed_forward.routed_experts.w1_EFD"
            ),
            "language_model.model.layers.{}.block_sparse_moe.experts.{}.w2.weight": (
                "layers.{}.feed_forward.routed_experts.w2.weight"
            ),
            "language_model.model.layers.{}.block_sparse_moe.experts.{}.w3.weight": (
                "layers.{}.feed_forward.routed_experts.w3_EFD"
            ),
            "language_model.model.layers.{}.block_sparse_moe.gate.weight": "layers.{}.feed_forward.router.gate.weight",
            "language_model.model.layers.{}.block_sparse_moe.gate.e_score_correction_bias": "layers.{}.feed_forward.expert_bias_E",
            "language_model.model.layers.{}.block_sparse_moe.routed_expert_down_proj.weight": (
                "layers.{}.feed_forward.routed_down.weight"
            ),
            "language_model.model.layers.{}.block_sparse_moe.routed_expert_up_proj.weight": (
                "layers.{}.feed_forward.routed_up.weight"
            ),
            "language_model.model.layers.{}.block_sparse_moe.routed_expert_norm.weight": (
                "layers.{}.feed_forward.routed_norm.weight"
            ),
            "language_model.model.layers.{}.block_sparse_moe.shared_experts.gate_proj.weight": (
                "layers.{}.feed_forward.shared_experts.w1.weight"
            ),
            "language_model.model.layers.{}.block_sparse_moe.shared_experts.up_proj.weight": (
                "layers.{}.feed_forward.shared_experts.w3.weight"
            ),
            "language_model.model.layers.{}.block_sparse_moe.shared_experts.down_proj.weight": (
                "layers.{}.feed_forward.shared_experts.w2.weight"
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
            "language_model.model.layers.{}.self_attn.kv_b_proj.weight": "layers.{}.attention.inner_attention.wkv_b.weight",
            "language_model.model.layers.{}.self_attn.g_proj.weight": "layers.{}.attention.gate.weight",
            "language_model.model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
        }
        self.kda_from_hf_map = {
            "language_model.model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.q_proj.weight",
            "language_model.model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.k_proj.weight",
            "language_model.model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.v_proj.weight",
            "language_model.model.layers.{}.self_attn.q_conv1d.weight": "layers.{}.attention.q_conv.weight",
            "language_model.model.layers.{}.self_attn.k_conv1d.weight": "layers.{}.attention.k_conv.weight",
            "language_model.model.layers.{}.self_attn.v_conv1d.weight": "layers.{}.attention.v_conv.weight",
            "language_model.model.layers.{}.self_attn.f_a_proj.weight": "layers.{}.attention.forget_a.weight",
            "language_model.model.layers.{}.self_attn.f_b_proj.weight": "layers.{}.attention.forget_b.weight",
            "language_model.model.layers.{}.self_attn.b_proj.weight": "layers.{}.attention.beta.weight",
            "language_model.model.layers.{}.self_attn.g_proj.weight": "layers.{}.attention.output_gate.weight",
            "language_model.model.layers.{}.self_attn.o_norm.weight": "layers.{}.attention.output_norm.weight",
            "language_model.model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.output_proj.weight",
            "language_model.model.layers.{}.self_attn.A_log": "layers.{}.attention.A_log",
            "language_model.model.layers.{}.self_attn.dt_bias": "layers.{}.attention.dt_bias",
        }

        # Loading MXFP4 dequantizes routed experts to BF16; HF saves write
        # unquantized weights rather than re-quantizing them. Remap the released
        # index from X.weight_packed to X.weight so saves retain the original
        # file assignments for the unquantized tensors.
        if self.fqn_to_index_mapping is not None:
            self.fqn_to_index_mapping = {
                key.removesuffix("_packed"): index
                for key, index in self.fqn_to_index_mapping.items()
            }
        # Restore KDA DTensor layouts after converting the checkpoint's padded
        # A_log and flattened dt_bias back to Titan's parameter shapes.
        self._kda_parameter_layouts: dict[
            str,
            tuple[DeviceMesh, tuple[Placement, ...]],
        ] = {}

    def get_hf_storage_reader(
        self,
        path: str,
        from_quantized: bool = False,
        *,
        thread_count: int | None = None,
    ) -> HuggingFaceStorageReader:
        if from_quantized:
            from torch.distributed.checkpoint.quantized_hf_storage import (
                QuantizedHuggingFaceStorageReader,
            )

            return QuantizedHuggingFaceStorageReader(
                path=path,
                thread_count=4 if thread_count is None else thread_count,
                target_dtype=torch.bfloat16,
            )
        return super().get_hf_storage_reader(path, thread_count=thread_count)

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
            self.kda_from_hf_map
            if isinstance(layer_config.attention, KDA.Config)
            else self.mla_from_hf_map
        )
        return attention_map.get(abstract_key)

    def to_hf(
        self, state_dict: dict[str, Any], *, to_quantized_layout: bool = False
    ) -> dict[str, Any]:
        """Convert a TorchTitan state dict to HuggingFace format.

        With ``to_quantized_layout``, routed experts use the compressed-tensors MXFP4
        names (``weight_packed``) and A_log is padded to match the released
        checkpoint's layout for load planning. Ordinary HF exports leave A_log
        unpadded.
        """
        state_dict = self._native_fused_linears_to_hf(
            state_dict,
            split_routed_experts=True,
        )
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
            if self._is_expert_weight_key(key):
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
                    moe_config = self.kimi_config.layers[int(layer_num)].feed_forward
                    assert isinstance(moe_config, KimiLatentMoE.Config)
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
                if abstract_key in {
                    "layers.{}.attention.A_log",
                    "layers.{}.attention.dt_bias",
                }:
                    if isinstance(value, DTensor):
                        # Padding and flattening change tensor coordinates.
                        # Replicate before matching HF shapes, then restore the
                        # original mesh and placements in from_hf().
                        self._kda_parameter_layouts[key] = (
                            value.device_mesh,
                            value.placements,
                        )
                        value = value.redistribute(
                            placements=(Replicate(),) * value.device_mesh.ndim
                        )
                    if abstract_key == "layers.{}.attention.A_log":
                        delta_config = self.kimi_config.layers[int(layer_num)].attention
                        if not isinstance(delta_config, KDA.Config):
                            raise ValueError(f"TT key '{key}' targets a non-KDA layer.")
                        if to_quantized_layout:
                            # The released MXFP4 checkpoint pads A_log to
                            # head_dim. Match that shape for DCP planning;
                            # from_hf() removes the padding after loading.
                            value = torch.nn.functional.pad(
                                value,
                                (0, delta_config.head_dim - delta_config.n_heads),
                            )
                    else:
                        # HF stores dt_bias flat; Titan uses
                        # [n_heads, head_dim], restored in from_hf().
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

        # The released HF checkpoint includes unused layer-0 attention residual
        # weights that Titan omits. Create placeholders only on the PP stage
        # that owns layer 0, using its local feed-forward residual shapes.
        if (
            self.kimi_config.layers[0].attention_res_norm is None
            and "layers.0.ffn_res_norm.weight" in state_dict
        ):
            hf_state_dict[
                "language_model.model.layers.0.self_attention_res_norm.weight"
            ] = torch.ones_like(state_dict["layers.0.ffn_res_norm.weight"])
            hf_state_dict[
                "language_model.model.layers.0.self_attention_res_proj.weight"
            ] = torch.zeros_like(state_dict["layers.0.ffn_res_proj.weight"])

        if unmapped:
            raise ValueError(
                "KimiK3StateDictAdapter found TorchTitan keys without a "
                f"mapping: {unmapped}."
            )
        if to_quantized_layout:
            hf_state_dict = {
                (
                    key.removesuffix(".weight") + ".weight_packed"
                    if ".block_sparse_moe.experts." in key
                    else key
                ): value
                for key, value in hf_state_dict.items()
            }
        return hf_state_dict

    def from_hf(
        self, hf_state_dict: dict[str, Any], *, is_quantized_layout: bool = False
    ) -> dict[str, Any]:
        """Convert a HuggingFace state dict to TorchTitan.

        With ``is_quantized_layout``, routed experts must use the ``weight_packed``
        names from ``to_hf(..., to_quantized_layout=True)``.
        """
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
                if is_quantized_layout:
                    if not key.endswith(".weight_packed"):
                        unmapped.append(key)
                        continue
                    key = key.removesuffix("_packed")
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
                    moe_config = self.kimi_config.layers[int(layer_num)].feed_forward
                    assert isinstance(moe_config, KimiLatentMoE.Config)
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
                if new_abstract_key in {
                    "layers.{}.attention.A_log",
                    "layers.{}.attention.dt_bias",
                }:
                    delta_config = self.kimi_config.layers[int(layer_num)].attention
                    if not isinstance(delta_config, KDA.Config):
                        raise ValueError(f"HF key '{key}' targets a non-KDA layer.")
                    if new_abstract_key == "layers.{}.attention.A_log":
                        value = value[: delta_config.n_heads]
                    else:
                        value = value.reshape(
                            delta_config.n_heads,
                            delta_config.head_dim,
                        )
                    layout = self._kda_parameter_layouts.get(
                        new_abstract_key.format(layer_num)
                    )
                    if layout is not None:
                        assert isinstance(value, DTensor)
                        value = value.redistribute(
                            device_mesh=layout[0],
                            placements=layout[1],
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
        return self._native_fused_linears_from_hf(
            state_dict,
            fuse_routed_experts=True,
        )

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

"""HuggingFace checkpoint adapter for dense and packed Kimi K3 weights."""

import json
import re
from pathlib import Path
from typing import Any, cast, TYPE_CHECKING

import torch
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.tensor import DTensor, Replicate, Shard

from torchtitan.components.checkpointer.hf_storage import (
    HuggingFaceStorageReaderWithViews,
    LogicalPrefixSpec,
    PackedPairSpec,
)
from torchtitan.models.common.feed_forward import FeedForward
from torchtitan.models.common.linear import GroupedLinear, Linear, RouterGateLinear
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.utils import MoEStateDictAdapter
from torchtitan.protocols.state_dict_adapter import dtensor_safe
from torchtitan.quantization.mx_qat.checkpoint import (
    decode_mxfp4,
    MXFP4CheckpointPolicy,
)

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
                "layers.{}.moe.routed_experts.w1_EFD"
            ),
            "language_model.model.layers.{}.block_sparse_moe.experts.{}.w2.weight": (
                "layers.{}.moe.routed_experts.w2.weight"
            ),
            "language_model.model.layers.{}.block_sparse_moe.experts.{}.w3.weight": (
                "layers.{}.moe.routed_experts.w3_EFD"
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

    def hf_linear_weight_mapping(self) -> dict[str, str | None]:
        """Map the supported HF Linear hierarchy to Titan parameter FQNs.

        Reuse the adapter's architecture mapping. RouterGateLinear represents
        HF KimiMoEGate's raw parameter, not an HF Linear. Grouped expert tensors
        represent one HF Linear per expert; no quantization-name regex is used.
        """
        linear_weights = {
            f"{fqn}.weight": f"{fqn}.weight"
            for fqn, config, _, _ in self.kimi_config.traverse(Linear.Config)
            if not isinstance(config, RouterGateLinear.Config)
        }
        # MoonViT builds repeated layers from one block config and renames MLP
        # fields. Inspect one meta block to recover its actual parameter names.
        vision = self.kimi_config.vision_encoder
        if vision is not None:
            with torch.device("meta"):
                block = vision.block.build()
            linear_weights.update(
                {
                    f"vision_encoder.layers.{layer}.{name}.weight": f"vision_encoder.layers.{layer}.{name}.weight"
                    for layer in range(vision.num_layers)
                    for name, module in block.named_modules()
                    if isinstance(module, Linear)
                }
            )
        # Match the logical W1/W3 keys emitted by the base adapter's native
        # fused-linear conversion to their single stored W13 parameter.
        for fqn, _, _, _ in self.kimi_config.traverse(FeedForward.Config):
            for projection in ("w1", "w3"):
                linear_weights[f"{fqn}.{projection}.weight"] = f"{fqn}.w13.weight"
        grouped_weights = {}
        for fqn, config, _, _ in self.kimi_config.traverse(RoutedExperts.Config):
            for projection in ("w1_EFD", "w3_EFD"):
                grouped_weights[f"{fqn}.{projection}"] = (
                    f"{fqn}.w13.weight",
                    config.w13.group_size,
                )
            grouped_weights[f"{fqn}.w2.weight"] = (
                f"{fqn}.w2.weight",
                config.w2.group_size,
            )
        result: dict[str, str | None] = {}
        for mapping in (self.from_hf_map, self.mla_from_hf_map, self.kda_from_hf_map):
            for hf_template, titan_template in mapping.items():
                layers = (
                    range(
                        vision.num_layers
                        if hf_template.startswith("vision_tower.")
                        and vision is not None
                        else len(self.kimi_config.layers)
                    )
                    if "{}" in titan_template
                    else (None,)
                )
                for layer in layers:
                    titan_key = titan_template.format(layer)
                    if titan_key in linear_weights:
                        result[hf_template.format(layer)] = linear_weights[titan_key]
                    elif titan_key in grouped_weights:
                        target, num_experts = grouped_weights[titan_key]
                        for expert in range(num_experts):
                            result[hf_template.format(layer, expert)] = target
        # HF has an unused layer-zero residual projection absent from Titan.
        if self.kimi_config.layers[0].attention_res_proj is None:
            result[
                "language_model.model.layers.0.self_attention_res_proj.weight"
            ] = None
        # The HF vision projection is fused; Titan stores its three slices.
        # Current Kimi recipes require vision to remain unquantized.
        if self.kimi_config.vision_encoder is not None:
            for layer in range(self.kimi_config.vision_encoder.num_layers):
                result[
                    f"vision_tower.encoder.blocks.{layer}.wqkv.weight"
                ] = f"vision_encoder.layers.{layer}.attn.wqkv.weight"
        return result

    def mxfp4_policy(self, path: str) -> MXFP4CheckpointPolicy:
        config_path = Path(path) / "config.json"
        if not config_path.is_file():
            raise ValueError(f"Quantized Kimi checkpoint is missing {config_path}.")
        config = json.loads(config_path.read_text())
        quantization = config.get("text_config", config).get("quantization_config")
        if not isinstance(quantization, dict):
            raise ValueError("Kimi checkpoint is missing quantization_config metadata.")
        index_path = Path(path) / "model.safetensors.index.json"
        if not index_path.is_file():
            raise ValueError(f"Quantized Kimi checkpoint is missing {index_path}.")
        index = json.loads(index_path.read_text())
        return MXFP4CheckpointPolicy.from_manifest(
            quantization, self.hf_linear_weight_mapping(), index.get("weight_map")
        )

    def qat_weight_fqns(self, policy: MXFP4CheckpointPolicy) -> set[str]:
        """Translate a manifest policy without silently quantizing BF16 experts.

        Several HF experts share one Titan parameter. QAT can only select that
        parameter when every corresponding HF weight is packed.
        """
        mapping = self.hf_linear_weight_mapping()
        selected = {
            target for key in policy.weight_fqns if (target := mapping[key]) is not None
        }
        partial = {
            target
            for key, target in mapping.items()
            if target in selected and key not in policy.weight_fqns
        }
        if partial:
            raise ValueError(
                "MX QAT cannot mix packed and BF16 experts within a shared parameter: "
                f"{sorted(partial)}"
            )
        return selected

    @staticmethod
    def _validate_qat_weight_config(config, policy: MXFP4CheckpointPolicy) -> None:
        if (
            config.dtype != torch.float4_e2m1fn_x2
            or config.block_size != policy.block_size
        ):
            raise ValueError("MX QAT weight format disagrees with checkpoint policy")

    def _validate_qat_policy(self, policy: MXFP4CheckpointPolicy) -> None:
        """Reject a QAT recipe whose selected parameters differ from import."""
        mapping = self.hf_linear_weight_mapping()
        selected = set()
        has_qat = False
        for fqn, config, _, _ in self.kimi_config.traverse(Linear.Config):
            if getattr(type(config)._owner, "_mx_qat", False):
                has_qat = True
                self._validate_qat_weight_config(
                    cast(Any, config).weight_fake_quant_config, policy
                )
                selected.add(f"{fqn}.weight")
        for fqn, config, _, _ in self.kimi_config.traverse(GroupedLinear.Config):
            if getattr(type(config)._owner, "_mx_qat", False):
                has_qat = True
                self._validate_qat_weight_config(
                    cast(Any, config).weight_fake_quant_config, policy
                )
                selected.update(
                    key
                    for key in mapping.values()
                    if key and key.rsplit(".", 1)[0] == fqn
                )
        if not has_qat:
            return  # Packed import into a BF16 model remains supported.
        expected = self.qat_weight_fqns(policy)
        if selected != expected:
            raise ValueError(
                "MX QAT selection disagrees with checkpoint policy: "
                f"missing={sorted(expected - selected)}, unexpected={sorted(selected - expected)}"
            )

    def _reshape_dt_bias(
        self, value: torch.Tensor, shape: tuple[int, ...]
    ) -> torch.Tensor:
        """Preserve leading-axis FSDP shards, including ranks with no heads."""
        if isinstance(value, DTensor) and any(
            not isinstance(p, Replicate) and not (type(p) is Shard and p.dim == 0)
            for p in value.placements
        ):
            raise ValueError("KDA dt_bias reshape supports only Replicate and Shard(0)")
        return self._reshape_dt_bias_replicated(value, shape)

    @dtensor_safe
    def _reshape_dt_bias_replicated(
        self, value: torch.Tensor, shape: tuple[int, ...]
    ) -> torch.Tensor:
        # Reuse the adapter's gather/restore helper only for this small bias.
        return value.reshape(shape)

    def get_hf_storage_reader(
        self,
        path: str,
        from_quantized: bool = False,
    ) -> HuggingFaceStorageReader:
        # The released 96-head KDA stores A_log in a 128-element vector.
        # Other architectures must not inherit this checkpoint-specific rule.
        prefixes = {
            f"language_model.model.layers.{index}.self_attn.A_log": LogicalPrefixSpec(
                logical_length=layer.delta_attention.num_heads,
                padded_length=128,
            )
            for index, layer in enumerate(self.kimi_config.layers)
            if layer.delta_attention is not None
            and layer.delta_attention.num_heads == 96
            and layer.delta_attention.head_dim == 128
        }
        if not from_quantized and not prefixes:
            return super().get_hf_storage_reader(path, from_quantized=False)
        spec = None
        if from_quantized:
            policy = self.mxfp4_policy(path)
            self._validate_qat_policy(policy)
            spec = PackedPairSpec(
                packed_suffix=".weight_packed",
                scale_suffix=".weight_scale",
                virtual_suffix=".weight",
                block_size=policy.block_size,
                packed_values_per_byte=2,
                target_dtype=torch.bfloat16,
                target_fqns=policy.weight_fqns,
                decode=decode_mxfp4,
            )
        return HuggingFaceStorageReaderWithViews(
            path=path,
            spec=spec,
            logical_prefixes=prefixes,
        )

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert a TorchTitan state dict to unquantized HuggingFace format."""
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
                    value = self._reshape_dt_bias(value, (-1,))
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
            # Match the base adapter's QKV handling: concatenate replicated
            # projections instead of letting DTensor choose column shards.
            # Copying those back to FSDP row shards requires an all-to-all.
            for projection, value in qkv.items():
                if isinstance(value, DTensor):
                    qkv[projection] = value.redistribute(
                        value.device_mesh, [Replicate()] * value.device_mesh.ndim
                    )
            hf_state_dict[
                f"vision_tower.encoder.blocks.{layer_num}.wqkv.weight"
            ] = torch.cat((qkv["q"], qkv["k"], qkv["v"]), dim=0)

        # HF retains unused layer-zero attention-residual parameters. Only
        # the stage owning layer zero emits them, using same-layer templates
        # so export also works when a pipeline stage contains just one layer.
        if self.kimi_config.layers[0].attention_res_norm is None:
            for suffix, fill in (("norm", torch.ones_like), ("proj", torch.zeros_like)):
                template = f"language_model.model.layers.0.mlp_res_{suffix}.weight"
                if template in hf_state_dict:
                    hf_state_dict[
                        f"language_model.model.layers.0.self_attention_res_{suffix}.weight"
                    ] = fill(hf_state_dict[template])

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
                    if isinstance(value, DTensor):
                        value = value.redistribute(
                            value.device_mesh, [Replicate()] * value.device_mesh.ndim
                        )
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
                    value = self._reshape_dt_bias(
                        value, (delta_config.num_heads, delta_config.head_dim)
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

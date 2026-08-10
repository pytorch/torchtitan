# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unquantized HuggingFace checkpoint adapter for Kimi K3.

The released Kimi K3 checkpoint uses MXFP4 expert weights.  That format is
intentionally outside the first implementation.  This adapter targets an
unquantized HuggingFace state dict, which is sufficient for constructing the
same reduced model on both sides of the numerical parity test.
"""

import re
from typing import Any

import torch

from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .model import KimiK3Model


_TEXT_GLOBAL_FROM_HF = {
    "language_model.model.embed_tokens.weight": "tok_embeddings.weight",
    "language_model.model.output_attn_res_norm.weight": ("output_res_norm.weight"),
    "language_model.model.output_attn_res_proj.weight": ("output_res_proj.weight"),
    "language_model.model.norm.weight": "norm.weight",
    "language_model.lm_head.weight": "lm_head.weight",
}

_TEXT_LAYER_FROM_HF = {
    # Layer norms and attention residuals.
    "input_layernorm.weight": "attention_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "self_attention_res_norm.weight": "attention_res_norm.weight",
    "self_attention_res_proj.weight": "attention_res_proj.weight",
    "mlp_res_norm.weight": "ffn_res_norm.weight",
    "mlp_res_proj.weight": "ffn_res_proj.weight",
    # Dense MLP.
    "mlp.gate_proj.weight": "feed_forward.w1.weight",
    "mlp.up_proj.weight": "feed_forward.w3.weight",
    "mlp.down_proj.weight": "feed_forward.w2.weight",
}

_MLA_FROM_HF = {
    "self_attn.q_a_proj.weight": "attention.wq_a.weight",
    "self_attn.q_a_layernorm.weight": "attention.q_norm.weight",
    "self_attn.q_b_proj.weight": "attention.wq_b.weight",
    "self_attn.kv_a_proj_with_mqa.weight": "attention.wkv_a.weight",
    "self_attn.kv_a_layernorm.weight": "attention.kv_norm.weight",
    "self_attn.kv_b_proj.weight": "attention.wkv_b.weight",
    "self_attn.g_proj.weight": "attention.gate.weight",
    "self_attn.o_proj.weight": "attention.wo.weight",
}

_KDA_FROM_HF = {
    "self_attn.q_proj.weight": "delta_attention.q_proj.weight",
    "self_attn.k_proj.weight": "delta_attention.k_proj.weight",
    "self_attn.v_proj.weight": "delta_attention.v_proj.weight",
    "self_attn.q_conv1d.weight": "delta_attention.q_conv.weight",
    "self_attn.k_conv1d.weight": "delta_attention.k_conv.weight",
    "self_attn.v_conv1d.weight": "delta_attention.v_conv.weight",
    "self_attn.f_a_proj.weight": "delta_attention.forget_a.weight",
    "self_attn.f_b_proj.weight": "delta_attention.forget_b.weight",
    "self_attn.b_proj.weight": "delta_attention.beta.weight",
    "self_attn.g_proj.weight": "delta_attention.output_gate.weight",
    "self_attn.o_norm.weight": "delta_attention.output_norm.weight",
    "self_attn.o_proj.weight": "delta_attention.output_proj.weight",
    "self_attn.A_log": "delta_attention.A_log",
    "self_attn.dt_bias": "delta_attention.dt_bias",
}

_MOE_FROM_HF = {
    "block_sparse_moe.gate.weight": "moe.router.gate.weight",
    "block_sparse_moe.gate.e_score_correction_bias": "moe.expert_bias_E",
    "block_sparse_moe.routed_expert_down_proj.weight": ("moe.routed_down.weight"),
    "block_sparse_moe.routed_expert_up_proj.weight": ("moe.routed_up.weight"),
    "block_sparse_moe.routed_expert_norm.weight": ("moe.routed_norm.weight"),
    "block_sparse_moe.shared_experts.gate_proj.weight": (
        "moe.shared_experts.w1.weight"
    ),
    "block_sparse_moe.shared_experts.up_proj.weight": ("moe.shared_experts.w3.weight"),
    "block_sparse_moe.shared_experts.down_proj.weight": (
        "moe.shared_experts.w2.weight"
    ),
}

# KimiGroupedExperts stacks all experts' weights into one (E, F, D) param per
# projection; HF stores them as separate per-expert 2D tensors.
_EXPERT_PROJECTION_TO_GROUPED_PARAM = {
    "w1": "w1_EFD",
    "w2": "w2_EDF",
    "w3": "w3_EFD",
}
_GROUPED_PARAM_TO_EXPERT_PROJECTION = {
    v: k for k, v in _EXPERT_PROJECTION_TO_GROUPED_PARAM.items()
}

_VISION_GLOBAL_FROM_HF = {
    "vision_tower.patch_embed.proj.weight": ("vision_encoder.patch_embed.weight"),
    "vision_tower.patch_embed.pos_emb.weight": "vision_encoder.pos_embed",
    "vision_tower.encoder.final_layernorm.weight": ("vision_encoder.final_norm.weight"),
    "mm_projector.proj.0.weight": ("vision_encoder.projector.linear_1.weight"),
    "mm_projector.proj.2.weight": ("vision_encoder.projector.linear_2.weight"),
    "mm_projector.post_norm.weight": ("vision_encoder.projector.post_norm.weight"),
}

_VISION_LAYER_FROM_HF = {
    "norm0.weight": "norm1.weight",
    "norm1.weight": "norm2.weight",
    "wo.weight": "attn.proj.weight",
    "mlp.fc0.weight": "mlp.linear_fc1.weight",
    "mlp.fc1.weight": "mlp.linear_fc2.weight",
}


class KimiK3StateDictAdapter(StateDictAdapter):
    """Convert between unquantized Kimi K3 HF and TorchTitan state dicts."""

    def __init__(
        self,
        model_config: KimiK3Model.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)
        self.kimi_config = model_config
        # {(layer_idx, projection): {expert_idx: 2D tensor}}, filled in from_hf
        # while HF's per-expert keys arrive one at a time; stacked into
        # KimiGroupedExperts' (E, F, D) parameter once all experts are seen.
        self._expert_weights_by_layer_projection: dict[
            tuple[str, str], dict[int, torch.Tensor]
        ] = {}

    @staticmethod
    def _raise_if_quantized_key(key: str) -> None:
        quantized_markers = (
            "weight_scale",
            "weight_packed",
            "compressed",
            "scale_shape",
        )
        if any(marker in key for marker in quantized_markers):
            raise NotImplementedError(
                "Kimi K3 v1 only supports unquantized HuggingFace state "
                f"dicts; encountered quantized key '{key}'."
            )

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert an unquantized HuggingFace state dict to TorchTitan."""
        state_dict: dict[str, Any] = {}
        unmapped: list[str] = []

        for hf_key, value in hf_state_dict.items():
            self._raise_if_quantized_key(hf_key)
            if hf_key.endswith("rotary_emb.inv_freq"):
                continue

            tt_key = _TEXT_GLOBAL_FROM_HF.get(hf_key)
            if tt_key is not None:
                state_dict[tt_key] = value
                continue

            tt_key = _VISION_GLOBAL_FROM_HF.get(hf_key)
            if tt_key is not None:
                if hf_key == "vision_tower.patch_embed.proj.weight":
                    value = value.reshape(value.shape[0], -1)
                state_dict[tt_key] = value
                continue

            text_match = re.fullmatch(
                r"language_model\.model\.layers\.(\d+)\.(.+)",
                hf_key,
            )
            if text_match is not None:
                layer_idx, suffix = text_match.groups()
                expert_match = re.fullmatch(
                    r"block_sparse_moe\.experts\.(\d+)\." r"(w1|w2|w3)\.weight",
                    suffix,
                )
                if expert_match is not None:
                    expert_idx, projection = expert_match.groups()
                    moe_config = self.kimi_config.layers[int(layer_idx)].moe
                    assert moe_config is not None
                    grouped_key = (
                        f"layers.{layer_idx}.moe.routed_experts."
                        f"{_EXPERT_PROJECTION_TO_GROUPED_PARAM[projection]}"
                    )
                    experts = self._expert_weights_by_layer_projection.setdefault(
                        (layer_idx, projection), {}
                    )
                    experts[int(expert_idx)] = value
                    if len(experts) == moe_config.num_experts:
                        sorted_experts = [
                            experts[i] for i in range(moe_config.num_experts)
                        ]
                        state_dict[grouped_key] = torch.stack(sorted_experts, dim=0)
                        del self._expert_weights_by_layer_projection[
                            (layer_idx, projection)
                        ]
                    continue

                layer_config = self.kimi_config.layers[int(layer_idx)]
                mapped_suffix = _TEXT_LAYER_FROM_HF.get(suffix)
                if mapped_suffix is None:
                    attention_map = (
                        _MLA_FROM_HF
                        if layer_config.attention is not None
                        else _KDA_FROM_HF
                    )
                    mapped_suffix = attention_map.get(suffix)
                if mapped_suffix is None:
                    mapped_suffix = _MOE_FROM_HF.get(suffix)
                if mapped_suffix is None:
                    unmapped.append(hf_key)
                    continue
                if suffix == "self_attn.dt_bias":
                    delta_config = layer_config.delta_attention
                    if delta_config is None:
                        raise ValueError(f"HF key '{hf_key}' targets a non-KDA layer.")
                    value = value.reshape(
                        delta_config.num_heads,
                        delta_config.head_dim,
                    )
                state_dict[f"layers.{layer_idx}.{mapped_suffix}"] = value
                continue

            vision_match = re.fullmatch(
                r"vision_tower\.encoder\.blocks\.(\d+)\.(.+)",
                hf_key,
            )
            if vision_match is not None:
                layer_idx, suffix = vision_match.groups()
                if suffix == "wqkv.weight":
                    q, k, v = torch.chunk(value, 3, dim=0)
                    base = f"vision_encoder.layers.{layer_idx}.attn"
                    state_dict[f"{base}.wq.weight"] = q
                    state_dict[f"{base}.wk.weight"] = k
                    state_dict[f"{base}.wv.weight"] = v
                    continue
                mapped_suffix = _VISION_LAYER_FROM_HF.get(suffix)
                if mapped_suffix is None:
                    unmapped.append(hf_key)
                    continue
                state_dict[f"vision_encoder.layers.{layer_idx}.{mapped_suffix}"] = value
                continue

            unmapped.append(hf_key)

        if unmapped:
            raise ValueError(
                "KimiK3StateDictAdapter found HuggingFace keys without a "
                f"mapping: {unmapped}."
            )
        if self._expert_weights_by_layer_projection:
            incomplete = list(self._expert_weights_by_layer_projection.keys())
            self._expert_weights_by_layer_projection.clear()
            raise ValueError(
                "KimiK3StateDictAdapter received an incomplete set of "
                f"routed-expert weights for (layer, projection): {incomplete}."
            )
        return state_dict

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert a TorchTitan state dict to unquantized HuggingFace format."""
        text_global_to_hf = {value: key for key, value in _TEXT_GLOBAL_FROM_HF.items()}
        vision_global_to_hf = {
            value: key for key, value in _VISION_GLOBAL_FROM_HF.items()
        }
        text_layer_to_hf = {
            value: key
            for mapping in (
                _TEXT_LAYER_FROM_HF,
                _MLA_FROM_HF,
                _KDA_FROM_HF,
                _MOE_FROM_HF,
            )
            for key, value in mapping.items()
        }
        vision_layer_to_hf = {
            value: key for key, value in _VISION_LAYER_FROM_HF.items()
        }

        hf_state_dict: dict[str, Any] = {}
        vision_qkv: dict[str, dict[str, Any]] = {}
        unmapped: list[str] = []

        for tt_key, value in state_dict.items():
            hf_key = text_global_to_hf.get(tt_key)
            if hf_key is not None:
                hf_state_dict[hf_key] = value
                continue

            hf_key = vision_global_to_hf.get(tt_key)
            if hf_key is not None:
                if tt_key == "vision_encoder.patch_embed.weight":
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
                continue

            text_match = re.fullmatch(r"layers\.(\d+)\.(.+)", tt_key)
            if text_match is not None:
                layer_idx, suffix = text_match.groups()
                expert_match = re.fullmatch(
                    r"moe\.routed_experts\." r"(w1_EFD|w2_EDF|w3_EFD)",
                    suffix,
                )
                if expert_match is not None:
                    (grouped_param,) = expert_match.groups()
                    projection = _GROUPED_PARAM_TO_EXPERT_PROJECTION[grouped_param]
                    for expert_idx, expert_weight in enumerate(value.unbind(0)):
                        hf_state_dict[
                            f"language_model.model.layers.{layer_idx}."
                            f"block_sparse_moe.experts.{expert_idx}."
                            f"{projection}.weight"
                        ] = expert_weight
                    continue

                mapped_suffix = text_layer_to_hf.get(suffix)
                if mapped_suffix is None:
                    unmapped.append(tt_key)
                    continue
                if suffix == "delta_attention.dt_bias":
                    value = value.reshape(-1)
                hf_state_dict[
                    f"language_model.model.layers.{layer_idx}.{mapped_suffix}"
                ] = value
                continue

            vision_match = re.fullmatch(
                r"vision_encoder\.layers\.(\d+)\.(.+)",
                tt_key,
            )
            if vision_match is not None:
                layer_idx, suffix = vision_match.groups()
                qkv_match = re.fullmatch(r"attn\.w(q|k|v)\.weight", suffix)
                if qkv_match is not None:
                    vision_qkv.setdefault(layer_idx, {})[qkv_match.group(1)] = value
                    continue
                mapped_suffix = vision_layer_to_hf.get(suffix)
                if mapped_suffix is None:
                    unmapped.append(tt_key)
                    continue
                hf_state_dict[
                    f"vision_tower.encoder.blocks.{layer_idx}.{mapped_suffix}"
                ] = value
                continue

            unmapped.append(tt_key)

        for layer_idx, qkv in vision_qkv.items():
            missing = {"q", "k", "v"} - qkv.keys()
            if missing:
                raise ValueError(
                    f"Vision layer {layer_idx} is missing QKV parts: {sorted(missing)}."
                )
            hf_state_dict[
                f"vision_tower.encoder.blocks.{layer_idx}.wqkv.weight"
            ] = torch.cat((qkv["q"], qkv["k"], qkv["v"]), dim=0)

        if unmapped:
            raise ValueError(
                "KimiK3StateDictAdapter found TorchTitan keys without a "
                f"mapping: {unmapped}."
            )
        return hf_state_dict

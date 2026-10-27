# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

from torchtitan.models.common.rope import CosSinRoPE
from torchtitan.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter

if TYPE_CHECKING:
    from .model import MiniMaxM3Model

__all__ = ["MiniMaxM3StateDictAdapter"]

_VISION_PREFIXES = ("vision_tower.", "multi_modal_projector.", "patch_merge_mlp.")


class MiniMaxM3StateDictAdapter(DeepSeekV3StateDictAdapter):
    """Map ``MiniMaxM3SparseForConditionalGeneration`` checkpoints to torchtitan."""

    hf_experts_key_fragment = "block_sparse_moe.experts"
    hf_rope_config_cls = CosSinRoPE.Config
    _MTP_SHARED_COPIES: dict[str, str] = {}

    def __init__(
        self,
        model_config: MiniMaxM3Model.Config,
        hf_assets_path: str | None,
    ):
        # pyrefly: ignore [bad-argument-type]
        super().__init__(model_config, hf_assets_path)
        self.vision_encoder = model_config.vision_encoder
        tower = "vision_tower.vision_model."
        block = f"{tower}encoder.layers.{{}}."
        vision_pairs = {
            f"{tower}embeddings.patch_embedding.weight": "vision_encoder.patch_embed.weight",
            f"{tower}pre_layrnorm.weight": "vision_encoder.pre_norm.weight",
            f"{tower}pre_layrnorm.bias": "vision_encoder.pre_norm.bias",
        }
        for hf_name, titan_name in (
            ("layer_norm1", "norm1"),
            ("layer_norm2", "norm2"),
            ("self_attn.q_proj", "attn.wq"),
            ("self_attn.k_proj", "attn.wk"),
            ("self_attn.v_proj", "attn.wv"),
            ("self_attn.out_proj", "attn.proj"),
            ("mlp.fc1", "mlp.linear_fc1"),
            ("mlp.fc2", "mlp.linear_fc2"),
        ):
            for kind in ("weight", "bias"):
                vision_pairs[
                    f"{block}{hf_name}.{kind}"
                ] = f"vision_encoder.layers.{{}}.{titan_name}.{kind}"
        for hf_name, titan_name in (
            ("multi_modal_projector.linear_1", "linear_1"),
            ("multi_modal_projector.linear_2", "linear_2"),
            ("patch_merge_mlp.linear_1", "merge_linear_1"),
            ("patch_merge_mlp.linear_2", "merge_linear_2"),
        ):
            for kind in ("weight", "bias"):
                vision_pairs[
                    f"{hf_name}.{kind}"
                ] = f"vision_encoder.projector.{titan_name}.{kind}"
        self.vision_from_hf_map = vision_pairs

    def _build_from_hf_map(  # pyrefly: ignore[bad-override]
        self, model_config: MiniMaxM3Model.Config
    ) -> dict[str, str]:
        del model_config
        text = "language_model.model."
        layer = f"{text}layers.{{}}."
        moe = f"{layer}block_sparse_moe."
        return {
            f"{text}embed_tokens.weight": "tok_embeddings.weight",
            # Attention, with the MSA indexer on sparse layers
            f"{layer}self_attn.q_proj.weight": "layers.{}.attention.qkv_linear.wq.weight",
            f"{layer}self_attn.k_proj.weight": "layers.{}.attention.qkv_linear.wk.weight",
            f"{layer}self_attn.v_proj.weight": "layers.{}.attention.qkv_linear.wv.weight",
            f"{layer}self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            f"{layer}self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
            f"{layer}self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
            f"{layer}self_attn.index_q_proj.weight": "layers.{}.attention.indexer.wq.weight",
            f"{layer}self_attn.index_k_proj.weight": "layers.{}.attention.indexer.wk.weight",
            f"{layer}self_attn.index_q_norm.weight": "layers.{}.attention.indexer.q_norm.weight",
            f"{layer}self_attn.index_k_norm.weight": "layers.{}.attention.indexer.k_norm.weight",
            # Dense feed-forward (first layers)
            f"{layer}mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            f"{layer}mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            f"{layer}mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            # MoE
            f"{moe}experts.{{}}.w1.weight": "layers.{}.feed_forward.routed_experts.w1_EFD",
            f"{moe}experts.{{}}.w3.weight": "layers.{}.feed_forward.routed_experts.w3_EFD",
            f"{moe}experts.{{}}.w2.weight": "layers.{}.feed_forward.routed_experts.w2.weight",
            f"{moe}gate.weight": "layers.{}.feed_forward.router.gate.weight",
            f"{moe}e_score_correction_bias": "layers.{}.feed_forward.expert_bias_E",
            f"{moe}shared_experts.gate_proj.weight": "layers.{}.feed_forward.shared_experts.w1.weight",
            f"{moe}shared_experts.up_proj.weight": "layers.{}.feed_forward.shared_experts.w3.weight",
            f"{moe}shared_experts.down_proj.weight": "layers.{}.feed_forward.shared_experts.w2.weight",
            # Norms and head
            f"{layer}input_layernorm.weight": "layers.{}.attention_norm.weight",
            f"{layer}post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            f"{text}norm.weight": "norm.weight",
            "language_model.lm_head.weight": "lm_head.weight",
        }

    def from_hf(
        self, hf_state_dict: dict[str, Any], *, is_quantized_layout: bool = False
    ) -> dict[str, Any]:
        """Map the vision keys here and the decoder keys through DeepSeek V3's path."""
        decoder_hf: dict[str, Any] = {}
        vision: dict[str, Any] = {}
        unmapped: list[str] = []
        for key, value in hf_state_dict.items():
            if not key.startswith(_VISION_PREFIXES):
                decoder_hf[key] = value
                continue
            if key in self.vision_from_hf_map:
                new_key = self.vision_from_hf_map[key]
                if new_key == "vision_encoder.patch_embed.weight":
                    # HF Conv3d (out, C, T, kH, kW) -> Linear (out, C*T*kH*kW).
                    value = value.reshape(value.shape[0], -1)
                vision[new_key] = value
                continue
            abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
            if abstract_key not in self.vision_from_hf_map:
                unmapped.append(key)
                continue
            # pyrefly: ignore [missing-attribute]
            layer_num = re.search(r"\d+", key).group(0)
            vision[self.vision_from_hf_map[abstract_key].format(layer_num)] = value

        if unmapped:
            raise ValueError(
                f"MiniMaxM3StateDictAdapter: {len(unmapped)} vision key(s) have no "
                f"mapping: {unmapped}."
            )
        if vision and self.vision_encoder is None:
            raise ValueError(
                "The checkpoint has vision weights but the model has no vision encoder."
            )
        state_dict = super().from_hf(
            decoder_hf, is_quantized_layout=is_quantized_layout
        )
        state_dict.update(vision)
        return state_dict

    def to_hf(
        self, state_dict: dict[str, Any], *, to_quantized_layout: bool = False
    ) -> dict[str, Any]:
        """Map the vision keys here and the decoder keys through DeepSeek V3's path."""
        to_hf_map = {v: k for k, v in self.vision_from_hf_map.items()}
        decoder: dict[str, Any] = {}
        hf_state_dict: dict[str, Any] = {}
        for key, value in state_dict.items():
            if not key.startswith("vision_encoder."):
                decoder[key] = value
            elif key == "vision_encoder.patch_embed.weight":
                assert self.vision_encoder is not None
                encoder = self.vision_encoder
                # Linear (out, C*T*kH*kW) -> HF Conv3d (out, C, T, kH, kW).
                hf_state_dict[to_hf_map[key]] = value.reshape(
                    -1,
                    encoder.in_channels,
                    encoder.temporal_patch_size,
                    encoder.patch_size,
                    encoder.patch_size,
                )
            elif key in to_hf_map:
                hf_state_dict[to_hf_map[key]] = value
            else:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)
                hf_state_dict[to_hf_map[abstract_key].format(layer_num)] = value
        hf_state_dict.update(
            super().to_hf(decoder, to_quantized_layout=to_quantized_layout)
        )
        return hf_state_dict

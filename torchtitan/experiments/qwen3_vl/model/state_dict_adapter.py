# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
State dict adapter for Qwen3-VL.

Converts between HuggingFace Qwen3-VL checkpoint format and torchtitan format.

Key differences from Qwen3 text-only adapter:
- HF uses `model.language_model.` prefix for text layers (not `model.`)
- HF uses `model.visual.blocks.{i}.*` for vision blocks (TT uses `visual.layers.{i}.*`)
- HF uses Conv3d for patch embedding, TT uses Linear (requires weight reshape)
- Vision encoder has fused QKV, LayerNorm with bias
- DeepStack merger list parameters
"""

import re
from typing import Any

from torch.distributed.tensor import DTensor
from torchtitan.models.utils import MoEStateDictAdapter

from .args import Qwen3VLModelArgs


class Qwen3VLStateDictAdapter(MoEStateDictAdapter):
    def __init__(self, model_args: Qwen3VLModelArgs, hf_assets_path: str | None):
        super().__init__(model_args, hf_assets_path)

        self.from_hf_map = {
            # ===== Language Model =====
            "model.language_model.embed_tokens.weight": "tok_embeddings.weight",
            # Attention
            "model.language_model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
            "model.language_model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
            "model.language_model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
            "model.language_model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.language_model.layers.{}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
            "model.language_model.layers.{}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
            "model.language_model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            # Non-MoE MLP
            "model.language_model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.language_model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.language_model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            # Layer norms
            "model.language_model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.language_model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            # MoE
            "model.language_model.layers.{}.mlp.experts.{}.gate_proj.weight": "layers.{}.moe.experts.w1",
            "model.language_model.layers.{}.mlp.experts.{}.up_proj.weight": "layers.{}.moe.experts.w3",
            "model.language_model.layers.{}.mlp.experts.{}.down_proj.weight": "layers.{}.moe.experts.w2",
            "model.language_model.layers.{}.mlp.gate.weight": "layers.{}.moe.router.gate.weight",
            # Final norm and output
            "model.language_model.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
            # ===== Vision Encoder =====
            # Patch embedding (Conv3d in HF, Linear in TT - weight reshape needed)
            "model.visual.patch_embed.proj.weight": "visual.patch_embed.proj.weight",
            "model.visual.patch_embed.proj.bias": "visual.patch_embed.proj.bias",
            # Position embeddings
            "model.visual.pos_embed.weight": "visual.pos_embed.weight",
            # Vision transformer blocks (HF: blocks, TT: layers)
            "model.visual.blocks.{}.norm1.weight": "visual.layers.{}.norm1.weight",
            "model.visual.blocks.{}.norm1.bias": "visual.layers.{}.norm1.bias",
            "model.visual.blocks.{}.norm2.weight": "visual.layers.{}.norm2.weight",
            "model.visual.blocks.{}.norm2.bias": "visual.layers.{}.norm2.bias",
            "model.visual.blocks.{}.attn.qkv.weight": "visual.layers.{}.attn.qkv.weight",
            "model.visual.blocks.{}.attn.qkv.bias": "visual.layers.{}.attn.qkv.bias",
            "model.visual.blocks.{}.attn.proj.weight": "visual.layers.{}.attn.proj.weight",
            "model.visual.blocks.{}.attn.proj.bias": "visual.layers.{}.attn.proj.bias",
            "model.visual.blocks.{}.mlp.linear_fc1.weight": "visual.layers.{}.mlp.linear_fc1.weight",
            "model.visual.blocks.{}.mlp.linear_fc1.bias": "visual.layers.{}.mlp.linear_fc1.bias",
            "model.visual.blocks.{}.mlp.linear_fc2.weight": "visual.layers.{}.mlp.linear_fc2.weight",
            "model.visual.blocks.{}.mlp.linear_fc2.bias": "visual.layers.{}.mlp.linear_fc2.bias",
            # Merger (maps vision dim to LLM dim)
            "model.visual.merger.norm.weight": "visual.merger.norm.weight",
            "model.visual.merger.norm.bias": "visual.merger.norm.bias",
            "model.visual.merger.linear_fc1.weight": "visual.merger.linear_fc1.weight",
            "model.visual.merger.linear_fc1.bias": "visual.merger.linear_fc1.bias",
            "model.visual.merger.linear_fc2.weight": "visual.merger.linear_fc2.weight",
            "model.visual.merger.linear_fc2.bias": "visual.merger.linear_fc2.bias",
            # DeepStack mergers
            "model.visual.deepstack_merger_list.{}.norm.weight": "visual.deepstack_merger_list.{}.norm.weight",
            "model.visual.deepstack_merger_list.{}.norm.bias": "visual.deepstack_merger_list.{}.norm.bias",
            "model.visual.deepstack_merger_list.{}.linear_fc1.weight": "visual.deepstack_merger_list.{}.linear_fc1.weight",
            "model.visual.deepstack_merger_list.{}.linear_fc1.bias": "visual.deepstack_merger_list.{}.linear_fc1.bias",
            "model.visual.deepstack_merger_list.{}.linear_fc2.weight": "visual.deepstack_merger_list.{}.linear_fc2.weight",
            "model.visual.deepstack_merger_list.{}.linear_fc2.bias": "visual.deepstack_merger_list.{}.linear_fc2.bias",
        }

    def _is_indexed_key(self, key: str) -> bool:
        """Check if a key contains a layer/block/merger index (number between dots)."""
        return bool(re.search(r"\.\d+\.", key))

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert torchtitan state dict to HuggingFace format."""
        to_hf_map = {v: k for k, v in self.from_hf_map.items() if v is not None}
        hf_state_dict = {}

        for key, value in state_dict.items():
            if "moe.experts" in key:
                # MoE expert handling (same pattern as Qwen3)
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                if abstract_key not in to_hf_map:
                    continue
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)
                new_abstract_key = to_hf_map[abstract_key]

                if isinstance(value, DTensor):
                    self.grouped_expert_weight_placements[
                        abstract_key
                    ] = value.placements
                    self.grouped_expert_weight_shape[abstract_key] = value.shape
                    self.grouped_expert_weight_mesh[abstract_key] = value.device_mesh

                    local_expert_fqn = self._get_local_experts_weights(
                        new_abstract_key,
                        abstract_key,
                        layer_num,
                        value,
                    )
                    hf_state_dict.update(local_expert_fqn)
                else:
                    split_values = self._split_experts_weights(
                        value,
                        # pyrefly: ignore [missing-attribute]
                        self.model_args.moe_args.num_experts,
                    )
                    # pyrefly: ignore [missing-attribute]
                    for expert_num in range(self.model_args.moe_args.num_experts):
                        new_key = new_abstract_key.format(layer_num, expert_num)
                        hf_state_dict[new_key] = split_values[expert_num].squeeze()

            elif self._is_indexed_key(key):
                # Keys with layer/block/merger indices
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                if abstract_key not in to_hf_map:
                    continue
                # pyrefly: ignore [missing-attribute]
                idx = re.search(r"\d+", key).group(0)
                new_key = to_hf_map[abstract_key].format(idx)
                hf_state_dict[new_key] = value

            else:
                # Direct mapping (tok_embeddings, norm, output, visual.patch_embed, etc.)
                if key not in to_hf_map:
                    continue
                # Some HF Qwen3-VL variants (e.g. 2B) tie lm_head.weight
                # with embed_tokens.weight. Skip output.weight so DCP
                # doesn't try to load the non-existent lm_head.weight.
                # pyrefly: ignore [missing-attribute]
                if key == "output.weight" and self.model_args.enable_weight_tying:
                    continue
                new_key = to_hf_map[key]
                new_value = value
                # Reshape Linear weight to Conv3d for patch embedding
                if key == "visual.patch_embed.proj.weight":
                    encoder = self.model_args.encoder
                    new_value = value.reshape(
                        value.shape[0],
                        encoder.in_channels,
                        encoder.temporal_patch_size,
                        encoder.patch_size,
                        encoder.patch_size,
                    )
                hf_state_dict[new_key] = new_value

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert HuggingFace state dict to torchtitan format."""
        state_dict = {}
        expert_weights_by_layer = {}

        # HF Qwen3-VL ties lm_head.weight with embed_tokens.weight,
        # so lm_head.weight may not be stored in the checkpoint.
        if "lm_head.weight" not in hf_state_dict:
            assert "model.language_model.embed_tokens.weight" in hf_state_dict
            hf_state_dict["lm_head.weight"] = hf_state_dict[
                "model.language_model.embed_tokens.weight"
            ]

        for key, value in hf_state_dict.items():
            if "mlp.experts" in key:
                # MoE expert handling (same pattern as Qwen3)
                abstract_key = re.sub(r"(\d+)", "{}", key, count=2)
                if abstract_key not in self.from_hf_map:
                    continue
                layer_num, expert_num = re.findall(r"\d+", key)[:2]
                titan_abstract_key = self.from_hf_map[abstract_key]
                if titan_abstract_key is None:
                    continue
                new_key = titan_abstract_key.format(layer_num)

                if layer_num not in expert_weights_by_layer:
                    expert_weights_by_layer[layer_num] = {}
                if titan_abstract_key not in expert_weights_by_layer[layer_num]:
                    expert_weights_by_layer[layer_num][titan_abstract_key] = {}
                expert_weights_by_layer[layer_num][titan_abstract_key][
                    int(expert_num)
                ] = value

                if titan_abstract_key in self.local_experts_indices:
                    stacked_value = self._concatenate_expert_weights_dtensor(
                        expert_weights_by_layer,
                        titan_abstract_key,
                        layer_num,
                    )
                else:
                    stacked_value = self._concatenate_expert_weights(
                        expert_weights_by_layer,
                        titan_abstract_key,
                        layer_num,
                        # pyrefly: ignore [missing-attribute]
                        self.model_args.moe_args.num_experts,
                    )

                if stacked_value is not None:
                    state_dict[new_key] = stacked_value

            elif self._is_indexed_key(key):
                # Keys with layer/block/merger indices
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                if abstract_key not in self.from_hf_map:
                    continue
                new_key = self.from_hf_map[abstract_key]
                if new_key is None:
                    continue
                # pyrefly: ignore [missing-attribute]
                idx = re.search(r"\d+", key).group(0)
                new_key = new_key.format(idx)
                state_dict[new_key] = value

            else:
                # Direct mapping
                if key not in self.from_hf_map:
                    continue
                new_key = self.from_hf_map[key]
                if new_key is None:
                    continue
                new_value = value
                # Reshape Conv3d weight to Linear for patch embedding
                if key == "model.visual.patch_embed.proj.weight":
                    new_value = value.reshape(value.shape[0], -1)
                state_dict[new_key] = new_value

        return state_dict

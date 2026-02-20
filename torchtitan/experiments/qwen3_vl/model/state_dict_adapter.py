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
- HF MoE checkpoints use grouped expert weights (3D tensors) with fused
  gate_up_proj and no `.weight` suffix, requiring split/fuse and transpose
"""

import re
from typing import Any

import torch

from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .args import Qwen3VLModelArgs


class Qwen3VLStateDictAdapter(StateDictAdapter):
    def __init__(self, model_args: Qwen3VLModelArgs, hf_assets_path: str | None):
        super().__init__(model_args, hf_assets_path)
        self.model_args = model_args

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
            # MoE (grouped experts, handled specially due to fused gate_up_proj and transpose)
            # gate_up_proj is mapped to w1+w3 via custom logic, not through from_hf_map
            "model.language_model.layers.{}.mlp.experts.down_proj": "layers.{}.moe.experts.w2",
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

        # Collect MoE w1/w3 per layer to fuse into gate_up_proj
        moe_w1_by_layer: dict[str, Any] = {}
        moe_w3_by_layer: dict[str, Any] = {}

        for key, value in state_dict.items():
            if self._is_indexed_key(key):
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore [missing-attribute]
                idx = re.search(r"\d+", key).group(0)

                # Collect w1/w3 for fusing into gate_up_proj
                if abstract_key == "layers.{}.moe.experts.w1":
                    moe_w1_by_layer[idx] = value
                    continue
                elif abstract_key == "layers.{}.moe.experts.w3":
                    moe_w3_by_layer[idx] = value
                    continue

                # Handle down_proj transpose: TT w2 [E, dim, hidden] -> HF [E, hidden, dim]
                if abstract_key == "layers.{}.moe.experts.w2":
                    new_key = f"model.language_model.layers.{idx}.mlp.experts.down_proj"
                    hf_state_dict[new_key] = value.transpose(-2, -1)
                    continue

                if abstract_key not in to_hf_map:
                    continue
                new_key = to_hf_map[abstract_key].format(idx)
                hf_state_dict[new_key] = value

            else:
                if key not in to_hf_map:
                    continue
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

        # Fuse w1 (gate) and w3 (up) into gate_up_proj per layer
        # TT w1/w3: [E, hidden_dim, dim] -> transpose to [E, dim, hidden_dim] -> cat on last dim
        for layer_idx in moe_w1_by_layer:
            w1 = moe_w1_by_layer[layer_idx].transpose(-2, -1)  # [E, dim, hidden_dim]
            w3 = moe_w3_by_layer[layer_idx].transpose(-2, -1)  # [E, dim, hidden_dim]
            gate_up = torch.cat([w1, w3], dim=-1)  # [E, dim, 2*hidden_dim]
            hf_key = f"model.language_model.layers.{layer_idx}.mlp.experts.gate_up_proj"
            hf_state_dict[hf_key] = gate_up

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert HuggingFace state dict to torchtitan format."""
        state_dict = {}

        # HF Qwen3-VL ties lm_head.weight with embed_tokens.weight,
        # so lm_head.weight may not be stored in the checkpoint.
        if "lm_head.weight" not in hf_state_dict:
            assert "model.language_model.embed_tokens.weight" in hf_state_dict
            hf_state_dict["lm_head.weight"] = hf_state_dict[
                "model.language_model.embed_tokens.weight"
            ]

        for key, value in hf_state_dict.items():
            if self._is_indexed_key(key):
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore [missing-attribute]
                idx = re.search(r"\d+", key).group(0)

                # Handle fused gate_up_proj: split and transpose
                # HF gate_up_proj: [E, dim, 2*hidden_dim] -> split -> transpose each to [E, hidden_dim, dim]
                if abstract_key == "model.language_model.layers.{}.mlp.experts.gate_up_proj":
                    w1_hf, w3_hf = value.chunk(2, dim=-1)  # each [E, dim, hidden_dim]
                    state_dict[f"layers.{idx}.moe.experts.w1"] = w1_hf.transpose(-2, -1)
                    state_dict[f"layers.{idx}.moe.experts.w3"] = w3_hf.transpose(-2, -1)
                    continue

                # Handle down_proj transpose: HF [E, hidden, dim] -> TT w2 [E, dim, hidden]
                if abstract_key == "model.language_model.layers.{}.mlp.experts.down_proj":
                    state_dict[f"layers.{idx}.moe.experts.w2"] = value.transpose(-2, -1)
                    continue

                if abstract_key not in self.from_hf_map:
                    continue
                new_key = self.from_hf_map[abstract_key]
                if new_key is None:
                    continue
                new_key = new_key.format(idx)
                state_dict[new_key] = value

            else:
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

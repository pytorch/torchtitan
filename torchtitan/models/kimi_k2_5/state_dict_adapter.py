# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""State dict adapter for Kimi K2.5 (MoonViT3d + DeepSeekV3).

Handles bidirectional conversion between the HuggingFace/SGLang checkpoint
layout and the torchtitan layout for both the vision tower and the language
model.

The language-model mapping mirrors ``DeepSeekV3StateDictAdapter`` (MLA + MoE).
The vision-tower key names follow the SGLang Kimi-K2.5 reference
(``vision_tower.*`` / ``mm_projector.*``); two structural transforms are
applied:

- **Patch embed**: HF stores a ``Conv2d`` weight ``(out, C, kH, kW)``; torchtitan
  uses a ``Linear`` weight ``(out, C*kH*kW)`` — a pure reshape.
- **Fused vs separate QKV**: HF stores a single fused vision-attention
  projection (``wqkv`` / ``attn.qkv_proj``); torchtitan uses separate
  ``wq``/``wk``/``wv`` (TP-correct), so the fused weight/bias is split on load
  and re-concatenated on save.
"""

import re
from typing import Any

import torch
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.tensor import DTensor

from torchtitan.models.utils import MoEStateDictAdapter

from .model import KimiK25Model

# HF fused vision-attention projection prefix (per the SGLang reference
# ``load_weights``: ``wqkv.`` is renamed to ``attn.qkv_proj.``).
_HF_VISION_QKV = "vision_tower.encoder.blocks.{}.attn.qkv_proj"


class KimiK25StateDictAdapter(MoEStateDictAdapter):
    """State dict adapter for Kimi K2.5."""

    def __init__(
        self,
        model_config: KimiK25Model.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)

        self.patch_size = model_config.vision_encoder.patch_size
        self.in_channels = model_config.vision_encoder.in_channels

        # --- Vision tower (1:1 keys; qkv handled separately) ---
        self.vision_from_hf_map = {
            # Patch embedding (Conv2d weight reshaped to Linear on load).
            "vision_tower.patch_embed.proj.weight": "vision_encoder.patch_embed.weight",
            "vision_tower.patch_embed.proj.bias": "vision_encoder.patch_embed.bias",
            # Learnable spatial position embedding.
            "vision_tower.patch_embed.pos_emb.weight": "vision_encoder.pos_emb.weight",
            # Encoder block norms.
            "vision_tower.encoder.blocks.{}.norm0.weight": "vision_encoder.layers.{}.norm0.weight",
            "vision_tower.encoder.blocks.{}.norm0.bias": "vision_encoder.layers.{}.norm0.bias",
            "vision_tower.encoder.blocks.{}.norm1.weight": "vision_encoder.layers.{}.norm1.weight",
            "vision_tower.encoder.blocks.{}.norm1.bias": "vision_encoder.layers.{}.norm1.bias",
            # Encoder block attention output projection.
            "vision_tower.encoder.blocks.{}.attn.proj.weight": "vision_encoder.layers.{}.attn.proj.weight",
            "vision_tower.encoder.blocks.{}.attn.proj.bias": "vision_encoder.layers.{}.attn.proj.bias",
            # Encoder block MLP (MLP2: linear_1/linear_2).
            "vision_tower.encoder.blocks.{}.mlp.linear_1.weight": "vision_encoder.layers.{}.mlp.fc1.weight",
            "vision_tower.encoder.blocks.{}.mlp.linear_1.bias": "vision_encoder.layers.{}.mlp.fc1.bias",
            "vision_tower.encoder.blocks.{}.mlp.linear_2.weight": "vision_encoder.layers.{}.mlp.fc2.weight",
            "vision_tower.encoder.blocks.{}.mlp.linear_2.bias": "vision_encoder.layers.{}.mlp.fc2.bias",
            # Final encoder norm.
            "vision_tower.encoder.final_layernorm.weight": "vision_encoder.final_norm.weight",
            "vision_tower.encoder.final_layernorm.bias": "vision_encoder.final_norm.bias",
            # Multimodal projector.
            "mm_projector.pre_norm.weight": "vision_encoder.projector.pre_norm.weight",
            "mm_projector.pre_norm.bias": "vision_encoder.projector.pre_norm.bias",
            "mm_projector.linear_1.weight": "vision_encoder.projector.linear_1.weight",
            "mm_projector.linear_1.bias": "vision_encoder.projector.linear_1.bias",
            "mm_projector.linear_2.weight": "vision_encoder.projector.linear_2.weight",
            "mm_projector.linear_2.bias": "vision_encoder.projector.linear_2.bias",
        }

        # --- Language model (DeepSeekV3: MLA + MoE) ---
        self.from_hf_map = {
            "model.embed_tokens.weight": "tok_embeddings.weight",
            "model.layers.{}.self_attn.kv_a_proj_with_mqa.weight": "layers.{}.attention.wkv_a.weight",
            "model.layers.{}.self_attn.kv_a_layernorm.weight": "layers.{}.attention.kv_norm.weight",
            "model.layers.{}.self_attn.kv_b_proj.weight": "layers.{}.attention.wkv_b.weight",
            "model.layers.{}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            "model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            "model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            "model.layers.{}.mlp.experts.{}.gate_proj.weight": "layers.{}.moe.experts.w1_EFD",
            "model.layers.{}.mlp.experts.{}.up_proj.weight": "layers.{}.moe.experts.w3_EFD",
            "model.layers.{}.mlp.experts.{}.down_proj.weight": "layers.{}.moe.experts.w2_EDF",
            "model.layers.{}.mlp.gate.weight": "layers.{}.moe.router.gate.weight",
            "model.layers.{}.mlp.shared_experts.gate_proj.weight": "layers.{}.moe.shared_experts.w1.weight",
            "model.layers.{}.mlp.shared_experts.up_proj.weight": "layers.{}.moe.shared_experts.w3.weight",
            "model.layers.{}.mlp.shared_experts.down_proj.weight": "layers.{}.moe.shared_experts.w2.weight",
            "model.layers.{}.mlp.gate.e_score_correction_bias": "layers.{}.moe.expert_bias_E",
            "model.norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
        }

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

        # Merge in the 1:1 vision keys.
        self.from_hf_map.update(self.vision_from_hf_map)

    def get_hf_storage_reader(
        self, path: str, from_quantized: bool = False
    ) -> HuggingFaceStorageReader:
        if from_quantized:
            from torch.distributed.checkpoint.quantized_hf_storage import (
                QuantizedHuggingFaceStorageReader,
            )

            return QuantizedHuggingFaceStorageReader(
                path=path,
                target_dtype=torch.float32,
                block_size=128,
                thread_count=4,
            )
        return HuggingFaceStorageReader(path)

    # --- Vision patch-embed Conv2d <-> Linear ---

    def _patch_embed_to_linear(self, weight: torch.Tensor) -> torch.Tensor:
        """HF Conv2d ``(out, C, kH, kW)`` -> torchtitan Linear ``(out, C*kH*kW)``."""
        if weight.ndim == 4:
            return weight.reshape(weight.shape[0], -1)
        return weight

    def _patch_embed_to_conv(self, weight: torch.Tensor) -> torch.Tensor:
        """torchtitan Linear ``(out, C*kH*kW)`` -> HF Conv2d ``(out, C, kH, kW)``."""
        if weight.ndim == 2:
            return weight.reshape(
                -1, self.in_channels, self.patch_size, self.patch_size
            )
        return weight

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        to_hf_map = {v: k for k, v in self.from_hf_map.items()}
        hf_state_dict: dict[str, Any] = {}

        # Buffer separate vision q/k/v per layer to re-fuse into one HF tensor.
        vision_qkv: dict[str, dict[str, torch.Tensor]] = {}

        for key, value in state_dict.items():
            if "moe.experts" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                new_abstract_key = to_hf_map[abstract_key]

                if isinstance(value, DTensor):
                    self.grouped_expert_weight_placements[
                        abstract_key
                    ] = value.placements
                    self.grouped_expert_weight_shape[abstract_key] = value.shape
                    self.grouped_expert_weight_mesh[abstract_key] = value.device_mesh
                    hf_state_dict.update(
                        self._get_local_experts_weights(
                            new_abstract_key, abstract_key, layer_num, value
                        )
                    )
                else:
                    moe_layer = next(
                        l for l in self.model_config.layers if l.moe is not None
                    )
                    split_values = self._split_experts_weights(
                        value, moe_layer.moe.num_experts
                    )
                    for expert_num in range(moe_layer.moe.num_experts):
                        new_key = new_abstract_key.format(layer_num, expert_num)
                        hf_state_dict[new_key] = split_values[expert_num].squeeze()

            elif re.search(r"vision_encoder\.layers\.\d+\.attn\.w[qkv]\.", key):
                # Buffer separate q/k/v to re-fuse below.
                layer_num = re.search(r"\d+", key).group(0)
                proj = re.search(r"attn\.(w[qkv])\.(weight|bias)", key)
                which, kind = proj.group(1), proj.group(2)
                vision_qkv.setdefault((layer_num, kind), {})[which] = value

            elif "patch_embed.weight" in key:
                hf_state_dict[to_hf_map[key]] = self._patch_embed_to_conv(value)

            elif "vision_encoder.layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                hf_state_dict[to_hf_map[abstract_key].format(layer_num)] = value

            elif "vision_encoder" in key:
                hf_state_dict[to_hf_map[key]] = value

            elif "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                hf_state_dict[to_hf_map[abstract_key].format(layer_num)] = value

            else:
                hf_state_dict[to_hf_map[key]] = value

        # Re-fuse vision q/k/v -> single HF qkv tensor per (layer, weight|bias).
        for (layer_num, kind), parts in vision_qkv.items():
            fused = torch.cat([parts["wq"], parts["wk"], parts["wv"]], dim=0)
            hf_state_dict[f"{_HF_VISION_QKV.format(layer_num)}.{kind}"] = fused

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict: dict[str, Any] = {}
        expert_weights_by_layer: dict = {}

        for key, value in hf_state_dict.items():
            # Some Kimi-K2.5 releases prefix language-model weights.
            if key.startswith("language_model."):
                key = key.replace("language_model.", "", 1)

            if "mlp.experts" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=2)
                layer_num, expert_num = re.findall(r"\d+", key)
                titan_abstract_key = self.from_hf_map[abstract_key]
                new_key = titan_abstract_key.format(layer_num)

                expert_weights_by_layer.setdefault(layer_num, {}).setdefault(
                    titan_abstract_key, {}
                )[int(expert_num)] = value

                if titan_abstract_key in self.local_experts_indices:
                    stacked = self._concatenate_expert_weights_dtensor(
                        expert_weights_by_layer, titan_abstract_key, layer_num
                    )
                else:
                    stacked = self._concatenate_expert_weights(
                        expert_weights_by_layer,
                        titan_abstract_key,
                        layer_num,
                        next(
                            l for l in self.model_config.layers if l.moe is not None
                        ).moe.num_experts,
                    )
                if stacked is not None:
                    state_dict[new_key] = stacked

            elif re.search(r"attn\.qkv_proj\.(weight|bias)", key) and key.startswith(
                "vision_tower."
            ):
                # Split fused HF vision qkv -> separate wq/wk/wv.
                layer_num = re.search(r"\d+", key).group(0)
                kind = "weight" if key.endswith("weight") else "bias"
                q, k, v = torch.chunk(value, 3, dim=0)
                base = f"vision_encoder.layers.{layer_num}.attn"
                state_dict[f"{base}.wq.{kind}"] = q
                state_dict[f"{base}.wk.{kind}"] = k
                state_dict[f"{base}.wv.{kind}"] = v

            elif key.startswith("vision_tower.") or key.startswith("mm_projector."):
                if key in self.from_hf_map:
                    new_key = self.from_hf_map[key]
                    if new_key == "vision_encoder.patch_embed.weight":
                        value = self._patch_embed_to_linear(value)
                    state_dict[new_key] = value
                else:
                    abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                    if abstract_key in self.from_hf_map:
                        layer_num = re.search(r"\d+", key).group(0)
                        state_dict[
                            self.from_hf_map[abstract_key].format(layer_num)
                        ] = value

            elif "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                state_dict[self.from_hf_map[abstract_key].format(layer_num)] = value

            elif key in self.from_hf_map:
                state_dict[self.from_hf_map[key]] = value

        return state_dict

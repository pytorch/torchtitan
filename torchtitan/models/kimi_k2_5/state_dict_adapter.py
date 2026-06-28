# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""State dict adapter for Kimi K2.5 (MoonViT3d + DeepSeekV3).

The language model is DeepSeekV3, so this subclasses
``DeepSeekV3StateDictAdapter`` and delegates every LM key to it; only the vision
tower / projector are handled here.

Vision name/shape mappings (``HF -> torchtitan``; reversed on save):

- attention qkv:  ``wqkv`` (fused)                       ->  ``attn.wq``/``wk``/``wv`` (split)
- attention proj: ``wo``                                 ->  ``attn.proj``
- projector mlp:  ``mm_projector.proj.0``/``2``            ->  ``projector.linear_1``/``2``
- patch embed:    ``Conv2d`` weight                      ->  ``Linear`` weight (reshape)
"""

import re
from typing import Any

import torch

from torchtitan.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter

from .model import KimiK25Model


class KimiK25StateDictAdapter(DeepSeekV3StateDictAdapter):
    def __init__(
        self,
        model_config: KimiK25Model.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)

        self.vision_encoder = model_config.vision_encoder
        if self.vision_encoder is None:
            return

        self.patch_size = self.vision_encoder.patch_size
        self.in_channels = self.vision_encoder.in_channels

        # Vision tower: HF name -> torchtitan name (fused qkv handled separately).
        self.vision_from_hf_map = {
            # Patch embedding (Conv2d weight reshaped to Linear on load).
            "vision_tower.patch_embed.proj.weight": "vision_encoder.patch_embed.weight",
            "vision_tower.patch_embed.proj.bias": "vision_encoder.patch_embed.bias",
            # Learnable spatial position embedding.
            "vision_tower.patch_embed.pos_emb.weight": "vision_encoder.pos_embed",
            # Block norms: HF norm0/norm1 (pre-attn / pre-mlp) -> tt norm1/norm2.
            "vision_tower.encoder.blocks.{}.norm0.weight": "vision_encoder.layers.{}.norm1.weight",
            "vision_tower.encoder.blocks.{}.norm0.bias": "vision_encoder.layers.{}.norm1.bias",
            "vision_tower.encoder.blocks.{}.norm1.weight": "vision_encoder.layers.{}.norm2.weight",
            "vision_tower.encoder.blocks.{}.norm1.bias": "vision_encoder.layers.{}.norm2.bias",
            # Attention output projection: HF wo -> tt attn.proj.
            "vision_tower.encoder.blocks.{}.wo.weight": "vision_encoder.layers.{}.attn.proj.weight",
            "vision_tower.encoder.blocks.{}.wo.bias": "vision_encoder.layers.{}.attn.proj.bias",
            # Block MLP: HF fc0/fc1 -> tt linear_fc1/linear_fc2.
            "vision_tower.encoder.blocks.{}.mlp.fc0.weight": "vision_encoder.layers.{}.mlp.linear_fc1.weight",
            "vision_tower.encoder.blocks.{}.mlp.fc0.bias": "vision_encoder.layers.{}.mlp.linear_fc1.bias",
            "vision_tower.encoder.blocks.{}.mlp.fc1.weight": "vision_encoder.layers.{}.mlp.linear_fc2.weight",
            "vision_tower.encoder.blocks.{}.mlp.fc1.bias": "vision_encoder.layers.{}.mlp.linear_fc2.bias",
            # Final encoder norm.
            "vision_tower.encoder.final_layernorm.weight": "vision_encoder.final_norm.weight",
            "vision_tower.encoder.final_layernorm.bias": "vision_encoder.final_norm.bias",
            # Multimodal projector (1T Kimi-K2.5 spelling; Kimi-VL's
            # multi_modal_projector.linear_1/2 is normalized to this in from_hf).
            "mm_projector.pre_norm.weight": "vision_encoder.projector.pre_norm.weight",
            "mm_projector.pre_norm.bias": "vision_encoder.projector.pre_norm.bias",
            "mm_projector.proj.0.weight": "vision_encoder.projector.linear_1.weight",
            "mm_projector.proj.0.bias": "vision_encoder.projector.linear_1.bias",
            "mm_projector.proj.2.weight": "vision_encoder.projector.linear_2.weight",
            "mm_projector.proj.2.bias": "vision_encoder.projector.linear_2.bias",
        }

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        if self.vision_encoder is None:
            return super().from_hf(hf_state_dict)

        lm_hf: dict[str, Any] = {}
        vision: dict[str, Any] = {}
        unmapped: list[str] = []
        for key, value in hf_state_dict.items():
            # RoPE inv_freq is recomputed at runtime -- no target to map.
            if key.endswith("rotary_emb.inv_freq"):
                continue

            if not (
                key.startswith("vision_tower.")
                or key.startswith("mm_projector.")
                or key.startswith("multi_modal_projector.")
            ):
                # LM key. The inherited DeepSeek-V3 adapter (super) is keyed on
                # bare ``model.*`` names, but the released checkpoints nest the
                # text tower under ``language_model.`` -- strip that prefix so super
                # recognizes the keys. (The ``language_model.layers.`` variant,
                # which omits the ``model.`` segment, first gets ``model.`` inserted.)
                if key.startswith("language_model.layers."):
                    key = key.replace(
                        "language_model.layers.", "language_model.model.layers.", 1
                    )
                if key.startswith("language_model."):
                    key = key.replace("language_model.", "", 1)
                lm_hf[key] = value
                continue

            # Projector: the two released checkpoints disagree -- Kimi-VL spells it
            # ``multi_modal_projector.linear_1/2``, the 1T Kimi-K2.5 spells it
            # ``mm_projector.proj.0/2``. Normalize to the latter (the map/save form).
            key = key.replace("multi_modal_projector.", "mm_projector.")
            key = key.replace("mm_projector.linear_1", "mm_projector.proj.0")
            key = key.replace("mm_projector.linear_2", "mm_projector.proj.2")

            if re.search(r"\.wqkv\.(weight|bias)$", key):
                # Split fused HF vision qkv -> separate wq/wk/wv.
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)
                kind = "weight" if key.endswith("weight") else "bias"
                if value.shape[0] % 3 != 0:
                    raise ValueError(
                        f"fused vision QKV '{key}' has first dim "
                        f"{value.shape[0]}, not divisible by 3 (q|k|v)."
                    )
                q, k, v = torch.chunk(value, 3, dim=0)
                base = f"vision_encoder.layers.{layer_num}.attn"
                vision[f"{base}.wq.{kind}"] = q
                vision[f"{base}.wk.{kind}"] = k
                vision[f"{base}.wv.{kind}"] = v
            elif key in self.vision_from_hf_map:
                new_key = self.vision_from_hf_map[key]
                if new_key == "vision_encoder.patch_embed.weight":
                    # HF Conv2d (out, C, kH, kW) -> Linear (out, C*kH*kW).
                    value = value.reshape(value.shape[0], -1)
                vision[new_key] = value
            else:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                if abstract_key in self.vision_from_hf_map:
                    # pyrefly: ignore [missing-attribute]
                    layer_num = re.search(r"\d+", key).group(0)
                    vision[
                        self.vision_from_hf_map[abstract_key].format(layer_num)
                    ] = value
                else:
                    unmapped.append(key)

        if unmapped:
            raise ValueError(
                f"KimiK25StateDictAdapter: {len(unmapped)} vision key(s) have no "
                f"mapping: {unmapped}. Add them to vision_from_hf_map, or filter "
                f"them above if they are disposable buffers."
            )

        # DeepSeekV3 handles the LM keys (incl. RoPE validation + experts).
        state_dict = super().from_hf(lm_hf)
        state_dict.update(vision)
        return state_dict

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        if self.vision_encoder is None:
            return super().to_hf(state_dict)

        to_hf_map = {v: k for k, v in self.vision_from_hf_map.items()}
        lm_titan: dict[str, Any] = {}
        hf_state_dict: dict[str, Any] = {}
        # Buffer separate vision q/k/v per layer to re-fuse into one HF tensor.
        vision_qkv: dict[tuple[str, str], dict[str, torch.Tensor]] = {}

        for key, value in state_dict.items():
            if not key.startswith("vision_encoder."):
                lm_titan[key] = value
            elif re.search(r"vision_encoder\.layers\.\d+\.attn\.w[qkv]\.", key):
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)
                proj = re.search(r"attn\.(w[qkv])\.(weight|bias)", key)
                # pyrefly: ignore [missing-attribute]
                which, kind = proj.group(1), proj.group(2)
                vision_qkv.setdefault((layer_num, kind), {})[which] = value
            elif "patch_embed.weight" in key:
                # Linear (out, C*kH*kW) -> HF Conv2d (out, C, kH, kW).
                hf_state_dict[to_hf_map[key]] = value.reshape(
                    -1, self.in_channels, self.patch_size, self.patch_size
                )
            elif "vision_encoder.layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", key).group(0)
                hf_state_dict[to_hf_map[abstract_key].format(layer_num)] = value
            else:
                hf_state_dict[to_hf_map[key]] = value

        # Fuse vision q/k/v -> single HF qkv tensor per (layer, weight|bias).
        for (layer_num, kind), parts in vision_qkv.items():
            fused = torch.cat([parts["wq"], parts["wk"], parts["wv"]], dim=0)
            hf_state_dict[
                f"vision_tower.encoder.blocks.{layer_num}.wqkv.{kind}"
            ] = fused

        # Re-add the ``language_model.`` nesting that super (DeepSeek-V3) drops.
        hf_state_dict.update(
            {f"language_model.{k}": v for k, v in super().to_hf(lm_titan).items()}
        )
        return hf_state_dict

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
State dict adapter for Qwen3.5.

Converts between HuggingFace Qwen3.5 checkpoint format and torchtitan format.

MoE expert weights require two transformations:
- **Transpose**: HF and TT use transposed layouts for grouped 3D expert weights.
  E.g. HF down_proj [E, hidden, dim] <-> TT w2 [E, dim, hidden].
- **Fuse/split gate_up_proj**: HF fuses gate_proj and up_proj into a single
  gate_up_proj [E, dim, 2*hidden_dim]. TT stores them separately as
  w1 [E, hidden_dim, dim] and w3 [E, hidden_dim, dim].

Other notable conversions:
- Conv3d patch embedding (HF) <-> Linear (TT) via weight reshape
- Vision block naming: HF `blocks` <-> TT `layers`
- Vision QKV: HF fused qkv <-> TT separate wq/wk/wv
- GatedDeltaNet QKV: HF fused in_proj_qkv <-> TT separate in_proj_q/k/v
- GatedDeltaNet Conv1d: HF fused conv1d <-> TT separate conv_q/k/v
"""

import re
from typing import Any

import torch

from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .model import Qwen35Model


class Qwen35StateDictAdapter(StateDictAdapter):
    def __init__(self, model_config: Qwen35Model.Config, hf_assets_path: str | None):
        super().__init__(model_config, hf_assets_path)
        self.model_config = model_config

        self.from_hf_map = {
            # ===== Language Model =====
            "model.language_model.embed_tokens.weight": "tok_embeddings.weight",
            # Full attention layers (self_attn.*)
            "model.language_model.layers.{}.self_attn.q_proj.weight": "layers.{}.attn.wq.weight",
            "model.language_model.layers.{}.self_attn.k_proj.weight": "layers.{}.attn.wk.weight",
            "model.language_model.layers.{}.self_attn.v_proj.weight": "layers.{}.attn.wv.weight",
            "model.language_model.layers.{}.self_attn.o_proj.weight": "layers.{}.attn.wo.weight",
            "model.language_model.layers.{}.self_attn.q_norm.weight": "layers.{}.attn.q_norm.weight",
            "model.language_model.layers.{}.self_attn.k_norm.weight": "layers.{}.attn.k_norm.weight",
            "model.language_model.layers.{}.self_attn.rotary_emb.inv_freq": None,
            # GatedDeltaNet layers (linear_attn.*)
            # QKV and Conv1d: HF fused → TT separate (handled in to_hf/from_hf)
            "model.language_model.layers.{}.linear_attn.in_proj_qkv.weight": None,
            "model.language_model.layers.{}.linear_attn.conv1d.weight": None,
            "model.language_model.layers.{}.linear_attn.in_proj_z.weight": "layers.{}.attn.in_proj_z.weight",
            "model.language_model.layers.{}.linear_attn.in_proj_a.weight": "layers.{}.attn.in_proj_a.weight",
            "model.language_model.layers.{}.linear_attn.in_proj_b.weight": "layers.{}.attn.in_proj_b.weight",
            "model.language_model.layers.{}.linear_attn.A_log": "layers.{}.attn.A_log",
            "model.language_model.layers.{}.linear_attn.dt_bias": "layers.{}.attn.dt_bias",
            "model.language_model.layers.{}.linear_attn.norm.weight": "layers.{}.attn.norm.weight",
            "model.language_model.layers.{}.linear_attn.out_proj.weight": "layers.{}.attn.out_proj.weight",
            # Non-MoE MLP
            "model.language_model.layers.{}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            "model.language_model.layers.{}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            "model.language_model.layers.{}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            # Layer norms
            "model.language_model.layers.{}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            "model.language_model.layers.{}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            # MoE (grouped 3D format, handled specially in to_hf/from_hf)
            "model.language_model.layers.{}.mlp.experts.down_proj": "layers.{}.moe.experts.w2_EDF",
            "model.language_model.layers.{}.mlp.gate.weight": "layers.{}.moe.router.gate.weight",
            # MoE shared expert
            "model.language_model.layers.{}.mlp.shared_expert.gate_proj.weight": "layers.{}.moe.shared_experts.w1.weight",
            "model.language_model.layers.{}.mlp.shared_expert.up_proj.weight": "layers.{}.moe.shared_experts.w3.weight",
            "model.language_model.layers.{}.mlp.shared_expert.down_proj.weight": "layers.{}.moe.shared_experts.w2.weight",
            "model.language_model.layers.{}.mlp.shared_expert_gate.weight": "layers.{}.moe.shared_experts.gate.weight",
            # Final norm and output
            "model.language_model.norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
            # ===== Vision Encoder =====
            # Patch embedding (Conv3d in HF, Linear in TT — weight reshape needed)
            "model.visual.patch_embed.proj.weight": "vision_encoder.patch_embed.weight",
            "model.visual.patch_embed.proj.bias": "vision_encoder.patch_embed.bias",
            # Position embeddings
            "model.visual.pos_embed.weight": "vision_encoder.pos_embed",
            # Vision transformer blocks (HF: blocks, TT: layers)
            "model.visual.blocks.{}.norm1.weight": "vision_encoder.layers.{}.norm1.weight",
            "model.visual.blocks.{}.norm1.bias": "vision_encoder.layers.{}.norm1.bias",
            "model.visual.blocks.{}.norm2.weight": "vision_encoder.layers.{}.norm2.weight",
            "model.visual.blocks.{}.norm2.bias": "vision_encoder.layers.{}.norm2.bias",
            # Vision QKV: HF fused → TT separate (handled in to_hf/from_hf)
            "model.visual.blocks.{}.attn.qkv.weight": None,
            "model.visual.blocks.{}.attn.qkv.bias": None,
            "model.visual.blocks.{}.attn.proj.weight": "vision_encoder.layers.{}.attn.proj.weight",
            "model.visual.blocks.{}.attn.proj.bias": "vision_encoder.layers.{}.attn.proj.bias",
            "model.visual.blocks.{}.mlp.linear_fc1.weight": "vision_encoder.layers.{}.mlp.linear_fc1.weight",
            "model.visual.blocks.{}.mlp.linear_fc1.bias": "vision_encoder.layers.{}.mlp.linear_fc1.bias",
            "model.visual.blocks.{}.mlp.linear_fc2.weight": "vision_encoder.layers.{}.mlp.linear_fc2.weight",
            "model.visual.blocks.{}.mlp.linear_fc2.bias": "vision_encoder.layers.{}.mlp.linear_fc2.bias",
            # Merger
            "model.visual.merger.norm.weight": "vision_encoder.merger.norm.weight",
            "model.visual.merger.norm.bias": "vision_encoder.merger.norm.bias",
            "model.visual.merger.linear_fc1.weight": "vision_encoder.merger.linear_fc1.weight",
            "model.visual.merger.linear_fc1.bias": "vision_encoder.merger.linear_fc1.bias",
            "model.visual.merger.linear_fc2.weight": "vision_encoder.merger.linear_fc2.weight",
            "model.visual.merger.linear_fc2.bias": "vision_encoder.merger.linear_fc2.bias",
        }

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert torchtitan state dict to HuggingFace Qwen3.5 format."""
        to_hf_map = {v: k for k, v in self.from_hf_map.items() if v is not None}
        hf_state_dict = {}

        moe_w1_by_layer: dict[str, Any] = {}
        moe_w3_by_layer: dict[str, Any] = {}
        vision_qkv_by_layer: dict[str, dict[str, Any]] = {}
        deltanet_qkv_by_layer: dict[str, dict[str, Any]] = {}

        for tt_key, value in state_dict.items():
            if "moe.experts" in tt_key:
                tt_abstract_key = re.sub(r"(\d+)", "{}", tt_key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", tt_key).group(0)

                if tt_abstract_key == "layers.{}.moe.experts.w1_EFD":
                    moe_w1_by_layer[layer_num] = value
                    continue
                elif tt_abstract_key == "layers.{}.moe.experts.w3_EFD":
                    moe_w3_by_layer[layer_num] = value
                    continue
                elif tt_abstract_key == "layers.{}.moe.experts.w2_EDF":
                    hf_key = (
                        f"model.language_model.layers.{layer_num}.mlp.experts.down_proj"
                    )
                    hf_state_dict[hf_key] = value.transpose(-2, -1)
                    continue

                if tt_abstract_key not in to_hf_map:
                    continue
                hf_state_dict[to_hf_map[tt_abstract_key].format(layer_num)] = value

            elif re.search(r"\.\d+\.", tt_key):
                tt_abstract_key = re.sub(r"(\d+)", "{}", tt_key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", tt_key).group(0)

                # Collect deltanet q/k/v projections and conv weights for fusing
                if tt_abstract_key in (
                    "layers.{}.attn.in_proj_q.weight",
                    "layers.{}.attn.in_proj_k.weight",
                    "layers.{}.attn.in_proj_v.weight",
                    "layers.{}.attn.conv_q.weight",
                    "layers.{}.attn.conv_k.weight",
                    "layers.{}.attn.conv_v.weight",
                ):
                    if layer_num not in deltanet_qkv_by_layer:
                        deltanet_qkv_by_layer[layer_num] = {}
                    short_key = tt_abstract_key.split("attn.")[-1].replace("{}", "")
                    deltanet_qkv_by_layer[layer_num][short_key] = value
                    continue

                # Collect vision wq/wk/wv for fusing into qkv
                if tt_abstract_key in (
                    "vision_encoder.layers.{}.attn.wq.weight",
                    "vision_encoder.layers.{}.attn.wq.bias",
                    "vision_encoder.layers.{}.attn.wk.weight",
                    "vision_encoder.layers.{}.attn.wk.bias",
                    "vision_encoder.layers.{}.attn.wv.weight",
                    "vision_encoder.layers.{}.attn.wv.bias",
                ):
                    if layer_num not in vision_qkv_by_layer:
                        vision_qkv_by_layer[layer_num] = {}
                    short_key = tt_abstract_key.split("attn.")[-1].replace("{}", "")
                    vision_qkv_by_layer[layer_num][short_key] = value
                    continue

                if tt_abstract_key not in to_hf_map:
                    continue
                hf_state_dict[to_hf_map[tt_abstract_key].format(layer_num)] = value

            else:
                if tt_key not in to_hf_map:
                    continue
                if tt_key == "lm_head.weight" and getattr(
                    self.model_config, "enable_weight_tying", False
                ):
                    continue
                hf_value = value
                # Linear weight (out, C*T*H*W) → Conv3d weight (out, C, T, H, W)
                if tt_key == "vision_encoder.patch_embed.weight":
                    # pyrefly: ignore [missing-attribute]
                    encoder = self.model_config.vision_encoder
                    hf_value = value.reshape(
                        value.shape[0],
                        encoder.in_channels,
                        encoder.temporal_patch_size,
                        encoder.patch_size,
                        encoder.patch_size,
                    )
                hf_state_dict[to_hf_map[tt_key]] = hf_value

        # Fuse MoE w1 (gate) + w3 (up) → gate_up_proj
        for layer_num in moe_w1_by_layer:
            w1 = moe_w1_by_layer[layer_num].transpose(-2, -1)
            w3 = moe_w3_by_layer[layer_num].transpose(-2, -1)
            hf_state_dict[
                f"model.language_model.layers.{layer_num}.mlp.experts.gate_up_proj"
            ] = torch.cat([w1, w3], dim=-1)

        # Fuse vision wq/wk/wv → qkv
        for layer_num, parts in vision_qkv_by_layer.items():
            for suffix in ("weight", "bias"):
                q = parts.get(f"wq.{suffix}")
                k = parts.get(f"wk.{suffix}")
                v = parts.get(f"wv.{suffix}")
                if q is not None and k is not None and v is not None:
                    hf_state_dict[
                        f"model.visual.blocks.{layer_num}.attn.qkv.{suffix}"
                    ] = torch.cat([q, k, v], dim=0)

        # Fuse deltanet in_proj_q/k/v → in_proj_qkv, conv_q/k/v → conv1d
        for layer_num, parts in deltanet_qkv_by_layer.items():
            q = parts.get("in_proj_q.weight")
            k = parts.get("in_proj_k.weight")
            v = parts.get("in_proj_v.weight")
            if q is not None and k is not None and v is not None:
                hf_state_dict[
                    f"model.language_model.layers.{layer_num}.linear_attn.in_proj_qkv.weight"
                ] = torch.cat([q, k, v], dim=0)
            cq = parts.get("conv_q.weight")
            ck = parts.get("conv_k.weight")
            cv = parts.get("conv_v.weight")
            if cq is not None and ck is not None and cv is not None:
                hf_state_dict[
                    f"model.language_model.layers.{layer_num}.linear_attn.conv1d.weight"
                ] = torch.cat([cq, ck, cv], dim=0)

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert HuggingFace Qwen3.5 state dict to torchtitan format."""
        tt_state_dict = {}

        # HF ties lm_head with embed_tokens — copy if missing
        if "lm_head.weight" not in hf_state_dict:
            embed_key = "model.language_model.embed_tokens.weight"
            if embed_key not in hf_state_dict:
                raise ValueError(
                    f"HF checkpoint missing both 'lm_head.weight' and '{embed_key}'"
                )
            hf_state_dict["lm_head.weight"] = hf_state_dict[embed_key]

        for hf_key, value in hf_state_dict.items():
            if re.search(r"\.\d+\.", hf_key):
                hf_abstract_key = re.sub(r"(\d+)", "{}", hf_key, count=1)
                # pyrefly: ignore [missing-attribute]
                idx = re.search(r"\d+", hf_key).group(0)

                # MoE gate_up_proj → split into w1 + w3 and transpose
                if (
                    hf_abstract_key
                    == "model.language_model.layers.{}.mlp.experts.gate_up_proj"
                ):
                    w1_hf, w3_hf = value.chunk(2, dim=-1)
                    tt_state_dict[f"layers.{idx}.moe.experts.w1_EFD"] = w1_hf.transpose(
                        -2, -1
                    )
                    tt_state_dict[f"layers.{idx}.moe.experts.w3_EFD"] = w3_hf.transpose(
                        -2, -1
                    )
                    continue

                # MoE down_proj → transpose
                if (
                    hf_abstract_key
                    == "model.language_model.layers.{}.mlp.experts.down_proj"
                ):
                    tt_state_dict[f"layers.{idx}.moe.experts.w2_EDF"] = value.transpose(
                        -2, -1
                    )
                    continue

                # GatedDeltaNet fused in_proj_qkv → split into q/k/v
                if (
                    hf_abstract_key
                    == "model.language_model.layers.{}.linear_attn.in_proj_qkv.weight"
                ):
                    # pyrefly: ignore [missing-attribute]
                    dn = self.model_config.layers[int(idx)].delta_net
                    kd = dn.in_proj_q.out_features
                    vd = dn.in_proj_v.out_features
                    q, k, v = value.split([kd, kd, vd], dim=0)
                    tt_state_dict[f"layers.{idx}.attn.in_proj_q.weight"] = q
                    tt_state_dict[f"layers.{idx}.attn.in_proj_k.weight"] = k
                    tt_state_dict[f"layers.{idx}.attn.in_proj_v.weight"] = v
                    continue

                # GatedDeltaNet fused conv1d → split into conv_q/k/v
                if (
                    hf_abstract_key
                    == "model.language_model.layers.{}.linear_attn.conv1d.weight"
                ):
                    # pyrefly: ignore [missing-attribute]
                    dn = self.model_config.layers[int(idx)].delta_net
                    kd = dn.in_proj_q.out_features
                    vd = dn.in_proj_v.out_features
                    cq, ck, cv = value.split([kd, kd, vd], dim=0)
                    tt_state_dict[f"layers.{idx}.attn.conv_q.weight"] = cq
                    tt_state_dict[f"layers.{idx}.attn.conv_k.weight"] = ck
                    tt_state_dict[f"layers.{idx}.attn.conv_v.weight"] = cv
                    continue

                # Vision fused QKV → split into wq/wk/wv
                if hf_abstract_key in (
                    "model.visual.blocks.{}.attn.qkv.weight",
                    "model.visual.blocks.{}.attn.qkv.bias",
                ):
                    suffix = "weight" if "weight" in hf_abstract_key else "bias"
                    q, k, v = value.chunk(3, dim=0)
                    tt_state_dict[f"vision_encoder.layers.{idx}.attn.wq.{suffix}"] = q
                    tt_state_dict[f"vision_encoder.layers.{idx}.attn.wk.{suffix}"] = k
                    tt_state_dict[f"vision_encoder.layers.{idx}.attn.wv.{suffix}"] = v
                    continue

                if hf_abstract_key not in self.from_hf_map:
                    continue
                tt_key = self.from_hf_map[hf_abstract_key]
                if tt_key is None:
                    continue
                tt_state_dict[tt_key.format(idx)] = value

            else:
                if hf_key not in self.from_hf_map:
                    continue
                tt_key = self.from_hf_map[hf_key]
                if tt_key is None:
                    continue
                tt_value = value
                # Conv3d weight (out, C, T, H, W) → Linear weight (out, C*T*H*W)
                if hf_key == "model.visual.patch_embed.proj.weight":
                    tt_value = value.reshape(value.shape[0], -1)
                tt_state_dict[tt_key] = tt_value

        return tt_state_dict

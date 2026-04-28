# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
State dict adapter for Qwen3-VL.

Converts between HuggingFace Qwen3-VL checkpoint format and torchtitan format.

MoE expert weights require two transformations:
- **Transpose**: HF and TT use transposed layouts for grouped 3D expert weights.
  E.g. HF down_proj [E, hidden, dim] <-> TT w2 [E, dim, hidden].
- **Fuse/split gate_up_proj**: HF fuses gate_proj and up_proj into a single
  gate_up_proj [E, dim, 2*hidden_dim]. TT stores them separately as
  w1 [E, hidden_dim, dim] and w3 [E, hidden_dim, dim].

Other notable conversions:
- Conv3d patch embedding (HF) <-> Linear (TT) via weight reshape
- Vision block naming: HF `blocks` <-> TT `layers`
"""

import re
from typing import Any

import torch

from torchtitan.models.common.attention import FusedQKVLinear
from torchtitan.protocols.state_dict_adapter import StateDictAdapter

from .model import Qwen3VLModel


class Qwen3VLStateDictAdapter(StateDictAdapter):
    def __init__(self, model_config: Qwen3VLModel.Config, hf_assets_path: str | None):
        super().__init__(model_config, hf_assets_path)
        self.model_config = model_config
        self.fuse_qkv = isinstance(
            model_config.layers[0].attention.qkv_linear, FusedQKVLinear.Config
        )

        if self.fuse_qkv:
            qkv_map = {
                "model.language_model.layers.{}.self_attn.q_proj.weight": None,
                "model.language_model.layers.{}.self_attn.k_proj.weight": None,
                "model.language_model.layers.{}.self_attn.v_proj.weight": None,
            }
        else:
            qkv_map = {
                "model.language_model.layers.{}.self_attn.q_proj.weight": "layers.{}.attention.qkv_linear.wq.weight",
                "model.language_model.layers.{}.self_attn.k_proj.weight": "layers.{}.attention.qkv_linear.wk.weight",
                "model.language_model.layers.{}.self_attn.v_proj.weight": "layers.{}.attention.qkv_linear.wv.weight",
            }

        self.from_hf_map = {
            # ===== Language Model =====
            "model.language_model.embed_tokens.weight": "tok_embeddings.weight",
            # Attention
            **qkv_map,
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
            # MoE (grouped 3D format, handled specially in to_hf/from_hf)
            # gate_up_proj is fused gate+up, mapped to w1+w3 via custom logic
            "model.language_model.layers.{}.mlp.experts.down_proj": "layers.{}.moe.experts.w2",
            "model.language_model.layers.{}.mlp.gate.weight": "layers.{}.moe.router.gate.weight",
            # Final norm and output
            "model.language_model.norm.weight": "norm.weight",
            "lm_head.weight": "lm_head.weight",
            # ===== Vision Encoder =====
            # Patch embedding (Conv3d in HF, Linear in TT - weight reshape needed)
            "model.visual.patch_embed.proj.weight": "vision_encoder.patch_embed.proj.weight",
            "model.visual.patch_embed.proj.bias": "vision_encoder.patch_embed.proj.bias",
            # Position embeddings
            "model.visual.pos_embed.weight": "vision_encoder.pos_embed",
            # Vision transformer blocks (HF: blocks, TT: layers)
            "model.visual.blocks.{}.norm1.weight": "vision_encoder.layers.{}.norm1.weight",
            "model.visual.blocks.{}.norm1.bias": "vision_encoder.layers.{}.norm1.bias",
            "model.visual.blocks.{}.norm2.weight": "vision_encoder.layers.{}.norm2.weight",
            "model.visual.blocks.{}.norm2.bias": "vision_encoder.layers.{}.norm2.bias",
            "model.visual.blocks.{}.attn.qkv.weight": "vision_encoder.layers.{}.attn.qkv.weight",
            "model.visual.blocks.{}.attn.qkv.bias": "vision_encoder.layers.{}.attn.qkv.bias",
            "model.visual.blocks.{}.attn.proj.weight": "vision_encoder.layers.{}.attn.proj.weight",
            "model.visual.blocks.{}.attn.proj.bias": "vision_encoder.layers.{}.attn.proj.bias",
            "model.visual.blocks.{}.mlp.linear_fc1.weight": "vision_encoder.layers.{}.mlp.linear_fc1.weight",
            "model.visual.blocks.{}.mlp.linear_fc1.bias": "vision_encoder.layers.{}.mlp.linear_fc1.bias",
            "model.visual.blocks.{}.mlp.linear_fc2.weight": "vision_encoder.layers.{}.mlp.linear_fc2.weight",
            "model.visual.blocks.{}.mlp.linear_fc2.bias": "vision_encoder.layers.{}.mlp.linear_fc2.bias",
            # Merger (maps vision dim to LLM dim)
            "model.visual.merger.norm.weight": "vision_encoder.merger.norm.weight",
            "model.visual.merger.norm.bias": "vision_encoder.merger.norm.bias",
            "model.visual.merger.linear_fc1.weight": "vision_encoder.merger.linear_fc1.weight",
            "model.visual.merger.linear_fc1.bias": "vision_encoder.merger.linear_fc1.bias",
            "model.visual.merger.linear_fc2.weight": "vision_encoder.merger.linear_fc2.weight",
            "model.visual.merger.linear_fc2.bias": "vision_encoder.merger.linear_fc2.bias",
            # DeepStack mergers
            "model.visual.deepstack_merger_list.{}.norm.weight": "vision_encoder.deepstack_merger_list.{}.norm.weight",
            "model.visual.deepstack_merger_list.{}.norm.bias": "vision_encoder.deepstack_merger_list.{}.norm.bias",
            "model.visual.deepstack_merger_list.{}.linear_fc1.weight": "vision_encoder.deepstack_merger_list.{}.linear_fc1.weight",
            "model.visual.deepstack_merger_list.{}.linear_fc1.bias": "vision_encoder.deepstack_merger_list.{}.linear_fc1.bias",
            "model.visual.deepstack_merger_list.{}.linear_fc2.weight": "vision_encoder.deepstack_merger_list.{}.linear_fc2.weight",
            "model.visual.deepstack_merger_list.{}.linear_fc2.bias": "vision_encoder.deepstack_merger_list.{}.linear_fc2.bias",
        }

    def _get_attention_dims(self) -> tuple[int, int, int]:
        """Return (n_heads, n_kv_heads, head_dim) from model config."""
        attn = self.model_config.layers[0].attention
        n_heads = attn.n_heads
        n_kv_heads = attn.n_kv_heads if attn.n_kv_heads is not None else n_heads
        head_dim = (
            attn.head_dim
            if attn.head_dim is not None
            else self.model_config.dim // n_heads
        )
        return n_heads, n_kv_heads, head_dim

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert torchtitan state dict to HuggingFace Qwen3-VL format."""
        if self.fuse_qkv:
            to_hf_map = {v: k for k, v in self.from_hf_map.items() if v is not None}
            n_heads, n_kv_heads, head_dim = self._get_attention_dims()
        else:
            to_hf_map = {v: k for k, v in self.from_hf_map.items() if v is not None}
        hf_state_dict = {}

        # Collect MoE w1/w3 per layer to fuse into gate_up_proj
        moe_w1_by_layer: dict[str, Any] = {}
        moe_w3_by_layer: dict[str, Any] = {}

        for tt_key, value in state_dict.items():
            if "moe.experts" in tt_key:
                tt_abstract_key = re.sub(r"(\d+)", "{}", tt_key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", tt_key).group(0)

                # Collect w1/w3 for fusing into gate_up_proj
                if tt_abstract_key == "layers.{}.moe.experts.w1":
                    moe_w1_by_layer[layer_num] = value
                    continue
                elif tt_abstract_key == "layers.{}.moe.experts.w3":
                    moe_w3_by_layer[layer_num] = value
                    continue

                # Handle down_proj: TT w2 [E, dim, hidden] -> HF [E, hidden, dim]
                if tt_abstract_key == "layers.{}.moe.experts.w2":
                    hf_key = (
                        f"model.language_model.layers.{layer_num}.mlp.experts.down_proj"
                    )
                    hf_state_dict[hf_key] = value.transpose(-2, -1)
                    continue

                if tt_abstract_key not in to_hf_map:
                    continue
                hf_key = to_hf_map[tt_abstract_key].format(layer_num)
                hf_state_dict[hf_key] = value
            # Indexed key: contains a layer/block/merger index (e.g. ".0.")
            elif re.search(r"\.\d+\.", tt_key):
                tt_abstract_key = re.sub(r"(\d+)", "{}", tt_key, count=1)
                # pyrefly: ignore [missing-attribute]
                layer_num = re.search(r"\d+", tt_key).group(0)

                # Handle fused QKV: split wqkv into separate q/k/v projections
                if (
                    self.fuse_qkv
                    and tt_abstract_key == "layers.{}.attention.qkv_linear.wqkv.weight"
                ):
                    wq, wk, wv = self.fused_to_separate_qkv(
                        value,
                        n_heads,  # pyrefly: ignore [unbound-name]
                        n_kv_heads,  # pyrefly: ignore [unbound-name]
                        head_dim,  # pyrefly: ignore [unbound-name]
                    )
                    hf_state_dict[
                        f"model.language_model.layers.{layer_num}.self_attn.q_proj.weight"
                    ] = wq
                    hf_state_dict[
                        f"model.language_model.layers.{layer_num}.self_attn.k_proj.weight"
                    ] = wk
                    hf_state_dict[
                        f"model.language_model.layers.{layer_num}.self_attn.v_proj.weight"
                    ] = wv
                    continue

                if tt_abstract_key not in to_hf_map:
                    continue
                hf_key = to_hf_map[tt_abstract_key].format(layer_num)
                hf_state_dict[hf_key] = value

            else:
                if tt_key not in to_hf_map:
                    continue
                if tt_key == "lm_head.weight" and self.model_config.enable_weight_tying:
                    continue
                hf_key = to_hf_map[tt_key]
                hf_value = value
                # Linear weight (out, C*T*H*W) -> Conv3d weight (out, C, T, H, W)
                # Plain reshape since both use channel-first (c pt ph pw) layout.
                if tt_key == "vision_encoder.patch_embed.proj.weight":
                    encoder = self.model_config.vision_encoder
                    hf_value = value.reshape(
                        value.shape[0],
                        encoder.in_channels,
                        encoder.temporal_patch_size,
                        encoder.patch_size,
                        encoder.patch_size,
                    )
                hf_state_dict[hf_key] = hf_value

        # Fuse w1 (gate) and w3 (up) into gate_up_proj per layer
        # TT w1/w3: [E, hidden_dim, dim] -> transpose to [E, dim, hidden_dim] -> cat on last dim
        for layer_num in moe_w1_by_layer:
            w1 = moe_w1_by_layer[layer_num].transpose(-2, -1)  # [E, dim, hidden_dim]
            w3 = moe_w3_by_layer[layer_num].transpose(-2, -1)  # [E, dim, hidden_dim]
            gate_up = torch.cat([w1, w3], dim=-1)  # [E, dim, 2*hidden_dim]
            hf_key = f"model.language_model.layers.{layer_num}.mlp.experts.gate_up_proj"
            hf_state_dict[hf_key] = gate_up

        return hf_state_dict

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert HuggingFace Qwen3-VL state dict to torchtitan format."""
        tt_state_dict = {}
        # Collect Q/K/V per layer for fusing (only used when fuse_qkv=True)
        pending_qkv: dict[str, dict[str, torch.Tensor]] = {}

        if self.fuse_qkv:
            n_heads, n_kv_heads, head_dim = self._get_attention_dims()

        # HF Qwen3-VL ties lm_head.weight with embed_tokens.weight,
        # so lm_head.weight may not be stored in the checkpoint.
        if "lm_head.weight" not in hf_state_dict:
            if "model.language_model.embed_tokens.weight" not in hf_state_dict:
                raise ValueError(
                    "HF checkpoint missing both 'lm_head.weight' and "
                    "'model.language_model.embed_tokens.weight'"
                )
            hf_state_dict["lm_head.weight"] = hf_state_dict[
                "model.language_model.embed_tokens.weight"
            ]

        for hf_key, value in hf_state_dict.items():
            # Indexed key: contains a layer/block/merger index (e.g. ".0.")
            if re.search(r"\.\d+\.", hf_key):
                hf_abstract_key = re.sub(r"(\d+)", "{}", hf_key, count=1)
                # pyrefly: ignore [missing-attribute]
                idx = re.search(r"\d+", hf_key).group(0)

                # Handle fused QKV: collect q/k/v and fuse when all 3 are ready
                if self.fuse_qkv and hf_abstract_key in (
                    "model.language_model.layers.{}.self_attn.q_proj.weight",
                    "model.language_model.layers.{}.self_attn.k_proj.weight",
                    "model.language_model.layers.{}.self_attn.v_proj.weight",
                ):
                    if idx not in pending_qkv:
                        pending_qkv[idx] = {}
                    proj = hf_abstract_key.split(".")[-2]  # q_proj, k_proj, v_proj
                    pending_qkv[idx][proj] = value
                    if len(pending_qkv[idx]) == 3:
                        fused = self.separate_to_fused_qkv(
                            pending_qkv[idx]["q_proj"],
                            pending_qkv[idx]["k_proj"],
                            pending_qkv[idx]["v_proj"],
                            n_heads,  # pyrefly: ignore [unbound-name]
                            n_kv_heads,  # pyrefly: ignore [unbound-name]
                            head_dim,  # pyrefly: ignore [unbound-name]
                        )
                        tt_state_dict[
                            f"layers.{idx}.attention.qkv_linear.wqkv.weight"
                        ] = fused
                        del pending_qkv[idx]
                    continue

                # Handle fused gate_up_proj: split and transpose
                # HF gate_up_proj: [E, dim, 2*hidden_dim] -> split -> transpose each to [E, hidden_dim, dim]
                if (
                    hf_abstract_key
                    == "model.language_model.layers.{}.mlp.experts.gate_up_proj"
                ):
                    w1_hf, w3_hf = value.chunk(2, dim=-1)  # each [E, dim, hidden_dim]
                    tt_state_dict[f"layers.{idx}.moe.experts.w1"] = w1_hf.transpose(
                        -2, -1
                    )
                    tt_state_dict[f"layers.{idx}.moe.experts.w3"] = w3_hf.transpose(
                        -2, -1
                    )
                    continue

                # Handle down_proj transpose: HF [E, hidden, dim] -> TT w2 [E, dim, hidden]
                if (
                    hf_abstract_key
                    == "model.language_model.layers.{}.mlp.experts.down_proj"
                ):
                    tt_state_dict[f"layers.{idx}.moe.experts.w2"] = value.transpose(
                        -2, -1
                    )
                    continue

                if hf_abstract_key not in self.from_hf_map:
                    continue
                tt_key = self.from_hf_map[hf_abstract_key]
                if tt_key is None:
                    continue
                tt_key = tt_key.format(idx)
                tt_state_dict[tt_key] = value

            else:
                if hf_key not in self.from_hf_map:
                    continue
                tt_key = self.from_hf_map[hf_key]
                if tt_key is None:
                    continue
                tt_value = value
                # Conv3d weight (out, C, T, H, W) -> Linear weight (out, C*T*H*W)
                # Plain reshape since both use channel-first (c pt ph pw) layout.
                if hf_key == "model.visual.patch_embed.proj.weight":
                    tt_value = value.reshape(value.shape[0], -1)
                tt_state_dict[tt_key] = tt_value

        if self.fuse_qkv and pending_qkv:
            raise ValueError(
                f"Incomplete Q/K/V projections for layers: {list(pending_qkv.keys())}"
            )

        return tt_state_dict

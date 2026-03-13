# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
State dict adapter for Qwen3-Next / Qwen3.5 models.

Handles conversion between HuggingFace and TorchTitan checkpoint formats.
Supports two HF formats:
  - Qwen3-Next (linear_split_projections=False): fused projections, `model.` prefix
  - Qwen3.5 VLM (linear_split_projections=True): split projections, `model.language_model.` prefix,
    fused gate_up_proj for sparse experts
"""

import re
from typing import Any

import torch
from torch.distributed.tensor import DTensor
from torchtitan.models.utils import MoEStateDictAdapter

from .args import Qwen3NextModelArgs


class Qwen3NextStateDictAdapter(MoEStateDictAdapter):
    @staticmethod
    def _permute_qproj_hf_to_tt(weight: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
        """Permute q_proj weight from HF per-head interleaved layout to TT global split.

        HF: [h0_q(hd), h0_gate(hd), h1_q(hd), h1_gate(hd), ...] along dim 0
        TT: [h0_q(hd), h1_q(hd), ..., h0_gate(hd), h1_gate(hd), ...] along dim 0
        """
        # weight shape: [n_heads * 2 * head_dim, dim]
        w = weight.view(n_heads, 2, head_dim, -1)
        # w[:, 0, :, :] = all query parts, w[:, 1, :, :] = all gate parts
        q = w[:, 0, :, :].reshape(n_heads * head_dim, -1)
        g = w[:, 1, :, :].reshape(n_heads * head_dim, -1)
        return torch.cat([q, g], dim=0)

    @staticmethod
    def _permute_qproj_tt_to_hf(weight: torch.Tensor, n_heads: int, head_dim: int) -> torch.Tensor:
        """Permute q_proj weight from TT global split to HF per-head interleaved layout."""
        # weight shape: [n_heads * 2 * head_dim, dim]
        half = n_heads * head_dim
        q = weight[:half].view(n_heads, head_dim, -1)
        g = weight[half:].view(n_heads, head_dim, -1)
        # Interleave: [h0_q, h0_g, h1_q, h1_g, ...]
        return torch.stack([q, g], dim=1).reshape(n_heads * 2 * head_dim, -1)

    def __init__(self, model_args: Qwen3NextModelArgs, hf_assets_path: str | None):
        super().__init__(model_args, hf_assets_path)

        # Qwen3.5 VLM checkpoints use 'model.language_model.' prefix;
        # qwen3-next (if released standalone) would use 'model.'
        self.is_qwen35 = model_args.linear_split_projections
        pfx = "model.language_model" if self.is_qwen35 else "model"

        self.from_hf_map = {
            f"{pfx}.embed_tokens.weight": "tok_embeddings.weight",
            # Full Attention
            f"{pfx}.layers.{{}}.self_attn.q_proj.weight": "layers.{}.attention.wq.weight",
            f"{pfx}.layers.{{}}.self_attn.k_proj.weight": "layers.{}.attention.wk.weight",
            f"{pfx}.layers.{{}}.self_attn.v_proj.weight": "layers.{}.attention.wv.weight",
            f"{pfx}.layers.{{}}.self_attn.o_proj.weight": "layers.{}.attention.wo.weight",
            f"{pfx}.layers.{{}}.self_attn.q_norm.weight": "layers.{}.attention.q_norm.weight",
            f"{pfx}.layers.{{}}.self_attn.k_norm.weight": "layers.{}.attention.k_norm.weight",
            # Linear Attention - common keys
            f"{pfx}.layers.{{}}.linear_attn.conv1d.weight": "layers.{}.attention.conv1d.weight",
            f"{pfx}.layers.{{}}.linear_attn.dt_bias": "layers.{}.attention.dt_bias",
            f"{pfx}.layers.{{}}.linear_attn.A_log": "layers.{}.attention.A_log",
            f"{pfx}.layers.{{}}.linear_attn.norm.weight": "layers.{}.attention.norm.weight",
            f"{pfx}.layers.{{}}.linear_attn.out_proj.weight": "layers.{}.attention.out_proj.weight",
            # Layer norms
            f"{pfx}.layers.{{}}.input_layernorm.weight": "layers.{}.attention_norm.weight",
            f"{pfx}.layers.{{}}.post_attention_layernorm.weight": "layers.{}.ffn_norm.weight",
            # MoE - router and shared gate
            f"{pfx}.layers.{{}}.mlp.gate.weight": "layers.{}.moe.router.gate.weight",
            f"{pfx}.layers.{{}}.mlp.shared_expert_gate.weight": "layers.{}.moe.shared_gate.weight",
            # Shared expert (FeedForward with nn.Linear, so .weight suffix on both sides)
            f"{pfx}.layers.{{}}.mlp.shared_expert.gate_proj.weight": "layers.{}.moe.shared_experts.w1.weight",
            f"{pfx}.layers.{{}}.mlp.shared_expert.up_proj.weight": "layers.{}.moe.shared_experts.w3.weight",
            f"{pfx}.layers.{{}}.mlp.shared_expert.down_proj.weight": "layers.{}.moe.shared_experts.w2.weight",
            # Non-MoE FFN (for variants without MoE on every layer)
            f"{pfx}.layers.{{}}.mlp.gate_proj.weight": "layers.{}.feed_forward.w1.weight",
            f"{pfx}.layers.{{}}.mlp.up_proj.weight": "layers.{}.feed_forward.w3.weight",
            f"{pfx}.layers.{{}}.mlp.down_proj.weight": "layers.{}.feed_forward.w2.weight",
            # Final
            f"{pfx}.norm.weight": "norm.weight",
            "lm_head.weight": "output.weight",
        }

        # Linear attention projection keys depend on format
        if model_args.linear_split_projections:
            self.from_hf_map.update({
                f"{pfx}.layers.{{}}.linear_attn.in_proj_qkv.weight": "layers.{}.attention.in_proj_qkv.weight",
                f"{pfx}.layers.{{}}.linear_attn.in_proj_z.weight": "layers.{}.attention.in_proj_z.weight",
                f"{pfx}.layers.{{}}.linear_attn.in_proj_b.weight": "layers.{}.attention.in_proj_b.weight",
                f"{pfx}.layers.{{}}.linear_attn.in_proj_a.weight": "layers.{}.attention.in_proj_a.weight",
            })
        else:
            self.from_hf_map.update({
                f"{pfx}.layers.{{}}.linear_attn.in_proj_qkvz.weight": "layers.{}.attention.in_proj_qkvz.weight",
                f"{pfx}.layers.{{}}.linear_attn.in_proj_ba.weight": "layers.{}.attention.in_proj_ba.weight",
            })

        # For Qwen3-Next (non-3.5), sparse experts use per-expert indexed keys
        if not self.is_qwen35:
            self.from_hf_map.update({
                f"{pfx}.layers.{{}}.mlp.experts.{{}}.gate_proj.weight": "layers.{}.moe.experts.w1",
                f"{pfx}.layers.{{}}.mlp.experts.{{}}.up_proj.weight": "layers.{}.moe.experts.w3",
                f"{pfx}.layers.{{}}.mlp.experts.{{}}.down_proj.weight": "layers.{}.moe.experts.w2",
            })

        self.pfx = pfx

    def to_hf(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert TorchTitan state dict to HF format."""
        if not self.is_qwen35:
            return self._to_hf_qwen3next(state_dict)
        return self._to_hf_qwen35(state_dict)

    def from_hf(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        """Convert HF state dict to TorchTitan format."""
        if not self.is_qwen35:
            return self._from_hf_qwen3next(hf_state_dict)
        return self._from_hf_qwen35(hf_state_dict)

    # ---- Qwen3.5 format (stacked experts with fused gate_up_proj) ----

    def _to_hf_qwen35(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        to_hf_map = {v: k for k, v in self.from_hf_map.items()}
        hf_state_dict = {}
        n_heads = self.model_args.n_heads
        head_dim = self.model_args.head_dim
        # Collect w1/w3 per layer for fusing into gate_up_proj
        pending_experts: dict[str, dict[str, Any]] = {}

        for key, value in state_dict.items():
            if "moe.experts" in key and "shared" not in key:
                layer_num = re.search(r"\d+", key).group(0)
                if layer_num not in pending_experts:
                    pending_experts[layer_num] = {}

                if key.endswith(".w1"):
                    pending_experts[layer_num]["w1"] = value
                elif key.endswith(".w3"):
                    pending_experts[layer_num]["w3"] = value
                elif key.endswith(".w2"):
                    hf_key = f"{self.pfx}.layers.{layer_num}.mlp.experts.down_proj"
                    hf_state_dict[hf_key] = value

            elif "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                if abstract_key not in to_hf_map:
                    continue
                layer_num = re.search(r"\d+", key).group(0)
                hf_key = to_hf_map[abstract_key].format(layer_num)
                # Permute q_proj from TT global split to HF interleaved
                if "attention.wq.weight" in key:
                    value = self._permute_qproj_tt_to_hf(value, n_heads, head_dim)
                hf_state_dict[hf_key] = value

            else:
                if key not in to_hf_map:
                    continue
                hf_state_dict[to_hf_map[key]] = value

        # Fuse w1 (gate) + w3 (up) → gate_up_proj
        for layer_num, experts in pending_experts.items():
            if "w1" in experts and "w3" in experts:
                gate_up = torch.cat([experts["w1"], experts["w3"]], dim=1)
                hf_key = f"{self.pfx}.layers.{layer_num}.mlp.experts.gate_up_proj"
                hf_state_dict[hf_key] = gate_up

        return hf_state_dict

    def _from_hf_qwen35(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict = {}
        n_heads = self.model_args.n_heads
        head_dim = self.model_args.head_dim

        for key, value in hf_state_dict.items():
            # Skip vision and MTP keys
            if "visual" in key or key.startswith("mtp."):
                continue

            if "mlp.experts.gate_up_proj" in key:
                # Fused gate_up_proj → w1 (gate) + w3 (up)
                layer_num = re.search(r"\d+", key).group(0)
                w1, w3 = torch.chunk(value, 2, dim=1)
                state_dict[f"layers.{layer_num}.moe.experts.w1"] = w1.contiguous()
                state_dict[f"layers.{layer_num}.moe.experts.w3"] = w3.contiguous()

            elif "mlp.experts.down_proj" in key:
                layer_num = re.search(r"\d+", key).group(0)
                state_dict[f"layers.{layer_num}.moe.experts.w2"] = value

            elif "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                if abstract_key not in self.from_hf_map:
                    continue
                layer_num = re.search(r"\d+", key).group(0)
                tt_key = self.from_hf_map[abstract_key].format(layer_num)
                # Permute q_proj from HF interleaved to TT global split
                if "q_proj.weight" in key:
                    value = self._permute_qproj_hf_to_tt(value, n_heads, head_dim)
                state_dict[tt_key] = value

            else:
                if key not in self.from_hf_map:
                    continue
                state_dict[self.from_hf_map[key]] = value

        return state_dict

    # ---- Qwen3-Next format (per-expert indexed keys) ----

    def _to_hf_qwen3next(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        to_hf_map = {v: k for k, v in self.from_hf_map.items()}
        hf_state_dict = {}

        for key, value in state_dict.items():
            if "moe.experts" in key or "moe.shared_experts" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                if abstract_key not in to_hf_map:
                    continue
                layer_num = re.search(r"\d+", key).group(0)
                new_abstract_key = to_hf_map[abstract_key]

                if isinstance(value, DTensor):
                    self.grouped_expert_weight_placements[
                        abstract_key
                    ] = value.placements
                    self.grouped_expert_weight_shape[abstract_key] = value.shape
                    local_expert_fqn = self._get_local_experts_weights(
                        new_abstract_key, abstract_key, layer_num, value,
                    )
                    hf_state_dict.update(local_expert_fqn)
                else:
                    n_experts = (
                        self.model_args.moe_args.num_experts
                        if "shared" not in key
                        else self.model_args.moe_args.num_shared_experts
                    )
                    split_values = self._split_experts_weights(value, n_experts)
                    for expert_num in range(n_experts):
                        new_key = new_abstract_key.format(layer_num, expert_num)
                        hf_state_dict[new_key] = split_values[expert_num].squeeze()

            elif "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                if abstract_key not in to_hf_map:
                    continue
                layer_num = re.search(r"\d+", key).group(0)
                hf_state_dict[to_hf_map[abstract_key].format(layer_num)] = value

            else:
                if key not in to_hf_map:
                    continue
                hf_state_dict[to_hf_map[key]] = value

        return hf_state_dict

    def _from_hf_qwen3next(self, hf_state_dict: dict[str, Any]) -> dict[str, Any]:
        state_dict = {}
        expert_weights_by_layer: dict[str, dict[str, dict]] = {}

        for key, value in hf_state_dict.items():
            if "mlp.experts." in key or "mlp.shared_expert." in key:
                abstract_key = re.sub(
                    r"(\d+)", "{}", key, count=2 if "experts" in key else 1
                )
                if "experts" in key:
                    layer_num, expert_num = re.findall(r"\d+", key)
                else:
                    layer_num = re.search(r"\d+", key).group(0)
                    expert_num = None
                titan_abstract_key = self.from_hf_map[abstract_key]
                new_key = titan_abstract_key.format(layer_num)

                if layer_num not in expert_weights_by_layer:
                    expert_weights_by_layer[layer_num] = {}
                if titan_abstract_key not in expert_weights_by_layer[layer_num]:
                    expert_weights_by_layer[layer_num][titan_abstract_key] = {}
                expert_weights_by_layer[layer_num][titan_abstract_key][
                    expert_num
                ] = value

                if isinstance(value, DTensor):
                    stacked_value = self._concatenate_expert_weights_dtensor(
                        expert_weights_by_layer,
                        titan_abstract_key,
                        layer_num,
                        value.device_mesh,
                    )
                else:
                    n_experts = (
                        self.model_args.moe_args.num_experts
                        if "experts" in key
                        else self.model_args.moe_args.num_shared_experts
                    )
                    stacked_value = self._concatenate_expert_weights(
                        expert_weights_by_layer,
                        titan_abstract_key,
                        layer_num,
                        n_experts,
                    )

                if stacked_value is not None:
                    state_dict[new_key] = stacked_value

            elif "layers" in key:
                abstract_key = re.sub(r"(\d+)", "{}", key, count=1)
                layer_num = re.search(r"\d+", key).group(0)
                new_key = self.from_hf_map[abstract_key]
                state_dict[new_key.format(layer_num)] = value

            else:
                new_key = self.from_hf_map[key]
                state_dict[new_key] = value

        return state_dict

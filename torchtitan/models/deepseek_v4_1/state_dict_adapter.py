# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checkpoint conversion for the released inference namespaces and formats.

M: image markers, D: model width.
"""

import json
import re
from pathlib import Path

import torch
from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.tensor import DTensor, Replicate

from torchtitan.models.common.moe.state_dict_adapter import MoEStateDictAdapter
from .checkpoint_storage import BlockScaledHuggingFaceStorageReader, BlockScaledLayout


class DeepSeekV41StateDictAdapter(MoEStateDictAdapter):
    def __init__(self, model_config, hf_assets_path=None):
        super().__init__(model_config, hf_assets_path)
        self.from_hf_map = {
            "embed.weight": "tok_embeddings.weight",
            "norm.weight": "norm.weight",
            "head.weight": "lm_head.weight",
        }
        self._expert_counts = {}
        self._vision_qkv = {}
        self._vision_gate_up = {}
        for layer_id, layer in enumerate(model_config.layers):
            self._add_layer(f"layers.{layer_id}", f"layers.{layer_id}", layer)
        if model_config.vision is not None:
            self._add_vision(model_config.vision)
        if self.fqn_to_index_mapping is not None:
            self.fqn_to_index_mapping = {
                key: index
                for key, index in self.fqn_to_index_mapping.items()
                if key in self.from_hf_map
            }

    def _add_vision(self, config):
        mapping = self.from_hf_map
        for name in ("weight", "bias"):
            mapping[f"vision.patch_embed.proj.{name}"] = f"vision.patch_embed.{name}"
            for projection in ("w1", "w2"):
                mapping[f"aligner.{projection}.{name}"] = f"aligner.{projection}.{name}"
        mapping["vision.norm.weight"] = "vision.norm.weight"
        for marker in ("image_start", "image_newline", "image_end"):
            mapping[marker] = "image_embeddings.weight"
        for index, _ in enumerate(config.layers):
            hf_prefix, prefix = f"vision.blocks.{index}", f"vision.layers.{index}"
            for norm in ("norm1", "norm2"):
                mapping[f"{hf_prefix}.{norm}.weight"] = f"{prefix}.{norm}.weight"
            for param in ("weight", "bias"):
                mapping[f"{hf_prefix}.attn.wo.{param}"] = f"{prefix}.attn.proj.{param}"
                fused = f"{hf_prefix}.attn.wqkv.{param}"
                logical = tuple(
                    f"{prefix}.attn.{projection}.{param}"
                    for projection in ("wq", "wk", "wv")
                )
                self._vision_qkv[fused] = logical
                mapping[fused] = logical[0]
            fused = f"{hf_prefix}.mlp.w1.weight"
            logical = (f"{prefix}.mlp.w1.weight", f"{prefix}.mlp.w3.weight")
            self._vision_gate_up[fused] = logical
            mapping[fused] = logical[0]
            mapping[f"{hf_prefix}.mlp.w2.weight"] = f"{prefix}.mlp.w2.weight"

    def _add_layer(self, hf, native, config):
        mapping = self.from_hf_map
        for name in ("wq_a", "q_norm", "wq_b", "wkv", "kv_norm", "wo_a", "wo_b"):
            mapping[f"{hf}.attn.{name}.weight"] = f"{native}.attention.{name}.weight"
        mapping[f"{hf}.attn.attn_sink"] = f"{native}.attention.attn_sink.weight"
        if config.attention.compressor is not None:
            for name in ("wkv", "norm", "wgate"):
                if getattr(config.attention.compressor, name, None) is not None:
                    mapping[
                        f"{hf}.attn.compressor.{name}.weight"
                    ] = f"{native}.attention.compressor.{name}.weight"
        if config.attention.indexer is not None:
            for name in ("wq_b", "weights_proj", "wk", "k_norm"):
                if getattr(config.attention.indexer, name) is not None:
                    mapping[
                        f"{hf}.attn.indexer.{name}.weight"
                    ] = f"{native}.attention.indexer.{name}.weight"
        for sublayer in ("attn", "ffn"):
            for name in ("fn", "base", "scale"):
                mapping[
                    f"{hf}.hc_{sublayer}_{name}"
                ] = f"{native}.hc_{sublayer}.hc_{name}"
        mapping[f"{hf}.attn_norm.weight"] = f"{native}.attention_norm.weight"
        mapping[f"{hf}.ffn_norm.weight"] = f"{native}.ffn_norm.weight"
        mapping[f"{hf}.ffn.gate.weight"] = f"{native}.feed_forward.router.gate.weight"
        mapping[f"{hf}.ffn.gate.bias"] = f"{native}.feed_forward.expert_bias_E"
        mapping[f"{hf}.ffn.gate.bias_vl"] = f"{native}.feed_forward.expert_bias_vl_E"
        for name in ("w1", "w2", "w3"):
            mapping[
                f"{hf}.ffn.shared_experts.{name}.weight"
            ] = f"{native}.feed_forward.shared_experts.{name}.weight"
        self._expert_counts[native] = config.feed_forward.num_experts
        for expert in range(config.feed_forward.num_experts):
            for hf_name, native_name in (
                ("w1.weight", "w1_EFD"),
                ("w2.weight", "w2.weight"),
                ("w3.weight", "w3_EFD"),
            ):
                mapping[
                    f"{hf}.ffn.experts.{expert}.{hf_name}"
                ] = f"{native}.feed_forward.routed_experts.{native_name}"
        if config.engram is not None:
            for name in ("embed.weight", "wkv.weight", "q_weight", "k_weight"):
                mapping[f"{hf}.engram.{name}"] = f"{native}.engram.{name}"

    def to_hf(self, state_dict, *, to_quantized_layout=False):
        state_dict = self._native_fused_linears_to_hf(
            state_dict, split_routed_experts=True
        )
        to_hf_map = {
            native: hf
            for hf, native in self.from_hf_map.items()
            if ".ffn.experts." not in hf
        }
        output = {}
        for fused, logical in (self._vision_qkv | self._vision_gate_up).items():
            if all(key in state_dict for key in logical):
                output[fused] = torch.cat(
                    [state_dict.pop(key) for key in logical], dim=0
                )
        if "image_embeddings.weight" in state_dict:
            markers_MD = state_dict.pop("image_embeddings.weight")
            if isinstance(markers_MD, DTensor):
                markers_MD = markers_MD.redistribute(
                    placements=[
                        Replicate() if placement.is_shard(0) else placement
                        for placement in markers_MD.placements
                    ]
                )
            for key, value in zip(
                ("image_start", "image_newline", "image_end"),
                markers_MD.unbind(0),
                strict=True,
            ):
                output[key] = value
        for key, value in state_dict.items():
            if self._is_expert_weight_key(key):
                match = re.match(r"(layers)\.(\d+)\.(.*)", key)
                assert match is not None
                root, layer_id, tail = match.groups()
                hf_root = root
                projection = {
                    "feed_forward.routed_experts.w1_EFD": "w1",
                    "feed_forward.routed_experts.w2.weight": "w2",
                    "feed_forward.routed_experts.w3_EFD": "w3",
                }[tail]
                hf_abstract = f"{hf_root}.{{}}.ffn.experts.{{}}.{projection}.weight"
                native_abstract = f"{root}.{{}}.{tail}"
                if isinstance(value, DTensor):
                    self.grouped_expert_weight_placements[
                        native_abstract
                    ] = value.placements
                    self.grouped_expert_weight_shape[native_abstract] = value.shape
                    self.grouped_expert_weight_mesh[native_abstract] = value.device_mesh
                    output.update(
                        self._get_local_experts_weights(
                            hf_abstract, native_abstract, layer_id, value
                        )
                    )
                else:
                    output.update(
                        {
                            hf_abstract.format(layer_id, expert): weight
                            for expert, weight in enumerate(value.unbind(0))
                        }
                    )
            else:
                if key not in to_hf_map:
                    raise ValueError(f"Unmapped V4.1 native checkpoint key: {key}")
                output[to_hf_map[key]] = (
                    value.squeeze(-1) if key.endswith("attn_sink.weight") else value
                )
        return output

    def from_hf(self, hf_state_dict, *, is_quantized_layout=False):
        output, grouped = {}, {}
        hf_state_dict = dict(hf_state_dict)
        for fused, logical in (self._vision_qkv | self._vision_gate_up).items():
            if fused in hf_state_dict:
                output.update(
                    zip(
                        logical,
                        hf_state_dict.pop(fused).chunk(len(logical), dim=0),
                        strict=True,
                    )
                )
        markers = ("image_start", "image_newline", "image_end")
        if all(key in hf_state_dict for key in markers):
            output["image_embeddings.weight"] = torch.stack(
                [hf_state_dict.pop(key) for key in markers]
            )
        for key, value in hf_state_dict.items():
            if key not in self.from_hf_map:
                raise ValueError(f"Unmapped V4.1 HF checkpoint key: {key}")
            native = self.from_hf_map[key]
            if ".ffn.experts." in key:
                match = re.match(r"(layers)\.(\d+)\.ffn\.experts\.(\d+)\.", key)
                assert match is not None
                root, layer_id, expert_id = match.groups()
                native_root = root
                abstract = native.replace(
                    f"{native_root}.{layer_id}.", f"{native_root}.{{}}.", 1
                )
                grouped.setdefault(layer_id, {}).setdefault(abstract, {})[
                    int(expert_id)
                ] = value
                if abstract in self.local_experts_indices:
                    weight = self._concatenate_expert_weights_dtensor(
                        grouped, abstract, layer_id
                    )
                else:
                    weight = self._concatenate_expert_weights(
                        grouped,
                        abstract,
                        layer_id,
                        self._expert_counts[f"{native_root}.{layer_id}"],
                    )
                if weight is not None:
                    output[native] = weight
            else:
                output[native] = (
                    value.unsqueeze(-1) if key.endswith(".attn.attn_sink") else value
                )
        if grouped:
            raise ValueError(
                "HF checkpoint contains an incomplete group of V4.1 expert weights."
            )
        return self._native_fused_linears_from_hf(output, fuse_routed_experts=True)

    def get_hf_storage_reader(self, path, from_quantized=False, *, thread_count=None):
        reader_kwargs = {} if thread_count is None else {"thread_count": thread_count}
        if not from_quantized:
            return HuggingFaceStorageReader(path, **reader_kwargs)
        with open(Path(path) / "model.safetensors.index.json") as file:
            weight_map = json.load(file)["weight_map"]
        layouts = {}
        for key in weight_map:
            if not key.endswith(".weight"):
                continue
            scale_key = key.removesuffix(".weight") + ".scale"
            if scale_key in weight_map:
                packed = ".ffn.experts." in key
                row_scaled = packed or ".engram.embed." in key
                layouts[key] = BlockScaledLayout(
                    scale_key, 1 if row_scaled else 32, 32, packed_fp4=packed
                )
        return BlockScaledHuggingFaceStorageReader(
            path, layouts=layouts, **reader_kwargs
        )

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Flash-Next checkpoint mapping, including its 128 row-wise n-gram shards."""

# N = embedding rows, D = input features, H = head groups, C = projection channels.

from __future__ import annotations

import re
from typing import Any, cast, TYPE_CHECKING

import torch
from torch.distributed.checkpoint.protocol import CheckpointableTensor
from torch.distributed.tensor import DTensor, Shard

from torchtitan.models.common.attention.gdn import GDN
from torchtitan.models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter

from .attention import QSAAttention, SplitLinear

if TYPE_CHECKING:
    from .model import Qwen38FlashNextModel


class Qwen38FlashNextStateDictAdapter(Qwen35StateDictAdapter):
    def __init__(
        self, model_config: Qwen38FlashNextModel.Config, hf_assets_path: str | None
    ):
        super().__init__(model_config, hf_assets_path)
        self.flash_config = model_config
        self._ngram_weight_layouts: dict[int, DTensor] = {}
        self._split_projection_layouts: dict[str, tuple[int, int, int]] = {}
        for layer_id, layer in enumerate(model_config.layers):
            if isinstance(layer.attention, GDN.Config):
                field = "in_proj"
                linear = layer.attention.in_proj
            else:
                assert isinstance(layer.attention, QSAAttention.Config)
                field = "wq"
                linear = layer.attention.wq
            assert isinstance(linear, SplitLinear.Config)
            self._split_projection_layouts[f"layers.{layer_id}.attention.{field}"] = (
                linear.num_groups,
                linear.projection.out_features // linear.num_groups,
                linear.gates.out_features // linear.num_groups,
            )
        prefix = self.hf_language_model_prefix
        for norm in ("input_layernorm", "post_attention_layernorm"):
            self.from_hf_map.pop(f"{prefix}.layers.{{}}.{norm}.weight")
        self.from_hf_map.pop(f"{prefix}.norm.weight")
        for connection in ("attn_hyper_connection", "mlp_hyper_connection"):
            for field in (
                "hc_norm",
                "input_mix_weight_down",
                "input_mix_weight_up",
                "block_inject_weight",
            ):
                self.from_hf_map[
                    f"{prefix}.layers.{{}}.{connection}.{field}.weight"
                ] = f"layers.{{}}.{connection}.{field}.weight"
        for field in ("hc_norm", "input_mix_weight_down", "input_mix_weight_up"):
            self.from_hf_map[
                f"{prefix}.hyper_connection_mixer.{field}.weight"
            ] = f"hyper_connection_mixer.{field}.weight"
        for field in ("index_qk_proj", "q_layernorm", "k_layernorm"):
            self.from_hf_map[
                f"{prefix}.layers.{{}}.self_attn.indexer.{field}.weight"
            ] = f"layers.{{}}.attention.indexer.{field}.weight"
        for field in (
            "key_proj",
            "value_proj",
            "norm_key",
            "norm_query",
            "norm_conv",
            "conv1d",
        ):
            self.from_hf_map[
                f"{prefix}.layers.{{}}.ple.{field}.weight"
            ] = f"layers.{{}}.ple.{field}.weight"
        for field in (
            "layer_multipliers",
            "ngram_heads_vocab_sizes",
            "ngram_heads_offsets",
            "ngram_embedding.weight",
        ):
            self.from_hf_map[
                f"{prefix}.layers.{{}}.ple.ple_embedding.{field}"
            ] = f"layers.{{}}.ple.ple_embedding.{field}"

    def to_hf(
        self, state_dict: dict[str, Any], *, to_quantized_layout: bool = False
    ) -> dict[str, Any]:
        state_dict = dict(state_dict)
        for prefix, (num_groups, _, _) in self._split_projection_layouts.items():
            projection_key = f"{prefix}.projection.weight"
            if projection_key not in state_dict:
                continue
            projections = []
            for field in ("projection", "gates"):
                key = f"{prefix}.{field}.weight"
                weight_CD = state_dict.pop(key)
                if isinstance(weight_CD, DTensor):
                    self._qkv_linear_sharding[key] = (
                        weight_CD.device_mesh,
                        weight_CD.placements,
                    )
                    # A DP row shard may split a head group, as with GDN's
                    # existing HF conversion. Keep whole groups on column shards.
                    weight_CD = weight_CD.redistribute(
                        weight_CD.device_mesh, [Shard(1)] * weight_CD.device_mesh.ndim
                    )
                projections.append(weight_CD.unflatten(0, (num_groups, -1)))
            projection_HCD, gates_HCD = projections
            state_dict[f"{prefix}.weight"] = torch.cat(
                (projection_HCD, gates_HCD), 1
            ).flatten(0, 1)
        result = super().to_hf(state_dict, to_quantized_layout=to_quantized_layout)
        for layer_id, layer in enumerate(self.flash_config.layers):
            if layer.ple is None:
                continue
            base = f"{self.hf_language_model_prefix}.layers.{layer_id}.ple.ple_embedding.ngram_embedding"
            key = base + ".weight"
            if key not in result:
                continue
            weight_ND: torch.Tensor = result.pop(key)
            parts = layer.ple.ple_embedding.split_ngram_parts
            self._ngram_weight_layouts.pop(layer_id, None)
            if isinstance(weight_ND, DTensor):
                # Keep only a meta template; retaining the export buffer would pin its storage.
                self._ngram_weight_layouts[layer_id] = cast(
                    DTensor, weight_ND.to(device="meta")
                )
                num_rows, num_features = weight_ND.shape
                mesh = weight_ND.device_mesh
                coordinate = mesh.get_coordinate()
                assert coordinate is not None
                local_shape = list(weight_ND.shape)
                offsets = [0, 0]
                # Apply shards in mesh-axis order; HSDP shards the rows twice.
                # NGramTable requires rows divisible by the DP degree, so every
                # shard is even.
                for mesh_axis, placement in enumerate(weight_ND.placements):
                    if isinstance(placement, Shard):
                        dim = placement.dim
                        assert local_shape[dim] % mesh.size(mesh_axis) == 0
                        local_shape[dim] //= mesh.size(mesh_axis)
                        offsets[dim] += coordinate[mesh_axis] * local_shape[dim]
                row_start, feature_start = offsets
                weight_ND = weight_ND.to_local(
                    grad_placements=weight_ND.placements
                ).detach()
                # Split local HF fragments, since DTensor.chunk would gather all rows.
                rows_per_part = num_rows // parts
                row_end = row_start + weight_ND.shape[0]
                # HF names describe global parts, not chunks of the owner buffer.
                for part in range(
                    row_start // rows_per_part,
                    (row_end + rows_per_part - 1) // rows_per_part,
                ):
                    part_start = part * rows_per_part
                    begin = max(row_start, part_start)
                    end = min(row_end, part_start + rows_per_part)
                    shard_ND = torch.narrow(
                        weight_ND, 0, begin - row_start, end - begin
                    )
                    # NOTE: DCP's public protocol explicitly supports coordinate
                    # fields on ordinary tensors; no wrapper is required.
                    # https://github.com/pytorch/pytorch/blob/1392e1f8e4903657d4391f3bd8fcc9ca8a74b758/torch/distributed/checkpoint/protocol.py
                    fragment = cast(CheckpointableTensor, shard_ND)
                    fragment.global_shape = (rows_per_part, num_features)
                    fragment.global_offsets = ((begin - part_start, feature_start),)
                    fragment.local_offsets = ((0, 0),)
                    fragment.local_sizes = (tuple(shard_ND.shape),)
                    result[f"{base}.shard_{part}.weight"] = shard_ND
                continue
            for part, shard_ND in enumerate(weight_ND.chunk(parts, 0)):
                result[f"{base}.shard_{part}.weight"] = shard_ND
        return result

    def from_hf(
        self, hf_state_dict: dict[str, Any], *, is_quantized_layout: bool = False
    ) -> dict[str, Any]:
        # The released checkpoint's MTP draft head is outside this model.
        state_dict = {
            key: value
            for key, value in hf_state_dict.items()
            if not key.startswith("mtp.")
        }
        shards_by_layer: dict[int, dict[int, Any]] = {}
        pattern = re.compile(
            re.escape(self.hf_language_model_prefix)
            + r"\.layers\.(\d+)\.ple\.ple_embedding\.ngram_embedding\.shard_(\d+)\.weight$"
        )
        for key in tuple(state_dict):
            match = pattern.fullmatch(key)
            if match is not None:
                layer_id, part = map(int, match.groups())
                shards_by_layer.setdefault(layer_id, {})[part] = state_dict.pop(key)
        for layer_id, shards in shards_by_layer.items():
            if layer_id >= len(self.flash_config.layers):
                raise ValueError(
                    f"Checkpoint has n-gram shards for unknown PLE layer {layer_id}"
                )
            ple = self.flash_config.layers[layer_id].ple
            if ple is None:
                raise ValueError(
                    f"Checkpoint has n-gram shards for non-PLE layer {layer_id}"
                )
            parts = ple.ple_embedding.split_ngram_parts
            owned = all(
                isinstance(value, CheckpointableTensor) for value in shards.values()
            )
            # DCP fills rank-local placeholders made by to_hf on this adapter;
            # complete shards would otherwise be wrapped as one rank's rows.
            if owned != (layer_id in self._ngram_weight_layouts):
                raise ValueError(
                    "Rank-local HF n-gram fragments require to_hf placeholders "
                    "from the same adapter, and a row-owned table requires them"
                )
            if not owned and set(shards) != set(range(parts)):
                raise ValueError(
                    f"PLE layer {layer_id} requires all {parts} n-gram shards"
                )
            base = f"{self.hf_language_model_prefix}.layers.{layer_id}.ple.ple_embedding.ngram_embedding.weight"
            if base in state_dict:
                raise ValueError(
                    "Checkpoint contains both consolidated and split n-gram weights"
                )
            ordered_parts = sorted(shards)
            weight_ND = torch.cat([shards[part] for part in ordered_parts], 0)
            if layer_id in self._ngram_weight_layouts:
                layout = self._ngram_weight_layouts[layer_id]
                weight_ND = DTensor.from_local(
                    weight_ND,
                    layout.device_mesh,
                    layout.placements,
                    shape=layout.shape,
                    stride=layout.stride(),
                    run_check=False,
                )
            state_dict[base] = weight_ND
        result = super().from_hf(state_dict, is_quantized_layout=is_quantized_layout)
        for prefix, (
            num_groups,
            projection_channels,
            gate_channels,
        ) in self._split_projection_layouts.items():
            key = f"{prefix}.weight"
            if key not in result:
                continue
            weight_CD = result.pop(key)
            if isinstance(weight_CD, DTensor):
                weight_CD = weight_CD.redistribute(
                    weight_CD.device_mesh, [Shard(1)] * weight_CD.device_mesh.ndim
                )
            weight_HCD = weight_CD.unflatten(0, (num_groups, -1))
            projection_HCD, gates_HCD = weight_HCD.split(
                (projection_channels, gate_channels), 1
            )
            for field, weight_HCD in (
                ("projection", projection_HCD),
                ("gates", gates_HCD),
            ):
                key = f"{prefix}.{field}.weight"
                weight_CD = weight_HCD.flatten(0, 1)
                if isinstance(weight_CD, DTensor) and key in self._qkv_linear_sharding:
                    mesh, placements = self._qkv_linear_sharding[key]
                    weight_CD = weight_CD.redistribute(mesh, placements)
                result[key] = weight_CD
        return result

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bidirectional GLM-5 checkpoint conversion for local tensors and DTensors.

TorchTitan and Hugging Face use different parameter names and different routed
expert packing. Most parameters need only a key rename. Routed expert gate/up
weights are separate in TorchTitan but fused along hidden dimension 1 in HF.

That fusion is trivial for local tensors. It is not trivial when TP already
shards dimension 1: concatenating two local shards would produce
``gate_rank0, up_rank0, gate_rank1, up_rank1`` globally instead of HF's
``all_gate, all_up`` order. The all-to-all helpers below exchange half-rank
chunks so conversion preserves both global value order and DTensor placement.
The inverse path uses the exact reverse exchange. Preserving DTensor placements
also lets TorchTitan's distributed checkpoint writer save and load HuggingFace
safetensors without first materializing one full model on every rank.
"""

from __future__ import annotations

import re
from typing import Any, TYPE_CHECKING

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from torch.distributed.tensor.placement_types import _StridedShard, Shard

from torchtitan.models.common.moe import MoE
from torchtitan.models.common.moe.state_dict_adapter import MoEStateDictAdapter
from torchtitan.models.common.rope import ComplexRoPE

if TYPE_CHECKING:
    from .model import Glm5Model


def _contiguous_stride(shape: torch.Size) -> tuple[int, ...]:
    stride = [1] * len(shape)
    for dim in range(len(shape) - 2, -1, -1):
        stride[dim] = stride[dim + 1] * shape[dim + 1]
    return tuple(stride)


class Glm5StateDictAdapter(MoEStateDictAdapter):
    """Bidirectional adapter for Transformers GLM-MoE-DSA checkpoints."""

    HF_TO_TITAN_TOP_LEVEL = {
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "lm_head.weight",
    }
    HF_TO_TITAN_LAYER = {
        "self_attn.q_a_proj.weight": "attention.wq_a.weight",
        "self_attn.q_a_layernorm.weight": "attention.q_norm.weight",
        "self_attn.q_b_proj.weight": "attention.wq_b.weight",
        "self_attn.kv_a_proj_with_mqa.weight": "attention.wkv_a.weight",
        "self_attn.kv_a_layernorm.weight": "attention.kv_norm.weight",
        "self_attn.kv_b_proj.weight": "attention.wkv_b.weight",
        "self_attn.o_proj.weight": "attention.wo.weight",
        "self_attn.indexer.wq_b.weight": "attention.indexer.wq_b.weight",
        "self_attn.indexer.wk.weight": "attention.indexer.wk.weight",
        "self_attn.indexer.k_norm.weight": "attention.indexer.k_norm.weight",
        "self_attn.indexer.k_norm.bias": "attention.indexer.k_norm.bias",
        "self_attn.indexer.weights_proj.weight": (
            "attention.indexer.weights_proj.weight"
        ),
        "mlp.gate_proj.weight": "feed_forward.w1.weight",
        "mlp.up_proj.weight": "feed_forward.w3.weight",
        "mlp.down_proj.weight": "feed_forward.w2.weight",
        "mlp.gate.weight": "feed_forward.router.gate.weight",
        "mlp.gate.e_score_correction_bias": "feed_forward.expert_bias_E",
        "mlp.shared_experts.gate_proj.weight": "feed_forward.shared_experts.w1.weight",
        "mlp.shared_experts.up_proj.weight": "feed_forward.shared_experts.w3.weight",
        "mlp.shared_experts.down_proj.weight": "feed_forward.shared_experts.w2.weight",
        "input_layernorm.weight": "attention_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
    }
    HF_TO_TITAN_MTP = {
        "enorm.weight": "enorm.weight",
        "hnorm.weight": "hnorm.weight",
        "eh_proj.weight": "eh_proj.weight",
        "shared_head.norm.weight": "mtp_norm.weight",
    }
    _IGNORED_HF_KEYS = {
        "model.rotary_emb.inv_freq",
        "model.rotary_emb.original_inv_freq",
    }
    _HF_LAYER_KEY = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
    _TITAN_LAYER_KEY = re.compile(r"^layers\.(\d+)\.(.+)$")
    _TITAN_MTP_KEY = re.compile(r"^mtp_layers\.(\d+)\.(.+)$")
    _FUSED_GATE_UP = "mlp.experts.gate_up_proj"
    _FUSED_DOWN = "mlp.experts.down_proj"
    _TITAN_GATE = "feed_forward.routed_experts.w1_EFD"
    _TITAN_UP = "feed_forward.routed_experts.w3_EFD"
    _TITAN_DOWN = "feed_forward.routed_experts.w2.weight"

    def __init__(
        self,
        model_config: Glm5Model.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)
        self.glm_config = model_config
        self.from_hf_map = {
            **self.HF_TO_TITAN_TOP_LEVEL,
            **{
                f"model.layers.{{}}.{hf_suffix}": f"layers.{{}}.{titan_suffix}"
                for hf_suffix, titan_suffix in self.HF_TO_TITAN_LAYER.items()
            },
        }
        self._to_hf_top_level = {v: k for k, v in self.HF_TO_TITAN_TOP_LEVEL.items()}
        self._to_hf_layer = {v: k for k, v in self.HF_TO_TITAN_LAYER.items()}
        self._to_hf_mtp = {v: k for k, v in self.HF_TO_TITAN_MTP.items()}

    def _layer_config(self, layer_index: int):
        if 0 <= layer_index < len(self.glm_config.layers):
            return self.glm_config.layers[layer_index]
        mtp_index = layer_index - len(self.glm_config.layers)
        if 0 <= mtp_index < len(self.glm_config.mtp_layers):
            return self.glm_config.mtp_layers[mtp_index]
        raise KeyError(f"unmapped HF key: layer index {layer_index} is unsupported")

    @staticmethod
    def _norm_shape(normalized_shape: int | tuple[int, ...]) -> tuple[int, ...]:
        return (
            (normalized_shape,)
            if isinstance(normalized_shape, int)
            else tuple(normalized_shape)
        )

    @staticmethod
    def _linear_shape(config: Any) -> tuple[int, int]:
        return (config.out_features, config.in_features)

    def _expected_shape(self, titan_key: str) -> tuple[int, ...]:
        if titan_key == "tok_embeddings.weight":
            return (
                self.glm_config.tok_embeddings.num_embeddings,
                self.glm_config.tok_embeddings.embedding_dim,
            )
        if titan_key == "norm.weight":
            return self._norm_shape(self.glm_config.norm.normalized_shape)
        if titan_key == "lm_head.weight":
            return self._linear_shape(self.glm_config.lm_head)

        match = self._TITAN_LAYER_KEY.fullmatch(titan_key)
        mtp_match = self._TITAN_MTP_KEY.fullmatch(titan_key)
        if match is None and mtp_match is None:
            raise KeyError(f"unmapped TorchTitan key: {titan_key}")
        if mtp_match is not None:
            mtp_index = int(mtp_match.group(1))
            layer_index = len(self.glm_config.layers) + mtp_index
            suffix = mtp_match.group(2)
        else:
            assert match is not None
            layer_index = int(match.group(1))
            suffix = match.group(2)
        layer = self._layer_config(layer_index)
        if mtp_match is not None:
            mtp_configs = {
                "enorm.weight": layer.enorm,
                "hnorm.weight": layer.hnorm,
                "eh_proj.weight": layer.eh_proj,
                "mtp_norm.weight": layer.mtp_norm,
            }
            if suffix in {"enorm.weight", "hnorm.weight", "mtp_norm.weight"}:
                return self._norm_shape(mtp_configs[suffix].normalized_shape)
            if suffix == "eh_proj.weight":
                return self._linear_shape(mtp_configs[suffix])
        attention = layer.attention

        linear_configs = {
            "attention.wq_a.weight": attention.wq_a,
            "attention.wq_b.weight": attention.wq_b,
            "attention.wkv_a.weight": attention.wkv_a,
            "attention.wkv_b.weight": attention.wkv_b,
            "attention.wo.weight": attention.wo,
        }
        if attention.indexer is not None:
            linear_configs.update(
                {
                    "attention.indexer.wq_b.weight": attention.indexer.wq_b,
                    "attention.indexer.wk.weight": attention.indexer.wk,
                    "attention.indexer.weights_proj.weight": (
                        attention.indexer.weights_proj
                    ),
                }
            )
        if suffix in linear_configs:
            return self._linear_shape(linear_configs[suffix])

        norm_configs = {
            "attention.q_norm.weight": attention.q_norm,
            "attention.kv_norm.weight": attention.kv_norm,
            "attention_norm.weight": layer.attention_norm,
            "ffn_norm.weight": layer.ffn_norm,
        }
        if attention.indexer is not None:
            norm_configs.update(
                {
                    "attention.indexer.k_norm.weight": attention.indexer.k_norm,
                    "attention.indexer.k_norm.bias": attention.indexer.k_norm,
                }
            )
        if suffix in norm_configs:
            return self._norm_shape(norm_configs[suffix].normalized_shape)

        if suffix.startswith("feed_forward.") and layer.feed_forward is not None:
            projection = suffix.removeprefix("feed_forward.").removesuffix(".weight")
            if projection in {"w1", "w3"}:
                return self._linear_shape(layer.feed_forward.w13)
            if projection == "w2":
                return self._linear_shape(layer.feed_forward.w2)

        if isinstance(layer.feed_forward, MoE.Config):
            moe = layer.feed_forward
            if suffix == "feed_forward.router.gate.weight":
                return self._linear_shape(moe.router.gate)
            if suffix == "feed_forward.expert_bias_E":
                return (moe.num_experts,)
            if suffix in {self._TITAN_GATE, self._TITAN_UP}:
                experts = moe.routed_experts.w13
                return (
                    experts.group_size,
                    experts.out_features,
                    experts.in_features,
                )
            if suffix == self._TITAN_DOWN:
                experts = moe.routed_experts.w2
                return (
                    experts.group_size,
                    experts.out_features,
                    experts.in_features,
                )
            if suffix.startswith("feed_forward.shared_experts."):
                if moe.shared_experts is None:
                    raise KeyError(
                        f"unmapped TorchTitan key: layer has no shared experts: {titan_key}"
                    )
                projection = suffix.removeprefix(
                    "feed_forward.shared_experts."
                ).removesuffix(".weight")
                if projection in {"w1", "w3"}:
                    return self._linear_shape(moe.shared_experts.w13)
                if projection == "w2":
                    return self._linear_shape(moe.shared_experts.w2)

        raise KeyError(f"unmapped TorchTitan key: {titan_key}")

    def _validate_tensor_shape(self, titan_key: str, value: Any) -> None:
        expected = self._expected_shape(titan_key)
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
            actual = tuple(value.shape) if isinstance(value, torch.Tensor) else None
            raise ValueError(
                f"invalid shape for {titan_key}: expected {expected}, got {actual}"
            )

    def _validate_fused_experts(self, layer_index: int, value: Any) -> None:
        layer = self._layer_config(layer_index)
        if not isinstance(layer.feed_forward, MoE.Config):
            raise KeyError(
                f"unmapped HF key: layer {layer_index} has no routed GLM-5 experts"
            )
        experts = layer.feed_forward.routed_experts.w13
        expected = (
            experts.group_size,
            2 * experts.out_features,
            experts.in_features,
        )
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != expected:
            actual = tuple(value.shape) if isinstance(value, torch.Tensor) else None
            raise ValueError(
                f"invalid fused expert shape: expected {expected}, got {actual}"
            )

    def _validate_fused_down(self, layer_index: int, value: Any) -> None:
        num_main_layers = len(self.glm_config.layers)
        titan_key = (
            f"layers.{layer_index}.{self._TITAN_DOWN}"
            if layer_index < num_main_layers
            else f"mtp_layers.{layer_index - num_main_layers}.{self._TITAN_DOWN}"
        )
        self._validate_tensor_shape(titan_key, value)

    @staticmethod
    def _is_sharded_on_dim(value: DTensor, dim: int) -> bool:
        return any(
            isinstance(placement, (Shard, _StridedShard)) and placement.dim == dim
            for placement in value.placements
        )

    @staticmethod
    def _tp_axis_for_fusion(value: DTensor) -> int:
        mesh_axes = [
            mesh_axis
            for mesh_axis, placement in enumerate(value.placements)
            if isinstance(placement, (Shard, _StridedShard)) and placement.dim == 1
        ]
        if len(mesh_axes) != 1:
            raise NotImplementedError(
                "GLM-5 fused gate/up conversion requires exactly one mesh axis "
                "sharding tensor dimension 1"
            )
        mesh_axis = mesh_axes[0]
        placement = value.placements[mesh_axis]
        if isinstance(placement, _StridedShard):
            raise NotImplementedError(
                "GLM-5 fused gate/up conversion does not support a "
                "_StridedShard placement on tensor dimension 1"
            )
        return mesh_axis

    @classmethod
    def _all_to_all_fuse_gate_up(cls, gate: DTensor, up: DTensor) -> DTensor:
        """Fuse TP-sharded gate/up weights into HF global ordering.

        Each source rank sends its gate shard to one destination half and its
        up shard to the corresponding destination in the other half. The local
        reshape then joins the received pair along hidden dimension 1 while the
        returned DTensor keeps the original Shard(1) placement.
        """
        mesh_axis = cls._tp_axis_for_fusion(gate)
        tp_size = gate.device_mesh.size(mesh_axis)
        if tp_size == 1:
            fused = torch.cat((gate, up), dim=1)
            assert isinstance(fused, DTensor)
            return fused
        if tp_size % 2 != 0:
            raise NotImplementedError(
                "GLM-5 fused gate/up conversion requires an even TP size"
            )
        tp_rank = gate.device_mesh.get_local_rank(mesh_axis)
        group = gate.device_mesh.get_group(mesh_axis)

        local_gate = gate._local_tensor
        local_up = up._local_tensor
        if local_up.shape != local_gate.shape:
            raise ValueError(
                "routed expert gate and up DTensors must have matching local shapes"
            )
        num_local_experts, shard_width, dim = local_gate.shape

        # Pack on the expert axis so all_to_all_single can use element counts as
        # split sizes without flattening or losing [E,F,D] structure.
        packed = torch.cat((local_gate, local_up), dim=0)
        output = torch.empty_like(packed)
        half_tp = tp_size // 2
        input_split_sizes = [0] * tp_size
        input_split_sizes[tp_rank // 2] = num_local_experts
        input_split_sizes[half_tp + tp_rank // 2] = num_local_experts
        output_split_sizes = [0] * tp_size
        source_start = 2 * (tp_rank % half_tp)
        output_split_sizes[source_start] = num_local_experts
        output_split_sizes[source_start + 1] = num_local_experts
        dist.all_to_all_single(
            output,
            packed,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )

        local_fused = (
            output.view(2, num_local_experts, shard_width, dim)
            .movedim(0, 1)
            .reshape(num_local_experts, 2 * shard_width, dim)
        )
        fused_shape = list(gate.shape)
        fused_shape[1] += up.shape[1]
        global_shape = torch.Size(fused_shape)
        return DTensor.from_local(
            local_fused,
            device_mesh=gate.device_mesh,
            placements=gate.placements,
            run_check=False,
            shape=global_shape,
            stride=_contiguous_stride(global_shape),
        )

    @classmethod
    def _all_to_all_split_gate_up(cls, fused: DTensor) -> tuple[DTensor, DTensor]:
        """Reverse distributed HF fusion and restore TorchTitan gate/up shards."""
        mesh_axis = cls._tp_axis_for_fusion(fused)
        tp_size = fused.device_mesh.size(mesh_axis)
        if tp_size == 1:
            gate, up = fused.chunk(2, dim=1)
            assert isinstance(gate, DTensor) and isinstance(up, DTensor)
            return gate, up
        if tp_size % 2 != 0:
            raise NotImplementedError(
                "GLM-5 fused gate/up conversion requires an even TP size"
            )
        tp_rank = fused.device_mesh.get_local_rank(mesh_axis)
        group = fused.device_mesh.get_group(mesh_axis)

        local_fused = fused._local_tensor
        if local_fused.shape[1] % 2 != 0:
            raise ValueError("local fused gate/up dimension must be even")
        num_local_experts, fused_width, dim = local_fused.shape
        shard_width = fused_width // 2
        packed = (
            local_fused.view(num_local_experts, 2, shard_width, dim)
            .movedim(1, 0)
            .reshape(2 * num_local_experts, shard_width, dim)
            .contiguous()
        )
        output = torch.empty_like(packed)
        half_tp = tp_size // 2
        input_split_sizes = [0] * tp_size
        target_start = 2 * (tp_rank % half_tp)
        input_split_sizes[target_start] = num_local_experts
        input_split_sizes[target_start + 1] = num_local_experts
        output_split_sizes = [0] * tp_size
        output_split_sizes[tp_rank // 2] = num_local_experts
        output_split_sizes[half_tp + tp_rank // 2] = num_local_experts
        dist.all_to_all_single(
            output,
            packed,
            output_split_sizes=output_split_sizes,
            input_split_sizes=input_split_sizes,
            group=group,
        )

        local_gate = output[:num_local_experts]
        local_up = output[num_local_experts:]
        source_shape = list(fused.shape)
        source_shape[1] //= 2
        global_shape = torch.Size(source_shape)
        stride = _contiguous_stride(global_shape)
        gate = DTensor.from_local(
            local_gate,
            device_mesh=fused.device_mesh,
            placements=fused.placements,
            run_check=False,
            shape=global_shape,
            stride=stride,
        )
        up = DTensor.from_local(
            local_up,
            device_mesh=fused.device_mesh,
            placements=fused.placements,
            run_check=False,
            shape=global_shape,
            stride=stride,
        )
        return gate, up

    @classmethod
    def _fuse_gate_up(cls, gate: Any, up: Any) -> Any:
        """Fuse grouped gate/up weights without corrupting TP shard order."""

        gate_is_dtensor = isinstance(gate, DTensor)
        up_is_dtensor = isinstance(up, DTensor)
        if gate_is_dtensor != up_is_dtensor:
            raise ValueError(
                "routed expert gate and up weights must both be DTensors or "
                "both be local tensors"
            )
        if not gate_is_dtensor:
            return torch.cat((gate, up), dim=1)

        if gate.device_mesh != up.device_mesh or gate.placements != up.placements:
            raise ValueError(
                "routed expert gate and up DTensors must use the same device "
                "mesh and placements"
            )

        if cls._is_sharded_on_dim(gate, 1):
            return cls._all_to_all_fuse_gate_up(gate, up)
        return torch.cat((gate, up), dim=1)

    @classmethod
    def _split_gate_up(cls, fused: Any) -> tuple[Any, Any]:
        """Undo HuggingFace gate/up fusion and restore TP placements."""

        if not isinstance(fused, DTensor):
            return fused.chunk(2, dim=1)

        if cls._is_sharded_on_dim(fused, 1):
            return cls._all_to_all_split_gate_up(fused)
        gate, up = fused.chunk(2, dim=1)
        return gate, up

    def from_hf(
        self,
        hf_state_dict: dict[str, Any],
        *,
        is_quantized_layout: bool = False,
    ) -> dict[str, Any]:
        """Validate and convert one HF state dict into TorchTitan key/layout form."""
        del is_quantized_layout
        self._validate_hf_rope_config(ComplexRoPE.Config)
        state_dict: dict[str, Any] = {}

        for hf_key, value in hf_state_dict.items():
            if hf_key in self._IGNORED_HF_KEYS:
                continue
            if hf_key in self.HF_TO_TITAN_TOP_LEVEL:
                titan_key = self.HF_TO_TITAN_TOP_LEVEL[hf_key]
                self._validate_tensor_shape(titan_key, value)
                state_dict[titan_key] = value
                continue

            match = self._HF_LAYER_KEY.fullmatch(hf_key)
            if match is None:
                raise KeyError(f"unmapped HF key: {hf_key}")
            layer_index = int(match.group(1))
            suffix = match.group(2)
            layer = self._layer_config(layer_index)
            mtp_index = layer_index - len(self.glm_config.layers)
            is_mtp = mtp_index >= 0
            prefix = f"mtp_layers.{mtp_index}" if is_mtp else f"layers.{layer_index}"

            if suffix == self._FUSED_GATE_UP:
                self._validate_fused_experts(layer_index, value)
                gate, up = self._split_gate_up(value)
                gate_key = f"{prefix}.{self._TITAN_GATE}"
                up_key = f"{prefix}.{self._TITAN_UP}"
                state_dict[gate_key] = gate
                state_dict[up_key] = up
                continue
            if suffix == self._FUSED_DOWN:
                self._validate_fused_down(layer_index, value)
                state_dict[f"{prefix}.{self._TITAN_DOWN}"] = value
                continue

            titan_suffix = (
                self.HF_TO_TITAN_MTP.get(suffix)
                if is_mtp and suffix in self.HF_TO_TITAN_MTP
                else self.HF_TO_TITAN_LAYER.get(suffix)
            )
            if titan_suffix is None:
                raise KeyError(f"unmapped HF key: {hf_key}")
            titan_key = f"{prefix}.{titan_suffix}"
            self._validate_tensor_shape(titan_key, value)
            state_dict[titan_key] = value

        return self._native_fused_linears_from_hf(
            state_dict,
            fuse_routed_experts=True,
        )

    def to_hf(
        self,
        state_dict: dict[str, Any],
        *,
        to_quantized_layout: bool = False,
    ) -> dict[str, Any]:
        """Validate and convert TorchTitan state into HF names and expert packing."""
        del to_quantized_layout
        state_dict = self._native_fused_linears_to_hf(
            state_dict,
            split_routed_experts=True,
        )
        hf_state_dict: dict[str, Any] = {}
        fused_experts: dict[int, dict[str, Any]] = {}

        for titan_key, value in state_dict.items():
            hf_key = self._to_hf_top_level.get(titan_key)
            if hf_key is not None:
                self._validate_tensor_shape(titan_key, value)
                hf_state_dict[hf_key] = value
                continue

            match = self._TITAN_LAYER_KEY.fullmatch(titan_key)
            mtp_match = self._TITAN_MTP_KEY.fullmatch(titan_key)
            if match is None and mtp_match is None:
                raise KeyError(f"unmapped TorchTitan key: {titan_key}")
            if mtp_match is not None:
                mtp_index = int(mtp_match.group(1))
                layer_index = len(self.glm_config.layers) + mtp_index
                suffix = mtp_match.group(2)
            else:
                assert match is not None
                layer_index = int(match.group(1))
                suffix = match.group(2)
            self._layer_config(layer_index)

            if suffix in {self._TITAN_GATE, self._TITAN_UP, self._TITAN_DOWN}:
                self._validate_tensor_shape(titan_key, value)
                fused_experts.setdefault(layer_index, {})[suffix] = value
                continue

            hf_suffix = (
                self._to_hf_mtp.get(suffix)
                if mtp_match is not None and suffix in self._to_hf_mtp
                else self._to_hf_layer.get(suffix)
            )
            if hf_suffix is None:
                raise KeyError(f"unmapped TorchTitan key: {titan_key}")
            self._validate_tensor_shape(titan_key, value)
            hf_state_dict[f"model.layers.{layer_index}.{hf_suffix}"] = value

        for layer_index, weights in fused_experts.items():
            required = {self._TITAN_GATE, self._TITAN_UP, self._TITAN_DOWN}
            missing = required.difference(weights)
            if missing:
                raise KeyError(
                    "incomplete routed expert mapping for layer "
                    f"{layer_index}: missing {sorted(missing)}"
                )
            hf_state_dict[
                f"model.layers.{layer_index}.{self._FUSED_GATE_UP}"
            ] = self._fuse_gate_up(weights[self._TITAN_GATE], weights[self._TITAN_UP])
            hf_state_dict[f"model.layers.{layer_index}.{self._FUSED_DOWN}"] = weights[
                self._TITAN_DOWN
            ]

        return hf_state_dict

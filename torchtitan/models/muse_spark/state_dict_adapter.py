# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import json
import os
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast, Literal, TYPE_CHECKING

import torch

from torch.distributed.checkpoint import HuggingFaceStorageReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from torch.distributed.tensor import DTensor, Replicate, Shard

from torchtitan.models.common.feed_forward import DenseFeedForward
from torchtitan.models.common.moe import MoE
from torchtitan.models.common.moe.state_dict_adapter import MoEStateDictAdapter

if TYPE_CHECKING:
    from torchtitan.models.muse_spark.model import MuseSparkModel


_HF_LOAD_THREAD_COUNT = 8


_TEXT_TOP_LEVEL_RENAMES = {
    "model.embed_tokens.weight": "model.language_model.embed_tokens.weight",
    "model.norm.weight": "model.language_model.norm.weight",
    "lm_head.weight": "lm_head.weight",
}

_TEXT_LAYER_RENAMES = {
    "self_attn.pre_qkv_proj_norm.weight": "input_layernorm.weight",
    "self_attn.q_proj.weight": "self_attn.q_proj.weight",
    "self_attn.k_proj.weight": "self_attn.k_proj.weight",
    "self_attn.v_proj.weight": "self_attn.v_proj.weight",
    "self_attn.o_proj.weight": "self_attn.o_proj.weight",
    "self_attn.o_gate_proj.weight": "self_attn.gate_proj.weight",
    "post_attn_norm.weight": "post_attention_layernorm.weight",
    "pre_ffn_norm.weight": "pre_feedforward_layernorm.weight",
    "post_ffn_norm.weight": "post_feedforward_layernorm.weight",
    "add_attn.gate": "post_attention_residual_gate.gate",
    "add_ffn.gate": "post_feedforward_residual_gate.gate",
    "feed_forward.fc1_weight.w1.weight": "mlp.gate_proj.weight",
    "feed_forward.fc1_weight.w3.weight": "mlp.up_proj.weight",
    "feed_forward.fc2_weight.w2.weight": "mlp.down_proj.weight",
    "feed_forward.router.weight": "mlp.gate.weight",
    "feed_forward.expert_bias": "mlp.gate.e_score_correction_bias",
    "feed_forward.pooler.weight": "mlp.pre_expert_proj.weight",
    "feed_forward.pre_expert_norm.weight": "mlp.pre_expert_norm.weight",
    "feed_forward.experts.w13_weight": "mlp.experts.gate_up_proj",
    "feed_forward.experts.w2_weight": "mlp.experts.down_proj",
    "feed_forward.post_expert_norm.weight": "mlp.experts.post_expert_norm.weight",
    "feed_forward.unpooler.weight": "mlp.post_expert_proj.weight",
}

_VISION_TOP_LEVEL_RENAMES = {
    "vision_model.positional_embedding_vlm": (
        "model.vision_tower.patch_embedder.position_embedding_table.weight"
    ),
    "vision_model.conv1_linear.weight": (
        "model.vision_tower.patch_embedder.patch_embedding.weight"
    ),
    "vision_model.ln_pre.weight": "model.vision_tower.ln_pre.weight",
    "vision_model.ln_pre.bias": "model.vision_tower.ln_pre.bias",
    "vision_model.ln_post.weight": "model.vision_tower.ln_post.weight",
    "vision_model.ln_post.bias": "model.vision_tower.ln_post.bias",
    "vision_model.adapter.c_fc.weight": "model.vision_adapter.fc1.weight",
    "vision_model.adapter.c_proj.weight": "model.vision_adapter.fc2.weight",
    "modalities.vision_perception.encoder.vision_projection.weight": (
        "model.vision_projection.weight"
    ),
}

_VISION_LAYER_RENAMES = {
    "attn.q_proj.weight": "attn.q_proj.weight",
    "attn.q_proj.bias": "attn.q_proj.bias",
    "attn.k_proj.weight": "attn.k_proj.weight",
    "attn.k_proj.bias": "attn.k_proj.bias",
    "attn.v_proj.weight": "attn.v_proj.weight",
    "attn.v_proj.bias": "attn.v_proj.bias",
    "attn.out_proj.weight": "attn.proj.weight",
    "attn.out_proj.bias": "attn.proj.bias",
    "ln_1.weight": "norm1.weight",
    "ln_1.bias": "norm1.bias",
    "ln_2.weight": "norm2.weight",
    "ln_2.bias": "norm2.bias",
    "mlp.c_fc.weight": "mlp.fc1.weight",
    "mlp.c_fc.bias": "mlp.fc1.bias",
    "mlp.c_proj.weight": "mlp.fc2.weight",
    "mlp.c_proj.bias": "mlp.fc2.bias",
}

_AUDIO_TOP_LEVEL_RENAMES = {
    "norm.weight": "norm.weight",
    "o_proj.weight": "proj.weight",
    "o_proj.bias": "proj.bias",
    "dual_norm.weight": "post_proj_norm.weight",
}

_AUDIO_PATCH_RENAMES = {
    "linear.weight": "patch_embedder.proj.weight",
    "linear.bias": "patch_embedder.proj.bias",
    "norm.weight": "patch_embedder.norm.weight",
}

_AUDIO_BLOCK_RENAMES = {
    "conv.conv.weight": "conv.weight",
    "conv.conv.bias": "conv.bias",
    "norm.weight": "norm.weight",
    "linear1.weight": "mlp.fc1.weight",
    "linear1.bias": "mlp.fc1.bias",
    "linear2.weight": "mlp.fc2.weight",
    "linear2.bias": "mlp.fc2.bias",
}


def _transformers_checkpoint_key(key: str) -> str:
    if key in _TEXT_TOP_LEVEL_RENAMES:
        return _TEXT_TOP_LEVEL_RENAMES[key]
    if key in _VISION_TOP_LEVEL_RENAMES:
        return _VISION_TOP_LEVEL_RENAMES[key]

    match = re.match(r"model\.layers\.(\d+)\.(.+)", key)
    if match is not None:
        layer_id, suffix = match.groups()
        return f"model.language_model.layers.{layer_id}.{_TEXT_LAYER_RENAMES[suffix]}"

    match = re.match(r"vision_model\.encoder\.layers\.(\d+)\.(.+)", key)
    if match is not None:
        layer_id, suffix = match.groups()
        return f"model.vision_tower.layers.{layer_id}.{_VISION_LAYER_RENAMES[suffix]}"

    audio_prefix = "modalities.soft_audio_perception.interaction_encoder."
    if key.startswith(audio_prefix):
        suffix = key.removeprefix(audio_prefix)
        if suffix in _AUDIO_TOP_LEVEL_RENAMES:
            return f"model.audio_tower.{_AUDIO_TOP_LEVEL_RENAMES[suffix]}"
        match = re.match(r"stages\.(\d+)\.(\d+)\.(.+)", suffix)
        if match is not None:
            stage_id, block_id_text, module = match.groups()
            block_id = int(block_id_text)
            if block_id == 0:
                return f"model.audio_tower.layers.{stage_id}.{_AUDIO_PATCH_RENAMES[module]}"
            return (
                f"model.audio_tower.layers.{stage_id}.conv_blocks."
                f"{block_id - 1}.{_AUDIO_BLOCK_RENAMES[module]}"
            )
    raise ValueError(f"unsupported Muse Spark checkpoint key: {key}")


@dataclass(frozen=True)
class _MappingSpec:
    checkpoint_key: str
    native_keys: tuple[str, ...]
    transform: Literal["identity", "transpose", "packed_w13", "rope"] = "identity"
    num_heads: int | None = None


@dataclass(frozen=True)
class _TensorMetadata:
    shape: tuple[int, ...]
    dtype: torch.dtype


def _permute_rope_rows(tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
    head_dim = tensor.shape[0] // num_heads
    return (
        tensor.reshape(num_heads, head_dim // 2, 2, *tensor.shape[1:])
        .transpose(1, 2)
        .reshape(tensor.shape)
    )


def _unpermute_rope_rows(tensor: torch.Tensor, num_heads: int) -> torch.Tensor:
    head_dim = tensor.shape[0] // num_heads
    return (
        tensor.reshape(num_heads, 2, head_dim // 2, *tensor.shape[1:])
        .transpose(1, 2)
        .reshape(tensor.shape)
    )


def _is_row_sharded(tensor: torch.Tensor) -> bool:
    return isinstance(tensor, DTensor) and any(
        placement.is_shard(0) for placement in tensor.placements
    )


def _is_feature_sharded(w13: torch.Tensor) -> bool:
    """Whether ``w13.flatten(1, 2)`` would give each rank non-contiguous rows.

    ``w13`` is ``(E, 2, F, D)`` (dim 1: gate, up), and ``gate_up_proj`` is
    ``(E, 2F, D)``. With ``Shard(2)`` on ranks A and B and F = 4, one expert's
    rows are:

        gate_up_proj row:  0   1   2   3   4   5   6   7
                           g0  g1  g2  g3  u0  u1  u2  u3
        owner:             A   A   B   B   A   A   B   B

    torch_checkpointing loads a shard as one contiguous slice, so it would
    silently fill A's view with g0-g3. Such a ``w13`` is copied instead.
    Dim 1 is never sharded in TorchTitan but is checked the same way.
    """
    return isinstance(w13, DTensor) and any(
        isinstance(placement, Shard) and placement.dim in (1, 2)
        for placement in w13.placements
    )


def _permute_rope_tensors(
    tensors: Sequence[torch.Tensor],
    num_heads: int,
    permute: Callable[[torch.Tensor, int], torch.Tensor],
) -> list[torch.Tensor]:
    """Apply a RoPE row permutation to row-sharded DTensors that share rows and sharding.

    A row shard can hold part of a head, which a head-wise reshape cannot split,
    so the tensors are gathered first. Viewing each as ``(rows, -1)`` and
    concatenating along columns lets the whole group share one gather; the result
    is sharded like the inputs again.
    """
    first = cast(DTensor, tensors[0])
    mesh, placements = first.device_mesh, first.placements
    columns = [tensor.reshape(tensor.shape[0], -1) for tensor in tensors]
    fused = cast(DTensor, torch.cat(columns, dim=1)).redistribute(
        mesh, [Replicate()] * mesh.ndim
    )
    fused = cast(DTensor, permute(fused, num_heads)).redistribute(mesh, placements)
    widths = [column.shape[1] for column in columns]
    return [
        part.reshape(tensor.shape)
        for part, tensor in zip(fused.split(widths, dim=1), tensors, strict=True)
    ]


def _rope_group_key(tensor: DTensor, num_heads: int) -> tuple[Any, ...]:
    return (
        tensor.device_mesh,
        tensor.placements,
        tensor.shape[0],
        tensor.dtype,
        num_heads,
    )


class MuseSparkStateDictAdapter(MoEStateDictAdapter):
    """Map TorchTitan Muse Spark Open state to the Transformers checkpoint schema.

    ``to_hf`` returns real tensors that any checkpointer can load into, keeping
    the order ``to_hf(target DTensors) -> load -> from_hf``:

    - Packed expert ``gate_up_proj`` (gate then up rows along dimension 1) is
      a view of the native fused ``(E, 2, F, D)`` ``w13`` weight, unless
      ``w13`` is sharded on its gate/up or feature dimension.
    - Q and K use split-half RoPE order in the checkpoint, so they are
      permuted copies.

    Pipeline ranks may provide any union of complete top-level model components.
    This matches ``ModelWrapper`` state from one or more local pipeline chunks
    while still rejecting partial layers and unknown keys.
    """

    def __init__(
        self,
        model_config: MuseSparkModel.Config,
        hf_assets_path: str | None,
    ) -> None:
        super().__init__(model_config, hf_assets_path=None)
        self.hf_assets_path = hf_assets_path
        self._native_metadata = self._build_native_metadata(model_config)
        self._lora_native_keys = {
            key
            for key in self._native_metadata
            if re.search(r"\.lora_[ab]\.weight$", key)
        }
        self._mapping_specs = self._build_mapping_specs()
        self._validate_mapping_specs()
        self._native_keys_by_pipeline_component = (
            self._build_native_keys_by_pipeline_component()
        )
        self._checkpoint_keys_by_pipeline_component = (
            self._build_checkpoint_keys_by_pipeline_component()
        )
        self._checkpoint_metadata = self._build_checkpoint_metadata()
        self.fqn_to_index_mapping = None

    def _routed_expert_fqns(self) -> Iterable[str]:
        """Yield the native routed-expert path for every Muse MoE layer."""
        for fqn, _, _, _ in self.model_config.traverse(MoE.Config):
            yield f"{fqn}.routed_experts"

    def _native_fused_linears_to_hf(
        self,
        state_dict: dict[str, Any],
        *,
        split_routed_experts: bool = False,
    ) -> dict[str, Any]:
        """Expose logical W1/W3 keys after an expert-backend transform."""
        result = super()._native_fused_linears_to_hf(
            state_dict,
            split_routed_experts=split_routed_experts,
        )
        if split_routed_experts:
            for experts_fqn in self._routed_expert_fqns():
                self._split_stacked_linear(
                    result,
                    fused_key=f"{experts_fqn}.w13.weight",
                    logical_keys=(
                        f"{experts_fqn}.w1_EFD",
                        f"{experts_fqn}.w3_EFD",
                    ),
                    dim=1,
                )
        return result

    def _native_fused_linears_from_hf(
        self,
        state_dict: dict[str, Any],
        *,
        fuse_routed_experts: bool = False,
    ) -> dict[str, Any]:
        """Restore native W13 keys after an expert-backend transform."""
        result = super()._native_fused_linears_from_hf(
            state_dict,
            fuse_routed_experts=fuse_routed_experts,
        )
        if fuse_routed_experts:
            for experts_fqn in self._routed_expert_fqns():
                self._stack_logical_linears(
                    result,
                    fused_key=f"{experts_fqn}.w13.weight",
                    logical_keys=(
                        f"{experts_fqn}.w1_EFD",
                        f"{experts_fqn}.w3_EFD",
                    ),
                    dim=1,
                )
        return result

    def _build_native_metadata(
        self,
        config: MuseSparkModel.Config,
    ) -> dict[str, _TensorMetadata]:
        with torch.device("meta"):
            model = config.build()
        state_dict = self._native_fused_linears_to_hf(
            model.state_dict(),
            split_routed_experts=True,
        )
        return {
            key: _TensorMetadata(tuple(value.shape), value.dtype)
            for key, value in state_dict.items()
        }

    def _build_mapping_specs(self) -> list[_MappingSpec]:
        specs = self._build_unwrapped_mapping_specs()
        from torchtitan.models.muse_spark.model import MuseSparkAttention

        config = cast(Any, self.model_config)
        result: list[_MappingSpec] = []
        for spec in specs:
            native_key = spec.native_keys[0]
            transform = spec.transform
            num_heads: int | None = None
            if native_key.endswith("attention.qkv_linear.wq.weight"):
                layer_id = int(native_key.split(".")[1])
                attention = cast(
                    MuseSparkAttention.Config, config.layers[layer_id].attention
                )
                transform = "rope"
                num_heads = attention.n_heads
            elif native_key.endswith("attention.qkv_linear.wk.weight"):
                layer_id = int(native_key.split(".")[1])
                attention = cast(
                    MuseSparkAttention.Config, config.layers[layer_id].attention
                )
                transform = "rope"
                num_heads = attention.n_kv_heads
            elif re.search(r"vision_encoder\.layers\.\d+\.attn\.w[qk]\.", native_key):
                assert config.vision_encoder is not None
                transform = "rope"
                num_heads = config.vision_encoder.num_heads
            elif native_key.endswith("feed_forward.router.gate.weight"):
                transform = "identity"
            result.append(
                _MappingSpec(
                    _transformers_checkpoint_key(spec.checkpoint_key),
                    spec.native_keys,
                    transform,
                    num_heads,
                )
            )
        return result

    def _build_unwrapped_mapping_specs(self) -> list[_MappingSpec]:
        from torchtitan.models.muse_spark.model import (
            MuseSparkAttention,
            MuseSparkRMSNorm,
        )

        config = cast(Any, self.model_config)
        specs = [
            _MappingSpec(
                "model.embed_tokens.weight",
                ("tok_embeddings.embedding.weight",),
            )
        ]
        for layer_id, layer in enumerate(config.layers):
            checkpoint = f"model.layers.{layer_id}"
            native = f"layers.{layer_id}"
            attention = cast(MuseSparkAttention.Config, layer.attention)
            post_attention_norm = cast(
                MuseSparkRMSNorm.Config,
                layer.post_attention_norm,
            )
            specs.extend(
                [
                    _MappingSpec(
                        f"{checkpoint}.self_attn.q_proj.weight",
                        (f"{native}.attention.qkv_linear.wq.weight",),
                    ),
                    _MappingSpec(
                        f"{checkpoint}.self_attn.k_proj.weight",
                        (f"{native}.attention.qkv_linear.wk.weight",),
                    ),
                    _MappingSpec(
                        f"{checkpoint}.self_attn.v_proj.weight",
                        (f"{native}.attention.qkv_linear.wv.weight",),
                    ),
                    _MappingSpec(
                        f"{checkpoint}.self_attn.o_proj.weight",
                        (f"{native}.attention.wo.weight",),
                    ),
                    _MappingSpec(
                        f"{checkpoint}.self_attn.pre_qkv_proj_norm.weight",
                        (f"{native}.attention_norm.weight",),
                    ),
                    _MappingSpec(
                        f"{checkpoint}.add_attn.gate",
                        (f"{native}.attention_residual.gate",),
                    ),
                    _MappingSpec(
                        f"{checkpoint}.pre_ffn_norm.weight",
                        (f"{native}.ffn_norm.weight",),
                    ),
                    _MappingSpec(
                        f"{checkpoint}.post_ffn_norm.weight",
                        (f"{native}.post_ffn_norm.weight",),
                    ),
                    _MappingSpec(
                        f"{checkpoint}.add_ffn.gate",
                        (f"{native}.ffn_residual.gate",),
                    ),
                ]
            )
            if attention.output_gate is not None:
                specs.append(
                    _MappingSpec(
                        f"{checkpoint}.self_attn.o_gate_proj.weight",
                        (f"{native}.attention.output_gate.weight",),
                    )
                )
            if post_attention_norm.elementwise_affine:
                specs.append(
                    _MappingSpec(
                        f"{checkpoint}.post_attn_norm.weight",
                        (f"{native}.post_attention_norm.weight",),
                    )
                )

            if isinstance(layer.feed_forward, DenseFeedForward.Config):
                specs.extend(
                    [
                        _MappingSpec(
                            f"{checkpoint}.feed_forward.fc1_weight.w1.weight",
                            (f"{native}.feed_forward.w1.weight",),
                        ),
                        _MappingSpec(
                            f"{checkpoint}.feed_forward.fc1_weight.w3.weight",
                            (f"{native}.feed_forward.w3.weight",),
                        ),
                        _MappingSpec(
                            f"{checkpoint}.feed_forward.fc2_weight.w2.weight",
                            (f"{native}.feed_forward.w2.weight",),
                        ),
                    ]
                )
                continue

            moe = f"{native}.feed_forward"
            experts = f"{moe}.routed_experts"
            specs.extend(
                [
                    _MappingSpec(
                        f"{checkpoint}.feed_forward.router.weight",
                        (f"{moe}.router.gate.weight",),
                        "transpose",
                    ),
                    _MappingSpec(
                        f"{checkpoint}.feed_forward.expert_bias",
                        (f"{moe}.expert_bias_E",),
                    ),
                    _MappingSpec(
                        f"{checkpoint}.feed_forward.pooler.weight",
                        (f"{moe}.pooler.weight",),
                    ),
                    _MappingSpec(
                        f"{checkpoint}.feed_forward.pre_expert_norm.weight",
                        (f"{moe}.pre_expert_norm.weight",),
                    ),
                    _MappingSpec(
                        f"{checkpoint}.feed_forward.experts.w13_weight",
                        (f"{experts}.w1_EFD", f"{experts}.w3_EFD"),
                        "packed_w13",
                    ),
                    _MappingSpec(
                        f"{checkpoint}.feed_forward.experts.w2_weight",
                        (f"{experts}.w2.weight",),
                    ),
                    _MappingSpec(
                        f"{checkpoint}.feed_forward.post_expert_norm.weight",
                        (f"{experts}.output_postprocess.weight",),
                    ),
                    _MappingSpec(
                        f"{checkpoint}.feed_forward.unpooler.weight",
                        (f"{moe}.unpooler.weight",),
                    ),
                ]
            )

        specs.extend(
            [
                _MappingSpec("model.norm.weight", ("norm.weight",)),
                _MappingSpec("lm_head.weight", ("lm_head.weight",)),
            ]
        )
        specs.extend(self._build_vision_mapping_specs(config))
        specs.extend(self._build_audio_mapping_specs(config))
        return specs

    @staticmethod
    def _build_vision_mapping_specs(
        config: MuseSparkModel.Config,
    ) -> list[_MappingSpec]:
        vision_encoder = config.vision_encoder
        if vision_encoder is None:
            return []

        checkpoint = "vision_model"
        native = "vision_encoder"
        specs = [
            _MappingSpec(
                f"{checkpoint}.positional_embedding_vlm",
                (f"{native}.positional_embedding_vlm",),
            ),
            _MappingSpec(
                f"{checkpoint}.conv1_linear.weight",
                (f"{native}.conv1_linear.weight",),
            ),
        ]
        for norm in ("ln_pre", "ln_post"):
            for parameter in ("weight", "bias"):
                specs.append(
                    _MappingSpec(
                        f"{checkpoint}.{norm}.{parameter}",
                        (f"{native}.{norm}.{parameter}",),
                    )
                )

        for layer_id in range(vision_encoder.num_layers):
            checkpoint_layer = f"{checkpoint}.encoder.layers.{layer_id}"
            native_layer = f"{native}.layers.{layer_id}"
            for checkpoint_name, native_name in (
                ("q_proj", "wq"),
                ("k_proj", "wk"),
                ("v_proj", "wv"),
            ):
                for parameter in ("weight", "bias"):
                    specs.append(
                        _MappingSpec(
                            f"{checkpoint_layer}.attn.{checkpoint_name}.{parameter}",
                            (f"{native_layer}.attn.{native_name}.{parameter}",),
                        )
                    )
            for checkpoint_name, native_name in (
                ("attn.out_proj", "attn.proj"),
                ("ln_1", "norm1"),
                ("ln_2", "norm2"),
                ("mlp.c_fc", "mlp.linear_fc1"),
                ("mlp.c_proj", "mlp.linear_fc2"),
            ):
                for parameter in ("weight", "bias"):
                    specs.append(
                        _MappingSpec(
                            f"{checkpoint_layer}.{checkpoint_name}.{parameter}",
                            (f"{native_layer}.{native_name}.{parameter}",),
                        )
                    )

        for projection in ("c_fc", "c_proj"):
            specs.append(
                _MappingSpec(
                    f"{checkpoint}.adapter.{projection}.weight",
                    (f"vision_adapter.{projection}.weight",),
                )
            )
        specs.append(
            _MappingSpec(
                "modalities.vision_perception.encoder.vision_projection.weight",
                ("vision_projection.weight",),
            )
        )
        return specs

    @staticmethod
    def _build_audio_mapping_specs(
        config: MuseSparkModel.Config,
    ) -> list[_MappingSpec]:
        audio_encoder = config.audio_encoder
        if audio_encoder is None:
            return []

        checkpoint = "modalities.soft_audio_perception.interaction_encoder"
        native = "audio_encoder"
        specs = []
        for stage_id in range(len(audio_encoder.layers)):
            checkpoint_stage = f"{checkpoint}.stages.{stage_id}"
            native_stage = f"{native}.layers.{stage_id}"
            for parameter in ("weight", "bias"):
                specs.append(
                    _MappingSpec(
                        f"{checkpoint_stage}.0.linear.{parameter}",
                        (f"{native_stage}.patchifier.linear.{parameter}",),
                    )
                )
            specs.append(
                _MappingSpec(
                    f"{checkpoint_stage}.0.norm.weight",
                    (f"{native_stage}.patchifier.norm.weight",),
                )
            )
            for block_id, block_name in ((1, "block1"), (2, "block2")):
                for module_name, parameters in (
                    ("conv.conv", ("weight", "bias")),
                    ("norm", ("weight",)),
                    ("linear1", ("weight", "bias")),
                    ("linear2", ("weight", "bias")),
                ):
                    for parameter in parameters:
                        specs.append(
                            _MappingSpec(
                                f"{checkpoint_stage}.{block_id}.{module_name}."
                                f"{parameter}",
                                (
                                    f"{native_stage}.{block_name}.{module_name}.{parameter}",
                                ),
                            )
                        )

        for module_name, parameters in (
            ("norm", ("weight",)),
            ("o_proj", ("weight", "bias")),
            ("dual_norm", ("weight",)),
        ):
            for parameter in parameters:
                specs.append(
                    _MappingSpec(
                        f"{checkpoint}.{module_name}.{parameter}",
                        (f"{native}.{module_name}.{parameter}",),
                    )
                )
        return specs

    def _validate_mapping_specs(self) -> None:
        checkpoint_keys: set[str] = set()
        native_keys: set[str] = set()
        for spec in self._mapping_specs:
            if spec.checkpoint_key in checkpoint_keys:
                raise ValueError(f"duplicate checkpoint mapping: {spec.checkpoint_key}")
            checkpoint_keys.add(spec.checkpoint_key)
            for native_key in spec.native_keys:
                if native_key in native_keys:
                    raise ValueError(f"duplicate native mapping: {native_key}")
                if native_key not in self._native_metadata:
                    raise ValueError(
                        f"mapping references unknown native state: {native_key}"
                    )
                native_keys.add(native_key)

        unmapped = set(self._native_metadata) - native_keys - self._lora_native_keys
        if unmapped:
            raise ValueError("unmapped native state: " + ", ".join(sorted(unmapped)))

    @staticmethod
    def _pipeline_component_for_native_key(key: str) -> str:
        if key.startswith("layers."):
            return ".".join(key.split(".", 2)[:2])
        return key.split(".", 1)[0]

    def _build_native_keys_by_pipeline_component(self) -> dict[str, set[str]]:
        result: dict[str, set[str]] = {}
        for spec in self._mapping_specs:
            for key in spec.native_keys:
                component = self._pipeline_component_for_native_key(key)
                result.setdefault(component, set()).add(key)
        return result

    def _build_checkpoint_keys_by_pipeline_component(
        self,
    ) -> dict[str, set[str]]:
        result: dict[str, set[str]] = {}
        for spec in self._mapping_specs:
            components = {
                self._pipeline_component_for_native_key(key) for key in spec.native_keys
            }
            if len(components) != 1:
                raise ValueError(
                    "checkpoint mapping crosses pipeline components: "
                    f"{spec.checkpoint_key}"
                )
            component = next(iter(components))
            result.setdefault(component, set()).add(spec.checkpoint_key)
        return result

    @staticmethod
    def _validate_pipeline_local_key_set(
        provided: set[str],
        expected_by_component: dict[str, set[str]],
        *,
        missing_label: str,
        unexpected_label: str,
    ) -> None:
        expected = set().union(*expected_by_component.values())
        unexpected = provided - expected
        if unexpected:
            raise ValueError(f"{unexpected_label}: " + ", ".join(sorted(unexpected)))

        included_components = {
            component
            for component, component_keys in expected_by_component.items()
            if component_keys & provided
        }
        required = set().union(
            *(expected_by_component[component] for component in included_components)
        )
        missing = required - provided

        if missing or not provided:
            missing = missing or expected
            raise ValueError(f"{missing_label}: " + ", ".join(sorted(missing)))

    def _build_checkpoint_metadata(self) -> dict[str, _TensorMetadata]:
        metadata = {}
        for spec in self._mapping_specs:
            native = [self._native_metadata[key] for key in spec.native_keys]
            if spec.transform in {"identity", "rope"}:
                checkpoint = native[0]
                if spec.transform == "rope" and (
                    spec.num_heads is None
                    or checkpoint.shape[0] % (2 * spec.num_heads) != 0
                ):
                    raise ValueError(
                        "RoPE mapping requires an even head dimension: "
                        f"{spec.checkpoint_key}"
                    )
            elif spec.transform == "transpose":
                if len(native[0].shape) != 2:
                    raise ValueError(
                        f"transpose mapping requires a matrix: {spec.checkpoint_key}"
                    )
                checkpoint = _TensorMetadata(
                    (native[0].shape[1], native[0].shape[0]),
                    native[0].dtype,
                )
            else:
                w1, w3 = native
                if (
                    len(w1.shape) != 3
                    or len(w3.shape) != 3
                    or w1.shape[0] != w3.shape[0]
                    or w1.shape[2] != w3.shape[2]
                    or w1.dtype != w3.dtype
                ):
                    raise ValueError(
                        "packed_w13 mapping requires compatible rank-3 weights: "
                        f"{spec.checkpoint_key}"
                    )
                checkpoint = _TensorMetadata(
                    (w1.shape[0], w1.shape[1] + w3.shape[1], w1.shape[2]),
                    w1.dtype,
                )
            metadata[spec.checkpoint_key] = checkpoint

        return metadata

    def _classify_checkpoint_keys(
        self,
        keys: set[str],
    ) -> tuple[set[str], set[str], set[str], set[str]]:
        expected = set(self._checkpoint_metadata)
        mapped = keys & expected
        extra = keys - mapped
        ignored_prefixes = (
            "model.vision_tower.",
            "model.vision_adapter.",
            "model.vision_projection.",
            "model.audio_tower.",
        )
        modalities = {key for key in extra if key.startswith(ignored_prefixes)}
        unknown = extra - modalities
        return mapped, modalities, set(), unknown

    def _validate_checkpoint_key_set(
        self,
        keys: set[str],
    ) -> tuple[set[str], set[str], set[str]]:
        mapped, modalities, vision, unknown = self._classify_checkpoint_keys(keys)
        missing = set(self._checkpoint_metadata) - mapped
        if missing:
            raise ValueError("missing checkpoint keys: " + ", ".join(sorted(missing)))
        if unknown:
            raise ValueError(
                "unexpected checkpoint keys: " + ", ".join(sorted(unknown))
            )
        return mapped, modalities, vision

    def _validate_checkpoint_index(self) -> None:
        if self.hf_assets_path is None:
            self.fqn_to_index_mapping = None
            return
        index_path = os.path.join(
            self.hf_assets_path,
            "model.safetensors.index.json",
        )
        try:
            with open(index_path) as index_file:
                index = json.load(index_file)
        except FileNotFoundError:
            self.fqn_to_index_mapping = None
            return

        weight_map = index.get("weight_map")
        if not isinstance(weight_map, Mapping):
            raise ValueError("checkpoint index must contain a weight_map object")
        weight_map = cast(Mapping[str, Any], weight_map)
        mapped, _, _ = self._validate_checkpoint_key_set(set(weight_map))
        fqn_to_index_mapping = {}
        for key in mapped:
            filename = weight_map[key]
            if not isinstance(filename, str):
                raise ValueError(f"checkpoint shard filename must be a string: {key}")
            match = re.search(r"-(\d+)-of-\d+\.safetensors$", filename)
            if match is None:
                raise ValueError(
                    f"invalid sharded safetensors filename for {key}: {filename}"
                )
            fqn_to_index_mapping[key] = int(match.group(1))
        self.fqn_to_index_mapping = fqn_to_index_mapping

    @staticmethod
    def _validate_tensor(
        key: str,
        tensor: torch.Tensor,
        metadata: _TensorMetadata,
    ) -> None:
        if tuple(tensor.shape) != metadata.shape:
            raise ValueError(
                f"shape mismatch for {key}: expected {metadata.shape}, "
                f"got {tuple(tensor.shape)}"
            )

    def to_hf(
        self, state_dict: dict[str, Any], *, to_quantized_layout: bool = False
    ) -> dict[str, Any]:
        del to_quantized_layout
        self._validate_checkpoint_index()
        fused_w13: dict[tuple[str, ...], Any] = {
            (f"{fqn}.w1_EFD", f"{fqn}.w3_EFD"): state_dict[f"{fqn}.w13.weight"]
            for fqn in self._routed_expert_fqns()
            if f"{fqn}.w13.weight" in state_dict
        }
        state_dict = self._native_fused_linears_to_hf(
            state_dict,
            split_routed_experts=True,
        )
        provided_native = set(state_dict) - self._lora_native_keys
        self._validate_pipeline_local_key_set(
            provided_native,
            self._native_keys_by_pipeline_component,
            missing_label="missing native keys",
            unexpected_label="unexpected native keys",
        )

        result: dict[str, Any] = {}
        sharded_rope: dict[tuple[Any, ...], list[tuple[str, DTensor]]] = {}
        for spec in self._mapping_specs:
            if not set(spec.native_keys) <= provided_native:
                continue
            tensors = []
            for native_key in spec.native_keys:
                # pop, so each replicated Q/K copy is freed once it is permuted.
                tensor = state_dict.pop(native_key)
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError(f"native state must be a tensor: {native_key}")
                self._validate_tensor(
                    native_key, tensor, self._native_metadata[native_key]
                )
                tensors.append(tensor)
            if (
                spec.transform == "packed_w13"
                and len({tensor.dtype for tensor in tensors}) != 1
            ):
                raise ValueError(
                    "packed_w13 native weights must use the same dtype: "
                    f"{spec.native_keys}"
                )
            w13 = fused_w13.get(spec.native_keys)
            if spec.transform == "rope" and _is_row_sharded(tensors[0]):
                assert spec.num_heads is not None
                tensor = cast(DTensor, tensors[0])
                key = _rope_group_key(tensor, spec.num_heads)
                sharded_rope.setdefault(key, []).append((spec.checkpoint_key, tensor))
                continue
            if spec.transform == "identity":
                checkpoint_tensor = tensors[0]
            elif spec.transform == "transpose":
                checkpoint_tensor = tensors[0].transpose(0, 1).contiguous()
            elif spec.transform == "rope":
                assert spec.num_heads is not None
                checkpoint_tensor = _permute_rope_rows(tensors[0], spec.num_heads)
            elif w13 is not None and not _is_feature_sharded(w13):
                # Gate and up rows are adjacent in the fused (E, 2, F, D) weight,
                # so flattening it gives the checkpoint's (E, 2F, D) layout
                # without a copy.
                checkpoint_tensor = w13.flatten(1, 2)
            else:
                checkpoint_tensor = torch.cat(tensors, dim=1)
            result[spec.checkpoint_key] = checkpoint_tensor
        for key, entries in sharded_rope.items():
            permuted = _permute_rope_tensors(
                [tensor for _, tensor in entries], key[-1], _permute_rope_rows
            )
            result.update(zip((name for name, _ in entries), permuted, strict=True))
        return result

    def from_hf(
        self, hf_state_dict: dict[str, Any], *, is_quantized_layout: bool = False
    ) -> dict[str, Any]:
        del is_quantized_layout
        keys = set(hf_state_dict)
        mapped, modalities, vision, unknown = self._classify_checkpoint_keys(keys)
        if modalities or vision:
            self._validate_checkpoint_key_set(
                keys,
            )
        else:
            if unknown:
                raise ValueError(
                    "unexpected checkpoint keys: " + ", ".join(sorted(unknown))
                )
            self._validate_pipeline_local_key_set(
                mapped,
                self._checkpoint_keys_by_pipeline_component,
                missing_label="missing checkpoint keys",
                unexpected_label="unexpected checkpoint keys",
            )

        result = {}
        sharded_rope: dict[tuple[Any, ...], list[tuple[str, DTensor]]] = {}
        # Mapping order, not set order: every rank must gather the RoPE groups
        # in the same order.
        for spec in self._mapping_specs:
            if spec.checkpoint_key not in mapped:
                continue
            tensor = hf_state_dict[spec.checkpoint_key]
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(
                    f"checkpoint state must be a tensor: {spec.checkpoint_key}"
                )
            self._validate_tensor(
                spec.checkpoint_key,
                tensor,
                self._checkpoint_metadata[spec.checkpoint_key],
            )
            if spec.transform == "identity":
                result[spec.native_keys[0]] = tensor
            elif spec.transform == "transpose":
                result[spec.native_keys[0]] = tensor.transpose(0, 1)
            elif spec.transform == "rope" and _is_row_sharded(tensor):
                assert spec.num_heads is not None
                tensor = cast(DTensor, tensor)
                key = _rope_group_key(tensor, spec.num_heads)
                entry = (spec.native_keys[0], tensor)
                sharded_rope.setdefault(key, []).append(entry)
            elif spec.transform == "rope":
                assert spec.num_heads is not None
                # to_hf returns RoPE weights as copies, so reorder in place
                # rather than hold a second copy of every replicated Q and K.
                with torch.no_grad():
                    tensor.copy_(_unpermute_rope_rows(tensor, spec.num_heads))
                result[spec.native_keys[0]] = tensor
            else:
                # Restore w13 directly; after a load into to_hf's view of w13,
                # this is a view of the parameter itself.
                w13_key = spec.native_keys[0].removesuffix("w1_EFD") + "w13.weight"
                result[w13_key] = tensor.unflatten(1, (2, -1))
        for key, entries in sharded_rope.items():
            unpermuted = _permute_rope_tensors(
                [tensor for _, tensor in entries], key[-1], _unpermute_rope_rows
            )
            result.update(zip((name for name, _ in entries), unpermuted, strict=True))
        return self._native_fused_linears_from_hf(
            result,
            fuse_routed_experts=True,
        )

    def get_hf_storage_reader(
        self,
        path: str,
        from_quantized: bool = False,
        *,
        thread_count: int | None = None,
    ) -> HuggingFaceStorageReader:
        if thread_count is None:
            thread_count = _HF_LOAD_THREAD_COUNT
        reader = super().get_hf_storage_reader(
            path, from_quantized, thread_count=thread_count
        )
        metadata = reader.read_metadata().state_dict_metadata
        mapped, _, _ = self._validate_checkpoint_key_set(set(metadata))
        for key in mapped:
            actual = metadata[key]
            if not isinstance(actual, TensorStorageMetadata):
                raise ValueError(f"checkpoint metadata must describe a tensor: {key}")
            expected = self._checkpoint_metadata[key]
            if tuple(actual.size) != expected.shape:
                raise ValueError(
                    f"shape mismatch for {key}: expected {expected.shape}, "
                    f"got {tuple(actual.size)}"
                )
        return reader

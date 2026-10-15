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
from typing import Any, cast, TYPE_CHECKING

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
        self.from_hf_map = self._build_from_hf_map()
        self._validate_mapping()
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

    def _build_from_hf_map(self) -> dict[str, str]:
        config = cast(Any, self.model_config)
        mapping = {
            "model.language_model.embed_tokens.weight": (
                "tok_embeddings.embedding.weight"
            ),
        }
        for layer_id, layer in enumerate(config.layers):
            checkpoint = f"model.language_model.layers.{layer_id}"
            native = f"layers.{layer_id}"
            mapping.update(
                {
                    f"{checkpoint}.self_attn.q_proj.weight": (
                        f"{native}.attention.qkv_linear.wq.weight"
                    ),
                    f"{checkpoint}.self_attn.k_proj.weight": (
                        f"{native}.attention.qkv_linear.wk.weight"
                    ),
                    f"{checkpoint}.self_attn.v_proj.weight": (
                        f"{native}.attention.qkv_linear.wv.weight"
                    ),
                    f"{checkpoint}.self_attn.o_proj.weight": (
                        f"{native}.attention.wo.weight"
                    ),
                    f"{checkpoint}.input_layernorm.weight": (
                        f"{native}.attention_norm.weight"
                    ),
                    f"{checkpoint}.post_attention_residual_gate.gate": (
                        f"{native}.attention_residual.gate"
                    ),
                    f"{checkpoint}.pre_feedforward_layernorm.weight": (
                        f"{native}.ffn_norm.weight"
                    ),
                    f"{checkpoint}.post_feedforward_layernorm.weight": (
                        f"{native}.post_ffn_norm.weight"
                    ),
                    f"{checkpoint}.post_feedforward_residual_gate.gate": (
                        f"{native}.ffn_residual.gate"
                    ),
                }
            )
            if layer.attention.output_gate is not None:
                mapping[
                    f"{checkpoint}.self_attn.gate_proj.weight"
                ] = f"{native}.attention.output_gate.weight"
            if layer.post_attention_norm.elementwise_affine:
                mapping[
                    f"{checkpoint}.post_attention_layernorm.weight"
                ] = f"{native}.post_attention_norm.weight"

            if isinstance(layer.feed_forward, DenseFeedForward.Config):
                mapping.update(
                    {
                        f"{checkpoint}.mlp.gate_proj.weight": (
                            f"{native}.feed_forward.w1.weight"
                        ),
                        f"{checkpoint}.mlp.up_proj.weight": (
                            f"{native}.feed_forward.w3.weight"
                        ),
                        f"{checkpoint}.mlp.down_proj.weight": (
                            f"{native}.feed_forward.w2.weight"
                        ),
                    }
                )
                continue

            moe = f"{native}.feed_forward"
            experts = f"{moe}.routed_experts"
            mapping.update(
                {
                    f"{checkpoint}.mlp.gate.weight": f"{moe}.router.gate.weight",
                    f"{checkpoint}.mlp.gate.e_score_correction_bias": (
                        f"{moe}.expert_bias_E"
                    ),
                    f"{checkpoint}.mlp.pre_expert_proj.weight": f"{moe}.pooler.weight",
                    f"{checkpoint}.mlp.pre_expert_norm.weight": (
                        f"{moe}.pre_expert_norm.weight"
                    ),
                    f"{checkpoint}.mlp.experts.gate_up_proj": f"{experts}.w13.weight",
                    f"{checkpoint}.mlp.experts.down_proj": f"{experts}.w2.weight",
                    f"{checkpoint}.mlp.experts.post_expert_norm.weight": (
                        f"{experts}.output_postprocess.weight"
                    ),
                    f"{checkpoint}.mlp.post_expert_proj.weight": (
                        f"{moe}.unpooler.weight"
                    ),
                }
            )

        mapping.update(
            {
                "model.language_model.norm.weight": "norm.weight",
                "lm_head.weight": "lm_head.weight",
            }
        )
        mapping.update(self._build_vision_from_hf_map(config))
        mapping.update(self._build_audio_from_hf_map(config))
        return mapping

    @staticmethod
    def _build_vision_from_hf_map(config: MuseSparkModel.Config) -> dict[str, str]:
        vision_encoder = config.vision_encoder
        if vision_encoder is None:
            return {}

        checkpoint = "model.vision_tower"
        native = "vision_encoder"
        mapping = {
            f"{checkpoint}.patch_embedder.position_embedding_table.weight": (
                f"{native}.positional_embedding_vlm"
            ),
            f"{checkpoint}.patch_embedder.patch_embedding.weight": (
                f"{native}.conv1_linear.weight"
            ),
        }
        for norm in ("ln_pre", "ln_post"):
            for parameter in ("weight", "bias"):
                mapping[
                    f"{checkpoint}.{norm}.{parameter}"
                ] = f"{native}.{norm}.{parameter}"

        for layer_id in range(vision_encoder.num_layers):
            checkpoint_layer = f"{checkpoint}.layers.{layer_id}"
            native_layer = f"{native}.layers.{layer_id}"
            for checkpoint_name, native_name in (
                ("attn.q_proj", "attn.wq"),
                ("attn.k_proj", "attn.wk"),
                ("attn.v_proj", "attn.wv"),
                ("attn.proj", "attn.proj"),
                ("norm1", "norm1"),
                ("norm2", "norm2"),
                ("mlp.fc1", "mlp.linear_fc1"),
                ("mlp.fc2", "mlp.linear_fc2"),
            ):
                for parameter in ("weight", "bias"):
                    mapping[
                        f"{checkpoint_layer}.{checkpoint_name}.{parameter}"
                    ] = f"{native_layer}.{native_name}.{parameter}"

        for checkpoint_name, native_name in (("fc1", "c_fc"), ("fc2", "c_proj")):
            mapping[
                f"model.vision_adapter.{checkpoint_name}.weight"
            ] = f"vision_adapter.{native_name}.weight"
        mapping["model.vision_projection.weight"] = "vision_projection.weight"
        return mapping

    @staticmethod
    def _build_audio_from_hf_map(config: MuseSparkModel.Config) -> dict[str, str]:
        audio_encoder = config.audio_encoder
        if audio_encoder is None:
            return {}

        checkpoint = "model.audio_tower"
        native = "audio_encoder"
        mapping = {}
        for stage_id in range(len(audio_encoder.layers)):
            checkpoint_stage = f"{checkpoint}.layers.{stage_id}"
            native_stage = f"{native}.layers.{stage_id}"
            for parameter in ("weight", "bias"):
                mapping[
                    f"{checkpoint_stage}.patch_embedder.proj.{parameter}"
                ] = f"{native_stage}.patchifier.linear.{parameter}"
            mapping[
                f"{checkpoint_stage}.patch_embedder.norm.weight"
            ] = f"{native_stage}.patchifier.norm.weight"
            for block_id, block_name in ((0, "block1"), (1, "block2")):
                checkpoint_block = f"{checkpoint_stage}.conv_blocks.{block_id}"
                for checkpoint_name, native_name, parameters in (
                    ("conv", "conv.conv", ("weight", "bias")),
                    ("norm", "norm", ("weight",)),
                    ("mlp.fc1", "linear1", ("weight", "bias")),
                    ("mlp.fc2", "linear2", ("weight", "bias")),
                ):
                    for parameter in parameters:
                        mapping[
                            f"{checkpoint_block}.{checkpoint_name}.{parameter}"
                        ] = f"{native_stage}.{block_name}.{native_name}.{parameter}"

        for checkpoint_name, native_name, parameters in (
            ("norm", "norm", ("weight",)),
            ("proj", "o_proj", ("weight", "bias")),
            ("post_proj_norm", "dual_norm", ("weight",)),
        ):
            for parameter in parameters:
                mapping[
                    f"{checkpoint}.{checkpoint_name}.{parameter}"
                ] = f"{native}.{native_name}.{parameter}"
        return mapping

    @staticmethod
    def _mapped_native_keys(native_key: str) -> tuple[str, ...]:
        if native_key.endswith(".routed_experts.w13.weight"):
            prefix = native_key.removesuffix("w13.weight")
            return (f"{prefix}w1_EFD", f"{prefix}w3_EFD")
        return (native_key,)

    def _rope_num_heads(self, native_key: str) -> int | None:
        config = cast("MuseSparkModel.Config", self.model_config)
        if native_key.endswith(
            ("attention.qkv_linear.wq.weight", "attention.qkv_linear.wk.weight")
        ):
            layer_id = int(native_key.split(".")[1])
            attention = config.layers[layer_id].attention
            return (
                attention.n_heads
                if native_key.endswith("wq.weight")
                else attention.n_kv_heads
            )
        if re.fullmatch(
            r"vision_encoder\.layers\.\d+\.attn\.w[qk]\.(weight|bias)", native_key
        ):
            assert config.vision_encoder is not None
            return config.vision_encoder.num_heads
        return None

    def _validate_mapping(self) -> None:
        native_keys: set[str] = set()
        for mapped_key in self.from_hf_map.values():
            for native_key in self._mapped_native_keys(mapped_key):
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
        for native_key in self.from_hf_map.values():
            for key in self._mapped_native_keys(native_key):
                component = self._pipeline_component_for_native_key(key)
                result.setdefault(component, set()).add(key)
        return result

    def _build_checkpoint_keys_by_pipeline_component(
        self,
    ) -> dict[str, set[str]]:
        result: dict[str, set[str]] = {}
        for checkpoint_key, native_key in self.from_hf_map.items():
            component = self._pipeline_component_for_native_key(native_key)
            result.setdefault(component, set()).add(checkpoint_key)
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
        for checkpoint_key, native_key in self.from_hf_map.items():
            native_keys = self._mapped_native_keys(native_key)
            native = [self._native_metadata[key] for key in native_keys]
            if len(native_keys) == 1:
                checkpoint = native[0]
                num_heads = self._rope_num_heads(native_key)
                if num_heads is not None and checkpoint.shape[0] % (2 * num_heads) != 0:
                    raise ValueError(
                        "RoPE mapping requires an even head dimension: "
                        f"{checkpoint_key}"
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
                        f"{checkpoint_key}"
                    )
                checkpoint = _TensorMetadata(
                    (w1.shape[0], w1.shape[1] + w3.shape[1], w1.shape[2]),
                    w1.dtype,
                )
            metadata[checkpoint_key] = checkpoint

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
        fused_w13 = {
            f"{fqn}.w13.weight": state_dict[f"{fqn}.w13.weight"]
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
        to_hf_map = {value: key for key, value in self.from_hf_map.items()}
        for mapped_key, checkpoint_key in to_hf_map.items():
            native_keys = self._mapped_native_keys(mapped_key)
            if not set(native_keys) <= provided_native:
                continue
            tensors = []
            for native_key in native_keys:
                # pop, so each replicated Q/K copy is freed once it is permuted.
                tensor = state_dict.pop(native_key)
                if not isinstance(tensor, torch.Tensor):
                    raise ValueError(f"native state must be a tensor: {native_key}")
                self._validate_tensor(
                    native_key, tensor, self._native_metadata[native_key]
                )
                tensors.append(tensor)
            if len(native_keys) == 2 and len({tensor.dtype for tensor in tensors}) != 1:
                raise ValueError(
                    "packed_w13 native weights must use the same dtype: "
                    f"{native_keys}"
                )
            w13 = fused_w13.get(mapped_key)
            num_heads = self._rope_num_heads(mapped_key)
            if num_heads is not None and _is_row_sharded(tensors[0]):
                tensor = cast(DTensor, tensors[0])
                key = _rope_group_key(tensor, num_heads)
                sharded_rope.setdefault(key, []).append((checkpoint_key, tensor))
                continue
            if num_heads is not None:
                checkpoint_tensor = _permute_rope_rows(tensors[0], num_heads)
            elif w13 is not None and not _is_feature_sharded(w13):
                # Gate and up rows are adjacent in the fused (E, 2, F, D) weight,
                # so flattening it gives the checkpoint's (E, 2F, D) layout
                # without a copy.
                checkpoint_tensor = w13.flatten(1, 2)
            elif len(native_keys) == 2:
                checkpoint_tensor = torch.cat(tensors, dim=1)
            else:
                checkpoint_tensor = tensors[0]
            result[checkpoint_key] = checkpoint_tensor
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
        for checkpoint_key, native_key in self.from_hf_map.items():
            if checkpoint_key not in mapped:
                continue
            tensor = hf_state_dict[checkpoint_key]
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f"checkpoint state must be a tensor: {checkpoint_key}")
            self._validate_tensor(
                checkpoint_key,
                tensor,
                self._checkpoint_metadata[checkpoint_key],
            )
            num_heads = self._rope_num_heads(native_key)
            if num_heads is not None and _is_row_sharded(tensor):
                tensor = cast(DTensor, tensor)
                key = _rope_group_key(tensor, num_heads)
                entry = (native_key, tensor)
                sharded_rope.setdefault(key, []).append(entry)
            elif num_heads is not None:
                # to_hf returns RoPE weights as copies, so reorder in place
                # rather than hold a second copy of every replicated Q and K.
                with torch.no_grad():
                    tensor.copy_(_unpermute_rope_rows(tensor, num_heads))
                result[native_key] = tensor
            elif native_key.endswith(".routed_experts.w13.weight"):
                # Restore w13 directly; after a load into to_hf's view of w13,
                # this is a view of the parameter itself.
                result[native_key] = tensor.unflatten(1, (2, -1))
            else:
                result[native_key] = tensor
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

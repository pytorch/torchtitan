# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Inkling text weights retain their published layouts, including packed experts."""

from __future__ import annotations

import json
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import cast, TYPE_CHECKING

import torch
from safetensors import safe_open
from torch import nn
from torch.distributed import checkpoint as dcp
from torch.distributed.tensor import DTensor

from torchtitan.models.base.state_dict_adapter import StateDictAdapter

if TYPE_CHECKING:
    from torchtitan.experiments.inkling.model import InklingModel

_PREFIX = "model.llm."
_LORA = (".lora_a.weight", ".lora_b.weight")
_NVFP4 = {
    "amax_type": "per_token",
    "block_size": 16,
    "format": "nvfp4_e2m1",
    "rounding": "nearest",
    "scale_dtype": "float8_e4m3fn",
    "scale_selection_rule": "fouroversix_mae",
}
_MXFP8 = {
    "block_size": 32,
    "format": "mxfp8_e4m3",
    "rounding": "nearest",
    "scale_dtype": "ue8m0",
}
# Safetensors dtype names; None denotes a tensor forbidden in this format.
_NVFP4_STORAGE_DTYPES = {
    "": "U8",
    "_scale": "F8_E4M3",
    "_amax": "F32",
    "_outer_scale_denominator": "F32",
}
_MXFP8_STORAGE_DTYPES = {
    "": "F8_E4M3",
    "_scale": "U8",
    "_amax": None,
    "_outer_scale_denominator": None,
}


def _validate_checkpoint(source: Path, model: InklingModel.Config) -> None:
    text = json.loads((source / "config.json").read_bytes())["text_config"]

    def require(key: str, value: object) -> None:
        if text.get(key) != value:
            raise ValueError(
                f"Unsupported Inkling text_config.{key}: expected {value!r}, got {text.get(key)!r}"
            )

    shape = model.shape
    expected = {
        "model_type": "inkling2_loop_text",
        "hidden_size": shape.hidden_size,
        "num_hidden_layers": shape.num_layers,
        "num_attention_heads": shape.num_q_heads,
        "num_key_value_heads": shape.num_kv_heads,
        "head_dim": shape.head_dim,
        "swa_num_attention_heads": shape.num_q_heads,
        "swa_num_key_value_heads": shape.num_kv_heads,
        "swa_head_dim": shape.head_dim,
        "d_rel": shape.d_rel,
        "rel_extent": shape.rel_extent,
        "sliding_window_size": shape.sliding_window,
        "local_layer_ids": [
            i for i in range(shape.num_layers) if i not in shape.full_layers
        ],
        "sconv_kernel_size": shape.conv_kernel,
        "vocab_size": shape.padded_vocab_size,
        "unpadded_vocab_size": shape.vocab_size,
        "rms_norm_eps": shape.rms_eps,
        "qk_l2_clip_threshold": shape.qk_clip,
        "rel_l2_clip_threshold": shape.rel_clip,
        "log_scaling_alpha": shape.log_scaling_alpha,
        "log_scaling_n_floor": shape.log_scaling_floor,
        "n_layers_per_attn_res_block": shape.block_layers,
        "final_logit_softcapping": shape.logit_softcap,
        "q_bias": False,
        "o_bias": False,
        "use_value_norm_no_gamma": True,
        "attn_res_normalize_values": True,
        "n_shared_experts": 0,
        "use_gate_bias": True,
        "loop_config": {
            "core_start": shape.loop_start,
            "core_end": shape.loop_end,
            "num_passes": shape.num_passes,
            "gate_rank": shape.gate_rank,
            "is_varloop": False,
        },
    }
    # Some HF configurations supply layer_types or moe_intermediate_size in
    # addition to the published keys. WarpDrive gives these aliases precedence.
    if "layer_types" in text:
        layer_types = text["layer_types"]
        full = tuple(i for i, kind in enumerate(layer_types) if kind == "hybrid")
        if len(layer_types) != shape.num_layers or full != shape.full_layers:
            raise ValueError("Unsupported Inkling text_config.layer_types")
    for key, value in expected.items():
        require(key, value)
    if text.get("model_max_length", 0) < model.max_context_length:
        raise ValueError("Native context length exceeds text_config.model_max_length")
    index = json.loads((source / "model.safetensors.index.json").read_bytes())
    weight_map = index["weight_map"]
    config = json.loads((source / "hf_quant_config.json").read_bytes())
    declarations = defaultdict(list)
    for group in config["model"]["parameter_groups"]:
        for key in group["tensors"]:
            declarations[key].append(group["required_activation_quantization"])
    packed_dtypes: dict[str, str | None] = {}
    for i, layer in enumerate(model.layers):
        moe = layer.mlp
        require("intermediate_size", moe.intermediate_size)
        if "moe_intermediate_size" in text:
            require("moe_intermediate_size", moe.intermediate_size)
        require("n_routed_experts", moe.num_experts)
        require("num_experts_per_tok", moe.experts_per_token)
        require("route_scale", moe.route_scale)
        require("mlp_gate_softcap", moe.gate_softcap)
        require("mlp_up_softcap", moe.up_softcap)
        nvfp4 = moe.expert_format == "nvfp4"
        storage_dtypes = _NVFP4_STORAGE_DTYPES if nvfp4 else _MXFP8_STORAGE_DTYPES
        for matrix in ("w13_weight", "w2_weight"):
            key = f"{_PREFIX}layers.{i}.mlp.experts.{matrix}"
            if declarations[key] != [_NVFP4 if nvfp4 else _MXFP8]:
                raise ValueError(
                    f"Missing or unsupported expert quantization for {key}"
                )
            for suffix, dtype in storage_dtypes.items():
                packed_dtypes[key + suffix] = dtype

    _validate_packed_storage(source, weight_map, packed_dtypes)


def _validate_packed_storage(
    source: Path, weight_map: dict[str, str], packed_dtypes: dict[str, str | None]
) -> None:
    required_keys = {key for key, dtype in packed_dtypes.items() if dtype is not None}
    for key in packed_dtypes:
        if (key in weight_map) != (key in required_keys):
            raise ValueError(
                f"Missing or unexpected packed expert key in weight map: {key}"
            )

    # DCP scans all local safetensors files and copy_ can cast the source dtype.
    # Check actual headers, including unindexed duplicates, without loading tensors.
    found_keys = set()
    for shard in source.glob("*.safetensors"):
        with safe_open(shard, framework="pt", device="cpu") as tensors:
            for key in packed_dtypes.keys() & tensors.keys():
                dtype = packed_dtypes[key]
                if dtype is None:
                    raise ValueError(
                        f"Unexpected packed expert key {key} in {shard.name}"
                    )
                if weight_map[key] != shard.name:
                    raise ValueError(
                        f"Packed expert key {key} found in {shard.name}, "
                        f"but indexed in {weight_map[key]}"
                    )
                actual_dtype = tensors.get_slice(key).get_dtype()
                if actual_dtype != dtype:
                    raise ValueError(
                        f"Packed expert key {key} in {shard.name} has wrong storage dtype: "
                        f"expected {dtype}, got {actual_dtype}"
                    )
                found_keys.add(key)
    if missing_keys := required_keys - found_keys:
        key = min(missing_keys)
        raise ValueError(f"Missing packed expert key {key} in {weight_map[key]}")


class InklingStateDictAdapter(StateDictAdapter):
    def to_hf(
        self, state_dict: dict[str, torch.Tensor], *, to_quantized_layout: bool = False
    ) -> dict[str, torch.Tensor]:
        return {_PREFIX + key: tensor for key, tensor in state_dict.items()}

    def from_hf(
        self,
        hf_state_dict: dict[str, torch.Tensor],
        *,
        is_quantized_layout: bool = False,
    ) -> dict[str, torch.Tensor]:
        return {
            key.removeprefix(_PREFIX): tensor
            for key, tensor in hf_state_dict.items()
            if key.startswith(_PREFIX)
        }

    def get_hf_storage_reader(
        self,
        path: str,
        from_quantized: bool = False,
        *,
        thread_count: int | None = None,
    ) -> dcp.HuggingFaceStorageReader:
        _validate_checkpoint(Path(path), cast("InklingModel.Config", self.model_config))
        # Inkling consumes the packed tensors directly, not a dequantizing reader.
        return dcp.HuggingFaceStorageReader(path, thread_count=thread_count or 1)


@torch.no_grad()
def export_merged_checkpoint(model: nn.Module, source: Path, destination: Path) -> None:
    """Export a single-device attention-LoRA model on top of its original checkpoint.

    Unlike an ordinary text-only HF save, this preserves the unchanged vision tower,
    speculative drafter, packed expert payloads, quantization declarations and tokenizer.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists() or destination.is_relative_to(source):
        raise ValueError(
            "Destination must be a new directory outside the source checkpoint"
        )
    index = json.loads((source / "model.safetensors.index.json").read_bytes())[
        "weight_map"
    ]
    adapted = defaultdict(list)
    for name, param in model.named_parameters():
        if param.requires_grad and not name.endswith(_LORA):
            raise ValueError(f"Cannot export a non-LoRA trainable parameter: {name}")
        if not name.endswith(_LORA[0]):
            continue
        projection_name = name.removesuffix(_LORA[0])
        key = f"{_PREFIX}{projection_name}.weight"
        shard = index[key]
        if Path(shard).name != shard:
            raise ValueError(f"Expected a shard filename, got {shard!r}")
        adapted[shard].append((key, model.get_submodule(projection_name)))
    if not adapted:
        raise ValueError("No LoRA projections to merge")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as directory:
        output = Path(directory) / "checkpoint"
        # HF cache snapshots symlink into read-only blobs. Never patch the source.
        shutil.copytree(source, output, copy_function=shutil.copyfile)
        for shard, projections in adapted.items():
            with (
                safe_open(source / shard, framework="pt", device="cpu") as base,
                (output / shard).open("r+b") as file,
            ):
                # Patch only the BF16 byte ranges; packed experts need no round trip
                # through a tensor serializer, even when they share the same shard.
                header = json.loads(file.read(int.from_bytes(file.read(8), "little")))
                data_start = file.tell()
                for key, projection in projections:
                    weight = base.get_tensor(key)
                    a, b = projection.lora_a.weight, projection.lora_b.weight
                    if isinstance(a, DTensor) or isinstance(b, DTensor):
                        raise TypeError(
                            "Merge requires unsharded single-device LoRA factors"
                        )
                    if weight.dtype != torch.bfloat16 or weight.shape != (
                        b.shape[0],
                        a.shape[1],
                    ):
                        raise ValueError(
                            f"Expected a BF16 output-by-input projection for {key}"
                        )
                    merged = torch.addmm(
                        weight.float(),
                        b.to(device="cpu", dtype=torch.float32),
                        a.to(device="cpu", dtype=torch.float32),
                        alpha=projection._lora_scaling,
                    ).bfloat16()
                    file.seek(data_start + header[key]["data_offsets"][0])
                    file.write(merged.view(torch.uint8).numpy())
        output.rename(destination)

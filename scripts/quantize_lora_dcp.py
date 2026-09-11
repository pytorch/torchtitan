#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Convert a bf16 LoRA DCP checkpoint to the packed-MXFP4 layout.

Loads the checkpoint once into host RAM, repacks, and writes a DCP
checkpoint whose keys match the model that
``LoRAConverter.Config(quantize_base='mxfp4', quantize_experts='mxfp4')``
builds:

    <linear>.weight   ->  <linear>.base_qdata  (uint8, [out, in/2])
                          <linear>.base_scale  (e8m0 viewed uint8, [out, in/32])
    <experts>.w1_EFD  ->  <experts>.w1_EFD_qdata / _scale   (likewise w2/w3)
    everything else   ->  copied through unchanged

WHICH keys pack is not guessed from suffixes: the target flavor is built on
meta and walked, so the mapping is exactly the packed model's state-dict
contract. Together with the meta packed-layout build this is the
quantize-then-shard path for QLoRA on meta-first torchtitan: no rank ever
materializes the full bf16 model.

The SOURCE is a checkpoint of the same flavor WITHOUT the quantize options
(e.g. ``kimi_k3_debugmodel_lora``): it already carries the adapter keys, and
only the frozen bases change representation.

Usage:
  python scripts/quantize_lora_dcp.py \\
      --module kimi_k3 --config kimi_k3_debugmodel_qlora_mxfp4 \\
      --src <ckpt>/step-N --dst <out_dir>
"""

from __future__ import annotations

import argparse
import importlib
import os

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter
from torchao.prototype.mx_formats.mx_tensor import MXTensor

from torchtitan.components.lora import LoRALinearBase, MXFP4ExpertsBase


def _packed_key_map(module_name: str, config_name: str) -> dict[str, tuple[str, str]]:
    """source key -> (qdata key, scale key), derived from the packed model."""
    registry = importlib.import_module(
        f"torchtitan.models.{module_name}.config_registry"
    )
    trainer_config = getattr(registry, config_name)()
    with torch.device("meta"):
        model = trainer_config.model_spec.model.build()
    mapping: dict[str, tuple[str, str]] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, LoRALinearBase) and mod._quantize_base == "mxfp4":
            mapping[f"{name}.weight"] = (
                f"{name}.base_qdata",
                f"{name}.base_scale",
            )
        if isinstance(mod, MXFP4ExpertsBase):
            for wname in mod._mxfp4_shapes:
                mapping[f"{name}.{wname}"] = (
                    f"{name}.{wname}_qdata",
                    f"{name}.{wname}_scale",
                )
    if not mapping:
        raise ValueError(
            f"{config_name} builds no MXFP4-packed modules; nothing to convert."
        )
    return mapping


def _load_all(src: str) -> dict:
    """Load every entry in one DCP pass (per-key loads rescan the metadata
    and turn quadratic). Tensor entries land in preallocated buffers; bytes
    entries (e.g. titan's train_state) come through DCP's pickling path.
    Peak host RAM is the full unpacked checkpoint -- fine at debug scale; a
    tensor-streaming pass is the upgrade path for 48B-class sources.
    """
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    reader = FileSystemReader(src)
    sd = {
        key: (
            torch.empty(tmeta.size, dtype=tmeta.properties.dtype)
            if isinstance(tmeta, TensorStorageMetadata)
            else None
        )
        for key, tmeta in reader.read_metadata().state_dict_metadata.items()
    }
    dcp.load(sd, storage_reader=FileSystemReader(src))
    return sd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--module", required=True, help="model module, e.g. kimi_k3")
    ap.add_argument(
        "--config", required=True, help="the PACKED flavor the output will load into"
    )
    ap.add_argument("--src", required=True, help="source DCP dir (step-N)")
    ap.add_argument("--dst", required=True, help="output DCP dir")
    ap.add_argument(
        "--prefix",
        default="",
        help="checkpoint key prefix in front of module FQNs; titan "
        "checkpoints carry bare FQNs, so the default is empty",
    )
    args = ap.parse_args()

    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29399")
        dist.init_process_group("gloo", rank=0, world_size=1)

    mapping = {
        f"{args.prefix}{k}": (f"{args.prefix}{q}", f"{args.prefix}{s}")
        for k, (q, s) in _packed_key_map(args.module, args.config).items()
    }
    source = _load_all(args.src)
    out: dict[str, torch.Tensor] = {}
    n_packed = n_copied = 0
    for key, t in source.items():
        if key in mapping and isinstance(t, torch.Tensor):
            qkey, skey = mapping.pop(key)
            rows = t.numel() // t.shape[-1]
            mx = MXTensor.to_mx(
                t.reshape(rows, t.shape[-1]).to(torch.bfloat16),
                elem_dtype=torch.float4_e2m1fn_x2,
                block_size=32,
            )
            out[qkey] = mx.qdata.contiguous()
            out[skey] = mx.scale.view(torch.uint8).contiguous()
            n_packed += 1
        else:
            out[key] = t
            n_copied += 1
        del t

    if mapping:
        missing = sorted(mapping)[:5]
        raise ValueError(
            f"{len(mapping)} packed keys have no source tensor (first few: "
            f"{missing}); the source checkpoint does not match the flavor -- "
            "check --prefix and that the source is the unquantized twin."
        )
    dcp.save(out, storage_writer=FileSystemWriter(args.dst))
    print(f"packed {n_packed} weights, copied {n_copied} tensors -> {args.dst}")


if __name__ == "__main__":
    main()

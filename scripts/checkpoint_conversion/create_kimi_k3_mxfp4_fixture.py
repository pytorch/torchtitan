#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Create a release-format MXFP4 checkpoint for the Kimi-K3 debug model."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import torch
import torchao
from safetensors.torch import save_file
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchtitan.models.kimi_k3 import KimiK3StateDictAdapter, model_registry
from torchtitan.models.kimi_k3.quantization import MXFP4_QUANTIZATION_CONFIG
from torchtitan.quantization.mx_qat.checkpoint import MXFP4CheckpointPolicy

_DEFAULT_MAX_SHARD_BYTES = 1 << 30


def _git_revision(path: Path) -> str:
    return subprocess.check_output(
        ("git", "-C", str(path), "rev-parse", "HEAD"),
        text=True,
    ).strip()


def _tensor_bytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_hf_state_dict_to_mxfp4(
    hf_state_dict: dict[str, torch.Tensor],
    policy: MXFP4CheckpointPolicy,
) -> tuple[dict[str, torch.Tensor], int]:
    """Pack the exact resolved policy, including non-expert Linear weights."""
    missing = policy.weight_fqns - hf_state_dict.keys()
    if missing:
        raise ValueError(f"Fixture is missing selected weights: {sorted(missing)}")
    converted: dict[str, torch.Tensor] = {}
    pair_count = 0
    for key, value in hf_state_dict.items():
        value = value.detach().to(device="cpu").contiguous()
        if key not in policy.weight_fqns:
            converted[key] = value
            continue

        mx = MXTensor.to_mx(
            value,
            elem_dtype=torch.float4_e2m1fn_x2,
            block_size=policy.block_size,
        )
        prefix = key.removesuffix(".weight")
        converted[f"{prefix}.weight_packed"] = mx.qdata.contiguous()
        converted[f"{prefix}.weight_scale"] = mx.scale.view(torch.uint8).contiguous()
        pair_count += 1
    return converted, pair_count


def _partition_shards(
    state_dict: dict[str, torch.Tensor],
    max_shard_bytes: int,
) -> list[dict[str, torch.Tensor]]:
    shards: list[dict[str, torch.Tensor]] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0
    for key in sorted(state_dict):
        value = state_dict[key]
        value_bytes = _tensor_bytes(value)
        if current and current_bytes + value_bytes > max_shard_bytes:
            shards.append(current)
            current = {}
            current_bytes = 0
        current[key] = value
        current_bytes += value_bytes
    if current:
        shards.append(current)
    return shards


def write_sharded_checkpoint(
    state_dict: dict[str, torch.Tensor],
    output: Path,
    max_shard_bytes: int,
) -> dict[str, Any]:
    shards = _partition_shards(state_dict, max_shard_bytes)
    weight_map: dict[str, str] = {}
    shard_hashes: dict[str, str] = {}
    total_size = 0
    for index, shard in enumerate(shards, start=1):
        filename = f"model-{index:05d}-of-{len(shards):05d}.safetensors"
        path = output / filename
        save_file(shard, path)
        shard_hashes[filename] = _sha256(path)
        for key, value in shard.items():
            weight_map[key] = filename
            total_size += _tensor_bytes(value)
    index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
    (output / "model.safetensors.index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n"
    )
    return {
        "shard_count": len(shards),
        "shard_sha256": shard_hashes,
        "tensor_count": len(weight_map),
        "total_size": total_size,
    }


def create_fixture(
    output: Path,
    *,
    seed: int,
    max_shard_bytes: int = _DEFAULT_MAX_SHARD_BYTES,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output directory must be empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(seed)
    model_spec = model_registry("debugmodel", seq_len=128)
    model = model_spec.model.build()
    model.init_states()
    model.to(dtype=torch.bfloat16)
    adapter = KimiK3StateDictAdapter(model_spec.model, hf_assets_path=None)
    hf_state_dict = adapter.to_hf(model.state_dict())
    policy = MXFP4CheckpointPolicy.from_config(
        MXFP4_QUANTIZATION_CONFIG, adapter.hf_linear_weight_mapping()
    )
    converted, pair_count = convert_hf_state_dict_to_mxfp4(hf_state_dict, policy)
    storage = write_sharded_checkpoint(converted, output, max_shard_bytes)
    (output / "config.json").write_text(
        json.dumps(
            {
                "model_type": "kimi_k3",
                "text_config": {
                    "quantization_config": MXFP4_QUANTIZATION_CONFIG,
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    repo_root = Path(__file__).resolve().parents[2]
    manifest = {
        "block_size": 32,
        "packed_pair_count": pair_count,
        "model": "kimi_k3_debugmodel",
        "seed": seed,
        "torch_version": torch.__version__,
        "torchao_version": getattr(torchao, "__version__", "unknown"),
        "torchtitan_commit": _git_revision(repo_root),
        **storage,
    }
    (output / "fixture-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument(
        "--max-shard-bytes",
        type=int,
        default=_DEFAULT_MAX_SHARD_BYTES,
    )
    args = parser.parse_args()
    manifest = create_fixture(
        args.output,
        seed=args.seed,
        max_shard_bytes=args.max_shard_bytes,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

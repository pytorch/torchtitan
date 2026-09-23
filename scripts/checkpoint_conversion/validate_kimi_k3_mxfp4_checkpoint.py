# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Preflight every Kimi HF tensor shape without allocating model weights.

Read safetensors headers through the same reader used by DCP. Packed tensors
are checked as logical weights, and known padding tails are read and validated.
This checks loading compatibility, not numerical accuracy or training behavior.
"""

import argparse
import json
from pathlib import Path

import torch
from torch.distributed.checkpoint.metadata import Metadata, TensorStorageMetadata
from torchtitan.models.kimi_k3 import kimi_k3_configs, model_registry
from torchtitan.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter


def validate_shapes(metadata: Metadata, expected: dict[str, torch.Size]) -> None:
    """Require every model tensor and reject unexplained checkpoint tensors."""
    actual = metadata.state_dict_metadata
    # The adapter explicitly ignores these legacy HF buffers on import.
    ignored = {name for name in actual if name.endswith("rotary_emb.inv_freq")}
    missing = sorted(expected.keys() - actual.keys())
    unexpected = sorted(actual.keys() - expected.keys() - ignored)
    mismatches = []
    for name in sorted(expected.keys() & actual.keys()):
        tensor = actual[name]
        if not isinstance(tensor, TensorStorageMetadata):
            mismatches.append(f"{name}: expected a tensor")
        elif tensor.size != expected[name]:
            mismatches.append(
                f"{name}: checkpoint {tuple(tensor.size)}, model {tuple(expected[name])}"
            )
    if missing or unexpected or mismatches:
        raise ValueError(
            "Checkpoint shape preflight failed: "
            f"missing={missing[:20]} (total {len(missing)}), "
            f"unexpected={unexpected[:20]} (total {len(unexpected)}), "
            f"mismatches={mismatches[:20]} (total {len(mismatches)})"
        )


def validate_checkpoint(
    checkpoint: Path, *, model_flavor: str = "Kimi-K3", from_quantized: bool = True
) -> dict[str, int | str]:
    config = model_registry(model_flavor, seq_len=128)
    adapter = KimiK3StateDictAdapter(config, hf_assets_path=None)
    with torch.device("meta"):
        model = config.build()
        expected = {
            name: tensor.shape
            for name, tensor in adapter.to_hf(model.state_dict()).items()
        }
    reader = adapter.get_hf_storage_reader(str(checkpoint), from_quantized)
    metadata = reader.read_metadata()
    validate_shapes(metadata, expected)
    return {"model_flavor": model_flavor, "validated_tensors": len(expected)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-flavor", choices=kimi_k3_configs, default="Kimi-K3")
    parser.add_argument("--unquantized", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            validate_checkpoint(
                args.checkpoint,
                model_flavor=args.model_flavor,
                from_quantized=not args.unquantized,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

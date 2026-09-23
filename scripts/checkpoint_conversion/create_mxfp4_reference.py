# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Regenerate tests/assets/mxfp4-reference.json with compressed-tensors==0.18.0.

This independent fixture does not import TorchAO or TorchTitan. It exercises
every E2M1 value, multiple E8M0 scales, and rounding/saturation of off-grid values.
"""

import importlib.metadata
import json
from pathlib import Path

import torch
from compressed_tensors.compressors.mxfp4 import MXFP4PackedCompressor
from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme


def main():
    version = importlib.metadata.version("compressed-tensors")
    if version != "0.18.0":
        raise RuntimeError(f"Expected compressed-tensors==0.18.0, got {version}")
    levels = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]
    )
    scales = torch.tensor([[0.5, 1], [2, 4]], dtype=torch.float32)
    values = levels.repeat(8).reshape(2, 64)
    values[1] += 0.2
    weight = values * scales.repeat_interleave(32, dim=-1)
    scheme = QuantizationScheme(
        targets=["Linear"],
        weights=QuantizationArgs(
            num_bits=4,
            type="float",
            strategy="group",
            group_size=32,
            symmetric=True,
            dynamic=False,
            scale_dtype=torch.uint8,
        ),
    )
    packed = MXFP4PackedCompressor.compress(
        {"weight": weight, "weight_scale": scales}, scheme
    )
    decoded = MXFP4PackedCompressor.decompress(packed, scheme)["weight"]
    result = {
        "serializer": f"compressed-tensors=={version}",
        "generator": "scripts/checkpoint_conversion/create_mxfp4_reference.py",
        "weight_packed": packed["weight_packed"].tolist(),
        "weight_scale": packed["weight_scale"].tolist(),
        "dequantized": decoded.tolist(),
    }
    path = Path(__file__).parents[2] / "tests/assets/mxfp4-reference.json"
    path.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()

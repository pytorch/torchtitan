# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile
import unittest
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from scripts.checkpoint_conversion.create_kimi_k3_mxfp4_fixture import (
    convert_hf_state_dict_to_mxfp4,
    write_sharded_checkpoint,
)
from torchao.prototype.mx_formats.mx_tensor import MXTensor
from torchtitan.components.checkpointer.packed_hf_storage import (
    PackedPairHuggingFaceStorageReader,
    PackedPairSpec,
)
from torchtitan.quantization.mx_qat.checkpoint import (
    decode_mxfp4,
    MXFP4CheckpointPolicy,
)

_HF_WEIGHT = "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight"


class KimiK3MXFP4CheckpointIntegrationTest(unittest.TestCase):
    def test_release_format_fixture_loads_across_shards(self) -> None:
        weight = torch.linspace(-6, 6, 128, dtype=torch.bfloat16).reshape(2, 64)
        dense = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
        expected = MXTensor.to_mx(
            weight,
            elem_dtype=torch.float4_e2m1fn_x2,
            block_size=32,
        ).dequantize(torch.bfloat16)
        converted, pair_count = convert_hf_state_dict_to_mxfp4(
            {_HF_WEIGHT: weight, "dense.weight": dense},
            MXFP4CheckpointPolicy(frozenset({_HF_WEIGHT})),
        )
        self.assertEqual(pair_count, 1)
        self.assertNotIn(_HF_WEIGHT, converted)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            manifest = write_sharded_checkpoint(
                converted,
                output,
                max_shard_bytes=64,
            )
            self.assertGreater(manifest["shard_count"], 1)
            destination = {
                _HF_WEIGHT: torch.empty_like(weight),
                "dense.weight": torch.empty_like(dense),
            }
            reader = PackedPairHuggingFaceStorageReader(
                str(output),
                PackedPairSpec(
                    packed_suffix=".weight_packed",
                    scale_suffix=".weight_scale",
                    virtual_suffix=".weight",
                    block_size=32,
                    packed_values_per_byte=2,
                    target_dtype=torch.bfloat16,
                    target_fqns=frozenset({_HF_WEIGHT}),
                    decode=decode_mxfp4,
                ),
            )
            dcp.load(destination, storage_reader=reader)

        torch.testing.assert_close(
            destination[_HF_WEIGHT], expected, rtol=0, atol=0, equal_nan=True
        )
        torch.testing.assert_close(destination["dense.weight"], dense, rtol=0, atol=0)
        self.assertFalse(
            any(key.endswith(("weight_packed", "weight_scale")) for key in destination)
        )


if __name__ == "__main__":
    unittest.main()

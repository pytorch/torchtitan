# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import unittest
from pathlib import Path

import torch
from torchtitan.quantization.mx_qat.checkpoint import (
    decode_mxfp4,
    MXFP4CheckpointPolicy,
)


def _config():
    return {
        "format": "mxfp4-pack-quantized",
        "quant_method": "compressed-tensors",
        "config_groups": {
            "group_0": {
                "targets": ["Linear"],
                "weights": {
                    "num_bits": 4,
                    "type": "float",
                    "strategy": "group",
                    "group_size": 32,
                    "dynamic": False,
                    "symmetric": True,
                    "scale_dtype": "torch.uint8",
                },
            }
        },
        "ignore": ["re:.*shared"],
    }


class MXFP4PolicyTest(unittest.TestCase):
    def test_manifest_selects_only_actual_pairs_from_eligible_linears(self):
        weights = ["model.expert.weight", "model.residual.weight", "model.fused.weight"]
        policy = MXFP4CheckpointPolicy.from_manifest(
            _config(),
            weights,
            {
                "model.expert.weight_packed": "weights.safetensors",
                "model.expert.weight_scale": "scales.safetensors",
                "model.residual.weight": "dense.safetensors",
                "model.fused.weight": "dense.safetensors",
            },
        )
        self.assertEqual(policy.weight_fqns, frozenset({"model.expert.weight"}))

    def test_manifest_rejects_invalid_pairs(self):
        pair = {"model.expert.weight_packed": "a", "model.expert.weight_scale": "b"}
        cases = [
            ({"model.expert.weight_packed": "a"}, "missing scales"),
            ({"model.expert.weight_scale": "a"}, "orphan scales"),
            ({**pair, "model.expert.weight": "c"}, "both packed and ordinary"),
            (
                {"model.shared.weight_packed": "a", "model.shared.weight_scale": "b"},
                "outside",
            ),
            ({"unknown.weight_packed": "a", "unknown.weight_scale": "b"}, "outside"),
            ({"model.expert.weight_packed": None}, "weight_map"),
        ]
        for manifest, message in cases:
            with self.subTest(manifest=manifest), self.assertRaisesRegex(
                ValueError, message
            ):
                MXFP4CheckpointPolicy.from_manifest(
                    _config(), ["model.expert.weight", "model.shared.weight"], manifest
                )

    def test_resolves_actual_linear_hierarchy_and_prefix_regex(self):
        policy = MXFP4CheckpointPolicy.from_config(
            _config(),
            [
                "model.experts.0.w1.weight",
                "model.latent_proj.weight",
                "model.shared.gate.weight",
            ],
        )
        self.assertEqual(
            policy.weight_fqns,
            frozenset(
                {
                    "model.experts.0.w1.weight",
                    "model.latent_proj.weight",
                }
            ),
        )
        self.assertNotIn("model.embed_tokens.weight", policy.weight_fqns)

    def test_rejects_contradictory_numeric_metadata(self):
        for field, value in (
            ("num_bits", 8),
            ("type", "int"),
            ("group_size", 64),
            ("scale_dtype", "torch.float32"),
            ("symmetric", False),
            ("dynamic", True),
            ("strategy", "channel"),
        ):
            with self.subTest(field=field):
                config = _config()
                config["config_groups"]["group_0"]["weights"][field] = value
                with self.assertRaisesRegex(ValueError, "static symmetric float4"):
                    MXFP4CheckpointPolicy.from_config(config, ["linear.weight"])

    def test_exact_targets_and_ignore(self):
        config = _config()
        config["config_groups"]["group_0"]["targets"] = ["model.proj"]
        policy = MXFP4CheckpointPolicy.from_config(
            config, ["model.proj.weight", "model.other.weight"]
        )
        self.assertEqual(policy.weight_fqns, frozenset({"model.proj.weight"}))

    def test_independent_compressed_tensors_reference(self):
        path = Path(__file__).parents[2] / "assets" / "mxfp4-reference.json"
        fixture = json.loads(path.read_text())
        self.assertEqual(fixture["serializer"], "compressed-tensors==0.18.0")
        actual = decode_mxfp4(
            torch.tensor(fixture["weight_packed"], dtype=torch.uint8),
            torch.tensor(fixture["weight_scale"], dtype=torch.uint8),
            32,
            torch.bfloat16,
        )
        torch.testing.assert_close(
            actual,
            torch.tensor(fixture["dequantized"], dtype=torch.bfloat16),
            rtol=0,
            atol=0,
        )


if __name__ == "__main__":
    unittest.main()

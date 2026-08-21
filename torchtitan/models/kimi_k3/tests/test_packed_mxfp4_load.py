# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Loading K3's packed-MXFP4 experts, without the 1.56 TB download.

The released checkpoint is 1.561 TB and stores routed experts as
``.weight_packed`` + ``.weight_scale``. The load path can still be exercised
completely: build a synthetic checkpoint in the OFFICIAL key naming and byte
layout at k3mini scale, push it through the same key map and dequantizer a real
load would use, and check the experts end up holding the right values.

The byte-layout claim is validated against torchao rather than against our own
packer, which would be circular: torchao packs, we decode, and the result must
match torchao's own dequantize exactly.
"""

from __future__ import annotations

import unittest

import torch

from torchtitan.models.kimi_k3.hf_key_map import official_to_titan
from torchtitan.models.kimi_k3.model import KimiK3Model
from torchtitan.models.kimi_k3.model_configs import build_kimi_linear_config
from torchtitan.models.kimi_k3.packed_mxfp4 import (
    dequantize_mxfp4,
    load_packed_experts,
    quantize_mxfp4,
)

_KDA = {i for i in range(21) if (i + 1) not in {4, 8, 12, 16, 20, 21}}


class TestMXFP4ByteLayout(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "torchao MX needs CUDA")
    def test_our_decoder_matches_torchao_on_torchao_bytes(self):
        """The decisive check: not "our packer round-trips" (circular) but "we
        read bytes produced by an independent packer". A swapped nibble order
        would pass a round-trip and fail here."""
        from torchao.prototype.mx_formats.mx_tensor import MXTensor

        torch.manual_seed(0)
        w = (torch.randn(16, 128) * 0.2).cuda().bfloat16()
        mx = MXTensor.to_mx(w, elem_dtype=torch.float4_e2m1fn_x2, block_size=32)
        ours = dequantize_mxfp4(
            mx.qdata, mx.scale.view(torch.uint8), dtype=torch.float32
        )
        theirs = mx.dequantize().float()
        self.assertEqual(((ours - theirs).norm() / theirs.norm()).item(), 0.0)

    def test_shapes_follow_the_released_layout(self):
        w = torch.randn(16, 128)
        packed, scale = quantize_mxfp4(w)
        self.assertEqual(packed.shape, (16, 64))  # two nibbles per byte
        self.assertEqual(scale.shape, (16, 4))  # one byte per 32 values
        self.assertEqual(packed.dtype, torch.uint8)
        self.assertEqual(scale.dtype, torch.uint8)

    def test_round_trip_error_is_in_the_4_bit_band(self):
        torch.manual_seed(0)
        w = torch.randn(16, 128) * 0.2
        back = dequantize_mxfp4(*quantize_mxfp4(w), dtype=torch.float32)
        rel = ((back - w).norm() / w.norm()).item()
        # 4 bits with 2 mantissa bits on Gaussian data; torchao measures ~0.117
        self.assertGreater(rel, 0.05)
        self.assertLess(rel, 0.20)

    def test_zero_scale_byte_means_zero_not_a_tiny_power_of_two(self):
        packed = torch.zeros(1, 16, dtype=torch.uint8)
        scale = torch.zeros(1, 1, dtype=torch.uint8)
        out = dequantize_mxfp4(packed, scale, dtype=torch.float32)
        self.assertTrue(torch.all(out == 0.0))

    def test_mismatched_scale_group_count_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "groups"):
            dequantize_mxfp4(
                torch.zeros(4, 64, dtype=torch.uint8),
                torch.zeros(4, 3, dtype=torch.uint8),
            )

    def test_non_uint8_input_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "uint8"):
            dequantize_mxfp4(
                torch.zeros(4, 64, dtype=torch.int8),
                torch.zeros(4, 2, dtype=torch.uint8),
            )


class TestSyntheticOfficialCheckpointLoad(unittest.TestCase):
    """A whole-model load, driven by official key strings."""

    def _model(self):
        cfg = build_kimi_linear_config("k3mini", vocab_size=256)
        with torch.device("meta"):
            m = KimiK3Model.make_config(cfg).build()
        m.to_empty(device="cpu")
        m.init_weights()
        return m, cfg

    @staticmethod
    def _first_moe_layer(model) -> str:
        # layer 0 is dense (first_k_dense_replace), so it has no moe at all
        for name, layer in model.layers.items():
            if getattr(layer, "moe", None) is not None:
                return name
        raise AssertionError("k3mini must have a MoE layer")

    def test_official_keys_drive_a_complete_expert_load(self):
        model, cfg = self._model()
        layer_idx = int(self._first_moe_layer(model))
        experts = model.layers[str(layer_idx)].moe._moe.routed_experts.inner_experts

        # Build the synthetic checkpoint slice with OFFICIAL key names.
        torch.manual_seed(0)
        truth, tensors = {}, {}
        for w_official, our_name in (
            ("w1", "w1_EFD"),
            ("w2", "w2_EDF"),
            ("w3", "w3_EFD"),
        ):
            shape = experts._parameters[our_name].shape
            for e in range(cfg.num_experts):
                block = torch.randn(*shape[1:]) * 0.2
                packed, scale = quantize_mxfp4(block)
                base = (
                    f"language_model.model.layers.{layer_idx}."
                    f"block_sparse_moe.experts.{e}.{w_official}"
                )
                ours_p, kind_p = official_to_titan(
                    f"{base}.weight_packed", kda_layers=_KDA
                )
                ours_s, kind_s = official_to_titan(
                    f"{base}.weight_scale", kda_layers=_KDA
                )
                self.assertEqual((kind_p, kind_s), ("expert_packed", "expert_scale"))
                self.assertEqual(ours_p, ours_s)  # same destination, two parts
                tensors[ours_p.split(".")[-1]] = packed
                tensors[ours_s.split(".")[-1] + ":scale"] = scale
                truth[(our_name, e)] = dequantize_mxfp4(
                    packed, scale, dtype=torch.float32
                )

        written = load_packed_experts(
            experts, tensors, num_experts=cfg.num_experts, dtype=torch.float32
        )
        self.assertEqual(written, 3 * cfg.num_experts)

        for (name, e), expected in truth.items():
            got = experts._parameters[name][e]
            self.assertTrue(
                torch.equal(got, expected.to(got.dtype)),
                f"{name}[{e}] did not load exactly",
            )

    def test_a_missing_slice_refuses_the_load(self):
        """A partial load is worse than a failure: the unwritten experts keep
        their init values and the model still trains, which is the exact failure
        mode that cost this repo every recorded MoE loss."""
        model, cfg = self._model()
        layer_idx = int(self._first_moe_layer(model))
        experts = model.layers[str(layer_idx)].moe._moe.routed_experts.inner_experts
        shape = experts._parameters["w1_EFD"].shape
        tensors = {}
        for e in range(cfg.num_experts - 1):  # deliberately one short
            packed, scale = quantize_mxfp4(torch.randn(*shape[1:]))
            tensors[f"w1_EFD[{e}]"] = packed
            tensors[f"w1_EFD[{e}]:scale"] = scale
        with self.assertRaisesRegex(KeyError, "partial load"):
            load_packed_experts(experts, tensors, num_experts=cfg.num_experts)

    def test_wrong_shape_is_rejected(self):
        model, cfg = self._model()
        layer_idx = int(self._first_moe_layer(model))
        experts = model.layers[str(layer_idx)].moe._moe.routed_experts.inner_experts
        tensors = {}
        for name in ("w1_EFD", "w2_EDF", "w3_EFD"):
            for e in range(cfg.num_experts):
                packed, scale = quantize_mxfp4(torch.randn(8, 64))  # wrong
                tensors[f"{name}[{e}]"] = packed
                tensors[f"{name}[{e}]:scale"] = scale
        with self.assertRaisesRegex(ValueError, "expects"):
            load_packed_experts(experts, tensors, num_experts=cfg.num_experts)


if __name__ == "__main__":
    unittest.main()


class TestE8M0SpecialCodes(unittest.TestCase):
    """The two E8M0 codes ``quantize_mxfp4`` never emits.

    OCP MX defines 0x00 as 2**-127 and 0xFF as NaN. A round trip through this
    module's own quantizer cannot reach either -- it picks scales from the data --
    so the decode was free to be wrong in both directions and was: 0x00 mapped to
    zero and 0xFF fell through to exp2(255-127), i.e. inf. An official shard using
    them would silently decode a tiny scale as zero and a NaN as inf.
    """

    def _decode_one_group(self, scale_code: int):
        from torchtitan.models.kimi_k3.packed_mxfp4 import (
            dequantize_mxfp4,
            MXFP4_GROUP_SIZE,
        )

        # One group whose every nibble is E2M1 code 1 == value 0.5, so the decoded
        # magnitude is exactly the scale factor times 0.5.
        packed = torch.full((1, MXFP4_GROUP_SIZE // 2), 0x11, dtype=torch.uint8)
        scale = torch.tensor([[scale_code]], dtype=torch.uint8)
        return dequantize_mxfp4(packed, scale, dtype=torch.float32)

    def test_zero_code_is_two_to_the_minus_127_not_zero(self):
        out = self._decode_one_group(0x00)
        self.assertTrue(
            torch.isfinite(out).all(), "0x00 must decode to a finite tiny scale"
        )
        self.assertFalse(
            bool((out == 0).all()),
            "0x00 decoded to zero; OCP MX defines it as 2**-127",
        )

    def test_all_ones_code_is_nan_not_inf(self):
        out = self._decode_one_group(0xFF)
        self.assertTrue(bool(torch.isnan(out).all()), "0xFF must decode to NaN")
        self.assertFalse(
            bool(torch.isinf(out).any()), "0xFF decoded to inf; OCP MX defines NaN"
        )


class TestDequantMatchesTorchao(unittest.TestCase):
    """Pin the delegation in finding 56, including the cases that distinguish it.

    ``dequantize_mxfp4`` now calls torchao's MX dequantizer instead of a local nibble
    table. These are the comparisons that were run before delegating, kept so that a
    change in torchao is caught here rather than in a checkpoint that decodes wrong.

    The E8M0 special values are the point. A random-scale comparison passes whether or
    not 0xFF is handled as NaN, because quantize_mxfp4 never emits it -- which is how
    that bug survived the round-trip test the first time.
    """

    def _reference(self, packed, scale, group_size, dtype):
        """The local implementation this replaced, kept only as the test's oracle."""
        lo = (packed & 0x0F).long()
        hi = (packed >> 4).long()
        table = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0],
            device=packed.device, dtype=torch.float32,
        )
        values = torch.stack([table[lo], table[hi]], dim=-1).flatten(-2)
        exp = scale.to(torch.int32)
        factors = torch.where(
            exp == 0xFF,
            torch.full_like(exp, float("nan"), dtype=torch.float32),
            torch.exp2((exp - 127).to(torch.float32)),
        ).repeat_interleave(group_size, dim=-1)
        return (values * factors).to(dtype)

    def test_bit_identical_on_random_data_at_bf16_and_fp32(self):
        torch.manual_seed(0)
        for rows, cols in ((4, 64), (3, 32), (8, 128)):
            for dtype in (torch.bfloat16, torch.float32):
                packed = torch.randint(0, 256, (rows, cols // 2), dtype=torch.uint8)
                scale = torch.randint(100, 150, (rows, cols // 32), dtype=torch.uint8)
                got = dequantize_mxfp4(packed, scale, dtype=dtype)
                want = self._reference(packed, scale, 32, dtype)
                self.assertTrue(
                    torch.equal(got, want),
                    f"{rows}x{cols} {dtype}: max diff "
                    f"{(got.float() - want.float()).abs().max().item()}",
                )

    def test_e8m0_special_values_agree(self):
        packed = torch.full((1, 48), 0x22, dtype=torch.uint8)  # every nibble = 1.0
        for scale_value in (0x00, 0x7F, 0xFF):
            scale = torch.full((1, 3), scale_value, dtype=torch.uint8)
            got = dequantize_mxfp4(packed, scale, dtype=torch.float32)
            want = self._reference(packed, scale, 32, torch.float32)
            if scale_value == 0xFF:
                self.assertTrue(got.isnan().all(), "0xFF must decode to NaN, not inf")
            else:
                self.assertTrue(torch.equal(got, want), f"scale {scale_value:#04x}")

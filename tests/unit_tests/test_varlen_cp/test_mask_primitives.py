# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.distributed.varlen_cp.mask_primitives import (
    AttnSlice,
    cu_seqlens_to_attn_slices,
    make_slice_mask,
    MaskType,
    split_slice_at_chunk_boundary,
)


class TestMaskType(unittest.TestCase):
    def test_mask_type_values(self):
        self.assertEqual(MaskType.FULL, 0)
        self.assertEqual(MaskType.CAUSAL, 1)
        self.assertEqual(MaskType.INVCAUSAL, 2)
        self.assertEqual(MaskType.BICAUSAL, 3)


class TestAttnSlice(unittest.TestCase):
    def test_basic_properties(self):
        s = AttnSlice(q_start=0, q_end=10, k_start=0, k_end=10, mask_type=MaskType.FULL)
        self.assertEqual(s.q_len, 10)
        self.assertEqual(s.k_len, 10)

    def test_work_estimate_full(self):
        s = AttnSlice(q_start=0, q_end=100, k_start=0, k_end=100, mask_type=MaskType.FULL)
        self.assertAlmostEqual(s.work_estimate, 10000.0)

    def test_work_estimate_causal(self):
        s = AttnSlice(q_start=0, q_end=100, k_start=0, k_end=100, mask_type=MaskType.CAUSAL)
        self.assertAlmostEqual(s.work_estimate, 5000.0)

    def test_work_estimate_invcausal(self):
        s = AttnSlice(
            q_start=0, q_end=100, k_start=0, k_end=100, mask_type=MaskType.INVCAUSAL
        )
        self.assertAlmostEqual(s.work_estimate, 5000.0)

    def test_work_estimate_bicausal(self):
        s = AttnSlice(
            q_start=0, q_end=100, k_start=0, k_end=100, mask_type=MaskType.BICAUSAL
        )
        self.assertAlmostEqual(s.work_estimate, 2500.0)

    def test_work_estimate_minimum(self):
        """Empty slices should have work_estimate >= 1.0."""
        s = AttnSlice(q_start=0, q_end=1, k_start=0, k_end=1, mask_type=MaskType.CAUSAL)
        self.assertGreaterEqual(s.work_estimate, 1.0)


class TestCuSeqlensToAttnSlices(unittest.TestCase):
    def test_single_doc(self):
        cu_seqlens = [0, 256]
        slices = cu_seqlens_to_attn_slices(cu_seqlens, is_causal=True)
        self.assertEqual(len(slices), 1)
        self.assertEqual(slices[0].q_start, 0)
        self.assertEqual(slices[0].q_end, 256)
        self.assertEqual(slices[0].mask_type, MaskType.CAUSAL)

    def test_multi_doc(self):
        cu_seqlens = [0, 128, 300, 512]
        slices = cu_seqlens_to_attn_slices(cu_seqlens, is_causal=True)
        self.assertEqual(len(slices), 3)
        self.assertEqual(slices[0], AttnSlice(0, 128, 0, 128, MaskType.CAUSAL))
        self.assertEqual(slices[1], AttnSlice(128, 300, 128, 300, MaskType.CAUSAL))
        self.assertEqual(slices[2], AttnSlice(300, 512, 300, 512, MaskType.CAUSAL))

    def test_full_mask(self):
        cu_seqlens = [0, 256]
        slices = cu_seqlens_to_attn_slices(cu_seqlens, is_causal=False)
        self.assertEqual(len(slices), 1)
        self.assertEqual(slices[0].mask_type, MaskType.FULL)

    def test_tensor_input(self):
        cu_seqlens = torch.tensor([0, 128, 256])
        slices = cu_seqlens_to_attn_slices(cu_seqlens, is_causal=True)
        self.assertEqual(len(slices), 2)

    def test_empty_doc(self):
        """Adjacent equal values in cu_seqlens create zero-length docs."""
        cu_seqlens = [0, 128, 128, 256]
        slices = cu_seqlens_to_attn_slices(cu_seqlens)
        # Zero-length docs should be skipped
        self.assertEqual(len(slices), 2)


class TestSplitSliceAtChunkBoundary(unittest.TestCase):
    def test_doc_within_one_chunk(self):
        """Document fits entirely within one chunk."""
        s = AttnSlice(q_start=10, q_end=50, k_start=10, k_end=50, mask_type=MaskType.CAUSAL)
        result = split_slice_at_chunk_boundary(s, chunk_size=64, total_seqlen=128)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].mask_type, MaskType.CAUSAL)
        self.assertEqual(result[0].q_start, 10)
        self.assertEqual(result[0].q_end, 50)

    def test_doc_spanning_two_chunks(self):
        """Document spans two chunks: diagonal blocks are CAUSAL, below-diagonal are FULL."""
        s = AttnSlice(q_start=48, q_end=80, k_start=48, k_end=80, mask_type=MaskType.CAUSAL)
        result = split_slice_at_chunk_boundary(s, chunk_size=64, total_seqlen=128)

        # Should produce 3 sub-slices:
        # (chunk 0, chunk 0): q=[48,64), k=[48,64), CAUSAL
        # (chunk 1, chunk 0): q=[64,80), k=[48,64), FULL (below diagonal)
        # (chunk 1, chunk 1): q=[64,80), k=[64,80), CAUSAL (diagonal)
        self.assertEqual(len(result), 3)

        # Check that we have the expected types
        types = {(r.q_start // 64, r.k_start // 64): r.mask_type for r in result}
        self.assertEqual(types[(0, 0)], MaskType.CAUSAL)
        self.assertEqual(types[(1, 0)], MaskType.FULL)
        self.assertEqual(types[(1, 1)], MaskType.CAUSAL)

    def test_full_mask_stays_full(self):
        """FULL mask type sub-blocks are all FULL."""
        s = AttnSlice(q_start=48, q_end=80, k_start=48, k_end=80, mask_type=MaskType.FULL)
        result = split_slice_at_chunk_boundary(s, chunk_size=64, total_seqlen=128)
        for r in result:
            self.assertEqual(r.mask_type, MaskType.FULL)


class TestMakeSliceMask(unittest.TestCase):
    def test_full_mask(self):
        mask = make_slice_mask(4, 4, MaskType.FULL)
        self.assertTrue(mask.all())

    def test_causal_square(self):
        mask = make_slice_mask(4, 4, MaskType.CAUSAL)
        expected = torch.tensor(
            [
                [True, False, False, False],
                [True, True, False, False],
                [True, True, True, False],
                [True, True, True, True],
            ]
        )
        self.assertTrue(torch.equal(mask, expected))

    def test_causal_rectangular(self):
        """Bottom-right aligned causal for q_len < k_len."""
        mask = make_slice_mask(2, 4, MaskType.CAUSAL)
        # q_len=2, k_len=4, offset = k_len - q_len = 2
        # Row 0: j <= 0+2 → j in {0,1,2}
        # Row 1: j <= 1+2 → j in {0,1,2,3}
        expected = torch.tensor(
            [
                [True, True, True, False],
                [True, True, True, True],
            ]
        )
        self.assertTrue(torch.equal(mask, expected))

    def test_invcausal_mask(self):
        mask = make_slice_mask(4, 4, MaskType.INVCAUSAL)
        expected = torch.tensor(
            [
                [True, True, True, True],
                [False, True, True, True],
                [False, False, True, True],
                [False, False, False, True],
            ]
        )
        self.assertTrue(torch.equal(mask, expected))

    def test_bicausal_mask(self):
        mask = make_slice_mask(4, 4, MaskType.BICAUSAL)
        expected = torch.tensor(
            [
                [True, False, False, False],
                [False, True, False, False],
                [False, False, True, False],
                [False, False, False, True],
            ]
        )
        self.assertTrue(torch.equal(mask, expected))


if __name__ == "__main__":
    unittest.main()

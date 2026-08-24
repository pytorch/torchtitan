# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import triton.language as tl

from torchtitan.distributed.minimal_async_ep.kernels import (
    _copy_rows_to_peer_ptrs_kernel,
    copy_full_counts_to_peers_kernel,
    copy_rows_to_peers_kernel,
    expand_topk_grad_kernel,
    fill_combine_metadata_kernel,
    fill_dispatch_metadata_kernel,
    invert_flat_indices_kernel,
    reduce_topk_slots_kernel,
    topk_scores_grad_kernel,
)


def assert_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(f"shape mismatch: {actual.shape} != {expected.shape}")
    if actual.dtype != expected.dtype:
        raise AssertionError(f"dtype mismatch: {actual.dtype} != {expected.dtype}")
    if not torch.equal(actual, expected):
        raise AssertionError(f"tensor mismatch:\nactual={actual}\nexpected={expected}")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestMinimalAsyncEPKernels(unittest.TestCase):
    def test_topk_index_kernels_match_reference(self):
        flat_indices = torch.tensor([2, 0, 3, 1], device="cuda", dtype=torch.int64)
        slot_to_row = invert_flat_indices_kernel(flat_indices, num_rows=4)
        assert_equal(
            slot_to_row,
            torch.tensor([1, 3, 0, 2], device="cuda", dtype=torch.int64),
        )

        routed_output = torch.tensor(
            [[10.0, 1.0], [20.0, 2.0], [30.0, 3.0], [40.0, 4.0]],
            device="cuda",
        )
        scores = torch.tensor([0.1, 0.2, 0.3, 0.4], device="cuda")
        out = reduce_topk_slots_kernel(
            routed_output,
            slot_to_row,
            scores,
            num_tokens=2,
            top_k=2,
            scores_are_slot_ordered=True,
        )
        expected_out = torch.stack(
            [
                routed_output[slot_to_row[0]] * scores[0]
                + routed_output[slot_to_row[1]] * scores[1],
                routed_output[slot_to_row[2]] * scores[2]
                + routed_output[slot_to_row[3]] * scores[3],
            ]
        )
        assert_equal(out, expected_out)

        grad_out = torch.tensor([[100.0, 10.0], [200.0, 20.0]], device="cuda")
        grad_routed = expand_topk_grad_kernel(
            grad_out,
            flat_indices,
            scores,
            top_k=2,
            dtype=torch.float32,
            scores_are_slot_ordered=True,
        )
        expected_grad_routed = torch.stack(
            [
                grad_out[flat_indices[row] // 2] * scores[flat_indices[row]]
                for row in range(flat_indices.numel())
            ]
        )
        assert_equal(grad_routed, expected_grad_routed)

        grad_scores = topk_scores_grad_kernel(
            routed_output,
            grad_out,
            flat_indices,
            top_k=2,
            dtype=torch.float32,
            scores_are_slot_ordered=True,
        )
        expected_grad_scores = torch.empty_like(scores)
        for row, flat_index in enumerate(flat_indices):
            token = flat_index // 2
            expected_grad_scores[flat_index] = torch.sum(
                routed_output[row] * grad_out[token]
            )
        assert_equal(grad_scores, expected_grad_scores)

    def test_topk_backward_uses_int64_for_row_stride_arithmetic(self):
        num_cols = 2048
        top_k = 8
        first_overflow_row = 2**31 // num_cols
        num_rows = first_overflow_row + 2
        num_tokens = (num_rows + top_k - 1) // top_k

        free_bytes, _ = torch.cuda.mem_get_info()
        required_bytes = (
            (num_rows + num_tokens) * num_cols * 2 + num_rows * 8 + 512 * 1024**2
        )
        if free_bytes < required_bytes:
            self.skipTest(
                f"need at least {required_bytes} free CUDA bytes, got {free_bytes}"
            )

        grad_out = torch.ones(
            num_tokens,
            num_cols,
            device="cuda",
            dtype=torch.bfloat16,
        )
        flat_indices = torch.arange(num_rows, device="cuda", dtype=torch.int64)
        grad_routed = expand_topk_grad_kernel(
            grad_out,
            flat_indices,
            None,
            top_k=top_k,
            dtype=torch.bfloat16,
        )
        rows = torch.tensor(
            [first_overflow_row - 1, first_overflow_row, num_rows - 1],
            device="cuda",
        )
        assert_equal(
            grad_routed[rows, 0],
            torch.ones(rows.numel(), device="cuda", dtype=torch.bfloat16),
        )

        grad_scores = topk_scores_grad_kernel(
            grad_routed,
            grad_out,
            flat_indices,
            top_k=top_k,
            dtype=torch.float32,
            scores_are_slot_ordered=True,
        )
        assert_equal(
            grad_scores[rows],
            torch.full(
                (rows.numel(),),
                num_cols,
                device="cuda",
                dtype=torch.float32,
            ),
        )

    def test_topk_reduce_uses_int64_for_output_stride_arithmetic(self):
        num_cols = 2048
        first_overflow_token = 2**31 // num_cols
        num_tokens = first_overflow_token + 2

        free_bytes, _ = torch.cuda.mem_get_info()
        required_bytes = num_tokens * num_cols * 2 + num_tokens * 8 + 512 * 1024**2
        if free_bytes < required_bytes:
            self.skipTest(
                f"need at least {required_bytes} free CUDA bytes, got {free_bytes}"
            )

        routed_output = torch.ones(
            num_tokens,
            num_cols,
            device="cuda",
            dtype=torch.uint8,
        )
        slot_to_row = torch.arange(num_tokens, device="cuda", dtype=torch.int64)
        out = reduce_topk_slots_kernel(
            routed_output,
            slot_to_row,
            None,
            num_tokens=num_tokens,
            top_k=1,
        )
        tokens = torch.tensor(
            [first_overflow_token - 1, first_overflow_token, num_tokens - 1],
            device="cuda",
        )
        assert_equal(
            out[tokens, 0],
            torch.ones(tokens.numel(), device="cuda", dtype=torch.uint8),
        )

    def test_metadata_kernels_match_reference(self):
        counts_storage = torch.tensor(
            [2, 1, 0, 0, 1, 0, 1, 0],
            device="cuda",
            dtype=torch.int64,
        )
        counts = counts_storage[::2]
        local_count_starts = counts.cumsum(0) - counts
        local_count_starts_storage = torch.empty(8, device="cuda", dtype=torch.int64)
        local_count_starts_storage[::2] = local_count_starts
        local_count_starts_storage[1::2] = torch.tensor(
            [1, 0, 0, 0],
            device="cuda",
            dtype=torch.int64,
        )
        local_count_starts = local_count_starts_storage[::2]
        local_dest_offsets_storage = torch.empty(8, device="cuda", dtype=torch.int64)
        local_dest_offsets_storage[::2] = torch.tensor(
            [0, 0, 5, 9],
            device="cuda",
            dtype=torch.int64,
        )
        local_dest_offsets_storage[1::2] = torch.tensor(
            [7, 0, 0, 0],
            device="cuda",
            dtype=torch.int64,
        )
        local_dest_offsets = local_dest_offsets_storage[::2]
        self.assertFalse(counts.is_contiguous())
        self.assertFalse(local_dest_offsets.is_contiguous())
        self.assertFalse(local_count_starts.is_contiguous())
        dst_ranks, dst_rows = fill_dispatch_metadata_kernel(
            counts,
            local_dest_offsets,
            local_count_starts,
            num_routed_tokens=4,
            num_local_experts=2,
            max_tokens_per_segment=2,
        )
        assert_equal(
            dst_ranks,
            torch.tensor([0, 0, 1, 1], device="cuda", dtype=torch.int64),
        )
        assert_equal(
            dst_rows,
            torch.tensor([0, 1, 5, 9], device="cuda", dtype=torch.int64),
        )

        segment_lens_storage = torch.tensor(
            [2, 1, 1, 0, 0, 0, 1, 0],
            device="cuda",
            dtype=torch.int64,
        )
        segment_lens = segment_lens_storage[::2]
        output_starts_values = segment_lens.cumsum(0) - segment_lens
        output_starts_storage = torch.empty(8, device="cuda", dtype=torch.int64)
        output_starts_storage[::2] = output_starts_values
        output_starts_storage[1::2] = torch.tensor(
            [3, 0, 0, 0],
            device="cuda",
            dtype=torch.int64,
        )
        output_starts = output_starts_storage[::2]
        source_input_starts_storage = torch.empty(
            2,
            8,
            device="cuda",
            dtype=torch.int64,
        )
        source_input_starts_storage[:, ::2] = torch.tensor(
            [[0, 0, 4, 6], [0, 0, 8, 10]],
            device="cuda",
            dtype=torch.int64,
        )
        source_input_starts_storage[:, 1::2] = torch.tensor(
            [[99, 99, 99, 99], [99, 99, 99, 99]],
            device="cuda",
            dtype=torch.int64,
        )
        source_input_starts = source_input_starts_storage[:, ::2]
        self.assertFalse(segment_lens.is_contiguous())
        self.assertFalse(output_starts.is_contiguous())
        self.assertFalse(source_input_starts.is_contiguous())
        combine_ranks, combine_rows, num_valid_rows = fill_combine_metadata_kernel(
            segment_lens,
            output_starts,
            source_input_starts,
            ep_rank=1,
            ep_size=2,
            num_local_experts=2,
            receive_capacity=6,
            max_tokens_per_segment=2,
        )
        assert_equal(
            combine_ranks[:4],
            torch.tensor([0, 0, 1, 1], device="cuda", dtype=torch.int64),
        )
        assert_equal(
            combine_rows[:4],
            torch.tensor([4, 5, 8, 10], device="cuda", dtype=torch.int64),
        )
        assert_equal(
            num_valid_rows,
            torch.tensor([4], device="cuda", dtype=torch.int64),
        )

    def test_copy_kernels_handle_strides_and_active_row_masks(self):
        counts = torch.tensor([2, 0, 1, 3], device="cuda", dtype=torch.int64)
        count_dsts = [
            torch.zeros(2, 4, device="cuda", dtype=torch.int64),
            torch.zeros(2, 4, device="cuda", dtype=torch.int64),
        ]
        count_ptrs = torch.tensor(
            [dst.data_ptr() for dst in count_dsts],
            device="cuda",
            dtype=torch.int64,
        )
        copy_full_counts_to_peers_kernel(
            counts,
            count_dsts,
            rank=1,
            ep_size=2,
            num_experts=4,
            dst_ptrs=count_ptrs,
        )
        torch.cuda.synchronize()
        for dst in count_dsts:
            assert_equal(dst[0], torch.zeros_like(counts))
            assert_equal(dst[1], counts)

        src_base = torch.arange(20, device="cuda", dtype=torch.float32).view(5, 4)
        src = src_base[:, ::2]
        row_dsts = [
            torch.zeros(6, 2, device="cuda", dtype=torch.float32),
            torch.zeros(6, 2, device="cuda", dtype=torch.float32),
        ]
        row_ptrs = torch.tensor(
            [dst.data_ptr() for dst in row_dsts],
            device="cuda",
            dtype=torch.int64,
        )
        dst_ranks = torch.tensor([1, 0, 1], device="cuda", dtype=torch.int64)
        dst_rows = torch.tensor([3, 4, 5], device="cuda", dtype=torch.int64)
        src_rows = torch.tensor([2, 0, 4], device="cuda", dtype=torch.int64)
        num_valid_rows = torch.tensor([2], device="cuda", dtype=torch.int64)

        copy_rows_to_peers_kernel(
            src,
            row_dsts,
            dst_ranks,
            dst_rows,
            ep_size=2,
            num_rows=3,
            num_cols=2,
            dst_ptrs=row_ptrs,
            block_m=2,
            src_rows=src_rows,
            num_valid_rows=num_valid_rows,
        )
        torch.cuda.synchronize()
        assert_equal(row_dsts[1][3], src[2])
        assert_equal(row_dsts[0][4], src[0])
        assert_equal(row_dsts[1][5], torch.zeros(2, device="cuda"))

    def test_copy_rows_uses_int64_for_source_stride_arithmetic(self):
        row = 1_048_576
        stride = 2048
        base_offset = 2**31
        high_offset = row * stride
        storage_numel = base_offset + high_offset + 1
        metadata_numel = row + 1

        free_bytes, _ = torch.cuda.mem_get_info()
        required_bytes = storage_numel + metadata_numel * 8 * 2 + 512 * 1024**2
        if free_bytes < required_bytes:
            self.skipTest(
                f"need at least {required_bytes} free CUDA bytes, got {free_bytes}"
            )

        src_storage = torch.empty(storage_numel, device="cuda", dtype=torch.uint8)
        src = src_storage[base_offset:]
        dst = torch.zeros(1, device="cuda", dtype=torch.uint8)
        dst_ptrs = torch.tensor([dst.data_ptr()], device="cuda", dtype=torch.int64)
        dst_ranks = torch.full((metadata_numel,), -1, device="cuda", dtype=torch.int64)
        dst_ranks[row] = 0
        dst_rows = torch.zeros(metadata_numel, device="cuda", dtype=torch.int64)
        num_valid_rows = torch.tensor(
            [metadata_numel], device="cuda", dtype=torch.int64
        )

        src_storage[0] = 17
        src_storage[base_offset + high_offset] = 93
        torch.cuda.synchronize()

        _copy_rows_to_peer_ptrs_kernel[(metadata_numel, 1)](
            src,
            dst_ptrs,
            dst_ranks,
            dst_rows,
            num_valid_rows,
            dst_rows,
            NUM_ROWS=metadata_numel,
            NUM_COLS=1,
            SRC_ROW_STRIDE=stride,
            SRC_COL_STRIDE=1,
            DST_ROW_STRIDE=1,
            DST_DTYPE=tl.uint8,
            HAS_NUM_VALID_ROWS=True,
            HAS_SRC_ROWS=False,
            BLOCK_M=1,
            BLOCK_N=1,
        )
        torch.cuda.synchronize()

        self.assertEqual(int(dst.item()), 93)


if __name__ == "__main__":
    unittest.main()

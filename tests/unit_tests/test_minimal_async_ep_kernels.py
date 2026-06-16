# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.distributed.minimal_async_ep.kernels import (
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

    def test_metadata_kernels_match_reference(self):
        counts = torch.tensor([2, 0, 1, 1], device="cuda", dtype=torch.int64)
        local_count_starts = counts.cumsum(0) - counts
        local_dest_offsets = torch.tensor(
            [0, 0, 5, 9],
            device="cuda",
            dtype=torch.int64,
        )
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

        segment_lens = torch.tensor([2, 1, 0, 1], device="cuda", dtype=torch.int64)
        output_starts = segment_lens.cumsum(0) - segment_lens
        source_input_starts = torch.tensor(
            [[0, 0, 4, 6], [0, 0, 8, 10]],
            device="cuda",
            dtype=torch.int64,
        )
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


if __name__ == "__main__":
    unittest.main()

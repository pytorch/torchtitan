# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import operator
import os
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist

import torchtitan.distributed.minimal_async_ep.api as minimal_async_ep_api
import triton.language as tl
from torch.fx.experimental.proxy_tensor import make_fx

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


def test_minimal_async_ep_fake_trace_has_launch_wait_edges():
    def exchange(x, expert_ids, counts, scores):
        (
            hidden,
            dispatch_ranks,
            dispatch_rows,
            combine_ranks,
            combine_rows,
            num_valid,
            expert_to_token,
            token_to_expert,
            _tokens_per_expert,
        ) = minimal_async_ep_api.dispatch_op(x, expert_ids, counts, 16, 2)
        hidden = minimal_async_ep_api.wait_dispatch_op(hidden, [x])
        expert_output = hidden * 2
        routed = minimal_async_ep_api.combine_op(
            expert_output,
            dispatch_ranks,
            dispatch_rows,
            combine_ranks,
            combine_rows,
            num_valid,
            8,
        )
        routed = minimal_async_ep_api.wait_combine_op(
            routed,
            [expert_output, combine_ranks, combine_rows, num_valid],
        )
        return minimal_async_ep_api.reduce_topk_op(
            routed,
            token_to_expert,
            expert_to_token,
            scores,
            4,
            2,
        )

    gm = make_fx(exchange, tracing_mode="fake")(
        torch.randn(4, 8),
        torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0]]),
        torch.full((4,), 2, dtype=torch.int64),
        torch.randn(8),
    )
    nodes = list(gm.graph.nodes)

    def one(target):
        matches = [node for node in nodes if node.target is target]
        assert len(matches) == 1
        return matches[0]

    dispatch = one(torch.ops.minimal_async_ep.dispatch.default)
    wait_dispatch = one(torch.ops.minimal_async_ep.wait_dispatch.default)
    combine = one(torch.ops.minimal_async_ep.combine.default)
    wait_combine = one(torch.ops.minimal_async_ep.wait_combine.default)
    dispatch_hidden = next(
        user
        for user in dispatch.users
        if user.target is operator.getitem and user.args[1] == 0
    )

    assert wait_dispatch.args == (dispatch_hidden, [dispatch.args[0]])
    assert list(dispatch_hidden.users) == [wait_dispatch]
    assert wait_combine.args[0] is combine
    assert wait_combine.args[1][0] is combine.args[0]
    assert list(combine.users) == [wait_combine]


def test_minimal_async_ep_wait_schemas_are_nonmutating_aliases():
    for op in (
        torch.ops.minimal_async_ep.wait_dispatch.default,
        torch.ops.minimal_async_ep.wait_combine.default,
    ):
        schema = str(op._schema)
        assert "Tensor(a) pending" in schema
        assert "Tensor(a!)" not in schema


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class TestMinimalAsyncEPKernels(unittest.TestCase):
    @unittest.skipUnless(
        dist.is_initialized() or "RANK" in os.environ,
        "requires torchrun launcher",
    )
    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
    def test_launch_wait_api_matches_reference(self):
        initialized_pg = dist.is_initialized()
        if not initialized_pg:
            dist.init_process_group("nccl")
        if dist.get_world_size() != 2:
            self.skipTest("test expects exactly two ranks")

        local_rank = int(os.environ.get("LOCAL_RANK", dist.get_rank()))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        minimal_async_ep_api._buffer_state = None

        def run_exchange(x, scores, expert_ids):
            counts = torch.bincount(expert_ids.flatten(), minlength=4)
            (
                hidden,
                dispatch_ranks,
                dispatch_rows,
                combine_ranks,
                combine_rows,
                num_valid,
                expert_to_token,
                token_to_expert,
                _tokens_per_expert,
            ) = minimal_async_ep_api.dispatch_op(x, expert_ids, counts, 16, 2)
            hidden = minimal_async_ep_api.wait_dispatch_op(hidden, [x])
            expert_output = hidden * 1.25
            routed = minimal_async_ep_api.combine_op(
                expert_output,
                dispatch_ranks,
                dispatch_rows,
                combine_ranks,
                combine_rows,
                num_valid,
                8,
            )
            routed = minimal_async_ep_api.wait_combine_op(
                routed,
                [expert_output, combine_ranks, combine_rows, num_valid],
            )
            return minimal_async_ep_api.reduce_topk_op(
                routed,
                token_to_expert,
                expert_to_token,
                scores.flatten(),
                4,
                2,
            )

        try:
            minimal_async_ep_api.init_buffer(
                dist.group.WORLD,
                hidden_dim=4,
                tokens_per_rank=4,
                num_local_experts=2,
                top_k=2,
                dtype=torch.float32,
                device=device,
            )
            rank = dist.get_rank()
            x = (
                torch.arange(16, device=device, dtype=torch.float32).view(4, 4)
                + rank * 100
            )
            scores = torch.linspace(0.1, 0.8, 8, device=device).view(4, 2)
            token_ids = torch.arange(4, device=device)
            expert_ids = torch.stack(
                ((token_ids + rank) % 4, (token_ids + rank + 2) % 4), dim=1
            )

            actual_x = x.clone().requires_grad_()
            actual_scores = scores.clone().requires_grad_()
            actual = run_exchange(actual_x, actual_scores, expert_ids)
            actual_grads = torch.autograd.grad(
                actual.square().sum(), (actual_x, actual_scores)
            )

            ref_x = x.clone().requires_grad_()
            ref_scores = scores.clone().requires_grad_()
            ref_expert_output = ref_x * 1.25
            expected = (
                ref_expert_output * ref_scores[:, :1]
                + ref_expert_output * ref_scores[:, 1:]
            )
            expected_grads = torch.autograd.grad(
                expected.square().sum(), (ref_x, ref_scores)
            )

            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for actual, expected in zip(actual_grads, expected_grads, strict=True):
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        finally:
            minimal_async_ep_api._buffer_state = None
            if not initialized_pg and dist.is_initialized():
                dist.destroy_process_group()

    def test_wait_ops_compile_with_inductor(self):
        for wait_op in (
            minimal_async_ep_api.wait_dispatch_op,
            minimal_async_ep_api.wait_combine_op,
        ):
            with self.subTest(wait_op=wait_op):

                def fn(x):
                    pending = x * 2
                    return wait_op(pending, [x]).sin()

                x = torch.randn(16, device="cuda", requires_grad=True)
                with patch.object(minimal_async_ep_api, "_wait_pending_event"):
                    actual = torch.compile(fn, backend="inductor", fullgraph=True)(x)
                expected = (x * 2).sin()
                torch.testing.assert_close(actual, expected)
                actual.sum().backward()
                torch.testing.assert_close(x.grad, 2 * (x * 2).cos())

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
            SRC_ROW_DIVISOR=1,
            BLOCK_M=1,
            BLOCK_N=1,
        )
        torch.cuda.synchronize()

        self.assertEqual(int(dst.item()), 93)


if __name__ == "__main__":
    unittest.main()

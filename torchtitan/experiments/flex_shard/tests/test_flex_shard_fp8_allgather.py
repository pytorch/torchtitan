# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch
import torch.nn as nn
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype
from torch.testing._internal.common_utils import run_tests, TestCase

from torchtitan.experiments.flex_shard import BucketSpec, flex_shard
from torchtitan.experiments.flex_shard.example.fp8_ragged_shard import (
    blockwise_dequant_weight,
    blockwise_quant_weight,
    blockwise_transpose,
    Fp8BlockwiseGroupedRaggedShard,
    Fp8TwoOrientationGroupedRaggedShard,
    make_fp8_blockwise_grouped_ragged_placement_fn,
    make_fp8_two_orientation_grouped_ragged_placement_fn,
    promote_to_square_block,
)
from torchtitan.experiments.flex_shard.example.ragged_shard import GroupedRaggedShard


device_type = torch.device(get_devtype())


class _TwoWeight(nn.Module):
    """Two 2D weights whose byte-balanced cut crosses the w1/w2 boundary.

    With block_size=4, world_size=2: w1 (16x8=128) + w2 (8x8=64), aligned to
    block*in=32. rank0 owns w1[0:96] (12 rows), rank1 owns w1[96:128] (4 rows) +
    w2 (8 rows) -- so w1 is split across ranks and the cut crosses into w2, while
    every rank still owns whole 4x4 tiles.
    """

    def __init__(self) -> None:
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(16, 8))
        self.w2 = nn.Parameter(torch.randn(8, 8))


class _Mlp(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.wq = nn.Linear(dim, dim, bias=False)
        self.wo = nn.Linear(dim, dim, bias=False)


class _MlpModel(nn.Module):
    """fp8 weights live in a submodule, so a multi-param bucket maps to one
    execution-unit module that reshard-after-forward can checkpoint-wrap."""

    def __init__(self, dim: int = 8) -> None:
        super().__init__()
        self.mlp = _Mlp(dim)


class _FakeMesh:
    """Minimal mesh for exercising bucket_storage_layout without a real PG."""

    def __init__(self, size: int, rank: int = 0) -> None:
        self._size = size
        self._rank = rank

    def size(self) -> int:
        return self._size

    def get_local_rank(self) -> int:
        return self._rank


class TestFp8AllGatherLayout(TestCase):
    """CPU-only tests for the tile-aligned layout (no FlexShard runtime)."""

    def test_alignment_is_one_tile_row(self) -> None:
        params = [("w", nn.Parameter(torch.zeros(16, 8)))]
        fp8 = Fp8BlockwiseGroupedRaggedShard(local_units=(1, 1), block_size=4)
        # block_size * suffix(=in=8) = 32, vs plain GroupedRaggedShard's suffix(=8).
        self.assertEqual(fp8._param_alignment_numel(params), 32)
        self.assertEqual(
            GroupedRaggedShard(local_units=(1, 1))._param_alignment_numel(params), 8
        )

    def test_rejects_indivisible_dims(self) -> None:
        fp8 = Fp8BlockwiseGroupedRaggedShard(local_units=(1, 1), block_size=4)
        with self.assertRaisesRegex(ValueError, "divisible by block_size"):
            fp8._param_alignment_numel([("w", nn.Parameter(torch.zeros(16, 6)))])  # in
        with self.assertRaisesRegex(ValueError, "divisible by block_size"):
            fp8._param_alignment_numel([("w", nn.Parameter(torch.zeros(6, 8)))])  # out

    def test_cut_lands_on_whole_tiles(self) -> None:
        block = 4
        in_dim = 8
        named = [
            ("w1", nn.Parameter(torch.zeros(16, in_dim))),
            ("w2", nn.Parameter(torch.zeros(8, in_dim))),
        ]
        world_size = 2
        placement = Fp8BlockwiseGroupedRaggedShard(
            local_units=(1,) * world_size, block_size=block
        )
        placements = {fqn: (placement,) for fqn, _ in named}
        layout = placement.bucket_storage_layout(
            named, placements, _FakeMesh(world_size)
        )
        bucket_layout = layout.param_layouts["w1"].bucket_layout
        tile_row = block * in_dim  # 32 elems = one 4x4 tile-row
        # every rank range is a whole number of tile-rows, and the bucket is a
        # whole number of block x block tiles
        for rank_numel in bucket_layout.rank_numels:
            self.assertEqual(rank_numel % tile_row, 0)
        self.assertEqual(bucket_layout.global_numel % (block * block), 0)

    def test_semantic_unshard_marker_preserves_autograd(self) -> None:
        from torchtitan.experiments.flex_shard.flex_shard.ops import (
            is_unshard_bucket_op,
            mark_unshard_bucket,
            UNSHARD_BUCKET_OP,
        )

        tensor = torch.randn(3, requires_grad=True)
        (marked,) = mark_unshard_bucket([tensor])
        self.assertIsNot(marked, tensor)
        self.assertIs(marked._base, tensor)
        self.assertTrue(is_unshard_bucket_op(UNSHARD_BUCKET_OP))

        marked.sum().backward()
        self.assertEqual(tensor.grad, torch.ones_like(tensor))

    def test_blockwise_transpose_invariance(self) -> None:
        # Square block x block tiling makes quant commute with transpose:
        # blockwise_transpose(quant(W)) is bit-identical to quant(W^T), and yields
        # the column-major (in, out) view the forward fp8 GEMM RHS expects.
        block = 4
        torch.manual_seed(0)
        w = torch.randn(16, 8)  # (out, in)
        quant, scale = blockwise_quant_weight(w, block)
        q_t, s_t = blockwise_transpose(quant, scale, block)
        ref_q, ref_s = blockwise_quant_weight(w.t().contiguous(), block)

        self.assertEqual(tuple(q_t.shape), (8, 16))  # (in, out)
        self.assertEqual(q_t.stride(), (1, 8))  # column-major view
        self.assertTrue(
            torch.equal(q_t.contiguous().view(torch.uint8), ref_q.view(torch.uint8))
        )
        self.assertTrue(torch.equal(s_t.contiguous(), ref_s))

    def test_blockwise_transpose_rejects_non_square_tiling(self) -> None:
        # A 1x4 (activation-style, non-square) scale must be rejected: the
        # fp8(W^T)=fp8(W)^T identity only holds for square tiles.
        quant = torch.zeros(16, 8, dtype=torch.float8_e4m3fn)
        non_square_scale = torch.zeros(16, 8 // 4, dtype=torch.float32)  # 1x4 groups
        with self.assertRaisesRegex(ValueError, "square"):
            blockwise_transpose(quant, non_square_scale, 4)

    def test_promote_to_square_block(self) -> None:
        self.assertEqual(promote_to_square_block(1, 128), 128)  # 1x128 -> 128x128
        self.assertEqual(promote_to_square_block(128, 1), 128)
        self.assertEqual(promote_to_square_block(64, 128), 128)
        self.assertEqual(promote_to_square_block(32, 96), 96)

    def test_promote_to_square_enables_transpose(self) -> None:
        # Solution 1 (promote): a non-square (1x4) quant can't be transposed, but
        # promoting to the enclosing square block (4x4) makes it transpose-reusable.
        torch.manual_seed(0)
        w = torch.randn(16, 8)
        q_ns, s_ns = blockwise_quant_weight(w, (1, 4))  # non-square scale (16, 2)
        square = promote_to_square_block(1, 4)  # = 4
        with self.assertRaisesRegex(ValueError, "square"):
            blockwise_transpose(q_ns, s_ns, square)
        q_sq, s_sq = blockwise_quant_weight(w, square)  # square scale (4, 2)
        q_t, _ = blockwise_transpose(q_sq, s_sq, square)  # no raise
        self.assertEqual(tuple(q_t.shape), (8, 16))

    def test_rectangular_quant_shapes(self) -> None:
        w = torch.randn(16, 8)
        _, s_fwd = blockwise_quant_weight(w, (1, 4))  # group along N (forward K)
        _, s_bwd = blockwise_quant_weight(w, (4, 1))  # group along M (backward K)
        self.assertEqual(tuple(s_fwd.shape), (16, 2))
        self.assertEqual(tuple(s_bwd.shape), (4, 8))

    def test_two_orientation_alignment(self) -> None:
        fp8 = Fp8TwoOrientationGroupedRaggedShard(local_units=(1, 1), block_size=4)
        params = [("w", nn.Parameter(torch.zeros(16, 8)))]
        self.assertEqual(fp8._param_alignment_numel(params), 32)  # block * in


class TestFp8AllGather(FSDPTest):
    @property
    def world_size(self) -> int:
        return 2

    def _mesh(self):
        return init_device_mesh(
            device_type.type, (self.world_size,), mesh_dim_names=("fsdp",)
        )

    @skip_if_lt_x_gpu(2)
    def test_fp8_allgather_matches_reference_quant(self) -> None:
        # fp8 all-gather: each rank quantizes its (tile-aligned) bf16 shard and the
        # bucket all-gathers fp8 + scales. Because the cut lands on whole tiles, the
        # reconstructed full fp8 + scales are BIT-IDENTICAL to quantizing the full
        # bf16 weight -- and the gathered bytes are ~half of bf16.
        block = 4
        mesh = self._mesh()
        torch.manual_seed(0)
        model = _TwoWeight().to(device=device_type, dtype=torch.bfloat16)
        reference = copy.deepcopy(model)

        flex_shard(
            model,
            buckets=[
                BucketSpec(
                    ["w1", "w2"],
                    placement_fn=make_fp8_blockwise_grouped_ragged_placement_fn(
                        block_size=block, local_units=(1,) * self.world_size
                    ),
                    mesh=mesh,
                    reshard_after_forward=False,
                )
            ],
        )

        storage = model.sharded_bucket_storages[0]
        infos = list(storage.param_infos.values())
        placement = infos[0].placement
        self.assertIsInstance(placement, Fp8BlockwiseGroupedRaggedShard)
        # the byte cut crosses a matrix boundary: w1 is split across ranks
        self.assertTrue(any(0 < i.local_numel < i.global_numel for i in infos))

        param_by_fqn = dict(model.named_parameters())
        tensors = [param_by_fqn[info.fqn] for info in infos]

        # Drive the fp8 unshard (the collective the forward would trigger).
        prepared = placement.prepare_unshard_bucket(tensors, infos, mesh, None)
        placement.run_prepared_unshard(prepared)
        full_params = placement.finish_prepared_unshard(prepared).full_params

        ref_by_fqn = dict(reference.named_parameters())
        for info, data in zip(infos, full_params, strict=True):
            ref_w = ref_by_fqn[info.fqn].detach()
            ref_fp8, ref_scale = blockwise_quant_weight(ref_w, block)
            # bit-identical fp8 data (compare raw bytes) and reciprocal scales
            self.assertTrue(
                torch.equal(data.view(torch.uint8), ref_fp8.view(torch.uint8))
            )
            self.assertTrue(torch.equal(data._blockwise_scale, ref_scale))
            self.assertEqual(data._block_size, block)
            self.assertEqual(data.dtype, torch.float8_e4m3fn)
            # dequant is a sane approximation of the original bf16 weight
            deq = blockwise_dequant_weight(data, data._blockwise_scale, block)
            torch.testing.assert_close(deq, ref_w.float(), atol=0.25, rtol=0.25)

        # Bandwidth: fp8 data is 1 byte/elem; fp8 + scales < bf16 (2 bytes/elem).
        gathered_fp8 = prepared.buffers[2]
        gathered_scale = prepared.buffers[3]
        self.assertEqual(gathered_fp8.element_size(), 1)
        fp8_bytes = gathered_fp8.numel() * 1 + gathered_scale.numel() * 4
        bf16_bytes = gathered_fp8.numel() * 2
        self.assertLess(fp8_bytes, bf16_bytes)

    @skip_if_lt_x_gpu(2)
    def test_fp8_allgather_composes_with_reshard_after_forward(self) -> None:
        # reshard_after_forward=True must install RAF saved-tensor hooks on the
        # bucket's execution-unit module and leave the fp8 unshard bit-identical.
        from torchtitan.experiments.flex_shard.flex_shard.reshard_after_forward import (
            _RAF_SAVED_TENSOR_HOOKS_INSTALLED_ATTR,
        )

        block = 4
        mesh = self._mesh()
        torch.manual_seed(0)
        model = _MlpModel(dim=8).to(device=device_type, dtype=torch.bfloat16)
        reference = copy.deepcopy(model)

        flex_shard(
            model,
            buckets=[
                BucketSpec(
                    ["mlp.wq.weight", "mlp.wo.weight"],
                    placement_fn=make_fp8_blockwise_grouped_ragged_placement_fn(
                        block_size=block, local_units=(1,) * self.world_size
                    ),
                    mesh=mesh,
                    reshard_after_forward=True,
                )
            ],
        )

        # RAF-only wrapping uses saved-tensor hooks. Existing AC wrappers are
        # composed separately by the core RAF path.
        self.assertTrue(
            getattr(model.mlp, _RAF_SAVED_TENSOR_HOOKS_INSTALLED_ATTR, False)
        )

        # The fp8 unshard is unaffected by reshard wrapping: drive it via the bucket
        # storage and compare bit-for-bit.
        storage = model.sharded_bucket_storages[0]
        infos = list(storage.param_infos.values())
        placement = infos[0].placement
        self.assertIsInstance(placement, Fp8BlockwiseGroupedRaggedShard)
        tensors = [storage.get_local_view(info.fqn) for info in infos]
        prepared = placement.prepare_unshard_bucket(tensors, infos, mesh, None)
        placement.run_prepared_unshard(prepared)
        full_params = placement.finish_prepared_unshard(prepared).full_params

        ref_by_fqn = dict(reference.named_parameters())
        for info, data in zip(infos, full_params, strict=True):
            ref_fp8, ref_scale = blockwise_quant_weight(
                ref_by_fqn[info.fqn].detach(), block
            )
            self.assertTrue(
                torch.equal(data.view(torch.uint8), ref_fp8.view(torch.uint8))
            )
            self.assertTrue(torch.equal(data._blockwise_scale, ref_scale))

    @skip_if_lt_x_gpu(2)
    def test_two_orientation_gathers_both_buffers(self) -> None:
        # Solution 2 (non-square): gather two fp8 buffers, one per orientation -- the
        # forward (1, block) tiling and the backward (block, 1) tiling. Each is
        # bit-identical to quantizing the full bf16 weight in that tiling, and the
        # gather moves two fp8 data buffers (~2x the square path).
        block = 4
        mesh = self._mesh()
        torch.manual_seed(0)
        model = _TwoWeight().to(device=device_type, dtype=torch.bfloat16)
        reference = copy.deepcopy(model)

        flex_shard(
            model,
            buckets=[
                BucketSpec(
                    ["w1", "w2"],
                    placement_fn=make_fp8_two_orientation_grouped_ragged_placement_fn(
                        block_size=block, local_units=(1,) * self.world_size
                    ),
                    mesh=mesh,
                    reshard_after_forward=False,
                )
            ],
        )

        storage = model.sharded_bucket_storages[0]
        infos = list(storage.param_infos.values())
        placement = infos[0].placement
        self.assertIsInstance(placement, Fp8TwoOrientationGroupedRaggedShard)

        param_by_fqn = dict(model.named_parameters())
        tensors = [param_by_fqn[info.fqn] for info in infos]
        prepared = placement.prepare_unshard_bucket(tensors, infos, mesh, None)
        placement.run_prepared_unshard(prepared)
        full_params = placement.finish_prepared_unshard(prepared).full_params

        ref_by_fqn = dict(reference.named_parameters())
        for info, data in zip(infos, full_params, strict=True):
            ref_w = ref_by_fqn[info.fqn].detach()
            ref_fwd, ref_s_fwd = blockwise_quant_weight(ref_w, (1, block))
            ref_bwd, ref_s_bwd = blockwise_quant_weight(ref_w, (block, 1))
            # forward buffer (returned param) + backward buffer (attached)
            self.assertTrue(
                torch.equal(data.view(torch.uint8), ref_fwd.view(torch.uint8))
            )
            self.assertTrue(torch.equal(data._scale_forward, ref_s_fwd))
            self.assertTrue(
                torch.equal(
                    data._fp8_backward.view(torch.uint8), ref_bwd.view(torch.uint8)
                )
            )
            self.assertTrue(torch.equal(data._scale_backward, ref_s_bwd))

        # Two fp8 data buffers were gathered (the ~2x-bandwidth cost of non-square).
        gathered_fwd = prepared.buffers[4]
        gathered_bwd = prepared.buffers[5]
        self.assertEqual(gathered_fwd.element_size(), 1)
        self.assertEqual(gathered_bwd.element_size(), 1)
        self.assertEqual(gathered_fwd.numel(), gathered_bwd.numel())


if __name__ == "__main__":
    run_tests()

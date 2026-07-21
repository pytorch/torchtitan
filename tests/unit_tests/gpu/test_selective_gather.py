# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Multi-GPU tests for the selective-gather transports.

Runs on the repo's multi-process harness, so the NCCL path is what is measured.
A full plan reproduces a plain all-gather bitwise. A sliding-window plan is the
selective case: asymmetric sends and receives, padded slots on the rank with no
predecessor, and only the planned output slots written. Both backwards reduce
each block's gradient over the ranks that read it.

The P2P class pins its backend so it runs everywhere. The LSA class needs
Hopper + CuTeDSL + nccl4py and skips without them.
"""

import unittest
from unittest import mock

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torchtitan.distributed.context_parallel.selective_gather import (
    backend as backend_mod,
    BlockGatherPlan,
    build_gin_metadata,
    build_plan_metadata,
    full_plan,
    run_p2p_gather,
    run_p2p_gather_backward,
    select_backend,
    selective_gather,
    SelectiveGatherContext,
    sliding_window_plan,
)

pytestmark = pytest.mark.multi_gpu

BLOCKS_PER_RANK = 2
BLOCK_NUMEL = 4


class TestP2PGatherNumerics(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    def _setup(self, device):
        mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("cp",)
        )
        pg = mesh.get_group("cp")
        shard_numel = BLOCKS_PER_RANK * BLOCK_NUMEL
        torch.manual_seed(1234 + pg.rank())
        kv_local = torch.randn(shard_numel, device=device)
        ctx = SelectiveGatherContext(
            mesh,
            mesh_axis="cp",
            shard_numel=shard_numel,
            block_numel=BLOCK_NUMEL,
            dtype=kv_local.dtype,
            device=device,
            # Pin the portable backend: this box may auto-select "lsa", but this
            # class validates the p2p transport specifically.
            backend="p2p",
        )
        return pg, kv_local, ctx

    def _planned_slots(self, plan, device):
        """Bool mask of the output slots the plan names (B == 1).

        Everything else is left uninitialized by design, so a test must never
        read it.
        """
        mask = torch.zeros(
            self.world_size * BLOCKS_PER_RANK * BLOCK_NUMEL,
            dtype=torch.bool,
            device=device,
        )
        ranks = plan.src_rank[0].tolist()
        blocks = plan.src_block[0].tolist()
        for rank, block in zip(ranks, blocks, strict=True):
            if rank >= 0:
                start = (rank * BLOCKS_PER_RANK + block) * BLOCK_NUMEL
                mask[start : start + BLOCK_NUMEL] = True
        return mask

    def _all_gathered(self, kv_local, pg):
        shards = [torch.empty_like(kv_local) for _ in range(self.world_size)]
        dist.all_gather(shards, kv_local, group=pg)
        return torch.cat(shards)

    @with_comms
    def test_full_plan_forward_matches_all_gather(self):
        device = torch.device(self.device_type)
        pg, kv_local, ctx = self._setup(device)
        plan = full_plan(1, self.world_size, BLOCKS_PER_RANK, BLOCK_NUMEL, device)
        meta = build_plan_metadata(plan, pg, blocks_per_rank=BLOCKS_PER_RANK)
        out = torch.empty(
            self.world_size * kv_local.numel(), dtype=kv_local.dtype, device=device
        )
        run_p2p_gather(ctx, meta, kv_local, out)
        self.assertTrue(torch.equal(out, self._all_gathered(kv_local, pg)))

    @with_comms
    def test_full_plan_backward_reduces_over_consumers(self):
        device = torch.device(self.device_type)
        pg, kv_local, ctx = self._setup(device)
        plan = full_plan(1, self.world_size, BLOCKS_PER_RANK, BLOCK_NUMEL, device)
        meta = build_plan_metadata(plan, pg, blocks_per_rank=BLOCKS_PER_RANK)
        kv_local.requires_grad_(True)
        # full_plan: every rank reads every block, so each block's gradient is
        # summed over all cp consumers -> grad == cp_size for a unit upstream grad.
        selective_gather(kv_local, ctx, meta).sum().backward()
        grad = kv_local.grad
        assert grad is not None
        self.assertTrue(torch.equal(grad, torch.full_like(grad, self.world_size)))

    @with_comms
    def test_sliding_window_plan_forward_and_backward(self):
        device = torch.device(self.device_type)
        pg, kv_local, ctx = self._setup(device)
        rank = pg.rank()
        # Rank 1 asks for the last block of rank 0, so rank 0 only sends and
        # rank 1 only receives. Rank 0 has no predecessor: its window slot is
        # -1 padding and the matching output slots stay uninitialized.
        plan = sliding_window_plan(
            cp_rank=rank,
            cp_size=self.world_size,
            blocks_per_rank=BLOCKS_PER_RANK,
            block_numel=BLOCK_NUMEL,
            window_blocks=1,
            device=device,
        )
        meta = build_plan_metadata(plan, pg, blocks_per_rank=BLOCKS_PER_RANK)
        planned = self._planned_slots(plan, device)

        kv_local.requires_grad_(True)
        out = selective_gather(kv_local, ctx, meta)
        reference = self._all_gathered(kv_local.detach(), pg)
        self.assertTrue(torch.equal(out.detach()[planned], reference[planned]))

        out[planned].sum().backward()
        # Rank 0's last block is read by both ranks; every other block has one
        # consumer.
        expected = torch.ones_like(kv_local.detach())
        if rank == 0:
            expected[(BLOCKS_PER_RANK - 1) * BLOCK_NUMEL :] = 2.0
        grad = kv_local.grad
        assert grad is not None
        self.assertTrue(torch.equal(grad, expected))

    @with_comms
    def test_multidimensional_input_keeps_its_shape(self):
        # The transport works on element counts, so a K/V-shaped shard is valid
        # input; its gradient has to come back in the same shape.
        device = torch.device(self.device_type)
        pg, flat, ctx = self._setup(device)
        kv_local = flat.view(BLOCKS_PER_RANK, BLOCK_NUMEL).clone().requires_grad_(True)
        plan = full_plan(1, self.world_size, BLOCKS_PER_RANK, BLOCK_NUMEL, device)
        meta = build_plan_metadata(plan, pg, blocks_per_rank=BLOCKS_PER_RANK)
        selective_gather(kv_local, ctx, meta).sum().backward()
        grad = kv_local.grad
        assert grad is not None
        self.assertEqual(grad.shape, kv_local.shape)
        self.assertTrue(torch.equal(grad, torch.full_like(grad, self.world_size)))

    @with_comms
    def test_rejects_num_valid_wrong_on_one_rank_only(self):
        # Rank 1's count is wrong; rank 0's plan is fine. Both must reject it,
        # or rank 0 walks into a collective rank 1 never joins.
        device = torch.device(self.device_type)
        pg, _, _ = self._setup(device)
        rank = pg.rank()
        plan = full_plan(1, self.world_size, BLOCKS_PER_RANK, BLOCK_NUMEL, device)
        if rank == 1:
            plan = BlockGatherPlan(
                block_numel=BLOCK_NUMEL,
                src_rank=plan.src_rank,
                src_block=plan.src_block,
                num_valid=plan.num_valid - 1,
            )
        with self.assertRaises(ValueError):
            build_plan_metadata(plan, pg, blocks_per_rank=BLOCKS_PER_RANK)

    @with_comms
    def test_rejects_non_int32_plan_on_one_rank_only(self):
        # A float plan used to be cast, which silently truncates it into a
        # different plan. Only rank 1 is invalid here: both ranks must reject it,
        # or rank 0 waits in the plan all_gather that rank 1 never reaches.
        device = torch.device(self.device_type)
        pg, _, _ = self._setup(device)
        plan = full_plan(1, self.world_size, BLOCKS_PER_RANK, BLOCK_NUMEL, device)
        if pg.rank() == 1:
            plan = BlockGatherPlan(
                block_numel=BLOCK_NUMEL,
                src_rank=plan.src_rank.float(),
                src_block=plan.src_block.float(),
                num_valid=plan.num_valid,
            )
        with self.assertRaises(ValueError):
            build_plan_metadata(plan, pg, blocks_per_rank=BLOCKS_PER_RANK)

    @with_comms
    def test_rejects_capacity_that_differs_across_ranks(self):
        # Gathering plans of different capacity is not even well defined, so the
        # disagreement has to be caught before that collective.
        device = torch.device(self.device_type)
        pg, _, _ = self._setup(device)
        window = 1 if pg.rank() == 0 else 2
        plan = sliding_window_plan(
            cp_rank=pg.rank(),
            cp_size=self.world_size,
            blocks_per_rank=BLOCKS_PER_RANK,
            block_numel=BLOCK_NUMEL,
            window_blocks=window,
            device=device,
        )
        with self.assertRaises(ValueError):
            build_plan_metadata(plan, pg, blocks_per_rank=BLOCKS_PER_RANK)

    @with_comms
    def test_rejects_blocks_per_rank_that_differs_across_ranks(self):
        # The bound the ids are validated against has to be the same everywhere,
        # or each rank accepts a different set of ids.
        device = torch.device(self.device_type)
        pg, _, _ = self._setup(device)
        plan = full_plan(1, self.world_size, BLOCKS_PER_RANK, BLOCK_NUMEL, device)
        bound = BLOCKS_PER_RANK + 1 if pg.rank() == 1 else BLOCKS_PER_RANK
        with self.assertRaises(ValueError):
            build_plan_metadata(plan, pg, blocks_per_rank=bound)

    @with_comms
    def test_rejects_block_numel_that_differs_across_ranks(self):
        # Plan shapes still match here, so only the agreement descriptor can
        # catch it -- and it decides the P2P message size.
        device = torch.device(self.device_type)
        pg, _, _ = self._setup(device)
        numel = BLOCK_NUMEL * 2 if pg.rank() == 1 else BLOCK_NUMEL
        plan = full_plan(1, self.world_size, BLOCKS_PER_RANK, numel, device)
        with self.assertRaises(ValueError):
            build_plan_metadata(plan, pg, blocks_per_rank=BLOCKS_PER_RANK)

    @with_comms
    def test_float64_gradient_is_not_narrowed(self):
        # The backward accumulates in at least float32; for float64 input that
        # must not mean narrowing to float32, which overflows to inf here.
        device = torch.device(self.device_type)
        pg, _, _ = self._setup(device)
        shard_numel = BLOCKS_PER_RANK * BLOCK_NUMEL
        kv_local = torch.randn(shard_numel, device=device, dtype=torch.float64)
        ctx = SelectiveGatherContext(
            pg,
            shard_numel=shard_numel,
            block_numel=BLOCK_NUMEL,
            dtype=torch.float64,
            device=device,
            backend="p2p",
        )
        plan = full_plan(1, self.world_size, BLOCKS_PER_RANK, BLOCK_NUMEL, device)
        meta = build_plan_metadata(plan, pg, blocks_per_rank=BLOCKS_PER_RANK)
        kv_local.requires_grad_(True)
        out = selective_gather(kv_local, ctx, meta)
        out.backward(torch.full_like(out, 1e300))
        grad = kv_local.grad
        assert grad is not None
        expected = torch.full_like(grad, self.world_size * 1e300)
        self.assertTrue(torch.equal(grad, expected))

    @with_comms
    def test_rejects_plan_without_own_blocks(self):
        device = torch.device(self.device_type)
        pg, kv_local, ctx = self._setup(device)
        remote_only = sliding_window_plan(
            cp_rank=pg.rank(),
            cp_size=self.world_size,
            blocks_per_rank=BLOCKS_PER_RANK,
            block_numel=BLOCK_NUMEL,
            window_blocks=1,
            device=device,
            include_own=False,
        )
        meta = build_plan_metadata(remote_only, pg, blocks_per_rank=BLOCKS_PER_RANK)
        kv_local.requires_grad_(True)
        with self.assertRaises(ValueError):
            selective_gather(kv_local, ctx, meta)

    @with_comms
    def test_rejects_duplicate_remote_block(self):
        device = torch.device(self.device_type)
        pg, _, _ = self._setup(device)
        # Rank 1 asks for rank 0's block 1 twice, which would send and
        # accumulate that block's gradient twice. Rank 0's row is padded to the
        # same capacity. Both ranks see the whole gathered plan, so both raise
        # and neither is left waiting in the next collective.
        rank = pg.rank()
        rows = ([rank, rank, -1], [0, 1, -1]) if rank == 0 else ([1, 0, 0], [0, 1, 1])
        duplicated = BlockGatherPlan(
            block_numel=BLOCK_NUMEL,
            src_rank=torch.tensor([rows[0]], dtype=torch.int32, device=device),
            src_block=torch.tensor([rows[1]], dtype=torch.int32, device=device),
            num_valid=torch.tensor(
                [2 if rank == 0 else 3], dtype=torch.int32, device=device
            ),
        )
        with self.assertRaises(ValueError):
            build_plan_metadata(duplicated, pg, blocks_per_rank=BLOCKS_PER_RANK)

    @with_comms
    def test_rejects_device_mismatch(self):
        device = torch.device(self.device_type)
        pg, kv_local, _ = self._setup(device)
        if device.type == "cpu":
            self.skipTest("needs indexed devices")
        # A context pinned to another rank's device.
        other = torch.device(device.type, (pg.rank() + 1) % self.world_size)
        ctx = SelectiveGatherContext(
            pg,
            shard_numel=kv_local.numel(),
            block_numel=BLOCK_NUMEL,
            dtype=kv_local.dtype,
            device=other,
            backend="p2p",
        )
        plan = full_plan(1, self.world_size, BLOCKS_PER_RANK, BLOCK_NUMEL, device)
        meta = build_plan_metadata(plan, pg, blocks_per_rank=BLOCKS_PER_RANK)
        kv_local.requires_grad_(True)
        with self.assertRaises(ValueError):
            selective_gather(kv_local, ctx, meta)

    @with_comms
    def test_rejects_block_bound_the_context_disagrees_with(self):
        device = torch.device(self.device_type)
        pg, kv_local, ctx = self._setup(device)
        # Metadata validated against a larger bound would have let through block
        # ids that index past the context's shard.
        plan = full_plan(1, self.world_size, BLOCKS_PER_RANK, BLOCK_NUMEL, device)
        meta = build_plan_metadata(plan, pg, blocks_per_rank=BLOCKS_PER_RANK + 1)
        kv_local.requires_grad_(True)
        with self.assertRaises(ValueError):
            selective_gather(kv_local, ctx, meta)

    @with_comms
    def test_direct_transport_rejects_the_same_mismatch(self):
        # run_p2p_gather is public, so it has to enforce the contract too, not
        # just the autograd path.
        device = torch.device(self.device_type)
        pg, kv_local, ctx = self._setup(device)
        plan = full_plan(1, self.world_size, BLOCKS_PER_RANK, BLOCK_NUMEL, device)
        meta = build_plan_metadata(plan, pg, blocks_per_rank=BLOCKS_PER_RANK + 1)
        out = torch.empty(
            self.world_size * kv_local.numel(), dtype=kv_local.dtype, device=device
        )
        with self.assertRaises(ValueError):
            run_p2p_gather(ctx, meta, kv_local, out)

    @with_comms
    def test_direct_backward_rejects_plan_without_own_blocks(self):
        # The backward reads the whole own region, so it needs the same plan
        # contract selective_gather enforces.
        device = torch.device(self.device_type)
        pg, kv_local, ctx = self._setup(device)
        remote_only = sliding_window_plan(
            cp_rank=pg.rank(),
            cp_size=self.world_size,
            blocks_per_rank=BLOCKS_PER_RANK,
            block_numel=BLOCK_NUMEL,
            window_blocks=1,
            device=device,
            include_own=False,
        )
        meta = build_plan_metadata(remote_only, pg, blocks_per_rank=BLOCKS_PER_RANK)
        d_out = torch.zeros(
            self.world_size * kv_local.numel(), dtype=kv_local.dtype, device=device
        )
        with self.assertRaises(ValueError):
            run_p2p_gather_backward(ctx, meta, d_out, torch.empty_like(kv_local))

    @with_comms
    def test_rejects_metadata_from_another_group(self):
        # Same size, different peers: the plan's local rank ids would name the
        # wrong ranks.
        device = torch.device(self.device_type)
        pg, kv_local, ctx = self._setup(device)
        plan = full_plan(1, self.world_size, BLOCKS_PER_RANK, BLOCK_NUMEL, device)
        meta = build_plan_metadata(plan, pg, blocks_per_rank=BLOCKS_PER_RANK)
        object.__setattr__(meta, "group_ranks", tuple(r + 100 for r in ctx.group_ranks))
        kv_local.requires_grad_(True)
        with self.assertRaises(ValueError):
            selective_gather(kv_local, ctx, meta)

    @with_comms
    def test_rejects_non_contiguous_destinations(self):
        device = torch.device(self.device_type)
        pg, kv_local, ctx = self._setup(device)
        plan = full_plan(1, self.world_size, BLOCKS_PER_RANK, BLOCK_NUMEL, device)
        meta = build_plan_metadata(plan, pg, blocks_per_rank=BLOCKS_PER_RANK)
        full_numel = self.world_size * kv_local.numel()
        # Right numel, strided: reshape would copy and the writes would be lost.
        strided_out = torch.empty(2 * full_numel, dtype=kv_local.dtype, device=device)[
            ::2
        ]
        with self.assertRaises(ValueError):
            run_p2p_gather(ctx, meta, kv_local, strided_out)

        strided_grad = torch.empty(
            2 * kv_local.numel(), dtype=kv_local.dtype, device=device
        )[::2]
        d_out = torch.zeros(full_numel, dtype=kv_local.dtype, device=device)
        with self.assertRaises(ValueError):
            run_p2p_gather_backward(ctx, meta, d_out, strided_grad)

    @with_comms
    def test_rejects_dtype_mismatch(self):
        device = torch.device(self.device_type)
        pg, kv_local, _ = self._setup(device)
        # A context declaring bfloat16 would post bfloat16 receives while the
        # peers send float32.
        ctx = SelectiveGatherContext(
            pg,
            shard_numel=kv_local.numel(),
            block_numel=BLOCK_NUMEL,
            dtype=torch.bfloat16,
            device=device,
            backend="p2p",
        )
        plan = full_plan(1, self.world_size, BLOCKS_PER_RANK, BLOCK_NUMEL, device)
        meta = build_plan_metadata(plan, pg, blocks_per_rank=BLOCKS_PER_RANK)
        kv_local.requires_grad_(True)
        with self.assertRaises(ValueError):
            selective_gather(kv_local, ctx, meta)

    @with_comms
    def test_rejects_unsupported_backend(self):
        device = torch.device(self.device_type)
        pg, kv_local, _ = self._setup(device)
        with self.assertRaises(ValueError):
            SelectiveGatherContext(
                pg,
                shard_numel=kv_local.numel(),
                block_numel=BLOCK_NUMEL,
                dtype=kv_local.dtype,
                device=device,
                backend="nvlink",
            )


def _lsa_capable():
    if not torch.cuda.is_available():
        return False
    from torchtitan.distributed.context_parallel.selective_gather import backend

    dev = torch.device("cuda:0")
    return (
        backend._cutlass_ok()
        and backend._nccl4py_ok()
        and backend._nccl_ok()
        and backend._hopper_ok(dev)
    )


# bf16 at 2048 elements keeps words_per_block a multiple of the 256-thread CTA
# tiling the LSA kernels require.
LSA_BLOCK_NUMEL = 2048


@unittest.skipUnless(_lsa_capable(), "LSA needs Hopper + CuTeDSL + nccl4py")
class TestLSAGatherNumerics(DTensorTestBase):
    """The LSA CuTeDSL backend reproduces a plain all-gather bitwise (full_plan).

    Mirrors TestP2PGatherNumerics against the "lsa" backend.
    """

    @property
    def world_size(self):
        return 2

    def _setup(self, device):
        mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("cp",)
        )
        pg = mesh.get_group("cp")
        shard_numel = BLOCKS_PER_RANK * LSA_BLOCK_NUMEL
        torch.manual_seed(1234 + pg.rank())
        kv_local = torch.randn(shard_numel, device=device, dtype=torch.bfloat16)
        plan = full_plan(1, self.world_size, BLOCKS_PER_RANK, LSA_BLOCK_NUMEL, device)
        meta = build_plan_metadata(plan, pg, blocks_per_rank=BLOCKS_PER_RANK)
        ctx = SelectiveGatherContext(
            mesh,
            mesh_axis="cp",
            shard_numel=shard_numel,
            block_numel=LSA_BLOCK_NUMEL,
            dtype=kv_local.dtype,
            device=device,
            max_consumers=meta.max_consumers,
            backend="lsa",
        )
        return pg, kv_local, ctx, meta

    @with_comms
    def test_full_plan_forward_matches_all_gather(self):
        device = torch.device(self.device_type)
        pg, kv_local, ctx, meta = self._setup(device)
        out = selective_gather(kv_local, ctx, meta)
        gathered = [torch.empty_like(kv_local) for _ in range(self.world_size)]
        dist.all_gather(gathered, kv_local, group=pg)
        self.assertTrue(torch.equal(out, torch.cat(gathered)))
        ctx.close()

    @with_comms
    def test_full_plan_backward_reduces_over_consumers(self):
        device = torch.device(self.device_type)
        _, kv_local, ctx, meta = self._setup(device)
        kv_local.requires_grad_(True)
        selective_gather(kv_local, ctx, meta).sum().backward()
        grad = kv_local.grad
        assert grad is not None
        self.assertTrue(torch.equal(grad, torch.full_like(grad, self.world_size)))
        ctx.close()

    @with_comms
    def test_rejects_metadata_from_another_group(self):
        # The shared guard has to run on the LSA path too: without it these
        # local rank ids reach raw window offsets instead of raising.
        device = torch.device(self.device_type)
        _, kv_local, ctx, meta = self._setup(device)
        object.__setattr__(meta, "group_ranks", tuple(r + 100 for r in ctx.group_ranks))
        kv_local.requires_grad_(True)
        with self.assertRaises(ValueError):
            selective_gather(kv_local, ctx, meta)
        ctx.close()

    @with_comms
    def test_rejects_dtype_mismatch(self):
        device = torch.device(self.device_type)
        _, kv_local, ctx, meta = self._setup(device)
        wrong = kv_local.detach().float().requires_grad_(True)
        with self.assertRaises(ValueError):
            selective_gather(wrong, ctx, meta)
        ctx.close()

    @with_comms
    def test_rejects_metadata_tensors_on_another_device(self):
        # The kernels dereference these on ctx.device, and neither .to(int32)
        # nor .contiguous() would move them back.
        device = torch.device(self.device_type)
        pg, kv_local, ctx, meta = self._setup(device)
        other = torch.device(device.type, (pg.rank() + 1) % self.world_size)
        object.__setattr__(meta.plan, "src_rank", meta.plan.src_rank.to(other))
        kv_local.requires_grad_(True)
        with self.assertRaises(ValueError):
            selective_gather(kv_local, ctx, meta)
        ctx.close()


class TestBackendSelection(DTensorTestBase):
    """Automatic selection has to be group-wide and operation-aware.

    LSA reads peer shards over NVLink, so it is only valid inside one host, and
    every rank must reach the same verdict: a split choice deadlocks the group.
    """

    @property
    def world_size(self):
        return 2

    def _group(self):
        mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("cp",)
        )
        return mesh.get_group("cp")

    def _lsa_sized_context(self, device, *, backend):
        """A context shaped so LSA is otherwise a valid choice."""
        return SelectiveGatherContext(
            self._group(),
            shard_numel=BLOCKS_PER_RANK * LSA_BLOCK_NUMEL,
            block_numel=LSA_BLOCK_NUMEL,
            dtype=torch.bfloat16,
            device=device,
            backend=backend,
        )

    @with_comms
    def test_p2p_for_a_dtype_lsa_cannot_run(self):
        # float64 would reach the kernel and raise NotImplementedError.
        device = torch.device(self.device_type)
        chosen = select_backend(
            device, self._group(), dtype=torch.float64, block_numel=LSA_BLOCK_NUMEL
        )
        self.assertEqual(chosen, "p2p")

    @with_comms
    def test_p2p_for_a_block_shape_lsa_cannot_tile(self):
        # BLOCK_NUMEL bf16 is 8 bytes, far under one 4096-byte CTA tile.
        device = torch.device(self.device_type)
        chosen = select_backend(
            device, self._group(), dtype=torch.bfloat16, block_numel=BLOCK_NUMEL
        )
        self.assertEqual(chosen, "p2p")

    @with_comms
    def test_p2p_without_a_group_to_agree_with(self):
        device = torch.device(self.device_type)
        chosen = select_backend(
            device, None, dtype=torch.bfloat16, block_numel=LSA_BLOCK_NUMEL
        )
        self.assertEqual(chosen, "p2p")

    @unittest.skipUnless(_lsa_capable(), "LSA needs Hopper + CuTeDSL + nccl4py")
    @with_comms
    def test_lsa_when_the_whole_group_can_run_it(self):
        # The positive control: without it the fallback tests pass vacuously on
        # a box that could never pick LSA anyway.
        device = torch.device(self.device_type)
        chosen = select_backend(
            device, self._group(), dtype=torch.bfloat16, block_numel=LSA_BLOCK_NUMEL
        )
        self.assertEqual(chosen, "lsa")

    @unittest.skipUnless(_lsa_capable(), "LSA needs Hopper + CuTeDSL + nccl4py")
    @with_comms
    def test_gin_when_the_ranks_are_on_different_hosts(self):
        # LSA cannot read peers over NVLink across nodes; GIN pushes over the
        # network instead.
        device = torch.device(self.device_type)
        pg = self._group()
        with mock.patch("socket.gethostname", return_value=f"host{pg.rank()}"):
            chosen = select_backend(
                device, pg, dtype=torch.bfloat16, block_numel=LSA_BLOCK_NUMEL
            )
        self.assertEqual(chosen, "gin")

    @unittest.skipUnless(_lsa_capable(), "LSA needs Hopper + CuTeDSL + nccl4py")
    @with_comms
    def test_p2p_when_an_inter_node_group_cannot_run_gin(self):
        # BLOCK_NUMEL is under one reduce tile, so GIN cannot run it either and
        # the group has no CuTeDSL option left.
        device = torch.device(self.device_type)
        pg = self._group()
        with mock.patch("socket.gethostname", return_value=f"host{pg.rank()}"):
            chosen = select_backend(
                device, pg, dtype=torch.bfloat16, block_numel=BLOCK_NUMEL
            )
        self.assertEqual(chosen, "p2p")

    @unittest.skipUnless(_lsa_capable(), "LSA needs Hopper + CuTeDSL + nccl4py")
    @with_comms
    def test_p2p_when_one_rank_is_not_capable(self):
        # Only rank 1 fails the probe. Both ranks must still choose p2p, and the
        # capable one must not block waiting on a collective the other skipped.
        device = torch.device(self.device_type)
        pg = self._group()
        with mock.patch.object(backend_mod, "_hopper_ok", lambda dev: pg.rank() == 0):
            chosen = select_backend(
                device, pg, dtype=torch.bfloat16, block_numel=LSA_BLOCK_NUMEL
            )
        self.assertEqual(chosen, "p2p")

    @unittest.skipUnless(_lsa_capable(), "LSA needs Hopper + CuTeDSL + nccl4py")
    @with_comms
    def test_context_falls_back_when_the_group_spans_lsa_teams(self):
        # A shared hostname does not prove one LSA team; lsa_pointer addresses a
        # peer inside the local team, so more than one means the wrong mapping.
        import nccl.core as nccl

        device = torch.device(self.device_type)
        with mock.patch.object(
            nccl.Communicator, "n_lsa_teams", property(lambda self: 2)
        ):
            ctx = self._lsa_sized_context(device, backend=None)
        self.assertEqual(ctx.backend, "p2p")
        ctx.close()

    @unittest.skipUnless(_lsa_capable(), "LSA needs Hopper + CuTeDSL + nccl4py")
    @with_comms
    def test_context_falls_back_without_device_api_support(self):
        import nccl.core as nccl

        device = torch.device(self.device_type)
        with mock.patch.object(
            nccl.Communicator, "device_api_support", property(lambda self: False)
        ):
            ctx = self._lsa_sized_context(device, backend=None)
        self.assertEqual(ctx.backend, "p2p")
        ctx.close()

    @unittest.skipUnless(_lsa_capable(), "LSA needs Hopper + CuTeDSL + nccl4py")
    @with_comms
    def test_forced_lsa_raises_when_the_group_spans_lsa_teams(self):
        # Forcing the backend is user configuration, so it fails loudly instead
        # of quietly running something else.
        import nccl.core as nccl

        device = torch.device(self.device_type)
        with mock.patch.object(
            nccl.Communicator, "n_lsa_teams", property(lambda self: 2)
        ):
            with self.assertRaises(ValueError):
                self._lsa_sized_context(device, backend="lsa")

    @with_comms
    def test_context_auto_selects_p2p_for_an_unsupported_dtype(self):
        # End to end through the context, which is what a caller actually hits.
        device = torch.device(self.device_type)
        ctx = SelectiveGatherContext(
            self._group(),
            shard_numel=BLOCKS_PER_RANK * LSA_BLOCK_NUMEL,
            block_numel=LSA_BLOCK_NUMEL,
            dtype=torch.float64,
            device=device,
        )
        self.assertEqual(ctx.backend, "p2p")
        ctx.close()


class TestGINMetadata(DTensorTestBase):
    """The GIN plan-transpose (build_gin_metadata) on the multi-process harness.

    full_plan on cp=2: every rank pushes both its blocks to the other rank and
    receives both of the other rank's blocks; each block has a single consumer.
    """

    @property
    def world_size(self):
        return 2

    @with_comms
    def test_full_plan_transpose(self):
        device = torch.device(self.device_type)
        mesh = init_device_mesh(
            self.device_type, (self.world_size,), mesh_dim_names=("cp",)
        )
        pg = mesh.get_group("cp")
        other = 1 - pg.rank()
        plan = full_plan(1, self.world_size, BLOCKS_PER_RANK, BLOCK_NUMEL, device)
        gin = build_gin_metadata(plan, pg)

        # Forward: push my two blocks to the other rank; own = my two blocks.
        self.assertEqual(gin.send_peer.tolist(), [other, other])
        self.assertEqual(sorted(gin.send_src_block.tolist()), [0, 1])
        self.assertEqual(sorted(gin.own_src_block.tolist()), [0, 1])
        # Receive the other rank's two blocks.
        self.assertEqual(gin.num_recv, 2)
        # Backward: push grads for those two remote blocks back to the other
        # rank, into staging slot 0 (each producer has a single consumer here).
        self.assertEqual(gin.grad_send_peer.tolist(), [other, other])
        self.assertEqual(gin.grad_send_slot.tolist(), [0, 0])
        self.assertEqual(gin.max_consumers, 1)
        # Copy-in stages exactly the two unique pushed blocks.
        self.assertEqual(sorted(gin.copyin_src_block.tolist()), [0, 1])


if __name__ == "__main__":
    unittest.main()

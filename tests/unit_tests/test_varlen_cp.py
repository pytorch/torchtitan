# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pure-logic CPU tests for ``CPVarlenMetadata.from_global``.

These do not require a real distributed environment; we mock the
``DeviceMesh`` with ``_MockMesh`` and iterate over ranks manually.
"""

import torch
from torch.distributed.tensor.experimental._attention import _HeadTailLoadBalancer
from torch.testing._internal.common_utils import run_tests, TestCase

from torchtitan.distributed.varlen_cp import CPVarlenMetadata
from torchtitan.models.common.attention import VarlenMetadata


class _MockMesh:
    """Minimal DeviceMesh stand-in for non-distributed unit tests."""

    ndim = 1

    def __init__(self, world_size: int, rank: int, device_type: str = "cpu") -> None:
        self._world_size = world_size
        self._rank = rank
        self.device_type = device_type

    def size(self) -> int:
        return self._world_size

    def get_local_rank(self) -> int:
        return self._rank


def _build_varlen_meta(cu_seq_q: list[int], B: int, seq_len: int) -> VarlenMetadata:
    """Build a global VarlenMetadata from a per-batch cumulative tensor.

    The given ``cu_seq_q`` covers a single batch element ``[0, ..., seq_len]``.
    It is replicated across ``B`` batches with offsets, matching what
    ``create_varlen_metadata_for_document`` produces.
    """
    if cu_seq_q[0] != 0 or cu_seq_q[-1] != seq_len:
        raise ValueError("cu_seq_q must start at 0 and end at seq_len")
    per_batch = torch.tensor(cu_seq_q, dtype=torch.int32)
    parts = [per_batch[:-1] + b * seq_len for b in range(B)]
    parts.append(torch.tensor([B * seq_len], dtype=torch.int32))
    cu = torch.cat(parts)
    seg_lens = torch.diff(cu)
    return VarlenMetadata(
        cu_seq_q=cu,
        cu_seq_k=cu,
        max_q=int(seg_lens.max().item()),
        max_k=int(seg_lens.max().item()),
    )


class TestCPVarlenMetadata(TestCase):
    def _run(
        self,
        global_meta: VarlenMetadata,
        cp_world_size: int,
        B: int,
        seq_len: int,
        load_balancer_factory=None,
    ) -> list[CPVarlenMetadata]:
        results = []
        for rank in range(cp_world_size):
            mesh = _MockMesh(cp_world_size, rank)
            lb = (
                load_balancer_factory(seq_len, cp_world_size)
                if load_balancer_factory
                else None
            )
            results.append(
                CPVarlenMetadata.from_global(
                    global_meta,
                    device_mesh=mesh,
                    batch_size=B,
                    seq_length=seq_len,
                    load_balancer=lb,
                )
            )
        return results

    @staticmethod
    def _seg_lens(meta: VarlenMetadata) -> tuple[list[int], list[int]]:
        return (
            torch.diff(meta.cu_seq_q).tolist(),
            torch.diff(meta.cu_seq_k).tolist(),
        )

    def test_single_doc_no_loadbalancer(self) -> None:
        # B=1, one doc spanning [0, 32). CP=2 -> rank 0 gets [0, 16),
        # rank 1 gets [16, 32). rank 1 is mid-doc so its seqlen_k = 32.
        meta = _build_varlen_meta([0, 32], B=1, seq_len=32)
        per_rank = self._run(meta, cp_world_size=2, B=1, seq_len=32)
        self.assertEqual(self._seg_lens(per_rank[0]), ([16], [16]))
        self.assertEqual(self._seg_lens(per_rank[1]), ([16], [32]))
        self.assertEqual(per_rank[0].max_q, 16)
        self.assertEqual(per_rank[1].max_k, 32)

    def test_single_doc_headtail(self) -> None:
        # seq_len=32, CP=2, headtail chunks=8 -> rearranged
        # [0..7, 24..31, 8..15, 16..23]. Rank 0 gets [0..7, 24..31] (one
        # contiguous + one tail chunk -> 2 segments). Rank 1 gets [8..23]
        # contiguous -> 1 segment.
        meta = _build_varlen_meta([0, 32], B=1, seq_len=32)
        per_rank = self._run(
            meta,
            cp_world_size=2,
            B=1,
            seq_len=32,
            load_balancer_factory=lambda s, w: _HeadTailLoadBalancer(
                s, w, torch.device("cpu")
            ),
        )
        self.assertEqual(self._seg_lens(per_rank[0]), ([8, 8], [8, 32]))
        self.assertEqual(self._seg_lens(per_rank[1]), ([16], [24]))

    def test_multi_doc_spanning(self) -> None:
        # docs [0,10), [10,50), [50,64). CP=2, no LB.
        # rank 0 has [0..32): doc 0 fully + doc 1 prefix [10..32).
        # rank 1 has [32..64): doc 1 mid [32..50) + doc 2 fully [50..64).
        meta = _build_varlen_meta([0, 10, 50, 64], B=1, seq_len=64)
        per_rank = self._run(meta, cp_world_size=2, B=1, seq_len=64)
        self.assertEqual(self._seg_lens(per_rank[0]), ([10, 22], [10, 22]))
        self.assertEqual(self._seg_lens(per_rank[1]), ([18, 14], [40, 14]))

    def test_multi_doc_headtail(self) -> None:
        # Same docs as test_multi_doc_spanning, with headtail balancer
        # (chunk=16). Forward permutation is [0..16, 48..64, 16..48].
        meta = _build_varlen_meta([0, 10, 50, 64], B=1, seq_len=64)
        per_rank = self._run(
            meta,
            cp_world_size=2,
            B=1,
            seq_len=64,
            load_balancer_factory=lambda s, w: _HeadTailLoadBalancer(
                s, w, torch.device("cpu")
            ),
        )
        self.assertEqual(self._seg_lens(per_rank[0]), ([10, 6, 2, 14], [10, 6, 40, 14]))
        self.assertEqual(self._seg_lens(per_rank[1]), ([32], [38]))

        # k_global_gather_indices compose the K-gather with the inverse permutation.
        # Rank 1 K covers original [10..48) -> [10..16) stays, [16..48) +16.
        expected_rank1 = torch.tensor(
            list(range(10, 16)) + list(range(32, 64)), dtype=torch.long
        )
        self.assertEqual(per_rank[1].k_global_gather_indices, expected_rank1)
        # Rank 0 K-gather covers originals [0..10), [10..16), [10..50), [50..64).
        expected_rank0 = torch.tensor(
            list(range(10))
            + list(range(10, 16))
            + list(range(10, 16))
            + list(range(32, 64))
            + [16, 17]
            + list(range(18, 32)),
            dtype=torch.long,
        )
        self.assertEqual(per_rank[0].k_global_gather_indices, expected_rank0)

    def test_one_token_segment(self) -> None:
        # 1-token doc + 7-token doc; seq_len=8, CP=2.
        meta = _build_varlen_meta([0, 1, 8], B=1, seq_len=8)
        per_rank = self._run(meta, cp_world_size=2, B=1, seq_len=8)
        self.assertEqual(self._seg_lens(per_rank[0]), ([1, 3], [1, 3]))
        self.assertEqual(self._seg_lens(per_rank[1]), ([4], [7]))

    def test_multi_batch_multi_doc_spanning(self) -> None:
        # B=2, per-batch docs [10, 22] -> packed cu_seq_q =
        # [0, 10, 32, 42, 64]. CP=2, shard_len=16.
        # Rank 0: per-batch [0..16) -> 4 segments; k_global_gather_indices must
        # pick out [0..10), [10..16), [32..42), [42..48) from packed K.
        meta = _build_varlen_meta([0, 10, 32], B=2, seq_len=32)
        per_rank = self._run(meta, cp_world_size=2, B=2, seq_len=32)
        self.assertEqual(self._seg_lens(per_rank[0]), ([10, 6, 10, 6], [10, 6, 10, 6]))
        self.assertEqual(self._seg_lens(per_rank[1]), ([16, 16], [22, 22]))
        expected_rank0_k = torch.tensor(
            list(range(10))
            + list(range(10, 16))
            + list(range(32, 42))
            + list(range(42, 48)),
            dtype=torch.long,
        )
        self.assertEqual(per_rank[0].k_global_gather_indices, expected_rank0_k)

    def test_multi_batch_headtail(self) -> None:
        # B=2, single doc per batch (cu_seq_q=[0,32,64]). CP=2, headtail
        # chunk=8 -> per-batch forward perm [0..8, 24..32, 8..16, 16..24].
        # Exercises the per-batch rearrange + per-batch inverse permutation
        # composition with B > 1, which is the most error-prone path.
        meta = _build_varlen_meta([0, 32], B=2, seq_len=32)
        per_rank = self._run(
            meta,
            cp_world_size=2,
            B=2,
            seq_len=32,
            load_balancer_factory=lambda s, w: _HeadTailLoadBalancer(
                s, w, torch.device("cpu")
            ),
        )
        # Rank 0 owns rearranged [0..8, 24..32] of each batch. In packed
        # globals those are [0..8, 24..32, 32..40, 56..64], giving 4
        # segments split by the global-discontinuities and the doc 0/1
        # boundary at packed position 16.
        self.assertEqual(self._seg_lens(per_rank[0]), ([8, 8, 8, 8], [8, 32, 8, 32]))
        expected_rank0_k = torch.tensor(
            list(range(0, 8))
            + list(range(0, 8))
            + list(range(16, 24))
            + list(range(24, 32))
            + list(range(8, 16))
            + list(range(32, 40))
            + list(range(32, 40))
            + list(range(48, 56))
            + list(range(56, 64))
            + list(range(40, 48)),
            dtype=torch.long,
        )
        self.assertEqual(per_rank[0].k_global_gather_indices, expected_rank0_k)
        # Rank 1 owns rearranged [8..24] of each batch. The two halves are
        # contiguous in globals, so each batch contributes one segment.
        self.assertEqual(self._seg_lens(per_rank[1]), ([16, 16], [24, 24]))
        expected_rank1_k = torch.tensor(
            list(range(0, 8))
            + list(range(16, 24))
            + list(range(24, 32))
            + list(range(32, 40))
            + list(range(48, 56))
            + list(range(56, 64)),
            dtype=torch.long,
        )
        self.assertEqual(per_rank[1].k_global_gather_indices, expected_rank1_k)

    def test_cp_world_size_4(self) -> None:
        # CP=4 exercises off-by-one in the per-rank boundary math.
        meta = _build_varlen_meta([0, 20, 64], B=1, seq_len=64)
        per_rank = self._run(meta, cp_world_size=4, B=1, seq_len=64)
        self.assertEqual(self._seg_lens(per_rank[0]), ([16], [16]))
        self.assertEqual(self._seg_lens(per_rank[1]), ([4, 12], [20, 12]))
        self.assertEqual(self._seg_lens(per_rank[2]), ([16], [28]))
        self.assertEqual(self._seg_lens(per_rank[3]), ([16], [44]))

    def test_seq_len_divisibility(self) -> None:
        meta = _build_varlen_meta([0, 30], B=1, seq_len=30)
        with self.assertRaisesRegex(ValueError, "divisible"):
            CPVarlenMetadata.from_global(
                meta, device_mesh=_MockMesh(4, 0), batch_size=1, seq_length=30
            )

    def test_seq_len_divisibility_with_load_balancer(self) -> None:
        # With a load balancer, each shard is split into 2 halves, so
        # seq_length must be divisible by 2 * cp_world_size, not just
        # cp_world_size. seq_length=6, CP=2 satisfies CP but not 2*CP.
        # The LB itself is never invoked because divisibility fires first.
        meta = _build_varlen_meta([0, 6], B=1, seq_len=6)
        lb = _HeadTailLoadBalancer(
            seq_length=8, world_size=2, device=torch.device("cpu")
        )
        with self.assertRaisesRegex(ValueError, "load balancers chunk"):
            CPVarlenMetadata.from_global(
                meta,
                device_mesh=_MockMesh(2, 0),
                batch_size=1,
                seq_length=6,
                load_balancer=lb,
            )

    def test_cross_attention_unsupported(self) -> None:
        meta = VarlenMetadata(
            cu_seq_q=torch.tensor([0, 16, 32], dtype=torch.int32),
            cu_seq_k=torch.tensor([0, 20, 40], dtype=torch.int32),
            max_q=16,
            max_k=20,
        )
        with self.assertRaisesRegex(ValueError, "self-attention"):
            CPVarlenMetadata.from_global(
                meta, device_mesh=_MockMesh(2, 0), batch_size=1, seq_length=32
            )

    def test_already_sharded_rejected(self) -> None:
        # Passing a CPVarlenMetadata as if it were a global metadata
        # should be rejected to prevent double-shard bugs.
        meta = _build_varlen_meta([0, 32], B=1, seq_len=32)
        sharded = CPVarlenMetadata.from_global(
            meta, device_mesh=_MockMesh(2, 0), batch_size=1, seq_length=32
        )
        with self.assertRaisesRegex(ValueError, "CPVarlenMetadata"):
            CPVarlenMetadata.from_global(
                sharded, device_mesh=_MockMesh(2, 0), batch_size=1, seq_length=32
            )


if __name__ == "__main__":
    run_tests()

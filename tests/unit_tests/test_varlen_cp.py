# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pure-logic CPU tests for ``CPVarlenMetadata.from_global`` and
``_VarlenPTRRLoadBalancer``.

These do not require a real distributed environment; we mock the
``DeviceMesh`` with ``_MockMesh`` and iterate over ranks manually.
"""

import torch
from torch.distributed.tensor.experimental._attention import _HeadTailLoadBalancer
from torch.testing._internal.common_utils import run_tests, TestCase

from torchtitan.distributed.varlen_cp import _VarlenPTRRLoadBalancer, CPVarlenMetadata
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

        # k_local_indices compose the K-gather with the inverse permutation.
        # Rank 1 K covers original [10..48) -> [10..16) stays, [16..48) +16.
        expected_rank1 = torch.tensor(
            list(range(10, 16)) + list(range(32, 64)), dtype=torch.long
        )
        self.assertEqual(per_rank[1].k_local_indices, expected_rank1)
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
        self.assertEqual(per_rank[0].k_local_indices, expected_rank0)

    def test_one_token_segment(self) -> None:
        # 1-token doc + 7-token doc; seq_len=8, CP=2.
        meta = _build_varlen_meta([0, 1, 8], B=1, seq_len=8)
        per_rank = self._run(meta, cp_world_size=2, B=1, seq_len=8)
        self.assertEqual(self._seg_lens(per_rank[0]), ([1, 3], [1, 3]))
        self.assertEqual(self._seg_lens(per_rank[1]), ([4], [7]))

    def test_multi_batch_multi_doc_spanning(self) -> None:
        # B=2, per-batch docs [10, 22] -> packed cu_seq_q =
        # [0, 10, 32, 42, 64]. CP=2, shard_len=16.
        # Rank 0: per-batch [0..16) -> 4 segments; k_local_indices must
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
        self.assertEqual(per_rank[0].k_local_indices, expected_rank0_k)

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
        self.assertEqual(per_rank[0].k_local_indices, expected_rank0_k)
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
        self.assertEqual(per_rank[1].k_local_indices, expected_rank1_k)

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
        # With a load balancer, seq_length must be divisible by
        # 2 * cp_world_size (the extra divisibility comes from the way
        # CP load balancers chunk each shard). seq_length=6, CP=2
        # satisfies CP but not 2*CP. The LB itself is never invoked
        # because divisibility fires first.
        meta = _build_varlen_meta([0, 6], B=1, seq_len=6)
        lb = _HeadTailLoadBalancer(
            seq_length=8, world_size=2, device=torch.device("cpu")
        )
        with self.assertRaisesRegex(ValueError, "extra divisibility"):
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


def _synthetic_doc_lengths(seq_len: int, kind: str, seed: int = 0) -> list[int]:
    """Generate doc lengths summing to seq_len.

    ``kind="mixture"`` (deterministic, hand-checkable):
        ~30 short docs U(16,64), 2 medium docs U(200,320), 1 long doc =
        remainder. Long doc placed mid-sequence (worst for headtail).
    """
    rng = torch.Generator().manual_seed(seed)
    if kind == "mixture":
        short_lens = [
            int(torch.randint(16, 65, (1,), generator=rng).item()) for _ in range(30)
        ]
        medium_lens = [
            int(torch.randint(200, 321, (1,), generator=rng).item()) for _ in range(2)
        ]
        used = sum(short_lens) + sum(medium_lens)
        if used >= seq_len:
            raise ValueError("mixture overflows seq_len; bump seq_len")
        long_len = seq_len - used
        mid = len(short_lens) // 2
        return short_lens[:mid] + medium_lens + [long_len] + short_lens[mid:]
    else:
        raise ValueError(f"unknown kind {kind!r}")


def _build_varlen_meta_from_doc_lens(
    doc_lens: list[int], B: int, seq_len: int
) -> VarlenMetadata:
    if sum(doc_lens) != seq_len:
        raise ValueError("doc_lens must sum to seq_len")
    cum = [0]
    for length in doc_lens:
        cum.append(cum[-1] + length)
    return _build_varlen_meta(cum, B=B, seq_len=seq_len)


def _per_rank_total_work(
    indices: torch.Tensor, cu_seq_q: torch.Tensor, B: int, S: int, W: int
) -> torch.Tensor:
    """Sum of visible-K-token-counts per rank for a given (B, S) permutation."""
    cu = cu_seq_q.to(torch.long)
    positions = torch.arange(B * S, dtype=torch.long)
    doc_id = torch.searchsorted(cu, positions, right=True) - 1
    work_per_token = positions - cu[doc_id] + 1
    work_per_token_2d = work_per_token.view(B, S)
    work_rearr = torch.gather(work_per_token_2d, 1, indices.to(torch.long))
    shard = S // W
    return work_rearr.view(B, W, shard).sum(dim=(0, 2))


class TestVarlenPTRRLoadBalancer(TestCase):
    @staticmethod
    def _check_perm(indices: torch.Tensor) -> None:
        B, S = indices.shape
        for b in range(B):
            torch.testing.assert_close(
                torch.sort(indices[b]).values,
                torch.arange(S, dtype=indices.dtype),
            )

    @staticmethod
    def _check_restore(lb: _VarlenPTRRLoadBalancer) -> None:
        fwd = lb._generate_indices(restore=False).to(torch.long)
        rev = lb._generate_indices(restore=True).to(torch.long)
        for b in range(fwd.shape[0]):
            roundtrip = fwd[b][rev[b]]
            torch.testing.assert_close(
                roundtrip, torch.arange(fwd.shape[1], dtype=torch.long)
            )

    def test_single_doc_hand_balance(self) -> None:
        # B=1, W=2, S=128, BS=32. One full-seq doc.
        # PTRR pairs heaviest with lightest: blocks should sum to
        # 528+3600 and 1552+2576 -- both 4128.
        S, BS, W = 128, 32, 2
        meta = _build_varlen_meta_from_doc_lens([S], B=1, seq_len=S)
        lb = _VarlenPTRRLoadBalancer(
            meta.cu_seq_q,
            batch_size=1,
            seq_length=S,
            world_size=W,
            block_size=BS,
        )
        idx = lb._generate_indices(restore=False)
        self._check_perm(idx)
        self._check_restore(lb)
        per_rank = _per_rank_total_work(idx, meta.cu_seq_q, B=1, S=S, W=W)
        torch.testing.assert_close(
            per_rank, torch.tensor([4128, 4128], dtype=per_rank.dtype)
        )

    def test_multi_batch_different_doc_structures(self) -> None:
        BS, W = 32, 2
        # batch 0: one doc of 128; batch 1: two docs of 64+64.
        cu = torch.tensor([0, 128, 192, 256], dtype=torch.int32)
        lb = _VarlenPTRRLoadBalancer(
            cu,
            batch_size=2,
            seq_length=128,
            world_size=W,
            block_size=BS,
        )
        idx = lb._generate_indices(restore=False)
        self._check_perm(idx)
        self._check_restore(lb)

    def test_balance_quality_beats_headtail(self) -> None:
        # On a skewed mixture, PTRR's per-rank-work spread should be
        # < 0.5x headtail's.
        S, BS, W = 4096, 128, 2
        doc_lens = _synthetic_doc_lengths(S, kind="mixture", seed=0)
        meta = _build_varlen_meta_from_doc_lens(doc_lens, B=1, seq_len=S)

        ptrr = _VarlenPTRRLoadBalancer(
            meta.cu_seq_q,
            batch_size=1,
            seq_length=S,
            world_size=W,
            block_size=BS,
        )
        ht = _HeadTailLoadBalancer(S, W, torch.device("cpu"))

        ptrr_work = _per_rank_total_work(
            ptrr._generate_indices(restore=False),
            meta.cu_seq_q,
            B=1,
            S=S,
            W=W,
        )
        ht_work = _per_rank_total_work(
            ht._generate_indices(restore=False),
            meta.cu_seq_q,
            B=1,
            S=S,
            W=W,
        )
        ptrr_spread = (ptrr_work.max() - ptrr_work.min()).item()
        ht_spread = (ht_work.max() - ht_work.min()).item()
        self.assertLess(
            ptrr_spread,
            0.5 * ht_spread,
            msg=(
                f"PTRR spread {ptrr_spread} should be < 0.5 * "
                f"headtail spread {ht_spread}"
            ),
        )

    def test_num_blocks_equals_world_size(self) -> None:
        S, BS, W = 64, 32, 2
        meta = _build_varlen_meta_from_doc_lens([S], B=1, seq_len=S)
        lb = _VarlenPTRRLoadBalancer(
            meta.cu_seq_q,
            batch_size=1,
            seq_length=S,
            world_size=W,
            block_size=BS,
        )
        idx = lb._generate_indices(restore=False)
        self._check_perm(idx)

    def test_block_size_divisibility_error(self) -> None:
        meta = _build_varlen_meta_from_doc_lens([128], B=1, seq_len=128)
        with self.assertRaisesRegex(ValueError, "divisible by block_size"):
            _VarlenPTRRLoadBalancer(
                meta.cu_seq_q,
                batch_size=1,
                seq_length=128,
                world_size=2,
                block_size=100,
            )

    def test_world_size_divisibility_error(self) -> None:
        meta = _build_varlen_meta_from_doc_lens([192], B=1, seq_len=192)
        with self.assertRaisesRegex(ValueError, "must be divisible by world_size"):
            _VarlenPTRRLoadBalancer(
                meta.cu_seq_q,
                batch_size=1,
                seq_length=192,
                world_size=4,
                block_size=32,
            )

    def test_through_cpvarlen_metadata_builder(self) -> None:
        # End-to-end: _VarlenPTRRLoadBalancer returns (B, S) indices,
        # which CPVarlenMetadata.from_global must accept (the per-row
        # argsort branch). Guards against the (1, S)-only regression.
        S, BS, W = 64, 32, 2
        meta = _build_varlen_meta_from_doc_lens([S], B=2, seq_len=S)
        lb = _VarlenPTRRLoadBalancer(
            meta.cu_seq_q,
            batch_size=2,
            seq_length=S,
            world_size=W,
            block_size=BS,
        )
        # Should construct without raising; sanity-check per-rank shapes.
        for rank in range(W):
            per_rank = CPVarlenMetadata.from_global(
                meta,
                device_mesh=_MockMesh(W, rank),
                batch_size=2,
                seq_length=S,
                load_balancer=lb,
            )
            # Q is sharded into B * shard_len positions.
            self.assertEqual(int(per_rank.cu_seq_q[-1].item()), 2 * (S // W))


if __name__ == "__main__":
    run_tests()

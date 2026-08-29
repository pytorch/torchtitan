# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The Ulysses full-sequence mask preserves packed-document boundaries.

Two gloo ranks each hold a contiguous positions shard; the gathered mask must
equal the mask built directly from the global positions. The synthetic stream
packs three documents so that one boundary falls ON the shard cut and one
falls INSIDE a shard -- the two cases a causal-only rebuild gets wrong.
"""

import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Three documents over 8 tokens: [0,1,2,3] [0,1] [0,1]. The cut at token 4
# lands exactly on the second document's start; the third document starts
# inside rank 1's shard.
_POSITIONS = [0, 1, 2, 3, 0, 1, 0, 1]


def _worker(rank: int, world_size: int, init_file: str, out_dir: str) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        from torchtitan.models.common.attention import (
            create_attention_mask,
            get_efficient_causal_mask_mod_for_packed_document,
        )
        from torchtitan.models.kimi_k3.sharding import full_sequence_document_mask

        positions_full = torch.tensor(_POSITIONS, dtype=torch.int64)
        shard = positions_full.chunk(world_size)[rank]

        gathered_mask = full_sequence_document_mask(None, shard, dist.group.WORLD)
        reference_mask = create_attention_mask(
            get_efficient_causal_mask_mod_for_packed_document(positions_full),
            None,
            None,
            len(_POSITIONS),
            len(_POSITIONS),
            device=positions_full.device,
        )
        from torch.nn.attention.flex_attention import create_mask

        n = len(_POSITIONS)
        dense = create_mask(gathered_mask.mask_mod, 1, 1, n, n, device="cpu")
        expected = create_mask(reference_mask.mask_mod, 1, 1, n, n, device="cpu")
        torch.save(
            {"equal": bool(torch.equal(dense, expected)), "dense": dense},
            os.path.join(out_dir, f"rank{rank}.pt"),
        )
    finally:
        dist.destroy_process_group()


class TestUlyssesDocumentMask(unittest.TestCase):
    def test_gathered_mask_matches_global_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            init_file = os.path.join(tmp, "rdzv")
            mp.spawn(_worker, args=(2, init_file, tmp), nprocs=2, join=True)
            results = [torch.load(os.path.join(tmp, f"rank{r}.pt")) for r in (0, 1)]
        for r, res in enumerate(results):
            self.assertTrue(res["equal"], f"rank {r} mask differs from reference")
        # Both ranks reassemble the same full sequence, so the masks agree.
        self.assertTrue(torch.equal(results[0]["dense"], results[1]["dense"]))
        # The boundary cases themselves: token 4 (doc 2 start, ON the cut) must
        # not attend to token 3; token 6 (doc 3 start, inside rank 1) must not
        # attend to token 5. A causal-only mask allows both.
        dense = results[0]["dense"].reshape(len(_POSITIONS), len(_POSITIONS))
        self.assertFalse(bool(dense[4, 3]))
        self.assertFalse(bool(dense[6, 5]))
        self.assertTrue(bool(dense[3, 0]))
        self.assertTrue(bool(dense[7, 6]))


if __name__ == "__main__":
    unittest.main()

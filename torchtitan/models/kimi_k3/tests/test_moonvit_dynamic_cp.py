# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dynamic CP: one image split along the patch dimension must match the whole.

Report 5.2.3 prescribes partitioning a single large image along the patch
dimension across devices and computing attention by gathering key-value pairs
across CP ranks. This pins the property that makes it a partition rather than a
different model: each rank's output for its own patches must equal the
unpartitioned result for those patches.

**Two ranks are required.** At world_size 1 the gather is the identity and the
padding mask has nothing to mask, so a single-process test passes whether or not
the gather and the mask are there at all -- the same blind spot that let a missing
all-reduce survive the suite once already.
"""

from __future__ import annotations

import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 2
DIM, HEADS, HEAD_DIM = 32, 2, 16


def _build(dim: int):
    from torchtitan.models.kimi_k3.moonvit import MoonViTConfig, MoonViTEncoderLayer

    cfg = MoonViTConfig(
        hidden_size=dim,
        intermediate_size=2 * dim,
        num_attention_heads=HEADS,
        qkv_hidden_size=HEADS * HEAD_DIM,
        num_hidden_layers=1,
        patch_size=2,
        text_hidden_size=dim,
    )
    torch.manual_seed(0)  # identical weights on every rank
    return MoonViTEncoderLayer(cfg)


def _body(rank: int, n_patches: int, queue) -> None:
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29731" if n_patches % WORLD == 0 else "29732"
        dist.init_process_group("gloo", rank=rank, world_size=WORLD)

        from torchtitan.models.kimi_k3.moonvit import CPPatchPlan

        layer = _build(DIM)
        torch.manual_seed(1)  # identical inputs on every rank
        x = torch.randn(n_patches, DIM)
        freqs = torch.polar(
            torch.ones(n_patches, HEAD_DIM // 2),
            torch.randn(n_patches, HEAD_DIM // 2),
        )
        cu = torch.tensor([0, n_patches], dtype=torch.int32)

        # Reference: the whole image on one rank, no partition.
        layer._cp_patch_plan = None
        ref = layer._attend(x, cu, freqs)

        # Partitioned: pad the tail so every rank holds an equal shard, which is
        # what a fixed-shape collective needs.
        shard = -(-n_patches // WORLD)
        padded = shard * WORLD
        x_pad = torch.zeros(padded, DIM)
        x_pad[:n_patches] = x
        f_pad = torch.ones(padded, HEAD_DIM // 2, dtype=freqs.dtype)
        f_pad[:n_patches] = freqs
        lo, hi = rank * shard, (rank + 1) * shard

        layer._cp_patch_plan = CPPatchPlan(
            group=dist.group.WORLD, valid_total=n_patches
        )
        got = layer._attend(x_pad[lo:hi], cu, f_pad[lo:hi])

        # Compare only the rows this rank really owns; the padded tail is garbage
        # by construction and is discarded when the shards are reassembled.
        valid = max(0, min(hi, n_patches) - lo)
        torch.testing.assert_close(
            got[:valid], ref[lo : lo + valid], rtol=2e-4, atol=2e-5
        )

        # The gather must be differentiable, or the tower trains on gradients
        # missing every other rank's contribution. wqkv is used by all ranks, so
        # its gradient here must be non-zero even for the rank whose own patches
        # contribute little.
        layer.zero_grad()
        got.square().sum().backward()
        g = layer.wqkv.weight.grad
        assert (
            g is not None and torch.isfinite(g).all() and g.abs().max() > 0
        ), "wqkv received no usable gradient through the gather-KV path"
        queue.put((rank, "ok", float(g.abs().max())))
    except Exception:
        import traceback

        queue.put((rank, "fail", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class TestMoonViTDynamicCP(unittest.TestCase):
    def _run(self, n_patches: int) -> None:
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        procs = [
            ctx.Process(target=_body, args=(r, n_patches, queue)) for r in range(WORLD)
        ]
        for p in procs:
            p.start()
        results = [queue.get(timeout=180) for _ in range(WORLD)]
        for p in procs:
            p.join(timeout=60)
        for rank, status, payload in results:
            self.assertEqual(status, "ok", f"rank {rank}:\n{payload}")

    def test_even_split_matches_the_whole_image(self):
        self._run(16)

    def test_padded_split_masks_the_tail(self):
        """An odd patch count pads; without the key mask the padding would join
        every softmax and the outputs would differ from the reference."""
        self._run(13)


if __name__ == "__main__":
    unittest.main()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The WHOLE tower under dynamic CP must reproduce the unpartitioned features.

``test_moonvit_dynamic_cp`` pins the attention path. This pins everything the
partition also touches and which attention alone cannot see:

* the divided_fixed absolute position embedding at the patch embed,
* 2-D RoPE inside every block,
* the projector's ``(kh, kw)`` patch merge, whose blocks must not straddle a rank,
* the order in which the shards' tokens reassemble.

Written because the end-to-end A/B did not close: partitioned training differed
from replicated by 1.9e-03 in step-1 loss. Chasing that produced two corrections
worth keeping.

**A tolerance loose enough to pass either way is not a measurement.** The first
version of this test used ``rtol=2e-3`` in fp32, where the only legitimate
difference is reduction order at 1e-6. It passed while a real 1e-3 defect was
present. It now runs at ``rtol=1e-5``.

**``--training.dtype float32`` does not reach the tower.** Measured: the tensors
arriving at ``MoonViT.forward`` are bf16 even in an fp32 run, because the "fsdp"
mesh here is ``dp_shard x cp`` -- so CP alone puts FSDP in the path, and FSDP
all-gathers the fp32 master into a bf16 compute copy. Reading
``patch_embed.proj.weight`` before the forward hook shows fp32 and is misleading.
So the training A/B has a floor of one bf16 rounding step, and the per-feature
delta it showed (1.56e-02 against a magnitude of 4.278, i.e. 3.6e-03 relative)
is exactly 2**-8. An earlier round read "the difference grows in fp32, so it is
not rounding" as evidence; the run was never fp32 where it mattered. This test
therefore verifies the arithmetic in a standalone fp32 reproducer, where nothing
casts.
"""

from __future__ import annotations

import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

WORLD = 2
DIM, HEADS, HEAD_DIM, PATCH = 32, 2, 16, 2


def _tower(dim: int, faithful: bool = False):
    from torchtitan.models.kimi_k3.moonvit import MoonViT, MoonViTConfig

    if faithful:
        # kimi_k3_debugmodel_report_arch's own vision config: 3 heads (which no TP
        # degree divides, hence the vit4h flavor elsewhere), head_dim 128, patch 14,
        # and a 64x64 position table interpolated down to the input grid.
        cfg = MoonViTConfig(
            hidden_size=256,
            num_attention_heads=3,
            qkv_hidden_size=384,
            intermediate_size=1024,
            patch_size=14,
            num_hidden_layers=4,
            merge_kernel_size=(2, 2),
            init_pos_emb_height=64,
            init_pos_emb_width=64,
            rope_max_grid=512,
            text_hidden_size=256,
        )
        torch.manual_seed(0)
        tower = MoonViT(cfg)
        tower.init_weights()
        return tower, cfg

    cfg = MoonViTConfig(
        hidden_size=dim,
        intermediate_size=2 * dim,
        num_attention_heads=HEADS,
        qkv_hidden_size=HEADS * HEAD_DIM,
        num_hidden_layers=2,
        patch_size=PATCH,
        text_hidden_size=dim,
        merge_kernel_size=(2, 2),
        init_pos_emb_height=16,
        init_pos_emb_width=16,
    )
    torch.manual_seed(0)  # identical weights on every rank
    tower = MoonViT(cfg)
    tower.init_weights()
    return tower, cfg


def _body(rank: int, grid: tuple[int, int, int], queue, faithful: bool = False) -> None:
    try:
        t, h, w = grid
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(29750 + h * 4 + t)
        dist.init_process_group("gloo", rank=rank, world_size=WORLD)

        from torchtitan.models.kimi_k3.moonvit import CPPatchPlan
        from torchtitan.models.kimi_k3.vit_cp_plan import row_partition

        tower, cfg = _tower(DIM, faithful)
        n = t * h * w
        torch.manual_seed(1)  # identical pixels on every rank
        patches = torch.randn(n, cfg.in_channels, cfg.patch_size, cfg.patch_size)
        full_grid = torch.tensor([[t, h, w]])

        with torch.no_grad():
            ref = tower(patches, full_grid)
        ref = ref[0] if isinstance(ref, list) else ref

        shards = row_partition(t, h, w, kh=2, group_size=WORLD)
        sh = shards[rank]
        band_max = max(s.row_end - s.row_start for s in shards)
        # Per frame: this rank's rows, then padding out to the widest band, so
        # every rank's tensor is (t, band_max, w) and the collective is
        # fixed-shape. Padding at the end of each frame's rows keeps the merged
        # padding at the tail of this rank's output.
        per_frame = []
        for a, b in sh.ranges:
            rows = patches[a:b]
            pad = torch.zeros(
                (band_max - (sh.row_end - sh.row_start)) * w,
                cfg.in_channels,
                cfg.patch_size,
                cfg.patch_size,
            )
            per_frame.append(torch.cat([rows, pad], dim=0) if pad.numel() else rows)
        local = torch.cat(per_frame, dim=0)
        local_grid = torch.tensor([[t, band_max, w]])
        plan = CPPatchPlan(
            group=dist.group.WORLD,
            valid_total=n,
            full_grid=(t, h, w),
            row_start=sh.row_start,
            band=band_max,
            real_rows=sh.row_end - sh.row_start,
        )
        with torch.no_grad():
            got = tower(local, local_grid, plan)
        got = got[0] if isinstance(got, list) else got

        # Reassemble in rank order and compare against the whole-image features.
        buf = [torch.empty_like(got) for _ in range(WORLD)]
        dist.all_gather(buf, got.contiguous())
        merged = torch.cat(buf, dim=0)[: ref.size(0)]
        delta = (merged - ref).abs().max().item()
        mag = ref.abs().max().item()
        # fp32 throughout, so the only legitimate difference is reduction order in
        # attention: order 1e-6, not 1e-3. A tolerance loose enough to pass either
        # way is not a measurement, which is how a 2e-3 rtol hid this once.
        torch.testing.assert_close(merged, ref, rtol=1e-5, atol=1e-6)
        queue.put((rank, "ok", {"max_abs_delta": delta, "magnitude": mag}))
    except Exception:
        import traceback

        queue.put((rank, "fail", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class TestMoonViTDynamicCPTower(unittest.TestCase):
    def _run(self, grid, faithful: bool = False) -> None:
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        procs = [
            ctx.Process(target=_body, args=(r, grid, queue, faithful))
            for r in range(WORLD)
        ]
        for p in procs:
            p.start()
        results = [queue.get(timeout=240) for _ in range(WORLD)]
        for p in procs:
            p.join(timeout=60)
        for rank, status, payload in results:
            self.assertEqual(status, "ok", f"rank {rank}:\n{payload}")

    def test_single_image_even_blocks(self):
        # h=8 -> 4 merge blocks -> 2 per rank, no padding.
        self._run((1, 8, 4))

    def test_single_image_odd_blocks_pads_the_tail(self):
        # h=6 -> 3 merge blocks over 2 ranks -> 2 then 1, so rank 1 pads.
        self._run((1, 6, 4))

    def test_report_arch_config_and_grid(self):
        """The exact tower and grid the multimodal matrix runs.

        The small grids above passed while the real one still differed in
        training, so this closes the gap between "a partition works" and "the
        partition this stack actually performs works".
        """
        self._run((1, 16, 16), faithful=True)

    def test_video_spanning_frames(self):
        """A shard that crosses a frame boundary. t>1 is where the first cut rule
        was wrong: merge-safe but not contiguous."""
        self._run((2, 8, 4))

    def test_video_with_deficit_interleaves_padding(self):
        """t>1 AND a deficit rank -- the combination none of the cases above hit.

        h=6 gives 3 merge blocks over 2 ranks, so rank 1 is short by one block and
        pads. With t>1 that padding is added PER FRAME by ``_slice_for_shard``
        (band rows per frame, then the frames concatenated), so the deficit rank's
        stream is [frame0 real, frame0 pad, frame1 real, frame1 pad], and the padded
        positions are INTERLEAVED rather than a trailing run.

        A prefix-only key mask therefore admits frame 0's padding into the softmax
        and masks frame 1's real keys. (1, 6, 4) has the deficit but only one frame;
        (2, 8, 4) has the frames but no deficit -- neither can see this.
        """
        self._run((2, 6, 4))


if __name__ == "__main__":
    unittest.main()

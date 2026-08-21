# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Dynamic CP under vision TP when attention is REPLICATED, not head-sharded.

The configuration that broke. ``parallelize.py`` only head-shards vision attention
when the head count divides the TP ranks; otherwise it warns and leaves attention
replicated. The head-sharded branch drops q/k/v to local tensors before the KV
gather, so it worked -- the replicated branch did not, and the gather hit a DTensor:

    NotImplementedError: Operator c10d.allgather_.default does not have a sharding
    strategy registered.

Every TP+CP matrix cell failed on a tower with 3 heads while the same cells passed on
a 4-head tower, which is why it read as "vision TP plus dynamic CP" rather than
"replicated attention plus dynamic CP".

**What this test covers, and what it deliberately does not.** The subject is the
conversion contract the fix introduces: a replicated DTensor taken to local for the
gather and re-wrapped afterwards, with the gradient neither dropped nor double
counted. The gather ITSELF is covered by ``test_moonvit_dynamic_cp.py``, so it is
replaced here by a local stand-in. That is not laziness about coverage -- on gloo,
``dist_nn.all_gather``'s backward cannot run on a process subgroup at all: its
scatter fallback passes a group-local index where a global rank is expected, so any
subgroup not starting at global rank 0 raises. NCCL takes the ``all_to_all`` branch
and does not hit it, which is why the GPU matrix cells run. A CPU test of tp2 x cp2
end to end is therefore not expressible; the end-to-end case is covered by the
``fsdp2_tp2_cp2`` / ``tp2_pp2_cp2`` / ``ep2_fsdp2_tp2_cp2`` matrix cells.

The load-bearing assertion is the GRADIENT, not the output. Taking a replicated
DTensor to local has to declare ``grad_placements=[Replicate()]``: every TP rank runs
the same full-head attention and receives the same full gradient, so a ``Partial``
declaration would be summed over the TP axis and scale it by tp_size. The output is
identical either way, so an output-only test passes with the wrong placement.
"""

from __future__ import annotations

import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F

WORLD = 2
DIM, HEADS, HEAD_DIM = 32, 2, 16
N_PATCHES = 16


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


def _local_attend(_self, q, k, v, _plan):
    """Stand-in for the gathering attention: same shapes, no collective.

    Keeps the surrounding code path exactly as production runs it -- the plan branch
    is taken, so the DTensor conversion under test happens -- while removing the one
    op gloo cannot run on a subgroup.
    """
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0),
        is_causal=False,
    )
    return out.squeeze(0).transpose(0, 1)


def _body(rank: int, queue) -> None:
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29743"
        dist.init_process_group("gloo", rank=rank, world_size=WORLD)

        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import distribute_module, DTensor, Replicate

        from torchtitan.models.kimi_k3.moonvit import (
            CPPatchPlan,
            MoonViTEncoderLayer,
        )

        tp_mesh = init_device_mesh("cpu", (WORLD,), mesh_dim_names=("tp",))
        MoonViTEncoderLayer._attend_gather_kv = _local_attend

        # Replicated over TP means every rank holds the SAME patches, which is what
        # makes DTensor(Replicate) truthful here. The CP split lives on a different
        # mesh axis and is the gather's business, not this contract's.
        torch.manual_seed(1)
        x = torch.randn(N_PATCHES, DIM)
        freqs = torch.polar(
            torch.ones(N_PATCHES, HEAD_DIM // 2),
            torch.randn(N_PATCHES, HEAD_DIM // 2),
        )
        cu = torch.tensor([0, N_PATCHES], dtype=torch.int32)
        plan = CPPatchPlan(group=dist.group.WORLD, valid_total=N_PATCHES)

        # Reference: plain tensors, the path that has always worked.
        plain = _build(DIM)
        plain._cp_patch_plan = plan
        ref = plain._attend(x, cu, freqs)
        ref.square().sum().backward()
        ref_grad = plain.wqkv.weight.grad.clone()

        # Under test: the layer replicated over the TP axis, so wqkv's output is a
        # DTensor(Replicate) and _tp_head_slice is None -- exactly what
        # parallelize.py leaves behind when the heads do not divide the TP ranks.
        under_test = _build(DIM)
        distribute_module(under_test, tp_mesh)
        under_test._cp_patch_plan = plan
        x_dt = DTensor.from_local(x, tp_mesh, [Replicate()], run_check=False)
        f_dt = DTensor.from_local(freqs, tp_mesh, [Replicate()], run_check=False)
        got = under_test._attend(x_dt, cu, f_dt)

        assert isinstance(got, DTensor), "wo must hand back a DTensor for the residual"
        torch.testing.assert_close(got.full_tensor(), ref, rtol=2e-4, atol=2e-5)

        got.square().sum().backward()
        grad = under_test.wqkv.weight.grad
        grad_full = grad.full_tensor() if isinstance(grad, DTensor) else grad
        assert torch.isfinite(grad_full).all(), "wqkv gradient is not finite"
        assert grad_full.abs().max() > 0, "wqkv received no gradient at all"
        # The check that catches a Partial declaration: it would land here at 2x.
        torch.testing.assert_close(grad_full, ref_grad, rtol=2e-4, atol=2e-5)

        queue.put((rank, "ok", float(grad_full.abs().max())))
    except Exception:
        import traceback

        queue.put((rank, "fail", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class TestMoonViTDynamicCPReplicatedTP(unittest.TestCase):
    def test_replicated_attention_keeps_the_gradient_unscaled(self):
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        procs = [ctx.Process(target=_body, args=(r, queue)) for r in range(WORLD)]
        for p in procs:
            p.start()
        results = [queue.get(timeout=300) for _ in range(WORLD)]
        for p in procs:
            p.join(timeout=60)
        for rank, status, payload in results:
            self.assertEqual(status, "ok", f"rank {rank}:\n{payload}")


if __name__ == "__main__":
    unittest.main()

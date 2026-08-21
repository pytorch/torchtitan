# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A Rowwise-styled LoRA base fed a PLAIN input must still be reduced.

The defect this pins: ``KimiLoRALinear.forward`` has a branch for "plain input
but a DTensor base weight", written for NoParallel descents (MoE shared experts,
where the weight is Replicate and ``to_local()`` is exact). It bypasses
``self.base`` entirely -- so the style's own collective never runs. With a
Rowwise base (``Shard(1)``, the CONTRACTED axis) that made ``base_out`` this
rank's PARTIAL product, escaping as a plain tensor, which everything downstream
assumes is replicated.

MLA's ``o_proj`` is the site: the attention output is built in plain-tensor land,
so it is the one Rowwise LoRA target that reaches the branch. Measured before the
fix, tp2, ``kimi_k3_mini_diag_4l_mla_lora``: layer 0's ``o_proj`` output differed
across ranks by 3.5e-01 against a magnitude of 2.3e-01, every activation after it
diverged, and 22 of 24 testable replicated LoRA gradients disagreed. The same
architecture without LoRA was bit-identical, and the dense FFN's ``down_proj`` --
also Rowwise, also LoRA-wrapped -- was clean, because it receives a DTensor from
the Colwise gate/up pair and so takes the ``self.base(x)`` path.

**This needs two ranks.** On a world_size=1 mesh the missing all-reduce is a
no-op and the test passes either way, which is exactly how the defect survived a
single-process suite.
"""

import os
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Replicate, Shard

IN, OUT, RANK_LORA, WORLD = 32, 8, 4, 2


def _body(rank: int, bias: bool, queue) -> None:
    try:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29631" if not bias else "29632"
        dist.init_process_group("gloo", rank=rank, world_size=WORLD)
        mesh = init_device_mesh("cpu", (WORLD,), mesh_dim_names=("tp",))

        from torchtitan.models.kimi_k3.lora import KimiLoRALinear

        torch.manual_seed(0)  # same init on both ranks
        base = nn.Linear(IN, OUT, bias=bias)
        mod = KimiLoRALinear(base, rank=RANK_LORA, alpha=8.0)
        # lora_b is zero-init by design (identity at step 0); fill it so the
        # adapter branch contributes and is checked too, not just the base.
        with torch.no_grad():
            mod.lora_b.copy_(torch.randn_like(mod.lora_b) * 0.1)

        w_full = mod.base.weight.detach().clone()
        b_full = mod.base.bias.detach().clone() if bias else None
        a_full = mod.lora_a.detach().clone()
        b_lora = mod.lora_b.detach().clone()
        scaling = mod._lora_scaling

        torch.manual_seed(1)  # same x on both ranks
        x_full = torch.randn(2, 3, IN)

        # Exactly what parallelize.py does for a Rowwise-styled LoRA module:
        # the style goes to .base, lora_a is Shard(1), lora_b is Replicate.
        mod.base.weight = nn.Parameter(
            distribute_tensor(mod.base.weight, mesh, [Shard(1)]), requires_grad=False
        )
        if bias:
            mod.base.bias = nn.Parameter(
                distribute_tensor(mod.base.bias, mesh, [Replicate()]),
                requires_grad=False,
            )
        mod.lora_a = nn.Parameter(distribute_tensor(mod.lora_a, mesh, [Shard(1)]))
        mod.lora_b = nn.Parameter(distribute_tensor(mod.lora_b, mesh, [Replicate()]))

        # The plain, per-rank input: this rank's slice of the contracted axis,
        # which is what MLA hands o_proj (its own heads' attention output).
        per = IN // WORLD
        x_local = x_full[..., rank * per : (rank + 1) * per].contiguous()

        got = mod(x_local)
        if isinstance(got, torch.Tensor) and hasattr(got, "to_local"):
            got = got.to_local()

        expected = F.linear(x_full, w_full, b_full) + scaling * F.linear(
            F.linear(x_full, a_full), b_lora
        )

        # 1. The value must be the full reduced product, not this rank's partial.
        torch.testing.assert_close(got, expected, rtol=1e-4, atol=1e-5)

        # 2. And it must be the SAME on every rank -- a partial value that
        #    happens to be close on one rank is still a divergent residual
        #    stream, which is how this defect presented.
        buf = [torch.empty_like(got) for _ in range(WORLD)]
        dist.all_gather(buf, got.contiguous())
        delta = (buf[1] - buf[0]).abs().max().item()
        assert delta == 0.0, f"output differs across ranks by {delta:.3e}"

        queue.put((rank, "ok", {"cross_rank_delta": delta}))
    except Exception:  # surface the real failure in the parent
        import traceback

        queue.put((rank, "fail", traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


class TestRowwiseLoRAPlainInput(unittest.TestCase):
    def _run(self, bias: bool) -> None:
        ctx = mp.get_context("spawn")
        queue = ctx.Queue()
        procs = [ctx.Process(target=_body, args=(r, bias, queue)) for r in range(WORLD)]
        for p in procs:
            p.start()
        results = [queue.get(timeout=180) for _ in range(WORLD)]
        for p in procs:
            p.join(timeout=60)
        for rank, status, payload in results:
            self.assertEqual(status, "ok", f"rank {rank}:\n{payload}")

    def test_rowwise_base_with_plain_input_is_reduced(self):
        self._run(bias=False)

    def test_rowwise_bias_is_added_once_not_per_rank(self):
        """Bias must be added AFTER the reduction, or it is counted tp times."""
        self._run(bias=True)


if __name__ == "__main__":
    unittest.main()

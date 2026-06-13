# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import unittest
import weakref

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch._dynamo.backends.common import aot_autograd
from torch._functorch.partitioners import min_cut_rematerialization_partition
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.distributed._region_checkpoint import checkpoint, unit

MS = CheckpointPolicy.MUST_SAVE


def _backward_mm_count(run):
    from torch.utils._python_dispatch import TorchDispatchMode

    mm_ops = (torch.ops.aten.mm.default, torch.ops.aten.addmm.default)

    class Counter(TorchDispatchMode):
        def __init__(self):
            self.n = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func in mm_ops:
                self.n += 1
            return func(*args, **(kwargs or {}))

    c = Counter()
    with c:
        run()
    return c.n


class TestRegionCheckpoint(unittest.TestCase):
    def test_frees_input_and_param_views_with_correct_grads(self):
        # A saved nn.Linear region on an N-D input saves a reshape view of the input
        # AND weight.t() (a view of the parameter), and is the last op so its
        # backward is first. Exercises slots for arg-view + param-view + the dummy
        # trigger: the input must be freed after forward and all grads must match a
        # no-AC run.
        torch.manual_seed(0)
        dim = 16
        lin = nn.Linear(dim, dim)
        holder = {}

        def block(x):
            h = (x * 2.0).relu()  # N-D intermediate -> region input is a reshape view
            holder["ref"] = weakref.ref(h)
            return unit(lin, h, name="r").sum()

        x = torch.randn(4, 3, dim, requires_grad=True)
        block(x).backward()
        gx, gw, gb = x.grad.clone(), lin.weight.grad.clone(), lin.bias.grad.clone()
        x.grad = lin.weight.grad = lin.bias.grad = None

        out = checkpoint(block, x, policy={"r": MS})
        gc.collect()
        self.assertIsNone(holder["ref"](), "saved region pinned its input activation")

        out.backward()
        torch.testing.assert_close(x.grad, gx)
        torch.testing.assert_close(lin.weight.grad, gw)
        torch.testing.assert_close(lin.bias.grad, gb)

    def test_attention_sdpa_region(self):
        # Save wq and wo around an SDPA; grads must match a no-AC run.
        torch.manual_seed(0)
        dim = 16

        class Attn(nn.Module):
            def __init__(self):
                super().__init__()
                self.wq = nn.Linear(dim, dim, bias=False)
                self.wk = nn.Linear(dim, dim, bias=False)
                self.wv = nn.Linear(dim, dim, bias=False)
                self.wo = nn.Linear(dim, dim, bias=False)

            def forward(self, x):
                q = unit(self.wq, x, name="wq")
                k = unit(self.wk, x, name="wk")
                v = unit(self.wv, x, name="wv")
                o = F.scaled_dot_product_attention(
                    q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)
                ).squeeze(1)
                return unit(self.wo, o, name="wo")

        def block(attn, x):
            return (attn(x) + x).sum()

        attn = Attn()
        x = torch.randn(2, 4, dim, requires_grad=True)
        block(attn, x).backward()
        ref = {n: p.grad.clone() for n, p in attn.named_parameters()}
        gx = x.grad.clone()
        for p in attn.parameters():
            p.grad = None
        x.grad = None

        out = checkpoint(block, attn, x, policy={"wq": MS, "wo": MS})
        out.backward()
        torch.testing.assert_close(x.grad, gx)
        for n, p in attn.named_parameters():
            torch.testing.assert_close(p.grad, ref[n])

    def test_multiple_checkpointed_blocks(self):
        # Per-invocation state: many checkpointed blocks, each with saved regions.
        torch.manual_seed(0)
        dim = 16

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.w1 = nn.Linear(dim, dim)
                self.w2 = nn.Linear(dim, dim)
                self.norm = nn.RMSNorm(dim)

            def forward(self, x):
                h = self.norm(x)
                return x + unit(self.w2, unit(self.w1, h, name="w1").relu(), name="w2")

        blocks = nn.ModuleList([Block() for _ in range(3)])
        x = torch.randn(2, 4, dim, requires_grad=True)

        h = x
        for b in blocks:
            h = b(h)
        h.sum().backward()
        gx = x.grad.clone()
        x.grad = None

        h = x
        for b in blocks:
            h = checkpoint(b, h, policy={"w1": MS, "w2": MS})
        h.sum().backward()
        torch.testing.assert_close(x.grad, gx)

    def test_compile_saves_named_regions_and_recompiles_on_policy_change(self):
        torch.manual_seed(0)
        dim = 16
        W = torch.randn(dim, dim, requires_grad=True)

        def attn(x):
            return F.linear(x, W)

        def block(x):
            return torch.sin(torch.relu(unit(attn, x, name="r")) @ W)

        seen = []

        def bw(g, _):
            seen.append(
                [n.target for n in g.graph.nodes].count(torch.ops.aten.mm.default)
            )
            return g

        backend = aot_autograd(
            fw_compiler=lambda g, _: g,
            bw_compiler=bw,
            partition_fn=min_cut_rematerialization_partition,
        )
        torch._dynamo.reset()
        f = torch.compile(
            lambda t, p: checkpoint(block, t, policy=p), backend=backend, fullgraph=True
        )
        x = torch.randn(4, dim)
        ref = block(x.detach().clone().requires_grad_(True))
        for pol in ({"r": MS}, {"r": CheckpointPolicy.MUST_RECOMPUTE}):
            xi = x.detach().clone().requires_grad_(True)
            out = f(xi, pol)
            out.sum().backward()
            torch.testing.assert_close(out, ref)
        # The policy dict is guarded, so changing it recompiles a new bw graph.
        self.assertEqual(len(seen), 2)


if __name__ == "__main__":
    unittest.main()

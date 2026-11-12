# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import tempfile

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import run_tests, TestCase

from torchtitan.experiments.graph_trainer.programmable_fusion import (
    FusionPlan,
    op_paths,
    programmable_fusion_pass,
)

aten = torch.ops.aten

# (module_fqn, is_backward) of each call_function node of _two_layers, in order.
_OWNERS = [
    ("layers.0", False),  # add
    ("layers.0.norm", False),  # mul
    ("layers.0.norm", False),  # relu
    ("layers.1", False),
    ("layers.1.norm", False),
    ("layers.1.norm", False),
    ("layers.1.norm", True),  # mul standing in for a backward op
]


def _two_layers(x, w0, w1):
    for w in (w0, w1):
        h = aten.add.Tensor(x, 1.0)
        x = aten.relu.default(aten.mul.Tensor(h, w))
    return x, aten.mul.Tensor(x, 2.0)


def _traced():
    args = (torch.randn(8), torch.randn(8), torch.randn(8))
    gm = make_fx(_two_layers)(*args)
    nodes = [n for n in gm.graph.nodes if n.op == "call_function"]
    for node, (fqn, backward) in zip(nodes, _OWNERS, strict=True):
        node.meta["custom"] = {"module_fqn": fqn}
        node.meta["autograd_backward"] = backward
    return gm, args


def _fuse(gm, plan, cache_dir):
    return programmable_fusion_pass(gm, config=plan.build(kernel_cache_dir=cache_dir))


class TestProgrammableFusion(TestCase):
    def setUp(self):
        self.cache_dir = tempfile.mkdtemp()

    def test_loop_regions_share_one_submodule(self):
        gm, args = _traced()
        self.assertEqual(
            list(op_paths(gm).values())[4:],
            [
                "fwd.layers.1.norm.mul#0",
                "fwd.layers.1.norm.relu#0",
                "bwd.layers.1.norm.mul#0",
            ],
        )
        expected = gm(*args)
        p = FusionPlan()
        for i in range(2):
            p.fuse("norm_fwd", [f"fwd.layers.{i}.norm"])
        gm = _fuse(gm, p, self.cache_dir)
        calls = [n.target for n in gm.graph.nodes if n.op == "call_module"]
        self.assertEqual(calls, ["norm_fwd", "norm_fwd"])
        self.assertEqual(gm(*args), expected)

    def test_exclude(self):
        gm, _ = _traced()
        p = FusionPlan()
        p.fuse("mul_fwd", ["fwd.layers.1.norm"], exclude=["fwd.layers.1.norm.relu#0"])
        gm = _fuse(gm, p, self.cache_dir)
        (call,) = gm.graph.find_nodes(op="call_module")
        self.assertEqual(
            [n.target for n in gm.mul_fwd.graph.nodes if n.op == "call_function"],
            [aten.mul.Tensor],
        )
        self.assertEqual(next(iter(call.users)).target, aten.relu.default)

    def test_explicit_errors(self):
        cases = [
            ("closest: \\['fwd.layers.1.norm'", [("x", ["fwd.layers.1.nrom"], [])]),
            ("removes nothing", [("x", ["fwd.layers.1.norm"], ["fwd.layers.0.norm"])]),
            (
                "mixes forward and backward",
                [("x", ["fwd.layers.1.norm", "bwd.layers.1"], [])],
            ),
            (
                "overlaps",
                [("a", ["fwd.layers.0"], []), ("b", ["fwd.layers.0.norm"], [])],
            ),
        ]
        for message, regions in cases:
            gm, _ = _traced()
            p = FusionPlan()
            for name, targets, exclude in regions:
                p.fuse(name, targets, exclude=exclude)
            with self.assertRaisesRegex(ValueError, message):
                _fuse(gm, p, self.cache_dir)


if __name__ == "__main__":
    run_tests()

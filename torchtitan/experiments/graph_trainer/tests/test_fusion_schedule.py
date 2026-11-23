# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch import fx

from torchtitan.experiments.graph_trainer.auto_fuser.schedule import (
    IndexedGraph,
    SchedulePlan,
)


def test_interleaved_alias_read_observes_write():
    graph = fx.Graph()
    x = graph.placeholder("x")
    view = graph.call_function(torch.ops.aten.view.default, (x, [-1]))
    write = graph.call_function(torch.ops.aten.add_.Scalar, (x, 1))
    read = graph.call_function(torch.ops.aten.clone.default, (view,))
    negate = graph.call_function(torch.ops.aten.neg.default, (write,))
    graph.output((read, negate))
    gm = fx.GraphModule({}, graph)
    expected = gm(torch.tensor([2.0]))
    plan = SchedulePlan(IndexedGraph(gm))
    plan.fuse("write_then_negate", [plan.indexed_graph.fwd.add_, negate])
    plan.apply()
    actual = gm(torch.tensor([2.0]))
    assert all(torch.equal(want, got) for want, got in zip(expected, actual))


def test_dependency_cannot_leave_and_reenter_a_region():
    graph = fx.Graph()
    x = graph.placeholder("x")
    first = graph.call_function(torch.ops.aten.neg.default, (x,))
    outside = graph.call_function(torch.ops.aten.cos.default, (first,))
    last = graph.call_function(torch.ops.aten.add.Tensor, (first, outside))
    graph.output(last)
    gm = fx.GraphModule({}, graph)
    plan = SchedulePlan(IndexedGraph(gm))
    plan.fuse("cycle", [first, last])
    with pytest.raises(ValueError, match="dependency leaves the region"):
        plan.apply()


def test_implicit_view_is_claimed_by_its_region():
    graph = fx.Graph()
    x = graph.placeholder("x")
    start = graph.call_function(torch.ops.aten.neg.default, (x,))
    view = graph.call_function(torch.ops.aten.reshape.default, (start, [-1]))
    end = graph.call_function(torch.ops.aten.cos.default, (view,))
    for node in (start, view, end):
        node.meta.update(custom={"module_fqn": "layers.0"}, autograd_backward=True)
    graph.output(end)
    plan = SchedulePlan(IndexedGraph(fx.GraphModule({}, graph)))
    layer = plan.indexed_graph.bwd.layers[0]
    plan.fuse("through_view", [layer.neg[0], layer.cos[:1]])
    with pytest.raises(ValueError, match="already in another region"):
        plan.fuse("overlapping", [layer.reshape])


def test_changed_literal_cannot_hit_a_verified_kernel(tmp_path, monkeypatch):
    import json

    from torchtitan.experiments.graph_trainer.auto_fuser.scheduled_kernels import (
        KernelSource,
        substitute_with_kernels,
    )

    graph = fx.Graph()
    x = graph.placeholder("x")
    x.meta["val"] = torch.ones(2)
    add = graph.call_function(torch.ops.aten.add.Scalar, (x, 2))
    add.meta["val"] = x.meta["val"] + 2
    graph.output(add)
    gm = fx.GraphModule({}, graph)
    plan = SchedulePlan(IndexedGraph(gm))
    plan.fuse("r000_add", [add])
    calls = plan.apply()
    key = "pattern_0/spec"
    directory = tmp_path / "kernels" / key
    directory.mkdir(parents=True)
    (directory / "problem.py").write_text(
        "def reference(in0):\n"
        "    v1 = torch.ops.aten.add.Scalar(in0, 1);  in0 = None\n"
        "    return (v1,)\n"
    )
    tensor = {"shape": [2], "stride": [1], "dtype": "float32", "device": "cpu"}
    (directory / "spec.json").write_text(
        json.dumps(
            {
                "env": {
                    "gpu": "cpu",
                    "deterministic": torch.are_deterministic_algorithms_enabled(),
                    "fill_uninitialized_memory": torch.utils.deterministic.fill_uninitialized_memory,
                },
                "inputs": [tensor],
                "outputs": [tensor],
            }
        )
    )
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda: "cpu")
    manifest = {"problems": {key: {"calls": 1, "region": "r000"}}}
    kernels = {key: KernelSource(path=directory / "kernel.py", sha256="unused")}
    with pytest.raises(ValueError, match="verified kernel calls differ"):
        substitute_with_kernels(
            gm, calls, problem_dir=tmp_path, manifest=manifest, kernels=kernels
        )

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.experiments.graph_trainer.cpu_offload import (
    apply_cpu_offload_pass,
    schedule_activation_reloads_pass,
)
from torchtitan.experiments.graph_trainer.fsdp_passes import (
    joint_transformer_block_bucketing_reordering_pass,
)
from torchtitan.experiments.graph_trainer.selective_activation_remat import (
    selective_activation_remat_pass,
)


@pytest.mark.parametrize("lookahead", [0, 1])
def test_final_reload_schedule_after_remat_and_bucketing(lookahead):
    graph = torch.fx.Graph()
    with FakeTensorMode():
        val = torch.empty((64, 64), device="cuda")
    x = graph.placeholder("x")
    y = graph.placeholder("y")
    x.meta["val"] = y.meta["val"] = val

    def op(target, args, scope, *, backward=False, policy=None):
        node = graph.call_function(target, args)
        node.meta.update(
            val=val,
            custom={"module_fqn": scope},
            autograd_backward=backward,
        )
        if policy is not None:
            node.meta["recompute"] = policy
        return node

    saved = op(
        torch.ops.aten.clone.default,
        (x,),
        "blocks.alpha",
        policy=CheckpointPolicy.MUST_CPU_OFFLOAD,
    )
    remat = op(
        torch.ops.aten.add.Tensor,
        (saved, y),
        "blocks.beta",
        policy=CheckpointPolicy.MUST_RECOMPUTE,
    )
    unpack = op(torch.ops.aten.clone.default, (y,), "blocks.gamma", backward=True)
    compute = op(torch.ops.aten.mm.default, (x, unpack), "blocks.gamma", backward=True)
    consumer = op(
        torch.ops.aten.mm.default, (remat, compute), "blocks.beta", backward=True
    )
    last = op(
        torch.ops.aten.mm.default, (saved, consumer), "blocks.alpha", backward=True
    )
    graph.output(last)
    gm = torch.fx.GraphModule(torch.nn.Module(), graph)
    buckets = ["blocks.alpha", "blocks.beta", "blocks.gamma"]
    apply_cpu_offload_pass(gm, prefetch_lookahead=0)
    selective_activation_remat_pass(gm)
    joint_transformer_block_bucketing_reordering_pass(
        gm, (), module_bucket_plans=buckets
    )
    reload = next(n for n in gm.graph.nodes if n.target == torch.ops.ao.reload.default)
    wait = next(iter(reload.users))
    forward_wait = wait.args[1]
    schedule_activation_reloads_pass(
        gm, prefetch_lookahead=lookahead, module_bucket_plans=buckets
    )
    positions = {node: i for i, node in enumerate(gm.graph.nodes)}
    first_user = min(wait.users, key=positions.__getitem__)
    assert first_user.meta["custom"]["module_fqn"] == "blocks.beta"
    assert wait.next is first_user
    assert wait.args == (reload, forward_wait)
    assert reload.args[0] is forward_wait.args[0]
    assert saved not in first_user.all_input_nodes
    assert all(
        positions[n] < positions[forward_wait] for n in forward_wait.all_input_nodes
    )
    if lookahead:
        assert (
            positions[unpack] < positions[reload] < positions[compute] < positions[wait]
        )
    else:
        assert reload.next is wait
    gm.graph.lint()

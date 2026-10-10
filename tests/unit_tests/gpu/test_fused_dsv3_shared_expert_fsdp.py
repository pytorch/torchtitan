# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FakePG proves collective/lifetime behavior; nonzero tests prove numerics."""

import subprocess
import sys

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves


pytest.importorskip("torchao")

from torchtitan.distributed.fsdp import linear_param_shard_placements  # noqa: E402
from torchtitan_recipes.overrides.fused_dsv3_shared_expert import (  # noqa: E402
    fused_dsv3_shared_expert,
)

from tests.unit_tests.gpu.dsv3_shared_expert_utils import (  # noqa: E402
    assert_bits_equal,
    shared_expert_config,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="GB300 shared-expert specialization",
)


class _ZeroCollectives(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.collectives = []
        self.fusions = []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        name = str(function)
        result = function(*args, **(kwargs or {}))
        if "dsv3_shared_expert" in name:
            self.fusions.append(name)
        if name.startswith("c10d.") and (
            "allgather" in name.replace("_", "")
            or "reducescatter" in name.replace("_", "")
        ):
            self.collectives.append(
                (
                    name,
                    [
                        tuple(t.shape)
                        for t in tree_leaves(args)
                        if isinstance(t, torch.Tensor)
                    ],
                )
            )
            # FakePG carries no remote data. Define gathered/reduced storage so
            # allocator contents cannot masquerade as numerical evidence.
            for tensor in tree_leaves(result):
                if isinstance(tensor, torch.Tensor):
                    tensor.zero_()
        return result


def _exercise(rank, grouping):
    dist.init_process_group("fake", store=FakeStore(), rank=rank, world_size=128)
    try:
        mesh = init_device_mesh("cuda", (128,), mesh_dim_names=("dp_shard",))
        models = []
        for fused in (False, True):
            config = shared_expert_config()
            if fused:
                config = fused_dsv3_shared_expert(
                    config, forward_quant=True, backward_quant=True
                )
            with torch.device("meta"):
                model = config.build()
            options = {
                "mesh": mesh,
                "mp_policy": MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.float32
                ),
                "reshard_after_forward": True,
            }
            placements = linear_param_shard_placements(model)
            if grouping != "root":
                fully_shard(model.w2, **options)
            if grouping == "both_children":
                fully_shard(model.w13, shard_placement_fn=placements.get, **options)
            fully_shard(model, shard_placement_fn=placements.get, **options)
            model.to_empty(device="cuda")
            model.init_states(buffer_device=torch.device("cuda"))
            models.append(model)
        torch.manual_seed(42)
        x = torch.randn(1, 4096, 7168, device="cuda", dtype=torch.bfloat16)
        inputs = [x.clone().requires_grad_() for _ in models]
        records = [_ZeroCollectives(), _ZeroCollectives()]
        for synchronize in (True, False, True):
            dy = torch.randn_like(x)
            results = []
            for model, input, record in zip(models, inputs, records):
                model.set_requires_gradient_sync(synchronize)
                with record:
                    output = model(input)
                    output.backward(dy)
                gradients = [
                    parameter.grad.to_local() if parameter.grad is not None else None
                    for parameter in model.parameters()
                ]
                results.append((output.detach(), input.grad, *gradients))
            for expected, actual in zip(*results):
                assert_bits_equal(expected, actual)
            if synchronize:
                for model in models:
                    assert all(
                        parameter.grad is not None for parameter in model.parameters()
                    )
                    model.zero_grad(set_to_none=True)
        assert records[0].collectives == records[1].collectives, (
            records[0].collectives,
            records[1].collectives,
        )
        assert any(
            "allgather" in name.replace("_", "") for name, _ in records[1].collectives
        )
        assert any(
            "reducescatter" in name.replace("_", "")
            for name, _ in records[1].collectives
        )
        if grouping == "both_children":
            assert records[1].fusions == []
        else:
            assert (
                records[1].fusions.count(
                    "torchtitan.dsv3_shared_expert_forward.default"
                )
                == 3
            )
            assert (
                records[1].fusions.count(
                    "torchtitan.dsv3_shared_expert_backward.default"
                )
                == 3
            )
        state = models[0].state_dict()
        models[1].load_state_dict(state)
        assert state["w13.weight"].shape == (2, 2048, 7168)
        assert state["w2.weight"].shape == (7168, 2048)
        torch.cuda.synchronize()
        print(
            {
                "rank": rank,
                "grouping": grouping,
                "collectives": records[1].collectives,
                "fusions": records[1].fusions,
            },
            flush=True,
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("rank", [0, 127])
@pytest.mark.parametrize("grouping", ["root", "w2_child", "both_children"])
def test_fsdp128_weight_refill_collective_order_and_native_child_fallback(
    rank, grouping
):
    result = subprocess.run(
        [sys.executable, __file__, str(rank), grouping],
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _exercise(int(sys.argv[1]), sys.argv[2])

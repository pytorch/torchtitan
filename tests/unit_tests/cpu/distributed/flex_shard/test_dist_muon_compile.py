# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import distribute_tensor, Replicate
from torchtitan.distributed.flex_shard import (
    BucketConfig,
    ComputeLayout,
    DistMuon,
    Owned,
)


@pytest.mark.parametrize("shape", [(8, 12), (2, 12, 8)])
def test_compiled_dist_muon_preserves_updates_and_state(tmp_path, shape) -> None:
    """Guard in-place updates, changing gradients/lr, and checkpoint restore."""
    dist.init_process_group(
        "gloo", init_method=(tmp_path / "store").as_uri(), rank=0, world_size=1
    )
    try:
        mesh = init_device_mesh("cpu", (1,), mesh_dim_names=("dp_shard",))
        generator = torch.Generator().manual_seed(42)
        initial = torch.randn(shape, generator=generator)
        parameters, optimizers = [], []
        for _ in range(2):
            parameter = torch.nn.Parameter(
                distribute_tensor(initial.clone(), mesh, (Replicate(),))
            )
            optimizer = DistMuon(
                [{"params": [parameter], "param_names": ["weight"]}],
                compute_sharding_by_fqn={
                    "weight": ComputeLayout(
                        {"dp_shard": Owned()}, compute_dtype=torch.bfloat16
                    )
                },
                bucket_configs=[BucketConfig(patterns=("weight",))],
                ns_steps=2,
            )
            parameters.append(parameter)
            optimizers.append(optimizer)
        optimizers[1].compile()
        for step in range(10):
            gradient = torch.randn(shape, generator=generator)
            for parameter, optimizer in zip(parameters, optimizers, strict=True):
                parameter.grad = distribute_tensor(
                    gradient.clone(), mesh, (Replicate(),)
                )
                optimizer.param_groups[0]["lr"] = 1e-3 / (step + 1)
                optimizer.step()
            assert torch.equal(parameters[0].to_local(), parameters[1].to_local())
            assert torch.equal(
                optimizers[0].state[parameters[0]]["momentum_buffer"].to_local(),
                optimizers[1].state[parameters[1]]["momentum_buffer"].to_local(),
            )
            if step == 4:
                optimizers[1].load_state_dict(optimizers[0].state_dict())
    finally:
        dist.destroy_process_group()

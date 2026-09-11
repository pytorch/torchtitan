# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed.tensor import DTensor
from torchtitan.components.optimizer import OptimizersContainer, ParamGroupConfig


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA fused Adam"
)


@pytest.fixture
def mesh(tmp_path):
    torch.cuda.set_device(0)
    dist.init_process_group(
        "nccl", init_method=f"file://{tmp_path / 'pg'}", rank=0, world_size=1
    )
    try:
        yield init_device_mesh("cuda", (1,))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("optimizer_name", ["Adam", "AdamW"])
@pytest.mark.parametrize("amsgrad", [False, True])
@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("param_dtype", [torch.float32, torch.bfloat16])
def test_bf16_checkpoint_matches_uninterrupted_training(
    optimizer_name, amsgrad, sharded, param_dtype, mesh, tmp_path
):
    def build():
        model = torch.nn.Linear(64, 64, bias=False, device="cuda", dtype=param_dtype)
        with torch.no_grad():
            model.weight.fill_(0.5)
        if sharded:
            fully_shard(model, mesh=mesh)
        container = OptimizersContainer.Config(
            implementation="fused_opt_states_bf16",
            param_groups=[
                ParamGroupConfig(
                    pattern=r".*",
                    optimizer_name=optimizer_name,
                    optimizer_kwargs={
                        "lr": 0.01,
                        "betas": (0.9, 0.95),
                        "weight_decay": 0.1,
                        "amsgrad": amsgrad,
                    },
                )
            ],
        ).build(model_parts=[model])
        return model, container

    def step(model, container, index):
        for param in model.parameters():
            param.grad = torch.full_like(param, (index + 1) / 7)
        container.step()
        container.zero_grad()

    def check_state(container, expected_step):
        for optimizer in container.optimizers:
            for group in optimizer.param_groups:
                for param in group["params"]:
                    state = optimizer.state[param]
                    assert state["step"].dtype == torch.float32
                    assert state["step"].device == param.device
                    assert state["step"].item() == expected_step
                    assert ("max_exp_avg_sq" in state) == amsgrad
                    for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
                        if key not in state:
                            continue
                        value = state[key]
                        assert value.dtype == torch.bfloat16
                        assert value.device == param.device
                        assert isinstance(value, DTensor) == sharded
                        if sharded:
                            assert value.placements == param.placements
                            assert value.device_mesh == param.device_mesh

    def assert_equal(actual, expected):
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            other = actual[key]
            if isinstance(value, DTensor):
                assert other.placements == value.placements
                value, other = value.to_local(), other.to_local()
            if isinstance(value, torch.Tensor):
                torch.testing.assert_close(other, value, rtol=0, atol=0)
            else:
                assert other == value

    model, source = build()
    for index in range(3):
        step(model, source, index)
    check_state(source, 3)
    checkpoint = copy.deepcopy(source.state_dict())
    dcp.save({"optimizer": source}, checkpoint_id=tmp_path / "checkpoint")

    resumed_model, resumed = build()
    resumed_model.load_state_dict(copy.deepcopy(model.state_dict()))
    resumed.state_dict()
    check_state(resumed, 0)
    dcp.load({"optimizer": resumed}, checkpoint_id=tmp_path / "checkpoint")
    check_state(resumed, 3)
    assert_equal(resumed.state_dict(), checkpoint)

    # The second update also catches divergence from storing the first resumed
    # update's moments in fp32 instead of rounding them back to bf16.
    for index in (3, 4):
        step(model, source, index)
        step(resumed_model, resumed, index)
        check_state(resumed, index + 1)
        assert_equal(resumed.state_dict(), source.state_dict())
        assert_equal(resumed_model.state_dict(), model.state_dict())

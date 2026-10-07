# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.elastic.utils.distributed import get_free_port
from torch.distributed.fsdp import fully_shard
from torch.testing._internal.distributed.fake_pg import FakeStore

from torchtitan.distributed.cuda_graph import cuda_graph_teardown, wrap_with_cuda_graph


def test_group_graph_replay_accumulates_existing_gradients() -> None:
    parameter = torch.nn.Parameter(torch.randn(1024, device="cuda"))

    def forward_backward(group: torch.Tensor) -> torch.Tensor:
        loss = (parameter * group).sum()
        loss.backward()
        return loss.detach()

    run_group = wrap_with_cuda_graph(
        forward_backward,
        num_warmup_iterations=2,
    )
    try:
        for step, count in enumerate((4, 1, 5, 2, 3)):
            if parameter.grad is not None:
                parameter.grad.zero_()
            values = [step + group_index + 1 for group_index in range(count)]
            groups = [torch.full_like(parameter, value) for value in values]
            losses = [run_group(group).clone() for group in groups]
            torch.cuda.synchronize()

            torch.testing.assert_close(
                parameter.grad,
                torch.full_like(parameter, sum(values)),
            )
            expected_losses = torch.stack(
                [
                    (parameter.detach() * torch.full_like(parameter, value)).sum()
                    for value in values
                ]
            )
            torch.testing.assert_close(torch.stack(losses), expected_losses)

        with pytest.raises(ValueError, match="same shape"):
            run_group(torch.ones(1, device="cuda"))
    finally:
        cuda_graph_teardown()


def test_group_graph_replay_accumulates_reduced_fsdp_gradients() -> None:
    dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=1)
    mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("dp_shard",))
    model = torch.nn.Linear(8, 8, bias=False, device="cuda")
    fully_shard(model, mesh=mesh)
    parameter = next(model.parameters())

    def forward_backward(group: torch.Tensor) -> torch.Tensor:
        loss = model(group).sum()
        loss.backward()
        return loss.detach()

    run_group = wrap_with_cuda_graph(
        forward_backward,
        num_warmup_iterations=2,
    )
    try:
        for step, count in enumerate((4, 1, 3, 2)):
            model.zero_grad(set_to_none=False)
            values = [step + group_index + 1 for group_index in range(count)]
            for value in values:
                group = torch.full((4, 8), float(value), device="cuda")
                run_group(group)
            torch.cuda.synchronize()

            assert parameter.grad is not None
            gradient = parameter.grad.to_local()
            torch.testing.assert_close(
                gradient,
                torch.full_like(gradient, 4 * sum(values)),
            )
    finally:
        cuda_graph_teardown()
        dist.destroy_process_group()


def _run_real_fsdp_group_graph_numerics(
    rank: int,
    world_size: int,
    port: int,
) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh(
            "cuda",
            (world_size,),
            mesh_dim_names=("dp_shard",),
        )
        graph_model = torch.nn.Linear(8, 8, bias=False, device="cuda")
        eager_model = torch.nn.Linear(8, 8, bias=False, device="cuda")
        eager_model.load_state_dict(graph_model.state_dict())
        fully_shard(graph_model, mesh=mesh)
        fully_shard(eager_model, mesh=mesh)
        graph_parameter = next(graph_model.parameters())
        eager_parameter = next(eager_model.parameters())
        graph_optimizer = torch.optim.SGD(graph_model.parameters(), lr=0.25)
        eager_optimizer = torch.optim.SGD(eager_model.parameters(), lr=0.25)

        def graph_forward_backward(group: torch.Tensor) -> torch.Tensor:
            loss = graph_model(group).sum()
            loss.backward()
            return loss.detach()

        run_graph_group = wrap_with_cuda_graph(
            graph_forward_backward,
            num_warmup_iterations=2,
        )
        for step, count in enumerate((4, 1, 3, 2)):
            graph_optimizer.zero_grad(set_to_none=False)
            eager_optimizer.zero_grad(set_to_none=False)
            graph_groups = [
                torch.full(
                    (4, 8),
                    float(step + group_index + rank + 1),
                    device="cuda",
                )
                for group_index in range(count)
            ]
            eager_groups = [group.clone() for group in graph_groups]
            for group in graph_groups:
                run_graph_group(group)
            for group in eager_groups:
                eager_loss = eager_model(group).sum()
                eager_loss.backward()
            torch.cuda.synchronize()

            assert graph_parameter.grad is not None
            assert eager_parameter.grad is not None
            graph_gradient = graph_parameter.grad.to_local()
            eager_gradient = eager_parameter.grad.to_local()
            expected_value = 4 * sum(
                step + group_index + (world_size - 1) / 2 + 1
                for group_index in range(count)
            )
            expected_gradient = torch.full_like(eager_gradient, expected_value)
            torch.testing.assert_close(
                eager_gradient,
                expected_gradient,
                rtol=0,
                atol=0,
            )
            torch.testing.assert_close(
                graph_gradient,
                expected_gradient,
                rtol=0,
                atol=0,
            )

            graph_optimizer.step()
            eager_optimizer.step()
            torch.testing.assert_close(
                graph_parameter.to_local(),
                eager_parameter.to_local(),
                rtol=0,
                atol=0,
            )
    finally:
        cuda_graph_teardown()
        dist.destroy_process_group()


@pytest.mark.multi_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
def test_group_graph_replay_matches_real_fsdp_numerics() -> None:
    mp.spawn(
        _run_real_fsdp_group_graph_numerics,
        args=(2, get_free_port()),
        nprocs=2,
        join=True,
    )

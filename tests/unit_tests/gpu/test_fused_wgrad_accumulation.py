# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from typing import Any, Literal

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.elastic.utils.distributed import get_free_port
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.distributed.pipelining import PipelineStage, Schedule1F1B
from torch.distributed.tensor import DTensor

from torchtitan.components.fused_wgrad import FusedWGradAccumLinear
from torchtitan.models.common.linear import Linear


def _build_linear(
    in_features: int,
    out_features: int,
    *,
    fused_wgrad: bool,
    bias: bool = False,
    wgrad_accum_dtype: Literal["bfloat16", "float32"] = "float32",
) -> Linear:
    if fused_wgrad:
        return FusedWGradAccumLinear.Config(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            wgrad_accum_dtype=wgrad_accum_dtype,
        ).build()
    return Linear.Config(
        in_features=in_features,
        out_features=out_features,
        bias=bias,
    ).build()


def _run_fsdp_backward_contributions(
    linear: Any,
    inputs: list[torch.Tensor],
    grad_outputs: list[torch.Tensor],
    *,
    expected_accumulation_dtype: torch.dtype | None = None,
) -> None:
    last_idx = len(inputs) - 1
    linear.set_is_last_backward(False)
    linear.set_reshard_after_backward(False)
    linear.set_requires_gradient_sync(False)

    for idx, (input_BK, grad_output_BN) in enumerate(
        zip(inputs, grad_outputs, strict=True)
    ):
        if idx == last_idx:
            linear.set_is_last_backward(True)
            linear.set_reshard_after_backward(True)
            linear.set_requires_gradient_sync(True)
        linear(input_BK).backward(grad_output_BN)
        if expected_accumulation_dtype is not None and idx != last_idx:
            assert linear.weight.grad is not None
            assert linear.weight.grad.dtype == expected_accumulation_dtype


def _operator_count(profile: torch.profiler.profile, operator: str) -> int:
    return sum(event.count for event in profile.key_averages() if event.key == operator)


def _accumulate_fp32_wgrad(
    grad_weight_NK: torch.Tensor | None,
    grad_output_BN: torch.Tensor,
    input_BK: torch.Tensor,
) -> torch.Tensor:
    if grad_weight_NK is None:
        return torch.mm(
            grad_output_BN.t(),
            input_BK,
            out_dtype=torch.float32,
        )
    torch.addmm(
        grad_weight_NK,
        grad_output_BN.t(),
        input_BK,
        out=grad_weight_NK,
        out_dtype=torch.float32,
    )
    return grad_weight_NK


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_wgrad_accumulates_in_parameter_grad():
    torch.manual_seed(42)
    num_contributions = 4
    in_features = 32
    out_features = 48

    baseline = (
        _build_linear(
            in_features,
            out_features,
            fused_wgrad=False,
            bias=True,
        )
        .cuda()
        .bfloat16()
    )
    fused = (
        _build_linear(
            in_features,
            out_features,
            fused_wgrad=True,
            bias=True,
        )
        .cuda()
        .bfloat16()
    )
    with torch.no_grad():
        fused.weight.copy_(baseline.weight)
        fused.bias.copy_(baseline.bias)

    generator = torch.Generator(device="cuda").manual_seed(1000)
    weight_grad_pointer: int | None = None
    for step in range(2):
        baseline.zero_grad(set_to_none=step == 0)
        fused.zero_grad(set_to_none=step == 0)
        expected_weight_grad_NK: torch.Tensor | None = None
        if step != 0:
            expected_weight_grad_NK = torch.zeros_like(
                fused.weight,
                dtype=torch.float32,
            )
        inputs = [
            torch.randn(
                16,
                in_features,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            for _ in range(num_contributions)
        ]
        grad_outputs = [
            torch.randn(
                16,
                out_features,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
            for _ in range(num_contributions)
        ]
        baseline_inputs = [value.detach().clone().requires_grad_() for value in inputs]
        fused_inputs = [value.detach().clone().requires_grad_() for value in inputs]

        for baseline_input, fused_input, grad_output_BN in zip(
            baseline_inputs,
            fused_inputs,
            grad_outputs,
            strict=True,
        ):
            baseline(baseline_input).backward(grad_output_BN)
            fused(fused_input).backward(grad_output_BN)
            expected_weight_grad_NK = _accumulate_fp32_wgrad(
                expected_weight_grad_NK,
                grad_output_BN,
                fused_input,
            )

            assert baseline.weight.grad is not None
            assert fused.weight.grad is not None
            assert fused.weight.grad.dtype == torch.float32
            if weight_grad_pointer is None:
                weight_grad_pointer = fused.weight.grad.data_ptr()
            assert fused.weight.grad.data_ptr() == weight_grad_pointer
            torch.testing.assert_close(
                fused.weight.grad,
                expected_weight_grad_NK,
            )

        assert baseline.bias.grad is not None
        assert fused.bias.grad is not None
        torch.testing.assert_close(fused.bias.grad, baseline.bias.grad)
        for fused_input, baseline_input in zip(
            fused_inputs, baseline_inputs, strict=True
        ):
            torch.testing.assert_close(fused_input.grad, baseline_input.grad)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_wgrad_rejects_noncontiguous_parameter_grad():
    linear = _build_linear(32, 48, fused_wgrad=True).cuda().bfloat16()
    linear.weight.grad_dtype = torch.float32
    linear.weight.grad = torch.empty(
        32,
        48,
        device="cuda",
        dtype=torch.float32,
    ).t()

    input_BK = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="contiguous"):
        linear(input_BK).sum().backward()


@pytest.mark.parametrize(
    ("wgrad_accum_dtype", "expected_dtype"),
    [("bfloat16", torch.bfloat16), ("float32", torch.float32)],
)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_wgrad_uses_configured_accumulation_dtype(
    wgrad_accum_dtype: Literal["bfloat16", "float32"],
    expected_dtype: torch.dtype,
):
    linear = FusedWGradAccumLinear.Config(
        in_features=32,
        out_features=48,
        wgrad_accum_dtype=wgrad_accum_dtype,
    ).build()
    linear = linear.cuda().bfloat16()
    input_BK = torch.randn(16, 32, device="cuda", dtype=torch.bfloat16)

    linear(input_BK).sum().backward()
    linear(input_BK).sum().backward()

    assert linear.weight.grad is not None
    assert linear.weight.grad.dtype == expected_dtype


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fused_wgrad_cuda_graph_reuses_parameter_grad():
    torch.manual_seed(42)
    num_contributions = 4
    in_features = 32
    out_features = 48

    baseline = (
        _build_linear(
            in_features,
            out_features,
            fused_wgrad=False,
        )
        .cuda()
        .bfloat16()
    )
    fused = (
        _build_linear(
            in_features,
            out_features,
            fused_wgrad=True,
        )
        .cuda()
        .bfloat16()
    )
    with torch.no_grad():
        fused.weight.copy_(baseline.weight)

    inputs = [
        torch.randn(16, in_features, device="cuda", dtype=torch.bfloat16)
        for _ in range(num_contributions)
    ]
    grad_outputs = [
        torch.randn(16, out_features, device="cuda", dtype=torch.bfloat16)
        for _ in range(num_contributions)
    ]

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        for input_BK, grad_output_BN in zip(inputs, grad_outputs, strict=True):
            fused(input_BK).backward(grad_output_BN)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    fused.zero_grad(set_to_none=True)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for input_BK, grad_output_BN in zip(inputs, grad_outputs, strict=True):
            fused(input_BK).backward(grad_output_BN)

    fused.zero_grad(set_to_none=False)
    graph.replay()
    torch.cuda.synchronize()

    for input_BK, grad_output_BN in zip(inputs, grad_outputs, strict=True):
        baseline(input_BK).backward(grad_output_BN)

    expected_weight_grad_NK = None
    for input_BK, grad_output_BN in zip(inputs, grad_outputs, strict=True):
        expected_weight_grad_NK = _accumulate_fp32_wgrad(
            expected_weight_grad_NK,
            grad_output_BN,
            input_BK,
        )

    assert fused.weight.grad is not None
    weight_grad_pointer = fused.weight.grad.data_ptr()
    torch.testing.assert_close(
        fused.weight.grad,
        expected_weight_grad_NK,
    )

    for _ in range(2):
        fused.zero_grad(set_to_none=False)
        graph.replay()
        torch.cuda.synchronize()
        assert fused.weight.grad is not None
        assert fused.weight.grad.data_ptr() == weight_grad_pointer
        torch.testing.assert_close(
            fused.weight.grad,
            expected_weight_grad_NK,
        )


def _run_fsdp_fused_wgrad(
    rank: int,
    world_size: int,
    port: int,
) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    try:
        torch.manual_seed(42)
        num_contributions = 4
        in_features = 32
        out_features = 48
        mesh = init_device_mesh(
            "cuda",
            (world_size,),
            mesh_dim_names=("dp_shard",),
        )

        baseline = _build_linear(
            in_features,
            out_features,
            fused_wgrad=False,
            bias=True,
        ).cuda()
        fused = _build_linear(
            in_features,
            out_features,
            fused_wgrad=True,
            bias=True,
        ).cuda()
        with torch.no_grad():
            fused.weight.copy_(baseline.weight)
            fused.bias.copy_(baseline.bias)

        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        fully_shard(
            baseline,
            mesh=mesh,
            mp_policy=mp_policy,
            reshard_after_forward=False,
        )
        fully_shard(
            fused,
            mesh=mesh,
            mp_policy=mp_policy,
            reshard_after_forward=False,
        )

        generator = torch.Generator(device="cuda").manual_seed(2000 + rank)
        for step in range(2):
            baseline.zero_grad(set_to_none=True)
            fused.zero_grad(set_to_none=True)
            inputs = [
                torch.randn(
                    16,
                    in_features,
                    device=device,
                    dtype=torch.bfloat16,
                    generator=generator,
                )
                for _ in range(num_contributions)
            ]
            grad_outputs = [
                torch.randn(
                    16,
                    out_features,
                    device=device,
                    dtype=torch.bfloat16,
                    generator=generator,
                )
                for _ in range(num_contributions)
            ]
            baseline_inputs = [
                value.detach().clone().requires_grad_() for value in inputs
            ]
            fused_inputs = [value.detach().clone().requires_grad_() for value in inputs]

            _run_fsdp_backward_contributions(
                baseline,
                baseline_inputs,
                grad_outputs,
            )
            if step == 0:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU]
                ) as profile:
                    _run_fsdp_backward_contributions(
                        fused,
                        fused_inputs,
                        grad_outputs,
                        expected_accumulation_dtype=torch.float32,
                    )
                assert _operator_count(profile, "aten::addmm") == (
                    num_contributions - 1
                )
            else:
                _run_fsdp_backward_contributions(
                    fused,
                    fused_inputs,
                    grad_outputs,
                    expected_accumulation_dtype=torch.float32,
                )

            expected_weight_grad_NK: torch.Tensor | None = None
            for input_BK, grad_output_BN in zip(inputs, grad_outputs, strict=True):
                expected_weight_grad_NK = _accumulate_fp32_wgrad(
                    expected_weight_grad_NK,
                    grad_output_BN,
                    input_BK,
                )
            assert expected_weight_grad_NK is not None
            dist.all_reduce(
                expected_weight_grad_NK,
                op=dist.ReduceOp.AVG,
                group=mesh.get_group(),
            )
            expected_local_weight_grad_NK = expected_weight_grad_NK.chunk(
                world_size,
                dim=0,
            )[rank].float()

            assert isinstance(baseline.weight.grad, DTensor)
            assert isinstance(fused.weight.grad, DTensor)
            assert isinstance(baseline.bias.grad, DTensor)
            assert isinstance(fused.bias.grad, DTensor)
            assert fused.weight.grad.dtype == torch.float32
            torch.testing.assert_close(
                fused.weight.grad.to_local(),
                expected_local_weight_grad_NK,
            )
            torch.testing.assert_close(
                fused.bias.grad.to_local(),
                baseline.bias.grad.to_local(),
            )
            for fused_input, baseline_input in zip(
                fused_inputs, baseline_inputs, strict=True
            ):
                torch.testing.assert_close(fused_input.grad, baseline_input.grad)
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
def test_fused_wgrad_with_fsdp():
    mp.spawn(
        _run_fsdp_fused_wgrad,
        args=(2, get_free_port()),
        nprocs=2,
        join=True,
    )


def _build_pipeline_linear(pp_rank: int, *, fused_wgrad: bool) -> Linear:
    in_features = 32 if pp_rank == 0 else 48
    out_features = 48 if pp_rank == 0 else 24
    return _build_linear(
        in_features,
        out_features,
        fused_wgrad=fused_wgrad,
    ).cuda()


def _run_pipeline_step(
    schedule: Schedule1F1B,
    pp_rank: int,
    input_BL: torch.Tensor,
    target_BL: torch.Tensor,
) -> list[torch.Tensor]:
    losses: list[torch.Tensor] = []
    if pp_rank == 0:
        schedule.step(input_BL)
    else:
        schedule.step(target=target_BL, losses=losses)
    return losses


def _run_pipeline_fsdp_fused_wgrad(
    rank: int,
    world_size: int,
    port: int,
) -> None:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    device = torch.device("cuda", rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )
    try:
        dp_degree = 2
        pp_degree = 2
        num_microbatches = 4
        mesh = init_device_mesh(
            "cuda",
            (dp_degree, pp_degree),
            mesh_dim_names=("dp_shard", "pp"),
        )
        dp_mesh = mesh["dp_shard"]
        pp_mesh = mesh["pp"]
        pp_rank = pp_mesh.get_local_rank()
        dp_rank = dp_mesh.get_local_rank()

        baseline = _build_pipeline_linear(pp_rank, fused_wgrad=False)
        fused = _build_pipeline_linear(pp_rank, fused_wgrad=True)

        weight_generator = torch.Generator(device="cuda").manual_seed(3000 + pp_rank)
        initial_weight_NK = torch.randn(
            fused.weight.shape,
            device=device,
            dtype=torch.float32,
            generator=weight_generator,
        )
        with torch.no_grad():
            baseline.weight.copy_(initial_weight_NK)
            fused.weight.copy_(initial_weight_NK)

        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
        )
        fully_shard(
            baseline,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=False,
        )
        fully_shard(
            fused,
            mesh=dp_mesh,
            mp_policy=mp_policy,
            reshard_after_forward=False,
        )

        baseline_schedule = Schedule1F1B(
            PipelineStage(
                baseline,
                pp_rank,
                pp_degree,
                device,
                group=pp_mesh.get_group(),
            ),
            n_microbatches=num_microbatches,
            loss_fn=nn.MSELoss(reduction="sum"),
            scale_grads=False,
        )
        fused_schedule = Schedule1F1B(
            PipelineStage(
                fused,
                pp_rank,
                pp_degree,
                device,
                group=pp_mesh.get_group(),
            ),
            n_microbatches=num_microbatches,
            loss_fn=nn.MSELoss(reduction="sum"),
            scale_grads=False,
        )

        data_generator = torch.Generator(device="cuda").manual_seed(4000 + dp_rank)
        for step in range(2):
            baseline.zero_grad(set_to_none=True)
            fused.zero_grad(set_to_none=True)
            input_BL = torch.randn(
                32,
                32,
                device=device,
                dtype=torch.bfloat16,
                generator=data_generator,
            )
            target_BL = torch.randn(
                32,
                24,
                device=device,
                dtype=torch.bfloat16,
                generator=data_generator,
            )

            baseline_losses = _run_pipeline_step(
                baseline_schedule,
                pp_rank,
                input_BL,
                target_BL,
            )
            dist.barrier()
            if step == 0:
                with torch.profiler.profile(
                    activities=[torch.profiler.ProfilerActivity.CPU]
                ) as profile:
                    fused_losses = _run_pipeline_step(
                        fused_schedule,
                        pp_rank,
                        input_BL,
                        target_BL,
                    )
                assert _operator_count(profile, "aten::addmm") == (num_microbatches - 1)
            else:
                fused_losses = _run_pipeline_step(
                    fused_schedule,
                    pp_rank,
                    input_BL,
                    target_BL,
                )
            dist.barrier()

            assert isinstance(baseline.weight.grad, DTensor)
            assert isinstance(fused.weight.grad, DTensor)
            assert fused.weight.grad.dtype == torch.float32
            baseline_local_grad = baseline.weight.grad.to_local()
            fused_local_grad = fused.weight.grad.to_local()
            relative_l2_error = (
                fused_local_grad - baseline_local_grad
            ).norm() / baseline_local_grad.norm()
            assert relative_l2_error < 1e-2
            if pp_rank == pp_degree - 1:
                assert len(baseline_losses) == num_microbatches
                assert len(fused_losses) == num_microbatches
                torch.testing.assert_close(
                    torch.stack(fused_losses),
                    torch.stack(baseline_losses),
                )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 4, reason="requires four GPUs")
def test_fused_wgrad_with_pipeline_and_fsdp():
    mp.spawn(
        _run_pipeline_fsdp_fused_wgrad,
        args=(4, get_free_port()),
        nprocs=4,
        join=True,
    )

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""In-place WGRAD accumulation of the bf16 ``Linear``."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.elastic.utils.distributed import get_free_port
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

from torchtitan.models.common.linear import Linear

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


def _build(*, inplace_wgrad_accum: bool, num_linears: int = 1) -> Linear:
    torch.manual_seed(0)
    module = Linear.Config(
        in_features=256,
        out_features=512,
        num_linears=num_linears,
        inplace_wgrad_accum=inplace_wgrad_accum,
    ).build()
    torch.nn.init.normal_(module.weight, std=0.02)
    return module.cuda()


def _count_addmm_calls(counter: list[int]):
    original_addmm = torch.addmm

    def counting_addmm(*args, **kwargs):
        counter[0] += 1
        return original_addmm(*args, **kwargs)

    return counting_addmm


def _relative_error(actual: torch.Tensor, exact: torch.Tensor) -> float:
    return ((actual.double() - exact).norm() / exact.norm()).item()


@pytest.mark.parametrize("num_linears", [1, 2])
def test_adds_bf16_wgrad_into_fp32_running_grad(num_linears, monkeypatch):
    # grad_dtype = fp32 stands in for FSDP's fp32 reduce dtype. The in-place path
    # adds the GEMM's fp32 accumulator into the running gradient instead of first
    # rounding each WGRAD to bf16, so it is at least as accurate.
    inputs = [
        torch.randn(64, 256, device="cuda", dtype=torch.bfloat16) for _ in range(4)
    ]
    grad_outputs = [
        torch.randn(64, num_linears * 512, device="cuda", dtype=torch.bfloat16)
        for _ in range(4)
    ]
    exact = sum(
        g.double().t() @ x.double() for x, g in zip(inputs, grad_outputs, strict=True)
    )

    def run(inplace_wgrad_accum):
        module = _build(
            inplace_wgrad_accum=inplace_wgrad_accum, num_linears=num_linears
        ).bfloat16()
        module.weight.grad_dtype = torch.float32
        for x, grad_output in zip(inputs, grad_outputs, strict=True):
            output = module(x)
            output.backward(grad_output.view(output.shape))
        return module.weight.grad

    reference = run(inplace_wgrad_accum=False)
    num_addmm_calls = [0]
    monkeypatch.setattr(torch, "addmm", _count_addmm_calls(num_addmm_calls))
    actual = run(inplace_wgrad_accum=True)

    assert num_addmm_calls[0] == len(inputs) - 1
    assert actual.dtype == torch.float32
    exact = exact.view(actual.shape)
    assert _relative_error(actual, exact) <= _relative_error(reference, exact)


def _run_fsdp_microbatches(rank: int, world_size: int, port: int) -> None:
    """PP-style microbatches with gradient sync disabled until the last one.

    FSDP gives the unsharded parameter grad_dtype = reduce_dtype, so the running
    fp32 gradient stays on it between microbatches and every later one adds
    into it in place.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    original_addmm = torch.addmm
    try:
        mesh = init_device_mesh("cuda", (world_size,))
        # Same data on every rank, so FSDP's average is the local gradient.
        torch.manual_seed(1)
        inputs = [
            torch.randn(64, 256, device="cuda", dtype=torch.bfloat16) for _ in range(3)
        ]
        grad_outputs = [
            torch.randn(64, 512, device="cuda", dtype=torch.bfloat16) for _ in range(3)
        ]
        exact = sum(
            g.double().t() @ x.double()
            for x, g in zip(inputs, grad_outputs, strict=True)
        )

        def run(inplace_wgrad_accum):
            module = _build(inplace_wgrad_accum=inplace_wgrad_accum)
            fully_shard(
                module,
                mesh=mesh,
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.float32
                ),
            )
            module.set_reshard_after_backward(False)
            module.set_requires_gradient_sync(False)
            for index, (x, grad_output) in enumerate(
                zip(inputs, grad_outputs, strict=True)
            ):
                if index == len(inputs) - 1:
                    module.set_requires_gradient_sync(True)
                    module.set_reshard_after_backward(True)
                module(x).backward(grad_output)
            return module.weight.grad.full_tensor()

        reference = run(inplace_wgrad_accum=False)
        num_addmm_calls = [0]
        torch.addmm = _count_addmm_calls(num_addmm_calls)
        actual = run(inplace_wgrad_accum=True)
        torch.addmm = original_addmm

        assert num_addmm_calls[0] == len(inputs) - 1, num_addmm_calls[0]
        assert _relative_error(actual, exact) <= _relative_error(reference, exact)
    finally:
        torch.addmm = original_addmm
        dist.destroy_process_group()


@pytest.mark.multi_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
def test_fsdp_microbatches_add_into_running_grad():
    mp.spawn(
        _run_fsdp_microbatches,
        args=(2, get_free_port()),
        nprocs=2,
        join=True,
    )

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
from torchtitan.models.common.linear.linear import _LinearFunction

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("pp_degree", [2, 4])
@pytest.mark.parametrize("performance", [False, True])
def test_dsv3_mlperf_router_bf16_forward_backward(pp_degree, performance):
    """The recipe gate uses BF16 storage and GEMMs at full DSV3 router geometry."""
    from torchtitan.components.data import IndexedJsonlSource
    from torchtitan.models.common.moe import MoE
    from torchtitan_recipes.models import deepseek_v3 as recipes

    source = IndexedJsonlSource.Config(patterns=("/unused/config-only.jsonl",))
    name = f"deepseek_v3_671b_mtp1_pp{pp_degree}_mlperf" + (
        "_perf" if performance else ""
    )
    config = getattr(recipes, name)(
        training_source=source,
        validation_source=source,
        hf_assets_path="/unused/assets",
    )
    _, moe, _, _ = next(config.model.traverse(MoE.Config))
    assert type(moe.router.gate) is Linear.Config
    torch.manual_seed(42)
    router = moe.router.build().cuda().eval()
    gate = router.gate.bfloat16()
    input_TD = torch.randn(
        4096,
        gate.in_features,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    grad_TE = torch.randn(
        4096,
        gate.out_features,
        device="cuda",
        dtype=torch.bfloat16,
    )
    saved_dtypes = []

    def pack(tensor):
        saved_dtypes.append(tensor.dtype)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        output_TE = gate(input_TD)
    grad_input_TD, grad_weight_ED = torch.autograd.grad(
        output_TE, (input_TD, gate.weight), grad_TE
    )
    assert (
        output_TE.dtype == grad_input_TD.dtype == grad_weight_ED.dtype == torch.bfloat16
    )
    assert saved_dtypes and all(dtype == torch.bfloat16 for dtype in saved_dtypes)
    assert torch.equal(output_TE, input_TD.detach().mm(gate.weight.detach().T))
    assert torch.equal(grad_input_TD, grad_TE.mm(gate.weight.detach()))
    assert torch.equal(grad_weight_ED, grad_TE.T.mm(input_TD.detach()))
    assert set(gate.state_dict()) == {"weight"}
    with torch.no_grad():
        weights_TK, indices_TK, _ = router(input_TD.detach())
        scores_TE = torch.sigmoid(output_TE.float())
        assert router.score_func(output_TE).dtype == torch.float32
        if performance:
            expected_indices_TK = (
                torch.arange(4096 * 8, device="cuda").reshape(4096, 8) % 256
            )
        else:
            grouped_TGE = scores_TE.reshape(4096, 8, 32)
            group_scores_TG = grouped_TGE.topk(2, dim=-1).values.sum(dim=-1)
            group_ids_TG = group_scores_TG.topk(4, dim=-1, sorted=False).indices
            group_mask_TG = torch.zeros_like(
                group_scores_TG, dtype=torch.bool
            ).scatter_(-1, group_ids_TG, True)
            choice_TE = grouped_TGE.masked_fill(
                ~group_mask_TG.unsqueeze(-1), 0
            ).flatten(-2)
            expected_indices_TK = choice_TE.topk(8, dim=-1, sorted=False).indices
        assert torch.equal(
            indices_TK.sort(-1).values, expected_indices_TK.sort(-1).values
        )
        selected_TK = scores_TE.gather(-1, indices_TK)
        expected_TK = (
            selected_TK / (selected_TK.sum(-1, keepdim=True) + 1e-20) * 2.5
        ).bfloat16()
        assert weights_TK.dtype == torch.bfloat16
        assert torch.equal(weights_TK, expected_TK)


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


@pytest.mark.parametrize("num_linears", [1, 2])
def test_first_wgrad_is_produced_in_grad_dtype(num_linears, monkeypatch):
    # Without a running gradient, the WGRAD comes out of the GEMM in the fp32
    # grad_dtype rather than being rounded to bf16 for autograd to upcast.
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16)
    grad_output = torch.randn(
        64, num_linears * 512, device="cuda", dtype=torch.bfloat16
    )
    module = _build(inplace_wgrad_accum=True, num_linears=num_linears).bfloat16()
    module.weight.grad_dtype = torch.float32
    returned_dtypes = []
    original_backward = _LinearFunction.backward

    def recording_backward(ctx, *grads):
        result = original_backward(ctx, *grads)
        returned_dtypes.append(result[1].dtype)
        return result

    monkeypatch.setattr(_LinearFunction, "backward", staticmethod(recording_backward))
    output = module(x)
    output.backward(grad_output.view(output.shape))

    assert returned_dtypes == [torch.float32]
    expected = torch.mm(grad_output.t(), x, out_dtype=torch.float32)
    assert torch.equal(module.weight.grad, expected.view(module.weight.shape))


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

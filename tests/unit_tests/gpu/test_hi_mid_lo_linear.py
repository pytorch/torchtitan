# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import itertools
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.elastic.utils.distributed import get_free_port
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.fx.experimental.proxy_tensor import make_fx

from torchtitan.distributed.local_compile import apply_local_compile
from torchtitan.models.common import hi_mid_lo_linear
from torchtitan.models.common.hi_mid_lo_linear import HiMidLoLinear


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("backward_mode", ["hi_mid", "hi_mid_lo"])
@pytest.mark.parametrize(
    ("input_dtype", "weight_dtype"),
    [
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32),
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.float32),
    ],
)
def test_hi_mid_lo_linear_compiles(input_dtype, weight_dtype, backward_mode):
    layer = HiMidLoLinear.Config(
        in_features=128,
        out_features=16,
        bias=True,
        backward_mode=backward_mode,
    ).build()
    layer = layer.to(device="cuda", dtype=weight_dtype)
    compiled = torch.compile(layer, fullgraph=True)
    input_TD = torch.randn(
        32, 128, device="cuda", dtype=input_dtype, requires_grad=True
    )

    output_TO = compiled(input_TD)
    output_TO.sum().backward()

    assert output_TO.dtype is torch.float32
    assert input_TD.grad is not None
    assert input_TD.grad.dtype is input_dtype
    assert layer.weight.grad is not None
    assert layer.weight.grad.dtype is weight_dtype


def _lm_head(in_features: int = 256, out_features: int = 1024) -> HiMidLoLinear:
    lm_head = HiMidLoLinear.Config(
        in_features=in_features, out_features=out_features
    ).build()
    torch.nn.init.normal_(lm_head.weight, std=0.02)
    return lm_head.to(device="cuda", dtype=torch.bfloat16)


def test_forward_matches_fp64_reference():
    lm_head = _lm_head()
    x = torch.randn(3, 5, 256, device="cuda", dtype=torch.bfloat16)
    reference = F.linear(x.double(), lm_head.weight.double())

    out = lm_head(x)

    assert out.dtype == torch.float32
    assert out.shape == reference.shape
    torch.testing.assert_close(out.double(), reference, rtol=1e-4, atol=1e-4)


def _relative_error(actual, exact):
    return ((actual.double() - exact).norm() / exact.norm()).item()


def _backward_errors_vs_bf16_floor(function, num_tokens, in_features, out_features):
    """Backward of ``function`` vs exact fp64 gradients, as multiples of the bf16 floor."""
    x = torch.randn(num_tokens, in_features, device="cuda", dtype=torch.bfloat16)
    weight = (torch.randn(out_features, in_features, device="cuda") * 0.02).bfloat16()
    grad_output = torch.randn(num_tokens, out_features, device="cuda")

    x_exact = x.double().requires_grad_()
    weight_exact = weight.double().requires_grad_()
    F.linear(x_exact, weight_exact).backward(grad_output.double())

    x_leaf = x.clone().requires_grad_()
    weight_leaf = weight.clone().requires_grad_()
    function(x_leaf, weight_leaf).backward(grad_output)

    # Autograd rounds both gradients to bf16 (default grad_dtype), so the bf16-rounded exact ones are
    # the floor.
    ratios = []
    for grad, exact in (
        (x_leaf.grad, x_exact.grad),
        (weight_leaf.grad, weight_exact.grad),
    ):
        assert grad.dtype == torch.bfloat16
        ratios.append(
            _relative_error(grad, exact) / _relative_error(exact.bfloat16(), exact)
        )
    return ratios


# out_features > num_tokens takes the LM-head layout, out_features < num_tokens the router one.
@pytest.mark.parametrize("num_pieces", [2, 3])
@pytest.mark.parametrize(
    "num_tokens,in_features,out_features", [(64, 256, 1024), (512, 256, 16)]
)
def test_backward_error_stays_at_bf16_rounding_floor(
    num_tokens, in_features, out_features, num_pieces
):
    ratios = _backward_errors_vs_bf16_floor(
        lambda input, weight: hi_mid_lo_linear._HiMidLoLinearFunction.apply(
            input, weight, num_pieces
        ),
        num_tokens,
        in_features,
        out_features,
    )
    assert max(ratios) < 1.05, ratios


def test_compiled_backward_keeps_mid():
    # An outer torch.compile must trace the backward and keep mid: the split upcasts with an
    # integer shift, so Inductor can't drop the round trip.
    # Compile a wrapper: compiling any ``Function.apply`` directly breaks later compiles of other
    # autograd Functions in the same process (test_qwen3_5_deltanet fails after it).
    def linear(input, weight):
        return hi_mid_lo_linear._HiMidLoLinearFunction.apply(input, weight, 2)

    # fullgraph=True: the LM-head backward must trace without a break.
    ratios = _backward_errors_vs_bf16_floor(
        torch.compile(linear, fullgraph=True), 64, 256, 1024
    )
    assert max(ratios) < 1.05, ratios


@pytest.mark.parametrize("compile", [False, True])
# 1 token takes the LM-head layout, 4 tokens the router one.
@pytest.mark.parametrize("num_tokens", [1, 4])
@pytest.mark.parametrize("num_pieces", [2, 3])
def test_third_piece_keeps_what_two_pieces_drop(num_pieces, num_tokens, compile):
    # (1 + 2^-8 + 2^-20) - (1 + 2^-8) = 2^-20: 2 pieces round the 2^-20 away, 3 keep it. The
    # split is compiled, so this also catches a split that Inductor folds (the third piece would
    # compile to zero). Earlier tests fill the split's Dynamo cache; past the limit it runs eagerly.
    torch._dynamo.reset()
    grad_output = torch.tensor([[1 + 2**-8 + 2**-20, 1 + 2**-8]], device="cuda")
    grad_output = grad_output.repeat(num_tokens, 1)
    weight = torch.tensor([[1.0] * 8, [-1.0] * 8], device="cuda", dtype=torch.bfloat16)
    x = torch.zeros(
        num_tokens, 8, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )

    def linear(input, weight):
        return hi_mid_lo_linear._HiMidLoLinearFunction.apply(input, weight, num_pieces)

    (torch.compile(linear) if compile else linear)(x, weight).backward(grad_output)

    expected = 2**-20 if num_pieces == 3 else 0.0
    assert torch.equal(x.grad, torch.full_like(x.grad, expected))


@contextlib.contextmanager
def _compiled_split():
    """Turn on the split's local_compile region, as the models that use HiMidLoLinear do."""
    apply_local_compile(["fp32_to_bf16_split"])
    try:
        yield
    finally:
        apply_local_compile([])


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("num_pieces", [2, 3])
def test_compiled_split_matches_eager_split(num_pieces, dim):
    # Compiled (its local_compile region on), the split must give the eager pieces bit for bit,
    # including signed zeros, ties (1 + 2^-8 sits halfway between two bf16s) and tiny and huge values.
    torch._dynamo.reset()
    torch.manual_seed(0)
    grad_output = torch.randn(64, 1024, device="cuda")
    grad_output *= torch.logspace(-30, 30, 1024, device="cuda")
    grad_output[0, :4] = torch.tensor([0.0, -0.0, 1 + 2**-8, -(1 + 2**-8)])
    eager = hi_mid_lo_linear._split_into_bf16_pieces.__wrapped__(
        grad_output, num_pieces, dim
    )

    with _compiled_split():
        compiled = hi_mid_lo_linear._split_into_bf16_pieces(
            grad_output, num_pieces, dim
        )

    assert torch.equal(compiled.view(torch.int16), eager.view(torch.int16))


def test_backward_compiles_the_split_once_per_pieces_and_layout():
    # The split compiles with symbolic shapes (dynamic=True): one graph per (number of pieces,
    # layout), whatever the token count (here >= 2) or out_features.
    torch._dynamo.reset()
    counters = torch._dynamo.utils.counters
    counters.clear()
    # 1024 and 2048 out_features take the LM-head layout, 16 and 32 the router one.
    cases = itertools.product((2, 3), (1024, 2048, 16, 32), (64, 65, 333, 999))
    with _compiled_split():
        for num_pieces, out_features, num_tokens in cases:
            x = torch.randn(
                num_tokens, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            weight = torch.randn(
                out_features,
                256,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            )
            output = hi_mid_lo_linear._HiMidLoLinearFunction.apply(
                x, weight, num_pieces
            )
            torch.autograd.grad(output, (x, weight), torch.randn_like(output))

    assert counters["stats"]["unique_graphs"] == 4


def _forward_backward(x, weight, grad_output):
    output = hi_mid_lo_linear._HiMidLoLinearFunction.apply(x, weight, 2)
    return torch.autograd.grad(output, (x, weight), grad_output)


@pytest.mark.parametrize("tracing_mode", ["real", "fake", "symbolic"])
@pytest.mark.parametrize("out_features", [1024, 16])
def test_backward_traces_with_make_fx(tracing_mode, out_features):
    # make_fx (graph_trainer, region off) must trace the backward; the replay below uses a new
    # grad_output, so a traced constant would fail it.
    x = torch.randn(64, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(
        out_features, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    traced = make_fx(_forward_backward, tracing_mode=tracing_mode)(
        x, weight, torch.randn(64, out_features, device="cuda")
    )

    grad_output = torch.randn(64, out_features, device="cuda")
    for actual, expected in zip(
        traced(x, weight, grad_output), _forward_backward(x, weight, grad_output)
    ):
        assert torch.equal(actual, expected)


# out_features > num_tokens takes the LM-head layout, out_features < num_tokens the router one.
@pytest.mark.parametrize("batch_invariant", [False, True])
@pytest.mark.parametrize("num_pieces", [2, 3])
@pytest.mark.parametrize("num_tokens,out_features", [(64, 1024), (512, 16)])
def test_weight_grad_stays_fp32_when_grad_dtype_is_fp32(
    num_tokens, out_features, num_pieces, batch_invariant, monkeypatch
):
    # grad_dtype = fp32 stands in for FSDP (https://github.com/pytorch/pytorch/pull/194434): the
    # fp32 grad_weight skips the bf16 rounding. Batch-invariant mode (RL) takes the fp32 fallback.
    monkeypatch.setattr(
        hi_mid_lo_linear, "is_in_batch_invariant_mode", lambda: batch_invariant
    )
    x = torch.randn(num_tokens, 256, device="cuda", dtype=torch.bfloat16)
    weight = (torch.randn(out_features, 256, device="cuda") * 0.02).bfloat16()
    weight.requires_grad_()
    weight.grad_dtype = torch.float32
    grad_output = torch.randn(num_tokens, out_features, device="cuda")

    hi_mid_lo_linear._HiMidLoLinearFunction.apply(x, weight, num_pieces).backward(
        grad_output
    )

    exact = grad_output.double().T @ x.double()
    assert weight.grad.dtype == torch.float32
    floor = _relative_error(exact.bfloat16(), exact)
    assert _relative_error(weight.grad, exact) < 0.01 * floor


def test_backward_handles_zero_tokens():
    x = torch.empty(0, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(
        16, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )

    hi_mid_lo_linear._HiMidLoLinearFunction.apply(x, weight, 3).sum().backward()

    assert x.grad.shape == x.shape
    assert torch.equal(weight.grad, torch.zeros_like(weight.grad))


def _run_fsdp_keeps_fp32_weight_grad(rank, world_size, port, compile):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    try:
        mesh = init_device_mesh("cuda", (world_size,))
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16, reduce_dtype=torch.float32
        )
        # (num_tokens, out_features, backward_mode): the LM head ships "hi_mid", routers "hi_mid_lo".
        for num_tokens, out_features, backward_mode in (
            (64, 1024, "hi_mid"),
            (512, 16, "hi_mid_lo"),
        ):
            # Same data on every rank, so FSDP's average is the local gradient.
            torch.manual_seed(0)
            layer = HiMidLoLinear.Config(
                in_features=256,
                out_features=out_features,
                backward_mode=backward_mode,
            ).build()
            layer = layer.cuda()
            torch.nn.init.normal_(layer.weight, std=0.02)
            fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
            forward = torch.compile(layer) if compile else layer
            x = torch.randn(num_tokens, 256, device="cuda").bfloat16()
            grad_output = torch.randn(num_tokens, out_features, device="cuda")

            forward(x).backward(grad_output)

            exact_grad = grad_output.double().T @ x.double()
            floor = _relative_error(exact_grad.bfloat16(), exact_grad)
            grad = layer.weight.grad.full_tensor()
            assert _relative_error(grad, exact_grad) < 0.01 * floor
    finally:
        dist.destroy_process_group()


@pytest.mark.multi_gpu
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two GPUs")
@pytest.mark.parametrize(
    "compile",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.xfail(
                strict=True,
                reason="AOTAutograd rounds grad_weight to bf16 (https://github.com/pytorch/pytorch/pull/197381)",
            ),
        ),
    ],
)
def test_fsdp_keeps_fp32_weight_grad(compile):
    mp.spawn(
        _run_fsdp_keeps_fp32_weight_grad,
        args=(2, get_free_port(), compile),
        nprocs=2,
        join=True,
    )


def test_batch_invariant_mode_computes_in_fp32(monkeypatch):
    lm_head = _lm_head()
    x = torch.randn(4, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    monkeypatch.setattr(hi_mid_lo_linear, "is_in_batch_invariant_mode", lambda: True)

    out = lm_head(x)
    out.sum().backward()

    assert torch.equal(out, F.linear(x.detach().float(), lm_head.weight.float()))
    assert x.grad.dtype is torch.bfloat16


def test_backward_runs_under_autocast_with_fp32_params():
    # Autocast makes the fp32 fallback's output bf16; the backward must still run.
    layer = HiMidLoLinear.Config(in_features=64, out_features=32).build().cuda()
    x = torch.randn(16, 64, device="cuda", requires_grad=True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = layer(x)
    out.float().sum().backward()

    assert x.grad.dtype is torch.float32
    assert layer.weight.grad.dtype is torch.float32

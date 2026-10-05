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
import torch.nn.functional as F
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.elastic.utils.distributed import get_free_port
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.fx.experimental.proxy_tensor import make_fx

from torchtitan.models.common import fp32_output_linear
from torchtitan.models.common.fp32_output_linear import FP32OutputLinear


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.mark.parametrize("higher_precision_bwd", [False, True])
@pytest.mark.parametrize(
    ("input_dtype", "weight_dtype"),
    [
        (torch.bfloat16, torch.bfloat16),
        (torch.float32, torch.float32),
        (torch.float32, torch.bfloat16),
        (torch.bfloat16, torch.float32),
    ],
)
def test_fp32_output_linear_compiles(input_dtype, weight_dtype, higher_precision_bwd):
    layer = FP32OutputLinear.Config(
        in_features=128,
        out_features=16,
        bias=True,
        higher_precision_bwd=higher_precision_bwd,
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


def _lm_head(in_features: int = 256, out_features: int = 1024) -> FP32OutputLinear:
    lm_head = FP32OutputLinear.Config(
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
@pytest.mark.parametrize("higher_precision_bwd", [False, True])
@pytest.mark.parametrize(
    "num_tokens,in_features,out_features", [(64, 256, 1024), (512, 256, 16)]
)
def test_backward_error_stays_at_bf16_rounding_floor(
    num_tokens, in_features, out_features, higher_precision_bwd
):
    ratios = _backward_errors_vs_bf16_floor(
        lambda input, weight: fp32_output_linear._FP32OutputLinearFunction.apply(
            input, weight, higher_precision_bwd
        ),
        num_tokens,
        in_features,
        out_features,
    )
    assert max(ratios) < 1.05, ratios


def test_compiled_backward_keeps_lo_half():
    # An outer torch.compile must trace the backward and keep lo: it records the split as one
    # custom op, so it can't fold the split's bf16 round trip.
    # Compile a wrapper: compiling any ``Function.apply`` directly breaks later compiles of other
    # autograd Functions in the same process (test_qwen3_5_deltanet fails after it).
    def linear(input, weight):
        return fp32_output_linear._FP32OutputLinearFunction.apply(input, weight, False)

    # fullgraph=True: the LM-head backward must trace without a break (the split is a custom op).
    ratios = _backward_errors_vs_bf16_floor(
        torch.compile(linear, fullgraph=True), 64, 256, 1024
    )
    assert max(ratios) < 1.05, ratios


@pytest.mark.parametrize("compile", [False, True])
# 1 token takes the LM-head layout, 4 tokens the router one.
@pytest.mark.parametrize("num_tokens", [1, 4])
@pytest.mark.parametrize("higher_precision_bwd", [False, True])
def test_third_piece_keeps_what_two_pieces_drop(
    higher_precision_bwd, num_tokens, compile
):
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
        return fp32_output_linear._FP32OutputLinearFunction.apply(
            input, weight, higher_precision_bwd
        )

    (torch.compile(linear) if compile else linear)(x, weight).backward(grad_output)

    expected = 2**-20 if higher_precision_bwd else 0.0
    assert torch.equal(x.grad, torch.full_like(x.grad, expected))


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("higher_precision_bwd", [False, True])
def test_compiled_split_matches_eager_split(higher_precision_bwd, dim):
    # The split is compiled. It must give the eager pieces bit for bit, including signed zeros, ties
    # (1 + 2^-8 sits halfway between two bf16s) and tiny and huge values.
    # Earlier tests fill the split's Dynamo cache; past the recompile limit it would run eagerly.
    torch._dynamo.reset()
    torch.manual_seed(0)
    grad_output = torch.randn(64, 1024, device="cuda")
    grad_output *= torch.logspace(-30, 30, 1024, device="cuda")
    grad_output[0, :4] = torch.tensor([0.0, -0.0, 1 + 2**-8, -(1 + 2**-8)])
    eager = fp32_output_linear._split_into_bf16_pieces_eager(
        grad_output, higher_precision_bwd, dim
    )

    compiled = fp32_output_linear._split_into_bf16_pieces(
        grad_output, higher_precision_bwd, dim
    )

    assert torch.equal(compiled.view(torch.int16), eager.view(torch.int16))


def test_compiled_split_runs_eagerly_past_the_recompile_limit():
    # Each call below needs a new graph. Past Dynamo's recompile limit it must run eagerly; with
    # fullgraph=True it would raise inside backward.
    torch.manual_seed(0)
    with torch._dynamo.config.patch(recompile_limit=1):
        for num_tokens, higher_precision_bwd, dim in (
            (64, False, 0),
            (64, True, 0),
            (64, True, 1),
            (1, True, 1),
        ):
            grad_output = torch.randn(num_tokens, 1024, device="cuda")
            eager = fp32_output_linear._split_into_bf16_pieces_eager(
                grad_output, higher_precision_bwd, dim
            )

            compiled = fp32_output_linear._split_into_bf16_pieces(
                grad_output, higher_precision_bwd, dim
            )

            assert torch.equal(compiled.view(torch.int16), eager.view(torch.int16))
    # Past the limit, Dynamo never compiles the split again in this process.
    torch._dynamo.reset()


@pytest.mark.parametrize("dim", [0, 1])
@pytest.mark.parametrize("higher_precision_bwd", [False, True])
def test_split_custom_op_passes_opcheck(higher_precision_bwd, dim):
    # Checks the schema, register_fake against the real output (FakeTensorMode), and a dynamic-shape
    # trace through AOTAutograd, as graph_trainer's make_fx and an outer torch.compile do.
    grad_output = torch.randn(64, 1024, device="cuda")

    torch.library.opcheck(
        fp32_output_linear._split_into_bf16_pieces,
        (grad_output, higher_precision_bwd, dim),
    )


# 1024 out_features takes the LM-head layout, 16 the router one.
@pytest.mark.parametrize("out_features", [1024, 16])
def test_backward_compiles_the_split_twice_for_all_token_counts(out_features):
    # The token dim starts static and turns symbolic the first time it changes, so ten token
    # counts compile two graphs, and out_features stays static.
    torch._dynamo.reset()
    counters = torch._dynamo.utils.counters
    counters.clear()
    weight = torch.randn(
        out_features, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    for num_tokens in (64, 65, 100, 128, 200, 333, 500, 512, 700, 999):
        x = torch.randn(
            num_tokens, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        _forward_backward(
            x, weight, torch.randn(num_tokens, out_features, device="cuda")
        )

    assert counters["stats"]["unique_graphs"] == 2


def _forward_backward(x, weight, grad_output):
    output = fp32_output_linear._FP32OutputLinearFunction.apply(x, weight, False)
    return torch.autograd.grad(output, (x, weight), grad_output)


@pytest.mark.parametrize("tracing_mode", ["real", "fake", "symbolic"])
@pytest.mark.parametrize("out_features", [1024, 16])
def test_backward_traces_with_make_fx(tracing_mode, out_features):
    # make_fx (graph_trainer) can't trace into the compiled split, so the backward must reach it
    # through the custom op: fake and symbolic tracing would fail, and real tracing would bake the
    # split's output into the graph (the replay below uses a new grad_output).
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
@pytest.mark.parametrize("higher_precision_bwd", [False, True])
@pytest.mark.parametrize("num_tokens,out_features", [(64, 1024), (512, 16)])
def test_weight_grad_stays_fp32_when_grad_dtype_is_fp32(
    num_tokens, out_features, higher_precision_bwd, batch_invariant, monkeypatch
):
    # grad_dtype = fp32 stands in for FSDP (https://github.com/pytorch/pytorch/pull/194434): the
    # fp32 grad_weight skips the bf16 rounding. Batch-invariant mode (RL) takes the fp32 fallback.
    monkeypatch.setattr(
        fp32_output_linear, "is_in_batch_invariant_mode", lambda: batch_invariant
    )
    x = torch.randn(num_tokens, 256, device="cuda", dtype=torch.bfloat16)
    weight = (torch.randn(out_features, 256, device="cuda") * 0.02).bfloat16()
    weight.requires_grad_()
    weight.grad_dtype = torch.float32
    grad_output = torch.randn(num_tokens, out_features, device="cuda")

    fp32_output_linear._FP32OutputLinearFunction.apply(
        x, weight, higher_precision_bwd
    ).backward(grad_output)

    exact = grad_output.double().T @ x.double()
    assert weight.grad.dtype == torch.float32
    floor = _relative_error(exact.bfloat16(), exact)
    assert _relative_error(weight.grad, exact) < 0.01 * floor


def test_backward_handles_zero_tokens():
    x = torch.empty(0, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(
        16, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )

    fp32_output_linear._FP32OutputLinearFunction.apply(x, weight, True).sum().backward()

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
        # (num_tokens, out_features, higher_precision_bwd): the LM head ships False, routers True.
        for num_tokens, out_features, higher_precision_bwd in (
            (64, 1024, False),
            (512, 16, True),
        ):
            # Same data on every rank, so FSDP's average is the local gradient.
            torch.manual_seed(0)
            layer = FP32OutputLinear.Config(
                in_features=256,
                out_features=out_features,
                higher_precision_bwd=higher_precision_bwd,
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
    monkeypatch.setattr(fp32_output_linear, "is_in_batch_invariant_mode", lambda: True)

    out = lm_head(x)
    out.sum().backward()

    assert torch.equal(out, F.linear(x.detach().float(), lm_head.weight.float()))
    assert x.grad.dtype is torch.bfloat16


def test_backward_runs_under_autocast_with_fp32_params():
    # Autocast makes the fp32 fallback's output bf16; the backward must still run.
    layer = FP32OutputLinear.Config(in_features=64, out_features=32).build().cuda()
    x = torch.randn(16, 64, device="cuda", requires_grad=True)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = layer(x)
    out.float().sum().backward()

    assert x.grad.dtype is torch.float32
    assert layer.weight.grad.dtype is torch.float32

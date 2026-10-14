# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import subprocess
import sys
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves
from torch.utils.checkpoint import checkpoint
from torchtitan.models.common.hi_mid_lo_linear import (
    _HiMidLoLinearFunction,
    _narrow_backward,
    HiMidLoLinear,
)
from torchtitan_recipes.overrides import fused_dsv3_router_gate as fusion

from tests.unit_tests.gpu.test_fused_dsv3_router import _config


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 3)
    or torch.__version__ != "2.16.0.dev20261007+cu130"
    or fusion._CUTEDSL_IMPORT_ERROR is not None,
    reason="Recorded GB300/PyTorch/CuTeDSL router-gate specialization",
)


def _assert_bits(expected, actual):
    assert expected.shape == actual.shape and expected.dtype == actual.dtype
    assert torch.equal(
        expected.detach().contiguous().reshape(-1).view(torch.uint8),
        actual.detach().contiguous().reshape(-1).view(torch.uint8),
    )


@pytest.fixture(autouse=True)
def _deterministic():
    previous = torch.are_deterministic_algorithms_enabled()
    reduced = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduced


@pytest.fixture(scope="module")
def _operands():
    torch.manual_seed(42)
    return (
        torch.randn(4096, 7168, device="cuda", dtype=torch.bfloat16),
        torch.randn(256, 7168, device="cuda", dtype=torch.bfloat16),
    )


@pytest.mark.parametrize("num_pieces", [2, 3])
@pytest.mark.parametrize("reduced", [False, True])
@pytest.mark.parametrize(
    "case", ["random", "ties", "zero", "tiny", "large", "nonfinite", "ieee"]
)
def test_operators_match_native_bits(_operands, num_pieces, reduced, case):
    x, w = _operands
    g = torch.randn(4096, 256, device="cuda")
    if case == "ties":
        index = torch.arange(g.numel(), dtype=torch.int32, device="cuda")
        bits = (((index & 127) + 0x3F80) << 16) | 0x8000 | (((index >> 7) & 1) << 31)
        g = bits.view(torch.float32).view_as(g)
    elif case == "zero":
        g.zero_()
        g[::2].neg_()
    elif case == "tiny":
        g.mul_(2.0**-120)
    elif case == "large":
        g.mul_(2.0**110)
    elif case == "nonfinite":
        g[0, :3] = torch.tensor(
            [float("nan"), float("inf"), -float("inf")], device="cuda"
        )
    elif case == "ieee":
        g = torch.randint(
            -(2**31), 2**31 - 1, g.shape, device="cuda", dtype=torch.int32
        ).view(torch.float32)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = reduced
    assert fusion._supports_gate_backward(g, w, num_pieces=num_pieces, wgrad=False)
    assert fusion._supports_gate_backward(g, x, num_pieces=num_pieces, wgrad=True)
    expected = _narrow_backward(
        g, x, w, num_pieces=num_pieces, needs_grad_input=True, needs_grad_weight=True
    )
    actual = fusion.backward_op(g, x, w, num_pieces, True, True)
    for reference, result in zip(expected, actual):
        _assert_bits(reference, result)


@pytest.mark.parametrize("needs", [(True, False), (False, True)])
def test_partial_backward_does_not_prepare_unused_operand(_operands, needs):
    x, w = _operands
    if needs[0]:
        x = x.T.contiguous().T
    else:
        w = w.T.contiguous().T
    g = torch.randn(4096, 256, device="cuda")
    expected = _narrow_backward(
        g,
        x,
        w,
        num_pieces=3,
        needs_grad_input=needs[0],
        needs_grad_weight=needs[1],
    )
    actual = fusion.backward_op(g, x, w, 3, *needs)
    for reference, result in zip(expected, actual):
        if reference is None:
            assert result is None
        else:
            _assert_bits(reference, result)


@pytest.mark.parametrize(
    "layout", ["transposed", "expanded", "negative", "unaligned", "operand_unaligned"]
)
def test_operator_layouts_match_native(_operands, layout):
    x, w = _operands
    g = torch.randn(4096, 256, device="cuda")
    if layout == "transposed":
        g = torch.randn(256, 4096, device="cuda").T
    elif layout == "expanded":
        g = torch.randn(1, device="cuda").expand(4096, 256)
    elif layout == "negative":
        g = torch._neg_view(g)
    elif layout == "unaligned":
        g = torch.cat((g.new_zeros(1), g.flatten()))[1:].view_as(g)
    else:
        x = torch.cat((x.new_zeros(1), x.flatten()))[1:].view_as(x)
        w = torch.cat((w.new_zeros(1), w.flatten()))[1:].view_as(w)
    expected = _narrow_backward(
        g, x, w, num_pieces=3, needs_grad_input=True, needs_grad_weight=True
    )
    # The dispatcher resolves negative views before entering a custom operator.
    kernel_mock = (
        {"wraps": fusion._backward_kernels}
        if layout == "negative"
        else {"side_effect": AssertionError("layout uses native backward")}
    )
    with patch.object(fusion, "_backward_kernels", **kernel_mock):
        actual = fusion.backward_op(g, x, w, 3, True, True)
    for reference, result in zip(expected, actual):
        _assert_bits(reference, result)


@pytest.mark.parametrize("needs", [(True, True), (True, False), (False, True)])
@pytest.mark.parametrize("recompute", [False, True])
def test_autograd_preserves_partial_gradients_accumulation_and_saves(
    _operands, needs, recompute
):
    x, w = _operands
    g = torch.randn(4096, 256, device="cuda")
    results = []
    for function in (
        _HiMidLoLinearFunction.apply,
        fusion.FusedDSv3RouterGateFunction.apply,
    ):
        input_TD = x.clone().requires_grad_(needs[0])
        weight_ED = w.clone().requires_grad_(needs[1])
        weight_ED.grad_dtype = torch.float32
        saved = []

        def pack(tensor):
            saved.append(tensor.data_ptr())
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            if recompute:
                output = checkpoint(
                    function, input_TD, weight_ED, 3, use_reentrant=False
                )
            else:
                output = function(input_TD, weight_ED, 3)
        for _ in range(2):
            output.backward(g, retain_graph=True)
        results.append((output, input_TD.grad, weight_ED.grad))
        if not recompute:
            assert saved == [input_TD.data_ptr(), weight_ED.data_ptr()]
        assert (input_TD.grad is not None) == needs[0]
        assert (weight_ED.grad is not None) == needs[1]
        if needs[1]:
            assert weight_ED.grad.dtype == torch.float32
    for expected, actual in zip(*results):
        if expected is None:
            assert actual is None
        else:
            _assert_bits(expected, actual)


class _RecordOperators(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.names = []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        if "dsv3_router" in str(function) or "dsv3_seqwise" in str(function):
            self.names.append(str(function))
        if "dsv3_router_gate" in str(function):
            assert fusion._supports_gate_backward(
                args[0], args[1], num_pieces=args[3], wgrad=True
            )
            assert fusion._supports_gate_backward(
                args[0], args[2], num_pieces=args[3], wgrad=False
            )
        return function(*args, **(kwargs or {}))


@pytest.mark.parametrize("composed", [False, True])
def test_module_keeps_gate_boundary_state_and_router_composition(composed):
    if composed:
        cfg = _config(dim=7168)
        native = cfg.build().cuda().bfloat16()
        with pytest.warns(UserWarning, match="roofline acceptance"):
            candidate = (
                fusion.fused_dsv3_router_with_gate(
                    cfg, experimental=True, seqwise_loss=True
                )
                .build()
                .cuda()
                .bfloat16()
            )
    else:
        cfg = HiMidLoLinear.Config(
            in_features=7168, out_features=256, bias=True, backward_mode="hi_mid_lo"
        )
        native = cfg.build().cuda().bfloat16()
        with pytest.warns(UserWarning, match="roofline acceptance"):
            candidate = (
                fusion.fused_dsv3_router_gate(cfg, experimental=True)
                .build()
                .cuda()
                .bfloat16()
            )
    candidate.load_state_dict(native.state_dict())
    x = torch.randn(4096, 7168, device="cuda", dtype=torch.bfloat16)
    g = torch.randn(4096, 8 if composed else 256, device="cuda")
    bias = torch.randn(256, device="cuda") * 0.025
    outputs, traces = [], []
    for model in (native, candidate):
        gate = model.gate if composed else model
        gate.weight.grad_dtype = torch.float32
        hook_calls = []
        hook = gate.register_forward_hook(lambda *args: hook_calls.append("forward"))
        input_TD = x.clone().requires_grad_()
        trace = _RecordOperators()
        with trace:
            if composed:
                output = model(
                    input_TD,
                    bias,
                    aux_loss_denominator=torch.tensor(4096.0, device="cuda"),
                )[0]
            else:
                output = model(input_TD)
            output.backward(g)
        hook.remove()
        assert hook_calls == ["forward"]
        outputs.append((output, input_TD.grad, gate.weight.grad))
        traces.append(trace.names)
    for expected, actual in zip(*outputs):
        _assert_bits(expected, actual)
    assert "torchtitan.dsv3_router_gate_backward.default" in traces[1]
    if composed:
        assert "torchtitan.dsv3_router_forward.default" in traces[1]
        _assert_bits(native.tokens_per_expert_E, candidate.tokens_per_expert_E)
        _assert_bits(native.aux_loss.instance_acc, candidate.aux_loss.instance_acc)
    assert native.state_dict().keys() == candidate.state_dict().keys()


def test_compiled_autograd_and_cuda_replay(_operands):
    x, w = _operands
    x = x.clone().requires_grad_()
    w = w.clone().requires_grad_()
    compiled = torch.compile(fusion.FusedDSv3RouterGateFunction.apply, fullgraph=True)
    g = torch.randn(4096, 256, device="cuda")
    reference = _HiMidLoLinearFunction.apply(x, w, 3)
    expected = torch.autograd.grad(reference, (x, w), g)
    unaligned = torch.cat((g.new_zeros(1), g.flatten()))[1:].view_as(g)
    for gradient in (g, unaligned):
        result = compiled(x, w, 3)
        actual = torch.autograd.grad(result, (x, w), gradient)
        _assert_bits(reference, result)
        for a, b in zip(expected, actual):
            _assert_bits(a, b)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        inputs = x.detach().requires_grad_(), w.detach().requires_grad_()

        def run():
            output = fusion.FusedDSv3RouterGateFunction.apply(*inputs, 3)
            return output, *torch.autograd.grad(output, inputs, g)

        for _ in range(3):
            run()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured = run()
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(3):
        with torch.no_grad():
            inputs[0].copy_(torch.randn_like(inputs[0]))
            inputs[1].copy_(torch.randn_like(inputs[1]))
            g.copy_(torch.randn_like(g))
        graph.replay()
        reference = _HiMidLoLinearFunction.apply(*inputs, 3)
        expected = (reference, *torch.autograd.grad(reference, inputs, g))
        for a, b in zip(expected, captured):
            _assert_bits(a, b)


def test_compiled_fp32_weight_gradient_matches_compiled_native(_operands):
    x, w = _operands
    g = torch.randn(4096, 256, device="cuda")
    results = []
    for function in (
        _HiMidLoLinearFunction.apply,
        fusion.FusedDSv3RouterGateFunction.apply,
    ):
        input_TD = x.clone().requires_grad_()
        weight_ED = w.clone().requires_grad_()
        weight_ED.grad_dtype = torch.float32
        output = torch.compile(function, fullgraph=True)(input_TD, weight_ED, 3)
        output.backward(g)
        assert weight_ED.grad.dtype == torch.float32
        results.append((output, input_TD.grad, weight_ED.grad))
    # Native AOTAutograd currently rounds the values despite the FP32 storage.
    # Compare compiled with compiled; this override does not fix that issue.
    for expected, actual in zip(*results):
        _assert_bits(expected, actual)


class _ZeroCollectives(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.collectives = []
        self.fusions = []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        name = str(function)
        result = function(*args, **(kwargs or {}))
        if "dsv3_router_gate" in name:
            assert fusion._supports_gate_backward(
                args[0], args[1], num_pieces=args[3], wgrad=True
            )
            assert fusion._supports_gate_backward(
                args[0], args[2], num_pieces=args[3], wgrad=False
            )
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
            # FakePG does not supply remote data. Define its output storage so
            # uninitialized memory cannot masquerade as numerical evidence.
            for tensor in tree_leaves(result):
                if isinstance(tensor, torch.Tensor):
                    tensor.zero_()
        return result


def _exercise(rank, reshard_after_forward):
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group("fake", store=FakeStore(), rank=rank, world_size=128)
    try:
        mesh = init_device_mesh("cuda", (128,), mesh_dim_names=("dp_shard",))
        models = []
        hook_calls = [[], []]
        for index, fused in enumerate((False, True)):
            config = HiMidLoLinear.Config(
                in_features=7168, out_features=256, backward_mode="hi_mid_lo"
            )
            if fused:
                config = fusion.fused_dsv3_router_gate(config, experimental=True)
            with torch.device("meta"):
                model = config.build()
            fully_shard(
                model,
                mesh=mesh,
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.float32
                ),
                reshard_after_forward=reshard_after_forward,
            )
            model.to_empty(device="cuda")
            with torch.no_grad():
                model.weight.zero_()
            model.register_forward_hook(
                lambda *args, calls=hook_calls[index]: calls.append("forward")
            )
            models.append(model)
        x = torch.randn(1, 4096, 7168, device="cuda", dtype=torch.bfloat16)
        inputs = [x.clone().requires_grad_() for _ in models]
        records = [_ZeroCollectives(), _ZeroCollectives()]
        for synchronize in (True, False, True):
            gradient = torch.randn(1, 4096, 256, device="cuda")
            results = []
            for model, input, record in zip(models, inputs, records):
                model.set_requires_gradient_sync(synchronize)
                with record:
                    output = model(input)
                    output.backward(gradient)
                grad_weight = model.weight.grad
                if grad_weight is not None:
                    assert grad_weight.dtype == torch.float32
                    grad_weight = grad_weight.to_local()
                results.append((output.detach(), input.grad, grad_weight))
            for expected, actual in zip(*results):
                if expected is None:
                    assert actual is None
                else:
                    _assert_bits(expected, actual)
            if synchronize:
                for model in models:
                    assert model.weight.grad is not None
                    model.zero_grad(set_to_none=True)
        assert hook_calls == [["forward"] * 3, ["forward"] * 3]
        assert records[0].collectives == records[1].collectives
        for collective in ("allgather", "reducescatter"):
            assert any(
                collective in name.replace("_", "")
                for name, _ in records[1].collectives
            )
        assert records[0].fusions == []
        assert (
            records[1].fusions.count("torchtitan.dsv3_router_gate_backward.default")
            == 3
        )
        state = models[0].state_dict()
        assert state.keys() == models[1].state_dict().keys()
        assert state["weight"].shape == (256, 7168)
        assert state["weight"].dtype == torch.float32
        models[1].load_state_dict(state)
        torch.cuda.synchronize()
        print(
            {
                "rank": rank,
                "reshard_after_forward": reshard_after_forward,
                "collectives": records[1].collectives,
                "fusions": records[1].fusions,
            },
            flush=True,
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("rank", [0, 127])
@pytest.mark.parametrize("reshard_after_forward", [False, True])
def test_fsdp128_preserves_gradients_refill_and_collectives(
    rank, reshard_after_forward
):
    result = subprocess.run(
        [sys.executable, __file__, str(rank), str(int(reshard_after_forward))],
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_training_peak_memory_does_not_exceed_native(_operands):
    x, w = _operands
    input_TD = x.clone().requires_grad_()
    weight_ED = w.clone().requires_grad_()
    weight_ED.grad_dtype = torch.float32
    coefficient_TE = torch.randn(4096, 256, device="cuda")
    peaks = []
    for function in (
        _HiMidLoLinearFunction.apply,
        fusion.FusedDSv3RouterGateFunction.apply,
    ):

        def step():
            input_TD.grad = weight_ED.grad = None
            output_TE = function(input_TD, weight_ED, 3)
            (output_TE * coefficient_TE).sum().backward()

        for _ in range(12):
            step()
        torch.cuda.synchronize()
        input_TD.grad = weight_ED.grad = None
        gc.collect()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(12):
            step()
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated() - baseline)
    assert peaks[1] <= peaks[0], peaks


if __name__ == "__main__":
    _exercise(int(sys.argv[1]), bool(int(sys.argv[2])))

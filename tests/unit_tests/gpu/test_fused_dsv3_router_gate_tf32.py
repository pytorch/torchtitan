# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import os
import subprocess
import sys
from dataclasses import replace
from functools import partial

import pytest
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecisionPolicy
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves
from torch.utils.checkpoint import checkpoint

from torchtitan_recipes.overrides import fused_dsv3_router_gate_tf32 as fusion

from tests.unit_tests.cpu.test_fused_dsv3_router_gate_tf32 import _config


pytestmark = pytest.mark.skipif(
    fusion.RouterGateLinear is None
    or fusion._NATIVE_MM is None
    or fusion._CUTEDSL_IMPORT_ERROR is not None
    or torch.__version__ != "2.15.0a0+giteecfcac"
    or not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 3)
    or os.environ.get("NVIDIA_TF32_OVERRIDE") == "0",
    reason="Pinned fbpackage TF32 gate on GB300",
)


@pytest.fixture(autouse=True)
def _deterministic():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


def _operands(seed=42):
    torch.manual_seed(seed)
    activation = torch.randn((4096, 7168), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((256, 7168), device="cuda", dtype=torch.bfloat16) / 64
    gradient = torch.randn((4096, 256), device="cuda", dtype=torch.float32)
    return gradient, weight, activation


def bitwise_differences(expected, actual):
    assert expected.shape == actual.shape and expected.dtype == actual.dtype
    return int(
        (
            expected.contiguous().view(torch.uint8)
            != actual.contiguous().view(torch.uint8)
        ).sum()
    )


def native_pair(gradient, weight, activation):
    activation = activation.detach().requires_grad_()
    weight = weight.detach().requires_grad_()
    logits = fusion._RouterGateLinearFunction.apply(
        activation, weight, "bf16", torch.float32, torch.float32, "tf32", torch.float32
    )
    return torch.autograd.grad(logits, (activation, weight), gradient)


def fused_pair(gradient, weight, activation):
    return (fusion.dgrad_op(gradient, weight), fusion.wgrad_op(gradient, activation))


def exact_pair(expected, actual):
    assert len(expected) == len(actual)
    differences = [
        bitwise_differences(left, right) for left, right in zip(expected, actual)
    ]
    assert differences == [0] * len(expected), differences


@pytest.mark.parametrize("seed", [0, 42, 20261009])
def test_native_function_parity_and_repeatability(seed):
    values = _operands(seed)
    expected = native_pair(*values)
    exact_pair(expected, native_pair(*values))
    exact_pair(expected, fused_pair(*values))
    assert all((torch.count_nonzero(value) > 0 for value in expected))


@pytest.mark.parametrize(
    "pattern",
    [
        "cancellation",
        "magnitudes",
        "tf32_ties",
        "bf16_ties",
        "signed_zero",
        "subnormal",
        "nonfinite",
        "nonfinite_inputs",
        "bf16_subnormal_inputs",
    ],
)
def test_numeric_boundaries(pattern):
    gradient, weight, activation = _operands()
    if pattern == "cancellation":
        weight[1::2].copy_(weight[::2])
        activation[1::2].copy_(activation[::2])
        gradient[:, 1::2] = -gradient[:, ::2]
        gradient[1::2] = -gradient[::2]
        gradient[::16, ::16] *= 1.0009765625
    elif pattern == "magnitudes":
        for value in (gradient, weight, activation):
            exponent = torch.randint(-60, 61, value.shape, device="cuda")
            value.copy_(torch.ldexp(value.float(), exponent))
    elif pattern == "tf32_ties":
        bits = torch.tensor(
            [
                1065357311,
                1065357312,
                1065357313,
                1065365503,
                1065365504,
                1065365505,
                -1082126336,
                -1082118144,
            ],
            dtype=torch.int32,
            device="cuda",
        ).view(torch.float32)
        gradient.copy_(bits.repeat(gradient.numel() // len(bits)).view_as(gradient))
    elif pattern == "bf16_ties":
        activation.zero_()
        activation[0].fill_(1)
        weight.zero_()
        weight[0].fill_(1)
        gradient.zero_()
        midpoint = torch.tensor(
            [1.00390625, 1.01171875, -1.00390625, -1.01171875], device="cuda"
        )
        below = torch.nextafter(midpoint, torch.full_like(midpoint, -torch.inf))
        above = torch.nextafter(midpoint, torch.full_like(midpoint, torch.inf))
        neighbors = torch.cat((below, midpoint, above))
        gradient[:, 0] = neighbors.repeat(342)[:4096]
        gradient[0] = neighbors.repeat(22)[:256]
    elif pattern == "signed_zero":
        gradient.zero_()
        gradient[:, ::2] = -0.0
    elif pattern == "subnormal":
        bits = torch.tensor(
            [
                1,
                4096,
                4194304,
                8388607,
                8388608,
                8392704,
                -2147483647,
                -2147479552,
                -2143289344,
                -2139095041,
                -2139095040,
                -2139090944,
            ],
            dtype=torch.int32,
            device="cuda",
        ).view(torch.float32)
        gradient.copy_(
            bits.repeat((gradient.numel() + 11) // 12)[: gradient.numel()].view_as(
                gradient
            )
        )
        activation.fill_(1)
        weight.fill_(1)
    elif pattern == "nonfinite":
        gradient[::8, 0] = torch.inf
        gradient[1::8, 0] = -torch.inf
        gradient[2::8, 0] = torch.nan
        gradient.view(torch.int32)[3::8, 0] = 2141266757
    elif pattern == "nonfinite_inputs":
        activation.view(torch.int16)[::8, ::8] = 32705
        weight.view(torch.int16)[::8, ::8] = -63
        gradient.view(torch.int32)[::8, ::8] = 2141266757
        activation[1::8, 1::8] = torch.inf
        weight[1::8, 1::8] = -torch.inf
    elif pattern == "bf16_subnormal_inputs":
        bits = torch.tensor(
            [1, 2, 127, 128, -32767, -32766, -32641, -32640],
            dtype=torch.int16,
            device="cuda",
        )
        for value in (activation, weight):
            value.view(torch.int16).copy_(
                bits.repeat(value.numel() // 8).view_as(value)
            )
        gradient[:, ::2] *= -1
    values = (gradient, weight, activation)
    exact_pair(native_pair(*values), fused_pair(*values))


class Operations(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.names = []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        self.names.append(str(function))
        return function(*args, **kwargs or {})


def models(config=None, *, dtype=torch.bfloat16):
    config = _config() if config is None else config
    native = config.build().to(device="cuda", dtype=dtype)
    fused = (
        fusion.fused_dsv3_router_gate_tf32(config, experimental=True)
        .build()
        .to(device="cuda", dtype=dtype)
    )
    with torch.no_grad():
        native.weight.normal_(std=0.015625)
    fused.load_state_dict(native.state_dict())
    return (native, fused)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_module_forward_gradients_accumulation_and_dispatch(dtype):
    torch.manual_seed(42)
    gates = models(dtype=dtype)
    _, _, values = _operands()
    values = values.view(1, 4096, 7168)
    upstreams = [torch.randn((1, 4096, 256), device="cuda") for _ in range(3)]
    previous_weight_grad = torch.randn_like(gates[0].weight)
    previous_activation_grad = torch.randn_like(values)
    snapshots = []
    for gate in gates:
        activation = values.clone().requires_grad_()
        activation.grad = previous_activation_grad.clone()
        gate.weight.grad = previous_weight_grad.clone()
        operations = Operations()
        steps = []
        with operations:
            for upstream in upstreams:
                logits = gate(activation)
                logits.backward(upstream)
                steps.extend(
                    [
                        logits.detach().clone(),
                        activation.grad.clone(),
                        gate.weight.grad.clone(),
                    ]
                )
        for suffix in ("dgrad", "wgrad"):
            assert operations.names.count(
                f"torchtitan.dsv3_router_gate_tf32_{suffix}.default"
            ) == (3 if gate is gates[1] else 0)
        snapshots.append(steps)
    exact_pair(*snapshots)
    assert list(gates[0].state_dict()) == list(gates[1].state_dict()) == ["weight"]


@pytest.mark.parametrize(
    "need_input,need_weight", [(False, True), (True, False), (True, True)]
)
def test_partial_gradients(need_input, need_weight):
    gradient, weight, activation = _operands()
    results = []
    for function in (
        fusion._RouterGateLinearFunction,
        fusion.FusedDSv3TF32RouterGateFunction,
    ):
        x = activation.detach().requires_grad_(need_input)
        w = weight.detach().requires_grad_(need_weight)
        logits = function.apply(
            x, w, "bf16", torch.float32, torch.float32, "tf32", torch.float32
        )
        logits.backward(gradient)
        assert (x.grad is None) is (not need_input)
        assert (w.grad is None) is (not need_weight)
        results.append([value for value in (x.grad, w.grad) if value is not None])
    exact_pair(*results)


@pytest.mark.parametrize("layout", ["strided", "transposed", "expanded", "unaligned"])
def test_native_fallback_for_gradient_layout(layout):
    gradient, weight, activation = _operands()
    if layout == "strided":
        storage = gradient.new_empty((4096, 512))
        storage[:, ::2].copy_(gradient)
        gradient = storage[:, ::2]
    elif layout == "transposed":
        gradient = gradient.t().contiguous().t()
    elif layout == "expanded":
        gradient = gradient[:1].expand(4096, -1)
    elif layout == "unaligned":
        storage = gradient.new_empty(gradient.numel() + 1)
        storage[1:].view_as(gradient).copy_(gradient)
        gradient = storage[1:].view_as(gradient)
    assert not fusion._supports_gate_backward(gradient, weight, wgrad=False)
    expected = native_pair(gradient, weight, activation)
    activation.requires_grad_()
    weight.requires_grad_()
    operations = Operations()
    with operations:
        logits = fusion.FusedDSv3TF32RouterGateFunction.apply(
            activation,
            weight,
            "bf16",
            torch.float32,
            torch.float32,
            "tf32",
            torch.float32,
        )
        actual = torch.autograd.grad(logits, (activation, weight), gradient)
    exact_pair(expected, actual)
    assert not any(
        ("torchtitan.dsv3_router_gate_tf32_" in name for name in operations.names)
    )


def test_native_fallback_for_policy_and_shape():
    from torchtitan.models.common.router_gate import RouterGemmConfig

    config = replace(
        _config(),
        backward=RouterGemmConfig(
            input_dtype=torch.float32, compute_mode="fp32", output_dtype=torch.float32
        ),
    )
    gates = models(config)
    _, _, activation = _operands()
    gradient = torch.randn((4096, 256), device="cuda")
    results = []
    for gate in gates:
        x = activation.detach().requires_grad_()
        calls = Operations()
        with calls:
            logits = gate(x)
            results.append(torch.autograd.grad(logits, (x, gate.weight), gradient))
        assert not any(
            ("torchtitan.dsv3_router_gate_tf32_" in name for name in calls.names)
        )
    exact_pair(*results)
    assert not fusion._supports_gate_backward(
        gradient[:32], gates[0].weight, wgrad=False
    )


def test_hooks_preserve_module_boundary():
    gates = models()
    _, _, activation = _operands()
    gradient = torch.randn((4096, 256), device="cuda")
    calls = []
    forward_handle = gates[1].register_forward_hook(
        lambda module, args, out: calls.append("forward")
    )
    backward_handle = gates[1].register_full_backward_hook(
        lambda *args: calls.append("backward")
    )
    try:
        results = []
        for gate in gates:
            x = activation.detach().requires_grad_()
            logits = gate(x)
            results.append(torch.autograd.grad(logits, (x, gate.weight), gradient))
        exact_pair(*results)
        assert calls == ["forward", "backward"]
    finally:
        forward_handle.remove()
        backward_handle.remove()


@pytest.mark.parametrize("score_gradient", [False, True])
def test_actual_learned_router_with_debug_balancing_disabled(score_gradient):
    from torchtitan.models.common.moe import TokenChoiceTopKRouter

    config = TokenChoiceTopKRouter.Config(
        num_experts=256,
        gate=_config(),
        num_expert_groups=8,
        num_limited_groups=4,
        top_k=8,
        score_func="sigmoid",
        route_norm=True,
        route_scale=2.5,
        _debug_force_load_balance=False,
    )
    native = config.build().to(device="cuda", dtype=torch.bfloat16)
    candidate = (
        replace(
            config,
            gate=fusion.fused_dsv3_router_gate_tf32(config.gate, experimental=True),
        )
        .build()
        .to(device="cuda", dtype=torch.bfloat16)
    )
    with torch.no_grad():
        native.gate.weight.normal_(std=0.015625)
    candidate.load_state_dict(native.state_dict())
    _, _, activation = _operands(9191)
    activation = activation.view(1, 4096, 7168)
    bias = torch.randn((256,), device="cuda") * 0.02
    grad_weights = torch.randn((1, 4096, 8), device="cuda") * 0.003
    grad_scores = torch.randn((1, 4096, 256), device="cuda") * 0.001
    snapshots = []
    indices = []
    for router in (native, candidate):
        x = activation.detach().requires_grad_()
        weights, ids, scores = router(x, bias)
        differentiated = (weights, scores) if score_gradient else (weights,)
        upstream = (grad_weights, grad_scores) if score_gradient else (grad_weights,)
        calls = Operations()
        with calls:
            gradients = torch.autograd.grad(
                differentiated, (x, router.gate.weight), upstream
            )
        assert all((torch.count_nonzero(value) > 0 for value in gradients))
        assert calls.names.count("torchtitan.dsv3_router_gate_tf32_wgrad.default") == (
            1 if router is candidate else 0
        )
        snapshots.append((weights, scores, *gradients))
        indices.append(ids)
    exact_pair(*snapshots)
    assert torch.equal(*indices)


def test_cuda_graph_replay():
    gradient, weight, activation = _operands()
    fused_pair(gradient, weight, activation)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = fused_pair(gradient, weight, activation)
    for _ in range(10):
        gradient.normal_()
        graph.replay()
        exact_pair(native_pair(gradient, weight, activation), actual)


@pytest.mark.parametrize("unaligned", [False, True])
def test_aot_autograd_compilation(unaligned):
    native, candidate = models()
    _, _, activation = _operands()
    gradient = torch.randn((4096, 256), device="cuda")
    if unaligned:
        storage = gradient.new_empty(gradient.numel() + 1)
        storage[1:].view_as(gradient).copy_(gradient)
        gradient = storage[1:].view_as(gradient)
    compiled = torch.compile(candidate, backend="aot_eager", fullgraph=True)
    results = []
    for gate in (native, compiled):
        x = activation.detach().requires_grad_()
        logits = gate(x)
        results.append(
            (logits, *torch.autograd.grad(logits, (x, gate.weight), gradient))
        )
    exact_pair(*results)


def test_checkpoint_and_saved_tensor_ownership():
    gradient, weight, activation = _operands()
    snapshots = []
    for function in (
        fusion._RouterGateLinearFunction,
        fusion.FusedDSv3TF32RouterGateFunction,
    ):
        for recompute in (False, True):
            x = activation.detach().requires_grad_()
            w = weight.detach().requires_grad_()
            saved = []

            def pack(tensor):
                saved.append(tensor.data_ptr())
                return tensor

            arguments = (
                x,
                w,
                "bf16",
                torch.float32,
                torch.float32,
                "tf32",
                torch.float32,
            )
            with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
                output = (
                    checkpoint(function.apply, *arguments, use_reentrant=False)
                    if recompute
                    else function.apply(*arguments)
                )
            snapshots.append((output, *torch.autograd.grad(output, (x, w), gradient)))
            if not recompute:
                assert saved == [x.data_ptr(), w.data_ptr()]
    for result in snapshots[1:]:
        exact_pair(snapshots[0], result)


def test_peak_training_memory_does_not_exceed_native():
    gradient, weight, activation = _operands()
    x = activation.requires_grad_()
    w = weight.requires_grad_()
    peaks = []
    for function in (
        fusion._RouterGateLinearFunction,
        fusion.FusedDSv3TF32RouterGateFunction,
    ):

        def step():
            x.grad = w.grad = None
            output = function.apply(
                x, w, "bf16", torch.float32, torch.float32, "tf32", torch.float32
            )
            (output * gradient).sum().backward()

        for _ in range(12):
            step()
        torch.cuda.synchronize()
        x.grad = w.grad = None
        gc.collect()
        torch.cuda.empty_cache()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(12):
            step()
        torch.cuda.synchronize()
        peaks.append(torch.cuda.max_memory_allocated() - baseline)
    assert peaks[1] <= peaks[0], peaks


class _ZeroCollectives(Operations):
    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        result = super().__torch_dispatch__(function, types, args, kwargs)
        name = str(function)
        if name.startswith("c10d.") and (
            "allgather" in name.replace("_", "") or "reduce_scatter" in name
        ):
            for value in tree_leaves(args[0]):
                if isinstance(value, torch.Tensor):
                    value.zero_()
        return result


def _exercise_fsdp(rank, reshard_after_forward):
    from torchtitan.distributed.fsdp import _fully_shard_configured

    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    dist.init_process_group("fake", rank=rank, world_size=128, store=FakeStore())
    try:
        mesh = init_device_mesh("cuda", (128,))
        gates = []
        for fused in (False, True):
            config = _config()
            config.param_init = {"weight": partial(torch.nn.init.normal_, std=0.015625)}
            if fused:
                config = fusion.fused_dsv3_router_gate_tf32(config, experimental=True)
            with torch.device("meta"):
                model = config.build()
            _fully_shard_configured(
                model,
                reshard_after_forward=reshard_after_forward,
                mesh=mesh,
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16
                ),
            )
            model.to_empty(device="cuda")
            torch.manual_seed(42)
            model.init_states(buffer_device=torch.device("cuda"))
            gates.append(model)
        activation = torch.randn((1, 4096, 7168), device="cuda", dtype=torch.bfloat16)
        upstreams = [torch.randn((1, 4096, 256), device="cuda") / 64 for _ in range(3)]
        records, snapshots = [], []
        for model in gates:
            operations = _ZeroCollectives()
            values = []
            with operations:
                for index, upstream in enumerate(upstreams):
                    model.set_requires_gradient_sync(index == 2)
                    x = activation.clone().requires_grad_()
                    output = model(x)
                    output.backward(upstream)
                    values.extend((output.detach().clone(), x.grad.clone()))
            grad_weight = model.weight.grad.to_local()
            assert grad_weight.dtype == model.weight.dtype == torch.float32
            values.append(grad_weight.detach().clone())
            for phase in ("dgrad", "wgrad"):
                assert operations.names.count(
                    f"torchtitan.dsv3_router_gate_tf32_{phase}.default"
                ) == (3 if model is gates[1] else 0)
            collectives = [name for name in operations.names if "c10d" in name]
            assert any("allgather" in name.replace("_", "") for name in collectives)
            assert any("reduce_scatter" in name for name in collectives)
            state = model.state_dict()
            assert list(state) == ["weight"]
            model.load_state_dict(state)
            snapshots.append(values)
            records.append(collectives)
        exact_pair(*snapshots)
        assert records[0] == records[1]
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("rank", [0, 127])
@pytest.mark.parametrize("reshard_after_forward", [False, True])
def test_fsdp128_preserves_gradients_collectives_and_checkpoint(
    rank, reshard_after_forward
):
    result = subprocess.run(
        [sys.executable, __file__, str(rank), str(int(reshard_after_forward))],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _exercise_fsdp(int(sys.argv[1]), bool(int(sys.argv[2])))

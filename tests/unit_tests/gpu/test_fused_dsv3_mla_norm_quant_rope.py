# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Production-shape byte parity, autograd, tracing, and FSDP lifetime checks."""

import subprocess
import sys
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.testing._internal.distributed.fake_pg import FakeStore
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves
from torch.utils.checkpoint import checkpoint

from torchtitan.config import derive
from torchtitan.distributed.fsdp import linear_param_shard_placements
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.nn_modules import RMSNorm
from torchtitan.models.common.rope import ComplexRoPE
from torchtitan.models.deepseek_v3.model import Attention
from torchtitan.quantization._fsdp_tensor import _UnshardedFSDPTensor
from torchtitan.quantization.mxfp8.linear import MXFP8Linear
from torchtitan_recipes.overrides import fused_dsv3_mla_norm_quant_rope as fusion
from torchtitan_recipes.overrides.fused_mla import (
    fused_mla_kv,
    fused_mla_q,
    FusedMLAAttention,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (10, 3)
    or torch.__version__ != "2.16.0.dev20261007+cu130",
    reason="Recorded GB300/PyTorch MLA specialization",
)


@pytest.fixture(autouse=True)
def _determinism():
    deterministic = torch.are_deterministic_algorithms_enabled()
    fill = torch.utils.deterministic.fill_uninitialized_memory
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    torch.manual_seed(42)
    yield
    torch.use_deterministic_algorithms(deterministic)
    torch.utils.deterministic.fill_uninitialized_memory = fill


def _assert_bits(expected, actual, *, strides=False):
    if isinstance(expected, (tuple, list)):
        assert len(expected) == len(actual)
        for reference, candidate in zip(expected, actual):
            _assert_bits(reference, candidate, strides=strides)
        return
    if expected is None or actual is None:
        assert expected is actual
        return
    assert (expected.shape, expected.dtype) == (actual.shape, actual.dtype)
    if strides:
        assert expected.stride() == actual.stride()
    expected_bytes = (
        expected.detach().resolve_neg().contiguous().reshape(-1).view(torch.uint8)
    )
    actual_bytes = (
        actual.detach().resolve_neg().contiguous().reshape(-1).view(torch.uint8)
    )
    assert torch.equal(
        expected_bytes, actual_bytes
    ), f"{int((expected_bytes != actual_bytes).sum())} different bytes in {tuple(expected.shape)} {expected.dtype}"


def _prepare_weights(module, *, prepared=True, grad_dtype=None):
    for child in module.modules():
        if isinstance(child, MXFP8Linear):
            weight = child.weight
            with torch.no_grad():
                weight._tensor.normal_(0, 0.02)
                child.weight = nn.Parameter(
                    _UnshardedFSDPTensor(
                        weight._tensor, weight._build_operands(weight._tensor)
                    )
                    if prepared
                    else weight._tensor
                )
            if grad_dtype is not None:
                child.weight.grad_dtype = grad_dtype


class _Chain(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_norm = RMSNorm.Config(normalized_shape=1536).build()
        self.kv_norm = RMSNorm.Config(normalized_shape=512).build()
        self.wq_b = MXFP8Linear.Config(
            in_features=1536,
            out_features=24576,
            input_activation_format_for_backward="mxfp8",
        ).build()
        self.wkv_b = MXFP8Linear.Config(
            in_features=512,
            out_features=32768,
            input_activation_format_for_backward="mxfp8",
        ).build()
        self.rope = ComplexRoPE.Config(dim=64, max_context_length=4096).build()


def _make_chain(*, prepared=True, grad_dtype=None):
    torch.manual_seed(42)
    with torch.device("cuda"):
        module = _Chain().bfloat16()
    module.rope.init_states(buffer_device=torch.device("cuda"))
    _prepare_weights(module, prepared=prepared, grad_dtype=grad_dtype)
    with torch.no_grad():
        module.q_norm.weight.uniform_(0.5, 1.5)
        module.kv_norm.weight.uniform_(0.5, 1.5)
    return module


@pytest.fixture(scope="module")
def chain():
    return _make_chain()


def _positions(kind="random32"):
    dtype = torch.int32 if kind.endswith("32") else torch.int64
    if kind.startswith("random"):
        return torch.randint(4096, (1, 4096), device="cuda", dtype=dtype)
    return torch.arange(4096, device="cuda", dtype=dtype).unsqueeze(0)


def _native_chain(module, value, positions, which):
    if which == "q":
        projected = module.wq_b(module.q_norm(value)).view(1, 4096, 128, 192)
        return (fused_mla_q(projected, module.rope.cache, positions, 128),)
    latent, positional = value.split((512, 64), dim=-1)
    projected = module.wkv_b(module.kv_norm(latent)).view(1, 4096, 128, 256)
    return fused_mla_kv(projected, positional, module.rope.cache, positions, 128)


def _fused_chain(module, value, positions, which, stages=fusion._STAGES):
    norm, projection, function = (
        (module.q_norm, module.wq_b, fusion.FusedMLAQChainFunction)
        if which == "q"
        else (module.kv_norm, module.wkv_b, fusion.FusedMLAKVChainFunction)
    )
    result = function.apply(
        *fusion._chain_inputs(
            norm, projection, value, module.rope.cache, positions, stages
        )
    )
    return (result,) if which == "q" else result


@pytest.mark.parametrize("which", ["q", "kv"])
@pytest.mark.parametrize("stages", [(), fusion._STAGES])
@pytest.mark.parametrize("positions_kind", ["arange32", "random64"])
def test_chain_bitwise(chain, which, stages, positions_kind):
    value = torch.randn(
        1, 4096, 1536 if which == "q" else 576, device="cuda", dtype=torch.bfloat16
    )
    positions = _positions(positions_kind)
    norm, projection = (
        (chain.q_norm, chain.wq_b) if which == "q" else (chain.kv_norm, chain.wkv_b)
    )
    gradients = (torch.randn(1, 4096, 128, 192, device="cuda", dtype=torch.bfloat16),)
    if which == "kv":
        gradients += (
            torch.randn(1, 4096, 128, 128, device="cuda", dtype=torch.bfloat16),
        )
    results = []
    for fused in (False, True):
        x = value.clone().requires_grad_()
        outputs = (
            _fused_chain(chain, x, positions, which, stages)
            if fused
            else _native_chain(chain, x, positions, which)
        )
        grads = torch.autograd.grad(
            outputs,
            (x, norm.weight, projection.weight),
            tuple(g.clone() for g in gradients),
        )
        results.append((outputs, grads))
    _assert_bits(*results)
    for native, fused in zip(results[0][0], results[1][0]):
        assert native.stride() == fused.stride()


@pytest.mark.parametrize("which", ["q", "kv"])
@pytest.mark.parametrize("trainable", ["input", "norm", "projection"])
def test_partial_gradients(which, trainable):
    module = _make_chain()
    norm, projection = (
        (module.q_norm, module.wq_b) if which == "q" else (module.kv_norm, module.wkv_b)
    )
    norm.weight.requires_grad_(trainable == "norm")
    projection.weight.requires_grad_(trainable == "projection")
    value = torch.randn(
        1, 4096, 1536 if which == "q" else 576, device="cuda", dtype=torch.bfloat16
    )
    positions = _positions()
    results = []
    for fused in (False, True):
        x = value.clone().requires_grad_(trainable == "input")
        target = {"input": x, "norm": norm.weight, "projection": projection.weight}[
            trainable
        ]
        outputs = (
            _fused_chain(module, x, positions, which)
            if fused
            else _native_chain(module, x, positions, which)
        )
        gradients = torch.autograd.grad(
            outputs, target, tuple(torch.full_like(out, 0.001) for out in outputs)
        )
        results.append((outputs, gradients))
    _assert_bits(*results)


@pytest.mark.parametrize("output_index", [0, 1])
def test_kv_backward_with_one_output_unused(chain, output_index):
    value = torch.randn(1, 4096, 576, device="cuda", dtype=torch.bfloat16)
    positions = _positions()
    results = []
    for fused in (False, True):
        x = value.clone().requires_grad_()
        outputs = (
            _fused_chain(chain, x, positions, "kv")
            if fused
            else _native_chain(chain, x, positions, "kv")
        )
        output = outputs[output_index]
        gradients = torch.autograd.grad(
            output,
            (x, chain.kv_norm.weight, chain.wkv_b.weight),
            torch.full_like(output, 0.001),
        )
        results.append((output, gradients))
    _assert_bits(*results)


@pytest.mark.parametrize("which", ["q", "kv"])
@pytest.mark.parametrize("case", ["normal", "zeros", "huge", "tiny", "nonfinite"])
@pytest.mark.parametrize("colwise", [False, True])
def test_norm_and_quantization_all_bytes(chain, which, case, colwise):
    width = 1536 if which == "q" else 576
    value = torch.randn(1, 4096, width, device="cuda")
    if case == "huge":
        value *= 2.0**60
    elif case == "tiny":
        value *= 2.0**-100
    value = value.bfloat16()
    if case == "zeros":
        value[:, ::2] = 0.0
        value[:, 1::2] = -0.0
    elif case == "nonfinite":
        value[:, ::17, :5] = torch.tensor(
            [float("nan"), float("inf"), -float("inf"), 0.0, -0.0],
            device="cuda",
            dtype=torch.bfloat16,
        )
    norm = chain.q_norm if which == "q" else chain.kv_norm
    if which == "kv":
        value = value[..., :512]
        assert value.stride(1) == 576
    normalized, rstd = torch.ops.aten._fused_rms_norm(
        value.contiguous(), [value.shape[-1]], norm.weight, norm.eps
    )
    expected = (rstd, *fusion._quantize(normalized.flatten(0, 1), colwise=colwise))
    actual = fusion.norm_quant_op(value, norm.weight, norm.eps, colwise)
    _assert_bits(expected, actual, strides=True)


@pytest.mark.parametrize("which", ["q", "kv"])
@pytest.mark.parametrize("case", ["normal", "tiny", "nonfinite", "expanded", "strided"])
def test_gradient_quantization_all_bytes(chain, which, case):
    positions = _positions()
    cache = torch.view_as_real(chain.rope.cache).contiguous()
    shape = (1, 4096, 128, 192)
    gradient = torch.randn(
        shape if case != "strided" else (*shape[:-1], 384),
        device="cuda",
        dtype=torch.bfloat16,
    )
    if case == "tiny":
        gradient *= 1e-38
    elif case == "nonfinite":
        gradient[..., ::11] = float("nan")
        gradient[..., 1::17] = float("inf")
        gradient[..., 2::17] = -0.0
    elif case == "expanded":
        gradient = gradient[:, :1, :1].expand(shape)
    elif case == "strided":
        gradient = gradient[..., :192]
    before = gradient.clone()
    if which == "q":
        native = gradient.clone(memory_format=torch.contiguous_format)
        fusion._fused_mla_q_rope_op(native, cache, positions, 128, True)
        expected = fusion._quantize(native.flatten(2).squeeze(0))
        actual = fusion.q_backward_quant_op(gradient, cache, positions)
    else:
        grad_v = torch.randn(1, 4096, 128, 256, device="cuda", dtype=torch.bfloat16)[
            ..., 128:
        ]
        packed, positional = fusion._fused_mla_kv_backward_op(
            gradient, grad_v, cache, positions, 128, 64
        )
        expected = (*fusion._quantize(packed.flatten(2).squeeze(0)), positional)
        actual = fusion.kv_backward_quant_op(gradient, grad_v, cache, positions)
    _assert_bits(expected, actual, strides=True)
    _assert_bits(before, gradient)


@pytest.mark.parametrize("which", ["q", "kv"])
@pytest.mark.parametrize("grad_dtype", [torch.bfloat16, torch.float32])
def test_gradient_accumulation_and_checkpoint(which, grad_dtype):
    module = _make_chain(grad_dtype=grad_dtype)
    norm, projection = (
        (module.q_norm, module.wq_b) if which == "q" else (module.kv_norm, module.wkv_b)
    )
    value = torch.randn(
        1, 4096, 1536 if which == "q" else 576, device="cuda", dtype=torch.bfloat16
    )
    positions = _positions()
    results = []
    for fused in (False, True):
        module.zero_grad(set_to_none=True)
        x = value.clone().requires_grad_()
        for _ in range(2):
            fn = (
                (lambda v: _fused_chain(module, v, positions, which))
                if fused
                else (lambda v: _native_chain(module, v, positions, which))
            )
            outputs = checkpoint(fn, x, use_reentrant=False)
            torch.autograd.backward(
                outputs, tuple(torch.full_like(out, 0.001) for out in outputs)
            )
        results.append(
            (x.grad.clone(), norm.weight.grad.clone(), projection.weight.grad.clone())
        )
        assert projection.weight.grad.dtype == grad_dtype
    _assert_bits(*results)


class _DeterministicAttention(nn.Module):
    def forward(self, q, k, v, **kwargs):
        return v * (q[..., :128] + k[..., 128:192].repeat(1, 1, 2))


def _attention_config():
    def linear(input_width, output_width):
        return MXFP8Linear.Config(
            in_features=input_width,
            out_features=output_width,
            input_activation_format_for_backward="mxfp8",
        )

    return Attention.Config(
        n_heads=128,
        dim=7168,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        wq_a=linear(7168, 1536),
        wq_b=linear(1536, 24576),
        wkv_a=linear(7168, 576),
        wkv_b=linear(512, 32768),
        wo=Linear.Config(in_features=16384, out_features=7168),
        q_norm=RMSNorm.Config(normalized_shape=1536),
        kv_norm=RMSNorm.Config(normalized_shape=512),
        rope=ComplexRoPE.Config(dim=64, max_context_length=4096),
    )


def _make_attention(fused):
    torch.manual_seed(42)
    cfg = _attention_config()
    cfg = (
        fusion.fused_mla_norm_quant_rope(cfg, experimental=True)
        if fused
        else derive(cfg, FusedMLAAttention.Config)
    )
    with torch.device("cuda"):
        module = cfg.build().bfloat16()
    module.init_states(buffer_device=torch.device("cuda"))
    _prepare_weights(module)
    module.inner_attention = _DeterministicAttention()
    return module


@pytest.mark.parametrize("explicit_positions", [False, True])
def test_attention_override_preserves_outputs_gradients_and_state(explicit_positions):
    x = torch.randn(4096, 7168, device="cuda", dtype=torch.bfloat16)
    dy = torch.randn_like(x) * 0.001
    positions = _positions().squeeze(0) if explicit_positions else None
    results = []
    states = []
    for fused in (False, True):
        module = _make_attention(fused)
        value = x.clone().requires_grad_()
        with patch.object(
            fusion.FusedMLAQChainFunction,
            "apply",
            wraps=fusion.FusedMLAQChainFunction.apply,
        ) as q_call, patch.object(
            fusion.FusedMLAKVChainFunction,
            "apply",
            wraps=fusion.FusedMLAKVChainFunction.apply,
        ) as kv_call:
            output = module(value, None, positions)
            grads = torch.autograd.grad(output, (value, *module.parameters()), dy)
            assert (q_call.call_count, kv_call.call_count) == (
                (1, 1) if fused else (0, 0)
            )
        results.append((output.detach(), grads))
        states.append({k: (v.shape, v.dtype) for k, v in module.state_dict().items()})
    _assert_bits(*results)
    assert states[0] == states[1]


def test_unsupported_save_policy_and_hooks_retain_native_calls():
    from torch.backends import python_native

    module = _make_attention(True)
    x = torch.randn(4096, 7168, device="cuda", dtype=torch.bfloat16)
    assert module._supports_chain_fusion(x)
    with patch("torch.are_deterministic_algorithms_enabled", return_value=False):
        assert not module._supports_chain_fusion(x)
    with python_native.cutedsl.disabled():
        assert not module._supports_chain_fusion(x)
    with python_native.operations_disabled("_fused_rms_norm"):
        assert not module._supports_chain_fusion(x)
    module.wq_b.input_activation_format_for_backward = "bf16"
    assert not module._supports_chain_fusion(x)
    module.wq_b.input_activation_format_for_backward = "mxfp8"
    calls = []
    hook = module.q_norm.register_forward_hook(lambda *args: calls.append("q_norm"))
    assert not module._supports_chain_fusion(x)
    with torch.no_grad(), patch.object(
        fusion.FusedMLAQChainFunction,
        "apply",
        side_effect=AssertionError("native child hook must run"),
    ):
        module(x, None)
    hook.remove()
    assert calls == ["q_norm"]


def test_attention_override_compiles_without_graph_breaks():
    module = _make_attention(True)
    value = torch.randn(
        4096, 7168, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    gradient = torch.full_like(value, 0.001)
    expected = module(value, None)
    expected_grad = torch.autograd.grad(
        expected, (value, *module.parameters()), gradient
    )
    # Exercise the complete module's Dynamo/AOT boundary. Kernel-level
    # Inductor parity is checked independently below; inner attention here
    # is a deterministic stand-in rather than a production attention backend.
    compiled = torch.compile(module, fullgraph=True, backend="aot_eager")
    actual = compiled(value, None)
    actual_grad = torch.autograd.grad(actual, (value, *module.parameters()), gradient)
    _assert_bits(expected, actual)
    _assert_bits(expected_grad, actual_grad)


def test_fake_trace_preserves_operator_boundaries(chain):
    from torch.fx.experimental.proxy_tensor import make_fx

    value = torch.randn(1, 4096, 1536, device="cuda", dtype=torch.bfloat16)
    positions = _positions()
    traced = make_fx(
        lambda x: _fused_chain(chain, x, positions, "q"),
        tracing_mode="fake",
        _allow_non_fake_inputs=True,
    )(value)
    operators = [
        str(node.target) for node in traced.graph.nodes if node.op == "call_function"
    ]
    assert "torchtitan.dsv3_mla_norm_quant.default" in operators
    assert "torchtitan.dsv3_mla_q_up_rope.default" in operators
    _assert_bits(_fused_chain(chain, value, positions, "q"), traced(value))


@pytest.mark.parametrize("which", ["q", "kv"])
def test_compile_autograd_and_cuda_graph_replay(which):
    module = _make_chain(prepared=False)
    value = torch.randn(
        1,
        4096,
        1536 if which == "q" else 576,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    positions = _positions()
    norm, projection = (
        (module.q_norm, module.wq_b) if which == "q" else (module.kv_norm, module.wkv_b)
    )

    def step(x):
        return _fused_chain(module, x, positions, which)

    expected = step(value)
    gradient = tuple(torch.full_like(out, 0.001) for out in expected)
    expected_grad = torch.autograd.grad(
        expected,
        (value, norm.weight, projection.weight),
        tuple(g.clone() for g in gradient),
    )
    compiled = torch.compile(step, fullgraph=True)
    actual = compiled(value)
    actual_grad = torch.autograd.grad(
        actual,
        (value, norm.weight, projection.weight),
        tuple(g.clone() for g in gradient),
    )
    _assert_bits(expected, actual)
    _assert_bits(expected_grad, actual_grad)
    # Fresh leaves and warmup on the capture stream avoid retaining the
    # default-stream AccumulateGrad nodes from the compilation checks.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        module = _make_chain(prepared=False)
        value = value.detach().clone().requires_grad_()
        norm, projection = (
            (module.q_norm, module.wq_b)
            if which == "q"
            else (module.kv_norm, module.wkv_b)
        )
        for _ in range(3):
            output = step(value)
            torch.autograd.grad(
                output, (value, norm.weight, projection.weight), gradient
            )
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        output = step(value)
        grads = torch.autograd.grad(
            output, (value, norm.weight, projection.weight), gradient
        )
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    _assert_bits(expected, output)
    _assert_bits(expected_grad, grads)


class _Collectives(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.collectives = []
        self.fusions = []

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        name = str(function)
        result = function(*args, **(kwargs or {}))
        if "dsv3_mla_" in name:
            self.fusions.append(name)
        if name.startswith("c10d.") and any(
            kind in name.replace("_", "") for kind in ("allgather", "reducescatter")
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
            for tensor in tree_leaves(result):
                if isinstance(tensor, torch.Tensor):
                    tensor.zero_()
        return result


def _exercise_fsdp(rank):
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    dist.init_process_group("fake", store=FakeStore(), rank=rank, world_size=128)
    try:
        mesh = init_device_mesh("cuda", (128,), mesh_dim_names=("dp_shard",))
        modules = []
        for fused in (False, True):
            cfg = _attention_config()
            cfg = (
                fusion.fused_mla_norm_quant_rope(cfg, experimental=True)
                if fused
                else derive(cfg, FusedMLAAttention.Config)
            )
            with torch.device("meta"):
                module = cfg.build()
            placements = linear_param_shard_placements(module)
            fully_shard(
                module,
                mesh=mesh,
                shard_placement_fn=placements.get,
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.float32
                ),
                reshard_after_forward=True,
            )
            module.to_empty(device="cuda")
            module.init_states(buffer_device=torch.device("cuda"))
            module.inner_attention = _DeterministicAttention()
            modules.append(module)
        x = torch.randn(4096, 7168, device="cuda", dtype=torch.bfloat16)
        records = [_Collectives(), _Collectives()]
        for synchronize in (True, False, True):
            results = []
            for module, record in zip(modules, records):
                module.set_requires_gradient_sync(synchronize)
                value = x.clone().requires_grad_()
                with record:
                    output = module(value, None)
                    output.backward(torch.full_like(output, 0.001))
                gradients = [
                    p.grad.to_local() if p.grad is not None else None
                    for p in module.parameters()
                ]
                results.append((output.detach(), value.grad, gradients))
            _assert_bits(*results)
            if synchronize:
                for module in modules:
                    module.zero_grad(set_to_none=True)
        assert records[0].collectives == records[1].collectives
        assert records[0].fusions == []
        for operation in (
            "norm_quant",
            "q_up_rope",
            "kv_up_rope",
            "q_backward_quant",
            "kv_backward_quant",
            "norm_backward",
        ):
            assert records[1].fusions.count(
                f"torchtitan.dsv3_mla_{operation}.default"
            ) == (6 if operation in ("norm_quant", "norm_backward") else 3)
        for operation in ("allgather", "reducescatter"):
            assert any(
                operation in name.replace("_", "") for name, _ in records[1].collectives
            )
        assert modules[0].state_dict().keys() == modules[1].state_dict().keys()
        torch.cuda.synchronize()
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("rank", [0, 127])
def test_fsdp128_refill_and_collectives(rank):
    result = subprocess.run(
        [sys.executable, __file__, str(rank)],
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    _exercise_fsdp(int(sys.argv[1]))

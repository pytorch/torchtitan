# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Raw-bit CE, saved state, and peak-memory checks for original node 747954."""

from unittest.mock import patch

import pytest
import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton
import triton.language as tl
from spmd_types.checker import typecheck
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.device_mesh import init_device_mesh
from torch.testing._internal.distributed.fake_pg import FakeStore

from torchtitan.components.loss import ChunkedLossWrapper, cross_entropy_loss
from torchtitan.distributed.spmd_types import set_current_spmd_mesh
from torchtitan.models.deepseek_v3.mtp import MTPLoss
from torchtitan_recipes.overrides._dsv3_mtp_cross_entropy import kernels, ops
from torchtitan_recipes.overrides._dsv3_mtp_cross_entropy.ce_exp import (
    exp as packed_exp,
)
from torchtitan_recipes.overrides._dsv3_mtp_cross_entropy.ops import (
    backward_op,
    cross_entropy_sum,
    forward_op,
    supports,
)
from torchtitan_recipes.overrides.fused_dsv3_mtp_loss import (
    _cross_entropy_loss,
    FusedDSv3MTPLoss,
)
from triton.language.extra.cuda import libdevice


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (10, 3),
    reason="GB300 cross-entropy specialization",
)


def native(logits, labels):
    return F.cross_entropy(logits.float(), labels, reduction="sum", ignore_index=-100)


def assert_bits(expected, actual):
    if isinstance(expected, tuple):
        assert isinstance(actual, tuple) and len(expected) == len(actual)
        for reference, candidate in zip(expected, actual):
            assert_bits(reference, candidate)
        return
    assert (expected.shape, expected.dtype) == (actual.shape, actual.dtype)
    reference = (
        expected.detach().resolve_neg().contiguous().reshape(-1).view(torch.uint8)
    )
    candidate = actual.detach().resolve_neg().contiguous().reshape(-1).view(torch.uint8)
    assert torch.equal(
        reference, candidate
    ), f"{int((reference != candidate).sum())} mismatched bytes"


@triton.jit
def _exp_rounding_probe(Values, Reference, Candidate, ELEMENTS: tl.constexpr):
    offsets = tl.program_id(0) * 1024 + tl.arange(0, 1024)
    values = tl.load(Values + offsets, offsets < ELEMENTS, other=0)
    tl.store(Reference + offsets, libdevice.exp(values), offsets < ELEMENTS)
    tl.store(Candidate + offsets, packed_exp(values), offsets < ELEMENTS)


@pytest.mark.parametrize("case", ["dense_range", "random_ieee_bits"])
def test_packed_exp_preserves_libdevice_rounding(case):
    torch.manual_seed(310)
    if case == "dense_range":
        values = torch.linspace(-105, 90, 2**23, device="cuda")
    else:
        values = (
            torch.randint(0, 2**32, (2**24,), dtype=torch.int64, device="cuda")
            .int()
            .view(torch.float32)
        )
    expected = torch.empty_like(values)
    actual = torch.empty_like(values)
    _exp_rounding_probe[(triton.cdiv(values.numel(), 1024),)](
        values, expected, actual, values.numel()
    )
    assert_bits(expected, actual)


def compare(logits, labels, scale, *, repeats=2):
    logits.requires_grad_(True)
    gradient = torch.tensor(scale, device=logits.device)
    expected = native(logits, labels)
    expected_gradient = torch.autograd.grad(expected, logits, gradient)[0]
    for _ in range(repeats):
        actual = cross_entropy_sum(logits, labels)
        actual_gradient = torch.autograd.grad(actual, logits, gradient)[0]
        assert_bits((expected, expected_gradient), (actual, actual_gradient))


@pytest.mark.parametrize("seed", [42, 821, 2981, 4507])
def test_production_shape_loss_and_gradient_bits(seed):
    torch.manual_seed(seed)
    logits = torch.randn(4096, 129280, device="cuda", dtype=torch.bfloat16)
    labels = torch.randint(129280, (4096,), device="cuda")
    labels[::41] = -100
    compare(logits, labels, 0.1 / (32 * 4095 * 128))


@pytest.mark.parametrize("scale", [1.0, -2.375, 0.0, -0.0, 1e-38])
def test_finite_extremes_and_signed_zeros(scale):
    logits = torch.zeros(9, 129280, dtype=torch.bfloat16, device="cuda")
    logits[0, ::2] = -0.0
    logits[1] = -88.0
    logits[1, 4] = 0.0
    logits[2] = -104.0
    logits[2, 0] = 0.0
    logits[3] = torch.finfo(torch.bfloat16).min
    logits[3, -1] = torch.finfo(torch.bfloat16).max
    logits[4] = torch.finfo(torch.bfloat16).tiny
    logits[5] = -torch.finfo(torch.bfloat16).tiny
    logits[6, 0] = 80.0
    logits[7, 0] = -80.0
    labels = torch.tensor([0, 4, 1, 129279, 0, 129279, -100, 0, -100], device="cuda")
    compare(logits, labels, scale)


@pytest.mark.parametrize("scale", [1.0, 0.0, -0.0, -2.375])
def test_nonfinite_and_ignored_rows(scale):
    logits = torch.zeros(7, 129280, dtype=torch.bfloat16, device="cuda")
    logits[0, 0] = float("inf")
    logits[1, 0] = -float("inf")
    logits[2, 0] = float("nan")
    logits[3] = -float("inf")
    logits[4, 0] = float("inf")
    logits[5, 0] = float("nan")
    logits[6] = float("nan")
    labels = torch.tensor([0, 0, 0, 0, -100, -100, -100], device="cuda")
    compare(logits, labels, scale)


def test_all_ignored_loss_is_positive_zero():
    logits = torch.randn(33, 129280, dtype=torch.bfloat16, device="cuda")
    labels = torch.full((33,), -100, device="cuda")
    compare(logits, labels, 0.0)


def test_fp32_log_probability_and_gradient_rounding_boundaries():
    torch.manual_seed(42)
    logits = torch.randn(33, 129280, dtype=torch.bfloat16, device="cuda")
    labels = torch.randint(129280, (33,), device="cuda")
    labels[::7] = -100
    loss, stats = kernels.forward(logits, labels)
    expected_lp = F.log_softmax(logits.float(), dim=-1)
    actual_lp = (logits.float() - stats[0, :, None]) - stats[1, :, None]
    assert_bits(expected_lp, actual_lp)
    grad_loss = torch.tensor(-2.375, device="cuda")
    nll_gradient = torch.ops.aten.nll_loss_backward.default(
        grad_loss,
        expected_lp,
        labels,
        None,
        2,
        -100,
        torch.tensor(28.0, device="cuda"),
    )
    expected_gradient = torch.ops.aten._log_softmax_backward_data.default(
        nll_gradient, expected_lp, 1, torch.float32
    )
    actual_gradient = torch.empty_like(expected_lp)
    kernels.backward(logits, labels, stats, grad_loss, output=actual_gradient)
    assert_bits(expected_gradient, actual_gradient)
    assert_bits(native(logits, labels), loss)


@pytest.mark.parametrize("tokens", [1, 31, 32, 33, 511, 512, 1024, 2048])
def test_native_scalar_reduction_order(tokens):
    torch.manual_seed(tokens)
    logits = torch.randn(tokens, 129280, device="cuda", dtype=torch.bfloat16)
    labels = torch.randint(129280, (tokens,), device="cuda")
    labels[::13] = -100
    assert_bits(native(logits, labels), cross_entropy_sum(logits, labels))


def test_lazy_negative_upstream_and_retained_gradient_accumulation():
    torch.manual_seed(83)
    logits = torch.randn(33, 129280, device="cuda", dtype=torch.bfloat16)
    labels = torch.randint(129280, (33,), device="cuda")
    labels[::9] = -100
    upstream = torch._neg_view(torch.tensor(0.375, device="cuda"))
    results = []
    initial_gradient = torch.randn_like(logits)
    for function in (native, cross_entropy_sum):
        x = logits.detach().clone().requires_grad_()
        x.grad = initial_gradient.clone()
        loss = function(x, labels)
        loss.backward(upstream, retain_graph=True)
        loss.backward(upstream)
        results.append((loss, x.grad))
    assert_bits(*results)


def test_saved_state_uses_bf16_input_and_two_fp32_rows():
    logits = torch.randn(
        4096, 129280, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    labels = torch.randint(129280, (4096,), device="cuda")
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(
        lambda t: saved.append(t) or t, lambda t: t
    ):
        loss = cross_entropy_sum(logits, labels)
    assert len(saved) == 3
    assert saved[0] is logits
    assert saved[1] is labels
    assert saved[2].shape == (2, 4096) and saved[2].dtype == torch.float32
    assert saved[2].nbytes == 32768
    torch.autograd.grad(loss, logits)


def test_fake_contract_and_layout_fallback():
    with FakeTensorMode():
        logits = torch.empty(4096, 129280, device="cuda", dtype=torch.bfloat16)
        labels = torch.empty(4096, device="cuda", dtype=torch.int64)
        assert supports(logits, labels)
        loss, stats = forward_op(logits, labels, -100)
        assert loss.shape == () and loss.dtype == torch.float32
        assert stats.shape == (2, 4096) and stats.stride() == (4096, 1)
        gradient = backward_op(logits, labels, stats, loss, -100)
        assert gradient.shape == logits.shape and gradient.dtype == logits.dtype
        assert not supports(logits[:, ::2], labels)
        assert not supports(logits.float(), labels)
        assert not supports(logits, labels.int())
        assert not supports(logits[:33], labels[:33])


@pytest.mark.parametrize("scale", [float("inf"), -float("inf"), float("nan")])
def test_nonfinite_upstream_preserves_ignored_gradients(scale):
    torch.manual_seed(91)
    logits = torch.randn(9, 129280, device="cuda", dtype=torch.bfloat16)
    labels = torch.randint(129280, (9,), device="cuda")
    labels[::2] = -100
    compare(logits, labels, scale)


@pytest.mark.parametrize("ignore_index", [-1, 7, 2**33 + 7])
def test_ignore_index_and_int64_label_identity(ignore_index):
    torch.manual_seed(93)
    logits = torch.randn(
        33, 129280, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    labels = torch.randint(129280, (33,), device="cuda")
    labels[::3] = ignore_index
    upstream = torch.tensor(-0.375, device="cuda")
    expected = F.cross_entropy(
        logits.float(), labels, reduction="sum", ignore_index=ignore_index
    )
    actual = cross_entropy_sum(logits, labels, ignore_index=ignore_index)
    expected_gradient = torch.autograd.grad(expected, logits, upstream)[0]
    actual_gradient = torch.autograd.grad(actual, logits, upstream)[0]
    assert_bits((expected, expected_gradient), (actual, actual_gradient))


def test_backward_rejects_wrong_saved_state_or_upstream_metadata():
    with FakeTensorMode():
        logits = torch.empty(4096, 129280, device="cuda", dtype=torch.bfloat16)
        labels = torch.empty(4096, device="cuda", dtype=torch.int64)
        loss, stats = forward_op(logits, labels, -100)
        with pytest.raises(ValueError, match="FP32 stats"):
            backward_op(logits, labels, stats.bfloat16(), loss, -100)
        with pytest.raises(ValueError, match="scalar FP32 gradient"):
            backward_op(logits, labels, stats, loss.reshape(1), -100)


def test_compile_forward_backward_and_cuda_graph_replay():
    torch.manual_seed(73)
    x = torch.randn(33, 129280, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(129280, (33,), device="cuda")
    upstream = torch.tensor(0.375, device="cuda")
    expected = native(x, labels)
    expected_gradient = torch.autograd.grad(expected, x, upstream)[0]
    actual = torch.compile(cross_entropy_sum, fullgraph=True)(x, labels)
    actual_gradient = torch.autograd.grad(actual, x, upstream)[0]
    assert_bits((expected, expected_gradient), (actual, actual_gradient))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        x = x.detach().requires_grad_()
        for _ in range(3):
            loss = cross_entropy_sum(x, labels)
            torch.autograd.grad(loss, x, upstream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            loss = cross_entropy_sum(x, labels)
            gradient = torch.autograd.grad(loss, x, upstream)[0]
    graph.replay()
    torch.cuda.synchronize()
    assert_bits((expected, expected_gradient), (loss, gradient))


@pytest.mark.parametrize("normalize", [False, True])
def test_mtp_integration_preserves_each_depth_denominator_and_gradient(normalize):
    torch.manual_seed(42)
    logits = tuple(
        torch.randn(4096, 129280, device="cuda", dtype=torch.bfloat16) for _ in range(3)
    )
    labels = tuple(torch.randint(129280, (4096,), device="cuda") for _ in range(3))
    labels[0][::41] = -100
    labels[1][::13] = -100
    labels[2].fill_(-100)
    counts = (
        torch.tensor([32 * 4096 * 128, 32 * 4095 * 128, 0], device="cuda")
        if normalize
        else None
    )
    upstream = torch.tensor(-0.375, device="cuda")
    results = []
    # Exercise the future accepted dispatch without changing the shipped gate.
    with patch.object(ops, "ACCEPTED", True), patch.object(
        ops, "cross_entropy_sum", wraps=cross_entropy_sum
    ) as fused:
        for cls in (MTPLoss, FusedDSv3MTPLoss):
            inputs = tuple(value.detach().requires_grad_() for value in logits)
            module = cls.Config(global_vocab_size=129280, mtp_scale=0.3).build()
            loss, metrics = module(inputs, labels, counts)
            gradients = torch.autograd.grad(loss, inputs, upstream)
            assert metrics == {}
            results.append((loss, *gradients))
        assert fused.call_count == 3
    assert_bits(*results)


def test_chunked_mtp_integration_preserves_hidden_and_lm_head_gradients():
    torch.manual_seed(42)
    hidden = tuple(
        torch.randn(8192, 16, device="cuda", dtype=torch.bfloat16) for _ in range(2)
    )
    labels = tuple(torch.randint(129280, (8192,), device="cuda") for _ in range(2))
    labels[0][::41] = -100
    labels[1][::13] = -100
    counts = torch.tensor([8192 * 128, 8191 * 128], device="cuda")
    weight = torch.randn(129280, 16, device="cuda", dtype=torch.bfloat16) * 0.02
    results = []
    with patch.object(ops, "ACCEPTED", True), patch.object(
        ops, "cross_entropy_sum", wraps=cross_entropy_sum
    ) as fused:
        for cls in (MTPLoss, FusedDSv3MTPLoss):
            wrapper = ChunkedLossWrapper.Config(
                num_chunks=2,
                loss_fn=cls.Config(global_vocab_size=129280, mtp_scale=0.3),
            ).build()
            lm_head = torch.nn.Linear(
                16, 129280, bias=False, device="cuda", dtype=torch.bfloat16
            )
            with torch.no_grad():
                lm_head.weight.copy_(weight)
            wrapper.set_lm_head(lm_head)
            inputs = tuple(value.detach().requires_grad_() for value in hidden)
            loss, metrics = wrapper(inputs, labels, counts)
            loss.backward()
            assert metrics == {}
            results.append(
                (loss, *(value.grad for value in inputs), lm_head.weight.grad)
            )
        assert fused.call_count == 4
    assert_bits(*results)


def test_model_callback_compiles_with_the_accepted_dispatch():
    torch.manual_seed(42)
    logits = torch.randn(
        4096, 129280, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    labels = torch.randint(129280, (4096,), device="cuda")
    labels[::41] = -100
    upstream = torch.tensor(-0.375, device="cuda")
    expected = native(logits, labels)
    expected_gradient = torch.autograd.grad(expected, logits, upstream)[0]
    with patch.object(ops, "ACCEPTED", True):
        compiled = torch.compile(_cross_entropy_loss, fullgraph=True)
        actual = compiled(logits, labels, global_vocab_size=129280)
        actual_gradient = torch.autograd.grad(actual, logits, upstream)[0]
    assert_bits((expected, expected_gradient), (actual, actual_gradient))


def test_compiled_dispatch_rechecks_negative_view_metadata():
    torch.manual_seed(42)
    logits = torch.randn(4096, 129280, device="cuda", dtype=torch.bfloat16)
    labels = torch.randint(129280, (4096,), device="cuda")
    compiled = torch.compile(_cross_entropy_loss, backend="eager", fullgraph=True)
    cases = (
        (logits, labels),
        (torch._neg_view(logits), labels),
        (logits, torch._neg_view(-labels)),
        (logits, labels),
    )
    with patch.object(ops, "ACCEPTED", True):
        for values, targets in cases:
            values = values.detach().requires_grad_()
            expected = native(values, targets)
            actual = compiled(values, targets)
            expected_gradient = torch.autograd.grad(expected, values)[0]
            actual_gradient = torch.autograd.grad(actual, values)[0]
            assert_bits((expected, expected_gradient), (actual, actual_gradient))


@pytest.mark.parametrize("local", [False, True])
def test_spmd_token_shards_produce_partial_loss(local):
    if dist.is_initialized():
        pytest.skip("standalone FakePG test requires an unused default group")
    dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=4)
    try:
        mesh = init_device_mesh("cuda", (2, 2, 1), mesh_dim_names=("dp", "cp", "tp"))
        logits = torch.randn(
            4096, 129280, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        labels = torch.randint(129280, (4096,), device="cuda")
        expected = native(logits, labels)
        expected_gradient = torch.autograd.grad(expected, logits)[0]
        with patch.object(ops, "ACCEPTED", True), set_current_spmd_mesh(
            mesh
        ), typecheck(local=local):
            local_type = {"dp": spmd.V, "cp": spmd.V, "tp": spmd.R}
            spmd.assert_type(
                logits,
                spmd.SpmdType(
                    local_type, partition_spec=spmd.PartitionSpec(("dp", "cp"), None)
                ),
            )
            spmd.assert_type(
                labels,
                spmd.SpmdType(
                    local_type, partition_spec=spmd.PartitionSpec(("dp", "cp"))
                ),
            )
            result = _cross_entropy_loss(logits, labels)
            spmd.assert_type(result, {"dp": spmd.P, "cp": spmd.P, "tp": spmd.R})
        gradient = torch.autograd.grad(result, logits)[0]
        assert_bits((expected, expected_gradient), (result, gradient))
    finally:
        dist.destroy_process_group()


def test_tensor_parallel_fallback_preserves_native_collectives():
    if dist.is_initialized():
        pytest.skip("standalone FakePG test requires an unused default group")
    dist.init_process_group("fake", store=FakeStore(), rank=0, world_size=2)
    try:
        mesh = init_device_mesh("cuda", (1, 1, 2), mesh_dim_names=("dp", "cp", "tp"))
        logits = torch.randn(33, 64640, device="cuda", dtype=torch.bfloat16)
        labels = torch.randint(64640, (33,), device="cuda")
        labels[::7] = -100
        results, collectives = [], []
        from torchtitan.components.loss import funcol

        with patch.object(ops, "ACCEPTED", True), patch.object(
            ops,
            "cross_entropy_sum",
            side_effect=AssertionError("TP must use native CE"),
        ), set_current_spmd_mesh(mesh):
            for function in (cross_entropy_loss, _cross_entropy_loss):
                inputs = logits.detach().requires_grad_()
                with patch.object(
                    funcol, "all_reduce", wraps=funcol.all_reduce
                ) as calls:
                    loss = function(inputs, labels, global_vocab_size=129280)
                    gradient = torch.autograd.grad(loss, inputs)[0]
                collectives.append(
                    [
                        (tuple(call.args[0].shape), call.kwargs["reduceOp"])
                        for call in calls.call_args_list
                    ]
                )
                results.append((loss, gradient))
        assert (
            collectives[0]
            == collectives[1]
            == [((33, 1), "MAX"), ((33, 1), "SUM"), ((33, 1), "SUM")]
        )
        assert_bits(*results)
    finally:
        dist.destroy_process_group()

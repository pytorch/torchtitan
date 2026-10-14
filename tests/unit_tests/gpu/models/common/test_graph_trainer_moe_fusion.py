# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from torchtitan.experiments.graph_trainer.autoresearch.kernels.router_aux import (
    router_aux_backward,
    router_aux_forward,
)
from torchtitan.experiments.graph_trainer.autoresearch.kernels.router_backward import (
    router_backward,
)
from torchtitan.experiments.graph_trainer.autoresearch.kernels.router_counts import (
    token_counts,
)
from torchtitan.experiments.graph_trainer.autoresearch.kernels.router_selection import (
    select_experts_fused,
)
from torchtitan.experiments.graph_trainer.autoresearch.kernels.router_weights import (
    router_weights_forward,
    router_weights_gate_backward,
)
from torchtitan.experiments.graph_trainer.autoresearch.moe_fusion import (
    _selection,
    _weights,
)
from torchtitan.models.common.hi_mid_lo_linear import _split_into_bf16_pieces

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _assert_same_bits(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.stride() == expected.stride()
    assert torch.equal(
        actual.contiguous().reshape(-1).view(torch.uint8),
        expected.contiguous().reshape(-1).view(torch.uint8),
    )


@pytest.fixture
def deterministic():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


@pytest.mark.parametrize("case", ["random", "ties", "tiny", "endpoints"])
def test_router_matches_eager_autograd_bits(case, deterministic):
    generator = torch.Generator(device="cuda").manual_seed(42)
    scores = torch.randn(4096, 256, device="cuda", generator=generator).sigmoid()
    bias = torch.zeros(256, device="cuda")
    if case == "ties":
        scores.fill_(0.5)
    elif case == "tiny":
        scores *= 1e-30
    elif case == "endpoints":
        scores[:, ::2] = 0
        scores[:, 1::2] = 1
    else:
        bias.normal_(std=0.01, generator=generator)
    scores.requires_grad_()
    indices = _selection(scores.detach(), bias)
    _assert_same_bits(select_experts_fused(scores, bias), indices)
    expected = _weights(scores, indices)
    actual = router_weights_forward(scores, indices)
    for value, reference in zip(actual, expected, strict=True):
        _assert_same_bits(value, reference)
    grad_weights = torch.randn(4096, 8, device="cuda", generator=generator)
    aux_grad = torch.randn(4096, 256, device="cuda", generator=generator) * 1e-9
    grad_scores = torch.autograd.grad(expected[0], scores, grad_weights)[0]
    grad_logits = torch.ops.aten.sigmoid_backward.default(
        aux_grad + grad_scores, scores
    )
    expected_pieces = _split_into_bf16_pieces(grad_logits, 3, dim=1)
    actual_pieces = router_weights_gate_backward(
        grad_weights, actual[1], actual[2], indices, aux_grad, scores
    )
    _assert_same_bits(actual_pieces, expected_pieces)


@pytest.mark.parametrize("case", ["random", "skewed", "balanced"])
def test_token_counts_match_dense_routing_map_bits(case, deterministic):
    generator = torch.Generator(device="cuda").manual_seed(42)
    if case == "random":
        scores = torch.rand(4096, 256, device="cuda", generator=generator)
        indices = scores.topk(8, dim=-1).indices
    elif case == "skewed":
        indices = torch.arange(8, device="cuda").repeat(4096, 1)
    else:
        indices = (
            torch.arange(4096, device="cuda")[:, None]
            + torch.arange(8, device="cuda")[None, :]
        ) % 256
    routing_map = torch.zeros(4096, 256, dtype=torch.bool, device="cuda")
    routing_map.scatter_(-1, indices, True)
    expected = (routing_map.sum(dim=0), routing_map.float().sum(dim=0))
    actual = token_counts(indices)
    for value, reference in zip(actual, expected, strict=True):
        _assert_same_bits(value, reference)


@pytest.mark.parametrize(
    "num_tokens,case",
    [
        (15, "random"),
        (16, "random"),
        (17, "random"),
        (4096, "strided_gradient"),
        (4096, "broadcast_gradient"),
        (4096, "negative_zero_gradient"),
        (4096, "clamp_limits"),
    ],
)
def test_aux_normalization_matches_captured_autograd_bits(
    num_tokens, case, deterministic
):
    generator = torch.Generator(device="cuda").manual_seed(42)
    scores = torch.randn(num_tokens, 256, device="cuda", generator=generator).sigmoid()
    grad = torch.randn(256, device="cuda", generator=generator)
    if case == "strided_gradient":
        grad = torch.randn(512, device="cuda", generator=generator)[::2]
    elif case == "broadcast_gradient":
        grad = torch.ones((), device="cuda").expand(256)
    elif case == "negative_zero_gradient":
        scores.neg_()
        grad = -torch.zeros_like(grad)
    elif case == "clamp_limits":
        scores[0].zero_()
        scores[1].fill_(2.0**-149)
        scores[2].fill_(1e-12 / 512)
        scores[3].fill_(1e-12 / 256)
        scores[4].fill_(1e-12 / 128)
    scores.requires_grad_()

    def joint(scores, grad):
        normalized = torch.nn.functional.normalize(scores, p=1, dim=-1)
        return torch.autograd.grad(normalized.sum(0), scores, grad)[0]

    raw_norm = torch.linalg.vector_norm(scores, ord=1, dim=-1, keepdim=True)
    clamped_norm = raw_norm.clamp_min(1e-12)
    expected = (scores / clamped_norm, raw_norm, clamped_norm)
    actual = router_aux_forward(scores)
    for value, reference in zip(actual, expected, strict=True):
        _assert_same_bits(value, reference)
    # The captured clamp backward uses strict > at epsilon.
    captured = make_fx(joint)(scores, grad)
    expected_grad = captured(scores, grad)
    actual_grad = router_aux_backward(grad, scores, actual[1], actual[2])
    _assert_same_bits(actual_grad, expected_grad)
    if case != "clamp_limits":
        _assert_same_bits(actual_grad, joint(scores, grad))


@pytest.mark.parametrize("case", ["random", "tiny", "clamp_limits", "negative_zero"])
def test_combined_router_backward_matches_captured_autograd_bits(case, deterministic):
    generator = torch.Generator(device="cuda").manual_seed(42)
    scores = torch.randn(4096, 256, device="cuda", generator=generator).sigmoid()
    grad_weights = torch.randn(4096, 8, device="cuda", generator=generator)
    grad_per_expert = torch.randn(256, device="cuda", generator=generator) * 1e-9
    if case == "tiny":
        scores *= 1e-30
    elif case == "clamp_limits":
        scores[0].zero_()
        scores[1].fill_(2.0**-149)
        scores[2].fill_(1e-12 / 512)
        scores[3].fill_(1e-12 / 256)
        scores[4].fill_(1e-12 / 128)
    elif case == "negative_zero":
        scores.neg_()
        grad_weights = -torch.zeros_like(grad_weights)
        grad_per_expert = -torch.zeros_like(grad_per_expert)
    scores.requires_grad_()
    indices = _selection(scores.detach(), torch.zeros(256, device="cuda"))

    def joint(scores, indices, grad_weights, grad_per_expert):
        weights = _weights(scores, indices)[0]
        grad_router = torch.autograd.grad(weights, scores, grad_weights)[0]
        normalized = torch.nn.functional.normalize(scores, p=1, dim=-1)
        grad_aux = torch.autograd.grad(normalized.sum(0), scores, grad_per_expert)[0]
        grad_logits = torch.ops.aten.sigmoid_backward.default(
            grad_aux + grad_router, scores
        )
        return _split_into_bf16_pieces(grad_logits, 3, dim=1)

    inputs = (scores, indices, grad_weights, grad_per_expert)
    expected = make_fx(joint)(*inputs)(*inputs)
    _, selected, denominator = _weights(scores, indices)
    raw_norm = torch.linalg.vector_norm(scores, ord=1, dim=-1, keepdim=True)
    actual = router_backward(
        grad_weights,
        selected,
        denominator,
        indices,
        grad_per_expert,
        scores,
        raw_norm,
        raw_norm.clamp_min(1e-12),
    )
    _assert_same_bits(actual, expected)
    if case != "clamp_limits":
        _assert_same_bits(actual, joint(*inputs))


@pytest.mark.parametrize("shape,seed", [((4096, 2, 2048), 42), ((128, 2, 128), 123)])
def test_shared_expert_matches_production_quantization_bits(shape, seed, deterministic):
    pytest.importorskip("torchao")
    if torch.cuda.get_device_capability()[0] < 10:
        pytest.skip("This MXFP8 kernel was validated on Blackwell")
    from torchtitan.experiments.graph_trainer.autoresearch.kernels.shared_expert import (
        swiglu_backward_mxfp8_quantize,
        swiglu_mxfp8_quantize,
    )
    from torchtitan.experiments.graph_trainer.autoresearch.moe_fusion import (
        _swiglu_backward_quantize,
        _swiglu_quantize,
    )
    from torchtitan_recipes.overrides import fused_swiglu  # noqa: F401

    generator = torch.Generator(device="cuda").manual_seed(seed)
    gate_up = torch.randn(
        shape, dtype=torch.bfloat16, device="cuda", generator=generator
    )
    grad_out = torch.randn(
        shape[0], shape[2], dtype=torch.bfloat16, device="cuda", generator=generator
    )
    for inputs, reference, fused in (
        ((gate_up,), _swiglu_quantize, swiglu_mxfp8_quantize),
        (
            (grad_out, gate_up),
            _swiglu_backward_quantize,
            swiglu_backward_mxfp8_quantize,
        ),
    ):
        for actual, expected in zip(fused(*inputs), reference(*inputs), strict=True):
            _assert_same_bits(actual, expected)

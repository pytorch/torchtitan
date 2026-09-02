# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch


pytest.importorskip("torchao")
pytest.importorskip("torchao.prototype.moe_training.kernels.mxfp8")

from torchtitan.components.quantization._fsdp_tensor import (  # noqa: E402
    _UnshardedFSDPTensor,
)
from torchtitan.components.quantization.mxfp8.grouped_experts import (  # noqa: E402
    get_mxfp8_grouped_experts_cls,
)
from torchtitan.components.quantization.mxfp8.tensor import (  # noqa: E402
    _GroupedExpertsShardedTensorWithMXFP8Compute,
    _quantize_mxfp8_grouped_weight,
)
from torchtitan.models.common.moe import GroupedExperts  # noqa: E402


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability() < (10, 0),
        reason="MXFP8 requires SM100 or later",
    ),
]

# Each expert's token group is padded to a multiple of 128 rows, which is what
# the converter's pad_multiple guarantees at runtime.
_NUM_EXPERTS = 4
_DIM = 128
_HIDDEN_DIM = 256
_TOKENS_PER_EXPERT = 128


@pytest.fixture(scope="module", autouse=True)
def _prime_autograd_backward_thread():
    """Issue one CUDA op on the autograd backward thread before any test runs.

    TorchAO's native TMA activation quantizer fails with an illegal instruction
    when it is the very first CUDA operation issued on a given thread. Real
    training never hits this because the backward thread runs many ops before
    reaching the experts, but a unit test whose graph contains nothing else
    would. Priming the thread keeps these tests focused on MXFP8 behavior.
    """

    class _Prime(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.clone()

        @staticmethod
        def backward(ctx, grad):
            torch.zeros(8, device=grad.device)
            return grad

    _Prime.apply(
        torch.zeros(8, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    ).sum().backward()


def _make_mxfp8_experts(input_activation_format_for_backward: str = "bf16"):
    experts_cls = get_mxfp8_grouped_experts_cls(GroupedExperts)
    module = (
        experts_cls.Config(
            dim=_DIM,
            hidden_dim=_HIDDEN_DIM,
            num_experts=_NUM_EXPERTS,
            input_activation_format_for_backward=input_activation_format_for_backward,
        )
        .build()
        .cuda()
        .bfloat16()
    )
    for parameter in module.parameters():
        torch.nn.init.normal_(parameter, std=0.02)
    return module


def _make_inputs():
    num_tokens = _NUM_EXPERTS * _TOKENS_PER_EXPERT
    x_RD = torch.randn(num_tokens, _DIM, device="cuda", dtype=torch.bfloat16)
    num_tokens_per_expert_E = torch.full(
        (_NUM_EXPERTS,), _TOKENS_PER_EXPERT, device="cuda", dtype=torch.int32
    )
    return x_RD, num_tokens_per_expert_E


def _sqnr(reference: torch.Tensor, actual: torch.Tensor) -> float:
    reference, actual = reference.float(), actual.float()
    noise = ((reference - actual) ** 2).mean()
    return (10 * torch.log10((reference**2).mean() / noise)).item()


@pytest.mark.parametrize("input_activation_format_for_backward", ["bf16", "mxfp8"])
def test_mxfp8_grouped_experts_match_bf16_reference(
    input_activation_format_for_backward,
):
    torch.manual_seed(0)
    experts = _make_mxfp8_experts(input_activation_format_for_backward)
    reference = (
        GroupedExperts.Config(
            dim=_DIM, hidden_dim=_HIDDEN_DIM, num_experts=_NUM_EXPERTS
        )
        .build()
        .cuda()
        .bfloat16()
    )
    reference.load_state_dict(experts.state_dict())

    x_RD, num_tokens_per_expert_E = _make_inputs()
    x_reference = x_RD.clone().requires_grad_()
    x_RD = x_RD.requires_grad_()

    out = experts(x_RD, num_tokens_per_expert_E)
    out_reference = reference(x_reference, num_tokens_per_expert_E)
    assert _sqnr(out_reference, out) > 20.0

    grad_out = torch.randn_like(out)
    out.backward(grad_out)
    out_reference.backward(grad_out)

    assert _sqnr(x_reference.grad, x_RD.grad) > 20.0
    for name, parameter in experts.named_parameters():
        reference_grad = reference.get_parameter(name).grad
        assert _sqnr(reference_grad, parameter.grad) > 20.0, name


def test_grouped_experts_quantize_per_call_without_fsdp():
    experts = _make_mxfp8_experts()
    # The wrappers are installed at construction, but with no data parallel
    # implementation driving their lifecycle they hold only the BF16 weights,
    # so each grouped GEMM builds its operands per call.
    for parameter in experts.parameters():
        # The sharded state is the type, so there is no operands to
        # inspect: an unsharded tensor would be a _UnshardedFSDPTensor instead.
        assert isinstance(parameter, _GroupedExpertsShardedTensorWithMXFP8Compute)
        assert not isinstance(parameter, _UnshardedFSDPTensor)
        assert parameter.dtype == torch.bfloat16


def test_grouped_weights_are_wrapped_preserving_logical_metadata():
    experts = _make_mxfp8_experts()
    for name, parameter in experts.named_parameters():
        assert isinstance(parameter, _GroupedExpertsShardedTensorWithMXFP8Compute), name
        assert parameter.dtype == torch.bfloat16
        expected_shape = (
            (_NUM_EXPERTS, _DIM, _HIDDEN_DIM)
            if name == "w2_EDF"
            else (_NUM_EXPERTS, _HIDDEN_DIM, _DIM)
        )
        assert parameter.shape == expected_shape, name
        assert parameter.requires_grad


def test_grouped_weights_are_wrapped_exactly_once():
    experts = _make_mxfp8_experts()
    # The wrapper holds the BF16 weight in ``_tensor``; that must be the plain
    # parameter data, not another wrapper.
    for parameter in experts.parameters():
        assert not isinstance(
            parameter._tensor, _GroupedExpertsShardedTensorWithMXFP8Compute
        )


def test_grouped_weight_quantization_shares_values_across_orientations():
    torch.manual_seed(0)
    weight_ENK = torch.randn(
        _NUM_EXPERTS, _HIDDEN_DIM, _DIM, device="cuda", dtype=torch.bfloat16
    )
    operands = _quantize_mxfp8_grouped_weight(weight_ENK)

    # Square 32x32 tiles make the quantized values transpose-invariant, so the
    # two operands differ only in physical layout.
    assert torch.equal(
        operands.weight_qdata_fprop_EKN.transpose(-2, -1).float(),
        operands.weight_qdata_dgrad_ENK.float(),
    )
    # torch._scaled_grouped_mm requires a right operand that is column-major
    # within each expert.
    assert operands.weight_qdata_fprop_EKN.stride()[-2] == 1
    assert operands.weight_qdata_dgrad_ENK.stride()[-2] == 1
    assert (
        operands.weight_qdata_fprop_EKN.untyped_storage().data_ptr()
        != operands.weight_qdata_dgrad_ENK.untyped_storage().data_ptr()
    )


def test_grouped_weight_quantization_rejects_unsupported_weights():
    with pytest.raises(ValueError, match="requires a 3D weight"):
        _quantize_mxfp8_grouped_weight(
            torch.randn(_HIDDEN_DIM, _DIM, device="cuda", dtype=torch.bfloat16)
        )
    with pytest.raises(ValueError, match="requires BF16 weights"):
        _quantize_mxfp8_grouped_weight(
            torch.randn(
                _NUM_EXPERTS, _HIDDEN_DIM, _DIM, device="cuda", dtype=torch.float32
            )
        )
    with pytest.raises(ValueError, match="divisible by 32"):
        _quantize_mxfp8_grouped_weight(
            torch.randn(_NUM_EXPERTS, 48, _DIM, device="cuda", dtype=torch.bfloat16)
        )


@pytest.mark.parametrize("input_activation_format_for_backward", ["bf16", "mxfp8"])
def test_mxfp8_grouped_experts_saves_selected_input_activation(
    input_activation_format_for_backward,
):
    experts = _make_mxfp8_experts(input_activation_format_for_backward)
    x_RD, num_tokens_per_expert_E = _make_inputs()
    x_RD = x_RD.requires_grad_()

    saved = []

    def pack_hook(tensor):
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda tensor: tensor):
        experts(x_RD, num_tokens_per_expert_E).sum().backward()

    saved_dtypes = {tensor.dtype for tensor in saved}
    if input_activation_format_for_backward == "mxfp8":
        assert torch.float8_e4m3fn in saved_dtypes
    else:
        # The BF16 policy still saves quantized weights, but never a quantized
        # copy of the routed input.
        num_bf16_activations = sum(
            tensor.dtype == torch.bfloat16 and tensor.ndim == 2 for tensor in saved
        )
        assert num_bf16_activations > 0

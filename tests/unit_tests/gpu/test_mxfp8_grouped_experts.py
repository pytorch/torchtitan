# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import importlib.util
from typing import NamedTuple

import pytest
import torch
from torch.utils.checkpoint import checkpoint


pytest.importorskip("torchao")
pytest.importorskip("torchao.prototype.moe_training.kernels.mxfp8")

from torchao.prototype.mx_formats.kernels import mxfp8_quantize_cuda  # noqa: E402
from torchao.prototype.mx_formats.utils import to_blocked  # noqa: E402
from torchtitan.components.quantization._fsdp_tensor import (  # noqa: E402
    _UnshardedFSDPTensor,
)
from torchtitan.components.quantization.mxfp8._common import (  # noqa: E402
    _MXFP8_BLOCK_SIZE,
    _MXFP8_FUSED_MLP_ROW_ALIGNMENT,
    _MXFP8_SCALING_MODE,
)
from torchtitan.components.quantization.mxfp8.grouped_experts import (  # noqa: E402
    _blocked_colwise_scales,
    _split_w13_grad,
    _w13_dgrad_operands,
    _w13_fprop_operands,
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


def _make_mxfp8_experts(
    input_activation_format_for_backward: str = "bf16", fuse_grouped_mlp: bool = False
):
    experts_cls = get_mxfp8_grouped_experts_cls(GroupedExperts)
    module = (
        experts_cls.Config(
            dim=_DIM,
            hidden_dim=_HIDDEN_DIM,
            num_experts=_NUM_EXPERTS,
            input_activation_format_for_backward=input_activation_format_for_backward,
            fuse_grouped_mlp=fuse_grouped_mlp,
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


# The fused expert MLP runs on TorchAO's cudnn_grouped_mlp ops, built for
# SM 10.0, over token groups padded to their fixed 256-row multiple.
_requires_fused_grouped_mlp_kernels = pytest.mark.skipif(
    importlib.util.find_spec(
        "torchao.prototype.moe_training.kernels.mxfp8.cudnn_grouped_mlp"
    )
    is None
    or (torch.cuda.is_available() and torch.cuda.get_device_capability() != (10, 0)),
    reason="the fused grouped MLP needs TorchAO's cudnn_grouped_mlp ops on SM 10.0",
)
_ZERO_TOKEN_EXPERT = 1


class _RoutedRows(NamedTuple):
    x: torch.Tensor
    grad_y: torch.Tensor
    num_tokens_per_expert: torch.Tensor
    active: int
    """Rows an expert owns, ``offsets_E[-1]``; the rest is the tail."""


def _make_fused_inputs(tail_value: float = float("nan")) -> _RoutedRows:
    """Expert-major ``x_RD`` and ``grad_y_RD`` shaped like the padded dispatcher's output.

    Each expert's group is padded to the fused path's 256-row multiple, one
    full group for expert ``_ZERO_TOKEN_EXPERT`` that received no tokens (as
    ``permute_and_pad`` gives it) and a mostly-padding group for expert 3.
    Routed rows are random, padding rows zero (the dispatcher gathers them
    from its zero sentinel row), and one group of tail rows past the last
    group holds ``tail_value``.
    """
    rows = _MXFP8_FUSED_MLP_ROW_ALIGNMENT
    routed = [rows - 3, 0, 2 * rows - 1, 7]
    padded = [max(-(-num_routed // rows), 1) * rows for num_routed in routed]
    active = sum(padded)
    generator = torch.Generator(device="cuda").manual_seed(0)
    x_RD = torch.zeros(active + rows, _DIM, device="cuda", dtype=torch.bfloat16)
    grad_y_RD = torch.zeros_like(x_RD)
    start = 0
    for num_routed, num_padded in zip(routed, padded):
        for buffer in (x_RD, grad_y_RD):
            buffer[start : start + num_routed] = torch.randn(
                num_routed,
                _DIM,
                device="cuda",
                dtype=torch.bfloat16,
                generator=generator,
            )
        start += num_padded
    x_RD[active:] = tail_value
    grad_y_RD[active:] = tail_value
    num_tokens_per_expert_E = torch.tensor(padded, device="cuda", dtype=torch.int32)
    return _RoutedRows(x_RD, grad_y_RD, num_tokens_per_expert_E, active)


def _run_step(module, rows: _RoutedRows, forward=None):
    """One forward/backward through ``forward`` (the module by default).

    Returns the active rows of ``y`` and ``x.grad`` and the full weight
    gradients, keyed by name.
    """
    # Not ``forward or module``: truth-testing a compiled module calls its
    # ``__len__``, which raises.
    forward = module if forward is None else forward
    x_RD = rows.x.detach().clone().requires_grad_()
    module.zero_grad(set_to_none=True)
    y_RD = forward(x_RD, rows.num_tokens_per_expert)
    y_RD.backward(rows.grad_y)
    outputs = {"y": y_RD.detach()[: rows.active], "x.grad": x_RD.grad[: rows.active]}
    for name, parameter in module.named_parameters():
        outputs[f"{name}.grad"] = parameter.grad.clone()
    return outputs


def _interleave_gate_up_rows(w1_EFD: torch.Tensor, w3_EFD: torch.Tensor):
    """BF16 gate and up weights -> the ``(E, 2F, D)`` weight in the fused FC1
    kernel's row order, alternating 32-row bands ``[gate0 | up0 | gate1 | ...]``."""
    e, f, d = w1_EFD.shape
    bands = (e, f // _MXFP8_BLOCK_SIZE, _MXFP8_BLOCK_SIZE, d)
    return torch.stack([w1_EFD.view(bands), w3_EFD.view(bands)], dim=2).reshape(
        e, 2 * f, d
    )


def _bitwise_equal(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    """Byte equality, so fp8 qdata and E8M0 scales compare without a cast."""
    return actual.shape == expected.shape and torch.equal(
        actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8)
    )


@pytest.mark.parametrize(
    ("num_experts", "hidden_dim", "dim"),
    [(_NUM_EXPERTS, _HIDDEN_DIM, _DIM), (3, 384, 128), (2, 640, 384)],
)
def test_fused_fc1_operands_match_quantizing_the_interleaved_weight(
    num_experts, hidden_dim, dim
):
    """The fp8-domain gate/up interleave is bitwise the quantized interleaved weight.

    Square 32x32 tiles never straddle a 32-row band, so interleaving the
    FSDP-cached FPROP and DGRAD operands of w1 and w3 (qdata and blocked
    scales) must reproduce the operands of the ``(E, 2F, D)`` weight in the
    fused FC1 kernel's row order; that is what lets a fused step quantize no
    weight. Covers an expert count that is not a power of two and dimensions
    that are multiples of 128 but not of 256.
    """
    torch.manual_seed(0)
    w1_EFD = torch.randn(
        num_experts, hidden_dim, dim, device="cuda", dtype=torch.bfloat16
    )
    w3_EFD = torch.randn_like(w1_EFD)
    w1 = _quantize_mxfp8_grouped_weight(w1_EFD)
    w3 = _quantize_mxfp8_grouped_weight(w3_EFD)
    reference = _quantize_mxfp8_grouped_weight(_interleave_gate_up_rows(w1_EFD, w3_EFD))

    qdata_E2FD, scales = _w13_fprop_operands(
        w1.weight_qdata_fprop_EKN,
        w1.weight_scale_fprop_swizzled,
        w3.weight_qdata_fprop_EKN,
        w3.weight_scale_fprop_swizzled,
    )
    assert qdata_E2FD.is_contiguous()
    assert _bitwise_equal(
        qdata_E2FD, reference.weight_qdata_fprop_EKN.transpose(-2, -1)
    )
    assert _bitwise_equal(scales, reference.weight_scale_fprop_swizzled)

    qdata_ED2F, scales = _w13_dgrad_operands(
        w1.weight_qdata_dgrad_ENK,
        w1.weight_scale_dgrad_swizzled,
        w3.weight_qdata_dgrad_ENK,
        w3.weight_scale_dgrad_swizzled,
    )
    assert qdata_ED2F.is_contiguous()
    assert _bitwise_equal(
        qdata_ED2F, reference.weight_qdata_dgrad_ENK.transpose(-2, -1)
    )
    assert _bitwise_equal(scales, reference.weight_scale_dgrad_swizzled)


def test_fused_fc1_weight_gradient_splits_back_into_gate_and_up():
    torch.manual_seed(0)
    grad_E2FD = torch.randn(
        _NUM_EXPERTS, 2 * _HIDDEN_DIM, _DIM, device="cuda", dtype=torch.bfloat16
    )
    grad_w1_EFD, grad_w3_EFD = _split_w13_grad(grad_E2FD)
    assert grad_w1_EFD.is_contiguous() and grad_w3_EFD.is_contiguous()
    assert torch.equal(_interleave_gate_up_rows(grad_w1_EFD, grad_w3_EFD), grad_E2FD)


@pytest.mark.parametrize(
    "group_ends",
    [[256, 512, 512, 1024], [256, 768, 1024]],
    ids=["zero-size-group", "non-power-of-two-groups"],
)
def test_fused_columnwise_scales_match_the_per_group_blocked_layout(group_ends):
    """One K-groups swizzle and a static prefix slice give the WGRAD ops' scales.

    The ops read each group's columnwise scales in its own blocked layout,
    concatenated. The swizzle lays the groups out back to back and pads only
    past them (here a tail group past ``offsets_E[-1]``), so the slice needs
    no device sync. A zero-size group (a repeated end offset) contributes
    nothing, and a group count that is not a power of two is padded for the
    swizzle's ``tl.arange``.
    """
    torch.manual_seed(0)
    rows = group_ends[-1] + _MXFP8_FUSED_MLP_ROW_ALIGNMENT
    offsets_E = torch.tensor(group_ends, device="cuda", dtype=torch.int32)
    x_RD = torch.randn(rows, _DIM, device="cuda", dtype=torch.bfloat16)
    _, _, _, scales = mxfp8_quantize_cuda(
        x_RD, rowwise=False, colwise=True, scaling_mode=_MXFP8_SCALING_MODE
    )

    blocked = _blocked_colwise_scales(scales, offsets_E, rows=rows)
    reference = torch.cat(
        [
            to_blocked(
                scales[:, start // _MXFP8_BLOCK_SIZE : end // _MXFP8_BLOCK_SIZE]
            ).reshape(-1)
            for start, end in zip([0, *group_ends[:-1]], group_ends)
            if end > start
        ]
    )
    assert blocked.numel() == _DIM * rows // _MXFP8_BLOCK_SIZE
    assert _bitwise_equal(blocked[: reference.numel()], reference)


@_requires_fused_grouped_mlp_kernels
@pytest.mark.parametrize("input_activation_format_for_backward", ["bf16", "mxfp8"])
def test_fused_grouped_mlp_tracks_the_unfused_experts(
    input_activation_format_for_backward,
):
    """The fused path stays within the quantization band of the per-GEMM path.

    Both consume the same 32x32 weight tiles, so they differ only where the
    SwiGLU rounds (the fused kernels apply it to FP32 accumulators, the
    per-GEMM path to BF16) and in the requantized ``h`` and ``dz``. Measured
    on GB200 across ``y``, ``x.grad`` and the three weight gradients:
    35.1-35.4 dB from the per-GEMM path and 23.6-24.2 dB from BF16, where the
    per-GEMM path itself scores 23.6-24.1 dB; a layout or dataflow bug lands
    near 0 dB on both gates.
    """
    torch.manual_seed(0)
    experts = _make_mxfp8_experts(
        input_activation_format_for_backward, fuse_grouped_mlp=True
    )
    unfused = _make_mxfp8_experts(input_activation_format_for_backward)
    unfused.load_state_dict(experts.state_dict())
    reference = (
        GroupedExperts.Config(
            dim=_DIM, hidden_dim=_HIDDEN_DIM, num_experts=_NUM_EXPERTS
        )
        .build()
        .cuda()
        .bfloat16()
    )
    reference.load_state_dict(experts.state_dict())
    rows = _make_fused_inputs()

    fused = _run_step(experts, rows)
    per_gemm = _run_step(unfused, rows)
    bf16 = _run_step(reference, rows)
    for name, value in fused.items():
        assert _sqnr(per_gemm[name], value) >= 30.0, name
        assert _sqnr(bf16[name], value) >= 21.0, name


@_requires_fused_grouped_mlp_kernels
@pytest.mark.parametrize("input_activation_format_for_backward", ["bf16", "mxfp8"])
def test_fused_grouped_mlp_reads_only_the_active_rows(
    input_activation_format_for_backward,
):
    """Rows no expert owns never reach an active output or a gradient.

    The fused path quantizes whole buffers (``x`` columnwise in the forward or
    the backward, by the saved format) and slices the per-group scales
    afterwards, so a leaking tail shows up as NaN in an active row or as a
    changed weight gradient. An expert that received no tokens owns only zero
    rows and gets exactly zero weight gradients.
    """
    torch.manual_seed(0)
    experts = _make_mxfp8_experts(
        input_activation_format_for_backward, fuse_grouped_mlp=True
    )

    poisoned = _run_step(experts, _make_fused_inputs())
    clean = _run_step(experts, _make_fused_inputs(tail_value=0.0))
    for name, value in poisoned.items():
        assert torch.isfinite(value).all(), name
        assert torch.equal(value, clean[name]), name
    for name in ("w1_EFD.grad", "w2_EDF.grad", "w3_EFD.grad"):
        assert torch.count_nonzero(poisoned[name][_ZERO_TOKEN_EXPERT]) == 0, name


@_requires_fused_grouped_mlp_kernels
@pytest.mark.parametrize("execution_mode", ["compile", "activation_checkpoint"])
def test_fused_grouped_mlp_runs_outside_plain_eager(execution_mode):
    """Compile and non-reentrant checkpointing reproduce eager bit for bit.

    Compile traces through the allow_in_graph Function, checkpointing
    recomputes the forward during backward. The kernels are deterministic, so
    anything but equality means the data movement around them changed.
    """
    torch.manual_seed(0)
    experts = _make_mxfp8_experts(fuse_grouped_mlp=True)
    rows = _make_fused_inputs()
    eager = _run_step(experts, rows)

    if execution_mode == "compile":
        forward = torch.compile(experts, fullgraph=True)
    else:

        def forward(x_RD, num_tokens_per_expert_E):
            return checkpoint(
                experts, x_RD, num_tokens_per_expert_E, use_reentrant=False
            )

    got = _run_step(experts, rows, forward=forward)
    for name, value in got.items():
        assert torch.equal(value, eager[name]), name


@_requires_fused_grouped_mlp_kernels
@pytest.mark.parametrize("input_activation_format_for_backward", ["bf16", "mxfp8"])
def test_fused_grouped_mlp_saves_selected_input_activation(
    input_activation_format_for_backward,
):
    """The fused Function saves ``x`` for WGRAD in the configured format only:
    the columnwise fp8 copy cast in the forward, or the BF16 rows themselves."""
    experts = _make_mxfp8_experts(
        input_activation_format_for_backward, fuse_grouped_mlp=True
    )
    rows = _make_fused_inputs(tail_value=0.0)
    x_RD = rows.x.requires_grad_()

    saved = []

    def pack_hook(tensor):
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda tensor: tensor):
        experts(x_RD, rows.num_tokens_per_expert).sum().backward()

    saved_input_dtypes = {
        tensor.dtype for tensor in saved if tensor.shape == x_RD.shape
    }
    if input_activation_format_for_backward == "mxfp8":
        assert saved_input_dtypes == {torch.float8_e4m3fn}
    else:
        assert saved_input_dtypes == {torch.bfloat16}


def test_fused_grouped_mlp_rejects_a_short_token_buffer():
    """A buffer below the kernels' 256-row group raises before any kernel runs."""
    experts = _make_mxfp8_experts(fuse_grouped_mlp=True)
    rows = _MXFP8_FUSED_MLP_ROW_ALIGNMENT // 2
    x_RD = torch.randn(rows, _DIM, device="cuda", dtype=torch.bfloat16)
    num_tokens_per_expert_E = torch.zeros(
        _NUM_EXPERTS, device="cuda", dtype=torch.int32
    )
    num_tokens_per_expert_E[0] = rows

    with pytest.raises(ValueError, match="at least 256"):
        experts(x_RD, num_tokens_per_expert_E)

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


pytest.importorskip("torchao")
pytest.importorskip("torchao.prototype.moe_training.kernels.mxfp8")

import torchtitan.components.quantization.mxfp8.linear as mxfp8_linear  # noqa: E402
from torchtitan.components.quantization._fsdp_tensor import (  # noqa: E402
    _UnshardedFSDPTensor,
)
from torchtitan.components.quantization.mxfp8.linear import MXFP8Linear  # noqa: E402
from torchtitan.components.quantization.mxfp8.tensor import (  # noqa: E402
    _LinearShardedTensorWithMXFP8Compute,
)


pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required"),
    pytest.mark.skipif(
        torch.cuda.is_available() and torch.cuda.get_device_capability() < (10, 0),
        reason="MXFP8 requires SM100 or later",
    ),
]


def _make_sharded_mxfp8_linear(
    in_features: int = 128,
    out_features: int = 96,
    *,
    bias: bool = True,
    input_activation_format_for_backward: str = "bf16",
) -> MXFP8Linear:
    """Build a layer in its as-constructed state, before any unshard."""
    return (
        MXFP8Linear.Config(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            input_activation_format_for_backward=input_activation_format_for_backward,
        )
        .build()
        .cuda()
        .bfloat16()
    )


def _make_mxfp8_linear(
    in_features: int = 128,
    out_features: int = 96,
    *,
    bias: bool = True,
    input_activation_format_for_backward: str = "bf16",
) -> MXFP8Linear:
    """Build a layer that is ready to run forward.

    ``forward`` requires a data parallel implementation to have built the
    unsharded tensor for the current unshard lifetime. These tests are single
    process and have none, so stand in for FSDP's post-all-gather hook and
    install the unsharded tensor directly.
    """
    return _install_unsharded_weight(
        _make_sharded_mxfp8_linear(
            in_features,
            out_features,
            bias=bias,
            input_activation_format_for_backward=input_activation_format_for_backward,
        )
    )


def _build_unsharded_tensor(
    sharded_weight: _LinearShardedTensorWithMXFP8Compute,
    weight_NK: torch.Tensor,
) -> _UnshardedFSDPTensor:
    """Quantize and wrap, as fsdp_post_all_gather does on the first unshard.

    These tests are single process, so the weight is never a DTensor and the
    DTensor half of _BuildUnshardedTensorFunction does not apply.
    """
    with torch.no_grad():
        return _UnshardedFSDPTensor(
            weight_NK, sharded_weight._build_operands(weight_NK)
        )


def _install_unsharded_weight(linear: MXFP8Linear) -> MXFP8Linear:
    """Stand in for FSDP's post-all-gather hook on a single-process layer.

    The unsharded tensor is storage-free, so anything that reads the weight's
    storage -- ``state_dict``, ``load_state_dict`` -- has to run before this.
    """
    sharded_weight = linear.weight
    linear.weight = nn.Parameter(
        _build_unsharded_tensor(sharded_weight, sharded_weight._tensor),
        requires_grad=sharded_weight.requires_grad,
    )
    return linear


@pytest.mark.parametrize("input_activation_format_for_backward", ["bf16", "mxfp8"])
def test_mxfp8_linear_saves_selected_input_activation(
    input_activation_format_for_backward,
):
    linear = _make_mxfp8_linear(
        input_activation_format_for_backward=input_activation_format_for_backward,
    )
    x = torch.randn(
        37,
        linear.in_features,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    saved_tensors = []

    def pack_hook(tensor):
        saved_tensors.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda tensor: tensor):
        output = linear(x)
        output.backward(torch.randn_like(output))

    assert output.shape == (37, linear.out_features)
    # The weight is saved as the single unsharded-tensor wrapper, not as its
    # individual DGRAD operands, so FSDP can free and refill that storage
    # around the reshard. Only the activation operands are saved as tensors.
    weight_saves = [
        tensor for tensor in saved_tensors if isinstance(tensor, _UnshardedFSDPTensor)
    ]
    activation_saves = [
        tensor
        for tensor in saved_tensors
        if not isinstance(tensor, _UnshardedFSDPTensor)
    ]
    assert len(weight_saves) == 1
    if input_activation_format_for_backward == "bf16":
        assert len(activation_saves) == 1
        assert activation_saves[0].dtype == torch.bfloat16
        assert (
            activation_saves[0].untyped_storage()._cdata == x.untyped_storage()._cdata
        )
    else:
        assert len(activation_saves) == 2
        assert all(tensor.dtype != torch.bfloat16 for tensor in activation_saves)
        assert (
            sum(tensor.dtype == torch.float8_e4m3fn for tensor in activation_saves) == 1
        )
        assert (
            sum(tensor.dtype == torch.float8_e8m0fnu for tensor in activation_saves)
            == 1
        )
    assert all(type(tensor) is torch.Tensor for tensor in activation_saves)


@pytest.mark.parametrize(
    ("input_activation_format_for_backward", "expected_quantize_calls"),
    [
        ("bf16", [(True, False), (True, True), (False, True)]),
        ("mxfp8", [(True, True), (True, True)]),
    ],
)
def test_mxfp8_input_activation_format_for_backward_controls_quantization_work(
    monkeypatch,
    input_activation_format_for_backward,
    expected_quantize_calls,
):
    original_quantize = mxfp8_linear.mxfp8_quantize_cuda
    quantize_calls = []

    def record_quantize(*args, **kwargs):
        quantize_calls.append((kwargs["rowwise"], kwargs["colwise"]))
        return original_quantize(*args, **kwargs)

    monkeypatch.setattr(mxfp8_linear, "mxfp8_quantize_cuda", record_quantize)
    linear = _make_mxfp8_linear(
        bias=False,
        input_activation_format_for_backward=input_activation_format_for_backward,
    )
    x = torch.randn(
        64,
        linear.in_features,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    linear(x).sum().backward()

    assert quantize_calls == expected_quantize_calls


def test_mxfp8_square_weight_dgrad_qdata_is_transpose_view():
    weight_NK = torch.randn(
        96,
        128,
        device="cuda",
        dtype=torch.bfloat16,
    )
    unsharded_tensor = _build_unsharded_tensor(
        _LinearShardedTensorWithMXFP8Compute(weight_NK), weight_NK
    )
    operands = unsharded_tensor.operands
    assert operands is not None
    inner_tensor_names, metadata = unsharded_tensor.__tensor_flatten__()
    rebuilt_unsharded_tensor = type(unsharded_tensor).__tensor_unflatten__(
        {name: getattr(unsharded_tensor, name) for name in inner_tensor_names},
        metadata,
        unsharded_tensor.shape,
        unsharded_tensor.stride(),
    )
    rebuilt_operands = rebuilt_unsharded_tensor.operands
    assert rebuilt_operands is not None

    # Inner tensors are named after the operands dataclass fields.
    # The FPROP qdata is a property, not a field, so FSDP does not manage it.
    assert inner_tensor_names == [
        "_weight_qdata_dgrad_NK",
        "_weight_scale_fprop_swizzled",
        "_weight_scale_dgrad_swizzled",
    ]
    assert (
        operands.weight_qdata_dgrad_NK.data_ptr()
        == operands.weight_qdata_fprop_KN.data_ptr()
    )
    assert torch.equal(
        operands.weight_qdata_dgrad_NK,
        operands.weight_qdata_fprop_KN.t(),
    )
    assert (
        rebuilt_operands.weight_qdata_dgrad_NK.data_ptr()
        == rebuilt_operands.weight_qdata_fprop_KN.data_ptr()
    )


def test_operands_fields_must_be_distinct_allocations():
    """FSDP owns each field's storage, so a field may not alias another.

    Derived views belong in properties, as ``_MXFP8LinearOperands`` does for
    its FPROP qdata. A format that made one a field instead would have FSDP
    free the same storage twice.
    """
    from dataclasses import dataclass

    from torchtitan.components.quantization._fsdp_tensor import _unsharded_inner_tensors

    qdata = torch.empty(64, 64, device="cuda", dtype=torch.float8_e4m3fn)

    @dataclass(frozen=True)
    class AliasingOperands:
        qdata_dgrad: torch.Tensor
        qdata_fprop: torch.Tensor

    with pytest.raises(ValueError, match="distinct allocations"):
        _unsharded_inner_tensors(AliasingOperands(qdata, qdata.t()))

    @dataclass(frozen=True)
    class DistinctOperands:
        qdata_dgrad: torch.Tensor
        scale: torch.Tensor

    scale = torch.empty(64, 2, device="cuda", dtype=torch.float8_e8m0fnu)
    assert len(_unsharded_inner_tensors(DistinctOperands(qdata, scale))) == 2


def test_mxfp8_linear_quantizes_per_call_without_an_unsharded_tensor():
    """A layer nobody unsharded builds its operands on every call.

    GraphTrainer under the spmd_types backend reaches forward this way: its
    runtime hands over a plain annotated local tensor, so the wrapper
    SimpleFSDP built never arrives. Eager FSDP2 always installs one.
    """
    linear = _make_sharded_mxfp8_linear()
    assert isinstance(linear.weight, _LinearShardedTensorWithMXFP8Compute)
    assert not isinstance(linear.weight, _UnshardedFSDPTensor)
    x = torch.randn(
        32,
        linear.in_features,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    output = linear(x)
    output.sum().backward()

    assert output.shape == (32, linear.out_features)
    assert x.grad is not None
    # The gradient reaches the wrapped parameter in high precision.
    assert linear.weight.grad is not None
    assert linear.weight.grad.dtype == torch.bfloat16


def test_mxfp8_linear_gradient_reaches_the_unsharded_tensor():
    linear = _make_mxfp8_linear()
    assert isinstance(linear.weight, _UnshardedFSDPTensor)
    x = torch.randn(
        32,
        linear.in_features,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    output = linear(x)
    output.sum().backward()

    assert output.shape == (32, linear.out_features)
    assert x.grad is not None
    # The gradient reaches the wrapped parameter in high precision.
    assert linear.weight.grad is not None
    assert linear.weight.grad.dtype == torch.bfloat16


@pytest.mark.parametrize("input_activation_format_for_backward", ["bf16", "mxfp8"])
@pytest.mark.parametrize("execution_mode", ["compile", "activation_checkpoint"])
def test_mxfp8_linear_runs_outside_plain_eager(
    execution_mode,
    input_activation_format_for_backward,
):
    """Both non-eager entry points must reach the weight gradient.

    Each one re-enters forward in a way that can lose the saved activation
    state: compile traces it, and non-reentrant checkpointing discards it and
    recreates it during recompute. The saved state differs per format, so both
    formats are exercised under both.
    """
    linear = _make_mxfp8_linear(
        bias=False,
        input_activation_format_for_backward=input_activation_format_for_backward,
    )
    x = torch.randn(
        64,
        linear.in_features,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    if execution_mode == "compile":
        output = torch.compile(linear, fullgraph=True)(x)
    else:
        output = checkpoint(linear, x, use_reentrant=False)
    output.backward(torch.randn_like(output))

    assert output.shape == (64, linear.out_features)
    assert x.grad is not None
    assert linear.weight.grad is not None
    # The gradient is a plain tensor, not the unsharded wrapper.
    assert type(linear.weight.grad) is torch.Tensor


def test_mxfp8_input_activation_formats_for_backward_match():
    bf16 = _make_sharded_mxfp8_linear(
        bias=False,
        input_activation_format_for_backward="bf16",
    )
    mxfp8 = _make_sharded_mxfp8_linear(
        bias=False,
        input_activation_format_for_backward="mxfp8",
    )
    # Copy the weight while it still has storage, then unshard both layers.
    mxfp8.load_state_dict(bf16.state_dict())
    _install_unsharded_weight(bf16)
    _install_unsharded_weight(mxfp8)

    x_hp = torch.randn(
        64,
        bf16.in_features,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    x_mxfp8 = x_hp.detach().clone().requires_grad_()
    grad_output = torch.randn(
        64,
        bf16.out_features,
        device="cuda",
        dtype=torch.bfloat16,
    )

    output_hp = bf16(x_hp)
    output_mxfp8 = mxfp8(x_mxfp8)
    output_hp.backward(grad_output)
    output_mxfp8.backward(grad_output)

    torch.testing.assert_close(output_hp, output_mxfp8, rtol=0, atol=0)
    torch.testing.assert_close(x_hp.grad, x_mxfp8.grad, rtol=0, atol=0)
    torch.testing.assert_close(
        bf16.weight.grad,
        mxfp8.weight.grad,
        rtol=0,
        atol=0,
    )


class _StubMesh:
    """Stands in for a DeviceMesh: fsdp_pre_all_gather only reads ``size()``."""

    def __init__(self, size: int) -> None:
        self._size = size

    def size(self) -> int:
        return self._size


class _StubMixedPrecisionPolicy:
    param_dtype = torch.bfloat16


def test_fsdp_pre_all_gather_pads_an_uneven_shard():
    """A dim-0 size that does not divide the mesh leaves the last rank short.

    All-gather needs every rank to contribute the same number of elements, so
    FSDP's contract is that the hook returns the *padded* shard and passes the
    logical size through as metadata. Driven directly rather than through FSDP
    because a dense MXFP8 weight cannot be unevenly sharded on two ranks: the
    kernels require out_features divisible by 32, which always divides 2.
    """
    # Logical (96, 128) over five ranks: ceil(96 / 5) == 20 rows each, so the
    # last rank holds only 16 and pads up to 20.
    shard_NK = torch.randn(16, 128, device="cuda", dtype=torch.bfloat16)
    sharded_weight = _LinearShardedTensorWithMXFP8Compute(shard_NK)

    (comm_NK,), metadata = sharded_weight.fsdp_pre_all_gather(
        _StubMesh(5),
        torch.Size([96, 128]),
        None,
        None,
        _StubMixedPrecisionPolicy(),
    )

    assert comm_NK.shape == (20, 128)
    assert torch.equal(comm_NK[:16], shard_NK)
    assert torch.count_nonzero(comm_NK[16:]) == 0
    # The logical size rides along so post-all-gather can drop the padding.
    assert tuple(metadata) == (96, 128)


def test_fsdp_pre_all_gather_casts_while_padding():
    """A shard needing both a cast and padding must get one buffer, not two.

    The pad path allocates directly in the comm dtype and lets the copy cast,
    so this pins that the cast still happens and the real rows survive it.
    """

    class _Fp32Policy:
        param_dtype = torch.bfloat16
        reduce_dtype = torch.bfloat16

    shard_NK = torch.randn(16, 128, device="cuda", dtype=torch.float32)
    sharded_weight = _LinearShardedTensorWithMXFP8Compute(shard_NK)

    (comm_NK,), metadata = sharded_weight.fsdp_pre_all_gather(
        _StubMesh(5),
        torch.Size([96, 128]),
        None,
        None,
        _Fp32Policy(),
    )

    assert comm_NK.dtype == torch.bfloat16
    assert comm_NK.shape == (20, 128)
    torch.testing.assert_close(comm_NK[:16], shard_NK.bfloat16())
    assert torch.count_nonzero(comm_NK[16:]) == 0
    assert tuple(metadata) == (96, 128)


def test_fsdp_post_all_gather_drops_the_padding():
    """The gathered buffer includes padding; quantization must not see it."""
    sharded_weight = _LinearShardedTensorWithMXFP8Compute(
        torch.randn(16, 128, device="cuda", dtype=torch.bfloat16)
    )
    # Five ranks contributing 20 padded rows each.
    gathered_NK = torch.randn(100, 128, device="cuda", dtype=torch.bfloat16)

    unsharded_tensor, unsharded_inner_tensors = sharded_weight.fsdp_post_all_gather(
        (gathered_NK,), torch.Size([96, 128]), torch.bfloat16
    )

    assert isinstance(unsharded_tensor, _UnshardedFSDPTensor)
    assert unsharded_tensor.shape == (96, 128)
    assert len(unsharded_inner_tensors) == 3


def test_fsdp_pre_all_gather_rejects_a_non_zero_shard_dim():
    """Only dim 0 is supported; the all-gather concatenates along it."""
    sharded_weight = _LinearShardedTensorWithMXFP8Compute(
        torch.randn(96, 64, device="cuda", dtype=torch.bfloat16)
    )
    with pytest.raises(NotImplementedError, match="sharding dimension 0 only"):
        sharded_weight.fsdp_pre_all_gather(
            _StubMesh(2),
            torch.Size([96, 128]),
            None,
            None,
            _StubMixedPrecisionPolicy(),
        )

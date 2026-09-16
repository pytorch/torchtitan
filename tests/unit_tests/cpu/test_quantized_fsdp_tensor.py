# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from contextlib import nullcontext
from dataclasses import dataclass

import pytest
import torch
import torch.nn.functional as F
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._subclasses.meta_utils import disable_inference_mode_for_fake_prop
from torch.utils._python_dispatch import transform_subclass

from torchtitan.quantization._fsdp_tensor import (
    _ShardedFSDPTensor,
    _UnshardedFSDPTensor,
)


@dataclass(frozen=True)
class _Operands:
    weight: torch.Tensor
    scale: torch.Tensor


class _CloneShardedTensor(_ShardedFSDPTensor):
    def _build_operands(self, logical_tensor, out=None):
        if out is None:
            return _Operands(logical_tensor.clone(), logical_tensor.new_ones(()))
        out.weight.copy_(logical_tensor)
        out.scale.fill_(1)
        return out


def _wrap(tensor, kind):
    if kind == "sharded":
        return _CloneShardedTensor(tensor)
    return _UnshardedFSDPTensor(tensor, _Operands(tensor, tensor.new_ones(())))


@pytest.mark.parametrize("kind", ["sharded", "unsharded"])
@pytest.mark.parametrize("source_inference", [False, True])
@pytest.mark.parametrize("use_inference", [False, True])
@pytest.mark.parametrize("operation", ["view", "detach", "alias", "as_strided"])
def test_view_preserves_inference_and_version_semantics(
    kind, source_inference, use_inference, operation
):
    with torch.inference_mode(source_inference):
        source = torch.arange(12.0).view(3, 4)
        wrapped = _wrap(source, kind)
    with torch.inference_mode(use_inference):
        if operation == "view":
            result = wrapped.view(4, 3)
        elif operation == "detach":
            result = wrapped.detach()
        elif operation == "alias":
            result = torch.ops.aten.alias.default(wrapped)
        else:
            result = wrapped.as_strided((4, 3), (3, 1))
        assert wrapped.is_inference() == source_inference
        assert result.is_inference() == source_inference
        if kind == "sharded":
            assert result._tensor.data_ptr() == source.data_ptr()
        else:
            assert result.operands is wrapped.operands
        if source_inference:
            with pytest.raises(RuntimeError, match="do not track version counter"):
                _ = result._version
        else:
            version = wrapped._version
            torch.autograd.graph.increment_version(wrapped)
            assert result._version == wrapped._version == version + 1


@pytest.mark.parametrize("source_inference", [False, True])
@pytest.mark.parametrize("use_inference", [False, True])
def test_sharded_clone_uses_current_inference_mode(source_inference, use_inference):
    with torch.inference_mode(source_inference):
        source = torch.arange(12.0).view(3, 4)
        wrapped = _CloneShardedTensor(source)
    with torch.inference_mode(use_inference):
        result = wrapped.clone()
    assert result.is_inference() == use_inference
    assert result._tensor.is_inference() == use_inference
    assert result._tensor.data_ptr() != source.data_ptr()
    torch.testing.assert_close(result._tensor, source)


@pytest.mark.parametrize("kind", ["sharded", "unsharded"])
@pytest.mark.parametrize("source_inference", [False, True])
@pytest.mark.parametrize("use_inference", [False, True])
@pytest.mark.parametrize("operation", ["empty_like", "new_zeros"])
def test_factory_uses_current_inference_mode(
    kind, source_inference, use_inference, operation
):
    with torch.inference_mode(source_inference):
        source = torch.ones(3, 4)
        wrapped = _wrap(source, kind)
    with torch.inference_mode(use_inference):
        result = (
            torch.empty_like(wrapped)
            if operation == "empty_like"
            else wrapped.new_zeros((3, 4))
        )
    assert result.is_inference() == use_inference
    if kind == "sharded":
        assert isinstance(result, _CloneShardedTensor)
        result = result._tensor
    else:
        assert type(result) is torch.Tensor
    assert result.is_inference() == use_inference
    assert result.data_ptr() != source.data_ptr()


@pytest.mark.parametrize("metadata_inference", [False, True])
def test_unsharded_view_follows_wrapper_not_operands(metadata_inference):
    with torch.inference_mode(not metadata_inference):
        operands = _Operands(torch.ones(3, 4), torch.ones(()))
    with torch.inference_mode(metadata_inference):
        metadata = torch.ones(3, 4)
        wrapped = _UnshardedFSDPTensor(metadata, operands)
    with torch.inference_mode(not metadata_inference):
        result = wrapped.view(4, 3)
    assert result.is_inference() == wrapped.is_inference() == metadata_inference
    assert result.operands is operands


@pytest.mark.parametrize("wrapper_inference", [False, True])
@pytest.mark.parametrize("operand_inference", [False, True])
def test_unsharded_tensor_reconstruction(wrapper_inference, operand_inference):
    with torch.inference_mode(operand_inference):
        operands = _Operands(torch.ones(3, 4), torch.ones(()))
    with torch.inference_mode(wrapper_inference):
        wrapped = _UnshardedFSDPTensor(torch.ones(3, 4), operands)
        reconstructed = transform_subclass(wrapped, lambda _, tensor: tensor)
    assert reconstructed.is_inference() == wrapper_inference
    assert reconstructed.operands.weight is operands.weight
    assert reconstructed.operands.scale is operands.scale


@pytest.mark.parametrize("wrapper_inference", [False, True])
@pytest.mark.parametrize("operand_inference", [False, True])
@pytest.mark.parametrize("normalize_inference", [False, True])
def test_unsharded_fake_tensor_conversion(
    wrapper_inference, operand_inference, normalize_inference
):
    with torch.inference_mode(operand_inference):
        operands = _Operands(torch.ones(3, 4), torch.ones(()))
    with torch.inference_mode(wrapper_inference):
        metadata = torch.ones(3, 4)
        wrapped = _UnshardedFSDPTensor(metadata, operands)
    context = (
        disable_inference_mode_for_fake_prop() if normalize_inference else nullcontext()
    )
    with context:
        mode = FakeTensorMode()
        fake = mode.from_tensor(wrapped)
        fake_metadata = mode.from_tensor(metadata)
    assert fake.is_inference() == fake_metadata.is_inference()
    assert fake.is_inference() == (wrapper_inference and not normalize_inference)
    assert fake.operands.weight.is_inference() == (
        operand_inference and not normalize_inference
    )
    assert fake.shape == wrapped.shape
    assert fake.dtype == wrapped.dtype


@pytest.mark.parametrize("create_inference", [False, True])
@pytest.mark.parametrize("refill_inference", [False, True])
def test_refill_preserves_operand_identity_and_versions(
    create_inference, refill_inference
):
    logical = torch.arange(12.0).view(3, 4)
    shard = _CloneShardedTensor(logical)
    with torch.inference_mode(create_inference):
        wrapped, tensors = shard.fsdp_post_all_gather(
            (logical,), logical.shape, logical.dtype
        )
    operands = wrapped.operands
    identities = tuple(id(tensor) for tensor in tensors)
    pointers = tuple(tensor.data_ptr() for tensor in tensors)
    versions = tuple(tensor._version for tensor in tensors if not tensor.is_inference())
    updated = logical + 1
    with torch.inference_mode(refill_inference):
        if create_inference and not refill_inference:
            with pytest.raises(RuntimeError, match="outside InferenceMode"):
                shard.fsdp_post_all_gather(
                    (updated,), updated.shape, updated.dtype, out=wrapped
                )
            return
        shard.fsdp_post_all_gather(
            (updated,), updated.shape, updated.dtype, out=wrapped
        )
    assert wrapped.operands is operands
    assert tuple(id(tensor) for tensor in tensors) == identities
    assert tuple(tensor.data_ptr() for tensor in tensors) == pointers
    assert (
        tuple(tensor._version for tensor in tensors if not tensor.is_inference())
        == versions
    )
    torch.testing.assert_close(operands.weight, updated)


def test_refill_preserves_versions_with_mixed_inference_operands():
    logical = torch.arange(12.0).view(3, 4)
    weight = logical.clone()
    with torch.inference_mode():
        scale = torch.zeros(())
    operands = _Operands(weight, scale)
    wrapped = _UnshardedFSDPTensor(logical, operands)
    version = weight._version
    with torch.inference_mode():
        _CloneShardedTensor(logical).fsdp_post_all_gather(
            (logical + 1,), logical.shape, logical.dtype, out=wrapped
        )
    assert wrapped.operands is operands
    assert wrapped.operands.weight is weight
    assert wrapped.operands.scale is scale
    assert weight._version == version
    assert scale.is_inference()
    torch.testing.assert_close(weight, logical + 1)
    torch.testing.assert_close(scale, torch.ones(()))


class _LinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight):
        ctx.save_for_backward(input, weight, weight.operands.weight)
        return F.linear(input, weight.operands.weight)

    @staticmethod
    def backward(ctx, grad_output):
        input, _, weight = ctx.saved_tensors
        return grad_output @ weight, grad_output.t() @ input


def test_training_autograd_and_same_weight_refill():
    torch.manual_seed(0)
    raw_weight = torch.randn(3, 4)
    shard = _CloneShardedTensor(raw_weight)
    holder, _ = shard.fsdp_post_all_gather(
        (raw_weight,), raw_weight.shape, raw_weight.dtype
    )
    weight = torch.nn.Parameter(holder)
    input = torch.randn(2, 4, requires_grad=True)
    ref_weight = raw_weight.clone().requires_grad_()
    ref_input = input.detach().clone().requires_grad_()
    output = _LinearFunction.apply(input, weight.view(3, 4))
    reference = F.linear(ref_input, ref_weight)
    shard.fsdp_post_all_gather(
        (raw_weight,), raw_weight.shape, raw_weight.dtype, out=holder
    )
    output.square().sum().backward()
    reference.square().sum().backward()
    torch.testing.assert_close(output, reference)
    torch.testing.assert_close(input.grad, ref_input.grad)
    torch.testing.assert_close(weight.grad, ref_weight.grad)

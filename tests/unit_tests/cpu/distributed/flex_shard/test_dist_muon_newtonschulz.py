# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exact arithmetic regression tests for Newton-Schulz temporary lifetimes."""

import weakref

import pytest
import torch

from torchtitan.distributed.flex_shard.dist_muon import _zeropower_via_newtonschulz


def _retained_lifetime_reference(
    update: torch.Tensor, *, ns_steps: int
) -> torch.Tensor:
    """Run the original operation order with obsolete temporaries kept alive."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    result = update.to(dtype=torch.bfloat16, copy=True)
    transposed = result.shape[-2] > result.shape[-1]
    if transposed:
        result = result.transpose(-2, -1)
    result.div_(result.norm(dim=(-2, -1), keepdim=True).clamp_min(1e-7))
    if result.ndim == 2:
        for _ in range(ns_steps):
            gram = result @ result.T
            gram_update = torch.addmm(gram, gram, gram, beta=b, alpha=c)
            result = torch.addmm(result, gram_update, result, beta=a)
    else:
        original_shape = result.shape
        matrices = result.reshape(-1, *original_shape[-2:])
        for _ in range(ns_steps):
            gram = matrices @ matrices.transpose(-2, -1)
            gram_update = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
            matrices = torch.baddbmm(matrices, gram_update, matrices, beta=a)
        result = matrices.reshape(original_shape)
    return result.transpose(-2, -1) if transposed else result


@pytest.mark.parametrize(
    "shape", [(7, 11), (11, 7), (3, 7, 11), (3, 11, 7), (2, 3, 7, 11), (2, 3, 11, 7)]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("ns_steps", [0, 1, 5])
@pytest.mark.parametrize("zero_input", [False, True])
def test_shorter_lifetimes_preserve_exact_arithmetic_and_ownership(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    ns_steps: int,
    zero_input: bool,
) -> None:
    """Preserve exact values, shape, strides, and independent output storage."""
    generator = torch.Generator().manual_seed(42)
    update = (
        torch.zeros(shape, dtype=dtype)
        if zero_input
        else torch.randn(shape, dtype=dtype, generator=generator)
    )
    unchanged = update.clone()
    expected = _retained_lifetime_reference(update, ns_steps=ns_steps)
    actual = _zeropower_via_newtonschulz(
        update,
        ns_coefficients=(3.4445, -4.7750, 2.0315),
        ns_steps=ns_steps,
        eps=1e-7,
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, check_stride=True)
    assert actual.dtype == torch.bfloat16
    assert actual.shape == update.shape
    assert actual.untyped_storage().data_ptr() != update.untyped_storage().data_ptr()
    assert torch.equal(update, unchanged)


@pytest.mark.parametrize("shape", [(7, 11), (11, 7), (3, 7, 11), (2, 3, 11, 7)])
def test_iteration_temporaries_are_released_after_last_use(
    shape: tuple[int, ...], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Do not retain old Gram tensors or the replaced initial BF16 allocation."""
    update = torch.randn(shape, generator=torch.Generator().manual_seed(42))
    operation_name = "addmm" if len(shape) == 2 else "baddbmm"
    original_operation = getattr(torch, operation_name)
    original_to = torch.Tensor.to
    copied: weakref.ReferenceType[torch.Tensor] | None = None
    gram: weakref.ReferenceType[torch.Tensor] | None = None
    gram_update: weakref.ReferenceType[torch.Tensor] | None = None
    matrix_updates = [0]

    def record_copy(tensor: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Observe the initial allocation without extending its lifetime."""
        nonlocal copied
        result = original_to(tensor, *args, **kwargs)
        if tensor is update:
            copied = weakref.ref(result)
        return result

    def record_matrix_operation(input: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Check last-use boundaries without storing strong input references."""
        nonlocal gram, gram_update
        if kwargs["beta"] == -4.7750:
            if gram_update is not None:
                assert gram_update() is None
            gram = weakref.ref(input)
            result = original_operation(input, *args, **kwargs)
            gram_update = weakref.ref(result)
            return result
        assert kwargs["beta"] == 3.4445
        assert gram is not None and gram() is None
        if matrix_updates[0]:
            assert copied is not None and copied() is None
        matrix_updates[0] += 1
        return original_operation(input, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", record_copy)
    monkeypatch.setattr(torch, operation_name, record_matrix_operation)
    _zeropower_via_newtonschulz(
        update,
        ns_coefficients=(3.4445, -4.7750, 2.0315),
        ns_steps=5,
        eps=1e-7,
    )
    assert matrix_updates[0] == 5
    assert gram_update is not None and gram_update() is None

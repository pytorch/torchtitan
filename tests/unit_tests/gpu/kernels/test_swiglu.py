# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine
#
# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

import pytest
import torch

from torchtitan.kernels import KernelBackend
from torchtitan.kernels.functional import swiglu_packed
from torchtitan.kernels.math import ceil_divide
from torchtitan.kernels.utils import is_cute_dsl_available, is_triton_available


_SHAPES = [(4, 8), (7, 15), (32, 256), (17, 4103)]
_DTYPES = [torch.float32, torch.float16, torch.bfloat16]
_SEED = 42

_DEFAULT_TOLERANCES = {
    torch.float32: dict(atol=1e-5, rtol=1e-5),
    torch.float16: dict(atol=1e-2, rtol=1e-2),
    torch.bfloat16: dict(atol=1e-2, rtol=1e-2),
}

_BACKEND_AVAILABILITY_CHECKS = {
    "triton": (is_triton_available, "triton"),
    "cuda": (is_cute_dsl_available, "cute_dsl (cutlass)"),
}


def skip_if_incompatible_kernel_backend(kernel_backend: KernelBackend) -> torch.device:
    if not kernel_backend.verify_accelerator():
        pytest.skip(f"skipping test because kernel_backend ({kernel_backend}) is incompatible with the accelerator")

    availability_check = _BACKEND_AVAILABILITY_CHECKS.get(kernel_backend.name)
    if availability_check is not None:
        is_available, name = availability_check
        if not is_available():
            pytest.skip(f"skipping test because {name} is unavailable")

    device = torch.device(kernel_backend.get_compatible_accelerator().value)
    if device.type == "cuda" and not torch.cuda.is_available():
        pytest.skip("skipping test because CUDA is unavailable")

    return device


def get_duplicated_tensors(
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype,
    std: float = 0.01,
    requires_grad: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """2 leaf tensors with identical values but independent autograd graphs, so a kernel path and a
    torch path can be run (and backpropagated) without their gradients interfering."""

    base = torch.randn(shape, device=device, dtype=dtype) * std
    a = base.clone().requires_grad_(requires_grad)
    b = base.clone().requires_grad_(requires_grad)
    return a, b


def assert_equal_tensors(x: torch.Tensor, y: torch.Tensor, exact_match: bool, **kwargs) -> None:
    """compares x and y; pass exact_match=True to require bitwise equality, otherwise falls back to
    a per-dtype default tolerance that can be overridden per call with atol_<dtype>/rtol_<dtype>
    kwargs (e.g. atol_bfloat16=1e-2)."""

    if exact_match:
        assert torch.equal(x, y)
        return

    dtype_name = str(x.dtype).rsplit(".", 1)[-1]
    defaults = _DEFAULT_TOLERANCES[x.dtype]

    atol = kwargs.get(f"atol_{dtype_name}", defaults["atol"])
    rtol = kwargs.get(f"rtol_{dtype_name}", defaults["rtol"])

    torch.testing.assert_close(x, y, atol=atol, rtol=rtol)


def _get_packed_shape(shape: tuple[int, int], dtype: torch.dtype) -> tuple[int, int]:
    # the cute_dsl (cuda) kernel does vectorized loads/stores and needs the last dim aligned to 16
    # bytes, and the interleaved gate/up layout needs it to be even on top of that
    multiple = 2 * (16 // dtype.itemsize)
    return (shape[0], ceil_divide(shape[-1] * 2, multiple) * multiple)


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_swiglu_packed_forward_kernel_vs_torch(dtype: torch.dtype, shape: tuple[int, int]) -> None:
    device = skip_if_incompatible_kernel_backend(KernelBackend.cuda)
    torch.manual_seed(_SEED)

    x_kernel, x_torch = get_duplicated_tensors(_get_packed_shape(shape, dtype), device=device, dtype=dtype)

    y_kernel = swiglu_packed(x_kernel, kernel_backend=KernelBackend.cuda)
    y_torch = swiglu_packed(x_torch, kernel_backend=KernelBackend.torch)

    assert_equal_tensors(y_kernel, y_torch, False)


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_swiglu_packed_backward_kernel_vs_torch(dtype: torch.dtype, shape: tuple[int, int]) -> None:
    device = skip_if_incompatible_kernel_backend(KernelBackend.cuda)
    torch.manual_seed(_SEED)

    x_kernel, x_torch = get_duplicated_tensors(_get_packed_shape(shape, dtype), device=device, dtype=dtype)

    y_kernel = swiglu_packed(x_kernel, kernel_backend=KernelBackend.cuda)
    y_torch = swiglu_packed(x_torch, kernel_backend=KernelBackend.torch)

    y_kernel.sum().backward()
    y_torch.sum().backward()

    assert_equal_tensors(y_kernel, y_torch, False)
    assert_equal_tensors(x_kernel.grad, x_torch.grad, False)


@pytest.mark.parametrize("shape", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
@torch._dynamo.config.patch(recompile_limit=1024)
def test_swiglu_packed_compiled_kernel_vs_torch(dtype: torch.dtype, shape: tuple[int, int]) -> None:
    device = skip_if_incompatible_kernel_backend(KernelBackend.cuda)
    torch.manual_seed(_SEED)

    x_kernel, x_torch = get_duplicated_tensors(_get_packed_shape(shape, dtype), device=device, dtype=dtype)

    swiglu_packed_compiled = torch.compile(swiglu_packed, fullgraph=True)
    y_kernel = swiglu_packed_compiled(x_kernel, kernel_backend=KernelBackend.cuda)
    y_torch = swiglu_packed(x_torch, kernel_backend=KernelBackend.torch)

    assert_equal_tensors(y_kernel, y_torch, False)

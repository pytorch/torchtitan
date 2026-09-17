# Copyright (c) 2026, trainstation team
# The following code is copied from https://github.com/open-lm-engine/lm-engine

# **************************************************
# Copyright (c) 2026, Mayank Mishra
# **************************************************

from __future__ import annotations

from enum import Enum

from .utils import is_torch_available


if is_torch_available():
    import torch


from .device import Accelerator


_IS_ROCM_AVAILABLE = is_torch_available() and torch.version.hip is not None


class KernelBackend(Enum):
    cuda = "cuda"
    jax = "jax"
    mps = "mps"
    nki = "nki"
    pallas = "pallas"
    rocm = "rocm"
    torch = "torch"
    triton = "triton"

    def get_compatible_accelerator(self) -> Accelerator:
        found_accelerator = Accelerator.get_accelerator()

        if self == KernelBackend.torch or (
            self == KernelBackend.triton and found_accelerator in [Accelerator.cuda, Accelerator.rocm]
        ):
            return found_accelerator

        mapping = {
            KernelBackend.cuda: Accelerator.cuda,
            KernelBackend.mps: Accelerator.mps,
            KernelBackend.nki: Accelerator.trainium,
            KernelBackend.pallas: Accelerator.tpu,
            KernelBackend.rocm: Accelerator.rocm,
        }

        return mapping.get(self, None)

    def verify_accelerator(self) -> bool:
        expected_accelerator = self.get_compatible_accelerator()
        found_accelerator = Accelerator.get_accelerator()
        return expected_accelerator == found_accelerator


def get_kernel_backend() -> KernelBackend:
    accelerator = Accelerator.get_accelerator()

    if accelerator == Accelerator.cuda:
        kernel_backend = KernelBackend.rocm if _IS_ROCM_AVAILABLE else KernelBackend.cuda
    elif accelerator == Accelerator.mps:
        kernel_backend = KernelBackend.mps
    elif accelerator == Accelerator.tpu:
        kernel_backend = KernelBackend.pallas
    elif accelerator == Accelerator.trainium:
        kernel_backend = KernelBackend.nki
    else:
        kernel_backend = KernelBackend.triton

    return kernel_backend

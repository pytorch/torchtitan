# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import importlib.util
import logging
import os
import subprocess
from collections.abc import Generator
from types import ModuleType

import torch
from torch._utils import _get_available_device_type, _get_device_module

from torchtitan.observability.logging import warn_once

logger = logging.getLogger(__name__)


def round_up(value: int, multiple: int) -> int:
    return ((value + multiple - 1) // multiple) * multiple


def has_cuda_capability(major: int, minor: int) -> bool:
    # torch.version.hip is None excludes ROCm (capability is a tuple on AMD too).
    return (
        torch.cuda.is_available()
        and torch.version.hip is None
        and torch.cuda.get_device_capability() >= (major, minor)
    )


def _cuda_flash_attention_backend() -> tuple[str, str] | None:
    if not torch.cuda.is_available() or torch.version.hip is not None:
        return None
    # FA4 advertises Hopper support, but as of writing it hangs under
    # torch.compile there, so Hopper (sm90) stays on FA3.
    # https://github.com/pytorch/torchtitan/pull/4413
    # PyTorch supports FA3 on major 9 and FA4 on majors 9/10 only.
    major, _ = torch.cuda.get_device_capability()
    return {
        9: ("FA3", "flash_attn_interface"),
        10: ("FA4", "flash_attn.cute.interface"),
    }.get(major)


def get_cuda_flash_attention_impl() -> str | None:
    """Return an installed FlashAttention implementation supported by this GPU."""
    backend = _cuda_flash_attention_backend()
    if backend is None:
        return None
    impl, module_path = backend
    try:
        available = importlib.util.find_spec(module_path) is not None
    except ModuleNotFoundError as error:
        # Looking up a submodule imports its parents. Only a missing backend
        # package means unavailable; a missing dependency is a broken install.
        if error.name is None or not (
            module_path == error.name or module_path.startswith(error.name + ".")
        ):
            raise
        available = False
    return impl if available else None


def activate_cuda_flash_attention() -> None:
    """Activate the installed CUDA backend, or retain PyTorch's default FA2."""
    from torch.nn.attention import (
        activate_flash_attention_impl,
        current_flash_attention_impl,
    )

    if not torch.cuda.is_available() or torch.version.hip is not None:
        return
    capability = torch.cuda.get_device_capability()
    backend = _cuda_flash_attention_backend()
    if backend is not None:
        impl, module_path = backend
        # Activation replaces process-wide dispatcher registrations.
        if current_flash_attention_impl() == impl:
            return
        try:
            if get_cuda_flash_attention_impl() is not None:
                activate_flash_attention_impl(impl)
                return
        except (ImportError, RuntimeError, ValueError) as error:
            raise RuntimeError(
                f"{impl} activation failed on SM {capability[0]}.{capability[1]}. "
                f"Check that '{module_path}' and its dependencies are installed "
                "and compatible with this PyTorch/CUDA build."
            ) from error
        reason = f"{impl} module '{module_path}' is not installed"
    else:
        reason = "No supported FA3/FA4 implementation"

    if current_flash_attention_impl() is None:
        warn_once(
            logger,
            f"{reason} on SM {capability[0]}.{capability[1]}; using FA2. "
            "FA2 requires equal Q/K/V head dimensions.",
        )


def validate_flash_attention_head_dims(qk_head_dim: int, *, v_head_dim: int) -> None:
    """Reject CUDA FA2 layouts with unequal query/key and value head dimensions."""
    from torch.nn.attention import current_flash_attention_impl

    if (
        qk_head_dim == v_head_dim
        or not torch.cuda.is_available()
        or torch.version.hip is not None
        or current_flash_attention_impl() not in (None, "FA2")
    ):
        return
    capability = torch.cuda.get_device_capability()
    backend = _cuda_flash_attention_backend()
    remedy = "Configure a compatible attention backend, such as FlexAttention."
    if backend is not None:
        impl, module_path = backend
        remedy = (
            f"Install a compatible {impl} wheel providing '{module_path}', "
            "or configure a compatible attention backend, such as FlexAttention."
        )
    raise ValueError(
        f"FA2 requires equal Q/K/V head dimensions, but got Q/K={qk_head_dim}, "
        f"V={v_head_dim} on SM {capability[0]}.{capability[1]}. {remedy}"
    )


def has_rocm_capability(major: int, minor: int) -> bool:
    is_rocm = torch.cuda.is_available() and torch.version.hip is not None
    return is_rocm and torch.cuda.get_device_capability() >= (
        major,
        minor,
    )


def get_device_info() -> tuple[str, ModuleType]:
    device_type = _get_available_device_type() or "cuda"
    device_module = _get_device_module(device_type)  # default device_module:torch.cuda
    return device_type, device_module


device_type, device_module = get_device_info()


def get_local_device() -> torch.device:
    """Return this process's device under LOCAL_RANK or visible-device launch.

    Launchers normally expose multiple accelerators per process and set
    LOCAL_RANK to identify which local device the process should use. Some
    launchers instead mask each process down to a single accelerator with
    CUDA_VISIBLE_DEVICES or another backend-specific visible-device mask. When
    exactly one device is visible, target device index 0; otherwise preserve
    the existing LOCAL_RANK mapping.
    """
    local_rank_str = os.environ.get("LOCAL_RANK")
    if local_rank_str is None:
        raise ValueError("LOCAL_RANK must be set before selecting a local device.")
    try:
        local_rank = int(local_rank_str)
    except ValueError as e:
        raise ValueError(
            "LOCAL_RANK environment variable must be a valid integer, "
            f"got: {local_rank_str}"
        ) from e
    if local_rank < 0:
        raise ValueError(f"LOCAL_RANK must be non-negative, got: {local_rank}")

    num_devices = None
    if hasattr(device_module, "device_count"):
        num_devices = device_module.device_count()
    device_index = 0 if num_devices == 1 else local_rank
    if num_devices is not None and num_devices > 1 and device_index >= num_devices:
        raise ValueError(
            f"LOCAL_RANK={local_rank} is outside the visible {device_type} "
            f"device count ({num_devices}). If each process is launched with a "
            f"single visible {device_type} device, set the visible-device mask "
            "per process so device_count() returns 1."
        )
    return torch.device(device_type, device_index)


# hardcoded BF16 type peak flops for NVIDIA A100, H20, H100, H200, B200 GPU,
# AMD MI250, MI300X, MI325X, MI350X, MI355X, Intel PVC, and AWS Trainium/Inferentia
def get_peak_flops(device_name: str) -> float:
    try:
        # Run the lspci command and capture the output
        result = subprocess.run(["lspci"], stdout=subprocess.PIPE, text=True)
        # Filter the output for lines containing both "NVIDIA" and "H100"
        filtered_lines = [
            line
            for line in result.stdout.splitlines()
            if "NVIDIA" in line and "H100" in line
        ]
        # Join all filtered lines into a single string
        device_name = " ".join(filtered_lines) or device_name
    except FileNotFoundError as e:
        logger.warning(f"Error running lspci: {e}, fallback to use device_name")
    if "A100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/a100/
        return 312e12
    elif "A6000" in device_name:
        # data from https://www.nvidia.com/content/dam/en-zz/Solutions/design-visualization/
        # quadro-product-literature/proviz-print-nvidia-rtx-a6000-datasheet-us-nvidia-1454980-r9-web%20(1).pdf
        # NOTE: 309.7 TFLOPS is with sparsity; dense value is half.
        return 154.85e12
    elif "H100" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h100/
        # NOTE: Specifications are one-half lower without sparsity.
        if "NVL" in device_name:
            return 835e12
        elif "PCIe" in device_name:
            return 756e12
        else:  # for H100 SXM and other variants
            return 989e12
    elif "H200" in device_name:
        # data from https://www.nvidia.com/en-us/data-center/h200/
        return 989e12
    elif "H20" in device_name:
        # NVIDIA H20 is a region-specific GPU variant.
        # Since first-hand specifications do not seem to be readily available on
        # NVIDIA's official global website, we refer to technical reports from
        # Tom's Hardware. The peak BF16/FP16 Tensor performance is reported as
        # 148 TFLOPS.
        # Ref: https://www.tomshardware.com/news/
        # nvidias-latest-regulation-compliant-gpu-for-china-has-been-delayed-to-early-next-year
        return 148e12
    elif "GB200" in device_name or "GB300" in device_name:
        # Grace Blackwell Superchips (Grace CPU + Blackwell GPU)
        # BF16 dense per GPU: 2,500 TFLOPS (half of 5,000 TFLOPS with sparsity)
        # GB200 data from https://www.nvidia.com/en-us/data-center/dgx-gb200
        # GB300 data from https://www.nvidia.com/en-us/data-center/dgx-gb300
        return 2.5e15
    elif "B300" in device_name or "B200" in device_name:
        # data from https://resources.nvidia.com/en-us-blackwell-architecture
        # Checked after GB300 to avoid false match on "GB300"
        return 2.25e15
    elif "MI350X" in device_name:
        # MI350X data from https://www.amd.com/en/products/accelerators/instinct/mi350/mi350x.html
        return 2300e12
    elif "MI355X" in device_name:
        # MI355X data from https://www.amd.com/en/products/accelerators/instinct/mi350/mi355x.html
        return 2500e12
    elif "MI300X" in device_name or "MI325X" in device_name:
        # MI300X data from https://www.amd.com/en/products/accelerators/instinct/mi300/mi300x.html
        # MI325X data from https://www.amd.com/en/products/accelerators/instinct/mi300/mi325x.html
        return 1300e12
    elif "MI250X" in device_name:
        # data from https://www.amd.com/en/products/accelerators/instinct/mi200/mi250x.html (per GCD)
        return 191.5e12
    elif "Data Center GPU Max 1550" in device_name:
        # Also known as Ponte Vecchio (PVC).
        # data from https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/intel-xe-gpu-architecture.html
        # Dot Product Accumulate Systolic (DPAS):
        # - Freq: 1300MHz
        # - #ops: 512
        # Full EU mode (i.e. 512 max compute units): 340.8 TFLOPS (BF16)
        # Standard EU mode (i.e. 448 max compute units): 298.2 TFLOPS (BF16)
        max_comp_units = torch.xpu.get_device_properties("xpu").max_compute_units
        return 512 * max_comp_units * 1300 * 10**6
    elif "l40s" in device_name.casefold():
        # data from: "https://resources.nvidia.com/en-us-l40s/l40s-datasheet-28413"
        return 362e12
    elif "neuron" in device_name:
        # AWS Trainium/Inferentia: query chip type via torch.neuron
        # TensorEngine BF16 TFLOPS per NeuronCore × default Logical NeuronCore (LNC) count per device
        neuron_device_name = device_module.get_device_properties().name
        if neuron_device_name in ("trn1", "trn1n", "inf2"):
            # NeuronCore-v2 TensorEngine: 90 BF16 TFLOPS/core, LNC=1
            # https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/neuron-core-v2.html
            return 90e12 * 1
        elif neuron_device_name in ("trn2", "trn2n", "trn2u", "trn3", "trn3u"):
            # NeuronCore-v3/NeuronCore-v4 TensorEngine: 79 BF16 TFLOPS/core, LNC=2
            # https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/neuron-core-v3.html
            # https://awsdocs-neuron.readthedocs-hosted.com/en/latest/about-neuron/arch/neuron-hardware/neuron-core-v4.html
            return 79e12 * 2
        else:
            logger.warning(
                f"Unknown neuron device: {neuron_device_name}, fallback to trn2/trn3"
            )
            return 79e12 * 2

    elif device_name.startswith("TPU"):
        # Google Cloud TPU: dense BF16 matrix-engine (MXU) peak, per device.
        # Source: https://cloud.google.com/tpu/docs/system-architecture-tpu-vm
        if "v4" in device_name:
            return 275e12
        elif "v5e" in device_name:
            return 197e12
        elif "v5p" in device_name:
            return 459e12
        elif "v6e" in device_name:
            return 918e12
        elif "v7" in device_name:
            # 2307 TFLOPS is the published per-chip figure; v7 exposes each of
            # its two TensorCores as a separate device, so halve for per-device.
            return 2307e12 / 2
        else:
            logger.warning(
                f"Peak flops undefined for TPU: {device_name}, fallback to A100"
            )
            return 312e12

    else:  # for other GPU types, assume A100
        logger.warning(f"Peak flops undefined for: {device_name}, fallback to A100")
        return 312e12


@contextlib.contextmanager
def set_default_dtype(dtype: torch.dtype) -> Generator[None, None, None]:
    """
    Context manager to set torch's default dtype.

    Args:
        dtype (torch.dtype): The desired default dtype inside the context manager.

    Returns:
        ContextManager: context manager for setting default dtype.

    Example:
        >>> with set_default_dtype(torch.bfloat16):
        >>>     x = torch.tensor([1, 2, 3])
        >>>     x.dtype
        torch.bfloat16


    """
    old_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(old_dtype)

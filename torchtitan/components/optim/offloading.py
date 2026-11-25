# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import weakref
from typing import Any

import torch
from torch import Tensor
from torch.distributed.tensor import DTensor
from torch.optim import Optimizer

from torchtitan.distributed.utils import as_dtensor_like, get_local_tensor

# Rounding that wastes less pinned memory than this is accepted, e.g. for small
# allocations below pinned_max_round_threshold_mb or a small model. Power-of-two
# rounding of large moments wastes up to their size, many GiB per rank.
_PINNED_ROUNDING_SLACK_BYTES = 2**30


# TODO: Use a supported host-mapped CUDA tensor once PyTorch has one (upstream
# issue not filed yet). This relies on two undocumented behaviors: PyTorch
# trusts the device given with __cuda_array_interface__, and pinned host memory
# is usable by kernels at its host address.
class _CUDAHostMemoryView:
    """Expose a pinned CPU tensor to CUDA.

    ``torch.as_tensor`` turns this object into a CUDA tensor that aliases the
    pinned memory through ``__cuda_array_interface__``. That CUDA tensor keeps
    this object, and so the CPU tensor, alive.

    The interface has no bfloat16 type code, so it describes the memory as raw
    bytes. The caller views the bytes as the target dtype.
    """

    def __init__(self, host: Tensor, device: torch.device) -> None:
        self.host = host
        # The caching host allocator reuses the memory once ``host`` is freed, but
        # it does not see kernels that use it through this view. Wait for them
        # first. This runs before ``host`` is released. Skipped at exit, when the
        # CUDA context may already be gone.
        weakref.finalize(self, torch.accelerator.synchronize, device).atexit = False
        # CUDA Array Interface v3 (the current spec version): one uint8 ("|u1")
        # per byte, at this address, writable (read-only flag False).
        self.__cuda_array_interface__ = {
            "shape": (host.nbytes,),
            "typestr": "|u1",
            "data": (host.data_ptr(), False),
            "version": 3,
        }


def _cuda_pinned_host_zeros_like(
    local: Tensor, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    """Return zeros shaped like the CUDA tensor ``local`` in pinned CPU memory.

    Returns the memory as a CUDA tensor and as the CPU tensor it aliases.
    ``local`` must not be empty.
    """
    # The pages land on the NUMA node this process is bound to.
    host = torch.zeros(local.shape, dtype=dtype, pin_memory=True)
    # The CUDA Array Interface normally describes device memory. Pinned host
    # memory is also addressable by CUDA kernels (unified addressing), so the
    # tensor gets an explicit device.
    raw = torch.as_tensor(_CUDAHostMemoryView(host, local.device), device=local.device)
    return raw.view(dtype).view(local.shape), host


def _cuda_pinned_host_bytes() -> int:
    """Return the bytes of pinned blocks handed out by the CUDA host allocator."""
    # The stats are empty before CUDA is initialized, when nothing is pinned.
    return torch.cuda.host_memory_stats().get("active_bytes.current", 0)


def _check_cuda_pinned_host_rounding(
    *, requested_bytes: int, bytes_before: int
) -> None:
    """Raise if pinned allocations since ``bytes_before`` were rounded up.

    ``bytes_before`` is ``_cuda_pinned_host_bytes()`` before the allocations. By
    default the CUDA host allocator rounds each allocation up to a power of two.
    """
    reserved_bytes = _cuda_pinned_host_bytes() - bytes_before
    if reserved_bytes > requested_bytes + _PINNED_ROUNDING_SLACK_BYTES:
        raise ValueError(
            "Pinned host memory was rounded up: requested "
            f"{requested_bytes / 2**30:.2f} GiB, reserved "
            f"{reserved_bytes / 2**30:.2f} GiB. Add pinned_max_round_threshold_mb:1 "
            "to PYTORCH_ALLOC_CONF."
        )


def _as_cpu_like(moment: Tensor, host: Tensor) -> Tensor:
    """Wrap the CPU tensor ``host`` like ``moment``, without copying it.

    ``DTensor.from_local`` would copy a CPU tensor to the mesh device.
    """
    if not isinstance(moment, DTensor):
        return host
    # TODO: Use a public API once DTensor can wrap a local tensor on another device
    # without a copy (upstream issue not filed yet). FSDP2 CPU offload uses this
    # private constructor the same way.
    return DTensor(host, moment._spec, requires_grad=False)


def _register_cpu_moment_hooks(
    optimizer: torch.optim.Optimizer, cpu_moments: dict[Tensor, dict[str, Tensor]]
) -> None:
    """Save and load pinned host moments through their CPU tensors.

    ``cpu_moments`` maps each parameter with pinned host moments to its CPU
    moments. Other parameters keep the default behavior. ``state_dict()`` returns
    the CPU moments, so checkpointers copy them on the CPU. A CUDA copy between two
    host buffers is synchronous and much slower. ``load_state_dict()`` keeps the
    CUDA moments in the state and copies loaded values into them, unless the values
    were loaded in place into the CPU moments. Without that, it would move the CPU
    tensors to the parameter device.
    """

    def params() -> list[Tensor]:
        return [param for group in optimizer.param_groups for param in group["params"]]

    def use_cpu_moments(
        _optimizer: Optimizer, state_dict: dict[str, Any]
    ) -> dict[str, Any]:
        # Checkpointers copy these CPU tensors on the host, outside stream order.
        # Wait for queued optimizer steps that write the moments.
        torch.accelerator.current_stream().synchronize()
        for index, param in enumerate(params()):
            if param in cpu_moments:
                state = state_dict["state"][index]
                state_dict["state"][index] = {**state, **cpu_moments[param]}
        return state_dict

    def load_into_pinned_memory(
        _optimizer: Optimizer, state_dict: dict[str, Any]
    ) -> dict[str, Any]:
        param_list = params()
        state = dict(state_dict["state"])
        for index, saved in state.items():
            param = param_list[index]
            if param not in cpu_moments:
                continue
            saved = dict(saved)
            for key, cpu_moment in cpu_moments[param].items():
                moment = optimizer.state[param][key]
                if saved[key] is not cpu_moment:
                    # Copy on the stream, after pending optimizer steps. A CPU
                    # copy into the pinned memory would race with them.
                    moment.copy_(saved[key])
                saved[key] = moment
            state[index] = saved
        return {**state_dict, "state": state}

    optimizer.register_state_dict_post_hook(use_cpu_moments)
    optimizer.register_load_state_dict_pre_hook(load_into_pinned_memory)


def init_cpu_offloaded_moments(
    optimizer: torch.optim.Optimizer,
    *,
    fraction: float,
    moment_dtype: torch.dtype | None,
) -> None:
    """Create the Adam state of the first parameters, with moments in pinned memory.

    Parameters are taken in optimizer order until their moments reach ``fraction``
    of all local moment bytes. Each gets the state ``init_optim_state`` would
    create, except that ``exp_avg`` and ``exp_avg_sq`` live in pinned CPU memory
    that CUDA kernels can access. ``init_optim_state`` skips parameters with
    state, so DCP loads into the pinned memory directly. The moment addresses never
    change, so a captured optimizer graph stays valid. The other parameters get
    device moments lazily, in the same fused step. ``moment_dtype=None`` keeps the
    parameter dtype.
    """
    params = [
        (name, param)
        for group in optimizer.param_groups
        for name, param in zip(group["param_names"], group["params"], strict=True)
    ]

    def moment_dtype_of(local: Tensor) -> torch.dtype:
        return moment_dtype or local.dtype

    total_bytes = sum(
        2 * local.numel() * moment_dtype_of(local).itemsize
        for local in (get_local_tensor(param) for _, param in params)
    )
    target_bytes = fraction * total_bytes
    offloaded_bytes = 0
    bytes_before = _cuda_pinned_host_bytes()
    cpu_moments: dict[Tensor, dict[str, Tensor]] = {}
    for name, param in params:
        if offloaded_bytes >= target_bytes:
            break
        local = get_local_tensor(param)
        # An empty shard needs no memory and keeps lazy device moments.
        if local.numel() == 0:
            continue
        if local.device.type != "cuda":
            raise ValueError(
                "CPU-offloaded Adam moments require CUDA parameters, got "
                f"{local.device}. Set moment_cpu_offload_fraction=0 with "
                "parallelism.fsdp_enable_cpu_offloading or create_seed_checkpoint."
            )
        if not local.is_contiguous():
            raise ValueError(
                "CPU-offloaded Adam moments require contiguous parameters, got "
                f"strides {local.stride()} for {name} with "
                f"shape {tuple(local.shape)}"
            )
        state = optimizer.state[param]
        state["step"] = torch.zeros((), dtype=torch.float32, device=local.device)
        cpu_moments[param] = {}
        for key in ("exp_avg", "exp_avg_sq"):
            moment, host = _cuda_pinned_host_zeros_like(local, moment_dtype_of(local))
            offloaded_bytes += host.nbytes
            state[key] = as_dtensor_like(param, moment)
            cpu_moments[param][key] = _as_cpu_like(state[key], host)
    _check_cuda_pinned_host_rounding(
        requested_bytes=offloaded_bytes, bytes_before=bytes_before
    )
    _register_cpu_moment_hooks(optimizer, cpu_moments)

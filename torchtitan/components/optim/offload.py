# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Optimizer-state CPU offload."""

import logging
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, overload

import torch
import torch.cuda._pin_memory_utils as pin_memory_utils
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._dtensor_spec import DTensorSpec, TensorMeta
from torch.optim import Optimizer

from torchtitan.config import Configurable
from torchtitan.distributed.utils import get_local_tensor

logger = logging.getLogger(__name__)

__all__ = ["OptimizerStateOffloader"]

_MOMENT_KEYS = ("exp_avg", "exp_avg_sq")

# Target bytes of both moments per chunk. step() holds two slots of the largest chunk on the
# GPU, and a parameter shard larger than this is a chunk of its own, so it sets the floor.
_CHUNK_SIZE_BYTES = 256 * 2**20


class OptimizerStateOffloader(Optimizer, Configurable):
    """Run fused Adam/AdamW with its moments in pinned CPU memory between steps.

    Definitions:
        Slab: one pinned CPU buffer per moment. Each parameter's moment is a view into it.
        Chunk: whole local parameter shards with about ``_CHUNK_SIZE_BYTES`` of moments.
        Slot: GPU memory for one chunk's two moments. ``step()`` uses two slots.

    ``step()`` streams the chunks through the two slots. While Adam updates one slot, the
    other stores its previous chunk and loads the next one, on separate H2D and D2H streams::

                       warm-up | chunk 0       | chunk 1           | chunk 2
        GPU compute:           | Adam 0 in A   | Adam 1 in B       | Adam 2 in A
        slot A copies: load 0  |               | store 0 -> load 2 |
        slot B copies:         | load 1        |                   | store 1 -> load 3

    Two streams vs one stream (load, update, store in order) on Qwen3-4B FSDP=2 rank
    shards, one GB300: fp32 moments 155 vs 176 ms per step, bf16 moments 65 vs 97 ms.

    Launch with ``torchrun --numa-binding=node`` so the slabs land on the GPU's NUMA node.

    Args:
        config: Has no options.
        optimizer: A fused ``torch.optim.Adam`` or ``torch.optim.AdamW``.
        state_dtype: dtype of the moments (``torch.bfloat16`` under ``moment_dtype="bfloat16"``).

    Example:

        AdamW.Config(pattern=r".*", state_offload=OptimizerStateOffloader.Config())
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        pass

    def __init__(
        self,
        config: Config,
        *,
        optimizer: Optimizer,
        state_dtype: torch.dtype,
    ) -> None:
        # Pinning goes through cudaHostRegister, and each moment is a contiguous slab view
        # that must match its parameter's memory layout.
        self.optimizer = optimizer
        for group in optimizer.param_groups:
            for param in group["params"]:
                if param.device.type != "cuda":
                    raise ValueError(
                        "state_offload requires CUDA-resident parameters, got "
                        f"{param.device}; unset it for training.enable_cpu_offload or "
                        "create_seed_checkpoint runs"
                    )
                if not get_local_tensor(param).is_contiguous():
                    raise ValueError(
                        "state_offload requires contiguous parameters, got strides "
                        f"{get_local_tensor(param).stride()} for shape "
                        f"{tuple(get_local_tensor(param).shape)}"
                    )

        # Register the wrapped optimizer's own group dicts, then share its state dict, so that
        # narrowing a group or binding a state tensor is visible to both objects.
        super().__init__(optimizer.param_groups, optimizer.defaults)
        self.param_groups = optimizer.param_groups
        self.state = optimizer.state

        # Partition parameters, allocate their canonical CPU state, then create the
        # independent streams that move chunks in each direction.
        params = [param for group in self.param_groups for param in group["params"]]
        self._chunks = _pack_params_by_state_bytes(
            params, chunk_size_bytes=_CHUNK_SIZE_BYTES, state_dtype=state_dtype
        )
        self._allocate_pinned_state(params, state_dtype)
        self._h2d_stream = torch.Stream()
        self._d2h_stream = torch.Stream()

        largest_chunk_mib = max(_chunk_numel(chunk) for chunk in self._chunks)
        largest_chunk_mib *= len(_MOMENT_KEYS) * state_dtype.itemsize / 2**20
        pinned_gb = sum(slab.nbytes for slab in self._slabs.values()) / 1e9
        logger.info(
            f"Optimizer-state offload: {len(params)} params in {len(self._chunks)} chunks "
            f"(largest {largest_chunk_mib:.0f} MiB; step() holds two GPU slots), "
            f"{pinned_gb:.2f} GB pinned host memory per rank"
        )

    @overload
    def step(self, closure: None = None) -> None:
        ...

    @overload
    def step(self, closure: Callable[[], float]) -> float:
        ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Run one optimizer step while streaming moment chunks through the GPU.

        1. Select parameters with gradients and load the first moment chunk.
        2. Alternate slots, overlapping each update with the surrounding copies.
        3. Restore the full parameter groups and wait for CPU moments to be current.

        Args:
            closure: Unsupported; it must be ``None``.
        """
        if closure is not None:
            raise ValueError("OptimizerStateOffloader does not support closures")
        compute_stream = torch.accelerator.current_stream()

        # Skip chunks without gradients. Adam skips grad-less parameters inside a chunk,
        # so their moments round-trip unchanged.
        chunks = [
            chunk
            for chunk in self._chunks
            if any(param.grad is not None for param in chunk)
        ]
        if not chunks:
            return

        # Two slots, allocated on the compute stream so the caching allocator can hand the
        # memory back to forward/backward after the step. Freed only after the side streams
        # are synchronized below, so no cross-stream lifetime hazard remains.
        max_numel = max(_chunk_numel(chunk) for chunk in chunks)
        state_dtype = self.state[chunks[0][0]]["exp_avg"].dtype
        device = chunks[0][0].device
        slots = [
            {
                key: torch.empty(max_numel, dtype=state_dtype, device=device)
                for key in _MOMENT_KEYS
            }
            for _ in range(2)
        ]
        slot_free = [self._d2h_stream.record_event(), self._d2h_stream.record_event()]
        self._h2d_stream.wait_stream(compute_stream)

        def h2d(chunk_index: int) -> torch.Event:
            slot = slots[chunk_index % 2]
            chunk = chunks[chunk_index]
            start, numel = self._slab_offsets[chunk[0]], _chunk_numel(chunk)
            with self._h2d_stream:
                self._h2d_stream.wait_event(slot_free[chunk_index % 2])
                for key in _MOMENT_KEYS:
                    slot[key][:numel].copy_(
                        self._slabs[key][start : start + numel], non_blocking=True
                    )
                return self._h2d_stream.record_event()

        # Prime slot A, then alternate slots so the next H2D and previous D2H overlap
        # the current chunk's optimizer kernels.
        group_params = [list(group["params"]) for group in self.param_groups]
        h2d_done = h2d(0)
        for chunk_index, chunk in enumerate(chunks):
            slot = slots[chunk_index % 2]
            compute_stream.wait_event(h2d_done)
            if chunk_index + 1 < len(chunks):
                h2d_done = h2d(chunk_index + 1)

            # Bind slot views as the state, step only this chunk with the unchanged optimizer,
            # then copy the updated moments back and rebind the pinned CPU state.
            cpu_state = {
                param: {key: self.state[param][key] for key in _MOMENT_KEYS}
                for param in chunk
            }
            offset = 0
            for param in chunk:
                numel = get_local_tensor(param).numel()
                for key in _MOMENT_KEYS:
                    view = slot[key][offset : offset + numel].view(
                        get_local_tensor(param).shape
                    )
                    self.state[param][key] = _state_like(param, view)
                offset += numel
            chunk_set = set(map(id, chunk))
            for group, params in zip(self.param_groups, group_params):
                group["params"] = [p for p in params if id(p) in chunk_set]
            self.optimizer.step()

            self._d2h_stream.wait_stream(compute_stream)
            start, numel = self._slab_offsets[chunk[0]], _chunk_numel(chunk)
            with self._d2h_stream:
                for key in _MOMENT_KEYS:
                    self._slabs[key][start : start + numel].copy_(
                        slot[key][:numel], non_blocking=True
                    )
                slot_free[chunk_index % 2] = self._d2h_stream.record_event()
            for param in chunk:
                self.state[param].update(cpu_state[param])

        for group, params in zip(self.param_groups, group_params):
            group["params"] = params
        # The CPU moments are current for checkpointing, and no copy still uses the slots
        # when they are freed.
        self._h2d_stream.synchronize()
        self._d2h_stream.synchronize()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Copy a loaded state dict into the existing pinned state, in place.

        Args:
            state_dict: Integer-keyed optimizer state produced by
                ``_unflatten_optim_state_dict`` in parameter order.

        Note:
            The base optimizer implementation would move state tensors to the parameter
            device. This override preserves the pinned slabs and copies values into them.
        """
        params = [param for group in self.param_groups for param in group["params"]]

        # Copy checkpoint values into the existing slab views instead of replacing
        # them with independently allocated tensors.
        for param_index, loaded in state_dict["state"].items():
            state = self.state[params[param_index]]
            for key, value in loaded.items():
                get_local_tensor(state[key]).copy_(get_local_tensor(value))
        for group, loaded_group in zip(self.param_groups, state_dict["param_groups"]):
            for key, value in loaded_group.items():
                if key not in ("params", "param_names"):
                    group[key] = value

    def _allocate_pinned_state(
        self, params: list[torch.Tensor], state_dtype: torch.dtype
    ) -> None:
        """One pinned slab per moment, each parameter's moment a view into it."""
        # Pin pageable memory, as DCP's async stager does: pin_memory=True would round each
        # slab up to a power of two, up to 2x the host memory.
        # TODO: torch.cuda._pin_memory_utils is private and CUDA-only. Move to in-place pinning
        # once PyTorch has it: https://github.com/pytorch/pytorch/issues/32167
        total_numel = sum(get_local_tensor(param).numel() for param in params)
        self._slabs = {}
        for key in _MOMENT_KEYS:
            slab = torch.zeros(total_numel, dtype=state_dtype)
            # A rank whose shards are all empty has a 0-byte slab with a null pointer,
            # which cudaHostRegister rejects.
            if slab.numel() > 0:
                pin_memory_utils.pin_memory(slab.data_ptr(), slab.nbytes)
                # Unpin when the slab is freed, but not at exit, after the CUDA context is gone.
                unpin = weakref.finalize(
                    slab, pin_memory_utils.unpin_memory, slab.data_ptr()
                )
                unpin.atexit = False
            self._slabs[key] = slab
        self._slab_offsets: dict[torch.Tensor, int] = {}

        # Recreate optimizer.state with parameter-shaped views into those slabs.
        offset = 0
        for param in params:
            local = get_local_tensor(param)
            state = self.state[param]
            self._slab_offsets[param] = offset
            state["step"] = torch.zeros((), dtype=torch.float32, device=param.device)
            for key in _MOMENT_KEYS:
                view = self._slabs[key][offset : offset + local.numel()].view(
                    local.shape
                )
                state[key] = _state_like(param, view)
            offset += local.numel()


def _pack_params_by_state_bytes(
    params: list[torch.Tensor], *, chunk_size_bytes: int, state_dtype: torch.dtype
) -> list[list[torch.Tensor]]:
    """Group parameter shards into transfer chunks by Adam-moment bytes.

    Each local parameter element has two moments, so a parameter contributes
    ``numel * 2 * state_dtype.itemsize`` bytes. Parameters stay in optimizer
    order and are never split: the next parameter starts a new chunk if adding
    it would exceed ``chunk_size_bytes``. A parameter larger than the target
    occupies its own chunk.

    Example:

        # fp32: two 4-byte moments = 8 bytes per parameter element.
        params = [torch.empty(60_000) for _ in range(3)]
        chunks = _pack_params_by_state_bytes(
            params, chunk_size_bytes=1024 * 1024, state_dtype=torch.float32
        )
        [len(chunk) for chunk in chunks]
        # -> [2, 1] because each parameter contributes 480,000 bytes.
    """
    bytes_per_element = len(_MOMENT_KEYS) * state_dtype.itemsize
    chunks: list[list[torch.Tensor]] = []
    current: list[torch.Tensor] = []
    current_bytes = 0

    # Preserve optimizer order while filling each chunk up to the byte target.
    for param in params:
        param_bytes = get_local_tensor(param).numel() * bytes_per_element
        if current and current_bytes + param_bytes > chunk_size_bytes:
            chunks.append(current)
            current, current_bytes = [], 0
        current.append(param)
        current_bytes += param_bytes
    if current:
        chunks.append(current)
    return chunks


def _chunk_numel(chunk: list[torch.Tensor]) -> int:
    return sum(get_local_tensor(param).numel() for param in chunk)


def _state_like(param: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
    """Give `local` (CPU or GPU) the parameter's mesh and placements so DCP saves it sharded.

    ``DTensor.from_local`` would move CPU state onto the CUDA mesh. The direct
    constructor keeps it on CPU and records the moment dtype in its metadata.

    TODO: the DTensor constructor and ``DTensorSpec`` are private (FSDP2's CPU offload
    uses them the same way). Switch once PyTorch has a public way to build a DTensor
    whose local tensor stays on the CPU.
    """
    if not isinstance(param, DTensor):
        return local
    if local.device.type == "cpu":
        # A DTensor inherits its local's storage offset. DCP's async stager reads a non-zero
        # data_ptr() as a dense tensor, so alias the slab view at offset 0.
        local = torch.from_dlpack(local)
    spec = DTensorSpec(
        mesh=param.device_mesh,
        placements=param.placements,
        tensor_meta=TensorMeta(
            shape=param.shape, stride=param.stride(), dtype=local.dtype
        ),
    )
    return DTensor(local, spec, requires_grad=False)

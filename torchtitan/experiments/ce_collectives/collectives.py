# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FSDP2 adapters for copy-engine communication within an NVLink domain."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
import triton

# TODO: Use public exports for the FSDP communication interfaces when available:
# https://github.com/pytorch/pytorch/pull/155189.
from torch.distributed.fsdp._fully_shard._fsdp_api import AllGather, ReduceScatter

from .kernels import barrier_kernel, local_sum_kernel


@dataclass
class _RegisteredBuffer:
    handle: Any
    buffer_ptrs: list[int]
    signal_pad_ptrs: torch.Tensor
    multicast_ptr: int


class _SymmetricAllocator:
    def __init__(self, group: dist.ProcessGroup, *, multicast: bool) -> None:
        if not 2 <= group.size() <= 72:
            raise ValueError("An NVLink group must contain 2-72 ranks.")
        if symm_mem.get_backend(torch.device("cuda")) != "CUDA":
            raise RuntimeError(
                "CE collectives require the CUDA symmetric-memory backend."
            )
        self.group = group
        self.multicast = multicast
        # A fake process group has no peers to rendezvous with. Its buffers still
        # come from the symmetric pool so fake runs report the CE allocations.
        self.fake = dist.get_backend(group) == "fake"
        self.device: torch.device | None = None
        self.pool: torch.cuda.MemPool | None = None
        # None marks a fake-group buffer: allocated, but without peer mappings.
        self.buffers: dict[int, _RegisteredBuffer | None] = {}

    def allocate(self, size, *, dtype, device) -> torch.Tensor:
        if device.type != "cuda" or torch.version.hip is not None:
            raise ValueError("CE collectives require NVIDIA CUDA devices.")
        if self.device is not None and device != self.device:
            raise ValueError("A CE communicator must use one local device.")
        if self.pool is None:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Warm up CE collectives before graph capture.")
            self.device = device
            self.pool = torch.cuda.MemPool(
                symm_mem.get_mempool_allocator(device),
                use_on_oom=False,
                no_split=True,
            )
        with torch.cuda.use_mem_pool(self.pool):
            tensor = torch.empty(size, dtype=dtype, device=device)
        pointer = tensor.data_ptr()
        if pointer not in self.buffers:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "Unwarmed CE buffer: warm up all shapes and the FSDP prefetch schedule."
                )
            if self.fake:
                self.buffers[pointer] = None
                return tensor
            handle = symm_mem.rendezvous(tensor, group=self.group)
            if self.multicast and not handle.multicast_ptr:
                raise RuntimeError(
                    "All-gather requires one multicast-capable NVLink domain."
                )
            offset = pointer - handle.buffer_ptrs[self.group.rank()]
            self.buffers[pointer] = _RegisteredBuffer(
                handle,
                [peer + offset for peer in handle.buffer_ptrs],
                torch.tensor(handle.signal_pad_ptrs, dtype=torch.uint64, device=device),
                handle.multicast_ptr + offset if handle.multicast_ptr else 0,
            )
        return tensor


class CEWorkspace:
    """Bounded exchange storage shared by collectives on one ordered stream."""

    def __init__(self, group: dist.ProcessGroup, *, size_bytes: int) -> None:
        if size_bytes <= 0:
            raise ValueError("CE workspace size must be positive.")
        self.group = group
        self.size_bytes = size_bytes
        self._allocator = _SymmetricAllocator(group, multicast=False)
        self._buffer: torch.Tensor | None = None

    def get_buffer(self, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if self._buffer is None:
            self._buffer = self._allocator.allocate(
                (self.size_bytes,), dtype=torch.uint8, device=device
            )
        if device != self._buffer.device:
            raise ValueError("A CE workspace must use one local device.")
        return self._buffer.view(dtype)

    @property
    def registered_buffer(self) -> _RegisteredBuffer | None:
        assert self._buffer is not None
        return self._allocator.buffers[self._buffer.data_ptr()]


def _barrier(buffer: _RegisteredBuffer, group: dist.ProcessGroup, channel: int) -> None:
    block_size = max(32, triton.next_power_of_2(group.size()))
    cast(Any, barrier_kernel)[(1,)](
        buffer.signal_pad_ptrs,
        group.rank(),
        group.size(),
        channel,
        block_size,
        num_warps=block_size // 32,
    )


def _driver():
    try:
        import cuda.bindings.driver as driver
    except ImportError as error:
        raise ImportError("CE collectives require cuda-bindings>=12.3.") from error
    return driver


def _copy(driver, destination: int, source: int, size: int, stream: int) -> None:
    if size and destination != source:
        (result,) = driver.cuMemcpyAsync(destination, source, size, stream)
        if result != driver.CUresult.CUDA_SUCCESS:
            raise RuntimeError(f"CE cuMemcpyAsync failed: {result}")


class CEMulticastAllGather(AllGather):
    """One multicast CE copy bracketed by two single-CTA barriers.

    FSDP owns copy-in, copy-out, streams and completion events. Calls must follow
    the same allocation and collective order on every rank. Warm up the entire
    workload, including prefetch, before CUDA graph capture.
    """

    def __init__(self, group: dist.ProcessGroup) -> None:
        self.group = group
        self._driver = _driver()
        self._allocator = _SymmetricAllocator(group, multicast=True)

    def allocate(
        self,
        size: Sequence[int | torch.SymInt],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        return self._allocator.allocate(size, dtype=dtype, device=device)

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        async_op: bool = False,
        *,
        workspace: CEWorkspace | None = None,
    ) -> dist.Work | None:
        shard_bytes = input_tensor.numel() * input_tensor.element_size()
        if (
            group is not self.group
            or not input_tensor.is_contiguous()
            or not output_tensor.is_contiguous()
            or input_tensor.dtype != output_tensor.dtype
            or input_tensor.device != output_tensor.device
            or output_tensor.numel() != input_tensor.numel() * group.size()
            or input_tensor.data_ptr()
            != output_tensor.data_ptr() + group.rank() * shard_bytes
        ):
            raise ValueError(
                "CE all-gather requires contiguous, equal, in-place rank shards."
            )
        if workspace is None:
            buffer = self._allocator.buffers[output_tensor.data_ptr()]
        else:
            buffer = workspace.registered_buffer
            if (
                workspace.group is not group
                or output_tensor.data_ptr() not in workspace._allocator.buffers
            ):
                raise ValueError("CE all-gather must use its group's workspace.")
            if buffer is not None and not buffer.multicast_ptr:
                raise RuntimeError(
                    "All-gather requires one multicast-capable NVLink domain."
                )
        stream = torch.cuda.current_stream(output_tensor.device)
        if buffer is None:
            # Fake group: match its NCCL all-gather, which copies the local shard
            # into every rank's slot. _copy skips this rank's own slot.
            for rank in range(group.size()):
                _copy(
                    self._driver,
                    output_tensor.data_ptr() + rank * shard_bytes,
                    input_tensor.data_ptr(),
                    shard_bytes,
                    stream.cuda_stream,
                )
            return None
        _barrier(buffer, group, 0)
        _copy(
            self._driver,
            buffer.multicast_ptr + group.rank() * shard_bytes,
            input_tensor.data_ptr(),
            shard_bytes,
            stream.cuda_stream,
        )
        _barrier(buffer, group, 1)
        # FSDP orders pool reuse after copy-out. record_stream() would instead
        # make allocation reuse depend on rank-local GPU progress.
        return None


def domain_rank_groups(
    world_size: int, ranks_per_domain: int
) -> tuple[list[list[int]], list[list[int]]]:
    """Return domain and same-position groups in parent-process-group rank order."""
    if not 2 <= ranks_per_domain <= 72 or world_size % ranks_per_domain:
        raise ValueError(
            "ranks_per_domain must be 2-72 and divide the FSDP group size."
        )
    domains = [
        list(range(start, start + ranks_per_domain))
        for start in range(0, world_size, ranks_per_domain)
    ]
    lanes = [
        list(range(lane, world_size, ranks_per_domain))
        for lane in range(ranks_per_domain)
    ]
    return domains, lanes


def _local_sum(
    input_tensor: torch.Tensor, output: torch.Tensor, num_sources: int, divisor: int = 1
) -> None:
    if output.numel():
        cast(Any, local_sum_kernel)[(triton.cdiv(output.numel(), 1024),)](
            input_tensor,
            output,
            output.numel(),
            num_sources,
            divisor,
            1024,
            num_warps=4,
            enable_fp_fusion=False,
        )


class CEAllToAllReduceScatter(ReduceScatter):
    """CE all-to-all + local sum, followed by a same-position NCCL exchange + sum.

    Parent ranks must be domain-major: rank = domain * ranks_per_domain + lane.
    Within one domain only the first exchange and sum run. The local sums use
    a fixed source order with FP32 compensated accumulation. Inter-domain
    partials remain FP32. Only the barriers are single-CTA; sums use GPU SMs.
    """

    def __init__(self, group: dist.ProcessGroup, *, ranks_per_domain: int) -> None:
        domains, lanes = domain_rank_groups(group.size(), ranks_per_domain)
        self.group = group
        self.num_domains = len(domains)
        self.local_group = group
        self.lane_group: dist.ProcessGroup | None = None
        if self.num_domains > 1:
            # split_group requires the parent NCCL communicator to be initialized.
            dist.barrier(group=group)
            self.local_group = cast(dist.ProcessGroup, dist.split_group(group, domains))
            self.lane_group = cast(dist.ProcessGroup, dist.split_group(group, lanes))
        self._driver = _driver()
        self._allocator = _SymmetricAllocator(self.local_group, multicast=False)

    def allocate(
        self,
        size: Sequence[int | torch.SymInt],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> torch.Tensor:
        # FSDP allocates both the full input and the final gradient here. Only
        # the exchange scratch needs peer mappings; gradients stay ordinary.
        return torch.empty(size, dtype=dtype, device=device)

    def __call__(
        self,
        output_tensor: torch.Tensor,
        input_tensor: torch.Tensor,
        group: dist.ProcessGroup,
        op: dist.ReduceOp,
        async_op: bool = False,
        *,
        workspace: CEWorkspace | None = None,
    ) -> dist.Work | None:
        if (
            group is not self.group
            or op not in (dist.ReduceOp.SUM, dist.ReduceOp.AVG)
            or input_tensor.dtype not in (torch.float32, torch.float16, torch.bfloat16)
            or input_tensor.dtype != output_tensor.dtype
            or input_tensor.device != output_tensor.device
            or not input_tensor.is_contiguous()
            or not output_tensor.is_contiguous()
            or input_tensor.numel() != output_tensor.numel() * group.size()
        ):
            raise ValueError(
                "CE reduce-scatter requires equal contiguous floating shards and SUM or AVG."
            )
        if not output_tensor.numel():
            return None
        local_group = self.local_group
        if workspace is None:
            storage = self._allocator.allocate(
                (input_tensor.numel(),),
                dtype=input_tensor.dtype,
                device=input_tensor.device,
            )
            buffer = self._allocator.buffers[storage.data_ptr()]
        else:
            if workspace.group is not local_group:
                raise ValueError(
                    "CE reduce-scatter must use its local group's workspace."
                )
            storage = workspace.get_buffer(
                dtype=input_tensor.dtype, device=input_tensor.device
            )
            buffer = workspace.registered_buffer
        chunk_capacity = storage.numel() // group.size()
        if not chunk_capacity:
            raise ValueError("CE workspace must hold at least one element per rank.")
        stream = torch.cuda.current_stream(input_tensor.device)
        divisor = group.size() if op == dist.ReduceOp.AVG else 1
        for offset in range(0, output_tensor.numel(), chunk_capacity):
            chunk_numel = min(chunk_capacity, output_tensor.numel() - offset)
            chunk_bytes = chunk_numel * input_tensor.element_size()
            received = storage[: group.size() * chunk_numel]
            output_chunk = output_tensor.flatten().narrow(0, offset, chunk_numel)
            if buffer is None:
                # Match fake reduce-scatter while retaining the real scratch
                # allocation and local sums for rank-local memory debugging.
                received.zero_()
                lane = local_group.rank()
                domain = group.rank() // local_group.size()
                _copy(
                    self._driver,
                    received.data_ptr()
                    + (lane * self.num_domains + domain) * chunk_bytes,
                    input_tensor.data_ptr()
                    + (group.rank() * output_tensor.numel() + offset)
                    * input_tensor.element_size(),
                    chunk_bytes,
                    stream.cuda_stream,
                )
            else:
                _barrier(buffer, local_group, 0)
                # Input [destination_domain, destination_lane, shard] is delivered
                # as [source_lane, destination_domain, chunk], without packing.
                for step in range(local_group.size()):
                    peer = (local_group.rank() + step) % local_group.size()
                    for domain in range(self.num_domains):
                        source = (
                            input_tensor.data_ptr()
                            + (
                                (domain * local_group.size() + peer)
                                * output_tensor.numel()
                                + offset
                            )
                            * input_tensor.element_size()
                        )
                        destination = (
                            buffer.buffer_ptrs[peer]
                            + (local_group.rank() * self.num_domains + domain)
                            * chunk_bytes
                        )
                        _copy(
                            self._driver,
                            destination,
                            source,
                            chunk_bytes,
                            stream.cuda_stream,
                        )
                _barrier(buffer, local_group, 1)
            if self.lane_group is None:
                _local_sum(received, output_chunk, local_group.size(), divisor)
            else:
                partial = torch.empty(
                    self.num_domains * chunk_numel,
                    dtype=torch.float32,
                    device=input_tensor.device,
                )
                _local_sum(received, partial, local_group.size())
                exchanged = torch.empty_like(partial)
                # NCCL Work.wait() inserts a dependency onto the caller's stream;
                # network traffic cannot use direct NVLink peer-memory CE copies.
                work = dist.all_to_all_single(
                    exchanged, partial, group=self.lane_group, async_op=True
                )
                assert work is not None
                work.wait()
                _local_sum(exchanged, output_chunk, self.num_domains, divisor)
        # Scratch is allocated and consumed on FSDP's reduce-scatter stream.
        # The allocator can reuse it after this stream's local sum completes.
        return None

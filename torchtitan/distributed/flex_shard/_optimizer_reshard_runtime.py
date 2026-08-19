# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pipelined packed all-to-all redistribution execution.

The packed redistribution and asynchronous compute overlap are informed by
Canzona and its FSDP implementation:

- Canzona: https://arxiv.org/abs/2602.06079
- FSDP-Canzona: https://github.com/liangyuwang/FSDP-Canzona
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from types import ModuleType
from typing import Any, Generic, TypeAlias

import torch
from torch import Tensor

from ._optimizer_reshard_schedule import (
    _BucketPlan,
    _ItemT,
    _LocalBucketPlan,
    _PackedAllToAllSchedule,
    _RedistributionBucketPlan,
    _TensorRegion,
)


__all__: list[str] = []


_NUM_PIPELINE_SLOTS = 2


_BufferKey: TypeAlias = tuple[torch.device, torch.dtype]


@dataclass(slots=True)
class _BufferRequirements:
    storage_exchange_numel: int | None = None
    compute_exchange_numel: int | None = None
    compute_scratch_numel: int | None = None
    storage_scratch_numel: int | None = None

    def include_communication(
        self,
        *,
        storage_exchange_numel: int,
        compute_exchange_numel: int,
    ) -> None:
        self.storage_exchange_numel = max(
            self.storage_exchange_numel or 0,
            storage_exchange_numel,
        )
        self.compute_exchange_numel = max(
            self.compute_exchange_numel or 0,
            compute_exchange_numel,
        )

    def include_compute_scratch(self, numel: int) -> None:
        self.compute_scratch_numel = max(self.compute_scratch_numel or 0, numel)

    def include_storage_scratch(self, numel: int) -> None:
        self.storage_scratch_numel = max(self.storage_scratch_numel or 0, numel)


@dataclass(slots=True)
class _ReservedBuffers:
    storage_exchange: Tensor | None = None
    compute_exchange: Tensor | None = None
    # These remain separate because transfer-stream preparation/finalization
    # can overlap caller-stream compute within the same pipeline slot.
    compute_scratch: Tensor | None = None
    storage_scratch: Tensor | None = None


def _reserve_tensor(
    tensor: Tensor | None,
    *,
    numel: int,
    dtype: torch.dtype,
    device: torch.device,
) -> Tensor:
    if tensor is None or tensor.numel() < numel:
        return torch.empty(numel, dtype=dtype, device=device)
    return tensor


def _reserved_view(tensor: Tensor | None, numel: int) -> Tensor:
    assert tensor is not None and tensor.numel() >= numel
    return tensor[:numel]


def _execute_packed_all_to_all(
    schedule: _PackedAllToAllSchedule,
    *,
    output: Tensor,
    input: Tensor,
) -> None:
    if not schedule.has_remote_transfers:
        output[: schedule.output_buffer_numel].copy_(
            input[: schedule.input_buffer_numel]
        )
        return
    torch.distributed.all_to_all_single(
        output[: schedule.output_buffer_numel],
        input[: schedule.input_buffer_numel],
        output_split_sizes=schedule.output_split_sizes,
        input_split_sizes=schedule.input_split_sizes,
        group=schedule.process_group,
    )


@dataclass(slots=True)
class _BufferSlot:
    buffers: dict[_BufferKey, _ReservedBuffers] = field(default_factory=dict)
    _recorded_compute_streams: set[torch.Stream] = field(default_factory=set)

    def reserve(
        self,
        requirements: dict[_BufferKey, _BufferRequirements],
        *,
        device_handle: ModuleType,
        compute_stream: torch.Stream,
        transfer_stream: torch.Stream | None,
    ) -> None:
        if transfer_stream is not None:
            with device_handle.stream(transfer_stream):
                for (device, dtype), required in requirements.items():
                    reserved = self.buffers.setdefault(
                        (device, dtype), _ReservedBuffers()
                    )
                    if required.storage_exchange_numel is not None:
                        reserved.storage_exchange = _reserve_tensor(
                            reserved.storage_exchange,
                            numel=required.storage_exchange_numel,
                            dtype=dtype,
                            device=device,
                        )
                    if required.compute_exchange_numel is not None:
                        reserved.compute_exchange = _reserve_tensor(
                            reserved.compute_exchange,
                            numel=required.compute_exchange_numel,
                            dtype=dtype,
                            device=device,
                        )
                    if required.storage_scratch_numel is not None:
                        reserved.storage_scratch = _reserve_tensor(
                            reserved.storage_scratch,
                            numel=required.storage_scratch_numel,
                            dtype=dtype,
                            device=device,
                        )

        with device_handle.stream(compute_stream):
            for (device, dtype), required in requirements.items():
                if required.compute_scratch_numel is None:
                    continue
                reserved = self.buffers.setdefault((device, dtype), _ReservedBuffers())
                reserved.compute_scratch = _reserve_tensor(
                    reserved.compute_scratch,
                    numel=required.compute_scratch_numel,
                    dtype=dtype,
                    device=device,
                )
        self._recorded_compute_streams.clear()
        self.record_compute_stream(compute_stream)

    def record_compute_stream(self, stream: torch.Stream) -> None:
        # Pipeline exchange buffers stay alive and use explicit slot events.
        # Compute scratch also serves local-only work, so record each distinct
        # caller for safe destruction or growth after queued compute.
        if stream in self._recorded_compute_streams:
            return
        for reserved in self.buffers.values():
            compute_scratch = reserved.compute_scratch
            if compute_scratch is not None and compute_scratch.device.type != "cpu":
                compute_scratch.record_stream(stream)
        self._recorded_compute_streams.add(stream)

    def communication_buffers(
        self, plan: _RedistributionBucketPlan[Any]
    ) -> tuple[Tensor, Tensor]:
        to_compute = plan.storage_to_compute_schedule
        to_storage = plan.compute_to_storage_schedule
        reserved = self.buffers[(plan.device, plan.dtype)]
        return (
            _reserved_view(
                reserved.storage_exchange,
                max(
                    to_compute.input_buffer_numel,
                    to_storage.output_buffer_numel,
                ),
            ),
            _reserved_view(
                reserved.compute_exchange,
                max(
                    to_compute.output_buffer_numel,
                    to_storage.input_buffer_numel,
                ),
            ),
        )

    def compute_buffer(
        self,
        shape: torch.Size | tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        reserved = self.buffers[(device, dtype)]
        return _reserved_view(reserved.compute_scratch, math.prod(shape)).view(shape)

    def storage_partition_buffer(
        self,
        shape: torch.Size | tuple[int, ...],
        *,
        dtype: torch.dtype,
        device: torch.device,
    ) -> Tensor:
        reserved = self.buffers[(device, dtype)]
        return _reserved_view(reserved.storage_scratch, math.prod(shape)).view(shape)


@dataclass(slots=True)
class _PipelineSlot:
    """Persistent buffer keepalive and reusable cross-stream handoff events."""

    buffers: _BufferSlot
    # Transfer producer -> compute consumer.
    compute_input_ready: torch.Event
    # Compute consumer -> transfer return/reuse.
    compute_done: torch.Event
    # Transfer finalization -> subsequent caller work.
    done: torch.Event


@dataclass(slots=True)
class _BucketWork(Generic[_ItemT]):
    plan: _RedistributionBucketPlan[_ItemT]
    slot: _PipelineSlot
    storage_buffer: Tensor
    compute_fragment_buffer: Tensor


@dataclass(slots=True)
class _CommunicationContext:
    device_handle: ModuleType
    transfer_stream: torch.Stream
    slots: tuple[_PipelineSlot, ...]

    @classmethod
    def create(
        cls,
        device: torch.device,
    ) -> _CommunicationContext:
        device_handle = torch.get_device_module(device)
        transfer_stream = device_handle.Stream(device=device, priority=0)

        def create_slot() -> _PipelineSlot:
            return _PipelineSlot(
                buffers=_BufferSlot(),
                compute_input_ready=device_handle.Event(),
                compute_done=device_handle.Event(),
                done=device_handle.Event(),
            )

        return cls(
            device_handle=device_handle,
            transfer_stream=transfer_stream,
            slots=tuple(create_slot() for _ in range(_NUM_PIPELINE_SLOTS)),
        )


def _include_compute_scratch_requirement(
    requirements: dict[_BufferKey, _BufferRequirements],
    item: _ItemT,
    local_tensor_spec: Callable[[_ItemT], tuple[torch.Size, torch.dtype, torch.device]],
) -> None:
    shape, dtype, device = local_tensor_spec(item)
    requirements.setdefault(
        (device, dtype), _BufferRequirements()
    ).include_compute_scratch(math.prod(shape))


class _BucketedRedistributionRuntime(Generic[_ItemT]):
    """Execute bucket plans with a rolling communication prefetch window.

    ``reserve_buffers`` must be called after planning and after any replan. It
    eagerly grows two rolling buffer slots and creates their stream events, so
    ``run`` only reuses reserved communication resources. One slot holds the
    current bucket and the other prefetches the next redistributed bucket.

    ``prepare`` writes an optimizer input into runtime-owned scratch, ``compute``
    updates its runtime-owned input in place, and ``finalize`` consumes a
    runtime-owned result before reuse. Callbacks run under the stream selected
    by the runtime and must not retain tensors, synchronize, or call
    ``Tensor.record_stream()``. Local-only buckets are prefetch barriers, so
    no later redistributed ``prepare`` runs before an intervening local bucket.

    Any exception is fatal: parameters or optimizer state may already be
    updated and communication may be in flight, so callers must not reuse this
    runtime or optimizer.
    """

    def __init__(
        self,
        device: torch.device,
    ) -> None:
        # pyrefly: ignore [read-only]
        self._device = device
        self._context: _CommunicationContext | None = None
        self._local_slot = _BufferSlot()

    def reserve_buffers(
        self,
        plans: Sequence[_BucketPlan[_ItemT]],
        *,
        local_tensor_spec: Callable[
            [_ItemT], tuple[torch.Size, torch.dtype, torch.device]
        ],
    ) -> None:
        """Eagerly grow runtime-owned tensors to the plans' maximum sizes."""
        context = self._context
        created_context = False
        if context is None and any(
            isinstance(plan, _RedistributionBucketPlan) for plan in plans
        ):
            context = _CommunicationContext.create(self._device)
            created_context = True

        device_handle = (
            context.device_handle
            if context is not None
            else torch.get_device_module(self._device)
        )
        compute_stream = device_handle.current_stream(self._device)
        local_requirements: dict[_BufferKey, _BufferRequirements] = {}
        slot_requirements = tuple({} for _ in context.slots) if context else ()
        redistributed_index = 0
        for plan in plans:
            if isinstance(plan, _LocalBucketPlan):
                for item in plan.items:
                    _include_compute_scratch_requirement(
                        local_requirements,
                        item,
                        local_tensor_spec,
                    )
                continue

            assert context is not None
            requirements = slot_requirements[
                redistributed_index % len(slot_requirements)
            ]
            redistributed_index += 1
            to_compute = plan.storage_to_compute_schedule
            to_storage = plan.compute_to_storage_schedule
            plan_requirements = requirements.setdefault(
                (plan.device, plan.dtype), _BufferRequirements()
            )
            plan_requirements.include_communication(
                storage_exchange_numel=max(
                    to_compute.input_buffer_numel,
                    to_storage.output_buffer_numel,
                ),
                compute_exchange_numel=max(
                    to_compute.output_buffer_numel,
                    to_storage.input_buffer_numel,
                ),
            )
            participant = plan.group.local_participant
            for redistribution_plan in plan.redistribution_plans:
                storage_partition = redistribution_plan.storage_partition(participant)
                plan_requirements.include_storage_scratch(
                    math.prod(storage_partition.tensor_shape)
                )
                compute_partition = redistribution_plan.compute_partition(participant)
                if compute_numel := math.prod(compute_partition.tensor_shape):
                    plan_requirements.include_compute_scratch(compute_numel)
            for item in plan.unredistributed_items:
                _include_compute_scratch_requirement(
                    requirements,
                    item,
                    local_tensor_spec,
                )

        self._local_slot.reserve(
            local_requirements,
            device_handle=device_handle,
            compute_stream=compute_stream,
            transfer_stream=None,
        )
        if context is not None:
            for slot, requirements in zip(
                context.slots, slot_requirements, strict=True
            ):
                slot.buffers.reserve(
                    requirements,
                    device_handle=device_handle,
                    compute_stream=compute_stream,
                    transfer_stream=context.transfer_stream,
                )
            if created_context:
                for slot in context.slots:
                    slot.compute_input_ready.record(context.transfer_stream)
                    slot.compute_done.record(compute_stream)
                    slot.done.record(context.transfer_stream)
            self._context = context

    def run(
        self,
        plans: Sequence[_BucketPlan[_ItemT]],
        *,
        local_tensor_spec: Callable[
            [_ItemT], tuple[torch.Size, torch.dtype, torch.device]
        ],
        prepare: Callable[[_ItemT, Tensor], None],
        compute: Callable[[_ItemT, Tensor], None],
        finalize: Callable[[_ItemT, Tensor], None],
    ) -> None:
        context = self._context
        handle = (
            context.device_handle
            if context is not None
            else torch.get_device_module(self._device)
        )
        caller = handle.current_stream(self._device)
        self._local_slot.record_compute_stream(caller)
        if context is not None:
            for slot in context.slots:
                slot.buffers.record_compute_stream(caller)

        transfer_stream: torch.Stream | None = None
        completed = False
        previous: _BucketWork[_ItemT] | None = None
        prefetched: deque[_BucketWork[_ItemT]] = deque()
        next_plan_to_prefetch = 0
        redistributed_index = 0

        def prefetch_next(slot: _PipelineSlot) -> _BucketWork[_ItemT] | None:
            nonlocal next_plan_to_prefetch
            if next_plan_to_prefetch >= len(plans):
                return None
            next_plan = plans[next_plan_to_prefetch]
            if isinstance(next_plan, _LocalBucketPlan):
                return None
            next_plan_to_prefetch += 1
            assert context is not None and transfer_stream is not None
            return self._enqueue_storage_to_compute(
                next_plan,
                slot,
                context,
                prepare=prepare,
            )

        try:
            for plan_index, plan in enumerate(plans):
                if isinstance(plan, _LocalBucketPlan):
                    assert not prefetched
                    with handle.stream(caller):
                        self._compute_without_redistribution(
                            plan.items,
                            self._local_slot,
                            local_tensor_spec=local_tensor_spec,
                            prepare=prepare,
                            compute=compute,
                            finalize=finalize,
                        )
                    next_plan_to_prefetch = plan_index + 1
                    continue

                assert context is not None
                if transfer_stream is None:
                    context.transfer_stream.wait_stream(caller)
                    transfer_stream = context.transfer_stream
                previous_work = previous
                if prefetched:
                    assert previous_work is not None
                    work = prefetched.popleft()
                    assert work.plan is plan
                    self._enqueue_compute_to_storage(
                        previous_work, context, finalize=finalize
                    )
                else:
                    slot = context.slots[redistributed_index % len(context.slots)]
                    if previous_work is not None and slot is previous_work.slot:
                        self._enqueue_compute_to_storage(
                            previous_work, context, finalize=finalize
                        )
                        work = self._enqueue_storage_to_compute(
                            plan,
                            slot,
                            context,
                            prepare=prepare,
                        )
                    else:
                        work = self._enqueue_storage_to_compute(
                            plan,
                            slot,
                            context,
                            prepare=prepare,
                        )
                        if previous_work is not None:
                            self._enqueue_compute_to_storage(
                                previous_work, context, finalize=finalize
                            )
                    next_plan_to_prefetch = plan_index + 1

                self._compute_bucket(
                    work,
                    work.slot,
                    caller,
                    context,
                    local_tensor_spec=local_tensor_spec,
                    prepare=prepare,
                    compute=compute,
                    finalize=finalize,
                )
                if previous_work is not None:
                    self._release(previous_work, caller)
                previous = work
                redistributed_index += 1

                while len(prefetched) < len(context.slots) - 1:
                    slot = context.slots[
                        (redistributed_index + len(prefetched)) % len(context.slots)
                    ]
                    if future_work := prefetch_next(slot):
                        prefetched.append(future_work)
                    else:
                        break

            assert not prefetched
            if previous is not None:
                assert context is not None
                self._enqueue_compute_to_storage(previous, context, finalize=finalize)
                self._release(previous, caller)
            completed = True
        finally:
            if not completed and transfer_stream is not None:
                # This fatal-only cleanup intentionally has no isolated mock-stream
                # test to keep the unit-test footprint minimal. An exception makes
                # the optimizer and training job unrecoverable.
                # Preserve allocator lifetime ordering for work already queued
                # on either stream without suppressing the active exception.
                transfer_stream.wait_stream(caller)
                caller.wait_stream(transfer_stream)

    @staticmethod
    def _enqueue_storage_to_compute(
        plan: _RedistributionBucketPlan[_ItemT],
        slot: _PipelineSlot,
        context: _CommunicationContext,
        *,
        prepare: Callable[[_ItemT, Tensor], None],
    ) -> _BucketWork[_ItemT]:
        handle = context.device_handle
        transfer = context.transfer_stream
        with handle.stream(transfer):
            (
                storage_buffer,
                compute_fragment_buffer,
            ) = slot.buffers.communication_buffers(plan)
            work = _BucketWork(plan, slot, storage_buffer, compute_fragment_buffer)
            _prepare_redistributed(
                plan,
                slot.buffers,
                storage_buffer,
                prepare=prepare,
            )
            _execute_packed_all_to_all(
                plan.storage_to_compute_schedule,
                output=compute_fragment_buffer,
                input=storage_buffer,
            )
            slot.compute_input_ready.record(transfer)
        return work

    @staticmethod
    def _compute_bucket(
        work: _BucketWork[_ItemT],
        slot: _PipelineSlot,
        caller_stream: torch.Stream,
        context: _CommunicationContext,
        *,
        local_tensor_spec: Callable[
            [_ItemT], tuple[torch.Size, torch.dtype, torch.device]
        ],
        prepare: Callable[[_ItemT, Tensor], None],
        compute: Callable[[_ItemT, Tensor], None],
        finalize: Callable[[_ItemT, Tensor], None],
    ) -> None:
        handle = context.device_handle
        with handle.stream(caller_stream):
            _BucketedRedistributionRuntime._compute_without_redistribution(
                work.plan.unredistributed_items,
                slot.buffers,
                local_tensor_spec=local_tensor_spec,
                prepare=prepare,
                compute=compute,
                finalize=finalize,
            )
            caller_stream.wait_event(slot.compute_input_ready)
            _compute_redistributed(
                work,
                slot.buffers,
                compute=compute,
            )
            slot.compute_done.record(caller_stream)

    @staticmethod
    def _enqueue_compute_to_storage(
        work: _BucketWork[_ItemT],
        context: _CommunicationContext,
        *,
        finalize: Callable[[_ItemT, Tensor], None],
    ) -> None:
        handle = context.device_handle
        transfer = context.transfer_stream
        with handle.stream(transfer):
            transfer.wait_event(work.slot.compute_done)
            _execute_packed_all_to_all(
                work.plan.compute_to_storage_schedule,
                output=work.storage_buffer,
                input=work.compute_fragment_buffer,
            )
            _finalize_redistributed(work, work.slot.buffers, finalize=finalize)
            work.slot.done.record(transfer)

    @staticmethod
    def _release(work: _BucketWork[_ItemT], caller_stream: torch.Stream) -> None:
        caller_stream.wait_event(work.slot.done)

    @staticmethod
    def _compute_without_redistribution(
        items: Sequence[_ItemT],
        slot: _BufferSlot,
        *,
        local_tensor_spec: Callable[
            [_ItemT], tuple[torch.Size, torch.dtype, torch.device]
        ],
        prepare: Callable[[_ItemT, Tensor], None],
        compute: Callable[[_ItemT, Tensor], None],
        finalize: Callable[[_ItemT, Tensor], None],
    ) -> None:
        for item in items:
            shape, dtype, device = local_tensor_spec(item)
            prepared = slot.compute_buffer(shape, dtype=dtype, device=device)
            prepare(item, prepared)
            compute(item, prepared)
            finalize(item, prepared)


def _prepare_redistributed(
    plan: _RedistributionBucketPlan[_ItemT],
    slot: _BufferSlot,
    storage_buffer: Tensor,
    *,
    prepare: Callable[[_ItemT, Tensor], None],
) -> None:
    schedule = plan.storage_to_compute_schedule
    participant = plan.group.local_participant
    for index, (item, redistribution_plan) in enumerate(
        zip(
            plan.redistributed_items,
            plan.redistribution_plans,
            strict=True,
        )
    ):
        partition = redistribution_plan.storage_partition(participant)
        prepared = slot.storage_partition_buffer(
            partition.tensor_shape,
            dtype=plan.dtype,
            device=plan.device,
        )
        prepare(item, prepared)
        spans = schedule.input_spans_by_parameter[index]
        for span in spans:
            packed = storage_buffer[
                span.buffer_offset : span.buffer_offset + span.numel
            ]
            packed.copy_(_tensor_region_view(prepared, span.region).reshape(-1))


def _compute_redistributed(
    work: _BucketWork[_ItemT],
    slot: _BufferSlot,
    *,
    compute: Callable[[_ItemT, Tensor], None],
) -> None:
    plan = work.plan
    participant = plan.group.local_participant
    to_compute = plan.storage_to_compute_schedule
    to_storage = plan.compute_to_storage_schedule
    for index, (item, redistribution_plan) in enumerate(
        zip(
            plan.redistributed_items,
            plan.redistribution_plans,
            strict=True,
        )
    ):
        partition = redistribution_plan.compute_partition(participant)
        if not math.prod(partition.tensor_shape):
            continue
        received_spans = to_compute.output_spans_by_parameter[index]
        compute_tensor = slot.compute_buffer(
            partition.tensor_shape,
            dtype=plan.dtype,
            device=plan.device,
        )
        for span in received_spans:
            received = work.compute_fragment_buffer[
                span.buffer_offset : span.buffer_offset + span.numel
            ]
            _tensor_region_view(compute_tensor, span.region).copy_(
                received.view(span.region.shape)
            )

        compute(item, compute_tensor)

        for span in to_storage.input_spans_by_parameter[index]:
            packed = work.compute_fragment_buffer[
                span.buffer_offset : span.buffer_offset + span.numel
            ]
            packed.copy_(_tensor_region_view(compute_tensor, span.region).reshape(-1))


def _finalize_redistributed(
    work: _BucketWork[_ItemT],
    slot: _BufferSlot,
    *,
    finalize: Callable[[_ItemT, Tensor], None],
) -> None:
    plan = work.plan
    participant = plan.group.local_participant
    schedule = work.plan.compute_to_storage_schedule
    for index, (item, redistribution_plan) in enumerate(
        zip(
            plan.redistributed_items,
            plan.redistribution_plans,
            strict=True,
        )
    ):
        partition = redistribution_plan.storage_partition(participant)
        update = slot.storage_partition_buffer(
            partition.tensor_shape,
            dtype=plan.dtype,
            device=plan.device,
        )
        spans = schedule.output_spans_by_parameter[index]
        for span in spans:
            packed = work.storage_buffer[
                span.buffer_offset : span.buffer_offset + span.numel
            ]
            _tensor_region_view(update, span.region).copy_(
                packed.view(span.region.shape)
            )
        finalize(item, update)


def _tensor_region_view(tensor: Tensor, region: _TensorRegion) -> Tensor:
    view = tensor[
        tuple(
            slice(offset, offset + size)
            for offset, size in zip(region.offsets, region.shape, strict=True)
        )
    ]
    return view

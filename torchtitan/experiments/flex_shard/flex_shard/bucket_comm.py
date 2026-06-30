# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass, field
from types import ModuleType
from typing import TYPE_CHECKING

import torch
from torch.distributed.device_mesh import _get_device_handle

from .placement_contract import (
    PlacementPreparedReduceGrad,
    PlacementPreparedUnshard,
    PlacementUnshardResult,
)

if TYPE_CHECKING:
    from torch.distributed.device_mesh import DeviceMesh

    from .bucket_storage import ParamInfo


class UnshardHandle:
    """Handle for a FlexShard bucket unshard operation."""

    def finish(self) -> list[torch.Tensor]:
        """Wait for the unshard and return full parameters."""
        raise NotImplementedError

    def wait(self) -> None:
        """Wait until the unshard result is usable on the current stream."""
        raise NotImplementedError

    def release_buffers(self) -> None:
        """Release temporary buffers owned by the unshard operation."""
        raise NotImplementedError


class ReduceGradHandle:
    """Handle for a FlexShard bucket gradient reduction operation."""

    def finish(self) -> list[torch.Tensor]:
        """Wait for the reduction and return local gradient shards."""
        raise NotImplementedError

    def wait(self) -> None:
        """Wait until the reduce-grad result is usable on the current stream."""
        raise NotImplementedError

    def synchronize(self) -> None:
        """Block the host until the reduce-grad result is complete."""
        raise NotImplementedError

    def release_buffers(self, release_sharded_grads: bool) -> None:
        """Release temporary buffers owned by the reduce-grad operation."""
        raise NotImplementedError

    def record_sharded_grads(
        self,
        sharded_grads: list[torch.Tensor],
        stream: torch.Stream,
    ) -> None:
        """Track sharded grads until queued work on stream is complete."""
        raise NotImplementedError


@dataclass
class PreparedReduceGrad:
    """Packed reduce-grad inputs whose collective launch can be deferred."""

    prepared: PlacementPreparedReduceGrad
    copy_in_done: torch.Event | None
    device_handle: ModuleType | None


def begin_bucket_unshard(
    tensors: list[torch.Tensor],
    infos: list[ParamInfo],
    mesh: DeviceMesh,
    unshard_stream: torch.Stream,
    debug_fqn: str | None = None,
) -> UnshardHandle:
    """Begin a bucket unshard and return a handle for full params."""
    placement = infos[0].placement
    for info in infos[1:]:
        if info.placement != placement:
            raise ValueError(
                "FlexShard bucket unshard requires all parameters in a "
                f"bucket to use the same placement, but {infos[0].fqn!r} uses "
                f"{placement!r} and {info.fqn!r} uses {info.placement!r}."
            )

    if torch.compiler.is_compiling():
        prepared = placement.prepare_unshard_bucket(tensors, infos, mesh, debug_fqn)
        result = _run_and_finish_unshard(prepared)
        return SyncUnshardResult(result.full_params)

    device = tensors[0].device
    device_handle = _get_device_handle(device.type)
    copy_in_done = device_handle.Event()
    copy_in_done.record(device_handle.current_stream(device))
    with device_handle.stream(unshard_stream):
        unshard_stream.wait_event(copy_in_done)
        prepared = placement.prepare_unshard_bucket(tensors, infos, mesh, debug_fqn)
        prepared.placement.run_prepared_unshard(prepared)
        event = device_handle.Event()
        event.record(unshard_stream)
    return AsyncUnshardResult(
        prepared=prepared,
        event=event,
        device_handle=device_handle,
    )


def prepare_reduce_grad(
    tensors: list[torch.Tensor],
    infos: list[ParamInfo],
    mesh: DeviceMesh,
    debug_fqn: str | None = None,
) -> PreparedReduceGrad:
    """Prepare reduce-grad inputs without launching the collective."""
    placement = infos[0].placement
    for info in infos[1:]:
        if info.placement != placement:
            raise ValueError(
                "FlexShard bucket reduce-grad requires all parameters in a "
                f"bucket to use the same placement, but {infos[0].fqn!r} uses "
                f"{placement!r} and {info.fqn!r} uses {info.placement!r}."
            )
    placement_prepared = placement.prepare_reduce_grad(
        tensors,
        infos,
        mesh,
        debug_fqn,
    )

    if torch.compiler.is_compiling():
        return PreparedReduceGrad(
            prepared=placement_prepared,
            copy_in_done=None,
            device_handle=None,
        )

    device = placement_prepared.buffers[0].device
    device_handle = _get_device_handle(device.type)
    copy_in_done = device_handle.Event()
    copy_in_done.record(device_handle.current_stream(device))
    return PreparedReduceGrad(
        prepared=placement_prepared,
        copy_in_done=copy_in_done,
        device_handle=device_handle,
    )


def launch_reduce_grad(
    prepared: PreparedReduceGrad,
    reduce_grad_stream: torch.Stream,
) -> ReduceGradHandle:
    """Launch a previously packed reduce-grad request."""
    if torch.compiler.is_compiling():
        result = prepared.prepared.placement.reduce_prepared_grad(prepared.prepared)
        return SyncReduceGradResult(result.sharded_grads)

    if prepared.device_handle is None or prepared.copy_in_done is None:
        raise AssertionError("Expected eager reduce-grad launch metadata.")
    with prepared.device_handle.stream(reduce_grad_stream):
        reduce_grad_stream.wait_event(prepared.copy_in_done)
        result = prepared.prepared.placement.reduce_prepared_grad(prepared.prepared)
        event = prepared.device_handle.Event()
        event.record(reduce_grad_stream)
    return AsyncReduceGradResult(
        sharded_grads=result.sharded_grads,
        event=event,
        buffers=[*prepared.prepared.buffers, *result.buffers],
        device_handle=prepared.device_handle,
    )


def begin_reduce_grad(
    tensors: list[torch.Tensor],
    infos: list[ParamInfo],
    mesh: DeviceMesh,
    reduce_grad_stream: torch.Stream,
    debug_fqn: str | None = None,
) -> ReduceGradHandle:
    """Begin a bucket reduce-grad and return a handle for local grad shards."""
    prepared = prepare_reduce_grad(tensors, infos, mesh, debug_fqn)
    return launch_reduce_grad(prepared, reduce_grad_stream)


@dataclass
class SyncUnshardResult(UnshardHandle):
    """Already-finished unshard result used during graph capture."""

    full_params: list[torch.Tensor]

    def finish(self) -> list[torch.Tensor]:
        return self.full_params

    def wait(self) -> None:
        return

    def release_buffers(self) -> None:
        return


@dataclass
class AsyncUnshardResult(UnshardHandle):
    """State needed to finish an async unshard launched on a side stream."""

    prepared: PlacementPreparedUnshard
    event: torch.Event | None
    device_handle: ModuleType
    _result: PlacementUnshardResult | None = field(default=None, init=False)
    _device: torch.device | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.prepared.buffers:
            self._device = self.prepared.buffers[0].device

    def finish(self) -> list[torch.Tensor]:
        self.wait()
        if self._result is None:
            self._result = self.prepared.placement.finish_prepared_unshard(
                self.prepared
            )
            if self._result.full_params:
                self._device = self._result.full_params[0].device
        results = self._result.full_params
        self.release_buffers()
        return results

    def wait(self) -> None:
        if self._device is None:
            return
        if self.event is not None:
            self.device_handle.current_stream(self._device).wait_event(self.event)

    def release_buffers(self) -> None:
        """Release raw unshard buffers after current-stream work is queued."""
        tensors = list(self.prepared.buffers)
        self.prepared.buffers.clear()
        if self._result is not None:
            tensors.extend(self._result.buffers)
            self._result.buffers.clear()
        if not tensors:
            return

        device = tensors[0].device
        stream = self.device_handle.current_stream(device)
        event = self.device_handle.Event()
        event.record(stream)
        handoff = StreamHandoff(tensors, event, stream, self.device_handle)
        handoff.release()


@dataclass
class SyncReduceGradResult(ReduceGradHandle):
    """Already-finished reduce-grad result used during graph capture."""

    sharded_grads: list[torch.Tensor]

    def finish(self) -> list[torch.Tensor]:
        return self.sharded_grads

    def wait(self) -> None:
        return

    def synchronize(self) -> None:
        return

    def release_buffers(self, release_sharded_grads: bool) -> None:
        if release_sharded_grads:
            self.sharded_grads.clear()

    def record_sharded_grads(
        self,
        sharded_grads: list[torch.Tensor],
        stream: torch.Stream,
    ) -> None:
        self.sharded_grads = sharded_grads


@dataclass
class AsyncReduceGradResult(ReduceGradHandle):
    """State needed to finish an async reduce-grad launched on a side stream."""

    sharded_grads: list[torch.Tensor]
    event: torch.Event | None
    buffers: list[torch.Tensor]
    device_handle: ModuleType
    _device: torch.device | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.sharded_grads:
            self._device = self.sharded_grads[0].device
        elif self.buffers:
            self._device = self.buffers[0].device

    def finish(self) -> list[torch.Tensor]:
        self.wait()
        return self.sharded_grads

    def wait(self) -> None:
        if self._device is None:
            return
        if self.event is not None:
            self.device_handle.current_stream(self._device).wait_event(self.event)

    def synchronize(self) -> None:
        if self.event is not None:
            self.event.synchronize()

    def release_buffers(self, release_sharded_grads: bool) -> None:
        """Release pending reduce-grad buffers after its completion wait."""
        tensors = list(self.buffers)
        self.buffers.clear()
        if release_sharded_grads:
            tensors.extend(self.sharded_grads)
            self.sharded_grads.clear()
        if not tensors:
            return
        device = tensors[0].device

        stream = self.device_handle.current_stream(device)
        event = self.device_handle.Event()
        event.record(stream)
        handoff = StreamHandoff(tensors, event, stream, self.device_handle)
        handoff.release()

    def record_sharded_grads(
        self,
        sharded_grads: list[torch.Tensor],
        stream: torch.Stream,
    ) -> None:
        self.sharded_grads = sharded_grads
        if self.sharded_grads:
            self._device = self.sharded_grads[0].device
        self.event = self.device_handle.Event()
        self.event.record(stream)


def _run_and_finish_unshard(prepared: PlacementPreparedUnshard):
    prepared.placement.run_prepared_unshard(prepared)
    return prepared.placement.finish_prepared_unshard(prepared)


class StreamHandoff:
    """Hold tensors until it is safe to release them on a target stream."""

    __slots__ = (
        "_tensors",
        "_event",
        "_release_stream",
        "_device_handle",
        "_released",
    )

    def __init__(
        self,
        tensors: list[torch.Tensor],
        ready_event: torch.Event | None,
        release_stream: torch.Stream,
        device_handle: ModuleType | None = None,
    ) -> None:
        if device_handle is None:
            device_handle = _get_device_handle(tensors[0].device.type)
        self._tensors = tensors
        self._event = ready_event
        self._release_stream = release_stream
        self._device_handle = device_handle
        self._released = False

    def wait(self, stream: torch.Stream) -> None:
        if self._released or self._event is None:
            return
        stream.wait_event(self._event)

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        if not self._tensors:
            return
        if self._event is not None:
            self._release_stream.wait_event(self._event)
        with self._device_handle.stream(self._release_stream):
            self._tensors.clear()

    def __del__(self) -> None:
        try:
            self.release()
        except Exception:
            pass

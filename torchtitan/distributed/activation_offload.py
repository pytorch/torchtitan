# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Eager CPU offloading of the activations retained for backward.

The offloader groups saves by transformer block call: each call of a block
installs torch_remat saved-tensor hooks for its forward, from forward hooks
on the block, so it works with or without remat. Inside a remat
checkpoint those hooks see exactly what the checkpoint keeps for backward,
with its kind and save region; without remat they see every autograd save.
This runtime offloads a selected subset to pinned CPU memory during forward
and reloads it during backward:

- Forward: ``pack`` queues a D2H copy on a dedicated offload stream. The GPU
  reference is dropped at the end of the next block's forward, after the
  compute stream waits for the copy, so the copy overlaps later compute.
- Backward: a gradient hook on block ``i + 1``'s output fires before that
  block's backward and queues the H2D reload of group ``i`` on a dedicated
  reload stream. ``unpack`` makes the compute stream wait for the reload, or
  reloads on demand when nothing prefetched it.

Every GPU block is allocated on the compute stream and freed in compute-stream
order; cross-stream dependencies are explicit event waits (no
``record_stream``), as in FSDP2. Pinned host buffers come from a pool the
offloader owns (see _PinnedBufferPool), so the same code is CUDA-graph
capturable without holding a host buffer per captured transfer.

State lives in per-call groups, never on the module: each call opens a
group, and the autograd graph keeps each group alive until backward reaches
it, so several microbatches can be in flight (gradient accumulation, pipeline
parallelism).

A group is one block, remat's checkpoint unit, and never finer: a remat
block's backward first replays the block, which needs all of the block's
checkpoint inputs and saved-region outputs at once, so finer groups would
only free memory earlier in the forward. Finer groups make sense only
together with remat checkpoints below the block. See
torchtitan/distributed/ACTIVATION_OFFLOAD.md for the design and
torchtitan/distributed/REMAT.md for the saved-tensor hook contract.
"""

import collections
import contextlib
import enum
import functools
import logging
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Any

import torch
import torch.nn as nn
import torch_remat as remat
from torch.distributed.fsdp import FSDPModule, fully_shard
from torch.distributed.tensor import DTensor
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._pytree import tree_flatten


logger = logging.getLogger(__name__)


@dataclass(kw_only=True, slots=True)
class ActivationOffloadConfig:
    """Offload selected activations retained for backward to CPU.

    Each transformer block (a child of the model part's ``layers``) is one
    offload group: the unit that is copied out, freed one block later, and
    reloaded one block ahead in backward. This is also remat's checkpoint
    unit, and it works with or without remat. Applies to the tensors autograd
    saves during a block's forward; under remat these are only what remat
    retains, since recomputed tensors are never stashed.

    Groups are not finer than a block: remat replays the whole block at the
    start of its backward, so sub-block groups would all be reloaded together
    and lower only the forward peak.
    """

    offload_block_input: bool = True
    """
    Offload each remat checkpoint's inputs that require grad, i.e. each
    transformer block's input hidden state. Under full remat this is the whole
    stash. Inputs shared across blocks (positions, masks) do not require grad
    and stay on GPU. An input sharing storage with the model part's own input
    (e.g. a pipeline stage's input, which the stage retains) stays on GPU,
    since offloading it would free nothing.
    """

    offload_regions: list[str] = field(default_factory=list)
    """
    Glob patterns of save regions whose retained tensors are offloaded: the
    tensors a region's ops save for backward and its output when later
    recomputation reads it. Names are relative to a transformer block, as in
    ``ActivationRematConfig.save_regions``; ``["*"]`` offloads everything
    the save regions retain. A storage is offloaded once for all its saves:
    a save of an already-offloaded storage is offloaded whatever its region,
    and a save of a storage another save kept on GPU stays on GPU, since
    offloading it would free nothing.
    """

    offload_saves_outside_regions: bool = False
    """
    Offload saves that belong to no save region. Inside a remat checkpoint
    every save belongs to one; without remat (offload instead of recompute)
    every save is one of these.
    """

    min_bytes: int = 1 << 20
    """Tensors smaller than this stay on GPU."""

    skip_last_layers: int = 1
    """
    Number of trailing transformer blocks of each model part to leave on GPU.
    The last block's backward starts right after its forward, so offloading it costs
    two copies and frees memory for almost no time.
    """

    reload_on_fsdp_copy_in_stream: bool = False
    """
    Issue reloads on FSDP2's all-gather copy-in stream instead of a private
    reload stream. Under FSDP CPU parameter offload that stream carries every
    H2D copy of a parameter shard, so parameter and activation H2D copies run
    strictly in enqueue order and never share the link concurrently. The
    offload hooks are registered after FSDP's, so on block ``i``'s output
    gradient FSDP queues its prefetch of block ``i - 1``'s parameters before
    the reload of group ``i - 1``. Requires the model part to be the FSDP
    root. Prototype: reads FSDP2's private comm context.
    """

    def __post_init__(self) -> None:
        if not (
            self.offload_block_input
            or self.offload_regions
            or self.offload_saves_outside_regions
        ):
            raise ValueError(
                "ActivationOffloadConfig offloads nothing: enable "
                "offload_block_input or offload_saves_outside_regions, or set "
                "offload_regions."
            )
        for name in ("min_bytes", "skip_last_layers"):
            if getattr(self, name) < 0:
                raise ValueError(f"ActivationOffloadConfig.{name} must be >= 0.")

    def apply(self, model_part: nn.Module) -> "ActivationOffloader":
        """Install offloading on the transformer blocks of ``model_part``.

        Call once per model part (each pipeline stage is one), after remat and
        parallelization have been applied.
        """
        return ActivationOffloader(self, model_part)


class _State(enum.Enum):
    # Original GPU tensor held; D2H copy queued.
    OFFLOADING = enum.auto()
    # GPU copy freed; only the pinned CPU copy remains.
    OFFLOADED = enum.auto()
    # H2D copy queued into a fresh GPU tensor, held until autograd frees the
    # payload. The CPU never learns when the copy completes; unpack makes the
    # compute stream wait on the reload event instead.
    RELOADING = enum.auto()


@dataclass(eq=False)
class _OffloadedStorage:
    """One offloaded storage, shared by every save that views it.

    Several saves of the same storage, whole or partial views with any shape,
    stride or dtype, share one transfer of the whole storage. Autograd frees it, and with it the CPU copy
    and any reloaded GPU copy, when it frees the last saved variable viewing
    it.
    """

    # A flat uint8 tensor over the whole storage: the original storage while
    # OFFLOADING, None once OFFLOADED, and the reloaded copy once RELOADING.
    device_tensor: torch.Tensor | None
    # The pinned host buffer (flat uint8) holding the storage's bytes. Returned
    # to the pool, and set to None, once the reload is queued.
    host_buffer: torch.Tensor | None
    # Why remat retained the tensor that first packed this storage.
    kind: remat.SavedTensorKind
    state: _State = _State.OFFLOADING
    reload_event: torch.Event | None = None


@dataclass
class OffloadStats:
    """Byte counts for one forward of one model part."""

    offloaded_bytes: int = 0
    # Offloaded but still referenced elsewhere when its GPU copy was dropped,
    # so the transfer freed no GPU memory; by block FQN and kind.
    still_resident_bytes: dict[str, int] = field(default_factory=dict)
    # GPU storage pinned by saved tensors kept on GPU, by reason. Each storage
    # counts once, at its real size: an expanded or sliced view pins its
    # storage, not its logical size.
    skipped_bytes: dict[str, int] = field(default_factory=dict)
    # Offloaded bytes by save region, or "block input".
    offloaded_bytes_by_region: dict[str, int] = field(default_factory=dict)
    _skipped_storages: set[int] = field(default_factory=set)


@dataclass(frozen=True, eq=False)
class _OffloadedView:
    """The payload autograd holds in place of one offloaded tensor: where the
    tensor sits in its offloaded storage."""

    storage: _OffloadedStorage
    dtype: torch.dtype
    size: torch.Size
    stride: tuple[int, ...]
    # In elements of ``dtype``, as Tensor.storage_offset() reports it.
    storage_offset: int | torch.SymInt


@dataclass(eq=False)
class _OffloadGroup:
    """The tensors offloaded by one call of one matched module."""

    module_fqn: str
    module: nn.Module
    # Storages of the module's parameters during this call, which are never
    # offloaded (see _pack). Collected at the first save that needs them,
    # inside the forward and so after FSDP2's pre-forward unshard.
    parameter_storages: set[int] | None = None
    # Keyed by storage pointer so repeated saves share one transfer.
    storages: dict[int, _OffloadedStorage] = field(default_factory=dict)
    # Storages a save of this call kept on GPU because its region was not
    # selected. Offloading another save of them would free nothing.
    kept_storages: set[int] = field(default_factory=set)
    # Recorded on the offload stream when the block's forward returns, after
    # every D2H copy of this group was queued, and only those before it.
    copy_done_event: torch.Event | None = None


class _PinnedBufferPool:
    """Pinned host buffers owned by one offloader, reused by exact size.

    The caching host allocator never reuses a block allocated during CUDA
    graph capture (it cannot know when replay is done with it), so a captured
    step that offloads N microbatches would hold N microbatches of pinned
    memory, for as long as the process lives. This pool reuses buffers in both
    eager mode and capture, ordered on the GPU: a buffer comes back with the
    event of the H2D copy that last read it, and the next D2H copy into it
    waits on that event.
    """

    def __init__(self) -> None:
        # nbytes -> [(buffer, event of its last read, recorded during capture)]
        self._free: dict[int, list[tuple[torch.Tensor, torch.Event, bool]]] = {}

    def acquire(self, nbytes: int, copy_stream: torch.cuda.Stream) -> torch.Tensor:
        """Return a buffer that ``copy_stream`` may write once it is current."""
        free = self._free.get(nbytes)
        if not free:
            return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
        buffer, last_read, recorded_during_capture = free.pop()
        # An event recorded in the other mode (eager vs capture) cannot be
        # waited on, and its read is already ordered before this copy:
        # torch.cuda.graph synchronizes the device before capture, and a
        # replay runs on the compute stream, which the copy stream waits on
        # before each copy.
        if recorded_during_capture == torch.cuda.is_current_stream_capturing():
            copy_stream.wait_event(last_read)
        return buffer

    def release(self, buffer: torch.Tensor, last_read: torch.Event) -> None:
        self._free.setdefault(buffer.nbytes, []).append(
            (buffer, last_read, torch.cuda.is_current_stream_capturing())
        )


class ActivationOffloader:
    """Offload runtime for the matched modules of one model part.

    Created by ``ActivationOffloadConfig.apply``. Registers forward hooks on
    the model part and on each offloaded module; each offloaded module's
    forward runs under this offloader's saved-tensor hooks.
    """

    def __init__(self, config: ActivationOffloadConfig, model_part: nn.Module):
        self.config = config
        # One stream per copy direction, so D2H and H2D copies overlap. With
        # reload_on_fsdp_copy_in_stream, the reload stream is replaced by
        # FSDP's copy-in stream once FSDP's lazy init has created it.
        self._offload_stream = torch.cuda.Stream()
        self._reload_stream = torch.cuda.Stream()
        if config.reload_on_fsdp_copy_in_stream and not isinstance(
            model_part, FSDPModule
        ):
            raise ValueError(
                "ActivationOffloadConfig.reload_on_fsdp_copy_in_stream requires "
                "a model part sharded with fully_shard."
            )
        self._model_part = model_part
        self._host_buffers = _PinnedBufferPool()
        # The group whose module forward is running, and its saved-tensor
        # hooks context, entered in the module's pre-hook and exited in its
        # post-hook.
        self._current_group: _OffloadGroup | None = None
        self._current_hooks: contextlib.AbstractContextManager[None] | None = None
        # The latest group of the current model-part forward, whose GPU copies
        # are not freed yet. They are freed, and its reload trigger installed,
        # at the end of the next group's forward (or of the model part's
        # forward).
        self._group_awaiting_free: _OffloadGroup | None = None
        # Storages of the current model-part forward's tensor inputs. Their
        # caller (e.g. the pipeline stage) retains them, so offloading a block
        # input that shares one would free nothing.
        self._model_part_input_storages: set[int] = set()
        # Stats of the latest (or in-progress) forward of this model part.
        self.forward_stats = OffloadStats()
        self._logged_summary = False
        # Reloads that no backward trigger prefetched; each one is exposed.
        self.num_on_demand_reloads = 0
        # Set by defer_to_schedule: a pipeline schedule frees and reloads each
        # microbatch's last groups (wait_offload, reload_next_backward).
        self._schedule_driven = False
        # Schedule-driven only. The groups of each microbatch forward of this
        # model part (a pipeline stage), in block order, oldest microbatch
        # first. A stage runs its backwards in microbatch order, so the front
        # belongs to the next backward.
        self._microbatch_groups: collections.deque[
            list[_OffloadGroup]
        ] = collections.deque()
        self._current_microbatch_groups: list[_OffloadGroup] = []
        # Schedule-driven only. Last groups of finished forwards whose GPU
        # copies wait_offload frees.
        self._groups_awaiting_wait_offload: list[_OffloadGroup] = []
        # The same blocks ActivationRematConfig.apply checkpoints.
        blocks = [
            (f"layers.{layer_id}", block)
            for layer_id, block in model_part.get_submodule("layers").named_children()
        ]
        num_offloaded = max(len(blocks) - config.skip_last_layers, 0)
        logger.info(
            "Activation offloading enabled for %d of %d transformer blocks: %s",
            num_offloaded,
            len(blocks),
            config,
        )
        for fqn, module in blocks[:num_offloaded]:
            module.register_forward_pre_hook(
                functools.partial(self._group_pre_forward, fqn)
            )
            # always_call: exit the saved-tensor hooks even if forward raises.
            module.register_forward_hook(self._group_post_forward, always_call=True)
        model_part.register_forward_pre_hook(
            self._model_part_pre_forward, with_kwargs=True
        )
        model_part.register_forward_hook(self._model_part_post_forward)

    # ------------------------------------------------------------ forward
    def _group_pre_forward(self, fqn: str, module: nn.Module, args: Any) -> None:
        # Under remat, the checkpoint's replay in backward calls the block's
        # forward again; that replay saves nothing to offload.
        if not torch.is_grad_enabled() or remat.is_recomputing():
            return
        assert (
            self._current_group is None
        ), f"transformer block {fqn} started inside {self._current_group.module_fqn}"
        group = _OffloadGroup(module_fqn=fqn, module=module)
        self._current_group = group
        # capture_context snapshots the group where a tensor is produced; see
        # _pack for why a save remat defers must still land in it.
        self._current_hooks = remat.saved_tensors_hooks(
            self._pack, self._unpack, capture_context=lambda: group
        )
        self._current_hooks.__enter__()

    def _group_post_forward(self, module: nn.Module, args: Any, output: Any) -> None:
        group = self._current_group
        if group is None or group.module is not module:
            return
        assert self._current_hooks is not None
        self._current_hooks.__exit__(None, None, None)
        self._current_group = None
        self._current_hooks = None
        if group.storages:
            # Queued after this group's D2H copies, so the event completes
            # once they all finish. The free point makes the compute stream
            # wait on it before dropping the group's GPU copies.
            group.copy_done_event = self._offload_stream.record_event()
            if self._schedule_driven:
                self._current_microbatch_groups.append(group)
        previous, self._group_awaiting_free = self._group_awaiting_free, group
        if previous is not None:
            # Group i - 1's copies had group i's forward to finish.
            self._free_gpu_copies(previous)
            # Group i's output triggers the reload of group i - 1.
            self._prefetch_on_grad(output, group_to_reload=previous)

    def _model_part_pre_forward(
        self, module: nn.Module, args: Any, kwargs: Any
    ) -> None:
        if self.config.reload_on_fsdp_copy_in_stream:
            # FSDP's root pre-forward hook is prepended, so its lazy init has
            # created the comm context's streams by now.
            # pyrefly: ignore [missing-attribute]
            comm_ctx = fully_shard.state(self._model_part)._comm_ctx
            self._reload_stream = comm_ctx.all_gather_copy_in_stream
        self._group_awaiting_free = None
        self._current_microbatch_groups = []
        self._model_part_input_storages = {
            t.untyped_storage().data_ptr()
            for t in tree_flatten((args, kwargs))[0]
            if isinstance(t, torch.Tensor)
        }
        self.forward_stats = OffloadStats()

    def _model_part_post_forward(
        self, module: nn.Module, args: Any, output: Any
    ) -> None:
        # The last group runs first in backward, so the model part's output
        # triggers the reload of the last group. On a non-last pipeline stage
        # this output's gradient arrives from the next stage and the backward
        # starts at once, so the trigger cannot overlap the reload and freeing
        # here would wait on the copies; hence skip_last_layers. A schedule
        # that calls defer_to_schedule frees the last group after its next
        # action instead and reloads it ahead of the backward; the trigger
        # stays as a fallback.
        last_group = self._group_awaiting_free
        if last_group is not None:
            self._group_awaiting_free = None
            if self._schedule_driven:
                self._groups_awaiting_wait_offload.append(last_group)
            else:
                self._free_gpu_copies(last_group)
            self._prefetch_on_grad(output, group_to_reload=last_group)
        # One entry per forward, even if it offloaded nothing, so the queue
        # stays aligned with the schedule's microbatches.
        if self._schedule_driven and torch.is_grad_enabled():
            self._microbatch_groups.append(self._current_microbatch_groups)
            self._current_microbatch_groups = []
        if not self._logged_summary and torch.is_grad_enabled():
            self._logged_summary = True
            stats = self.forward_stats
            logger.info(
                "Activation offloading, first forward of this model part: "
                "offloaded %.1f MiB %s, kept on GPU %s",
                stats.offloaded_bytes / 2**20,
                {
                    k: f"{v / 2**20:.1f} MiB"
                    for k, v in stats.offloaded_bytes_by_region.items()
                },
                {k: f"{v / 2**20:.1f} MiB" for k, v in stats.skipped_bytes.items()},
            )
            if stats.still_resident_bytes:
                logger.warning(
                    "Activation offloading copied %.1f MiB to CPU that stayed "
                    "resident on GPU because another reference held its storage "
                    "(e.g. a saved partial view kept on GPU). These copies cost "
                    "bandwidth without saving memory: %s",
                    sum(stats.still_resident_bytes.values()) / 2**20,
                    {
                        k: f"{v / 2**20:.1f} MiB"
                        for k, v in stats.still_resident_bytes.items()
                    },
                )

    # ---------------------------------------------- pipeline schedule hooks
    def defer_to_schedule(self) -> None:
        """Hand the timing of each microbatch's last groups to a pipeline
        schedule, which must then call ``wait_offload`` after the action that
        follows each forward of this stage, and ``reload_next_backward``
        before each backward.

        Like FSDP's UNSHARD/RESHARD actions, this lets the schedule overlap
        the transfers at the stage boundary with other microbatches' work, so
        the last blocks need not be skipped. Within the stage, gradient hooks
        still prefetch one block ahead.
        """
        self._schedule_driven = True

    def wait_offload(self) -> None:
        """Free the GPU copies of the last groups of finished forwards.

        Call it one action after the forward, so the copies had that action's
        compute to finish; the compute stream still waits on them.
        """
        for group in self._groups_awaiting_wait_offload:
            self._free_gpu_copies(group)
        self._groups_awaiting_wait_offload.clear()

    def reload_next_backward(self, num_groups: int = 1) -> None:
        """Queue the reload of the last ``num_groups`` groups of the oldest
        microbatch whose backward has not run yet.

        Call it one transfer latency before that backward. ``num_groups=1``
        matches the in-stage prefetch depth; reloading every group brings
        back the stage's whole stash for that microbatch at once, which only
        pays off for stages of one or two blocks.
        """
        if num_groups < 1:
            raise ValueError(f"num_groups must be >= 1, got {num_groups}")
        # Drop forwards whose backward already ran outside the schedule's
        # actions (e.g. a pipeline stage's metadata inference): their
        # gradient hooks reloaded, and so emptied, every group.
        while self._microbatch_groups and _backward_ran(self._microbatch_groups[0]):
            self._microbatch_groups.popleft()
        assert (
            self._schedule_driven and self._microbatch_groups
        ), "reload_next_backward needs defer_to_schedule and a pending forward"
        groups = self._microbatch_groups.popleft()
        for group in reversed(groups[-num_groups:]):
            self._reload_group(group)

    def _pack(self, tensor: torch.Tensor) -> Any:
        info = remat.current_saved_tensor_info()
        group = self._current_group
        if group is None:
            return tensor
        # remat defers a save region's output save to its first later consumer
        # and passes the group captured where it was produced. A consumer in a
        # later block would pack into a group whose copy-done event is already
        # recorded, so its free would race with this copy.
        assert info.context is None or info.context is group, (
            f"a save produced in {info.context.module_fqn} was deferred to "
            f"{group.module_fqn}; offload groups do not support saves deferred "
            "across transformer blocks"
        )
        kind = info.kind
        storage_ptr = tensor.untyped_storage().data_ptr()
        # An already-offloaded storage is offloaded for every save of it.
        selected = storage_ptr in group.storages
        if kind is remat.SavedTensorKind.CHECKPOINT_INPUT:
            if not tensor.requires_grad:
                return tensor
            selected = selected or self.config.offload_block_input
            region = "block input"
        elif info.region is None:
            region = "outside any region"
            selected = selected or self.config.offload_saves_outside_regions
        else:
            region = info.region
            selected = selected or any(
                fnmatch(region, pattern) for pattern in self.config.offload_regions
            )
        if not selected:
            group.kept_storages.add(storage_ptr)
            return self._skip(tensor, f"{region} not selected")
        if storage_ptr in group.kept_storages:
            return self._skip(tensor, "storage kept on GPU by another save")
        if storage_ptr in self._model_part_input_storages:
            return self._skip(tensor, "model-part input")
        if type(tensor) is not torch.Tensor or tensor.device.type != "cuda":
            return self._skip(tensor, "not a plain CUDA tensor")
        storage = tensor.untyped_storage()
        # Never offload a parameter. A save region that uses a weight
        # internally (rather than taking it as a region input, which remat
        # replays instead of packing) keeps it for backward. A save of the
        # nn.Parameter itself is declined above as not a plain tensor; a
        # plain-tensor alias of its storage (a view, or a TP weight's local
        # tensor) is caught here. Offloading it frees nothing (the module
        # holds it), and under FSDP2 the async D2H copy would race with
        # reshard freeing the unsharded storage after forward.
        if group.parameter_storages is None:
            group.parameter_storages = _parameter_storages(group.module)
        if storage.data_ptr() in group.parameter_storages:
            return tensor
        if storage.nbytes() < self.config.min_bytes:
            return self._skip(tensor, "below min_bytes")
        # Offload the whole storage, even for a partial view: other saves
        # often view the rest of it (e.g. q, k and v slices of a fused QKV
        # output). Once all of them are offloaded, nothing else holds the
        # storage, so dropping the GPU copy frees it.
        offloaded = group.storages.get(storage.data_ptr())
        if offloaded is None:
            offloaded = self._offload(storage, kind)
            group.storages[storage.data_ptr()] = offloaded
            by_region = self.forward_stats.offloaded_bytes_by_region
            by_region[region] = by_region.get(region, 0) + storage.nbytes()
        return _OffloadedView(
            offloaded,
            tensor.dtype,
            tensor.size(),
            tensor.stride(),
            tensor.storage_offset(),
        )

    def _skip(self, tensor: torch.Tensor, reason: str) -> torch.Tensor:
        stats = self.forward_stats
        if type(tensor) is torch.Tensor:
            storage = tensor.untyped_storage()
            if storage.data_ptr() in stats._skipped_storages:
                return tensor
            stats._skipped_storages.add(storage.data_ptr())
            nbytes = storage.nbytes()
        else:
            nbytes = tensor.nbytes
        stats.skipped_bytes[reason] = stats.skipped_bytes.get(reason, 0) + nbytes
        return tensor

    def _offload(
        self, storage: torch.UntypedStorage, kind: remat.SavedTensorKind
    ) -> _OffloadedStorage:
        device_tensor = torch.empty(0, dtype=torch.uint8, device=storage.device)
        device_tensor.set_(storage)
        host_buffer = self._host_buffers.acquire(storage.nbytes(), self._offload_stream)
        # Start the copy only after the compute stream has produced the tensor.
        self._offload_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._offload_stream):
            host_buffer.copy_(device_tensor, non_blocking=True)
        self.forward_stats.offloaded_bytes += storage.nbytes()
        return _OffloadedStorage(
            device_tensor=device_tensor, host_buffer=host_buffer, kind=kind
        )

    def _free_gpu_copies(self, group: _OffloadGroup) -> None:
        if not group.storages:
            return
        assert group.copy_done_event is not None
        # The compute stream owns the GPU memory: without this wait the
        # allocator could hand it to later compute while the copy reads it.
        # Waiting on this group's event, not the whole offload stream, keeps
        # later groups' copies from delaying compute.
        torch.cuda.current_stream().wait_event(group.copy_done_event)
        for offloaded in group.storages.values():
            if offloaded.state is _State.OFFLOADING:
                assert offloaded.device_tensor is not None
                storage = StorageWeakRef(offloaded.device_tensor.untyped_storage())
                nbytes = offloaded.device_tensor.untyped_storage().nbytes()
                offloaded.device_tensor = None
                offloaded.state = _State.OFFLOADED
                # Dropping our reference frees the GPU memory only if nothing
                # else holds the storage.
                if not storage.expired():
                    resident = self.forward_stats.still_resident_bytes
                    source = f"{group.module_fqn} {offloaded.kind.name}"
                    resident[source] = resident.get(source, 0) + nbytes

    # ----------------------------------------------------------- backward
    def _prefetch_on_grad(self, output: Any, *, group_to_reload: _OffloadGroup) -> None:
        """Install a backward trigger: reload ``group_to_reload`` once
        ``output``'s gradient is ready.

        The hook is registered during forward and fires during backward. For
        group ``i``, ``output`` is its module's output ``out_i`` and
        ``group_to_reload`` is group ``i - 1``::

            forward (hook registered)          backward (hook fires)
            -------------------------          ---------------------
            module i-1 fwd: packs group i-1
            module i fwd -> out_i
              _prefetch_on_grad(out_i,         grad of out_i ready
                group_to_reload=group i-1) -->   hook: queue H2D of group i-1
                                                 on the reload stream
                                               module i bwd (recompute, grads)
                                                 overlaps the H2D copy
                                               module i-1 bwd: unpack waits on
                                                 group i-1's reload event

        The model part's output gets the trigger for the last group, since
        the last group runs first in backward. This is the same pre-backward
        point FSDP2 uses to prefetch parameters. Several outputs share one
        trigger, which fires once.
        """

        def hook(grad: torch.Tensor) -> None:
            self._reload_group(group_to_reload)

        # Like FSDP2, hook every output: the first gradient to arrive starts
        # the reload. Later firings do nothing (see _reload_group).
        for tensor in tree_flatten(output)[0]:
            if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                tensor.register_hook(hook)

    def _reload_group(self, group: _OffloadGroup) -> None:
        """Queue the H2D reload of every still-offloaded storage of ``group``.

        The one entry point for reloads queued ahead of unpack. It is
        idempotent, so callers can race: whichever runs first reloads, and
        later calls find the group empty. A PP schedule that prefetches a
        stage's first backward group (see _model_part_post_forward) would
        call it too.
        """
        # Reloads are queued in pack order. The block's backward first replays
        # its forward, which reads the block input and then the SAVE_OUTPUTs in
        # that same order, so pack order suits it. TODO: optimization
        # opportunity for offload_regions: queue BACKWARD tensors last
        # and in reverse, since only the gradient pass after the replay reads
        # them. This matters only when the reloads barely finish in time.
        for offloaded in group.storages.values():
            if offloaded.state is _State.OFFLOADED:
                self._reload(offloaded)
        # Autograd keeps the caller's hook alive as long as the graph's nodes
        # are referenced (e.g. by a loss kept for logging). Dropping the
        # group's references leaves autograd's payloads as the only owners, so
        # each reloaded copy is freed with its saved tensor.
        group.storages.clear()

    def _reload(self, offloaded: _OffloadedStorage) -> None:
        # Allocate on the compute stream so the memory is freed in
        # compute-stream order, then copy into it on the reload stream.
        host_buffer = offloaded.host_buffer
        assert host_buffer is not None
        device_tensor = torch.empty_like(host_buffer, device="cuda")
        self._reload_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self._reload_stream):
            device_tensor.copy_(host_buffer, non_blocking=True)
            offloaded.reload_event = self._reload_stream.record_event()
        # The next D2H copy into this buffer waits on the reload event.
        self._host_buffers.release(host_buffer, offloaded.reload_event)
        offloaded.host_buffer = None
        offloaded.device_tensor = device_tensor
        offloaded.state = _State.RELOADING

    def _unpack(self, packed: Any) -> torch.Tensor:
        if not isinstance(packed, _OffloadedView):
            return packed
        offloaded = packed.storage
        if offloaded.state is _State.OFFLOADED:
            self.num_on_demand_reloads += 1
            self._reload(offloaded)
        if offloaded.state is _State.RELOADING:
            assert offloaded.reload_event is not None
            torch.cuda.current_stream().wait_event(offloaded.reload_event)
        assert offloaded.device_tensor is not None
        # The reloaded copy holds the whole storage, so the saved tensor is the
        # same view of it as of the original.
        return offloaded.device_tensor.view(packed.dtype).as_strided(
            packed.size, packed.stride, packed.storage_offset
        )


def _backward_ran(groups: list[_OffloadGroup]) -> bool:
    return bool(groups) and all(not group.storages for group in groups)


def _parameter_storages(module: nn.Module) -> set[int]:
    storages = set()
    for parameter in module.parameters():
        local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
        storages.add(local.untyped_storage().data_ptr())
    return storages

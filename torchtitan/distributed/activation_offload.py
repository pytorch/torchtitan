# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Eager CPU offloading of the activations remat retains for backward.

The remat policy picks what to offload (``ActivationRematConfig.offload_block_input``
and ``offload_regions``); this module moves it. Each call of a transformer block
is one offload group: forward hooks on the block install torch_remat
saved-tensor hooks around its remat checkpoint, so ``pack`` sees exactly what
the checkpoint keeps for backward, with its kind and save region.

- Forward: ``pack`` queues a D2H copy on the offload stream. The GPU copy is
  freed at the end of the next block's forward, once the compute stream waits
  on the copy, so the copy overlaps that block's compute.
- Backward: a pre-hook on the autograd node of block ``i + 1``'s output queues
  group ``i``'s H2D reload on the reload stream, one block ahead. ``unpack``
  makes the compute stream wait on it.

GPU memory is allocated and freed on the compute stream, with explicit event
waits across streams (no ``record_stream``), as in FSDP2. Pinned host buffers
come from a pool reused in stream order (see _PinnedBufferPool), which keeps
the code CUDA-graph capturable. Groups live on the autograd graph, not on the
module, so several microbatches can be in flight (gradient accumulation,
pipeline parallelism). See torchtitan/distributed/ACTIVATION_OFFLOAD.md for the
design and torchtitan/distributed/ACTIVATION_REMAT.md for the saved-tensor hook
contract.
"""

import collections
import contextlib
import enum
import functools
import logging
import weakref
from dataclasses import dataclass, field
from fnmatch import fnmatch
from typing import Any, TYPE_CHECKING

import torch
import torch.nn as nn
import torch_remat as remat
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._python_dispatch import is_traceable_wrapper_subclass
from torch.utils._pytree import tree_flatten

if TYPE_CHECKING:
    from torchtitan.distributed.activation_remat import ActivationRematConfig


logger = logging.getLogger(__name__)

# Smaller storages stay on GPU: routing metadata, loss scalars, attention
# log-sum-exps. Offloading them saves no measurable memory (43 MiB per forward
# on Qwen3 30B-A3B, out of 15.5 GB offloaded) for many small copies.
_MIN_OFFLOAD_BYTES = 1 << 20


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
    """One offloaded storage: the state that ties the offloader's hooks
    together.

    The pack hook, the free point, the reload trigger and the unpack hook are
    separate callbacks, run by autograd and module hooks at different times,
    and none of them calls another; this object is how they hand off. Who
    holds it, and what each hook does to it, for group i (block i's call):

        autograd's saved variable:          group i (_OffloadGroup):
        _OffloadedView (dtype, size,        storages[storage pointer]
        stride, storage_offset)                       |
                   | .storage                         |
                   v                                  v
        +-----------------------------------------------------+
        | _OffloadedStorage                                   |
        | state, device_tensor, host_buffer, reload_event     |
        +-----------------------------------------------------+
           |   |   |   |
           |   |   |   +-- fwd i: _pack -> OFFLOADING
           |   |   |       creates it holding the GPU storage, queues the
           |   |   |       D2H copy on the offload stream, returns an
           |   |   |       _OffloadedView pointing to it
           |   |   |
           |   |   +------ end of fwd i+1: _free_gpu_copies -> OFFLOADED
           |   |           once the compute stream waits on the group's
           |   |           copy_done_event, drops device_tensor
           |   |
           |   +---------- before bwd i+1: reload trigger -> _reload
           |               -> RELOADING: allocates a new device_tensor,
           |               queues the H2D copy on the reload stream and
           |               records reload_event; the host buffer goes back
           |               to the pool
           |
           +-------------- bwd i: _unpack(view)
                           the compute stream waits on reload_event, then
                           the tensor is rebuilt as a view of
                           device_tensor

    Several saves of the same storage, whole or partial views with any shape,
    stride or dtype, share one transfer of the whole storage; each save's
    ``_OffloadedView`` points here.

    Autograd frees it, and with it the reloaded GPU copy, when it frees the
    last saved variable viewing it.
    """

    # A flat uint8 tensor over the whole storage: the original storage while
    # OFFLOADING, None once OFFLOADED, and the reloaded copy once RELOADING.
    device_tensor: torch.Tensor | None
    # A view of a pinned host buffer (flat uint8) holding the storage's bytes.
    # Returned to the pool, and set to None, once the reload is queued.
    host_buffer: torch.Tensor | None
    # Why remat retained the tensor that first packed this storage.
    kind: remat.SavedTensorKind
    state: _State = _State.OFFLOADING
    # Recorded on the reload stream right after this storage's H2D copy. It
    # syncs the reload with its consumers on the GPU: _unpack makes the compute
    # stream wait on it before backward reads the tensor, and the next D2H copy
    # into the same pinned buffer waits on it before overwriting the buffer.
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
    # The saved tensor and its version when saved, for the in-place check
    # (see _check_version). A weak reference, so the payload never keeps the
    # GPU storage alive.
    tensor_ref: weakref.ReferenceType[torch.Tensor]
    version: int


@dataclass(frozen=True, eq=False)
class _KeptTensor:
    """The payload autograd holds for a save kept on GPU, with its version
    when saved, for the in-place check (see _check_version)."""

    tensor: torch.Tensor
    version: int


@dataclass(eq=False)
class _OffloadGroup:
    """The tensors offloaded by one call of one transformer block."""

    module_fqn: str
    # Storages of the block's parameters during this call, including the
    # inner tensors of subclass parameters, which are never offloaded (see
    # _pack). Collected when the group opens, after FSDP2's pre-forward hook
    # has unsharded them.
    parameter_storages: set[int]
    # Keyed by storage pointer so repeated saves share one transfer.
    storages: dict[int, _OffloadedStorage] = field(default_factory=dict)
    # Storages a save of this call kept on GPU because its region was not
    # selected. Offloading another save of them would free nothing.
    kept_storages: set[int] = field(default_factory=set)
    # Recorded on the offload stream when the block's forward returns, after
    # every D2H copy of this group was queued, and only those before it.
    copy_done_event: torch.Event | None = None


class _PinnedBufferPool:
    """Pinned host buffers owned by one offloader, reused in GPU stream order.

    Why not let the caching host allocator recycle them: in eager mode it
    would, reusing a freed block once events recorded on the streams of its
    copies have completed. During CUDA graph capture nothing runs, so there is
    no event to poll, and the copies run again at every replay; the allocator
    therefore never reuses (or unpins) a block that a copy used during capture.
    A captured step with N microbatches would give each microbatch new
    buffers, holding N microbatches of pinned memory for the life of the
    process.

    How the pool reuses them: it keeps its buffers and orders reuse with a GPU
    wait instead of a CPU poll. ``release`` stores a buffer with the event of
    the reload (H2D copy) that last read it, and ``acquire`` makes the stream
    of the next offload (D2H copy) into it wait on that event. In eager mode
    that is plain stream ordering; during capture the record and the wait
    become an edge of the graph, so every replay orders the next microbatch's
    offload after the previous microbatch's reload, and one set of buffers
    serves all microbatches.

    That edge exists only if the graph contains both the reload that releases
    a buffer and the offload that reuses it, since a replay reuses the host
    addresses recorded at capture. Both torchtitan graph modes, a whole
    forward_backward step and one microbatch group (with PP, a whole schedule
    step), meet this. Separate forward and backward graphs with several
    microbatches in flight would not: the next microbatch's forward replay
    would overwrite buffers not yet reloaded (see ACTIVATION_OFFLOAD.md).

    Buffers are keyed by power-of-two capacity and a request gets a view of
    its first ``nbytes``, so saved tensors whose size varies from step to step
    (e.g. expert inputs under EP) share buffers instead of growing the pool
    every step. The caching host allocator rounds pinned allocations up to a
    power of two anyway, so this costs no memory.
    """

    # TODO: enforce the graph boundary invariant above. Count captures with
    # torch.cuda.graphs.register_graph_capture_start_hook, record at acquire
    # whether a capture is running and which one, and raise at release when a
    # buffer acquired during a capture is released outside it.
    def __init__(self) -> None:
        # capacity -> free buffers of that capacity, oldest release first:
        # (buffer, event of its last read, recorded during capture).
        self._free: dict[
            int, collections.deque[tuple[torch.Tensor, torch.Event, bool]]
        ] = {}

    def acquire(self, nbytes: int, copy_stream: torch.cuda.Stream) -> torch.Tensor:
        """Return an ``nbytes`` view of a buffer that ``copy_stream`` may write
        once it is current."""
        # The smallest power of two >= nbytes; requesting exactly that makes
        # the caching host allocator's own rounding a no-op.
        capacity = 1 << max(nbytes - 1, 0).bit_length()
        free = self._free.get(capacity)
        if not free:
            return torch.empty(capacity, dtype=torch.uint8, pin_memory=True)[:nbytes]
        # Any free buffer is correct (the copy waits on its last read); the
        # oldest release is the least likely to stall the copy stream.
        buffer, last_read, recorded_during_capture = free.popleft()
        # An event recorded in the other mode (eager vs capture) cannot be
        # waited on, and its read is ordered anyway: torch.cuda.graph
        # synchronizes before capture, and a replay runs on the compute
        # stream, which the copy stream waits on before each copy.
        if recorded_during_capture == torch.cuda.is_current_stream_capturing():
            copy_stream.wait_event(last_read)
        return buffer[:nbytes]

    def release(self, view: torch.Tensor, last_read: torch.Event) -> None:
        """Return the buffer behind a view from ``acquire``."""
        # acquire returns a slice of a whole buffer: its base, of the capacity.
        buffer = view._base
        assert buffer is not None
        self._free.setdefault(buffer.nbytes, collections.deque()).append(
            (buffer, last_read, torch.cuda.is_current_stream_capturing())
        )


@dataclass
class _PipelineScheduleState:
    """Created by ``defer_to_schedule``: a pipeline schedule frees and reloads
    each microbatch's last groups, naming the microbatch by the index its
    actions carry (``set_microbatch``, ``wait_offload``, ``reload_backward``).
    """

    # Set by set_microbatch for the next forward, cleared when it finishes. A
    # forward without one (e.g. a stage's metadata inference) is not tracked.
    current_microbatch: int | None = None
    # The groups of the tracked forward in progress.
    current_groups: list[_OffloadGroup] = field(default_factory=list)
    # The groups of each tracked microbatch, in block order, until
    # reload_backward takes them.
    groups_by_microbatch: dict[int, list[_OffloadGroup]] = field(default_factory=dict)
    # The last group of each tracked microbatch, held on GPU until
    # wait_offload frees it.
    last_group_by_microbatch: dict[int, _OffloadGroup] = field(default_factory=dict)


class ActivationOffloader:
    """Offload runtime for the transformer blocks of one model part.

    Installed and returned by ``ActivationRematConfig.apply`` when the policy
    offloads anything. Registers forward hooks
    on the model part and on each offloaded block; each offloaded block's
    forward runs under this offloader's saved-tensor hooks.

    Diagnostics: ``forward_stats`` holds the byte counts of the latest forward
    (logged after the first one).
    """

    def __init__(self, model_part: nn.Module, *, remat_config: "ActivationRematConfig"):
        assert (
            get_activation_offloader(model_part) is None
        ), "activation offloading is already installed on this model part"
        _offloaders[model_part] = self
        # One stream per copy direction: one microbatch's offload and
        # another's reload can run at once (sharing host bandwidth, see
        # ACTIVATION_OFFLOAD.md), and a reload never queues behind an offload.
        # TODO: order reloads after FSDP2 CPU offload's H2D parameter copies
        # (its all_gather_copy_in_stream) once FSDP2 exposes that stream, so
        # the all-gather overlaps the reload instead of following both.
        self._offload_stream = torch.cuda.Stream()
        self._reload_stream = torch.cuda.Stream()
        self._host_buffers = _PinnedBufferPool()
        # The group whose block forward is running, and its saved-tensor
        # hooks: the block's pre-hook enters them and its post-hook exits
        # them, so they cannot be one ``with`` statement.
        self._current_group: _OffloadGroup | None = None
        self._current_hooks = contextlib.ExitStack()
        # The previous group of this model-part forward, not freed yet: its
        # free point and reload trigger come at the end of the next group's
        # forward (or of the model part's).
        self._previous_group: _OffloadGroup | None = None
        # Storages of this model-part forward's inputs. The caller (e.g. a
        # pipeline stage) retains them, so offloading them would free nothing.
        self._model_part_input_storages: set[int] = set()
        # Stats of the latest (or in-progress) forward of this model part.
        self.forward_stats = OffloadStats()
        self._logged_summary = False
        # Set by defer_to_schedule.
        self._pp_schedule: _PipelineScheduleState | None = None
        # The same blocks ActivationRematConfig.apply checkpoints.
        blocks = [
            (f"layers.{layer_id}", block)
            for layer_id, block in model_part.get_submodule("layers").named_children()
        ]
        num_offloaded = (
            len(blocks) if remat_config.offload_last_block else max(len(blocks) - 1, 0)
        )
        logger.info(
            "Activation offloading enabled for %d of %d transformer blocks "
            "(offload_last_block=%s)",
            num_offloaded,
            len(blocks),
            remat_config.offload_last_block,
        )
        # Which retained tensors each offloaded block sends to CPU.
        self._block_policies = {
            fqn: remat_config.block_policy(fqn) for fqn, _ in blocks[:num_offloaded]
        }
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
        """Block pre-forward hook: open the block's offload group and enter
        the saved-tensor hooks, so ``_pack`` sees the block's saves."""
        # Under remat, the checkpoint's replay in backward calls the block's
        # forward again; that replay saves nothing to offload.
        if not torch.is_grad_enabled() or remat.is_recomputing():
            return
        assert (
            self._current_group is None
        ), f"transformer block {fqn} started inside {self._current_group.module_fqn}"
        group = _OffloadGroup(
            module_fqn=fqn, parameter_storages=_parameter_storages(module)
        )
        self._current_group = group
        # capture_context snapshots the group where a tensor is produced; see
        # _pack for why a save remat defers must still land in it.
        self._current_hooks.enter_context(
            remat.saved_tensors_hooks(
                self._pack, self._unpack, capture_context=lambda: group
            )
        )

    def _group_post_forward(self, module: nn.Module, args: Any, output: Any) -> None:
        """Block post-forward hook: close the group and record its
        ``copy_done_event``; free the previous group (its free point) and
        install the previous group's reload trigger on this block's output."""
        group = self._current_group
        # No group was opened: grad disabled, or remat's replay.
        if group is None:
            return
        self._current_hooks.close()
        self._current_group = None
        if group.storages:
            # Queued after this group's D2H copies, so the event completes
            # once they all finish. The free point makes the compute stream
            # wait on it before dropping the group's GPU copies.
            group.copy_done_event = self._offload_stream.record_event()
            pp_schedule = self._pp_schedule
            if pp_schedule is not None and pp_schedule.current_microbatch is not None:
                pp_schedule.current_groups.append(group)
        previous, self._previous_group = self._previous_group, group
        if previous is not None:
            # Group i - 1's copies had group i's forward to finish.
            self._free_gpu_copies(previous)
            # Group i's output triggers the reload of group i - 1.
            self._prefetch_on_grad(output, group_to_reload=previous)

    def _model_part_pre_forward(
        self, module: nn.Module, args: Any, kwargs: Any
    ) -> None:
        """Model-part pre-forward hook: reset the per-forward state and record
        the model part's input storages, which are never offloaded."""
        self._previous_group = None
        if self._pp_schedule is not None:
            self._pp_schedule.current_groups = []
        self._model_part_input_storages = {
            t.untyped_storage().data_ptr()
            for t in tree_flatten((args, kwargs))[0]
            if isinstance(t, torch.Tensor)
        }
        self.forward_stats = OffloadStats()

    def _model_part_post_forward(
        self, module: nn.Module, args: Any, output: Any
    ) -> None:
        """Model-part post-forward hook: free the last group (or, for a
        microbatch the pipeline schedule tracks, leave it to ``wait_offload``)
        and install its reload trigger on the model part's output."""
        # The last group runs first in backward, so the model part's output
        # triggers its reload. On a non-last pipeline stage the backward
        # starts as soon as this output's gradient arrives, so the reload
        # cannot overlap and this free would wait on the copies; hence
        # offload_last_block=False. A schedule that tracks the microbatch
        # instead frees the last group after another action and reloads it
        # ahead of the backward; the trigger stays as a fallback.
        # The schedule state, if the schedule tagged this forward.
        tracked = None
        microbatch = -1
        pp_schedule = self._pp_schedule
        if pp_schedule is not None and pp_schedule.current_microbatch is not None:
            if torch.is_grad_enabled():
                tracked, microbatch = pp_schedule, pp_schedule.current_microbatch
            pp_schedule.current_microbatch = None
        last_group = self._previous_group
        if last_group is not None:
            self._previous_group = None
            if tracked is not None:
                tracked.last_group_by_microbatch[microbatch] = last_group
            else:
                self._free_gpu_copies(last_group)
            self._prefetch_on_grad(output, group_to_reload=last_group)
        if tracked is not None:
            tracked.groups_by_microbatch[microbatch] = tracked.current_groups
            tracked.current_groups = []
        if not self._logged_summary and torch.is_grad_enabled():
            self._logged_summary = True
            self._log_first_forward_summary()

    def _log_first_forward_summary(self) -> None:
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
        """Hand the timing of each microbatch's last group to a pipeline
        schedule. For each microbatch, the schedule calls ``set_microbatch``
        before the stage's forward, ``wait_offload`` after a later action, and
        ``reload_backward`` ahead of the stage's backward, passing the
        microbatch index its actions carry.

        Like FSDP's UNSHARD/RESHARD actions, this lets the schedule overlap
        the transfers at the stage boundary with other work on the rank, in
        any order the schedule runs microbatches, so the last blocks need not
        be skipped. Within the stage, gradient hooks still prefetch one block
        ahead.
        """
        self._pp_schedule = _PipelineScheduleState()

    def set_microbatch(self, microbatch_index: int) -> None:
        """Tag the next forward of this model part with ``microbatch_index``."""
        pp_schedule = self._pp_schedule
        assert pp_schedule is not None, "set_microbatch needs defer_to_schedule"
        assert pp_schedule.current_microbatch is None, (
            f"microbatch {pp_schedule.current_microbatch} was set but its "
            "forward has not run"
        )
        assert microbatch_index not in pp_schedule.groups_by_microbatch, (
            f"microbatch {microbatch_index} already has a forward whose "
            "backward was not reloaded"
        )
        pp_schedule.current_microbatch = microbatch_index

    def wait_offload(self, microbatch_index: int) -> None:
        """Free the GPU copies of ``microbatch_index``'s last group.

        Call it after at least one action following that microbatch's
        forward, so the copies had that action's compute to finish; the
        compute stream still waits on them. After the microbatch's backward
        it does nothing.
        """
        pp_schedule = self._pp_schedule
        assert pp_schedule is not None, "wait_offload needs defer_to_schedule"
        last_group = pp_schedule.last_group_by_microbatch.pop(microbatch_index, None)
        assert last_group is not None, (
            f"microbatch {microbatch_index} has no last group awaiting " "wait_offload"
        )
        self._free_gpu_copies(last_group)

    def reload_backward(self, microbatch_index: int, num_groups: int = 1) -> None:
        """Queue the reload of ``microbatch_index``'s last ``num_groups``
        groups, last block first.

        Call it one transfer latency before that microbatch's backward.
        ``num_groups=1`` matches the in-stage prefetch depth; reloading every
        group brings back the stage's whole stash for that microbatch at once,
        which only pays off for stages of one or two blocks.
        """
        if num_groups < 1:
            raise ValueError(f"num_groups must be >= 1, got {num_groups}")
        pp_schedule = self._pp_schedule
        assert pp_schedule is not None, "reload_backward needs defer_to_schedule"
        groups = pp_schedule.groups_by_microbatch.pop(microbatch_index, None)
        assert (
            groups is not None
        ), f"microbatch {microbatch_index} has no forward awaiting reload_backward"
        for group in reversed(groups[-num_groups:]):
            self._reload_group(group)

    def _pack(self, tensor: torch.Tensor) -> Any:
        packed = self._offload_or_keep(tensor)
        if packed is tensor:
            return _KeptTensor(tensor, tensor._version)
        return packed

    def _offload_or_keep(self, tensor: torch.Tensor) -> Any:
        """Return an ``_OffloadedView`` for a tensor to offload, or the tensor
        itself to keep it on GPU."""
        info = remat.current_saved_tensor_info()
        group = self._current_group
        if group is None:
            return tensor
        # A save region's output is packed only once a consumer needs it for
        # backward, so remat may defer its pack past the region; it then
        # passes the group captured where the output was produced
        # (info.context), while self._current_group is the consumer's block.
        # In the same block the two agree. A consumer in a later block would
        # need the producer's group, which already recorded copy_done_event
        # when its block's forward returned: a copy queued now runs after that
        # event, so the group's free point would drop the GPU storage while
        # the copy still reads it. Filing it under the consumer's group
        # instead would offload it under another block's policy and schedule.
        #
        # This is unlikely because of how remat applies to transformer
        # blocks: each block is one checkpoint, and its save regions
        # (attention, MLP projections) are consumed inside it. The block's
        # output is the residual add, outside any region, and reaches the
        # next block as that block's checkpoint input, not as a deferred
        # region output. Raise rather than guess if a model breaks this.
        assert info.context is None or info.context is group, (
            f"a save produced in {info.context.module_fqn} was deferred to "
            f"{group.module_fqn}; offload groups do not support saves deferred "
            "across transformer blocks"
        )
        kind = info.kind
        if type(tensor) is not torch.Tensor or tensor.device.type != "cuda":
            # A subclass parameter (FSDP2's unsharded MXFP8 weight) is a weight,
            # not a retained activation: all its storages are parameter storages.
            if _storages(tensor).issubset(group.parameter_storages):
                return tensor
            return self._skip(tensor, "not a plain CUDA tensor")
        policy = self._block_policies[group.module_fqn]
        storage_ptr = tensor.untyped_storage().data_ptr()
        # An already-offloaded storage is offloaded for every save of it.
        selected = storage_ptr in group.storages
        if kind is remat.SavedTensorKind.CHECKPOINT_INPUT:
            if not tensor.requires_grad:
                return tensor
            selected = selected or policy.offload_block_input
            region = "block input"
        elif info.region is None:
            # Retained for recomputation (recompute_needs_tensor), not by a
            # save region. Never offloaded; large ones show up in the summary.
            region = "outside any region"
        else:
            region = info.region
            selected = selected or any(
                fnmatch(region, pattern) for pattern in policy.offload_regions
            )
        if not selected:
            group.kept_storages.add(storage_ptr)
            return self._skip(tensor, f"{region} not selected")
        if storage_ptr in group.kept_storages:
            return self._skip(tensor, "storage kept on GPU by another save")
        if storage_ptr in self._model_part_input_storages:
            return self._skip(tensor, "model-part input")
        storage = tensor.untyped_storage()
        # Never offload a parameter: it frees nothing (the module holds it),
        # and under FSDP2 the async D2H copy would race reshard freeing the
        # unsharded storage. A save region that uses a weight internally
        # keeps it for backward; the nn.Parameter itself is declined above as
        # not a plain tensor, and plain-tensor aliases of its storage (a view,
        # a TP weight's local tensor, the MXFP8 qdata and scales that MXFP8
        # Dist-MoE saves) are caught here.
        if storage.data_ptr() in group.parameter_storages:
            return tensor
        if storage.nbytes() < _MIN_OFFLOAD_BYTES:
            return self._skip(tensor, "smaller than 1 MiB")
        # Offload the whole storage, even for a partial view: other saves
        # often view the rest (e.g. q, k and v slices of a fused QKV output),
        # and only once all are offloaded does dropping the GPU copy free it.
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
            weakref.ref(tensor),
            tensor._version,
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
        # Recorded only if the group copied something. Not group.storages:
        # under a pipeline schedule, RELOAD can run before WAIT_OFFLOAD (a last
        # stage's backward follows its forward at once), and reloading empties
        # the group while its D2H copies must still be joined here.
        if group.copy_done_event is None:
            return
        # The compute stream owns the GPU memory: without this wait the
        # allocator could hand it to later compute while the copy reads it.
        # This group's event, not the whole offload stream, so later groups'
        # copies do not delay compute.
        torch.cuda.current_stream().wait_event(group.copy_done_event)
        for offloaded in group.storages.values():
            if offloaded.state is _State.OFFLOADING:
                assert offloaded.device_tensor is not None
                nbytes = offloaded.device_tensor.untyped_storage().nbytes()
                # device_tensor still has the storage's size at pack.
                if nbytes < offloaded.device_tensor.numel():
                    raise RuntimeError(
                        "Activation offloading: a storage saved in "
                        f"{group.module_fqn} ({offloaded.kind.name}) was freed "
                        "(resized) by something other than autograd before "
                        "its copy to CPU was known to finish, so the copy may "
                        "have read reused memory. Storages managed elsewhere, "
                        "e.g. FSDP2's unsharded parameter data, cannot be "
                        "offloaded."
                    )
                storage = StorageWeakRef(offloaded.device_tensor.untyped_storage())
                # The free: pack replaced each saved tensor with an
                # _OffloadedView, so this is the only reference to the GPU
                # storage, and dropping it returns the memory to the caching
                # allocator on the compute stream, unless something else
                # still holds the storage.
                offloaded.device_tensor = None
                offloaded.state = _State.OFFLOADED
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

        Background: the autograd engine runs backward one node at a time (a
        ``grad_fn``, or ``AccumulateGrad`` for a leaf), and calls a node's
        hooks in a fixed order (``call_function`` in
        torch/csrc/autograd/engine.cpp)::

            gradients of the node's outputs arrive (summed)
            1. tensor pre-hooks    Tensor.register_hook on a tensor it produced
            2. retains_grad hooks  Tensor.retain_grad
            3. node pre-hooks      grad_fn.register_prehook
            4. the node itself     gradients with respect to its inputs
            5. node post-hooks     grad_fn.register_hook

        The order is by kind; hooks of one kind run in registration order.

        The trigger is a node pre-hook on ``out_i.grad_fn`` (step 3) so that
        it runs after FSDP2's pre-backward for block ``i``, a tensor hook on
        ``out_i`` (step 1), whichever was registered first. That hook
        unshards block ``i``: its copy-out on the compute stream starts with
        a small H2D copy of ``split_with_sizes_copy``'s metadata, and the
        reload, which waits on the compute stream, queues behind it. Queued
        first, the reload could hold that copy, and block ``i``'s backward,
        in the copy engine (H100, pytorch/torchtitan#467). A module backward
        pre-hook would also fire before FSDP2's, and it wraps the outputs in
        a custom autograd Function, so an in-place update of an output
        raises.
        """

        def hook(grad_outputs: tuple[torch.Tensor | None, ...]) -> None:
            self._reload_group(group_to_reload)

        # Like FSDP2, hook every output: the first gradient to arrive starts
        # the reload. Later firings do nothing (see _reload_group).
        for tensor in tree_flatten(output)[0]:
            if isinstance(tensor, torch.Tensor) and tensor.requires_grad:
                assert tensor.grad_fn is not None, (
                    "an offloaded block returned a leaf tensor, which has no "
                    "autograd node for the reload trigger"
                )
                # Not tensor.register_hook: a node pre-hook runs after FSDP2's
                # tensor hook on the same output (see the order above).
                tensor.grad_fn.register_prehook(hook)

    def _reload_group(self, group: _OffloadGroup) -> None:
        """Queue the H2D reload of every still-offloaded storage of ``group``.

        Idempotent, so callers can race: the first reloads, later calls find
        the group empty. Callers: the reload trigger, and ``reload_backward``
        under a pipeline schedule.
        """
        # Pack order suits the replay, which reads the block input and then
        # the SAVE_OUTPUTs in that order. TODO: for offload_regions, queue
        # BACKWARD tensors last and in reverse, since only the gradient pass
        # after the replay reads them; matters only when reloads barely
        # finish in time.
        for offloaded in group.storages.values():
            if offloaded.state is _State.OFFLOADED:
                self._reload(offloaded)
        # Autograd keeps the trigger, and so this group, alive while the
        # graph is referenced (e.g. by a loss kept for logging). Dropping the
        # group's references leaves autograd's payloads as the only owners,
        # so each reloaded copy is freed with its saved tensor.
        group.storages.clear()

    def _reload(self, offloaded: _OffloadedStorage) -> None:
        host_buffer = offloaded.host_buffer
        assert host_buffer is not None
        # Allocated on the compute stream, so freed in its order. The reload
        # stream waits on the compute stream, which orders the copy after this
        # allocation and after FSDP2's copy-out (see _prefetch_on_grad).
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
        if isinstance(packed, _KeptTensor):
            _check_version(packed.tensor, packed.version)
            return packed.tensor
        assert isinstance(packed, _OffloadedView)
        tensor = packed.tensor_ref()
        if tensor is not None:
            _check_version(tensor, packed.version)
        offloaded = packed.storage
        if offloaded.state is _State.OFFLOADED:
            # The trigger on the next block's output (or the model part's)
            # fires when that output receives a gradient, before this block's
            # backward can start.
            raise RuntimeError(
                "Activation offloading: backward read an offloaded "
                f"{offloaded.kind.name} tensor that no reload trigger reloaded. "
                "The trigger fires when the next block's output receives a "
                "gradient; an output that gets no gradient leaves the previous "
                "block's activations on CPU."
            )
        if offloaded.state is _State.RELOADING:
            assert offloaded.reload_event is not None
            torch.cuda.current_stream().wait_event(offloaded.reload_event)
        # Otherwise still OFFLOADING: the GPU copy has not been freed yet.
        assert offloaded.device_tensor is not None
        # The reloaded copy holds the whole storage, so the saved tensor is the
        # same view of it as of the original.
        return offloaded.device_tensor.view(packed.dtype).as_strided(
            packed.size, packed.stride, packed.storage_offset
        )


# Weak keys: an entry goes away with its model part.
_offloaders: weakref.WeakKeyDictionary[
    nn.Module, ActivationOffloader
] = weakref.WeakKeyDictionary()


def get_activation_offloader(model_part: nn.Module) -> ActivationOffloader | None:
    """The offloader installed on ``model_part``, or None when its remat
    policy offloads nothing. The PP schedule uses it to reach each stage's
    offloader."""
    return _offloaders.get(model_part)


def _check_version(tensor: torch.Tensor, saved_version: int) -> None:
    """Raise autograd's error for a saved tensor modified in place since it
    was saved.

    Autograd checks a saved tensor's version when backward reads it, but
    skips the check for tensors that saved-tensor hooks pack; pack records
    the version and unpack runs the check here instead. An offloaded tensor
    is checked only while something still references it; once it is gone,
    nothing can modify it through it.
    """
    if tensor._version != saved_version:
        raise RuntimeError(
            "one of the variables needed for gradient computation has been "
            f"modified by an inplace operation: [{tensor.type()} "
            f"{list(tensor.shape)}] is at version {tensor._version}; expected "
            f"version {saved_version} instead. Activation offloading's "
            "saved-tensor hooks bypass autograd's own check of this."
        )


def _storages(tensor: torch.Tensor) -> set[int]:
    """Storage pointers of ``tensor``, or of a wrapper subclass's inner
    tensors (a DTensor's local tensor, FSDP2's unsharded MXFP8 operands),
    which have no storage of their own."""
    if not is_traceable_wrapper_subclass(tensor):
        return {tensor.untyped_storage().data_ptr()}
    inner_names, _ = tensor.__tensor_flatten__()
    return set().union(*(_storages(getattr(tensor, name)) for name in inner_names))


def _parameter_storages(module: nn.Module) -> set[int]:
    return set().union(*(_storages(p) for p in module.parameters()))

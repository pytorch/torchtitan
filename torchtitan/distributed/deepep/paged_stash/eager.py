# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

r"""
Eager-trainer frontend for MoE paged stashing.

The eager counterpart of GraphTrainer's paged stash memory policy, over the same
runtime (``runtime.py``). Megatron opens ``saved_tensors_hooks`` around
Transformer Engine's fused grouped MLP and pages the tensors Transformer Engine
tagged. Here the regions are ``PagedAC.page_regions`` -- paging is an
activation checkpointing policy -- and nothing is tagged:

* **Region hooks.** A forward pre-hook on every paged region joins the
  stashes still in flight (Megatron's ``paged_stash_group_start``) and pushes a
  ``torch_remat.saved_tensors_hooks`` pair for the region's forward; the forward
  hook pops them and launches the region's stashes
  (``paged_stash_group_commit``). torch_remat owns what a checkpointed block
  retains and delegates it to that pair, so the region's saves arrive whether or
  not an enclosing block is checkpointed. Gradient
  hooks on the region's output and input mark its backward's start, which joins
  the stashes, and its end, which prefetches the next backward's reloads, as the
  backwards of Megatron's ``PipelinePostScheduleFunction`` and
  ``PipelinePreScheduleFunction`` do.
* **Automatic selection.** The static-capacity token dispatcher publishes the
  padded and live row counts of every dispatch
  (``hybridep.register_dispatch_observer``). A tensor saved inside the region
  after its dispatch is paged when its storage reads as rows of that capacity
  -- ``infer_slot_dims``' rule, which GraphTrainer applies to traced values --
  and is large enough to be worth a round trip. The region's own inputs and
  outputs are the caller's tokens, every row live, so they are never paged even
  when the token count equals the capacity. Saves of one storage, such as
  the two halves of a fused gate/up projection, share one stash, and each
  save's view is rebuilt on reload.
* **Owning the region.** Nothing in a paged region is recomputed.
  ``PagedAC.apply`` stamps the subtree so that every region it declares is
  retained and none is recomputed -- ownership stated in ``RegionAC``'s own
  vocabulary rather than forced on another policy from inside a hook. A
  checkpointed block's recompute therefore replays the paged region from what it
  retained instead of rerunning its GEMMs and all-to-alls, and these hooks stand
  down while ``torch_remat.is_recomputing()``. Backward reads two
  ``SavedTensorKind``s from an owned region and both are paged: ``BACKWARD``,
  autograd's own saves, and ``SAVE_OUTPUT``, a region output held real because
  a recomputed consumer reads it -- which is what
  ``remat.recompute_needs_tensor`` forces on the capacity-padded dispatch
  output, the largest thing here worth paging. ``CHECKPOINT_INPUT`` is the
  enclosing block's own input, which is not this region's to page. Pipeline
  metadata inference pages nothing and keeps its saves resident, since they
  belong to no step. The pager's own tensor ops run with dispatch modes
  disabled.
* **Schedule.** The measure step records the paged-layer schedule as it runs
  -- Megatron's ``_pp_schedule`` -- and the steps that run Python afterwards
  verify it. Each stage forward's pipeline context tells a region which stage it
  is in and keeps PP metadata inference out of it. The eager trainer captures a
  whole step in one CUDA graph, so the skips and prefetches decided at capture
  are right for every replay.
* **Validation.** The first step that pages runs eagerly, before capture. Each
  reload's live rows are checked against what was stashed, so a tensor whose live
  data is not a row prefix fails then, not silently on every later step.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, cast, Literal, TYPE_CHECKING

import torch
import torch.nn as nn
import torch_remat as remat
from torch.distributed.pipelining.schedules import (
    PipelineScheduleMulti,
    PipelineScheduleSingle,
)
from torch.utils import _pytree as pytree
from torch.utils._python_dispatch import _disable_current_modes
from torch.utils.hooks import RemovableHandle

from torchtitan.distributed.activation_checkpoint import paged_stash_regions, PagedAC
from torchtitan.distributed.deepep.paged_stash.runtime import (
    _num_pages,
    PagedStashManager,
    ScheduleEntry,
    StashRecord,
    StashSlot,
    wait_for_copy,
    worth_paging,
)

if TYPE_CHECKING:
    from torch.distributed.pipelining import PipelineStageInfo
    from torch.distributed.pipelining.schedules import _PipelineSchedule


logger = logging.getLogger(__name__)

_E8M0 = torch.float8_e8m0fnu
_SCALE_BLOCK = 32
_CHECKSUM_CHUNK_BYTES = 32 << 20


def _storage_key(t: torch.Tensor) -> tuple[int, int]:
    storage = t.untyped_storage()
    return storage.data_ptr(), storage.nbytes()


def _storage_slot_dims(t: torch.Tensor, capacity: int) -> tuple[int, int] | None:
    """``(token_divisor, hidden)`` reading ``t``'s whole storage as token rows
    of ``capacity``, or ``None`` if it does not.

    The stash pages a storage, not a view: it copies a prefix of rows of
    ``hidden`` elements from offset 0. ``t`` -- or the base it views -- has to
    witness that layout: a leading dimension of ``capacity`` rows whose row
    stride tiles the storage, or, for MXFP8 block scales (e8m0, 32 tokens per
    row), a flat buffer covering the storage. That is ``infer_slot_dims``' rule,
    applied to storages, so that a strided view of a fused tensor pages the
    whole tensor once.
    """
    divisor = _SCALE_BLOCK if t.dtype is _E8M0 else 1
    if capacity % divisor:
        return None
    rows = capacity // divisor
    total = t.untyped_storage().nbytes() // t.element_size()
    if rows <= 0 or total % rows:
        return None
    hidden = total // rows
    for witness in (t, t._base):
        if witness is None or witness.dtype is not t.dtype:
            continue
        if witness.dim() >= 1 and witness.shape[0] == rows:
            if witness.dim() == 1 or witness.stride(0) == hidden:
                return divisor, hidden
        if divisor != 1 and witness.dim() == 1 and witness.numel() == total:
            # A flat block-scale buffer has no rows to say otherwise.
            return divisor, hidden
    return None


def _live_checksum(rows: torch.Tensor, num_live_rows: int) -> torch.Tensor:
    """A position-weighted sum of the bytes of the first ``num_live_rows`` rows
    of ``rows``.

    Changes if a live row comes back with different contents or at another
    position, which is what a wrong layout, a race or a freelist bug looks like
    after a round trip through the pages. Summed a chunk of rows at a time: the
    int64 sum materializes its input at eight times the size.
    """
    data = rows.reshape(rows.shape[0], -1).view(torch.uint8)[:num_live_rows]
    chunk_rows = max(1, _CHECKSUM_CHUNK_BYTES // (8 * max(data.shape[1], 1)))
    total = torch.zeros((), dtype=torch.int64, device=rows.device)
    for start in range(0, data.shape[0], chunk_rows):
        chunk = data[start : start + chunk_rows]
        positions = torch.arange(
            start + 1, start + 1 + chunk.shape[0], device=rows.device
        )
        total += (chunk.sum(dim=1, dtype=torch.int64) * positions).sum()
    return total


def _num_live_rows(num_tokens: torch.Tensor, token_divisor: int) -> int:
    return int(num_tokens.reshape(-1)[0].item()) // token_divisor


def _alias_first_grad_input(
    args: tuple[Any, ...],
) -> tuple[tuple[Any, ...], torch.Tensor | None]:
    """Replace the first tensor argument that requires grad by an alias, whose
    gradient is then exactly this region's contribution to the argument's."""
    for i, arg in enumerate(args):
        if isinstance(arg, torch.Tensor) and arg.requires_grad:
            alias = arg.view_as(arg)
            return args[:i] + (alias,) + args[i + 1 :], alias
    return args, None


_StashState = Literal["pending", "paged", "resident", "dropped"]


@dataclass(eq=False)
class _Stash:
    """One capacity-padded storage saved inside one region forward, and what
    the region's commit makes of it: ``paged``, kept ``resident``, or
    ``dropped``."""

    flat: torch.Tensor | None
    """The whole storage as a flat tensor of ``dtype``, while it is held."""
    numel: int
    dtype: torch.dtype
    device: torch.device
    version: int
    layout: tuple[int, int] | None
    """``(token_divisor, hidden)`` if the stash can hold the storage."""
    description: str
    state: _StashState = "pending"
    backward_reads: bool = False
    """Whether backward reads it, rather than only checkpointing's replay."""
    record: StashRecord | None = None
    name: str = ""
    taken: bool = False
    checksum: torch.Tensor | None = None


@dataclass(frozen=True, eq=False)
class _StashedView:
    """What autograd holds for a save of a stashed storage: the stash, and the
    view of the storage the save was."""

    stash: _Stash
    dtype: torch.dtype
    size: torch.Size
    stride: tuple[int, ...]
    offset: int


@dataclass(eq=False)
class _RegionForward:
    """One forward of a paged-stash region: one layer of one microbatch."""

    fqn: str
    stage_index: int
    layer_id: int
    microbatch: int
    inputs: set[tuple[int, int]] = field(default_factory=set)
    """The storages of the region's tensor arguments."""
    capacity: int | None = None
    num_tokens: torch.Tensor | None = None
    stashes: dict[tuple[int, int], _Stash] = field(default_factory=dict)

    def entry(self, *, forward: bool) -> ScheduleEntry:
        return ScheduleEntry(forward, self.stage_index, self.layer_id, self.microbatch)


@dataclass(eq=False)
class _Frame:
    """What a region's forward pre-hook leaves for its forward hook."""

    contexts: contextlib.ExitStack
    region: _RegionForward | None = None
    input_alias: torch.Tensor | None = None


class EagerPagedStash:
    """Pages capacity-padded activations saved inside the configured region of
    the eager trainer's model.

    Args:
        config: The paged stash settings.
        model_parts: The model parts whose regions to page: one, or one per
            local pipeline stage.
        pp_schedule: The pipeline schedule, whose stages report each forward's
            stage and whether it is metadata inference. ``None`` without
            pipeline parallelism.
        device: The local device.
    """

    def __init__(
        self,
        ac_config: PagedAC.Config,
        *,
        model_parts: Sequence[nn.Module],
        pp_schedule: _PipelineSchedule | None,
        device: torch.device,
    ) -> None:
        from torchtitan.distributed.deepep.hybridep import register_dispatch_observer

        self.ac_config = ac_config
        self.config = ac_config.paged_stash
        config = self.config
        self.manager = PagedStashManager.get_instance()
        manager = self.manager
        manager.enabled = True
        manager.device = device
        manager.page_size = config.page_size
        manager.buffer_size_factor_cuda = config.buffer_size_factor_cuda
        manager.buffer_size_factor_cpu = config.buffer_size_factor_cpu
        manager.skip_immediate_backward = config.skip_immediate_backward

        self._tls = threading.local()
        self._handles: list[RemovableHandle] = []
        self._slots: dict[tuple[int, int, int], StashSlot] = {}
        self._forward_counts: dict[tuple[int, int], int] = {}
        self._events: list[ScheduleEntry] = []
        self._recorded: list[ScheduleEntry] | None = None
        self._measuring = False
        self._validating = False
        self._validated = False
        self._reloads_checked = 0
        self._dispatches = 0
        self._stashes = 0
        self._stashed_bytes = 0
        self._unread: set[str] = set()
        self._padded_unpaged: dict[str, set[str]] = {}

        # ``PagedAC.apply`` already stamped these regions as owned: every region
        # inside one is retained and none is recomputed. This finds the same
        # modules again to hang the per-layer hooks on, since the policy object
        # does not survive parallelization.
        regions: dict[int, tuple[str, nn.Module]] = {}
        for model_part in model_parts:
            regions.update(paged_stash_regions(model_part, ac_config.page_regions))
        if not regions:
            raise ValueError(
                f"PagedAC.page_regions = {list(ac_config.page_regions)!r} matches "
                "no module of this rank's model parts."
            )
        for layer_id, (fqn, module) in regions.items():
            self._install_region(module, fqn, layer_id)
        if pp_schedule is not None:
            stages = (
                pp_schedule._stages
                if isinstance(pp_schedule, PipelineScheduleMulti)
                else [cast(PipelineScheduleSingle, pp_schedule)._stage]
            )
            for stage in stages:
                self._handles.append(
                    stage.register_forward_context(self._stage_forward)
                )
        self._remove_observer: Callable[[], None] | None = register_dispatch_observer(
            self._on_dispatch
        )
        logger.info(
            "Paged stash enabled on %d region(s) matching %s: page_size=%d, "
            "factor_cuda=%.2f, factor_cpu=%.2f, overflow_check=%s",
            len(regions),
            list(ac_config.page_regions),
            config.page_size,
            config.buffer_size_factor_cuda,
            config.buffer_size_factor_cpu,
            config.overflow_check,
        )

    def close(self) -> None:
        """Remove every hook this frontend installed."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if self._remove_observer is not None:
            self._remove_observer()
            self._remove_observer = None

    # ------------------------------------------------------------------
    # Per-step lifecycle
    # ------------------------------------------------------------------
    def run_step(self, fn: Callable[..., Any], *args: Any) -> Any:
        """Run one attempt of a step's forward-backward, after the runner has
        advanced the manager into it."""
        self._begin_step()
        result = fn(*args)
        self._end_step()
        return result

    def _begin_step(self) -> None:
        self._forward_counts.clear()
        self._events = []
        self._dispatches = 0
        self._stashes = 0
        self._stashed_bytes = 0
        self._unread.clear()
        self._reloads_checked = 0
        self._measuring = self.manager.measuring
        # The first step that pages; see ``_materialize``.
        self._validating = self.manager.active and not self._validated

    def _end_step(self) -> None:
        if self._unread:
            # Its pages, and any reload prefetched for it, would be held until
            # the next step resets the stash.
            raise RuntimeError(
                "Paged stash: backward never read what these regions saved: "
                f"{', '.join(sorted(self._unread))}. A paged-stash region's "
                "saves must all be read by the step's backward."
            )
        if self._measuring:
            self._finish_measure_step()
        elif self._events and self._recorded is not None:
            # Replayed steps run no Python and record nothing; a step that ran
            # Python must have run the schedule the decisions were made for.
            if self._events != self._recorded:
                raise RuntimeError(
                    "Paged stash: this step ran its paged layers in a different "
                    f"order than the measure step ({len(self._events)} entries vs "
                    f"{len(self._recorded)}), so the stashes it skipped and the "
                    "reloads it prefetched were planned for another schedule."
                )
        if self._reloads_checked:
            self._validated = True
            logger.info(
                "Paged stash: all %d reloads of the first paging step matched "
                "their stashes.",
                self._reloads_checked,
            )

    def _finish_measure_step(self) -> None:
        self._recorded = list(self._events)
        self.manager.set_recorded_schedule(self._recorded)
        if self._dispatches == 0:
            logger.warning(
                "Paged stash: no static-capacity token dispatch ran inside a "
                "paged-stash region; nothing will be paged. Paging needs a "
                "dispatcher with a static capacity, e.g. HybridEP with "
                "non_blocking_capacity_factor set."
            )
            return
        if self._stashes == 0:
            logger.warning(
                "Paged stash: the paged-stash regions saved no capacity-padded "
                "activation for backward; nothing will be paged."
            )
            return
        logger.info(
            "Paged stash: the measure step stashed %d storage(s) (%.2f MiB "
            "padded) over %d paged-layer schedule entries",
            self._stashes,
            self._stashed_bytes / (1024 * 1024),
            len(self._recorded),
        )
        for fqn, tensors in sorted(self._padded_unpaged.items()):
            logger.warning(
                "Paged stash: %s saves capacity-padded tensor(s) the stash cannot "
                "hold, which stay resident at their padded size: %s",
                fqn,
                ", ".join(sorted(tensors)),
            )

    # ------------------------------------------------------------------
    # Where a forward runs
    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def _stage_forward(self, info: PipelineStageInfo) -> Iterator[None]:
        previous = getattr(self._tls, "stage", None)
        self._tls.stage = info
        try:
            yield
        finally:
            self._tls.stage = previous

    def _forward_mode(self) -> str:
        """How a region forward runs: ``forward``, paged; ``recompute``, when
        torch_remat replays the enclosing block inside backward; ``resident``,
        for pipeline metadata inference, whose saves are not part of any step;
        or ``off`` without autograd."""
        if not torch.is_grad_enabled():
            return "off"
        if remat.is_recomputing():
            return "recompute"
        info = getattr(self._tls, "stage", None)
        if info is not None and info.is_metadata_inference:
            return "resident"
        return "forward"

    def _on_dispatch(
        self, hidden: torch.Tensor, tokens_per_expert: torch.Tensor
    ) -> None:
        region = getattr(self._tls, "region", None)
        if region is None:
            return
        if region.capacity is not None:
            raise RuntimeError(
                f"Paged stash: region {region.fqn} dispatched twice; a paged-stash "
                "region must contain exactly one static-capacity token dispatch."
            )
        with _disable_current_modes():
            region.capacity = int(hidden.shape[0])
            region.num_tokens = tokens_per_expert.sum()
        self._dispatches += 1

    # ------------------------------------------------------------------
    # Region hooks
    # ------------------------------------------------------------------
    def _install_region(self, module: nn.Module, fqn: str, layer_id: int) -> None:
        frames: list[_Frame] = []

        def pre_hook(
            module: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
        ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
            frame = _Frame(contextlib.ExitStack())
            frames.append(frame)
            if self._forward_mode() != "forward":
                # ``off``: no autograd, so nothing is saved. ``recompute``:
                # paging owns this region, so torch_remat replays it from what
                # it retained instead of rerunning it -- this hook still fires,
                # but no save of ours reaches backward. ``resident``: pipeline
                # metadata inference, whose saves belong to no step. None of
                # them stashes, and none may run the group start, which would
                # free a live layer's padded activations.
                return None

            stage = getattr(self._tls, "stage", None)
            stage_index = stage.stage_index if stage is not None else 0
            key = (stage_index, layer_id)
            microbatch = self._forward_counts.get(key, 0)
            self._forward_counts[key] = microbatch + 1
            region = _RegionForward(
                fqn,
                stage_index,
                layer_id,
                microbatch,
            )
            frame.region = region
            with _disable_current_modes():
                # Megatron's group start: the previous paged layer's copies
                # finish, and the padded activations they read are freed,
                # before this layer reads its dispatch output.
                self.manager.join_stashes()
                region.inputs = {
                    _storage_key(arg)
                    for arg in pytree.tree_leaves((args, kwargs))
                    if type(arg) is torch.Tensor
                }
                args, frame.input_alias = _alias_first_grad_input(args)
            frame.contexts.enter_context(
                remat.saved_tensors_hooks(
                    lambda tensor: self._pack(region, tensor), self._unpack
                )
            )
            previous = getattr(self._tls, "region", None)
            self._tls.region = region
            frame.contexts.callback(setattr, self._tls, "region", previous)
            return args, kwargs

        def forward_hook(
            module: nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
        ) -> None:
            frame = frames.pop()
            frame.contexts.close()
            if frame.region is not None and isinstance(output, torch.Tensor):
                if self._commit(frame.region, output):
                    self._watch_backward(frame.region, output, frame.input_alias)

        self._handles.append(
            module.register_forward_pre_hook(pre_hook, with_kwargs=True)
        )
        self._handles.append(
            module.register_forward_hook(
                forward_hook, with_kwargs=True, always_call=True
            )
        )

    def _pack(self, region: _RegionForward, tensor: torch.Tensor) -> Any:
        # Backward reads two kinds of save from a region PagedAC owns:
        # ``BACKWARD``, autograd's own, and ``SAVE_OUTPUT``, a region output
        # held real because a recomputed consumer reads it -- which is what
        # ``remat.recompute_needs_tensor`` forces on the capacity-padded
        # dispatch output. Both are held from forward to backward, so both are
        # worth paging. ``CHECKPOINT_INPUT`` is the enclosing block's own
        # input, which is not this region's to page.
        backward_reads = (
            remat.current_saved_tensor_info().kind
            is not remat.SavedTensorKind.CHECKPOINT_INPUT
        )
        with _disable_current_modes():
            # Tensor subclasses and parameters are never capacity-padded rows.
            if (
                region.capacity is None
                or type(tensor) is not torch.Tensor
                or not tensor.is_cuda
            ):
                return tensor.detach()
            key = _storage_key(tensor)
            if key in region.inputs:
                return tensor.detach()
            stash = region.stashes.get(key)
            if stash is None:
                dims = _storage_slot_dims(tensor, region.capacity)
                if dims is not None and worth_paging(key[1], dims[0]):
                    layout = dims
                elif (
                    tensor.dim() >= 1
                    and tensor.shape[0] == region.capacity
                    and worth_paging(key[1], 1)
                ):
                    layout = None  # padded, but not in rows the stash can hold
                else:
                    return tensor.detach()
                numel = key[1] // tensor.element_size()
                stash = _Stash(
                    flat=tensor.detach().as_strided((numel,), (1,), 0),
                    numel=numel,
                    dtype=tensor.dtype,
                    device=tensor.device,
                    version=tensor._version,
                    layout=layout,
                    description=(
                        f"{tensor.dtype} {tuple(tensor.shape)} "
                        f"stride={tensor.stride()}"
                    ),
                )
                region.stashes[key] = stash
            stash.backward_reads |= backward_reads
            return _StashedView(
                stash,
                tensor.dtype,
                tensor.size(),
                tensor.stride(),
                int(tensor.storage_offset()),
            )

    def _unpack(self, saved: Any) -> torch.Tensor:
        if not isinstance(saved, _StashedView):
            return saved
        with _disable_current_modes():
            stash = saved.stash
            if stash.state == "paged":
                flat = self._materialize(stash).view(-1)
            elif stash.state == "dropped":
                # Checkpointing's replay of the region hands this back as an
                # op's output, but every op that reads it replays from its
                # cache too. Only the layout has to match, which the replay's
                # Python may branch on.
                flat = torch.empty(stash.numel, dtype=stash.dtype, device=stash.device)
            else:
                assert stash.flat is not None
                flat = stash.flat
            if saved.dtype != flat.dtype:
                flat = flat.view(torch.uint8).view(saved.dtype)
            return flat.as_strided(saved.size, saved.stride, saved.offset)

    def _commit(self, region: _RegionForward, output: Any) -> bool:
        """Megatron's group commit: launch every stash of the region together,
        once the region has produced everything it saves. Returns whether there
        was any.

        A storage the region returns stays resident: its rows are the caller's
        tokens. One the enclosing checkpoint saved as its own input is
        dropped: it is the block's, not this region's. One backward reads is
        paged if the stash can hold it, and kept resident otherwise. From here on, the saves
        of a storage hold it; the region keeps none, so a reload is freed with
        the last save backward reads it for.
        """
        returned = {
            _storage_key(tensor)
            for tensor in pytree.tree_leaves(output)
            if type(tensor) is torch.Tensor
        }
        stashes = []
        for key, stash in region.stashes.items():
            if key in returned:
                stash.state = "resident"
            elif not stash.backward_reads:
                stash.state = "dropped"
                stash.flat = None
            elif stash.layout is None:
                stash.state = "resident"
                if self._measuring:
                    self._padded_unpaged.setdefault(region.fqn, set()).add(
                        stash.description
                    )
            else:
                stash.state = "paged"
                stashes.append(stash)
        region.stashes.clear()
        if not stashes:
            return False
        manager = self.manager
        assert region.num_tokens is not None and region.capacity is not None
        validate = self._validating and not torch.cuda.is_current_stream_capturing()
        with _disable_current_modes():
            for ordinal, stash in enumerate(stashes):
                assert stash.flat is not None and stash.layout is not None
                if stash.flat._version != stash.version:
                    raise RuntimeError(
                        f"Paged stash: a tensor {region.fqn} saved for backward was "
                        "modified in place before its stash, so the stash would "
                        "not hold what backward expects."
                    )
                token_divisor, hidden = stash.layout
                rows = stash.flat.view(region.capacity // token_divisor, hidden)
                slot = self._slot_for(region, ordinal, stash, rows.dtype)
                handle = torch.empty(
                    _num_pages(slot.max_stash_tokens, manager.page_size) + 1,
                    dtype=torch.int64,
                    device=rows.device,
                )
                if validate:
                    stash.checksum = _live_checksum(
                        rows, _num_live_rows(region.num_tokens, token_divisor)
                    )
                last = ordinal == len(stashes) - 1
                stash.record = manager.stash(
                    slot,
                    rows,
                    region.num_tokens,
                    handle=handle,
                    prefetch=self.config.prefetch_n_layers if last else 0,
                )
                stash.flat = None
                stash.name = f"{slot.node_name} (microbatch {region.microbatch})"
                self._unread.add(stash.name)
                self._stashed_bytes += rows.numel() * rows.element_size()
        self._stashes += len(stashes)
        self._events.append(region.entry(forward=True))
        return True

    def _slot_for(
        self,
        region: _RegionForward,
        ordinal: int,
        stash: _Stash,
        dtype: torch.dtype,
    ) -> StashSlot:
        """The slot of the ``ordinal``-th storage ``region``'s layer stashes,
        the same for every microbatch and step."""
        assert region.capacity is not None and stash.layout is not None
        token_divisor, hidden = stash.layout
        key = (region.stage_index, region.layer_id, ordinal)
        dims = (dtype, hidden, region.capacity, token_divisor)
        slot = self._slots.get(key)
        if slot is None:
            slot = StashSlot(
                slot_id=self.manager.next_slot_id(),
                dtype=dtype,
                hidden_size=hidden,
                max_num_tokens=region.capacity,
                token_divisor=token_divisor,
                layer_id=region.layer_id,
                node_name=f"{region.fqn}[{ordinal}]",
                stage_index=region.stage_index,
            )
            self.manager.register_slot(slot)
            self._slots[key] = slot
        elif (
            slot.dtype,
            slot.hidden_size,
            slot.max_num_tokens,
            slot.token_divisor,
        ) != dims:
            raise RuntimeError(
                f"Paged stash: {region.fqn} saved a different set of padded "
                f"tensors than before: its stash {ordinal} was {slot}, now "
                f"{dims}. Paged stashing needs the same saves every microbatch."
            )
        return slot

    def _materialize(self, stash: _Stash) -> torch.Tensor:
        record = stash.record
        assert record is not None and stash.layout is not None
        if not stash.taken:
            self.manager.take_record(record)
            stash.taken = True
            self._unread.discard(stash.name)
        rows = self.manager.materialize(record)
        wait_for_copy(rows)
        if stash.checksum is not None:
            expected = int(stash.checksum.item())
            num_live_rows = _num_live_rows(record.num_tokens, stash.layout[0])
            actual = int(_live_checksum(rows, num_live_rows).item())
            stash.checksum = None
            # An overflowed stash reloads garbage too, and is the runner's to
            # handle.
            overflow = self.manager.overflow
            if actual == expected:
                self._reloads_checked += 1
            elif overflow is None or not overflow.item():
                raise RuntimeError(
                    f"Paged stash: the reload of {record.slot.node_name} "
                    f"({record.dtype} {record.shape}, {stash.layout[0]} "
                    "token(s) per row) does not match what was stashed. Its "
                    "live tokens are not a prefix of its rows, so it cannot be "
                    "paged."
                )
        return rows

    def _watch_backward(
        self,
        region: _RegionForward,
        output: torch.Tensor,
        input_alias: torch.Tensor | None,
    ) -> None:
        if not output.requires_grad:
            return
        output.register_hook(lambda grad: self._backward_start(region))
        if input_alias is not None:
            input_alias.register_hook(lambda grad: self._backward_end(region))

    def _backward_start(self, region: _RegionForward) -> None:
        """Megatron's ``PipelinePostScheduleFunction.backward``."""
        with _disable_current_modes():
            self.manager.join_stashes()
        self._events.append(region.entry(forward=False))

    def _backward_end(self, region: _RegionForward) -> None:
        """Megatron's ``PipelinePreScheduleFunction.backward``: reload what the
        next backward reads."""
        with _disable_current_modes():
            self.manager.prefetch_after(
                region.entry(forward=False), self.config.prefetch_n_layers
            )


__all__ = ["EagerPagedStash"]

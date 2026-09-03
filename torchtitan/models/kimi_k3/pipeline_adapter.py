# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cross-stage caching adapter and ``pipelining_fn`` for AttnRes.

* A block residual is defined over the whole layer stack, so under PP it travels
  between stages as a second payload next to the hidden states.
* :class:`CrossStageCacheAdapter` (delta mode, opt-in) ships per hop only the
  blocks the receiver does not already hold; the receiver rebuilds the stack
  from its cached prefix plus the delta.
* The block stack is a live autograd path, not a cache -- gradients cross stage
  boundaries through it.
"""

from __future__ import annotations

import dataclasses

import inspect
import math
import threading
import warnings
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn

from torch.distributed.pipelining.schedules import (
    _PipelineSchedule,
    PipelineScheduleMulti,
    PipelineScheduleSingle,
)

# Resolve Interleaved1F1B at import time so the schedule guard is a direct
# isinstance check instead of a string-match.
try:
    from torch.distributed.pipelining.schedules import get_schedule_class

    _INTERLEAVED_1F1B_CLASS = get_schedule_class("Interleaved1F1B")
except Exception:  # pragma: no cover - fallback for older torch
    _INTERLEAVED_1F1B_CLASS = None

from torchtitan.distributed.pipeline_parallel import (
    _generate_llm_fqn_per_model_part,
    get_schedule_class as _tt_get_schedule_class,
    pipeline_llm,
)
from torchtitan.models.kimi_k3.layout import (
    BlockLayoutTables,
    infer_block_layout_tables_from_stages,
    unstack_blocks,
)
from torchtitan.tools.logging import logger


# ----- Topology knobs, resolved once from config --------------------------- #
# These decide the pipeline topology, so every rank must resolve them
# identically; a per-rank disagreement hangs a collective with nothing
# pointing at the cause. They are read from call sites deep in the split
# where no config is in scope, so the resolved record is module-global:
# registered once at the pipelining entry, read back below.


@dataclass
class _TopologyKnobs:
    """Resolved topology."""

    attn_res_cache: bool = False


_TOPOLOGY: _TopologyKnobs | None = None
_WARNED_UNREGISTERED = False


def _register_topology(*, attn_res_cache: bool) -> _TopologyKnobs:
    """Resolve the topology once. Idempotent, first call wins."""
    global _TOPOLOGY

    resolved = _TopologyKnobs(attn_res_cache=bool(attn_res_cache))
    if _TOPOLOGY is not None and _TOPOLOGY != resolved:
        logger.warning(
            "topology re-registered with a different resolution: keeping %r, "
            "ignoring %r. Two entry points were handed different configs.",
            _TOPOLOGY,
            resolved,
        )
        return _TOPOLOGY
    _TOPOLOGY = resolved
    return _TOPOLOGY


def _topology() -> _TopologyKnobs:
    """The resolved topology, or the defaults with a warning."""
    global _WARNED_UNREGISTERED

    if _TOPOLOGY is not None:
        return _TOPOLOGY
    if not _WARNED_UNREGISTERED:
        _WARNED_UNREGISTERED = True
        logger.warning(
            "topology knob read before register_topology(); using defaults. "
            "Config fields are NOT being honoured on this path."
        )
    return _TopologyKnobs()


def _reset_topology_for_testing() -> None:
    """Tests need to re-resolve; production code must not call this."""
    global _TOPOLOGY, _WARNED_UNREGISTERED

    _TOPOLOGY = None
    _WARNED_UNREGISTERED = False


def adapter_enabled() -> bool:
    """Gate for the delta-mode block transport.

    Engaging it changes the order the block grads are summed, so it is not
    bitwise against the naive transport. It requires Interleaved1F1B (a rank
    with one stage has nothing to reuse); plain 1F1B runs the naive transport.
    """
    return _topology().attn_res_cache


# ----- Rank-shared cache across virtual stages ----------------------------- #


class RankLocalCache:
    """Per-rank, per-microbatch forward-block cache shared across VP stages.

    Every adapter on the same physical rank reads/writes the SAME cache
    -- one rank, one cache, any number of virtual stages. Holds only the cached
    block tensors (autograd-live against their original source) and
    producer metadata for layout bookkeeping.

    Grad-send-back has no state here: backward rides the autograd graph
    via PP's built-in SEND_B, so there's nothing for this cache to
    track on the backward path.
    """

    def __init__(self) -> None:
        self._blocks: dict[int, list[torch.Tensor]] = {}
        self._producer_meta: dict[int, list[tuple[int, int, int]]] = {}
        # Every backward marks its mb here so the step-end drop sweep
        # on the last virtual stage knows which mbs to evict.
        self._seen_mbs: set[int] = set()
        # Captured grads for the local-only Capture/Augment bridge, keyed by
        # (mb_index, producer_stage_id, block_idx): consumer-side
        # Capture.backward accumulates here; producer-side Augment.backward
        # pops and sums it into its incoming grad.
        self._captured_grads: dict[tuple[int, int, int], torch.Tensor] = {}
        # How many Capture.backward calls deposited into each slot; the
        # producer-side hook compares it to expected_same_rank_captures to turn
        # silent grad loss (a consumer backward never fired) into an error.
        self._capture_counts: dict[tuple[int, int, int], int] = {}
        # Commits whose producer installed no augment hook (no gradient path through
        # them). A consumer must not deposit into those slots; see mark_no_hook.
        self._no_hook: set[tuple[int, int, int]] = set()

    def append(
        self,
        mb_index: int,
        block: torch.Tensor,
        meta: tuple[int, int, int],
    ) -> None:
        self._blocks.setdefault(mb_index, []).append(block)
        self._producer_meta.setdefault(mb_index, []).append(meta)

    def get_blocks(self, mb_index: int) -> list[torch.Tensor]:
        return self._blocks.get(mb_index, [])

    def get_meta(self, mb_index: int) -> list[tuple[int, int, int]]:
        return self._producer_meta.get(mb_index, [])

    def put_forward(
        self,
        mb_index: int,
        blocks: list[torch.Tensor],
        producer_meta: list[tuple[int, int, int]] | None = None,
    ) -> None:
        """Back-compat shim used by unit tests: overwrite the per-mb list."""
        self._blocks[mb_index] = list(blocks)
        if producer_meta is not None:
            self._producer_meta[mb_index] = list(producer_meta)

    def drop(self, mb_index: int) -> None:
        self._blocks.pop(mb_index, None)
        self._producer_meta.pop(mb_index, None)
        self._seen_mbs.discard(mb_index)
        # Drop any leftover captured-grad slots + counters for this mb
        # (defensive; the on_microbatch_end assertion should have already
        # caught any real leak).
        for key in list(self._captured_grads.keys()):
            if key[0] == mb_index:
                self._captured_grads.pop(key, None)
        for key in list(self._capture_counts.keys()):
            if key[0] == mb_index:
                self._capture_counts.pop(key, None)
        for key in list(self._no_hook):
            if key[0] == mb_index:
                self._no_hook.discard(key)

    # ----- captured-grad slot helpers -------------------------------- #

    def capture_grad(
        self,
        key: tuple[int, int, int],
        grad: torch.Tensor,
    ) -> None:
        """Accumulate (sum) ``grad`` into the captured-grad slot at
        ``key`` and bump the capture counter. Multiple consumer-side
        Captures for the same producer block (V>2, one cached block
        read by >1 later virtual stage on the same rank) sum into the
        same slot.

        The first deposit is ``detach().clone()``-ed so the slot never aliases
        storage the autograd engine or FSDP2 may reuse.
        """
        prior = self._captured_grads.get(key)
        if prior is None:
            self._captured_grads[key] = grad.detach().clone()
        else:
            # `prior` is already our own detached clone, so out-of-place
            # addition keeps the semantics clean without aliasing.
            self._captured_grads[key] = prior + grad
        self._capture_counts[key] = self._capture_counts.get(key, 0) + 1

    def pop_grad(
        self,
        key: tuple[int, int, int],
    ) -> tuple[torch.Tensor | None, int]:
        """Return-and-clear ``(captured_grad, capture_count)`` for
        ``key``. ``captured_grad`` is ``None`` / ``capture_count`` is
        ``0`` when no consumer deposited into this slot during the
        current mb's backward window.

        The producer-side hook uses ``capture_count`` to validate
        against the static expectation from
        :meth:`BlockLayoutTables.expected_same_rank_captures` -- any
        mismatch means a producer's forward graph never saw a
        consumer's backward, which would silently drop grad.
        """
        grad = self._captured_grads.pop(key, None)
        count = self._capture_counts.pop(key, 0)
        return grad, count

    def mark_no_hook(self, key: tuple[int, int, int]) -> None:
        """Record that this commit has NO producer-side augment hook.

        A consumer must then leave the cached block alone: cache entries are
        stored detached, so ``blk.requires_grad`` cannot tell a consumer whether
        the producer had a gradient path, and a deposit with no hook to pop it
        would be a lost gradient. Both sides consult this record instead.
        """
        self._no_hook.add(key)

    def has_augment_hook(self, key: tuple[int, int, int]) -> bool:
        return key not in self._no_hook

    def clear_capture_slots(self) -> int:
        """Drop every captured-grad slot and counter. Returns how many there were.

        For the step-end sweep: an aborted backward can leave a slot behind that
        no mb-keyed ``drop`` reaches.
        """
        count = len(self._captured_grads)
        self._captured_grads.clear()
        self._capture_counts.clear()
        return count

    def has_captured_for_mb(self, mb_index: int) -> bool:
        """True iff any captured-grad slot for ``mb_index`` survives.
        Called by the mb-end assertion as a lingering-bug canary.
        """
        return any(k[0] == mb_index for k in self._captured_grads)


# One RankLocalCache per pipeline-group rank, shared by every adapter
# on that rank. Lock-protected against concurrent construction.
_rank_caches: dict[int, RankLocalCache] = {}
_rank_caches_lock = threading.Lock()


def _get_or_create_rank_cache(pp_rank: int) -> RankLocalCache:
    """Return (creating if absent) the shared cache for ``pp_rank``."""
    cache = _rank_caches.get(pp_rank)
    if cache is not None:
        return cache
    with _rank_caches_lock:
        cache = _rank_caches.get(pp_rank)
        if cache is None:
            cache = RankLocalCache()
            _rank_caches[pp_rank] = cache
        return cache


def _reset_rank_caches_for_testing() -> None:
    """Clear the module-level registry. Unit-test isolation only."""
    with _rank_caches_lock:
        _rank_caches.clear()


# ----- Local-only grad bridge for own-rank cached commits ------------------ #
#
# * Same-rank reuse must not let a consumer backward walk into the producer
#   graph: that frees it, and the producer backward then fails with "backward
#   through the graph a second time".
# * So the cache stores a DETACHED copy; the consumer wraps it in a Capture that
#   deposits the grad in a rank-local slot, and a producer-side hook sums the
#   slot into the producer incoming grad. No collectives on this path.
# * Recv-originated blocks stay attached on purpose: PP built-in backward P2P
#   already carries their gradient to the producing rank.


def _install_augment_hook(
    block_tensor: torch.Tensor,
    slot_key: tuple[int, int, int],
    rank_cache: "RankLocalCache",
    *,
    expected_captures: int | None = None,
) -> bool:
    """Sum later virtual stages' captured block grads into the producer's incoming grad.

    Raises when the observed capture count diverges from what the layout tables predict:
    a silently missing capture is a lost gradient that no loss curve shows.
    """
    if not block_tensor.requires_grad:
        return False

    def _hook(grad: torch.Tensor) -> torch.Tensor:
        captured, count = rank_cache.pop_grad(slot_key)
        if expected_captures is not None and count != expected_captures:
            mb_index, producer_stage, block_idx_in_producer = slot_key
            raise RuntimeError(
                f"AttnRes adapter: capture-count mismatch at slot {slot_key} "
                f"(mb={mb_index}, producer stage={producer_stage}, commit "
                f"{block_idx_in_producer} of that stage): observed {count} "
                f"deposits, the layout expected {expected_captures}. Fewer "
                "means a same-rank consumer's backward did not fire and its "
                "grad contribution is lost; more means a consumer deposited "
                "into the wrong slot. Either way this micro-batch's gradient "
                "for that block is wrong, so the step is refused rather than "
                "taken."
            )
        if captured is None:
            return grad
        return grad + captured

    block_tensor.register_hook(_hook)
    return True


class _LocalCacheCapture(torch.autograd.Function):
    """Identity forward; backward deposits grad in the slot and STOPS.

    The input tensor comes from the rank cache where it was stored in
    DETACHED form (see ``RankLocalCache.append``), so even if autograd
    were to attempt to traverse upstream from Capture's input, there
    is no upstream graph to walk. ``None`` for the tensor-input grad
    is belt-and-suspenders; detach is the primary guarantee.

    Multiple later virtual stages on this rank reading the same cached
    own-commit block each fire ``backward`` once per mb; each call sums
    into the slot via :meth:`RankLocalCache.capture_grad`.
    """

    @staticmethod
    def forward(ctx, block_tensor, slot_key, rank_cache):  # type: ignore[override]
        ctx.slot_key = slot_key
        ctx.rank_cache = rank_cache
        # Return a distinct Tensor wrapper so Function.apply builds a
        # fresh grad_fn node here. ``view(shape)`` is zero-copy.
        return block_tensor.view(block_tensor.shape)

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        ctx.rank_cache.capture_grad(ctx.slot_key, grad_out)
        return None, None, None


# ----- Microbatch-index threading ------------------------------------------ #

# Each adapter stashes its *current* mb index under its own object id.
# Forward and backward of a single mb run on the same thread.
_mb_state = threading.local()


def _current_mb_index(adapter_key: int) -> int | None:
    d = getattr(_mb_state, "indices", None)
    if not d:
        return None
    return d.get(adapter_key)


def _set_mb_index(adapter_key: int, mb_index: int | None) -> None:
    d = getattr(_mb_state, "indices", None)
    if d is None:
        d = {}
        _mb_state.indices = d
    if mb_index is None:
        d.pop(adapter_key, None)
    else:
        d[adapter_key] = mb_index


# ----- state_dict key rewriting -------------------------------------------- #
# The wrapped model lives under ``self.wrapped``; HF state_dict adapters key
# off raw names like ``tok_embeddings.weight``, so strip on save, re-prepend on load.

_WRAPPED_PREFIX = "wrapped."


def _strip_wrapped_prefix_hook(
    module: nn.Module, state_dict: dict, prefix: str, local_metadata: dict
) -> dict:
    """Save hook: drop the adapter's ``wrapped.`` namespace."""
    target = prefix + _WRAPPED_PREFIX
    rewrites = [k for k in state_dict if k.startswith(target)]
    for old_key in rewrites:
        new_key = prefix + old_key[len(target) :]
        state_dict[new_key] = state_dict.pop(old_key)
    return state_dict


def _prepend_wrapped_prefix_pre_hook(
    state_dict: dict,
    prefix: str,
    local_metadata: dict,
    strict: bool,
    missing_keys: list,
    unexpected_keys: list,
    error_msgs: list,
) -> None:
    """Load pre-hook: add the ``wrapped.`` namespace back."""
    target = prefix + _WRAPPED_PREFIX
    rewrites = [
        k for k in state_dict if k.startswith(prefix) and not k.startswith(target)
    ]
    for old_key in rewrites:
        inner = old_key[len(prefix) :]
        state_dict[target + inner] = state_dict.pop(old_key)


# ----- The adapter module -------------------------------------------------- #


class CrossStageCacheAdapter(nn.Module):
    """Wraps an ``AttnResModel`` stage with cross-stage caching.

    In delta mode (``layout_tables`` supplied) each forward pulls earlier
    blocks from the shared :class:`RankLocalCache`, receives the incoming
    delta, rebuilds the full block stack in block-index order, and lets
    backward flow through the autograd graph. Cached-prefix blocks are
    handled two ways depending on who committed them:

    * Different rank (producer_rank != self.pp_rank) -> cached block
      is a slice of an older ``recv_delta_tensor``. Passed through
      unwrapped; its grad flows back via that tensor and PP's built-in
      ``SEND_B`` drains it to the producer rank.
    * Same rank (producer_rank == self.pp_rank) -> cached block
      came from an earlier virtual stage on this rank and was stored
      DETACHED in the cache (no autograd link to the producer). At
      read time it is wrapped in :class:`_LocalCacheCapture`; Capture's
      backward deposits the grad in a rank-local slot. The matching
      producer-side hook installed by :func:`_install_augment_hook`
      pops the slot and SUMS the captured grad into the producer's
      incoming grad when the producer's own backward runs.

    Without layout tables the adapter is a naive passthrough.

    Adapters sharing a ``pp_rank`` share ONE :class:`RankLocalCache`.
    """

    def __init__(
        self,
        wrapped: nn.Module,
        *,
        stage_id: int,
        num_stages: int,
        group: "dist.ProcessGroup | None" = None,
        stage_to_rank: dict[int, int] | None = None,
        pp_rank: int | None = None,
        layout_tables: BlockLayoutTables | None = None,
    ) -> None:
        super().__init__()
        self.wrapped = wrapped
        self.stage_id = stage_id
        self.num_stages = num_stages
        self._group = group
        self._stage_to_rank = stage_to_rank or {i: i for i in range(num_stages)}
        if pp_rank is None:
            pp_rank = self._stage_to_rank.get(stage_id, stage_id)
        self.pp_rank = pp_rank
        self._cache = _get_or_create_rank_cache(self.pp_rank)
        self._layout = layout_tables
        # Delta mode is the layout tables' presence: the model returns the
        # carrier it was handed with this stage's commits appended, and
        # _finish_forward takes the tail past what went in.
        self._delta_mode = layout_tables is not None

        # Hide ``wrapped.`` from state_dict consumers.
        self._register_state_dict_hook(_strip_wrapped_prefix_hook)
        self._register_load_state_dict_pre_hook(
            _prepend_wrapped_prefix_pre_hook, with_module=False
        )

    # Torchtitan trainer iterates model_parts and calls init_weights /
    # init_states; __getattr__ delegates the rest.
    def init_weights(self, *args, **kwargs) -> None:
        self.wrapped.init_weights(*args, **kwargs)

    def init_states(self, *args, **kwargs) -> None:
        self.wrapped.init_states(*args, **kwargs)

    def __getattr__(self, name: str):
        """Fall back to the wrapped model for unknown attributes."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        wrapped = self.__dict__.get("_modules", {}).get("wrapped")
        if wrapped is None:
            raise AttributeError(
                f"'CrossStageCacheAdapter' object has no attribute '{name}' "
                "and wrapped model is not yet bound."
            )
        return getattr(wrapped, name)

    def _adapter_key(self) -> int:
        return id(self)

    def _current_mb(self) -> int:
        mb = _current_mb_index(self._adapter_key())
        assert mb is not None, (
            "CrossStageCacheAdapter.forward called without an mb index; "
            "stage.forward_one_chunk monkey-patch missing."
        )
        return mb

    @staticmethod
    def _has_blocks_signature(args) -> bool:
        """True if ``args[1]`` is the [T, N, D] block carrier (middle/last stage)."""
        return (
            len(args) >= 2 and isinstance(args[1], torch.Tensor) and args[1].dim() == 3
        )

    def _call_wrapped_naive(self, args, kwargs):
        """Dispatch to the wrapped model with the appropriate signature."""
        if self._has_blocks_signature(args):
            partial, new_blocks_tensor, *rest = args
            return self.wrapped(partial, new_blocks_tensor, *rest, **kwargs)
        return self.wrapped(*args, **kwargs)

    def forward(self, *args, **kwargs):
        """Dispatch to delta-P2P, shape inference, or naive passthrough."""
        # ``PipelineStage._shape_inference`` invokes ``self.submod(...)``
        # directly, bypassing the ``forward_one_chunk`` patch that stashes
        # the mb index. Route to the shape-inference helper in that case.
        if _current_mb_index(self._adapter_key()) is None:
            return self._forward_shape_inference(*args, **kwargs)
        if self._delta_mode:
            return self._forward_delta(*args, **kwargs)
        return self._call_wrapped_naive(args, kwargs)

    def _forward_shape_inference(self, *args, **kwargs):
        """Run wrapped model and reshape its blocks output to the delta
        size the runtime will emit; pipelining uses this return shape to
        size the next stage's recv buffer.
        """
        wrapped_out = self._call_wrapped_naive(args, kwargs)
        if not isinstance(wrapped_out, tuple):  # last stage
            return wrapped_out

        partial_out, new_blocks_out = wrapped_out
        if not self._delta_mode or self._layout is None:
            return partial_out, new_blocks_out

        expected_K = len(self._layout.delta_to_send(self.stage_id))
        if expected_K == new_blocks_out.shape[1]:
            return partial_out, new_blocks_out

        # The placeholder must match what the runtime emits, because the
        # downstream recv buffer is sized from it:
        # * shape [T, K, D], T the flattened batch-sequence -- derived from
        #   partial_out so it holds even for a stage that commits nothing;
        # * requires_grad mirroring the runtime delta, since pipelining derives
        #   the recv-buffer and grad-send metadata from these tensors and a
        #   False placeholder drops the delta backward edge.
        # The carrier's token axis is whatever the hidden state's is: this
        # model folds batch and sequence into one axis, so it is shape[0].
        # Multiplying the first two dimensions -- correct when the hidden state
        # carried a separate batch axis -- gives T * D here, and the recv buffer
        # sized from it then fails the residual's concatenation.
        num_tokens = partial_out.shape[0]
        return partial_out, partial_out.new_zeros(
            (num_tokens, expected_K, partial_out.shape[-1]),
            requires_grad=partial_out.requires_grad,
        )

    def _forward_delta(self, *args, **kwargs):
        """Interleaved1F1B delta forward.

        Cached-prefix blocks whose producer is on a DIFFERENT rank are
        passed through unwrapped: their autograd graph already goes
        back to the original ``recv_delta_tensor`` and PP's built-in
        ``SEND_B`` drains it to the producer rank. Cached-prefix
        blocks whose producer is ON THIS RANK (earlier virtual stage)
        are wrapped in :class:`_LocalCacheCapture` at read time,
        severing the consumer->producer autograd link and depositing
        their grad in a rank-local captured-grad slot for the matching
        producer-side :class:`_LocalCacheAugment` to re-inject during
        the producer's own backward pass.
        """
        mb = self._current_mb()
        layout = self._layout
        assert layout is not None, "_forward_delta called without layout tables"

        if self.stage_id == 0:
            partial_out, new_blocks_tensor = self.wrapped(*args, **kwargs)
            return self._finish_forward(
                mb,
                partial_out,
                new_blocks_tensor,
                prev_recv_tensor=None,
                incoming_block_indices=[],
                carried_in=0,
            )

        if not self._has_blocks_signature(args):
            return self.wrapped(*args, **kwargs)
        partial, recv_delta_tensor, *rest = args

        # Unstack incoming delta; wire order MUST match sender's layout.
        incoming_block_indices = layout.delta_to_send(self.stage_id - 1)
        recv_list = unstack_blocks(recv_delta_tensor)
        assert len(recv_list) == len(incoming_block_indices), (
            f"Incoming delta size mismatch at stage {self.stage_id} mb {mb}: "
            f"expected {len(incoming_block_indices)}, got {len(recv_list)}."
        )

        # Pull earlier cached blocks from the rank cache. Recv-originated
        # entries stay attached so PP's SEND_B drains their grad; own-rank
        # commits were stored DETACHED and get requires_grad + a Capture whose
        # slot the producer-side Augment hook sums in (see the bridge above).
        earlier_blocks_raw = list(self._cache.get_blocks(mb))
        earlier_meta = list(self._cache.get_meta(mb))
        cached_indices = [layout.commits_at(meta[1])[meta[2]] for meta in earlier_meta]
        earlier_blocks: list[torch.Tensor] = []
        # Eval / no_grad path: skip the Capture wrapping. There is no backward
        # to capture, and requires_grad_(True) + autograd.Function.apply both
        # fail under torch.no_grad() (the Validator's pp_schedule.eval()).
        grad_active = torch.is_grad_enabled()
        for blk, meta in zip(earlier_blocks_raw, earlier_meta):
            producer_rank, producer_stage, block_idx_in_producer = meta
            slot_key = (mb, producer_stage, block_idx_in_producer)
            if (
                producer_rank == self.pp_rank
                and grad_active
                and self._cache.has_augment_hook(slot_key)
            ):
                if not blk.requires_grad:
                    blk.requires_grad_(True)
                earlier_blocks.append(
                    _LocalCacheCapture.apply(blk, slot_key, self._cache)
                )
            else:
                earlier_blocks.append(blk)

        # Rebuild the full blocks tensor in block-index order.
        pairs = list(zip(cached_indices, earlier_blocks)) + list(
            zip(incoming_block_indices, recv_list)
        )
        pairs.sort(key=lambda p: p[0])
        ordered_blocks = [p[1] for p in pairs]
        blocks_tensor = (
            torch.stack(ordered_blocks, dim=1) if ordered_blocks else recv_delta_tensor
        )

        wrapped_ret = self.wrapped(partial, blocks_tensor, *rest, **kwargs)

        if self.stage_id == self.num_stages - 1:
            # Last stage: keepalive keeps recv tensor on the autograd graph.
            return self._keepalive_touch(wrapped_ret, recv_delta_tensor)

        partial_out, new_blocks_tensor = wrapped_ret
        return self._finish_forward(
            mb,
            partial_out,
            new_blocks_tensor,
            prev_recv_tensor=recv_delta_tensor,
            incoming_block_indices=incoming_block_indices,
            carried_in=blocks_tensor.shape[1],
        )

    def _finish_forward(
        self,
        mb: int,
        partial_out: torch.Tensor,
        returned_blocks_tensor: torch.Tensor,
        *,
        prev_recv_tensor: torch.Tensor | None,
        incoming_block_indices: list[int],
        carried_in: int,
    ):
        """Common tail for first + middle stages: append relayed and
        committed blocks to the shared rank cache, then stack the
        outgoing delta.
        """
        layout = self._layout
        assert layout is not None
        my_commits = layout.commits_at(self.stage_id)
        # The model returns the carrier it was handed with this stage's own
        # commits appended, not the commits alone, so the new blocks are the
        # tail past what went in.
        new_blocks_tensor = returned_blocks_tensor[:, carried_in:]
        assert new_blocks_tensor.shape[1] == len(my_commits), (
            f"Wrapped model returned {new_blocks_tensor.shape[1]} new blocks "
            f"at stage {self.stage_id} (carrier {carried_in} -> "
            f"{returned_blocks_tensor.shape[1]}), expected {len(my_commits)}."
        )

        # Append relayed blocks so later virtual stages on this rank see them;
        # producer metadata comes from the static layout. Slices stay
        # autograd-live against prev_recv_tensor, so SEND_B drains their grads.
        if prev_recv_tensor is not None:
            recv_list = unstack_blocks(prev_recv_tensor)
            for bidx, blk in zip(incoming_block_indices, recv_list):
                producer_stage = layout.producer_stage_of_block(bidx)
                producer_rank = self._stage_to_rank.get(producer_stage, producer_stage)
                block_idx_in_producer = layout.commits_at(producer_stage).index(bidx)
                self._cache.append(
                    mb,
                    blk,
                    (producer_rank, producer_stage, block_idx_in_producer),
                )

        # Append own commits. Each new block gets a grad hook that sums in, at
        # THIS stage's backward, grads a later same-rank Capture deposited; the
        # outgoing delta uses the attached block. The RANK CACHE gets a DETACHED
        # copy so a consumer backward cannot free the producer graph early.
        new_blocks_list = unstack_blocks(new_blocks_tensor)
        for local_idx, blk in enumerate(new_blocks_list):
            slot_key = (mb, self.stage_id, local_idx)
            expected_captures = layout.expected_same_rank_captures(
                self.stage_id,
                local_idx,
            )
            if not _install_augment_hook(
                blk,
                slot_key,
                self._cache,
                expected_captures=expected_captures,
            ):
                # No gradient path through this commit, so no consumer may deposit into
                # its slot either. Keeps the two sides of the bridge in agreement.
                self._cache.mark_no_hook(slot_key)
            # Cache entry must be detached so same-rank consumers cannot
            # reach the producer's forward graph via autograd.
            self._cache.append(
                mb,
                blk.detach(),
                (self.pp_rank, self.stage_id, local_idx),
            )
        # `new_blocks_list` (attached) is used below for the outgoing
        # delta. Keep the name alias for readability vs. the previous
        # `wrapped_new_blocks` variable.
        attached_new_blocks = new_blocks_list

        # Build the outgoing delta from (cache + new) by canonical bidx.
        # cache_by_bidx reads the rank cache directly, so relayed blocks in the
        # delta keep their autograd link to prev_recv_tensor for grad routing.
        out_indices = layout.delta_to_send(self.stage_id)
        cache_by_bidx = {
            layout.commits_at(meta[1])[meta[2]]: blk
            for meta, blk in zip(self._cache.get_meta(mb), self._cache.get_blocks(mb))
        }
        new_by_bidx = {
            my_commits[i]: attached_new_blocks[i]
            for i in range(len(attached_new_blocks))
        }
        send_pieces: list[torch.Tensor] = []
        for bidx in out_indices:
            if bidx in new_by_bidx:
                send_pieces.append(new_by_bidx[bidx])
            elif bidx in cache_by_bidx:
                send_pieces.append(cache_by_bidx[bidx])
            else:
                raise RuntimeError(
                    f"Outgoing delta asks for block {bidx} at stage "
                    f"{self.stage_id} but it's neither cached nor committed."
                )

        out_blocks_tensor = (
            torch.stack(send_pieces, dim=1)
            if send_pieces
            else partial_out.new_zeros((partial_out.shape[0], 0, partial_out.shape[-1]))
        )
        partial_out = self._keepalive_touch(partial_out, prev_recv_tensor)
        return partial_out, out_blocks_tensor

    @staticmethod
    def _keepalive_touch(payload, prev_recv_tensor: torch.Tensor | None):
        """Ensure ``prev_recv_tensor`` is on the autograd graph that
        produces ``payload``. Preserves tuple returns. Launch-bound (measured),
        well under a percent of a stage forward at any production shape.
        """
        if prev_recv_tensor is None:
            return payload
        touch = 0.0 * prev_recv_tensor.sum()
        if isinstance(payload, tuple):
            head, *tail = payload
            return (head + touch, *tail)
        return payload + touch

    def _drop_all_cached_and_clear(self) -> None:
        """Drop every mb the cache saw during the step and clear the
        seen-set. Called by the step-end monkey-patch after every
        adapter on this rank has finished backward. Honors the VP
        drop-guard: only the LAST virtual stage on the rank evicts;
        earlier virtual stages no-op so the shared cache survives for
        them.
        """
        if self._delta_mode:
            pp_size = self._layout.P if self._layout is not None else self.num_stages
            if self.stage_id + pp_size < self.num_stages:
                return
        # Union, not just the seen-set: only backward marks an mb seen, so a
        # forward-only pass caches blocks nothing would announce for eviction.
        # Nothing in the cache outlives the step, so drop the keys present.
        for mb_index in set(self._cache._seen_mbs) | set(self._cache._blocks):
            self._cache.drop(mb_index)
        # Defensive: ensure the seen-set is clear even if drop() didn't
        # remove every entry.
        self._cache._seen_mbs.clear()
        # Runs from a ``finally``, so also after a mid-backward failure, where a
        # deposit can be stranded. Outside the exception path a residual slot
        # means an on_microbatch_end assertion did not run.
        leaked = self._cache.clear_capture_slots()
        if leaked:
            logger.warning(
                "cross-stage cache: cleared %d captured-grad slot(s) at step end "
                "on rank %s; expected zero unless the step raised mid-backward.",
                leaked,
                self.pp_rank,
            )

    def on_microbatch_end(self, mb_index: int) -> None:
        """Mark ``mb_index`` seen; eviction happens at step end.

        * In delta mode, also assert every Capture deposit for this mb was
          drained -- a surviving slot means a producer backward never ran.
        * Interleaved1F1B runs backward in reverse virtual-stage order, so the
          rank's EARLIEST virtual stage (``stage_id < pp_size``) is the last to
          get here for an mb, and the only safe point for that assertion.
        """
        self._cache._seen_mbs.add(mb_index)
        if self._delta_mode and self._layout is not None:
            pp_size = self._layout.P
            # Earliest virtual stage on this rank (stage_id < pp_size): its
            # backward fires LAST among the rank's virtual stages for this mb,
            # so every slot for the mb should have been popped by an Augment.
            if self.stage_id < pp_size:
                assert not self._cache.has_captured_for_mb(mb_index), (
                    f"Captured grad slot for mb {mb_index} survived past "
                    f"stage {self.stage_id}'s backward on rank {self.pp_rank}; "
                    "producer-side _LocalCacheAugment never fired. "
                    "This indicates a producer forward graph was never "
                    "backward-traversed for this mb."
                )

    def extra_repr(self) -> str:
        return f"stage_id={self.stage_id}, num_stages={self.num_stages}"


# ----- Stage iteration + monkey-patching ----------------------------------- #


def _iter_schedule_stages(schedule: _PipelineSchedule):
    """Yield the ``PipelineStage`` objects a schedule holds."""
    if isinstance(schedule, PipelineScheduleSingle):
        yield schedule._stage
    elif isinstance(schedule, PipelineScheduleMulti):
        yield from schedule._stages
    else:
        raise RuntimeError(
            f"Unexpected pipeline schedule class {type(schedule).__name__}; "
            "extend _iter_schedule_stages."
        )


def _install_mb_index_patch(stage, adapter: CrossStageCacheAdapter) -> None:
    """Patch ``forward_one_chunk`` / ``backward_one_chunk`` to stash the
    schedule-owned mb index for the adapter. Per-(stage, adapter) via
    closure so multi-stage ranks (VP) demux correctly.

    Backward is a plain call: no retain_graph override, no custom
    transport. The cached-prefix autograd graph + PP's built-in SEND_B
    route all cross-rank grads; the adapter only needs the mb index
    threaded through forward + on_microbatch_end.
    """
    adapter_key = id(adapter)
    orig_fwd = stage.forward_one_chunk
    orig_bwd = stage.backward_one_chunk

    # ``save_forward_output`` was added to ``_PipelineStageBase.forward_one_chunk``
    # in torch nightly (>=2.10). On torch 2.9 stable the kwarg doesn't
    # exist, so passing it raises TypeError. Detect once and dispatch.
    _orig_fwd_sig = inspect.signature(orig_fwd)
    _has_save_kw = "save_forward_output" in _orig_fwd_sig.parameters

    def patched_fwd(fwd_chunk_id, args, kwargs=None, save_forward_output=True):
        _set_mb_index(adapter_key, fwd_chunk_id)
        try:
            if _has_save_kw:
                return orig_fwd(
                    fwd_chunk_id,
                    args,
                    kwargs,
                    save_forward_output=save_forward_output,
                )
            return orig_fwd(fwd_chunk_id, args, kwargs)
        finally:
            _set_mb_index(adapter_key, None)

    def patched_bwd(
        bwd_chunk_id,
        loss=None,
        full_backward: bool = True,
        last_backward: bool = False,
    ):
        # Plain backward: the Capture/hook bridge keeps each stage's forward
        # graph traversed exactly once per mb, the naive-PP baseline.
        _set_mb_index(adapter_key, bwd_chunk_id)
        try:
            return orig_bwd(
                bwd_chunk_id,
                loss=loss,
                full_backward=full_backward,
                last_backward=last_backward,
            )
        finally:
            # Mark the mb as seen so the step-end drop sweep evicts it.
            # We don't drop here: the shared rank cache is still live
            # for peers / later virtual stages.
            adapter.on_microbatch_end(bwd_chunk_id)
            _set_mb_index(adapter_key, None)

    stage.forward_one_chunk = patched_fwd
    stage.backward_one_chunk = patched_bwd


def _install_step_drop_patch(
    pp_schedule: _PipelineSchedule, adapters: list[CrossStageCacheAdapter]
) -> None:
    """Wrap ``pp_schedule.step`` so every registered adapter on this
    rank evicts its seen mbs from the shared cache EXACTLY ONCE after
    ``orig_step`` returns. The VP drop-guard inside
    :meth:`_drop_all_cached_and_clear` ensures only the last virtual
    stage on the rank actually frees memory.
    """
    orig_step = pp_schedule.step

    def patched_step(*args, **kwargs):
        try:
            return orig_step(*args, **kwargs)
        finally:
            for adapter in adapters:
                try:
                    adapter._drop_all_cached_and_clear()
                except Exception:
                    # Continue so one poisoned adapter cannot keep the others
                    # from clearing -- but say so: swallowed, a cache that
                    # stopped evicting is a slow leak with no symptom until OOM.
                    logger.warning(
                        "cross-stage cache sweep failed for one adapter; "
                        "continuing with the rest",
                        exc_info=True,
                    )

    pp_schedule.step = patched_step  # type: ignore[method-assign]


# ----- FQN split and DEP wiring -------------------------------------------- #

# Injected into the last PP stage when AttnRes is enabled.
_KIMI_ATTN_RES_LAST_STAGE_FQNS = ("output_res_proj", "output_res_norm")


def kimi_k3_module_fqns_per_model_part(
    model: nn.Module,
    *,
    model_config,
    parallelism,
    pp: int,
) -> list[list[str]] | None:
    """The pipeline split of a Kimi K3 model, built from its config.

    Core's layer distribution (``_generate_llm_fqn_per_model_part``) places the
    embedding, the layers and the head; on top of it this model needs the
    AttnRes aggregation modules (``output_res_proj``, ``output_res_norm``) on
    the stage that holds ``lm_head``, since the final block attention runs
    there, and the vision tower on the stage that holds the embedding, since
    vision features are spliced into the embeddings and nothing vision-side
    crosses a stage boundary. Returns None when the split does not apply (no
    pipeline parallelism, or a config without layers); the caller keeps
    whatever split the user configured.
    """
    if pp <= 1 or model_config is None:
        return None
    layers = getattr(model_config, "layers", None)
    if layers is None:
        return None
    num_layers = len(layers)
    input_weight = parallelism.pipeline_parallel_first_stage_less_layers
    output_weight = parallelism.pipeline_parallel_last_stage_less_layers
    layers_per_stage = parallelism.pipeline_parallel_layers_per_stage
    if layers_per_stage is not None:
        num_virtual_stages = math.ceil(
            (num_layers + input_weight + output_weight) / layers_per_stage
        )
    else:
        schedule_class = _tt_get_schedule_class(parallelism.pipeline_parallel_schedule)
        stages_per_rank = 1 if issubclass(schedule_class, PipelineScheduleSingle) else 2
        num_virtual_stages = pp * stages_per_rank
    fqns = _generate_llm_fqn_per_model_part(
        num_virtual_stages, num_layers, input_weight, output_weight
    )
    # Core spells the head ``output``; this model calls it ``lm_head``. Any
    # FQN matching no child makes core set that child to None on every stage.
    fqns = [["lm_head" if n == "output" else n for n in stage] for stage in fqns]
    tail = [n for n in _KIMI_ATTN_RES_LAST_STAGE_FQNS if hasattr(model, n)]
    fqns[-1].extend(tail)
    if getattr(model, "vision_encoder", None) is not None:
        embed_stage = next(
            (stage for stage in fqns if "tok_embeddings" in stage), fqns[0]
        )
        embed_stage.append("vision_encoder")
    return fqns


def pipeline_kimi_k3(model: nn.Module, *, attn_res_cache: bool = True, **kwargs):
    """``pipelining_fn`` for Kimi K3.

    Behavior:

    * Always: split the model with this model's names and the AttnRes
      aggregation modules on the last stage, then delegate to core
      ``pipeline_llm``.
    * When ``attn_res_cache`` is set AND the schedule is Interleaved1F1B: wrap
      each stage's ``submod`` in :class:`CrossStageCacheAdapter`, which ships
      only the blocks the receiver does not hold on each hop.
    * Otherwise: pass through (plain PP, the whole carrier on every hop).

    ``attn_res_cache`` is a property of the pipeline transport, not of the
    model, so it is an argument here; a recipe turns it off with
    ``functools.partial(pipeline_kimi_k3, attn_res_cache=False)`` as the
    ``pipelining_fn``. It changes the order the block gradients are summed, so
    the two transports are not bitwise against each other. Every rank must
    resolve it identically: a rank without the adapter while its peers have
    one hangs the first cross-stage hop with nothing pointing at the cause.
    """
    # Resolve the topology knobs ONCE. This entry can run before parallelize,
    # so whichever comes first registers; register_topology is idempotent and
    # reports a disagreement rather than letting order decide.
    _register_topology(attn_res_cache=attn_res_cache)

    parallelism = kwargs["parallelism"]
    if parallelism.module_fqns_per_model_part is None:
        fqns = kimi_k3_module_fqns_per_model_part(
            model,
            model_config=kwargs.get("model_config"),
            parallelism=parallelism,
            pp=kwargs["parallel_dims"].pp,
        )
        if fqns is not None:
            kwargs["parallelism"] = dataclasses.replace(
                parallelism, module_fqns_per_model_part=fqns
            )
    pp_schedule, model_parts, has_first_stage, has_last_stage = pipeline_llm(
        model, **kwargs
    )
    passthrough = (pp_schedule, model_parts, has_first_stage, has_last_stage)

    if not adapter_enabled():
        return passthrough

    if _INTERLEAVED_1F1B_CLASS is None or not isinstance(
        pp_schedule, _INTERLEAVED_1F1B_CLASS
    ):
        warnings.warn(
            "Kimi Linear cross-stage caching supports only Interleaved1F1B; "
            "running without the adapter."
        )
        return passthrough

    stages = list(_iter_schedule_stages(pp_schedule))
    parallel_dims = kwargs.get("parallel_dims")
    pp_size = parallel_dims.pp if parallel_dims is not None else len(stages)
    num_stages = pp_size * len(stages)
    stage_to_rank = {s: s % pp_size for s in range(num_stages)}

    # The block layout comes from the config, which is where it is defined:
    # a block opens every attn_res_block_size layers (see the residual's
    # opens_block test), so the layer count and that size give both numbers.
    # They used to be read as marker attributes off the stage's module, which
    # is how the reference tree distinguished its AttnRes model from its
    # baseline one; this tree has one model and attn_res_block_size is a
    # required config parameter, so the distinction -- and the silent
    # passthrough when the attributes were absent -- no longer applies.
    model_config = kwargs.get("model_config")
    layer_cfgs = getattr(model_config, "layers", None)
    if not layer_cfgs:
        warnings.warn(
            "Cannot determine the layer count from the model config; the "
            "cross-stage cache adapter falls back to passthrough."
        )
        return passthrough
    n_layers_total = len(layer_cfgs)
    layers_per_block = getattr(layer_cfgs[0], "attn_res_block_size", None)
    if not layers_per_block:
        warnings.warn(
            "Cannot determine attn_res_block_size from the model config; the "
            "cross-stage cache adapter falls back to passthrough."
        )
        return passthrough
    num_blocks = -(-n_layers_total // layers_per_block)

    try:
        layout_tables = infer_block_layout_tables_from_stages(
            stages,
            pp_size=pp_size,
            num_blocks=num_blocks,
            n_layers=n_layers_total,
            layers_per_block=layers_per_block,
        )
    except ValueError:
        # An unsupported configuration, not a rank-local mishap: falling back
        # leaves this rank without an adapter while its peers have one, and a
        # rank sending no delta hangs the first cross-stage hop.
        raise
    except Exception as e:  # pragma: no cover - defensive
        warnings.warn(
            f"Failed to build Kimi Linear block-layout tables ({e!r}); "
            "falling back to passthrough."
        )
        return passthrough

    installed_adapters: list[CrossStageCacheAdapter] = []
    for i, stage in enumerate(stages):
        adapter = CrossStageCacheAdapter(
            stage.submod,
            stage_id=stage.stage_index,
            num_stages=num_stages,
            group=getattr(stage, "group", None),
            stage_to_rank=stage_to_rank,
            pp_rank=getattr(stage, "group_rank", None),
            layout_tables=layout_tables,
        )
        stage.submod = adapter
        _install_mb_index_patch(stage, adapter)
        installed_adapters.append(adapter)
    # model_parts is deliberately left alone. The schedule runs what
    # stage.submod points at, which is the adapter; model_parts is what the
    # trainer reaches through for the real module -- it takes lm_head off the
    # last part and sets _skip_lm_head on it, builds the optimizer from it and
    # hands it to the checkpointer. Substituting the wrapper there sends
    # _skip_lm_head to the wrapper instead of the model, so the last stage keeps
    # applying lm_head and the loss applies it a second time; and it prefixes
    # every checkpoint key with the wrapper's attribute name.

    _install_step_drop_patch(pp_schedule, installed_adapters)

    # Say so on success, not only on fallback: the adapter is numerically
    # neutral by design, so without this line "wrapped" and "silently fell
    # back" are indistinguishable from the outside.
    logger.info(
        "cross-stage cache adapter wrapped %d stage(s): %s",
        len(installed_adapters),
        [s.stage_index for s in stages],
    )

    return pp_schedule, model_parts, has_first_stage, has_last_stage

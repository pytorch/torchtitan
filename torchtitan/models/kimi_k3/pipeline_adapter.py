# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Cross-stage caching adapter and ``pipelining_fn`` for AttnRes.

    :class:`CrossStageCacheAdapter` wraps a per-stage AttnRes decoder. In delta mode
    each hop ships only the blocks the receiver does not already hold; the receiver
    rebuilds the stack from its cached prefix plus the delta. The block stack is a
    live autograd path, not a cache -- gradients cross stage boundaries through it.

    See ``phase13_k3like_48b_posttrain/PP_ATTNRES_ADAPTER.md``.
    """

from __future__ import annotations

import math
import os
import threading
import warnings

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

from torchtitan.models.kimi_k3.attn_res import unstack_blocks
from torchtitan.models.kimi_k3.layout import (
    _infer_block_layout_tables_from_stages,
    BlockLayoutTables,
)
from torchtitan.tools.logging import logger


def adapter_enabled() -> bool:
    """Config gate for delta mode. Opt-in, and the trust now has a measurement.

    3000 steps per arm at pp2 x vp2, two commits per stage
    (matrix_scripts/run_delta_convergence.sh): naive 1.77437, delta 1.78017, and the
    SAME configuration reseeded 1.76570. The delta-vs-naive mean relative gap runs
    0.00245 over the first tenth to 0.00342 over the last; the reseed-vs-naive gap runs
    0.00465 to 0.00555. Neither grows, so it is not a divergence.

    But it is not noise either, and the distinction is the point. Delta's difference is
    DETERMINISTIC -- a summation-order difference from the mid-stage block-stack rebuild
    and from ``grad + captured`` where autograd would have accumulated -- so comparing it
    against a reseed spread bounds detectability, not bias.
    ``run_delta_sign_test.sh`` runs the pair across seeds and reads the sign of the tail
    difference, which is what separates a coincidence from a bias. At six seeds: +0.14%,
    +0.24%, +0.20%, -0.16%, +0.08%, +0.14% -- five positive, one negative, mean +0.107%.
    Five of six same-signed has probability 0.219 under no bias, so this is NOT a
    detectable systematic effect; the first three seeds all landing positive was the
    coincidence that 3/3 at p = 0.25 always risked being.

    The accurate statement is therefore narrow: the difference is DETERMINISTIC in
    mechanism -- same seed, same digits -- and unbiased in aggregate, with a magnitude
    (0.107% mean) well inside the 0.56% spread between two runs of one configuration. It
    behaves like noise without being noise, and neither transport is favoured.

    Still False by default, because engaging it needs more than trust: Interleaved1F1B
    (otherwise this returns and the adapter passes through), n_layers divisible by the
    stage count, and an even split (first/last_stage_less_layers 0). Flipping the default
    would leave most configurations on the passthrough they already take, while the ones
    that do qualify would change transport without a gate cell able to see it -- the
    58-cell gate never enters delta mode at all.
    """
    from torchtitan.models.kimi_k3.knobs import topology

    return topology().attn_res_cache


# ----- Rank-shared cache across virtual stages ----------------------------- #


class RankLocalCache:
    """Per-rank, per-microbatch forward-block cache shared across VP stages.

    Every adapter on the same physical rank reads/writes the SAME cache
    (Kimi §4.1 invariant). Holds only forward-path state: the cached
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
        # Captured grads for the local-only _LocalCacheAugment/Capture
        # dance. Keyed by (mb_index, producer_stage_id, block_idx). A
        # consumer-side Capture.backward accumulates grad here; the
        # producer-side Augment.backward pops and sums the captured
        # grad into its incoming grad when stage R's own backward runs.
        self._captured_grads: dict[tuple[int, int, int], torch.Tensor] = {}
        # Parallel counter: how many Capture.backward calls have deposited
        # into each slot. The producer-side hook compares this against
        # ``layout.expected_same_rank_captures(...)`` to turn silent grad
        # loss (a consumer's backward never fired) into a raised error.
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

        The first deposit is ``detach().clone()``-ed to decouple from
        whatever storage the autograd engine / FSDP2 post-backward
        pipeline hands us. The per-mb cost is O(Np*d) per consumer,
        which is insignificant next to the PP collective cost at
        realistic scale, and it removes a fragility: if a downstream
        framework were to reuse the grad tensor's storage, the slot
        value would silently corrupt.
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

        A consumer must then leave the cached block alone. The cache always stores a
        DETACHED copy, so ``blk.requires_grad`` is False for every entry and cannot tell
        a consumer whether the producer's own tensor had a gradient path -- which is why
        the consumer used to force ``requires_grad_(True)`` unconditionally. That made
        the two sides asymmetric: a deposit with no hook to pop it is a lost gradient
        caught only by the mb-end assertion. Measured as unreachable in training (the
        AttnRes graft projections are trainable even under LoRA, so the block always
        requires grad), so this records the invariant rather than fixing a live bug.
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
# A block committed by an earlier virtual stage and read back by a later one ON THE SAME
# RANK would otherwise have its forward graph freed by the consumer's backward, and the
# producer's own backward then fails with "backward through the graph a second time". A
# producer-side grad hook plus a consumer-side detached leaf sever that link structurally;
# both halves are rank-local, with no collectives. The detach is load-bearing, and
# recv-originated blocks are deliberately left attached so PP's SEND_B still carries their
# gradient to the producing rank.
#
# Full reasoning, including the two designs that did not hold, is in the PR-C body
# (Raising_PRs/k3_pr_c_pp_attnres/PR_BODY.md, "The local grad bridge").


_DBG = os.environ.get("ATTNRES_ADAPTER_DBG") == "1"


def _dbg(msg: str) -> None:
    if _DBG:
        rank = os.environ.get("RANK", "?")
        print(f"[adapter-dbg rank={rank}] {msg}", flush=True)


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

    See ``phase13_k3like_48b_posttrain/PP_AUGMENT_HOOK.md``.
    """
    if not block_tensor.requires_grad:
        return False

    def _hook(grad: torch.Tensor) -> torch.Tensor:
        captured, count = rank_cache.pop_grad(slot_key)
        _dbg(
            f"augment_hook slot={slot_key} "
            f"captured={'yes' if captured is not None else 'no'} "
            f"count={count} expected={expected_captures}"
        )
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
        _dbg(f"Capture.forward slot={slot_key}")
        # Return a distinct Tensor wrapper so Function.apply builds a
        # fresh grad_fn node here. ``view(shape)`` is zero-copy.
        return block_tensor.view(block_tensor.shape)

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        _dbg(f"Capture.backward slot={ctx.slot_key}")
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
# The adapter stores its wrapped model under ``self.wrapped``. The Llama3 HF
# state_dict_adapter keys off raw names like ``tok_embeddings.weight``, so
# we strip the prefix on save and re-prepend on load.

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

    * **Different rank** (producer_rank != self.pp_rank) → cached block
      is a slice of an older ``recv_delta_tensor``. Passed through
      unwrapped; its grad flows back via that tensor and PP's built-in
      ``SEND_B`` drains it to the producer rank.
    * **Same rank** (producer_rank == self.pp_rank) → cached block
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
        self._delta_mode = layout_tables is not None

        # Delta mode: wrapped returns only own commits. Naive mode: full stack.
        if hasattr(wrapped, "_return_only_new_blocks"):
            wrapped._return_only_new_blocks = bool(self._delta_mode)
        else:
            warnings.warn(
                "Wrapped model does not expose _return_only_new_blocks; "
                "adapter will run in naive (full-stack) mode.",
                stacklevel=2,
            )

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
            return self.wrapped(partial, *rest, blocks=new_blocks_tensor, **kwargs)
        return self.wrapped(*args, blocks=None, **kwargs)

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

        # The placeholder must have the shape the RUNTIME emits, because the
        # downstream recv buffer is sized from it. The runtime sends
        # ``torch.stack(send_pieces, dim=1)`` over ``[T, D]`` pieces, i.e.
        # ``[T, K, D]`` with T the flattened batch-sequence -- the carrier layout
        # stack_blocks documents. Deriving it from partial_out is what makes that
        # true for a stage that commits nothing, where new_blocks_out has no
        # column to read a per-block shape from.
        #
        # Both previous forms were wrong and both needed expected_K != N to show
        # it, which is why every delta run to date missed them: an empty commit
        # took partial_out.shape whole and produced a FOUR-dimensional
        # [K, B, L, D], and a non-empty one took new_blocks_out.shape[1:] and
        # produced [K, N, D] with the block axis first. The four-dimensional case
        # surfaced as a consumer failing _has_blocks_signature (which tests
        # dim() == 3) and then passing the carrier positionally into ``blocks``:
        # "got multiple values for argument 'blocks'", 32 layers at pp8 x vp2.
        tokens_times_batch = partial_out.shape[0] * partial_out.shape[1]
        # requires_grad must mirror the runtime delta emission: torch >= 2.12
        # derives the downstream recv-buffer and grad-send metadata from the
        # shape-inference tensors, and a requires_grad=False placeholder makes
        # the consumer stage drop the delta's backward edge (None grads at
        # SEND_B -> PipeliningMetadataError).
        return partial_out, partial_out.new_zeros(
            (tokens_times_batch, expected_K, partial_out.shape[-1]),
            requires_grad=partial_out.requires_grad,
        )

    def _forward_delta(self, *args, **kwargs):
        """Interleaved1F1B delta forward (spec §4.1).

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
            partial_out, new_blocks_tensor = self.wrapped(*args, blocks=None, **kwargs)
            return self._finish_forward(
                mb,
                partial_out,
                new_blocks_tensor,
                prev_recv_tensor=None,
                incoming_block_indices=[],
            )

        if not self._has_blocks_signature(args):
            return self.wrapped(*args, blocks=None, **kwargs)
        partial, recv_delta_tensor, *rest = args

        # Unstack incoming delta; wire order MUST match sender's layout.
        incoming_block_indices = layout.delta_to_send(self.stage_id - 1)
        recv_list = unstack_blocks(recv_delta_tensor)
        assert len(recv_list) == len(incoming_block_indices), (
            f"Incoming delta size mismatch at stage {self.stage_id} mb {mb}: "
            f"expected {len(incoming_block_indices)}, got {len(recv_list)}."
        )

        # Pull earlier cached blocks out of the rank cache. Recv-originated
        # entries were stored attached to their recv_delta_tensor, so leaving
        # them unwrapped lets PP's own SEND_B drain their grad to the producer
        # rank. Own-rank commits were stored DETACHED (see _finish_forward), so
        # they need requires_grad plus a _LocalCacheCapture wrapper: Capture
        # deposits the grad in a rank-local slot and the producer-side hook
        # from _install_augment_hook sums it in when the producer's backward
        # runs. Routing it through a slot rather than the graph is the point --
        # detached means autograd cannot walk into the producer and free its
        # saved tensors early.
        earlier_blocks_raw = list(self._cache.get_blocks(mb))
        earlier_meta = list(self._cache.get_meta(mb))
        cached_indices = [layout.commits_at(meta[1])[meta[2]] for meta in earlier_meta]
        earlier_blocks: list[torch.Tensor] = []
        # Eval / no_grad path: skip the Capture wrapping. With no
        # backward to run, there is nothing to capture into a slot, and
        # ``requires_grad_(True)`` + ``autograd.Function.apply`` both
        # fail under ``torch.no_grad()`` (which the torchtitan Validator
        # uses via ``pp_schedule.eval()``). Use the cached block tensors
        # raw — fwd math is identical.
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

        wrapped_ret = self.wrapped(partial, *rest, blocks=blocks_tensor, **kwargs)

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
        )

    def _finish_forward(
        self,
        mb: int,
        partial_out: torch.Tensor,
        new_blocks_tensor: torch.Tensor,
        *,
        prev_recv_tensor: torch.Tensor | None,
        incoming_block_indices: list[int],
    ):
        """Common tail for first + middle stages: append relayed and
        committed blocks to the shared rank cache, then stack the
        outgoing delta.
        """
        layout = self._layout
        assert layout is not None
        my_commits = layout.commits_at(self.stage_id)
        assert new_blocks_tensor.shape[1] == len(my_commits), (
            f"Wrapped model returned {new_blocks_tensor.shape[1]} new "
            f"blocks at stage {self.stage_id}, expected {len(my_commits)}."
        )

        # Append relayed blocks so later virtual stages on this rank see
        # them; producer metadata comes from the static layout. Slices
        # of ``prev_recv_tensor`` stay autograd-live against it, so
        # PP's SEND_B on backward will drain their grads upstream.
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

        # Append own commits. Each new block gets a grad hook that, during THIS
        # stage's backward, sums in any grad a later same-rank virtual stage's
        # _LocalCacheCapture deposited; the outgoing-delta path uses the
        # attached block, so the next stage's SEND_B reaches the same grad_fn.
        # The RANK CACHE gets a DETACHED copy: that severs it from the
        # producer's forward graph, so a later same-rank consumer's backward
        # physically cannot walk into the producer and free its saved tensors
        # early -- the double-backward crash the previous
        # _LocalCacheAugment.apply + view pattern hit under PP + FSDP + AC.
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

        # Build outgoing delta: subset of (cache + new), by canonical bidx.
        # ``cache_by_bidx`` reads from the rank cache directly so
        # relayed (recv-originated) blocks that show up in the outgoing
        # delta also route grad correctly via their existing autograd
        # link to ``prev_recv_tensor``.
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
            else partial_out.new_zeros(
                (partial_out.shape[0] * partial_out.shape[1], 0, partial_out.shape[-1])
            )
        )
        partial_out = self._keepalive_touch(partial_out, prev_recv_tensor)
        return partial_out, out_blocks_tensor

    @staticmethod
    def _keepalive_touch(payload, prev_recv_tensor: torch.Tensor | None):
        """Ensure ``prev_recv_tensor`` is on the autograd graph that
        produces ``payload``. Preserves tuple returns.

        Profiled rather than left as a suspicion (matrix_scripts/pp_cp_overheads.py):
        the touch costs 0.037 ms at 256x512, 4096x2048 and every shape between, rising
        to 0.18 ms only at 16384x2048. It is launch-bound, not arithmetic-bound, so the
        O(T*D) reduction it looks like is not what is being paid -- and a cheaper
        formulation that reads one element instead of reducing would save nothing.
        Against one projection it is 137% at 256x512 and 5.2% at 4096x2048; against a
        whole stage's forward, which runs dozens of projections, it is well under a
        percent at any production shape. Left as is.
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
        # Union, not just the seen-set: only backward marks an mb as seen, so a
        # forward-only pass (evaluation) caches blocks that nothing would ever
        # announce for eviction. Nothing in the cache outlives the step, so the
        # keys actually present are the right thing to drop.
        for mb_index in set(self._cache._seen_mbs) | set(self._cache._blocks):
            self._cache.drop(mb_index)
        # Defensive: ensure the seen-set is clear even if drop() didn't
        # remove every entry.
        self._cache._seen_mbs.clear()
        # The step-end patch calls this from a ``finally``, so it also runs when
        # the step raised. On that path a micro-batch's backward can stop between
        # a consumer's deposit and the producer's pop, and the mb-keyed drop above
        # only reaches slots whose mb still had cached blocks -- so clear the slot
        # tables outright. A grad tensor per slot is real memory, and a step that
        # dies in the backward of one micro-batch (OOM being the ordinary cause)
        # would otherwise accumulate them for as long as the process keeps
        # retrying. Reported rather than silent: outside the exception path a
        # residual slot means an on_microbatch_end assertion did not run.
        leaked = self._cache.clear_capture_slots()
        if leaked:
            logger.warning(
                "cross-stage cache: cleared %d captured-grad slot(s) at step end "
                "on rank %s; expected zero unless the step raised mid-backward.",
                leaked,
                self.pp_rank,
            )

    def on_microbatch_end(self, mb_index: int) -> None:
        """Mark ``mb_index`` as seen on this rank so the step-end sweep
        drops it. Actual eviction is deferred to ``pp_schedule.step``
        return; see :func:`_install_step_drop_patch`.

        In delta mode, this is also the moment to assert that every
        :class:`_LocalCacheCapture` deposit for this mb has been drained
        by a matching :class:`_LocalCacheAugment` -- a surviving slot
        would mean a producer's backward never ran, which is a bug.
        Interleaved1F1B runs backward in reverse virtual-stage order
        on each rank, so the EARLIEST virtual stage on this rank is
        the last to call on_microbatch_end for a given mb. That is
        the only point at which every producer-side Augment has had a
        chance to drain its slot, so we guard the assertion to fire
        there only (``stage_id < pp_size`` == "this rank's earliest
        virtual stage").
        """
        self._cache._seen_mbs.add(mb_index)
        if self._delta_mode and self._layout is not None:
            pp_size = self._layout.P
            # Earliest virtual stage on this rank: stage_id < pp_size.
            # Its backward fires LAST among the rank's virtual stages
            # for this mb, so by the time we reach here every slot
            # for this mb should have been popped by an Augment.
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
    import inspect as _inspect

    _orig_fwd_sig = _inspect.signature(orig_fwd)
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
        # Plain backward pass. The double-backward risk on own-rank
        # cached commits is now handled structurally by
        # :class:`_LocalCacheAugment` / :class:`_LocalCacheCapture`:
        # the consumer-side Capture severs the consumer->producer
        # autograd link (so the producer's graph is NOT traversed or
        # freed by this stage's backward), and the producer-side
        # Augment sums the captured grad into the producer's own
        # incoming grad when THE PRODUCER's backward runs. Each
        # stage's forward graph is thus traversed exactly once per mb,
        # which is the naive-PP baseline.
        _set_mb_index(adapter_key, bwd_chunk_id)
        _dbg(f"patched_bwd ENTER stage={adapter.stage_id} mb={bwd_chunk_id}")
        try:
            return orig_bwd(
                bwd_chunk_id,
                loss=loss,
                full_backward=full_backward,
                last_backward=last_backward,
            )
        finally:
            _dbg(f"patched_bwd EXIT stage={adapter.stage_id} mb={bwd_chunk_id}")
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
                    # from clearing -- but say so. Swallowing this silently
                    # turns a cache that stopped evicting into a slow memory
                    # leak with no symptom until OOM.
                    logger.warning(
                        "cross-stage cache sweep failed for one adapter; "
                        "continuing with the rest",
                        exc_info=True,
                    )

    pp_schedule.step = patched_step  # type: ignore[method-assign]


# ----- FQN-split injection ------------------------------------------------- #

_ATTN_RES_EXTRA_LAST_STAGE_FQNS = ("output_res_proj", "output_res_norm")


# ----- Custom pipelining_fn ------------------------------------------------ #


# ----- Kimi Linear / K3 pipelining wiring (merged from kimi_linear/) ----- #

# Kimi-specific FQNs injected into the last PP stage when AttnRes is enabled.
_KIMI_ATTN_RES_LAST_STAGE_FQNS = ("output_res_proj", "output_res_norm")


def _kimi_llm_fqns(
    num_stages: int,
    num_layers: int,
    input_weight: int = 1,
    output_weight: int = 1,
) -> list[list[str]]:
    """Kimi-named version of ``generate_llm_fqn_per_model_part``.

    Substitutes ``tok_embeddings``→``embed_tokens`` and
    ``output``→``lm_head``. Keeps the layer distribution logic
    (delegated to core's function, then re-mapped) so any future
    tweaks there apply to us automatically.
    """
    from torchtitan.distributed.pipeline_parallel import (
        _generate_llm_fqn_per_model_part as generate_llm_fqn_per_model_part,
    )

    raw = generate_llm_fqn_per_model_part(
        num_stages, num_layers, input_weight, output_weight
    )
    rename = {"tok_embeddings": "embed_tokens", "output": "lm_head"}
    return [[rename.get(n, n) for n in stage] for stage in raw]


def _unwrap_multimodal_for_pp(model: nn.Module, kwargs: dict) -> nn.Module:
    """Split the TEXT model, and re-wrap the stage that owns ``embed_tokens``.

    Core's ``_split_module`` iterates only top-level ``named_children()``. On the
    multimodal wrapper those children are ``vision_tower`` and
    ``language_model``, so no FQN scheme reaches the text stack: flat names
    (``embed_tokens``, ``layers.N``) match nothing, and dotted ones
    (``language_model.layers.N``) are not recursed into either. Every child then
    takes the "not in modules_to_keep" branch and is set to None, so the stage
    holds zero parameters and the optimizer reports
    ``pattern '.*' matched no parameters``.

    Vision features are spliced into the embeddings, so the tower belongs with
    whichever chunk kept ``embed_tokens`` -- nothing vision-side crosses a stage
    boundary. Re-wrapping happens inside ``parallelize_fn`` so the tower is
    present before SPMD is applied, not bolted on afterwards.

    Returns the module to hand to ``pipeline_llm``: the text model when this is
    the multimodal wrapper, otherwise ``model`` untouched.
    """
    tower = getattr(model, "vision_tower", None)
    inner = getattr(model, "language_model", None)
    if tower is None or inner is None:
        return model

    from torchtitan.models.kimi_k3.multimodal_model import KimiK3MultimodalModel

    mm_config = model.config
    inner_parallelize = kwargs["parallelize_fn"]

    step_inputs = None
    if dep_enabled():
        from torchtitan.models.kimi_k3.vit_prefetch import VisionStepInputs

        # Marker submodules so each vision chunk knows WHICH share it is. They must
        # really exist on the module being split: _split_module keeps children whose
        # name is in the chunk's FQN list and sets the rest to None, so a marker that
        # matched nothing (the previous scheme) leaves every share indistinguishable.
        # They hold no parameters, so they add nothing to any stage.
        for i in range(dep_vision_stages()):
            inner.add_module(f"{_DEP_VISION_FQN}{i}", nn.Module())
        step_inputs = VisionStepInputs()
        # Read back by the pipelining_fn, which needs to hook the schedule that does
        # not exist yet at this point.
        inner._dep_step_inputs_holder = step_inputs

    def _parallelize_with_tower(part: nn.Module, **pk):
        if dep_enabled():
            from torchtitan.models.kimi_k3.multimodal_model import KimiK3ViTStage

            # Which vision share is this? The marker submodule that survived
            # _split_module carries the index, so identity comes from the chunk
            # itself. Call order cannot be used: a rank holding several virtual
            # stages sees them in an order this function is not told. And "holds
            # embed_tokens and no layers" only identifies share 0 -- the later
            # shares hold neither, so they are indistinguishable from a text chunk
            # without the marker.
            share = _dep_vision_share_index(part)
            if share is not None:
                n_vit = dep_vision_stages()
                stage = KimiK3ViTStage.from_parts(mm_config, tower, part)
                if n_vit > 1:
                    bounds = stage.vision_tower.block_bounds(n_vit)
                    role = (
                        "head"
                        if share == 0
                        else ("tail" if share == n_vit - 1 else "body")
                    )
                    stage.set_dep_role(
                        role,
                        bounds=bounds[share],
                        num_shares=n_vit,
                        step_inputs=step_inputs,
                    )
                return inner_parallelize(stage, **pk)
            _register_mm_prefix_hooks(part)
            return inner_parallelize(part, **pk)

        # The embed_tokens-owning chunk is the first stage by construction.
        if getattr(part, "embed_tokens", None) is not None:
            part = KimiK3MultimodalModel.from_parts(mm_config, tower, part)
        else:
            _register_mm_prefix_hooks(part)
        return inner_parallelize(part, **pk)

    kwargs["parallelize_fn"] = _parallelize_with_tower
    return inner


def _install_vision_stage_wiring(pp_schedule, step_inputs) -> int:
    """Give every vision stage its micro-batch index, and the step its ``kwarg_mbs``.

    Required whenever the tower spans stages, not just for the run-ahead: a body or
    tail share reads ``grid_thw`` by micro-batch index, and without the index it takes
    the metadata-inference path -- passing activations through unprocessed, with no
    error. A silently unsplit tower and a silently un-spliced batch are exactly the
    failure shape to design out, so this is installed unconditionally under DEP and its
    count is logged.

    Returns how many stages were wired, so a caller can assert engagement instead of
    inferring it from numerics.
    """
    from torchtitan.models.kimi_k3.multimodal_model import KimiK3ViTStage
    from torchtitan.models.kimi_k3.vit_prefetch import install_step_hook

    if step_inputs is not None:
        install_step_hook(pp_schedule, step_inputs)

    wired = 0
    for stage in _iter_schedule_stages(pp_schedule):
        submod = getattr(stage, "submod", None)
        if not isinstance(submod, KimiK3ViTStage):
            continue
        orig_fwd = stage.forward_one_chunk

        def patched(fwd_chunk_id, args, kwargs=None, _f=orig_fwd, _m=submod, **kw):
            _m._dep_current_mb = fwd_chunk_id
            try:
                return _f(fwd_chunk_id, args, kwargs, **kw)
            finally:
                _m._dep_current_mb = None

        stage.forward_one_chunk = patched
        wired += 1

    if wired:
        logger.info(
            "DEP vision stage wiring: %d stage(s) on this rank, roles %s",
            wired,
            [
                getattr(s.submod, "_dep_role", "?")
                for s in _iter_schedule_stages(pp_schedule)
                if isinstance(getattr(s, "submod", None), KimiK3ViTStage)
            ],
        )
    return wired


def _dep_vision_share_index(part: nn.Module) -> int | None:
    """Which vision share ``part`` is, from the marker submodule that survived the split.

    Returns None for a text chunk. Reading identity off the chunk rather than off call
    order matters because a rank holding several virtual stages receives them in an
    order ``parallelize_fn`` is not told.
    """
    for name, child in part.named_children():
        if child is not None and name.startswith(_DEP_VISION_FQN):
            try:
                return int(name[len(_DEP_VISION_FQN) :])
            except ValueError:
                continue
    return None


_MM_INNER_PREFIX = "language_model."


def _register_mm_prefix_hooks(part: nn.Module) -> None:
    """Make a bare-text PP stage save and load under the wrapper's namespace.

    Only the stage owning ``embed_tokens`` is re-wrapped as the multimodal
    model, so its parameters are named ``language_model.*``. Every other stage
    is the bare text model and names them ``layers.*``,
    ``output_res_norm.weight`` and so on -- while a non-PP save, and the
    first stage's own save, use the prefixed form.

    That split namespace makes ANY checkpoint unloadable under PP for this
    model, not just a seed checkpoint: a resume fails with
    "Missing key in checkpoint state_dict: output_res_norm.weight". Cold
    starts never noticed because they load nothing.

    Fixed in the checkpoint path rather than by re-wrapping every stage: the
    wrapper's forward expects a tower and an image-splice path, and giving the
    middle stages one to satisfy a key-naming issue would trade a naming bug for
    a forward bug.
    """

    def _add_prefix(module, state_dict, prefix, local_metadata):
        del module, local_metadata
        for key in [k for k in state_dict if k.startswith(prefix)]:
            state_dict[prefix + _MM_INNER_PREFIX + key[len(prefix) :]] = state_dict.pop(
                key
            )

    def _strip_prefix(
        state_dict, prefix, local_metadata, strict, missing, unexpected, errors
    ):
        del local_metadata, strict, missing, unexpected, errors
        wrapped = prefix + _MM_INNER_PREFIX
        for key in [k for k in state_dict if k.startswith(wrapped)]:
            state_dict[prefix + key[len(wrapped) :]] = state_dict.pop(key)

    part._register_state_dict_hook(_add_prefix)
    part._register_load_state_dict_pre_hook(_strip_prefix)


_DEP_VISION_FQN = "__kimi_dep_vision__"
"""A name no text module has, so its PP chunk comes back parameterless."""


def dep_enabled() -> bool:
    """DEP is opt-in while it is being brought up.

    Off by default because it changes the stage count, so a run that enables it
    silently would report a different pipeline shape than the config asked for.
    """
    from torchtitan.models.kimi_k3.knobs import topology

    return topology().vit_dep


def _install_bubble_runtime_for(pp_schedule, prefetcher) -> None:
    """Wire the bubble planner and runtime to one vision stage's prefetcher.

    The plan is rebuilt per step from the schedule's own shape rather than cached,
    because the micro-batch count can change between steps (a short final batch) and a
    stale plan would anchor on actions the schedule no longer runs. Rebuilding is pure
    Python over a list of actions, so it is not worth caching against that risk.
    """
    from torchtitan.models.kimi_k3.dep_bubble_plan import build_plans
    from torchtitan.models.kimi_k3.dep_bubble_runtime import install_bubble_runtime
    from torchtitan.models.kimi_k3.knobs import topology

    cost_ratio = float(topology().vit_bubble_cost_ratio)

    def plan_for_step():
        n_mb = int(getattr(pp_schedule, "_n_microbatches", 0) or 0)
        pp_size = int(getattr(pp_schedule, "pp_group_size", 0) or 0)
        n_stages = int(getattr(pp_schedule, "_num_stages", 0) or 0)
        rank = int(getattr(pp_schedule, "rank", -1))
        if not (n_mb and pp_size and n_stages) or rank < 0:
            return None
        vp, rem = divmod(n_stages, pp_size)
        if rem or vp < 1:
            # A non-looped schedule has no interleaved action list to plan against.
            return None
        try:
            plans = build_plans(
                pp_size=pp_size,
                vp=vp,
                n_microbatches=n_mb,
                cost_ratio=cost_ratio,
            )
        except ValueError as err:
            # e.g. a micro-batch count Interleaved1F1B rejects. Saying so beats
            # silently running without the mechanism under test.
            logger.warning("DEP bubble plan unavailable: %s", err)
            return None
        return plans.get(rank)

    def encode_now(microbatches):
        # Synchronous on the CURRENT stream: this is the bubble, so the point is to
        # occupy it, not to overlap with it.
        for mb in microbatches:
            prefetcher.ensure_sync(mb)

    install_bubble_runtime(
        pp_schedule,
        plan_for_step=plan_for_step,
        encode_now=encode_now,
        upfront_encode=encode_now,
    )

    # The backward half. The queue lives on the OWNER module, because the seam that
    # cuts the graph is inside its forward and that is the only place the micro-batch
    # index and the features meet.
    from torchtitan.models.kimi_k3.dep_bubble_backward import (
        GradQueue,
        install_backward_slots,
    )

    queue = GradQueue(max_pending=int(topology().vit_bubble_max_pending))
    prefetcher._owner._vision_grad_queue = queue
    install_backward_slots(pp_schedule, queue)


def dep_vision_stages() -> int:
    """How many stages the vision tower occupies.

    Report 5.2.3 requires vision forward and backward to be "balanced across PP
    stages", so more than one is the target. It starts at 1 because the total stage
    count must stay divisible by ``pp_degree`` -- the schedule asserts that -- and
    the vision stages are taken OUT of the text budget rather than added on top.
    Growing this therefore trades text stages for vision stages, which is the
    balance the report is describing and which needs measurement to set.

    Above 1 the tower is split: share 0 takes ``patch_embed`` plus its blocks and
    ``embed_tokens``, the last share takes its blocks plus the projector and the
    splice, and what crosses each hop is a fixed-capacity patch stream alongside the
    text embeddings. See ``KimiK3ViTStage.set_dep_role``.
    """
    from torchtitan.models.kimi_k3.knobs import topology

    return max(1, topology().vit_dep_stages)


def _inject_kimi_k3_fqns(model: nn.Module, kwargs: dict) -> None:
    """Populate ``parallelism.module_fqns_per_model_part`` so the PP
    split uses Kimi module names and the last stage includes the
    AttnRes final-aggregation modules.
    """
    if not any(
        hasattr(model, n) for n in _KIMI_ATTN_RES_LAST_STAGE_FQNS
    ) and not hasattr(model, "embed_tokens"):
        return  # Not a Kimi model; pass through
    parallelism = kwargs.get("parallelism")
    if parallelism is None or parallelism.module_fqns_per_model_part is not None:
        return
    model_config = kwargs.get("model_config")
    pp = kwargs["parallel_dims"].pp
    if pp <= 1 or model_config is None:
        return

    # Layer count: kimi's config stores it at ``num_hidden_layers``.
    num_layers = getattr(model_config, "num_hidden_layers", None)
    if num_layers is None:
        return
    input_weight = parallelism.pipeline_parallel_first_stage_less_layers
    output_weight = parallelism.pipeline_parallel_last_stage_less_layers
    layers_per_stage = parallelism.pipeline_parallel_layers_per_stage

    if layers_per_stage is not None:
        num_virtual_stages = math.ceil(
            (num_layers + input_weight + output_weight) / layers_per_stage
        )
    else:
        from torchtitan.distributed.pipeline_parallel import get_schedule_class

        schedule_class = get_schedule_class(parallelism.pipeline_parallel_schedule)
        stages_per_rank = 1 if issubclass(schedule_class, PipelineScheduleSingle) else 2
        num_virtual_stages = pp * stages_per_rank

    n_vit = dep_vision_stages() if dep_enabled() else 0
    if n_vit:
        # Taken out of the text budget, not added on top: the schedule asserts
        # num_stages % pp_degree == 0, so appending would break pp=2 at the first
        # vision stage.
        if num_virtual_stages - n_vit < 1:
            raise ValueError(
                f"DEP wants {n_vit} vision stage(s) but only {num_virtual_stages} "
                "stages exist; raise pipeline_parallel_degree or lower "
                "KIMI_VIT_DEP_STAGES"
            )
        num_virtual_stages -= n_vit

    fqns = _kimi_llm_fqns(num_virtual_stages, num_layers, input_weight, output_weight)
    # Append AttnRes tail modules if present (last stage only).
    extras = [n for n in _KIMI_ATTN_RES_LAST_STAGE_FQNS if hasattr(model, n)]
    if extras:
        fqns[-1].extend(extras)
    if dep_enabled():
        # DEP: one stage ahead of the text ones that owns the vision tower. The
        # FQN deliberately matches NOTHING in the text model, so core's
        # _split_module -- which sets every non-matching child to None -- yields a
        # zero-parameter chunk. pipeline_llm then does `stages[i].submod = m` with
        # whatever parallelize_fn returns, so the empty chunk is replaced by the
        # ViT stage module. That is why this needs no core change and no rename of
        # language_model.*, which the alternative (hoisting the text stack's
        # children to the wrapper) would have forced.
        # The vision stage owns embed_tokens too, so the pipe carries the spliced
        # EMBEDDING stream rather than ids. Ids cannot travel the pipe: PP's
        # metadata inference pushes dummy values through it, and indexing an
        # embedding with those asserts out of bounds. The first text stage
        # therefore must NOT keep embed_tokens -- it receives pre-embedded input,
        # which the backbone already supports when embed_tokens is None.
        fqns[0] = [f for f in fqns[0] if f != "embed_tokens"]
        vision = [[f"{_DEP_VISION_FQN}{i}"] for i in range(n_vit)]
        vision[0].append("embed_tokens")
        fqns = vision + fqns
    parallelism.module_fqns_per_model_part = fqns


def _install_vision_prefetch(pp_schedule, model_parts) -> None:
    """Give the DEP vision stage a prefetcher and tell it which micro-batch it serves.

    The vision stage is NOT wrapped by :class:`CrossStageCacheAdapter` -- it holds the
    tower and embed_tokens, not AttnRes blocks -- so it does not get that wrapper's
    mb-index patch and needs its own. Same shape, and for the same reason: the index
    is schedule-owned and there is no other way to learn it from inside a forward.

    A no-op when the prefetch depth is 0, which is the default, so enabling DEP alone
    changes nothing here.
    """
    from torchtitan.models.kimi_k3.knobs import topology
    from torchtitan.models.kimi_k3.multimodal_model import KimiK3ViTStage
    from torchtitan.models.kimi_k3.vit_prefetch import (
        install_step_hook,
        prefetch_depth,
        VisionPrefetcher,
    )

    bubble = bool(topology().vit_bubble)
    if prefetch_depth() <= 0 and not bubble:
        return
    if prefetch_depth() > 0 and bubble:
        # Alternatives, not layers: the prefetch issues ahead on a side stream, the
        # bubble runtime places encodes in idle intervals on the main stream. Both at
        # once would have the prefetch satisfy every micro-batch before the planned slot
        # arrived, so the placements would report as fired while the side stream did the
        # work -- a green occupancy number for the wrong mechanism.
        raise ValueError(
            "KIMI_VIT_PREFETCH and KIMI_VIT_BUBBLE are alternatives; set exactly one. "
            f"Got prefetch={prefetch_depth()}, bubble={bubble}."
        )

    if dep_vision_stages() > 1 and prefetch_depth() > 0:
        # The run-ahead prefetches by calling encode_images, which assumes one stage
        # performs the whole encode. With the tower split that would run every block
        # on share 0 and defeat the split. Refuse rather than silently negate it.
        # Only the run-ahead is refused -- the bubble runtime is independent of the split.
        warnings.warn(
            f"KIMI_VIT_PREFETCH={prefetch_depth()} ignored: the run-ahead has no "
            f"cross-stage form yet, and KIMI_VIT_DEP_STAGES="
            f"{dep_vision_stages()} splits the tower. Running without the run-ahead."
        )
        if not bool(topology().vit_bubble):
            return

    vision_stage_modules = [m for m in model_parts if isinstance(m, KimiK3ViTStage)]
    if not vision_stage_modules:
        # Normal on a text-only rank: the vision stage is global stage 0, so only
        # one rank holds it. Logged rather than silent because "the run-ahead did
        # not install" and "the run-ahead did nothing" are otherwise the same
        # observation -- but WARN only where the stage was supposed to be, or
        # every text rank cries wolf on a correct run.
        owns_vision_stage = any(
            getattr(s, "stage_index", None) == 0
            for s in _iter_schedule_stages(pp_schedule)
        )
        message = (
            "DEP vision prefetch NOT installed: depth=%d, this rank's model parts "
            "are %s"
        )
        parts = [type(m).__name__ for m in model_parts]
        if owns_vision_stage:
            warnings.warn(
                f"KIMI_VIT_PREFETCH={prefetch_depth()} requested and this rank "
                f"owns pipeline stage 0, but no KimiK3ViTStage is present in its "
                f"model parts ({parts}); the run-ahead is OFF."
            )
        else:
            logger.info(message, prefetch_depth(), parts)
        return

    for module in vision_stage_modules:
        prefetcher = VisionPrefetcher(module)
        module._vision_prefetcher = prefetcher
        install_step_hook(pp_schedule, prefetcher)
        if bubble:
            _install_bubble_runtime_for(pp_schedule, prefetcher)

    # The micro-batch index patch lives in _install_vision_stage_wiring, which runs
    # first and unconditionally under DEP -- patching it here too would wrap
    # forward_one_chunk twice.
    for stage in _iter_schedule_stages(pp_schedule):
        if not isinstance(getattr(stage, "submod", None), KimiK3ViTStage):
            continue
        logger.info(
            "DEP vision prefetch installed: depth=%d on stage %s",
            prefetch_depth(),
            getattr(stage, "stage_index", "?"),
        )


def pipeline_kimi_k3_with_cache_adapter(model: nn.Module, **kwargs):
    """``pipelining_fn`` for Kimi Linear (baseline + AttnRes variants).

    Behavior:

    * Always: patch ``parallelism.module_fqns_per_model_part`` to use
      Kimi names and include final AttnRes modules on the last stage,
      then delegate to core ``pipeline_llm`` for the actual PP setup.
    * When ``TORCHTITAN_ATTNRES_CACHE=1`` AND the schedule is
      Interleaved1F1B AND the wrapped model is AttnRes (has
      ``num_blocks`` + ``layers_per_block`` attrs): wrap each stage's
      ``submod`` in ``CrossStageCacheAdapter`` (the
      implementation, reused unchanged — it duck-types the wrapped
      model's forward signature).
    * Otherwise: pass through (plain PP, no cache adapter).
    """
    # Resolve the topology knobs from config ONCE (finding 32). This entry can run
    # before parallelize, so whichever comes first registers; register_topology is
    # idempotent and reports a disagreement rather than letting order decide.
    from torchtitan.models.kimi_k3.knobs import register_topology

    if hasattr(model, "config"):
        register_topology(model.config)

    from torchtitan.distributed.pipeline_parallel import pipeline_llm

    model = _unwrap_multimodal_for_pp(model, kwargs)
    step_inputs = getattr(model, "_dep_step_inputs_holder", None)
    _inject_kimi_k3_fqns(model, kwargs)
    pp_schedule, model_parts, has_first_stage, has_last_stage = pipeline_llm(
        model, **kwargs
    )
    # Every kimi_k3 flavor registers THIS pipelining_fn, so the DEP wiring has to be
    # installed here; having it only in pipeline_llm_with_cache_adapter left it dead
    # code, and its absence read as "the prefetch changes nothing".
    if dep_enabled():
        # Wiring first, and unconditionally: a split tower's later shares need the
        # micro-batch index to find grid_thw, and without it they pass activations
        # through with no error at all.
        wired = _install_vision_stage_wiring(pp_schedule, step_inputs)
        # A split tower whose shares were never wired would run the
        # metadata-inference path for real micro-batches: activations passed
        # through, no tower, no splice, no error. So assert engagement -- but
        # against what THIS rank should own, not against a global count.
        #
        # The vision stages are the first dep_vision_stages() global stage
        # indices, so a rank owning none of them correctly wires zero. The first
        # version of this check read `wired == 0 and n_vit > 1`, which assumed
        # every rank owns a vision stage once the tower is split. That holds only
        # when n_vit == pp_degree; at pp=4 with n_vit=2 the two ranks holding
        # only text stages raised, and n_vit > 1 could not run at all.
        n_vit = dep_vision_stages()
        expected = sum(
            1
            for stage in _iter_schedule_stages(pp_schedule)
            if getattr(stage, "stage_index", None) is not None
            and stage.stage_index < n_vit
        )
        if wired != expected:
            raise RuntimeError(
                f"KIMI_VIT_DEP_STAGES={n_vit}: this rank owns {expected} vision "
                f"stage(s) by stage index but {wired} were wired; an unwired share "
                "passes activations through unprocessed and reports no error"
            )
        _install_vision_prefetch(pp_schedule, model_parts)
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

    # Detect AttnRes by Kimi-specific marker attributes on the wrapped model.
    inner0 = getattr(stages[0], "submod", None)
    num_blocks = getattr(inner0, "num_blocks", None)
    layers_per_block = getattr(inner0, "layers_per_block", None)
    if num_blocks is None or layers_per_block is None:
        warnings.warn(
            "Stage 0 model has no 'num_blocks'/'layers_per_block' — "
            "this is a baseline (non-AttnRes) Kimi Linear run; the "
            "cross-stage cache adapter only applies to AttnRes variants. "
            "Running without the adapter."
        )
        return passthrough

    # Layout tables: same math as attn_res, just with Kimi's layer count.
    model_config = kwargs.get("model_config")
    n_layers_total = getattr(model_config, "num_hidden_layers", None)
    if n_layers_total is None:
        warnings.warn(
            "Cannot determine total layer count; cache adapter falls back to passthrough."
        )
        return passthrough

    try:
        layout_tables = _infer_block_layout_tables_from_stages(
            stages,
            pp_size=pp_size,
            num_blocks=num_blocks,
            n_layers=n_layers_total,
            layers_per_block=layers_per_block,
        )
    except ValueError:
        # An unsupported configuration, not a rank-local mishap. Falling back
        # here would leave this rank without an adapter while its peers have
        # one, and a rank with no adapter sends no delta -- the first
        # cross-stage hop would hang instead of reporting the real problem.
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
        if i < len(model_parts):
            model_parts[i] = adapter

    _install_step_drop_patch(pp_schedule, installed_adapters)

    # Say so on success, not only on the fallback paths. The adapter is numerically
    # neutral by design, so loss reads the same whether it engaged or not -- without
    # this line "wrapped" and "silently fell back" are indistinguishable from the
    # outside.
    logger.info(
        "cross-stage cache adapter wrapped %d stage(s): %s",
        len(installed_adapters),
        [s.stage_index for s in stages],
    )

    return pp_schedule, model_parts, has_first_stage, has_last_stage

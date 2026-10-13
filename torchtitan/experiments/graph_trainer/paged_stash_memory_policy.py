# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

r"""
Paged stash memory policy for CUDA-graphable MoE training.

A port of Megatron-Core 0.19.0's paged stashing
(``megatron.core.transformer.moe.paged_stash``) onto GraphTrainer's
traced-graph flow.

Why Paged Stashing
------------------

When a MoE token dispatcher runs with a *static* capacity factor, every
permuted activation is allocated for the worst case while only
``sum(tokens_per_expert)`` rows hold data. With CUDA graphs on, every layer that
saves such an activation also needs its own capacity-padded buffer resident in
the graph pool, so the un-paged cost is ``num_layers x padded``.

Paged stashing reclaims that gap: once the forward consumers are done, the live
rows are copied into a shared pool of fixed-size pages and the padded tensor is
freed; backward pages it back in. The pool is shared by every layer -- one per
``(dtype, hidden_size)`` -- and sized from measured live rows rather than the
padded worst case. The saving grows with pipeline depth, because with PP those
activations otherwise sit idle through the whole bubble, once per in-flight
microbatch.

Usage
-----

Enable it with the ``sac_and_paged_stash`` memory policy on a model whose
dispatcher has a static capacity (HybridEP with ``non_blocking_capacity_factor``),
in a recipe derived from one that builds such a model::

    import dataclasses

    from torchtitan.models.common.moe import (
        RoundRobinTokenChoiceTopKRouter,
        TokenChoiceTopKRouter,
    )
    from torchtitan_recipes.tests.graph_trainer.deepseek_v3 import (
        graph_trainer_deepseek_v3_debugmodel_hybridep,
    )


    def paged_stash_debugmodel():
        config = graph_trainer_deepseek_v3_debugmodel_hybridep()
        config.compile.memory_policy = "sac_and_paged_stash"
        config.parallelism.data_parallel_shard_degree = 4
        config.parallelism.expert_parallel_degree = 2
        # Route round-robin, with exact balance (see below).
        router_fields = dataclasses.fields(TokenChoiceTopKRouter.Config)
        for layer in config.model.layers:
            if getattr(layer, "moe", None) is not None:
                router = layer.moe.router
                layer.moe.router = RoundRobinTokenChoiceTopKRouter.Config(
                    **{f.name: getattr(router, f.name) for f in router_fields}
                )
        return config


    def paged_stash_debugmodel_pp():
        # With pipeline parallelism, where the feature pays for itself.
        config = paged_stash_debugmodel()
        config.parallelism.pipeline_parallel_degree = 2
        config.parallelism.num_pp_microbatches = 4
        config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
        config.parallelism.data_parallel_shard_degree = 2
        return config

Launch either from the repo root with the recipe module on the path::

    NGPU=4 MODULE=my_recipes CONFIG=paged_stash_debugmodel ./run_train.sh

Both route tokens round-robin across experts, with exact balance, as the
integration tests do. Buffers are sized from the first step's routing, and the
debug model's untrained router shifts enough over the next few steps to outgrow
them, which the default ``deferred`` overflow check treats as fatal.

Options
-------

``compile.paged_stash.page_size``
    Tokens per page (default 64). Smaller pages waste less on stashes that do
    not fill a page.
``compile.paged_stash.buffer_size_factor_cuda``
    Headroom over the pages the measured step and the pipeline schedule
    require (default 1.10).
``compile.paged_stash.buffer_size_factor_cpu``
    Headroom for an optional pinned-host spill buffer (default 0 = no host
    spill).
``compile.paged_stash.prefetch_n_layers``
    Issue a backward's reloads this many paged-layer schedule entries early,
    across pipeline actions, to overlap page reads with compute (default 1,
    Megatron's). 0 reloads an activation when backward first reads it.
``compile.paged_stash.module_fqn``
    Module-FQN pattern of the modules annotated as the paged-stash region
    (default ``layers.*.moe.routed_experts``).
``compile.paged_stash.skip_immediate_backward``
    Keep the last paged layer resident instead of stashing it when the schedule
    runs that microbatch's backward next (default on). Matches Megatron's
    ``remove_paged_tensor_from_stash``.
``compile.paged_stash.overflow_check``
    ``deferred`` (default): reads the overflow flag from a pinned host copy a
    step or two later, never syncing, skips the optimizer steps in between on
    the GPU, and fails the job. Requires fused Adam or AdamW, the optimizers
    that can skip a step on device.
    ``blocking``: Megatron's per-step read, reruns an overflowing step with
    paging disabled, as Megatron does. Works with any optimizer.

How It Works
------------

1. **Annotation, not a context manager.** Megatron opens ``saved_tensors_hooks``
   around the fused grouped MLP and pages whatever Transformer Engine tagged
   inside it. There is no such window in a traced graph, so
   ``tag_paged_stash_activations`` annotates nodes -- following the same
   ``MUST_SAVE`` / ``MUST_RECOMPUTE`` convention as the rest of the
   memory-policy framework -- and ``apply_paged_stash_pass`` rewrites them.
   The region it decides is annotated on the model before tracing, the way
   GraphTrainer annotates module FQNs and EP regions: every activation there
   that backward reads and the stash can hold is stashed, whatever SAC chose.
   The rewrite runs after selective activation remat, whose recomputed
   duplicates would otherwise keep reading the padded original.

2. **The forward stash is a mutation.** GraphPP defines its forward graph as
   the dependency closure of the forward user outputs. A value-returning stash
   reaches no forward output, so the whole chain would be classified as
   backward and the activation would still be saved. Writing the stash as a
   mutation of a small ``handle`` tensor routes it through GraphPP's existing
   forward-mutation path, so the handle -- not the padded tensor -- is what
   crosses into backward.

3. **Megatron's stash and reload points.** A layer's stashes launch together
   at the end of its expert MLP and are joined, freeing the padded
   activations, at the next paged layer's group start -- just before its
   expert MLP reads the dispatch output -- as Megatron's
   ``paged_stash_group_commit`` and ``paged_stash_group_start`` do. Reloads
   are issued one schedule entry ahead, as Megatron's are: the paged-layer
   schedule is read off GraphPP's actions, so a backward's first layer is
   reloaded at the end of whatever forward or backward precedes it on the
   rank, even in another pipeline action.

4. **Lifecycle.** The first step measures: ``stash_`` reads the real token
   count and keeps just the live rows, as Megatron's capture iteration does,
   so the step that does not page needs no more memory than one that does. The second allocates the buffers, primes the Triton
   kernels (Triton cannot compile inside a CUDA graph capture), and pages for
   real. Both run eagerly. The trainer captures the third whole step in one
   CUDA graph, every pipeline action at its own place in the schedule, and
   replays it from then on. Later steps only reset freelists and flags, which
   allocates nothing. ``PagedStashRunner`` does that outside the graph, since
   it also reads the overflow flags on the host and may rerun a step.

5. **GraphPP replaces the virtual-pipeline estimate.** Megatron reconstructs
   its schedule to work out how many stashes are live at once.
   ``schedule_page_peak`` replays the real PP action list over a per-stage
   page footprint instead, which is both simpler and tighter.

6. **Sizing is in pages, not tokens.** Megatron accumulates raw token counts,
   which under-allocates when individual stashes are small relative to
   ``page_size``: four concurrent 16-token stashes need four pages, not one.

7. **Overflow is fatal, and checking for it never syncs.** Detection is
   identical to Megatron -- a device-side flag from the Triton kernels, next to
   ``host_spill`` and ``overbudget``, HybridEP's token-drop flag accumulated
   over every dispatch of the step. Megatron all-reduces the flags and reads
   them with ``.item()`` every step; here each step copies the rank's own flags
   to pinned host memory asynchronously and a later step reads the copy, and a
   rank that overflowed raises. The overflow flag is sticky -- ``begin_step``
   leaves it set -- so a check that reads it steps later cannot miss it, and a
   blocking read at shutdown covers the final steps. The steps run before the
   raise never apply their corrupted gradients: each optimizer step
   all-reduces the flag on device and passes it to fused Adam/AdamW as AMP's
   ``found_inf``, so every rank skips the update, and a skipped step's MoE
   expert-bias update is undone. ``blocking`` mode instead reads the
   all-reduced flags synchronously and reruns an overflowing step with paging
   disabled, as Megatron does. A token drop is reported but never fatal: it
   comes from the capacity factor, not the stash.

The Triton kernels, the ``PagedStashBuffer`` layout, the stash manager and
the overflow runner live in ``torchtitan.distributed.deepep.paged_stash``,
shared with the eager trainer; this module is GraphTrainer's frontend to them:
the region annotation, the tagging pass, the FX rewrite and its custom ops, and
the GraphPP schedule.

Limitations
-----------

- Requires a token dispatcher with a static capacity factor. With a
  blocking/dynamic dispatcher the permuted buffer is already exactly sized and
  the pass tags nothing.
- Only capacity-padded activations in the annotated region are paged;
  elsewhere SAC decides as usual.
- The stash copies a tensor's live tokens as a prefix of its bytes, so tensors
  in the region must store tokens in order. The pass reads the row layout off
  a tensor's shape or any reshape of it -- a flat block-scale buffer reads as
  token-ordered, as Transformer Engine's do in Megatron. A tensor no shape shows
  token rows for is left to SAC; if backward then rebuilds it from a padded
  input, that input crosses into backward unpaged, and a warning names it.
- With the default ``deferred`` overflow check, an overflow can surface as the
  trainer's "Loss or gradient norm is not finite" error rather than the paged
  stash one: the overflowing step's backward reads activations that were never
  written back, and if they make the gradient norm non-finite, that check stops
  the job first. Either way the step is never applied.
"""
from __future__ import annotations

import logging
import operator
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch
import torch.nn as nn
from torch._library.custom_ops import custom_op
from torch.distributed.pipelining.schedules import _PipelineScheduleRuntime
from torch.fx import has_side_effect, Node
from torch.fx.traceback import annotate_fn
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.distributed.deepep.paged_stash.config import PagedStashConfig
from torchtitan.distributed.deepep.paged_stash.runtime import (
    _num_pages,
    DEFAULT_MODULE_FQN,
    hybridep_overbudget_probe,
    infer_slot_dims,
    max_live_stash_generations,
    moe_expert_biases,
    PageCounts,
    PagedStashManager,
    PagedStashRunner,
    replay_schedule_page_peak,
    StashSlot,
    wait_for_copy,
    worth_paging,
)
from torchtitan.experiments.graph_trainer.common_utils import (
    _get_layer_id,
    _is_backward_node,
    _NOT_IN_LAYERS,
    matches_module_fqn_pattern,
)
from torchtitan.experiments.graph_trainer.cpu_offload import (
    _can_offload_node,
    _collect_offload_info,
    _get_aten_target,
    _get_storage_chain,
    _is_view,
    _OffloadInfo,
    _redirect_backward_consumers,
    _tensor_bytes,
    _tensor_is_contiguous,
)
from torchtitan.experiments.graph_trainer.memory_policy import (
    _default_memory_policy_pass,
)
from torchtitan.experiments.graph_trainer.registry import register_memory_policy

if TYPE_CHECKING:
    from torchtitan.components.optim.optimizer import OptimizersContainer
    from torchtitan.experiments.graph_trainer.trainer import GraphTrainer


logger = logging.getLogger(__name__)


# ===========================================================================
# Custom ops that page MoE activations in and out of the stash buffers
# ===========================================================================
#
# These are the FX-visible form of Megatron's ``PagedTensor.offload_to_stash`` /
# ``reload_from_stash`` and of the two autograd functions that drive them around
# the grouped MLP, ``paged_stash_group_start`` and ``paged_stash_group_commit``.
# Megatron runs all of it from ``saved_tensors_hooks`` and those functions'
# forward and backward; GraphTrainer has no autograd tape at graph-execution
# time, so the same points are ops that ``apply_paged_stash_pass`` places. Per
# paged layer, the resulting IR is::
#
#     Forward:  x = paged_stash.group_start(x)    # the dispatch output, before
#                                                  # the expert MLP reads it
#               ... the expert MLP ...
#               handle = aten.empty([max_pages + 1], int64)
#               paged_stash.stash_(handle, act, num_tokens, slot, prefetch)
#
#     Backward: act = paged_stash.reload(handle, num_tokens, slot, size, dtype)
#               act = paged_stash.wait_reload(act)
#               ... the expert MLP's backward ...
#               dx = paged_stash.reload_next(dx, slot, prefetch)
#                                                  # the dispatch output's gradient
#
# ``stash_`` launches the copy of the activation's live rows on the side stream,
# at the end of the expert MLP like Megatron's group commit. ``group_start`` is
# Megatron's group start: it joins every copy still in flight -- the previous
# paged layer's, which may have run in an earlier pipeline action -- and only
# then lets the padded activations go. A backward joins them too, at its first
# ``reload``, in case no group start ran in between.
#
# Reloads run one schedule entry ahead, as Megatron's do. A forward's last stash
# and every ``reload_next`` start the reloads of the next paged layer's backward
# whenever that is what the schedule runs next: the same backward's next layer,
# or the first layer of another microbatch's or stage's backward, possibly in a
# later pipeline action. ``reload`` hands backward the prefetched tensor, or
# pops the pages itself when nothing was prefetched, and ``wait_reload`` joins
# the copy before backward reads it.
#
# ``handle`` is one small int64 tensor: ``[:max_pages]`` records which page backs
# each chunk of the activation, and the last element is the spill flag. It is the
# only thing that crosses into backward, in place of the padded activation.
#
# The stash is a *mutation* of ``handle`` rather than a value-returning op, and
# that is load-bearing rather than cosmetic. GraphPP defines its forward graph
# as the dependency closure of the forward user outputs; a value-returning
# stash reaches no forward output, so the whole chain would be classified as
# backward, the activation would still be saved, and the pass would achieve
# nothing. Expressing the stash as a mutation in forward puts it through
# GraphPP's existing ``_forward_mutations_to_materialize`` path, which runs the
# mutation in the forward callable and saves the mutated base -- the handle --
# for backward. ``group_start`` stays in forward and ``reload_next`` in
# backward the other way: each passes a value through, so forward outputs
# depend on the first and backward outputs on the second.
#
# All stash and reload copies run on a single side stream. Megatron shares one
# stream between pack and unpack because the freelist is unsynchronized mutable
# state; that invariant is preserved here.


def _slot(slot_id: int) -> StashSlot:
    manager = PagedStashManager.get_instance()
    try:
        return manager.slots[slot_id]
    except KeyError:
        raise RuntimeError(
            f"paged stash: unknown slot id {slot_id}. The graph pass must "
            "register every slot before the graph runs."
        ) from None


@custom_op("paged_stash::stash_", mutates_args=("handle",))
def stash_(
    handle: torch.Tensor,
    activation: torch.Tensor,
    num_tokens: torch.Tensor,
    slot_id: int,
    prefetch: int = 0,
) -> None:
    """Copy the live rows of ``activation`` into the paged stash, into ``handle``.

    ``handle`` must be an ``int64`` tensor of ``max_pages + 1`` elements. Once
    the copy completes it holds the page id backing each ``page_size``-token
    chunk, plus a trailing flag recording whether the stash landed in the
    pinned host buffer.

    The caller allocates ``handle`` and the stash fills it in place rather than
    returning it, because GraphPP keeps an op in the forward callable only if a
    forward output depends on it or it mutates a forward value. A returned
    handle would feed nothing but backward's ``reload``, so the stash would run
    in backward and the padded activation would be saved for it; see the
    section comment above.

    The copy runs on the side stream, and the manager keeps ``activation``
    alive until the next ``group_start``, or a backward's first ``reload``,
    joins it. When the schedule runs this layer's backward next, nothing is
    copied and the activation stays resident instead (see
    ``PagedStashManager.should_skip_stash``).

    ``prefetch`` is nonzero on a layer's last stash, where Megatron's group
    commit runs: once the layer's copies are launched, the reloads of every
    backward among the next ``prefetch`` schedule entries start.
    """
    PagedStashManager.get_instance().stash(
        _slot(slot_id), activation, num_tokens, handle=handle, prefetch=prefetch
    )


@stash_.register_fake
def _stash_fake(
    handle: torch.Tensor,
    activation: torch.Tensor,
    num_tokens: torch.Tensor,
    slot_id: int,
    prefetch: int = 0,
) -> None:
    return None


@custom_op("paged_stash::reload", mutates_args=())
def reload(
    handle: torch.Tensor,
    num_tokens: torch.Tensor,
    slot_id: int,
    size: list[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    """Restore a stashed activation from its pages.

    Hands back the tensor a prefetch already started filling, or pops the pages
    now if nothing was prefetched (see ``_launch_reload``); a stash that stayed
    resident hands back the activation itself.
    """
    manager = PagedStashManager.get_instance()
    return manager.materialize(manager.take_stash_record(_slot(slot_id), handle))


@reload.register_fake
def _reload_fake(
    handle: torch.Tensor,
    num_tokens: torch.Tensor,
    slot_id: int,
    size: list[int],
    dtype: torch.dtype,
) -> torch.Tensor:
    return torch.empty(size, dtype=dtype, device=handle.device)


# The remaining ops pass their input through: the output aliases the input,
# which ``custom_op`` forbids, so they are defined through torch.library
# directly -- the same reason ``ao::wait_tensor`` is. Passing a value through is
# what places each one: the consumers of that value run after it, and in GraphPP
# it lands in the callable whose outputs depend on that value.
_lib = torch.library.Library("paged_stash", "FRAGMENT")
_lib.define("group_start(Tensor(a) tensor) -> Tensor(a)")
_lib.define("wait_reload(Tensor(a) tensor) -> Tensor(a)")
_lib.define("reload_next(Tensor(a) tensor, int slot_id, int count) -> Tensor(a)")


@torch.library.impl("paged_stash::group_start", "CompositeExplicitAutograd")
def _paged_stash_group_start(tensor: torch.Tensor) -> torch.Tensor:
    """Megatron's ``paged_stash_group_start``: join the copies still in flight
    before the expert MLP reads ``tensor``, which frees the padded activations
    they read."""
    PagedStashManager.get_instance().join_stashes()
    return tensor


@torch.library.impl("paged_stash::wait_reload", "CompositeExplicitAutograd")
def _paged_stash_wait_reload(tensor: torch.Tensor) -> torch.Tensor:
    """Join the side stream into compute before backward reads the reload."""
    wait_for_copy(tensor)
    return tensor


@torch.library.impl("paged_stash::reload_next", "CompositeExplicitAutograd")
def _paged_stash_reload_next(
    tensor: torch.Tensor, slot_id: int, count: int
) -> torch.Tensor:
    """Megatron's ``PipelinePreScheduleFunction.backward``: once the expert
    MLP's backward has produced ``tensor``, the gradient of its input, start the
    reloads of every backward among the next ``count`` schedule entries.
    ``slot_id`` is any of the layer's slots; it names the layer."""
    manager = PagedStashManager.get_instance()
    entry = manager.backward_entry(_slot(slot_id)) if manager.active else None
    if entry is not None:
        manager.prefetch_after(entry, count)
    return tensor


@torch.library.register_fake("paged_stash::group_start")
def _paged_stash_group_start_fake(tensor: torch.Tensor) -> torch.Tensor:
    return tensor


@torch.library.register_fake("paged_stash::wait_reload")
def _paged_stash_wait_reload_fake(tensor: torch.Tensor) -> torch.Tensor:
    return tensor


@torch.library.register_fake("paged_stash::reload_next")
def _paged_stash_reload_next_fake(
    tensor: torch.Tensor, slot_id: int, count: int
) -> torch.Tensor:
    return tensor


# The backward ops are side effects to FX, which keeps dead-code elimination and
# reordering off them. ``group_start`` must not be: GraphPP's backward
# extraction copies every side effect whose inputs it can compute, and backward
# can recompute the dispatch output from what forward saves, so it would run the
# dispatch's all-to-all a second time just to feed a join. The expert MLP
# reading its output is what keeps it in place.
has_side_effect(torch.ops.paged_stash.wait_reload.default)
has_side_effect(torch.ops.paged_stash.reload_next.default)


# ===========================================================================
# Annotation pass: which activations should be paged
# ===========================================================================
#
# Megatron marks stash candidates imperatively -- ``get_paged_stash_context()``
# opens ``saved_tensors_hooks`` around the fused grouped MLP, and every tensor
# Transformer Engine tagged with ``grouped_tensor_scale_inv`` inside that context
# gets paged. GraphTrainer has no such dynamic context, so the same selection is
# expressed as node annotations on the traced graph, following the existing
# memory-policy convention in ``memory_policy.py`` (``MUST_SAVE`` /
# ``MUST_RECOMPUTE`` / ``MUST_CPU_OFFLOAD``).
#
# The selection rule is the graph equivalent of Megatron's: an activation is
# pageable when it is produced inside the routed-expert region, its leading
# dimension is the *capacity-padded* permuted-token dimension, and only the first
# ``sum(tokens_per_expert)`` rows of it are live. That padded-vs-live gap is
# exactly what paged stashing reclaims, and it is why the token dispatcher must be
# running with a static capacity factor.
#
# The region is annotated on the model before tracing, the trace-time
# counterpart of Megatron's context: ``annotate_paged_stash_region`` wraps the
# region's modules with ``annotate_fn``, as ``annotate_module_fqns`` does for
# FQNs, so every node traced inside carries ``PAGED_STASH_REGION`` in
# ``node.meta["custom"]``. Opting into paged stashing hands it that region:
# ``tag_paged_stash_activations`` stashes every activation there that backward
# reads and the stash can hold, whatever SAC or CPU offload decided before it.


# node.meta key set by this pass and consumed by apply_paged_stash_pass.
PAGED_STASH_TAG = "paged_stash"

# node.meta["custom"] key marking the paged-stash region.
PAGED_STASH_REGION = "paged_stash_region"


@dataclass(frozen=True, slots=True)
class TokenCounts:
    """Where a layer's padded and live token counts come from.

    ``padded`` is the static leading dimension of every permuted activation in
    the layer -- the capacity-factor-sized buffer the dispatcher writes into.
    ``per_expert_node`` produces the device-side per-local-expert token counts;
    summing it gives the live row count, which is what the stash actually
    copies. This pair is the GraphTrainer analogue of Megatron's
    ``(max_num_tokens, num_tokens_tensor)``. ``hidden_node`` is the dispatch
    output, the expert MLP's input.
    """

    layer_id: int
    padded: int
    per_expert_node: Node
    hidden_node: Node


def _dispatch_token_count_output(node: Node) -> int | None:
    """Return the output index carrying per-local-expert token counts.

    Token dispatchers that expose a *static* permuted-token capacity are the
    only ones paged stashing applies to: a dynamically sized permuted buffer is
    already exactly as large as the live tokens, so there is nothing to reclaim.
    """
    target = node.target
    if target is torch.ops.hybridep.dispatch.default:
        # (hidden, scores, tokens_per_expert, handle)
        return 2
    return None


def find_layer_token_counts(gm: torch.fx.GraphModule) -> dict[int, TokenCounts]:
    """Map each MoE layer to its padded capacity and live-token-count node."""
    counts: dict[int, TokenCounts] = {}
    for node in gm.graph.nodes:
        if node.op != "call_function" or _is_backward_node(node):
            continue
        count_index = _dispatch_token_count_output(node)
        if count_index is None:
            continue
        layer_id = _get_layer_id(node)
        if layer_id == _NOT_IN_LAYERS:
            continue

        hidden_node: Node | None = None
        per_expert_node: Node | None = None
        for user in node.users:
            if user.target is not operator.getitem:
                continue
            if user.args[1] == 0:
                hidden_node = user
            elif user.args[1] == count_index:
                per_expert_node = user
        if hidden_node is None or per_expert_node is None:
            continue
        hidden_val = hidden_node.meta.get("val")
        if not isinstance(hidden_val, torch.Tensor) or hidden_val.dim() < 1:
            continue
        padded = hidden_val.shape[0]
        if isinstance(padded, torch.SymInt):
            # A symbolic permuted dimension means the dispatcher is in blocking
            # mode and already sizes exactly; nothing to page.
            continue
        counts[layer_id] = TokenCounts(
            layer_id=layer_id,
            padded=int(padded),
            per_expert_node=per_expert_node,
            hidden_node=hidden_node,
        )
    return counts


def _whole_storage_views(node: Node) -> Iterator[Node]:
    """Views of ``node``, through any chain of views, that reshape all of it.

    Same element count, no offset and contiguous, so each describes the same
    bytes in the same order. Backward views count too: remat moves a view whose
    only readers are in backward into backward.
    """
    numel = node.meta["val"].numel()
    pending = [user for user in node.users if _is_view(user)]
    while pending:
        view = pending.pop()
        val = view.meta.get("val")
        if (
            isinstance(val, torch.Tensor)
            and val.numel() == numel
            and val.storage_offset() == 0
            and _tensor_is_contiguous(val)
        ):
            yield view
        pending.extend(user for user in view.users if _is_view(user))


def _slot_dims(node: Node, padded: int) -> tuple[int, int] | None:
    """``(token_divisor, hidden_size)`` read off ``node``'s shape, or off a view
    that reshapes all of it.

    A traced graph often holds a buffer under a shape made for some kernel's
    API rather than for its tokens. A fused MXFP8 expert MLP, for instance,
    gathers its FC1-input block scales as ``(1536, 512)`` tiles, views them as
    ``(256, 3072)`` and saves them flattened for backward: three shapes of one
    token-ordered buffer, of which only the flat one reads as token rows. Each
    view describes the same bytes, so one that reads as token rows is as good a
    witness as the buffer's own shape. That keeps the region annotation the
    only thing paging needs; nothing in the region declares its layout.
    """
    dims = infer_slot_dims(node.meta["val"], padded)
    if dims is not None:
        return dims
    for view in _whole_storage_views(node):
        dims = infer_slot_dims(view.meta["val"], padded)
        if dims is not None:
            return dims
    return None


# Views that hand their outputs out through ``getitem``. CPU offload's view
# chain stops at the ``getitem``, which would hide a backward read through one:
# a fused gate/up projection, for instance, is read in backward through the
# ``unbind`` that separates its two halves.
_MULTI_OUTPUT_VIEW_OPS = frozenset(
    {
        torch.ops.aten.unbind,
        torch.ops.aten.split,
        torch.ops.aten.split_with_sizes,
        torch.ops.aten.chunk,
        torch.ops.aten.tensor_split,
    }
)


def _is_stash_view(node: Node) -> bool:
    """``_is_view``, extended through multi-output views and their ``getitem``."""
    if _is_view(node) or _get_aten_target(node) in _MULTI_OUTPUT_VIEW_OPS:
        return True
    return (
        node.target is operator.getitem
        and isinstance(node.args[0], Node)
        and _get_aten_target(node.args[0]) in _MULTI_OUTPUT_VIEW_OPS
    )


def _is_pageable(node: Node, padded: int) -> bool:
    """Whether ``node``'s output is a permuted activation worth paging.

    Starts from what CPU offload requires of any activation it moves out of the
    way until backward -- a materialized, contiguous, non-view tensor -- and
    adds the row layout the stash addresses. A view is paged through its base.
    """
    if _is_backward_node(node) or _is_stash_view(node) or not _can_offload_node(node):
        return False
    val = node.meta["val"]
    if val.device.type != "cuda":
        return False
    if val.dim() == 0 or any(isinstance(s, torch.SymInt) for s in val.shape):
        return False
    dims = _slot_dims(node, padded)
    if dims is None:
        return False
    divisor, _ = dims
    return worth_paging(_tensor_bytes(val), divisor)


def annotate_paged_stash_region(
    model: nn.Module, module_fqn_pattern: str = DEFAULT_MODULE_FQN
) -> None:
    """Annotate the modules matching ``module_fqn_pattern`` as the paged-stash
    region.

    Wraps each matching module's forward with ``annotate_fn``, as
    ``annotate_module_fqns`` does, so every node traced inside it, including
    inside its submodules, carries ``PAGED_STASH_REGION`` in
    ``node.meta["custom"]``. Call before the model is traced.
    """
    for fqn, module in model.named_modules():
        if matches_module_fqn_pattern(module_fqn_pattern, fqn):
            module.forward = annotate_fn({PAGED_STASH_REGION: True})(module.forward)


def tag_paged_stash_activations(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
) -> torch.fx.GraphModule:
    """Stash every activation in the paged-stash region that backward reads.

    Paged stashing owns the region ``annotate_paged_stash_region`` marks: each
    activation there that backward reads and the stash can hold is saved and
    paged, whatever SAC or CPU offload decided before, and nothing in it is
    recomputed. For the stock BF16 SwiGLU experts that is the dispatch output,
    the fused gate/up projection -- which backward reads through the ``unbind``
    that splits it -- the SiLU, the gate product and the down projection's
    output. Activations the stash cannot hold keep SAC's decision: the dispatch
    and combine collectives themselves, the per-row router scores, and the
    expert weights.
    """
    token_counts = find_layer_token_counts(gm)
    if not token_counts:
        logger.info(
            "Paged stash: no static-capacity token dispatch found; nothing tagged."
        )
        return gm

    tagged = 0
    tagged_bytes = 0
    per_layer: dict[int, int] = defaultdict(int)
    for node in gm.graph.nodes:
        if PAGED_STASH_REGION not in node.meta.get("custom", {}):
            continue
        counts = token_counts.get(_get_layer_id(node))
        if counts is None or not _is_pageable(node, counts.padded):
            continue
        _, has_backward = _get_storage_chain(node, is_view=_is_stash_view)
        if not has_backward:
            continue
        node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
        node.meta[PAGED_STASH_TAG] = True
        tagged += 1
        tagged_bytes += _tensor_bytes(node.meta["val"])
        per_layer[counts.layer_id] += 1

    if tagged:
        logger.info(
            "Paged stash: tagged %d activation(s) across %d layer(s) "
            "(%.2f MiB padded)",
            tagged,
            len(per_layer),
            tagged_bytes / (1024 * 1024),
        )
        for layer_id in sorted(per_layer):
            logger.debug(
                "  layer %d: %d tagged, padded tokens %d",
                layer_id,
                per_layer[layer_id],
                token_counts[layer_id].padded,
            )
    else:
        logger.info("Paged stash: no activations in the paged-stash region")
    return gm


# ===========================================================================
# Insert paged stash / reload ops around tagged activations
# ===========================================================================
#
# This is the structural counterpart of ``cpu_offload.apply_cpu_offload_pass``,
# and of Megatron's ``PipelinePreScheduleFunction`` /
# ``PipelinePostScheduleFunction`` pair. Megatron needs those autograd functions
# because it discovers stash and reload points while the tape runs; here the
# joint graph already contains both halves, so a single pass places every op
# and rewires backward directly.
#
# Rewrite performed for each paged layer::
#
#     forward:
#         x = hybridep.dispatch(...)[0]
#         x' = paged_stash.group_start(x)
#         ... the expert MLP, reading x' ...
#         # for every activation tagged by tag_paged_stash_activations, in order:
#         handle = aten.empty([max_pages + 1], int64)
#         paged_stash.stash_(handle, act, num_tokens, slot, prefetch)
#
#     backward:
#         # for every stashed activation:
#         act' = paged_stash.reload(handle, num_tokens, slot, size, dtype)
#         act' = paged_stash.wait_reload(act')
#         ... backward consumers rewired from act to act' ...
#         dx' = paged_stash.reload_next(dx, slot, prefetch)
#         hybridep.combine_bwd(dx', ...)  # the dispatch's backward
#
# After the rewrite nothing in backward reads a padded activation, so nothing
# keeps it once the next group start joins its copy; ``handle`` and
# ``num_tokens`` -- a few kilobytes -- are what cross into backward. Under
# GraphPP that means the stage's ``saved_values_for_backward`` carries the
# handle instead of the padded tensor, which is where the memory is actually
# recovered.
#
# The points are Megatron's. A layer's stashes launch together once the expert
# MLP has produced everything it saves (``paged_stash_group_commit``), so no
# copy competes with the expert GEMMs, and the next layer's ``group_start``
# joins them (``paged_stash_group_start``), so the copies overlap the combine
# and the next layer's attention and dispatch. Each backward's reloads are
# issued at the schedule entry before it -- a forward's last stash, or the
# ``reload_next`` that ends a backward's expert MLP (the backward of Megatron's
# group start) -- and wait in that microbatch's stash records, so the entry
# before may be another pipeline action's graph.
#
# The backward half is CPU offload's: its helpers find the backward consumers
# and rewire them to the reload, replaying any view chain.


aten = torch.ops.aten
paged_stash_ops = torch.ops.paged_stash


@dataclass(frozen=True, slots=True)
class _StashPlan:
    offload: _OffloadInfo
    slot: StashSlot
    num_tokens_node: Node


def _make_num_tokens_node(
    gm: torch.fx.GraphModule,
    per_expert_node: Node,
) -> Node:
    """Insert ``sum(tokens_per_expert)`` -- the live row count for a layer.

    Megatron passes ``tokens_per_expert.sum()`` into
    ``get_paged_stash_context(num_tokens_tensor=...)``; the same value is
    materialized here as a graph node so it stays on device and never forces a
    host sync, which is what keeps the stash CUDA-graph safe.
    """
    val = per_expert_node.meta.get("val")
    with gm.graph.inserting_after(per_expert_node):
        node = gm.graph.call_function(aten.sum.default, args=(per_expert_node,))
    node.meta.update(
        {k: v for k, v in per_expert_node.meta.items() if k not in ("val", "recompute")}
    )
    if isinstance(val, torch.Tensor):
        node.meta["val"] = val.sum()
    # This is a forward value even though backward reads it.
    node.meta.pop("autograd_backward", None)
    return node


def _plan_stashes(
    gm: torch.fx.GraphModule,
    tagged: list[Node],
    token_counts: dict[int, TokenCounts],
) -> tuple[list[_StashPlan], dict[Node, int]]:
    """Register a stash slot for every tagged node that backward reads.

    First inserts the per-layer live-token-count nodes the plans refer to, then
    returns the plans with the node index they were made against.
    """
    manager = PagedStashManager.get_instance()

    # One live-token-count node per MoE layer, shared by every stash in it.
    num_tokens_nodes: dict[int, Node] = {}
    for layer_id in sorted({_get_layer_id(n) for n in tagged}):
        counts = token_counts.get(layer_id)
        if counts is None:
            raise RuntimeError(
                f"paged stash: layer {layer_id} has tagged activations but no "
                "token-count node; tag_paged_stash_activations and "
                "apply_paged_stash_pass disagree about the dispatcher."
            )
        num_tokens_nodes[layer_id] = _make_num_tokens_node(gm, counts.per_expert_node)
    node_to_index = {n: i for i, n in enumerate(gm.graph.nodes)}

    plans: list[_StashPlan] = []
    for node in tagged:
        val = node.meta["val"]
        layer_id = _get_layer_id(node)
        padded = token_counts[layer_id].padded
        dims = _slot_dims(node, padded)
        if dims is None:
            # tag_paged_stash_activations only tags nodes with slot dims, so
            # this means the two disagree about the token count.
            raise RuntimeError(
                f"paged stash: tagged node {node.name} has no slot dims "
                f"for layer {layer_id}; tagging and rewriting disagree."
            )
        offload = _collect_offload_info(node, node_to_index, is_view=_is_stash_view)
        if offload is None:
            continue

        token_divisor, hidden_size = dims
        slot = StashSlot(
            slot_id=manager.next_slot_id(),
            dtype=val.dtype,
            hidden_size=int(hidden_size),
            max_num_tokens=padded,
            token_divisor=token_divisor,
            layer_id=layer_id,
            node_name=node.name,
            stage_index=manager.current_stage_index,
        )
        manager.register_slot(slot)
        logger.debug(
            "Paged stash slot %d: %s shape=%s numel=%d dtype=%s -> "
            "hidden=%d divisor=%d max_tokens=%d rows=%d",
            slot.slot_id,
            node.name,
            tuple(val.shape),
            val.numel(),
            val.dtype,
            slot.hidden_size,
            slot.token_divisor,
            slot.max_num_tokens,
            slot.max_stash_tokens,
        )
        plans.append(
            _StashPlan(
                offload=offload,
                slot=slot,
                num_tokens_node=num_tokens_nodes[layer_id],
            )
        )
    return plans, node_to_index


def _stash_group_commits(
    plans: list[_StashPlan], node_to_index: dict[Node, int]
) -> dict[int, Node]:
    """Per layer, the latest of its stashed activations and their token counts.

    That is where the layer's stashes launch, together: once the expert MLP has
    produced everything it saves, as Megatron's ``paged_stash_group_commit``
    launches them after the grouped MLP. Launching each copy as soon as its
    activation exists would put it alongside the expert GEMMs, and a copy
    kernel occupies every SM while it runs.
    """
    commits: dict[int, Node] = {}
    for plan in plans:
        for candidate in (plan.offload.node, plan.num_tokens_node):
            current = commits.get(plan.slot.layer_id)
            if current is None or node_to_index[candidate] > node_to_index[current]:
                commits[plan.slot.layer_id] = candidate
    return commits


def _meta_from(node: Node) -> dict[str, Any]:
    """``node``'s metadata minus what describes its value or its tagging, so
    the ops inserted for it are attributed to the same module in tlparse dumps
    and kernel annotations."""
    return {
        k: v
        for k, v in node.meta.items()
        if k not in ("val", "recompute", PAGED_STASH_TAG)
    }


def _insert_stash_ops(
    gm: torch.fx.GraphModule,
    plan: _StashPlan,
    *,
    page_size: int,
    prefetch: int,
    insert_after: Node,
) -> tuple[Node, Node]:
    """Insert the forward stash and backward reload for ``plan``.

    The stash is inserted after ``insert_after``: the layer's commit (see
    ``_stash_group_commits``) or the layer's previous stash, so a layer's
    stashes launch together and in order. ``prefetch`` is the stash's
    ``prefetch`` argument, set on the layer's last one. Returns the stash node
    and the ``wait_reload`` node, placed in front of the first backward
    consumer; rewiring the consumers to it is left to the caller.
    """
    node = plan.offload.node
    val = node.meta["val"]
    max_pages = _num_pages(plan.slot.max_stash_tokens, page_size)
    src_meta = _meta_from(node)

    with gm.graph.inserting_after(insert_after):
        # One int64 tensor per stashed activation: page ids followed by the
        # spill flag. Keeping it as a single tensor gives the forward
        # mutation a single base for GraphPP to save.
        handle = gm.graph.call_function(
            aten.empty.memory_format,
            args=([max_pages + 1],),
            kwargs={
                "dtype": torch.int64,
                "device": val.device,
                "pin_memory": False,
            },
        )
        handle.meta.update(src_meta)
        handle.meta["val"] = val.new_empty((max_pages + 1,), dtype=torch.int64)
    with gm.graph.inserting_after(handle):
        stash_node = gm.graph.call_function(
            paged_stash_ops.stash_.default,
            args=(handle, node, plan.num_tokens_node, plan.slot.slot_id, prefetch),
        )
        stash_node.meta.update(src_meta)
        stash_node.meta["val"] = None

    # Backward: page the activation back in just before its first consumer.
    with gm.graph.inserting_before(plan.offload.first_bwd_consumer):
        reload_node = gm.graph.call_function(
            paged_stash_ops.reload.default,
            args=(
                handle,
                plan.num_tokens_node,
                plan.slot.slot_id,
                list(val.size()),
                val.dtype,
            ),
        )
        reload_node.meta.update(src_meta)
        reload_node.meta["val"] = val
        reload_node.meta["autograd_backward"] = True

        reload_wait = gm.graph.call_function(
            paged_stash_ops.wait_reload.default,
            args=(reload_node,),
        )
        reload_wait.meta.update(src_meta)
        reload_wait.meta["val"] = val
        reload_wait.meta["autograd_backward"] = True
    return stash_node, reload_wait


def _insert_group_start(gm: torch.fx.GraphModule, hidden: Node) -> Node | None:
    """Route the expert MLP's reads of the dispatch output ``hidden`` through a
    ``group_start`` placed just before the first of them.

    Returns the ``group_start`` node, or None if nothing in forward reads
    ``hidden``.
    """
    order = {n: i for i, n in enumerate(gm.graph.nodes)}
    readers = [
        user
        for user in hidden.users
        if not _is_backward_node(user)
        and user.target is not paged_stash_ops.stash_.default
    ]
    if not readers:
        return None
    with gm.graph.inserting_before(min(readers, key=order.__getitem__)):
        start = gm.graph.call_function(paged_stash_ops.group_start.default, (hidden,))
    start.meta.update(_meta_from(hidden))
    start.meta["val"] = hidden.meta["val"]
    for reader in readers:
        reader.replace_input_with(hidden, start)
    return start


def _dispatch_backward(dispatch: Node) -> Node | None:
    """HybridEP's backward of ``dispatch``: the ``combine_bwd`` that reads the
    gradient of the dispatch output, which is the expert MLP's input.

    Autograd gives the ops of a forward op's backward that op's ``seq_nr``.
    """
    seq_nr = dispatch.meta.get("seq_nr")
    if seq_nr is None:
        return None
    for node in dispatch.graph.nodes:
        if (
            node.target is torch.ops.hybridep.combine_bwd.default
            and _is_backward_node(node)
            and node.meta.get("seq_nr") == seq_nr
        ):
            return node
    return None


def _insert_reload_next(
    gm: torch.fx.GraphModule, hidden: Node, slot_id: int, prefetch: int
) -> Node | None:
    """Pass the gradient of the dispatch output ``hidden`` through a
    ``reload_next`` as soon as the expert MLP's backward has produced it, on its
    way to the dispatch's backward.

    Returns the ``reload_next`` node, or None if the dispatch's backward is not
    in the graph.
    """
    dispatch_backward = _dispatch_backward(hidden.args[0])
    if dispatch_backward is None or not isinstance(dispatch_backward.args[0], Node):
        return None
    grad = dispatch_backward.args[0]
    anchor = (
        gm.graph.inserting_after(grad)
        if _is_backward_node(grad)
        else gm.graph.inserting_before(dispatch_backward)
    )
    with anchor:
        node = gm.graph.call_function(
            paged_stash_ops.reload_next.default, (grad, slot_id, prefetch)
        )
    # The dispatch backward's metadata, not the gradient's: an accumulated
    # gradient is a bare ``add`` that carries no module annotation.
    node.meta.update(_meta_from(dispatch_backward))
    node.meta["val"] = grad.meta.get("val")
    dispatch_backward.replace_input_with(grad, node)
    return node


def apply_paged_stash_pass(
    gm: torch.fx.GraphModule,
    example_inputs: tuple | None = None,
    *,
    page_size: int = 64,
    prefetch_lookahead: int = 1,
) -> torch.fx.GraphModule:
    """Insert paged stash/reload ops for nodes tagged ``paged_stash``.

    Args:
        gm: Joint forward+backward graph module.
        example_inputs: Unused; required by the graph pass interface.
        page_size: Tokens per page. Must match the value the manager allocates
            buffers with, so it is written back onto the manager here.
        prefetch_lookahead: How many paged-layer schedule entries ahead of a
            backward its reloads are issued, across pipeline actions. 1 is
            Megatron's; 0 pops each activation's pages when backward first
            reads it.

    Returns:
        The transformed GraphModule.
    """
    tagged = [n for n in gm.graph.nodes if n.meta.get(PAGED_STASH_TAG)]
    if not tagged:
        return gm

    PagedStashManager.get_instance().page_size = page_size
    token_counts = find_layer_token_counts(gm)
    plans, node_to_index = _plan_stashes(gm, tagged, token_counts)
    commits = _stash_group_commits(plans, node_to_index)
    # Each layer's last plan, whose stash ends the group.
    last_plans = {plan.slot.layer_id: plan for plan in plans}
    last_stash: dict[int, Node] = {}

    stashed_bytes = 0
    replay_count = 0
    for plan in plans:
        layer_id = plan.slot.layer_id
        commit = commits[layer_id]
        assert node_to_index[commit] < node_to_index[plan.offload.first_bwd_consumer], (
            f"paged stash: layer {layer_id} launches its stashes at "
            f"{commit.name}, after {plan.offload.node.name}'s first backward "
            f"reader {plan.offload.first_bwd_consumer.name}; the joint graph "
            "must trace forward before backward."
        )
        last_stash[layer_id], reloaded = _insert_stash_ops(
            gm,
            plan,
            page_size=page_size,
            prefetch=prefetch_lookahead if plan is last_plans[layer_id] else 0,
            insert_after=last_stash.get(layer_id, commit),
        )
        if _redirect_backward_consumers(gm, plan.offload, reloaded, node_to_index):
            replay_count += 1

        val = plan.offload.node.meta["val"]
        stashed_bytes += _tensor_bytes(val)
        logger.debug(
            "Paged stash: slot %d <- %s (%s %s, %.2f MiB padded)",
            plan.slot.slot_id,
            plan.offload.node.name,
            val.dtype,
            tuple(val.shape),
            _tensor_bytes(val) / (1024 * 1024),
        )

    unanchored = []
    for layer_id, plan in last_plans.items():
        hidden = token_counts[layer_id].hidden_node
        _insert_group_start(gm, hidden)
        if prefetch_lookahead > 0 and (
            _insert_reload_next(gm, hidden, plan.slot.slot_id, prefetch_lookahead)
            is None
        ):
            unanchored.append(layer_id)
    if unanchored:
        logger.warning(
            "Paged stash: found no dispatch backward in layer(s) %s, so the "
            "reloads that would follow their backward are issued when backward "
            "first reads them instead.",
            unanchored,
        )

    _warn_unpaged_padded_activations(gm)
    gm.graph.lint()
    gm.recompile()
    logger.info(
        "Paged stash: paged %d activation(s) (%.2f MiB padded), %d with view "
        "replay; reloads issued %d schedule entr%s ahead",
        len(plans),
        stashed_bytes / (1024 * 1024),
        replay_count,
        prefetch_lookahead,
        "y" if prefetch_lookahead == 1 else "ies",
    )
    return gm


def _warn_unpaged_padded_activations(gm: torch.fx.GraphModule) -> None:
    """Warn about capacity-padded activations in the paged-stash region that
    backward still reads once the stashes are in place.

    Tagging sees the graph before remat, and remat can add backward readers
    afterwards: a region tensor the stash could not hold is left to SAC, and if
    SAC recomputes it, backward rebuilds it from inputs that then cross the
    boundary at their full padded size. Nothing else reports that, and the
    input can be far larger than the tensor that pulled it in.
    """
    token_counts = find_layer_token_counts(gm)
    unpaged = []
    for node in gm.graph.nodes:
        if PAGED_STASH_REGION not in node.meta.get("custom", {}):
            continue
        counts = token_counts.get(_get_layer_id(node))
        if counts is None or not _is_pageable(node, counts.padded):
            continue
        _, has_backward = _get_storage_chain(node, is_view=_is_stash_view)
        if has_backward:
            unpaged.append(node)
    if unpaged:
        logger.warning(
            "Paged stash: %d capacity-padded activation(s) in the paged-stash "
            "region cross into backward unpaged (%.2f MiB padded): %s. Backward "
            "rebuilds something from them that the stash could not hold, "
            "usually a tensor whose shape does not show its token rows; "
            "storing that tensor token-major lets the stash take it instead.",
            len(unpaged),
            sum(_tensor_bytes(node.meta["val"]) for node in unpaged) / (1024 * 1024),
            ", ".join(node.name for node in unpaged),
        )


# ===========================================================================
# Size the paged stash from the GraphPP pipeline schedule
# ===========================================================================
#
# GraphPP's runtime schedule is a concrete list of ``_Action`` s per rank, so
# ``replay_schedule_page_peak`` can accumulate each stage's stash footprint over
# it directly -- the quantity Megatron reconstructs from its virtual-pipeline
# order -- and ``paged_layer_schedule`` reads Megatron's ``_pp_schedule`` off it.


def schedule_page_peak(
    schedule: _PipelineScheduleRuntime,
    stage_footprint: dict[int, PageCounts],
) -> PageCounts:
    """Peak stashed pages across every rank of ``schedule``.

    Taking the max over ranks rather than only the local rank keeps buffer sizes
    uniform, which matters because the overflow fallback is collective: one rank
    overflowing forces every rank to rerun.
    """
    try:
        pipeline_order = schedule.pipeline_order_with_comms
    except AttributeError as exc:
        raise ValueError(
            "Paged stash sizing requires a runtime PP schedule exposing "
            "pipeline_order_with_comms."
        ) from exc
    if not pipeline_order:
        return {}

    combined: PageCounts = {}
    for actions in pipeline_order.values():
        for key, pages in replay_schedule_page_peak(actions, stage_footprint).items():
            combined[key] = max(combined.get(key, 0), pages)
    return combined


def apply_graph_pp_schedule(schedule: _PipelineScheduleRuntime) -> None:
    """Install this rank's GraphPP actions on the manager, which reads the
    paged-layer schedule off them (see :func:`paged_layer_schedule`).

    Sizing takes the max over ranks so buffers stay uniform; the schedule is the
    opposite -- it must be *this* rank's action list, because what runs after a
    forward or backward is a property of the local schedule. A rank whose
    action list cannot be read gets an empty schedule, which skips nothing and
    prefetches nothing: correct, just not optimized.
    """
    manager = PagedStashManager.get_instance()
    if not manager.enabled:
        return
    # ``pipeline_order`` is keyed by pipeline-group rank, which differs from the
    # global rank whenever another parallelism (e.g. data parallel) sits
    # alongside PP.
    pipeline_order = getattr(schedule, "pipeline_order_with_comms", None) or {}
    manager.set_pipeline_actions(list(pipeline_order.get(schedule.rank, ())))


def apply_graph_pp_sizing(schedule: _PipelineScheduleRuntime) -> PageCounts:
    """Install the schedule-derived page peak on the manager.

    Called before every step so the estimate tracks the per-slot footprints as
    they are measured. It is a no-op until the measure step has populated those
    footprints, which is also when buffer allocation first needs it.
    """
    manager = PagedStashManager.get_instance()
    if not manager.enabled:
        return {}
    footprint = manager.stage_page_footprint()
    if not footprint:
        return {}

    estimate = schedule_page_peak(schedule, footprint)
    if estimate and estimate != manager.schedule_estimate:
        generations = max(
            (
                max_live_stash_generations(actions)
                for actions in schedule.pipeline_order_with_comms.values()
            ),
            default=1,
        )
        logger.info(
            "Paged stash: GraphPP schedule holds up to %d stashed microbatch(es) "
            "per rank; peak pages %s",
            generations,
            {f"{dtype}/{hidden}": pages for (dtype, hidden), pages in estimate.items()},
        )
    manager.schedule_estimate = estimate
    return estimate


def apply_graph_pp(schedule: _PipelineScheduleRuntime) -> None:
    """Feed the GraphPP schedule into buffer sizing and into the paged-layer
    schedule that skips stashes and prefetches reloads.

    This is the GraphTrainer replacement for Megatron's virtual-pipeline
    bookkeeping: instead of recording the schedule while it runs, read it off
    the runtime PP schedule. Called before every step, so the first allocation
    after the measure step already accounts for it.

    ``schedule`` is ``GraphRuntime.pipeline_liveness_schedule``: the stash
    lifetimes are read in PyTorch's action types, and the schedule GraphRuntime
    executes renames its backward actions.
    """
    apply_graph_pp_sizing(schedule)
    apply_graph_pp_schedule(schedule)


# ===========================================================================
# Memory policy registration and trainer setup
# ===========================================================================


@register_memory_policy("sac_and_paged_stash")
def _sac_and_paged_stash_memory_policy_pass(
    gm: torch.fx.GraphModule,
    *,
    config: GraphTrainer.Config,
) -> torch.fx.GraphModule:
    """SAC + MoE paged stash: default SAC everywhere, except that the paged
    stash decides the activations it can hold in its annotated region."""
    _default_memory_policy_pass(gm, config=config)
    tag_paged_stash_activations(gm)
    return gm


def build_paged_stash_runner(
    paged_stash_config: PagedStashConfig,
    *,
    optimizers: OptimizersContainer,
    model_parts: Sequence[nn.Module],
    device: torch.device,
    pp_enabled: bool,
    reset_cuda_graphs: Callable[[], None],
) -> PagedStashRunner:
    """Enable MoE paged stashing and return its step runner.

    Called only under ``compile.memory_policy = "sac_and_paged_stash"``. Buffers
    are not created here: their size comes from the first measured step. What
    is created up front is everything that must outlive a step -- the manager
    and its sizing configuration -- matching Megatron, which also defers
    ``allocate_stash_buffers`` to the end of its capture iteration.
    """
    # Annotate the region only when the feature is on, so a run without paged
    # stashing traces exactly the graph it did before. It must happen before
    # the first trace, which happens on the first step.
    for model in model_parts:
        annotate_paged_stash_region(model, paged_stash_config.module_fqn)

    manager = PagedStashManager.get_instance()
    manager.enabled = True
    manager.device = device
    manager.page_size = paged_stash_config.page_size
    manager.buffer_size_factor_cuda = paged_stash_config.buffer_size_factor_cuda
    manager.buffer_size_factor_cpu = paged_stash_config.buffer_size_factor_cpu
    manager.skip_immediate_backward = paged_stash_config.skip_immediate_backward
    # Without pipeline parallelism the schedule is the joint graph's own --
    # every forward, then every backward -- until the GraphPP one is installed.
    manager.set_pipeline_actions([] if pp_enabled else None)
    logger.info(
        "Paged stash enabled: page_size=%d, factor_cuda=%.2f, factor_cpu=%.2f, "
        "overflow_check=%s",
        manager.page_size,
        manager.buffer_size_factor_cuda,
        manager.buffer_size_factor_cpu,
        paged_stash_config.overflow_check,
    )
    return PagedStashRunner(
        optimizers,
        expert_biases=moe_expert_biases(model_parts),
        # Token drops in the HybridEP dispatcher fold into the same verdict as
        # stash overflow -- the ``overbudget`` slot of Megatron's
        # check_moe_overflow.
        overbudget_probes=[hybridep_overbudget_probe()],
        overflow_check=paged_stash_config.overflow_check,
        reset_cuda_graphs=reset_cuda_graphs,
        config_path="compile.paged_stash",
        disable_hint="compile.memory_policy = 'default'",
    )


__all__ = [
    "PAGED_STASH_REGION",
    "PAGED_STASH_TAG",
    "TokenCounts",
    "annotate_paged_stash_region",
    "apply_graph_pp",
    "apply_graph_pp_schedule",
    "apply_graph_pp_sizing",
    "apply_paged_stash_pass",
    "build_paged_stash_runner",
    "find_layer_token_counts",
    "reload",
    "schedule_page_peak",
    "stash_",
    "tag_paged_stash_activations",
]

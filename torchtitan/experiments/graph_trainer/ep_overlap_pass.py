# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Schedule already chunked EP token-exchange regions.

Contract
========
This pass is intentionally a scheduler only.  It consumes a graph that has
already been chunked by either eager chunking or ``ep_chunk_pass`` and must not
change tensor values, live-in/live-out materialization, or provenance.  The only
semantic input it relies on is chunk-body metadata collected by
``collect_chunked_regions``.

For each selected forward/backward region:

* exactly two chunk bodies, chunk 0 and chunk 1, must be present;
* true EP scheduling markers are
  ``_c10d_functional.all_to_all_single.default`` launches inside the selected
  chunk body;
* ``custom[_EP_TOKEN_EXCHANGE]`` marks true token-exchange all-to-all launches;
  waits may inherit the annotation from traceback and are normalized to
  ``custom[_EP_TOKEN_EXCHANGE_WAIT]``;
* marker counts and labels must match across chunks;
* forward emits marker pairs in chunk order 0 then 1, backward emits 1 then 0;
* MoE-root chunking pairs both chunks' first-marker setup before launching the
  first marker pair; wider transformer-root chunking keeps the regular
  wait-gated per-chunk closure schedule for every marker;
* token-count sync CPU copies, when present in the first marker closure, are
  launched before their CPU scalar/list consumers;
* after each marker pair, ready non-collective body work is emitted as filler
  before advancing to the next marker pair;
* all graph nodes remain in the sorted graph exactly once and the final graph
  must lint.

The same contract covers eager and graph chunking.  If a chunked region violates
the contract, the pass errors rather than producing a silent schedule change.

Pseudo-code
===========
1. Collect chunked regions and build node -> chunk-owner lookup from shared EP
   pass metadata.
2. For each region, collect true token-exchange all-to-all markers per chunk
   and normalize wait annotations inherited from traceback.
3. Validate both chunks have the same marker signature, then build dependency
   closures needed to launch each marker.
4. Emit wait-gated phases: marker pair in chunk order, ready filler work that
   does not need token-exchange waits, then final tail work with waits allowed.
5. Apply the requested region phases through a stable topological sort, lint,
   recompile, and validate that phase order materialized.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.fx as fx
from torch._dynamo.graph_deduplication import _stable_topological_sort
from torch.utils._ordered_set import OrderedSet

from torchtitan.experiments.graph_trainer.common_utils import (
    _EP_TOKEN_COUNT_SYNC,
    _EP_TOKEN_EXCHANGE,
    _EP_TOKEN_EXCHANGE_WAIT,
)
from torchtitan.experiments.graph_trainer.ep_pass_utils import (
    ChunkBody,
    ChunkedRegion,
    ChunkOwner,
    collect_chunked_regions,
    is_c10d_functional_node,
    ordered_nodes,
)
from torchtitan.tools.logging import logger


_GRAPH_BOUNDARY_OPS = {"placeholder", "get_attr"}
_EP_PHASES = {"dispatch", "combine"}


# Step 0: Small metadata helpers and local scheduling records.


@dataclass(frozen=True)
class _TokenExchange:
    label: str
    launch: fx.Node


@dataclass(frozen=True)
class _ScheduledRegion:
    region: ChunkedRegion
    phases: tuple[tuple[fx.Node, ...], ...]


@dataclass(frozen=True)
class _SyncClosureParts:
    pre_copy: tuple[fx.Node, ...]
    copies: tuple[fx.Node, ...]
    post_copy: tuple[fx.Node, ...]
    launches: tuple[fx.Node, ...]


def _custom_meta(node: fx.Node) -> dict[str, Any]:
    """Return mutable custom metadata when present, otherwise an empty dict."""
    custom = node.meta.get("custom")
    return custom if isinstance(custom, dict) else {}


def _ep_label(node: fx.Node) -> str:
    """Return the optional EP phase label for logs/wait metadata."""
    phase = _custom_meta(node).get(_EP_TOKEN_EXCHANGE)
    return phase if phase in _EP_PHASES else "all_to_all"


def _is_token_exchange_launch(node: fx.Node) -> bool:
    """Return whether a node is a token-exchange scheduling marker.

    Only all-to-all ops annotated with ``EP_token_exchange`` (dispatch or
    combine) are scheduling markers. The counts all-to-all (which only has
    ``EP: dispatch`` without ``EP_token_exchange``) is NOT a marker; it
    stays in the closure with shared expert + router compute so that
    compute is available as filler during the data dispatch window.
    """
    return (
        node.op == "call_function"
        and node.target == torch.ops._c10d_functional.all_to_all_single.default
        and _custom_meta(node).get(_EP_TOKEN_EXCHANGE) in _EP_PHASES
    )


def _is_c10d_functional_node(node: fx.Node) -> bool:
    """Return whether a node is a distributed functional op."""
    return is_c10d_functional_node(node)


def _is_token_count_sync_copy(node: fx.Node) -> bool:
    return (
        node.op == "call_function"
        and node.target == torch.ops.aten._to_copy.default
        and _custom_meta(node).get(_EP_TOKEN_COUNT_SYNC) == "dispatch"
    )


def _is_cpu_destination_copy(node: fx.Node) -> bool:
    device = node.kwargs.get("device")
    if device is not None:
        return torch.device(device).type == "cpu"

    val = node.meta.get("val")
    val_device = getattr(val, "device", None)
    return val_device is not None and val_device.type == "cpu"


def _set_copy_non_blocking(node: fx.Node, non_blocking: bool) -> None:
    kwargs = dict(node.kwargs)
    kwargs["non_blocking"] = non_blocking
    node.kwargs = kwargs


def _same_region_owner(
    node: fx.Node,
    *,
    owner_by_node: dict[fx.Node, ChunkOwner],
    root_fqn: str,
    is_backward: bool,
) -> ChunkOwner | None:
    """Return chunk ownership only when it belongs to the same selected region."""
    owner = owner_by_node.get(node)
    if (
        owner is not None
        and owner.root_fqn == root_fqn
        and owner.is_backward == is_backward
    ):
        return owner
    return None


def _is_wait_for_token_exchange(node: fx.Node, node_set: set[fx.Node]) -> bool:
    return (
        node.op == "call_function"
        and node.target == torch.ops._c10d_functional.wait_tensor.default
        and node.all_input_nodes
        and node.all_input_nodes[0] in node_set
        and _is_token_exchange_launch(node.all_input_nodes[0])
    )


def _collect_token_exchanges(
    body: ChunkBody,
    *,
    order: dict[fx.Node, int],
) -> tuple[_TokenExchange, ...]:
    """Step 2: collect true token-exchange launches for one chunk body."""
    node_set = set(body.nodes)
    exchanges: list[_TokenExchange] = []
    for node in body.nodes:
        if not _is_token_exchange_launch(node):
            phase = _custom_meta(node).get(_EP_TOKEN_EXCHANGE)
            if phase is None:
                continue
            if _is_wait_for_token_exchange(node, node_set):
                custom = dict(_custom_meta(node))
                custom.pop(_EP_TOKEN_EXCHANGE, None)
                custom[_EP_TOKEN_EXCHANGE_WAIT] = phase
                node.meta["custom"] = custom
                continue
            raise ValueError(
                "ep_overlap found EP token-exchange metadata on non-marker "
                f"node {node.name} ({node.target}). Only all_to_all launches "
                "and their wait_tensor nodes may carry this annotation."
            )
            continue

        label = _ep_label(node)
        waits = [
            user
            for user in node.users
            if user in node_set
            and user.op == "call_function"
            and user.target == torch.ops._c10d_functional.wait_tensor.default
        ]
        if len(waits) != 1:
            raise ValueError(
                f"ep_overlap expected one token-exchange wait for {node.name}, "
                f"found {len(waits)}."
            )
        wait = waits[0]
        if order[wait] <= order[node]:
            raise ValueError(
                "ep_overlap expected token-exchange wait to appear after its "
                f"launch, found launch={node.name} wait={wait.name} for "
                f"{body.owner}."
            )
        custom = dict(_custom_meta(wait))
        custom.pop(_EP_TOKEN_EXCHANGE, None)
        custom[_EP_TOKEN_EXCHANGE_WAIT] = label
        wait.meta["custom"] = custom
        exchanges.append(_TokenExchange(label=label, launch=node))
    return tuple(exchanges)


def _exchange_signature(exchanges: tuple[_TokenExchange, ...]) -> tuple[str, ...]:
    """Return the semantic marker sequence used to match chunk pairs."""
    return ("all_to_all",) * len(exchanges)


def _exchange_labels(exchanges: tuple[_TokenExchange, ...]) -> tuple[str, ...]:
    """Return optional marker labels for diagnostics."""
    return tuple(exchange.label for exchange in exchanges)


def _validate_exchange_labels(
    root: str,
    direction: str,
    exchanges_by_chunk: dict[int, tuple[_TokenExchange, ...]],
) -> None:
    labels0 = _exchange_labels(exchanges_by_chunk[0])
    labels1 = _exchange_labels(exchanges_by_chunk[1])
    if labels0 != labels1:
        raise ValueError(
            f"ep_overlap expected matching token-exchange labels for "
            f"{root!r} ({direction}), found chunk0={labels0} chunk1={labels1}."
        )


# Step 3: Build marker dependency closures and identify ready filler work.


def _hidden_body_deps(
    node: fx.Node,
    *,
    owner_by_node: dict[fx.Node, ChunkOwner],
    root_fqn: str,
    is_backward: bool,
) -> tuple[fx.Node, ...]:
    """Find same-region body deps behind unowned graph plumbing."""
    deps: list[fx.Node] = []
    seen: set[fx.Node] = set()
    stack = list(node.all_input_nodes)
    while stack:
        dep = stack.pop()
        if dep in seen:
            continue
        seen.add(dep)
        owner = _same_region_owner(
            dep,
            owner_by_node=owner_by_node,
            root_fqn=root_fqn,
            is_backward=is_backward,
        )
        if owner is not None:
            deps.append(dep)
        elif owner_by_node.get(dep) is None and dep.op not in _GRAPH_BOUNDARY_OPS:
            stack.extend(dep.all_input_nodes)
    return tuple(deps)


def _body_deps(
    node: fx.Node,
    *,
    body: ChunkBody,
    owner_by_node: dict[fx.Node, ChunkOwner],
) -> tuple[fx.Node, ...]:
    """Return same-region body deps, including deps behind unowned plumbing."""
    deps: list[fx.Node] = []
    for dep in node.all_input_nodes:
        owner = _same_region_owner(
            dep,
            owner_by_node=owner_by_node,
            root_fqn=body.owner.root_fqn,
            is_backward=body.owner.is_backward,
        )
        if owner is not None:
            deps.append(dep)
        elif owner_by_node.get(dep) is None and dep.op not in _GRAPH_BOUNDARY_OPS:
            deps.extend(
                _hidden_body_deps(
                    dep,
                    owner_by_node=owner_by_node,
                    root_fqn=body.owner.root_fqn,
                    is_backward=body.owner.is_backward,
                )
            )
    return tuple(dict.fromkeys(deps))


def _body_ancestors(
    nodes: tuple[fx.Node, ...],
    *,
    body: ChunkBody,
    owner_by_node: dict[fx.Node, ChunkOwner],
    node_set: set[fx.Node],
) -> set[fx.Node]:
    """Return same-body ancestors needed before the requested body nodes."""
    ancestors: set[fx.Node] = set()
    stack = [
        dep
        for node in nodes
        for dep in _body_deps(node, body=body, owner_by_node=owner_by_node)
        if dep in node_set
    ]
    while stack:
        dep = stack.pop()
        if dep in ancestors:
            continue
        ancestors.add(dep)
        stack.extend(
            next_dep
            for next_dep in _body_deps(dep, body=body, owner_by_node=owner_by_node)
            if next_dep in node_set
        )
    return ancestors


def _split_token_count_sync_closure(
    closure: tuple[fx.Node, ...],
    *,
    body: ChunkBody,
    launch_nodes: set[fx.Node],
    owner_by_node: dict[fx.Node, ChunkOwner],
) -> _SyncClosureParts | None:
    """Split the first token-exchange closure around sync D2H copy launches."""
    copies = tuple(node for node in closure if _is_token_count_sync_copy(node))
    if not copies:
        return None
    direction = "backward" if body.owner.is_backward else "forward"
    if len(copies) != 2:
        raise ValueError(
            "ep_overlap expected exactly two token-count sync CPU copies for "
            f"{body.owner.root_fqn!r} chunk {body.owner.chunk_id} ({direction}), "
            f"found {len(copies)}."
        )
    non_cpu_copies = [
        copy.name for copy in copies if not _is_cpu_destination_copy(copy)
    ]
    if non_cpu_copies:
        raise ValueError(
            "ep_overlap token-count sync optimization only supports CPU "
            f"_to_copy destinations; ambiguous/non-CPU copies for "
            f"{body.owner.root_fqn!r} chunk {body.owner.chunk_id} ({direction}): "
            f"{', '.join(non_cpu_copies)}."
        )

    closure_set = set(closure)
    copy_set = set(copies)
    pre_copy_set = _body_ancestors(
        copies,
        body=body,
        owner_by_node=owner_by_node,
        node_set=closure_set,
    )
    launches = tuple(node for node in closure if node in launch_nodes)
    blocked = pre_copy_set | copy_set | set(launches)
    return _SyncClosureParts(
        pre_copy=tuple(node for node in closure if node in pre_copy_set),
        copies=copies,
        post_copy=tuple(node for node in closure if node not in blocked),
        launches=launches,
    )


def _validate_token_count_sync_copies(
    body: ChunkBody,
    *,
    first_closure: tuple[fx.Node, ...],
) -> None:
    first_closure_set = set(first_closure)
    stray_copies = [
        node.name
        for node in body.nodes
        if _is_token_count_sync_copy(node) and node not in first_closure_set
    ]
    if stray_copies:
        direction = "backward" if body.owner.is_backward else "forward"
        raise ValueError(
            "ep_overlap only schedules token-count sync CPU copies in the first "
            f"token-exchange closure for {body.owner.root_fqn!r} chunk "
            f"{body.owner.chunk_id} ({direction}); found outside the first "
            f"closure: {', '.join(stray_copies)}."
        )


def _marker_closure(
    launch: fx.Node,
    *,
    body: ChunkBody,
    order: dict[fx.Node, int],
    owner_by_node: dict[fx.Node, ChunkOwner],
    exchange_index: int,
    exchange_indices: dict[fx.Node, int],
) -> tuple[fx.Node, ...]:
    """Return the body nodes required to launch one token exchange."""
    closure: list[fx.Node] = []
    visiting: set[fx.Node] = set()
    visited: set[fx.Node] = set()

    def visit(node: fx.Node, *, allow_peer_chunk: bool = False) -> None:
        if node in visited:
            return
        if node in visiting:
            raise ValueError(
                "ep_overlap found a cycle while building marker closure for "
                f"{launch.name} in {body.owner}."
            )

        owner = _same_region_owner(
            node,
            owner_by_node=owner_by_node,
            root_fqn=body.owner.root_fqn,
            is_backward=body.owner.is_backward,
        )
        if owner is None:
            for dep in sorted(
                _hidden_body_deps(
                    node,
                    owner_by_node=owner_by_node,
                    root_fqn=body.owner.root_fqn,
                    is_backward=body.owner.is_backward,
                ),
                key=order.__getitem__,
            ):
                visit(dep, allow_peer_chunk=True)
            return

        if owner.chunk_id != body.owner.chunk_id and not allow_peer_chunk:
            raise ValueError(
                "ep_overlap cannot schedule a token exchange whose dependency "
                f"{node.name} belongs to peer chunk {owner.chunk_id} of "
                f"{body.owner.root_fqn!r}."
            )
        if owner.chunk_id == body.owner.chunk_id:
            dep_exchange_idx = exchange_indices.get(node)
            if dep_exchange_idx is not None and dep_exchange_idx > exchange_index:
                raise ValueError(
                    "ep_overlap token-exchange order is not topologically valid: "
                    f"launch {launch.name} for {body.owner} needs later "
                    f"same-chunk launch {node.name}."
                )

        visiting.add(node)
        for dep in sorted(
            _body_deps(node, body=body, owner_by_node=owner_by_node),
            key=order.__getitem__,
        ):
            dep_owner = _same_region_owner(
                dep,
                owner_by_node=owner_by_node,
                root_fqn=body.owner.root_fqn,
                is_backward=body.owner.is_backward,
            )
            visit(
                dep,
                allow_peer_chunk=allow_peer_chunk
                or (
                    dep_owner is not None and dep_owner.chunk_id != body.owner.chunk_id
                ),
            )
        visiting.remove(node)
        visited.add(node)
        closure.append(node)

    visit(launch)
    return tuple(sorted(closure, key=order.__getitem__))


def _ready_nodes(
    *,
    candidates_by_chunk: dict[int, set[fx.Node]],
    emitted: set[fx.Node],
    region: ChunkedRegion,
    chunk_order: tuple[int, ...],
    order: dict[fx.Node, int],
    owner_by_node: dict[fx.Node, ChunkOwner],
    include_waits: bool,
) -> tuple[fx.Node, ...]:
    """Return currently schedulable body nodes from candidate filler sets."""
    ready: list[fx.Node] = []
    ready_set: set[fx.Node] = set()
    for chunk_id in chunk_order:
        body = region.bodies_by_chunk[chunk_id]
        candidates = sorted(
            candidates_by_chunk.get(chunk_id, set()) - emitted,
            key=order.__getitem__,
        )
        for node in candidates:
            if not include_waits and _is_c10d_functional_node(node):
                continue
            deps = _body_deps(node, body=body, owner_by_node=owner_by_node)
            if all(dep in emitted for dep in deps) and node not in ready_set:
                ready.append(node)
                ready_set.add(node)
    return tuple(ready)


def _append_ready_blocks(
    blocks: list[tuple[fx.Node, ...]],
    emitted: set[fx.Node],
    *,
    candidates_by_chunk: dict[int, set[fx.Node]],
    region: ChunkedRegion,
    chunk_order: tuple[int, ...],
    order: dict[fx.Node, int],
    owner_by_node: dict[fx.Node, ChunkOwner],
    include_waits: bool,
) -> bool:
    """Append ready filler/tail blocks until the candidate frontier stalls."""
    made_progress = False
    while True:
        ready = tuple(
            node
            for node in _ready_nodes(
                candidates_by_chunk=candidates_by_chunk,
                emitted=emitted,
                region=region,
                chunk_order=chunk_order,
                order=order,
                owner_by_node=owner_by_node,
                include_waits=include_waits,
            )
            if node not in emitted
        )
        if not ready:
            return made_progress
        blocks.append(ready)
        emitted.update(ready)
        made_progress = True


def _build_region_phases(
    *,
    region: ChunkedRegion,
    exchanges_by_chunk: dict[int, tuple[_TokenExchange, ...]],
    order: dict[fx.Node, int],
    owner_by_node: dict[fx.Node, ChunkOwner],
    pair_first_token_exchange: bool,
    rewrite_token_count_sync_copies: bool,
) -> tuple[tuple[fx.Node, ...], ...]:
    """Step 4: construct wait-gated phases for one scheduled region."""
    chunk_order = (1, 0) if region.is_backward else (0, 1)
    exchange_indices = {
        chunk_id: {
            exchange.launch: idx
            for idx, exchange in enumerate(exchanges_by_chunk[chunk_id])
        }
        for chunk_id in chunk_order
    }
    closures = {
        chunk_id: tuple(
            _marker_closure(
                exchange.launch,
                body=region.bodies_by_chunk[chunk_id],
                order=order,
                owner_by_node=owner_by_node,
                exchange_index=idx,
                exchange_indices=exchange_indices[chunk_id],
            )
            for idx, exchange in enumerate(exchanges_by_chunk[chunk_id])
        )
        for chunk_id in chunk_order
    }
    closure_nodes = {
        chunk_id: {node for closure in chunk_closures for node in closure}
        for chunk_id, chunk_closures in closures.items()
    }
    filler = {
        chunk_id: set(region.bodies_by_chunk[chunk_id].nodes) - closure_nodes[chunk_id]
        for chunk_id in chunk_order
    }
    launch_nodes = {
        exchange.launch
        for chunk_exchanges in exchanges_by_chunk.values()
        for exchange in chunk_exchanges
    }

    def future_candidates(exchange_idx: int) -> dict[int, set[fx.Node]]:
        return {
            chunk_id: {
                node
                for closure in closures[chunk_id][exchange_idx + 1 :]
                for node in closure
                if node not in launch_nodes
            }
            | filler[chunk_id]
            for chunk_id in chunk_order
        }

    blocks: list[tuple[fx.Node, ...]] = []
    emitted: set[fx.Node] = set()

    def append_pending(nodes: tuple[fx.Node, ...]) -> None:
        # A marker closure may legally pull peer-chunk dependencies through
        # hidden shape plumbing. Deduplicate here so one FX node cannot create
        # contradictory phase-order edges when it is ready from both chunks.
        seen: set[fx.Node] = set()
        pending_list: list[fx.Node] = []
        for node in nodes:
            if node in emitted or node in seen:
                continue
            seen.add(node)
            pending_list.append(node)
        pending = tuple(pending_list)
        if pending:
            blocks.append(pending)
            emitted.update(pending)

    def rewrite_sync_copies(copies: tuple[fx.Node, ...]) -> None:
        if not rewrite_token_count_sync_copies:
            return
        for idx, copy in enumerate(copies):
            _set_copy_non_blocking(copy, idx + 1 != len(copies))

    sync_parts_by_chunk: dict[int, _SyncClosureParts | None] = {}
    if exchanges_by_chunk[0]:
        for chunk_id in chunk_order:
            body = region.bodies_by_chunk[chunk_id]
            first_closure = closures[chunk_id][0]
            _validate_token_count_sync_copies(body, first_closure=first_closure)
            sync_parts_by_chunk[chunk_id] = _split_token_count_sync_closure(
                first_closure,
                body=body,
                launch_nodes=launch_nodes,
                owner_by_node=owner_by_node,
            )
        if any(parts is not None for parts in sync_parts_by_chunk.values()) and not all(
            parts is not None for parts in sync_parts_by_chunk.values()
        ):
            direction = "backward" if region.is_backward else "forward"
            raise ValueError(
                "ep_overlap expected token-count sync CPU copies for both chunks "
                f"of {region.root_fqn!r} ({direction}) when optimizing sync "
                "copy scheduling."
            )

    for exchange_idx in range(len(exchanges_by_chunk[0])):
        if exchange_idx == 0 and pair_first_token_exchange:
            paired_sync_parts = tuple(
                sync_parts_by_chunk[chunk_id] for chunk_id in chunk_order
            )
            if all(parts is not None for parts in paired_sync_parts):
                # Startup: issue both chunks' token-count D2H copies before any
                # CPU split-size reads, so only the final copy has to block.
                for parts in paired_sync_parts:
                    assert parts is not None
                    append_pending(parts.pre_copy)
                paired_copies = tuple(
                    copy
                    for parts in paired_sync_parts
                    for copy in (parts.copies if parts is not None else ())
                )
                rewrite_sync_copies(paired_copies)
                append_pending(paired_copies)
                for parts in paired_sync_parts:
                    assert parts is not None
                    append_pending(parts.post_copy)
                for parts in paired_sync_parts:
                    assert parts is not None
                    append_pending(parts.launches)
            else:
                # Startup: emit both chunks up to the first token exchange before
                # launching either chunk's exchange. This keeps count/sync setup
                # together in forward and keeps backward combine launches paired.
                for chunk_id in chunk_order:
                    closure = closures[chunk_id][exchange_idx]
                    deps = tuple(
                        node
                        for node in closure
                        if node not in emitted and node not in launch_nodes
                    )
                    append_pending(deps)
                for chunk_id in chunk_order:
                    closure = closures[chunk_id][exchange_idx]
                    launches = tuple(
                        node
                        for node in closure
                        if node not in emitted and node in launch_nodes
                    )
                    append_pending(launches)
        else:
            # Regular wait-gated scheduling: emit each chunk's full closure
            # sequentially. Later closures may intentionally include waits
            # whose users are the next token-exchange launch.
            for chunk_id in chunk_order:
                sync_parts = (
                    sync_parts_by_chunk.get(chunk_id) if exchange_idx == 0 else None
                )
                if sync_parts is not None:
                    rewrite_sync_copies(sync_parts.copies)
                    append_pending(sync_parts.pre_copy)
                    append_pending(sync_parts.copies)
                    append_pending(sync_parts.post_copy)
                    append_pending(sync_parts.launches)
                else:
                    closure = closures[chunk_id][exchange_idx]
                    append_pending(
                        tuple(node for node in closure if node not in emitted)
                    )
        _append_ready_blocks(
            blocks,
            emitted,
            candidates_by_chunk=future_candidates(exchange_idx),
            region=region,
            chunk_order=chunk_order,
            order=order,
            owner_by_node=owner_by_node,
            include_waits=False,
        )

    remaining = {
        chunk_id: set(region.bodies_by_chunk[chunk_id].nodes) - emitted
        for chunk_id in chunk_order
    }
    made_progress = True
    while made_progress:
        made_progress = False
        for chunk_id in chunk_order:
            made_progress |= _append_ready_blocks(
                blocks,
                emitted,
                candidates_by_chunk={chunk_id: remaining[chunk_id]},
                region=region,
                chunk_order=(chunk_id,),
                order=order,
                owner_by_node=owner_by_node,
                include_waits=True,
            )

    missing = [
        node
        for chunk_id in chunk_order
        for node in region.bodies_by_chunk[chunk_id].nodes
        if node not in emitted
    ]
    if missing:
        direction = "backward" if region.is_backward else "forward"
        raise ValueError(
            f"ep_overlap could not schedule all body nodes for {region.root_fqn!r} "
            f"({direction}); remaining: {', '.join(n.name for n in missing[:8])}."
        )
    logger.debug(
        "ep_overlap phases: root=%s direction=%s chunk_order=%s markers=%d "
        "phase_sizes=%s",
        region.root_fqn,
        "backward" if region.is_backward else "forward",
        chunk_order,
        len(exchanges_by_chunk[0]),
        [len(block) for block in blocks],
    )
    return tuple(blocks)


def _plan_region(
    region: ChunkedRegion,
    *,
    order: dict[fx.Node, int],
    owner_by_node: dict[fx.Node, ChunkOwner],
    pair_first_token_exchange: bool,
    rewrite_token_count_sync_copies: bool,
) -> _ScheduledRegion | None:
    """Steps 2-4: validate one chunked region and build its schedule phases."""
    root = region.root_fqn
    direction = "backward" if region.is_backward else "forward"
    if set(region.bodies_by_chunk) != {0, 1}:
        raise ValueError(
            f"ep_overlap expected both chunks for {root!r} ({direction}), "
            f"found {sorted(region.bodies_by_chunk)}."
        )

    exchanges_by_chunk = {
        chunk_id: _collect_token_exchanges(
            region.bodies_by_chunk[chunk_id],
            order=order,
        )
        for chunk_id in (0, 1)
    }
    if not exchanges_by_chunk[0] and not exchanges_by_chunk[1]:
        logger.debug(
            "ep_overlap skipped region without token exchanges: root=%s direction=%s",
            root,
            direction,
        )
        return None
    if not exchanges_by_chunk[0] or not exchanges_by_chunk[1]:
        raise ValueError(
            f"ep_overlap found EP token exchanges for only one chunk of "
            f"{root!r} ({direction})."
        )
    if _exchange_signature(exchanges_by_chunk[0]) != _exchange_signature(
        exchanges_by_chunk[1]
    ):
        raise ValueError(
            f"ep_overlap expected matching EP all-to-all counts for "
            f"{root!r} ({direction}), found "
            f"chunk0={_exchange_signature(exchanges_by_chunk[0])} "
            f"chunk1={_exchange_signature(exchanges_by_chunk[1])}."
        )
    _validate_exchange_labels(root, direction, exchanges_by_chunk)
    logger.debug(
        "ep_overlap planned region: root=%s direction=%s body_sizes=(%d,%d) "
        "marker_count=%d marker_labels=(%s,%s)",
        root,
        direction,
        len(region.bodies_by_chunk[0].nodes),
        len(region.bodies_by_chunk[1].nodes),
        len(exchanges_by_chunk[0]),
        _exchange_labels(exchanges_by_chunk[0]),
        _exchange_labels(exchanges_by_chunk[1]),
    )

    phases = _build_region_phases(
        region=region,
        exchanges_by_chunk=exchanges_by_chunk,
        order=order,
        owner_by_node=owner_by_node,
        pair_first_token_exchange=pair_first_token_exchange,
        rewrite_token_count_sync_copies=rewrite_token_count_sync_copies,
    )
    return _ScheduledRegion(region=region, phases=phases) if phases else None


def _phase_order_deps(
    scheduled_regions: list[_ScheduledRegion],
) -> dict[fx.Node, OrderedSet[fx.Node]]:
    """Return extra ordering deps that preserve each region's phase schedule."""
    deps: dict[fx.Node, OrderedSet[fx.Node]] = {}

    def add_dep(node: fx.Node, dep: fx.Node) -> None:
        if node is dep:
            return
        deps.setdefault(node, OrderedSet()).add(dep)

    for region in scheduled_regions:
        previous_phase_tail: fx.Node | None = None
        for phase in region.phases:
            previous_node: fx.Node | None = None
            for node in phase:
                if previous_node is not None:
                    add_dep(node, previous_node)
                elif previous_phase_tail is not None:
                    add_dep(node, previous_phase_tail)
                previous_node = node
            if previous_node is not None:
                previous_phase_tail = previous_node
    return deps


def _apply_schedule(
    gm: fx.GraphModule,
    scheduled_regions: list[_ScheduledRegion],
) -> None:
    """Step 5: apply scheduled order, lint, recompile, and validate phases."""
    _stable_topological_sort(gm.graph, _phase_order_deps(scheduled_regions))
    gm.graph.lint()
    gm.recompile()
    new_order = ordered_nodes(gm)
    for region in scheduled_regions:
        previous_max: int | None = None
        for phase in region.phases:
            if not phase:
                continue
            phase_min = min(new_order[node] for node in phase)
            phase_max = max(new_order[node] for node in phase)
            if previous_max is not None and previous_max >= phase_min:
                direction = "backward" if region.region.is_backward else "forward"
                raise ValueError(
                    "ep_overlap failed to materialize requested block order for "
                    f"{region.region.root_fqn!r} ({direction})."
                )
            previous_max = phase_max


def _schedule_ep_overlap_regions(
    gm: fx.GraphModule,
    *,
    module_pattern: str,
    require_all_to_all: bool,
    reorder: bool = True,
    pair_first_token_exchange: bool = False,
) -> int:
    """Run validation or scheduling for all chunked regions matching a pattern."""
    order = ordered_nodes(gm)
    chunked_regions = collect_chunked_regions(gm, module_pattern=module_pattern)
    owner_by_node = {
        node: body.owner
        for region in chunked_regions
        for body in region.bodies_by_chunk.values()
        for node in body.nodes
    }
    scheduled_regions = [
        planned
        for region in chunked_regions
        if (
            planned := _plan_region(
                region,
                order=order,
                owner_by_node=owner_by_node,
                pair_first_token_exchange=pair_first_token_exchange,
                rewrite_token_count_sync_copies=reorder,
            )
        )
        is not None
    ]
    logger.debug(
        "ep_overlap discovered %d chunked region(s), scheduled %d: pattern=%s",
        len(chunked_regions),
        len(scheduled_regions),
        module_pattern,
    )

    if scheduled_regions and reorder:
        _apply_schedule(gm, scheduled_regions)
    elif require_all_to_all:
        raise ValueError(
            f"ep_overlap did not find any chunked EP all-to-all regions for "
            f"pattern {module_pattern}."
        )
    return len(scheduled_regions)


def ep_overlap_validate_pass(
    gm: fx.GraphModule,
    example_inputs: tuple[Any, ...] | None = None,
    *,
    module_pattern: str,
    require_all_to_all: bool = False,
    pair_first_token_exchange: bool = False,
) -> fx.GraphModule:
    """Validate the already chunked graph without changing node order."""
    del example_inputs
    validated = _schedule_ep_overlap_regions(
        gm,
        module_pattern=module_pattern,
        require_all_to_all=require_all_to_all,
        reorder=False,
        pair_first_token_exchange=pair_first_token_exchange,
    )
    logger.info(
        "Validated %d ep_overlap chunked region(s): module=%s",
        validated,
        module_pattern,
    )
    return gm


def ep_overlap_schedule_pass(
    gm: fx.GraphModule,
    example_inputs: tuple[Any, ...] | None = None,
    *,
    module_pattern: str,
    require_all_to_all: bool = True,
    pair_first_token_exchange: bool = False,
) -> fx.GraphModule:
    """Reorder already chunked regions around EP all-to-alls."""
    del example_inputs
    scheduled = _schedule_ep_overlap_regions(
        gm,
        module_pattern=module_pattern,
        require_all_to_all=require_all_to_all,
        pair_first_token_exchange=pair_first_token_exchange,
    )
    logger.info(
        "Applied ep_overlap scheduling to %d chunked region(s): module=%s",
        scheduled,
        module_pattern,
    )
    return gm

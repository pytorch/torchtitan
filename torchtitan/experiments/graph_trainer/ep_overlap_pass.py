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
* true EP scheduling markers are graph-visible token-exchange launches inside
  the selected chunk body: annotated all-to-all launches for the c10d path, and
  MinimalAsyncEP dispatch/combine launch ops for the MinimalAsyncEP path;
* ``custom[_EP_TOKEN_EXCHANGE]`` marks true c10d token-exchange launches; waits
  may inherit the annotation from traceback and are normalized to
  ``custom[_EP_TOKEN_EXCHANGE_WAIT]``. MinimalAsyncEP marker labels are inferred
  from the launch/wait op names;
* marker counts and labels must match across chunks;
* forward emits marker pairs in chunk order 0 then 1, backward emits 1 then 0;
* MoE-root chunking pairs both chunks' first-marker setup before launching the
  first marker pair; wider transformer-root chunking keeps the regular
  wait-gated per-chunk closure schedule for every marker;
* token-count sync CPU copies, when present in the first marker closure, are
  launched before their CPU scalar/list consumers;
* the ``auto`` schedule emits ready non-collective body work as filler after
  each marker pair; named schedules order typed token-exchange and module-FQN
  anchors while preserving their dependency closures;
* all graph nodes remain in the sorted graph exactly once and the final graph
  must lint.

The same contract covers eager and graph chunking.  If a chunked region violates
the contract, the pass errors rather than producing a silent schedule change.

Pseudo-code
===========
1. Collect chunked regions and build node -> chunk-owner lookup from shared EP
   pass metadata.
2. For each region, collect true token-exchange markers per chunk and normalize
   wait annotations inherited from traceback.
3. Validate both chunks have the same marker signature, then build dependency
   closures needed to launch each marker.
4. Emit dependency-closed phases from either the default greedy fill or a
   validated named schedule.
5. Apply the requested region phases through a stable topological sort, lint,
   recompile, and validate that phase order materialized.
"""

from __future__ import annotations

import operator
import warnings

from dataclasses import dataclass
from typing import Any

import torch
import torch.fx as fx
from torch._dynamo.graph_deduplication import _stable_topological_sort
from torch.utils._ordered_set import OrderedSet

from torchtitan.experiments.graph_trainer.common_utils import (
    _ACTIVATION_RECOMPUTE,
    _EP_TOKEN_COUNT_SYNC,
    _EP_TOKEN_EXCHANGE,
    _EP_TOKEN_EXCHANGE_WAIT,
    _MODULE_FQN,
)
from torchtitan.experiments.graph_trainer.ep_overlap_schedules import (
    CustomScheduleContext,
    EpOverlapExecution,
    EpOverlapScheduleAnchor,
    matches_module_fqn_subtree,
    ModuleFQNAnchor,
    ReadyFillerAnchor,
    TokenExchangeAnchor,
)
from torchtitan.experiments.graph_trainer.ep_pass_utils import (
    ChunkBody,
    ChunkedRegion,
    ChunkOwner,
    collect_chunked_regions,
    ep_token_exchange_launch_phase,
    is_c10d_functional_node,
    ordered_nodes,
)
from torchtitan.experiments.graph_trainer.registry import EP_OVERLAP_SCHEDULE_REGISTRY
from torchtitan.tools.logging import logger


_GRAPH_BOUNDARY_OPS = {"placeholder", "get_attr"}
_MINIMAL_ASYNC_EP_WAITS = {
    "dispatch": frozenset({"wait_dispatch", "wait_dispatch_data"}),
    "combine": frozenset({"wait_combine"}),
}


# Step 0: Small metadata helpers and local scheduling records.


@dataclass(frozen=True)
class _TokenExchange:
    label: str
    launch: fx.Node
    execution: EpOverlapExecution


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


def _execution(node: fx.Node, *, is_backward: bool) -> EpOverlapExecution:
    if not is_backward:
        return "forward"
    return "recompute" if node.meta.get(_ACTIVATION_RECOMPUTE) else "backward"


def _minimal_async_ep_op_name(node: fx.Node) -> str | None:
    target = node.target
    if getattr(target, "namespace", None) != "minimal_async_ep":
        return None
    return target._schema.name.rsplit("::", 1)[-1]


def _is_minimal_async_ep_wait(node: fx.Node) -> bool:
    name = _minimal_async_ep_op_name(node)
    return any(name in waits for waits in _MINIMAL_ASYNC_EP_WAITS.values())


def _is_matching_minimal_async_ep_wait(wait: fx.Node, launch: fx.Node) -> bool:
    phase = ep_token_exchange_launch_phase(launch)
    return phase is not None and _minimal_async_ep_op_name(wait) in (
        _MINIMAL_ASYNC_EP_WAITS.get(phase) or ()
    )


def _token_exchange_launch_for_wait(node: fx.Node) -> fx.Node | None:
    if node.op != "call_function" or not node.all_input_nodes:
        return None
    launch = node.all_input_nodes[0]
    if (
        _is_minimal_async_ep_wait(node)
        and launch.op == "call_function"
        and launch.target == operator.getitem
        and launch.all_input_nodes
    ):
        launch = launch.all_input_nodes[0]
    return launch


def _is_token_exchange_projection(node: fx.Node, node_set: set[fx.Node]) -> bool:
    if (
        node.op != "call_function"
        or node.target != operator.getitem
        or not node.all_input_nodes
    ):
        return False
    launch = node.all_input_nodes[0]
    return launch in node_set and _is_token_exchange_launch(launch)


def _ep_label(node: fx.Node) -> str:
    """Return the optional EP phase label for logs/wait metadata."""
    phase = ep_token_exchange_launch_phase(node)
    return phase if phase is not None else "token_exchange"


def _is_token_exchange_launch(node: fx.Node) -> bool:
    """Return whether a node is a token-exchange scheduling marker.

    Only true token-exchange launches annotated with ``EP_token_exchange``
    (dispatch or combine) are scheduling markers. For the all-to-all backend,
    token-count all-to-all only carries ``EP: dispatch`` and intentionally is
    not a marker. MinimalAsyncEP has no separate token-count scheduling marker.
    """
    return ep_token_exchange_launch_phase(node) is not None


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
    launch = _token_exchange_launch_for_wait(node)
    if launch is None:
        return False
    if launch not in node_set or not _is_token_exchange_launch(launch):
        return False
    if launch.target == torch.ops._c10d_functional.all_to_all_single.default:
        return node.target == torch.ops._c10d_functional.wait_tensor.default
    return _is_matching_minimal_async_ep_wait(node, launch)


def _token_exchange_wait_users(
    launch: fx.Node,
    *,
    node_set: set[fx.Node],
) -> list[fx.Node]:
    candidates = set(launch.users)
    for user in launch.users:
        if user.op == "call_function" and user.target == operator.getitem:
            candidates.update(user.users)
    return [
        user
        for user in candidates
        if user in node_set and _is_wait_for_token_exchange(user, {launch})
    ]


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
            if _is_token_exchange_projection(node, node_set):
                custom = dict(_custom_meta(node))
                custom.pop(_EP_TOKEN_EXCHANGE, None)
                node.meta["custom"] = custom
                continue
            raise ValueError(
                "ep_overlap found EP token-exchange metadata on non-marker "
                f"node {node.name} ({node.target}). Only token-exchange "
                "launches and their waits may carry this annotation."
            )

        label = _ep_label(node)
        waits = _token_exchange_wait_users(node, node_set=node_set)
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
        exchanges.append(
            _TokenExchange(
                label=label,
                launch=node,
                execution=_execution(node, is_backward=body.owner.is_backward),
            )
        )
    return tuple(exchanges)


def _exchange_signature(exchanges: tuple[_TokenExchange, ...]) -> tuple[str, ...]:
    """Return the semantic marker sequence used to match chunk pairs."""
    return ("token_exchange",) * len(exchanges)


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


def _first_exchange_sync_parts(
    *,
    region: ChunkedRegion,
    closures: dict[int, tuple[tuple[fx.Node, ...], ...]],
    chunk_order: tuple[int, ...],
    launch_nodes: set[fx.Node],
    owner_by_node: dict[fx.Node, ChunkOwner],
) -> dict[int, _SyncClosureParts | None]:
    parts_by_chunk: dict[int, _SyncClosureParts | None] = {}
    for chunk_id in chunk_order:
        body = region.bodies_by_chunk[chunk_id]
        first_closure = closures[chunk_id][0]
        _validate_token_count_sync_copies(body, first_closure=first_closure)
        parts_by_chunk[chunk_id] = _split_token_count_sync_closure(
            first_closure,
            body=body,
            launch_nodes=launch_nodes,
            owner_by_node=owner_by_node,
        )
    if any(parts is not None for parts in parts_by_chunk.values()) and not all(
        parts is not None for parts in parts_by_chunk.values()
    ):
        direction = "backward" if region.is_backward else "forward"
        raise ValueError(
            "ep_overlap expected token-count sync CPU copies for both chunks "
            f"of {region.root_fqn!r} ({direction}) when optimizing sync "
            "copy scheduling."
        )
    return parts_by_chunk


def _rewrite_sync_copies(copies: tuple[fx.Node, ...], *, enabled: bool) -> None:
    if not enabled:
        return
    for idx, copy in enumerate(copies):
        _set_copy_non_blocking(copy, idx + 1 != len(copies))


def _paired_first_exchange_blocks(
    *,
    closures: dict[int, tuple[tuple[fx.Node, ...], ...]],
    sync_parts_by_chunk: dict[int, _SyncClosureParts | None],
    chunk_order: tuple[int, ...],
    launch_nodes: set[fx.Node],
    rewrite_token_count_sync_copies: bool,
) -> tuple[tuple[fx.Node, ...], ...]:
    """Pair both chunks' first setup while preserving their launch order."""
    blocks: list[tuple[fx.Node, ...]] = []
    emitted: set[fx.Node] = set()

    def append_pending(nodes: tuple[fx.Node, ...]) -> None:
        pending = tuple(node for node in nodes if node not in emitted)
        if pending:
            blocks.append(pending)
            emitted.update(pending)

    paired_parts = tuple(sync_parts_by_chunk[chunk_id] for chunk_id in chunk_order)
    if all(parts is not None for parts in paired_parts):
        for parts in paired_parts:
            assert parts is not None
            append_pending(parts.pre_copy)
        copies = tuple(
            copy
            for parts in paired_parts
            for copy in (parts.copies if parts is not None else ())
        )
        _rewrite_sync_copies(copies, enabled=rewrite_token_count_sync_copies)
        append_pending(copies)
        for parts in paired_parts:
            assert parts is not None
            append_pending(parts.post_copy)
        for parts in paired_parts:
            assert parts is not None
            append_pending(parts.launches)
        return tuple(blocks)

    for chunk_id in chunk_order:
        append_pending(
            tuple(node for node in closures[chunk_id][0] if node not in launch_nodes)
        )
    for chunk_id in chunk_order:
        append_pending(
            tuple(node for node in closures[chunk_id][0] if node in launch_nodes)
        )
    return tuple(blocks)


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
    for chunk_id in chunk_order:
        body = region.bodies_by_chunk[chunk_id]
        candidates = sorted(
            candidates_by_chunk.get(chunk_id, set()) - emitted,
            key=order.__getitem__,
        )
        for node in candidates:
            if not include_waits and (
                _is_c10d_functional_node(node) or _is_minimal_async_ep_wait(node)
            ):
                continue
            deps = _body_deps(node, body=body, owner_by_node=owner_by_node)
            if all(dep in emitted for dep in deps):
                ready.append(node)
    return tuple(dict.fromkeys(ready))


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


def _exchange_closures(
    *,
    region: ChunkedRegion,
    exchanges_by_chunk: dict[int, tuple[_TokenExchange, ...]],
    chunk_order: tuple[int, ...],
    order: dict[fx.Node, int],
    owner_by_node: dict[fx.Node, ChunkOwner],
) -> dict[int, tuple[tuple[fx.Node, ...], ...]]:
    exchange_indices = {
        chunk_id: {
            exchange.launch: idx
            for idx, exchange in enumerate(exchanges_by_chunk[chunk_id])
        }
        for chunk_id in chunk_order
    }
    return {
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


class _CustomScheduleMismatch(Exception):
    pass


def _custom_schedule_context(
    *,
    region: ChunkedRegion,
    exchanges_by_chunk: dict[int, tuple[_TokenExchange, ...]],
    memory_policy: str,
) -> CustomScheduleContext:
    return CustomScheduleContext(
        root_fqn=region.root_fqn,
        direction="backward" if region.is_backward else "forward",
        memory_policy=memory_policy,
        exchange_signature=tuple(
            (exchange.execution, exchange.label) for exchange in exchanges_by_chunk[0]
        ),
        module_fqns=frozenset(
            fqn
            for body in region.bodies_by_chunk.values()
            for node in body.nodes
            if isinstance((fqn := _custom_meta(node).get(_MODULE_FQN)), str)
        ),
    )


def _resolve_custom_anchor_targets(
    anchor: EpOverlapScheduleAnchor,
    *,
    region: ChunkedRegion,
    exchanges_by_chunk: dict[int, tuple[_TokenExchange, ...]],
) -> tuple[fx.Node, ...]:
    if isinstance(anchor, ReadyFillerAnchor):
        return ()

    body = region.bodies_by_chunk.get(anchor.chunk_id)
    if body is None:
        raise _CustomScheduleMismatch(f"chunk {anchor.chunk_id} is absent")

    if isinstance(anchor, TokenExchangeAnchor):
        matches = [
            exchange.launch
            for exchange in exchanges_by_chunk[anchor.chunk_id]
            if exchange.execution == anchor.execution and exchange.label == anchor.phase
        ]
        if anchor.occurrence < 0 or anchor.occurrence >= len(matches):
            raise _CustomScheduleMismatch(
                f"token anchor {anchor} matched {len(matches)} launch(es)"
            )
        return (matches[anchor.occurrence],)

    assert isinstance(anchor, ModuleFQNAnchor)
    matches = tuple(
        node
        for node in body.nodes
        if _execution(node, is_backward=region.is_backward) == anchor.execution
        and isinstance((fqn := _custom_meta(node).get(_MODULE_FQN)), str)
        and matches_module_fqn_subtree(anchor.module_fqn, fqn)
    )
    if not matches:
        raise _CustomScheduleMismatch(f"module anchor {anchor} matched no nodes")
    return matches


def _validate_custom_anchor_targets(
    *,
    anchors: tuple[EpOverlapScheduleAnchor, ...],
    targets: tuple[tuple[fx.Node, ...], ...],
    exchanges_by_chunk: dict[int, tuple[_TokenExchange, ...]],
) -> dict[fx.Node, int]:
    anchor_by_node: dict[fx.Node, int] = {}
    for index, nodes in enumerate(targets):
        for node in nodes:
            previous = anchor_by_node.setdefault(node, index)
            if previous != index:
                raise _CustomScheduleMismatch(
                    f"anchors {previous} and {index} both match {node.name}"
                )

    expected_launches = {
        exchange.launch
        for exchanges in exchanges_by_chunk.values()
        for exchange in exchanges
    }
    scheduled_launches = {
        node
        for anchor, nodes in zip(anchors, targets)
        if isinstance(anchor, TokenExchangeAnchor)
        for node in nodes
    }
    if scheduled_launches != expected_launches:
        missing = expected_launches - scheduled_launches
        extra = scheduled_launches - expected_launches
        raise _CustomScheduleMismatch(
            "token anchors do not cover the region exactly: "
            f"missing={[node.name for node in missing]}, "
            f"extra={[node.name for node in extra]}"
        )

    return anchor_by_node


def _build_custom_region_phases(
    *,
    region: ChunkedRegion,
    exchanges_by_chunk: dict[int, tuple[_TokenExchange, ...]],
    anchors: tuple[EpOverlapScheduleAnchor, ...],
    order: dict[fx.Node, int],
    owner_by_node: dict[fx.Node, ChunkOwner],
    pair_first_token_exchange: bool,
    rewrite_token_count_sync_copies: bool,
) -> tuple[tuple[fx.Node, ...], ...]:
    targets = tuple(
        _resolve_custom_anchor_targets(
            anchor,
            region=region,
            exchanges_by_chunk=exchanges_by_chunk,
        )
        for anchor in anchors
    )
    anchor_by_node = _validate_custom_anchor_targets(
        anchors=anchors,
        targets=targets,
        exchanges_by_chunk=exchanges_by_chunk,
    )
    chunk_order = (1, 0) if region.is_backward else (0, 1)
    closures = _exchange_closures(
        region=region,
        exchanges_by_chunk=exchanges_by_chunk,
        chunk_order=chunk_order,
        order=order,
        owner_by_node=owner_by_node,
    )
    exchange_index_by_launch = {
        exchange.launch: index
        for exchanges in exchanges_by_chunk.values()
        for index, exchange in enumerate(exchanges)
    }
    launch_nodes = set(exchange_index_by_launch)
    phases: list[tuple[fx.Node, ...]] = []
    emitted: set[fx.Node] = set()
    reserved = set(anchor_by_node)

    def append_phase(nodes: tuple[fx.Node, ...], *, anchor_index: int) -> None:
        later_anchor = next(
            (
                (node, index)
                for node in nodes
                if (index := anchor_by_node.get(node)) is not None
                and index > anchor_index
            ),
            None,
        )
        if later_anchor is not None:
            node, index = later_anchor
            raise _CustomScheduleMismatch(
                f"anchor {anchor_index} closure requires later anchor {index} "
                f"node {node.name}"
            )
        pending = tuple(node for node in nodes if node not in emitted)
        if pending:
            phases.append(pending)
            emitted.update(pending)

    start_index = 0
    if (
        pair_first_token_exchange
        and len(anchors) >= 2
        and all(isinstance(anchor, TokenExchangeAnchor) for anchor in anchors[:2])
    ):
        first = anchors[0]
        second = anchors[1]
        assert isinstance(first, TokenExchangeAnchor)
        assert isinstance(second, TokenExchangeAnchor)
        first_launch = targets[0][0]
        second_launch = targets[1][0]
        if (
            first.execution == second.execution
            and first.phase == second.phase
            and first.chunk_id != second.chunk_id
            and exchange_index_by_launch[first_launch] == 0
            and exchange_index_by_launch[second_launch] == 0
        ):
            paired_order = (first.chunk_id, second.chunk_id)
            sync_parts = _first_exchange_sync_parts(
                region=region,
                closures=closures,
                chunk_order=paired_order,
                launch_nodes=launch_nodes,
                owner_by_node=owner_by_node,
            )
            for block in _paired_first_exchange_blocks(
                closures=closures,
                sync_parts_by_chunk=sync_parts,
                chunk_order=paired_order,
                launch_nodes=launch_nodes,
                rewrite_token_count_sync_copies=rewrite_token_count_sync_copies,
            ):
                append_phase(block, anchor_index=1)
            start_index = 2

    for index in range(start_index, len(anchors)):
        anchor = anchors[index]
        if isinstance(anchor, ReadyFillerAnchor):
            candidates = {
                chunk_id: {
                    node
                    for node in region.bodies_by_chunk[chunk_id].nodes
                    if node not in reserved
                    and _execution(node, is_backward=region.is_backward)
                    == anchor.execution
                }
                for chunk_id in chunk_order
            }
            _append_ready_blocks(
                phases,
                emitted,
                candidates_by_chunk=candidates,
                region=region,
                chunk_order=chunk_order,
                order=order,
                owner_by_node=owner_by_node,
                include_waits=False,
            )
            continue
        if any(node in emitted for node in targets[index]):
            raise _CustomScheduleMismatch(
                f"anchor {index} target was required by an earlier anchor"
            )
        if isinstance(anchor, TokenExchangeAnchor):
            launch = targets[index][0]
            closure = closures[anchor.chunk_id][exchange_index_by_launch[launch]]
            append_phase(closure, anchor_index=index)
        else:
            body = region.bodies_by_chunk[anchor.chunk_id]
            ancestors = _body_ancestors(
                targets[index],
                body=body,
                owner_by_node=owner_by_node,
                node_set=set(body.nodes),
            )
            append_phase(
                tuple(
                    node
                    for node in body.nodes
                    if node in ancestors or node in targets[index]
                ),
                anchor_index=index,
            )

    missing_targets = set(anchor_by_node) - emitted
    if missing_targets:
        raise _CustomScheduleMismatch(
            "custom schedule did not emit targets: "
            f"{[node.name for node in missing_targets]}"
        )

    remaining = {
        chunk_id: set(region.bodies_by_chunk[chunk_id].nodes) - emitted
        for chunk_id in chunk_order
    }
    _append_ready_blocks(
        phases,
        emitted,
        candidates_by_chunk=remaining,
        region=region,
        chunk_order=chunk_order,
        order=order,
        owner_by_node=owner_by_node,
        include_waits=False,
    )
    made_progress = True
    while made_progress:
        made_progress = False
        for chunk_id in chunk_order:
            made_progress |= _append_ready_blocks(
                phases,
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
        raise _CustomScheduleMismatch(
            "custom schedule could not place all region nodes: "
            f"{[node.name for node in missing[:8]]}"
        )
    _validate_custom_phase_dependencies(phases)
    return tuple(phases)


def _validate_custom_phase_dependencies(
    phases: list[tuple[fx.Node, ...]],
) -> None:
    phase_by_node = {
        node: phase_index for phase_index, phase in enumerate(phases) for node in phase
    }
    for phase_index, phase in enumerate(phases):
        for node in phase:
            seen: set[fx.Node] = set()
            stack = list(node.all_input_nodes)
            while stack:
                dep = stack.pop()
                if dep in seen:
                    continue
                seen.add(dep)
                dep_phase = phase_by_node.get(dep)
                if dep_phase is not None and dep_phase > phase_index:
                    raise _CustomScheduleMismatch(
                        f"phase {phase_index} node {node.name} depends on later "
                        f"phase {dep_phase} node {dep.name}"
                    )
                stack.extend(dep.all_input_nodes)


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
    closures = _exchange_closures(
        region=region,
        exchanges_by_chunk=exchanges_by_chunk,
        chunk_order=chunk_order,
        order=order,
        owner_by_node=owner_by_node,
    )
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
        pending = tuple(node for node in nodes if node not in emitted)
        if pending:
            blocks.append(pending)
            emitted.update(pending)

    sync_parts_by_chunk: dict[int, _SyncClosureParts | None] = {}
    if exchanges_by_chunk[0]:
        sync_parts_by_chunk = _first_exchange_sync_parts(
            region=region,
            closures=closures,
            chunk_order=chunk_order,
            launch_nodes=launch_nodes,
            owner_by_node=owner_by_node,
        )

    for exchange_idx in range(len(exchanges_by_chunk[0])):
        if exchange_idx == 0 and pair_first_token_exchange:
            for block in _paired_first_exchange_blocks(
                closures=closures,
                sync_parts_by_chunk=sync_parts_by_chunk,
                chunk_order=chunk_order,
                launch_nodes=launch_nodes,
                rewrite_token_count_sync_copies=rewrite_token_count_sync_copies,
            ):
                append_pending(block)
        else:
            # Regular wait-gated scheduling: emit each chunk's full closure
            # sequentially. Later closures may intentionally include waits
            # whose users are the next token-exchange launch.
            for chunk_id in chunk_order:
                sync_parts = (
                    sync_parts_by_chunk.get(chunk_id) if exchange_idx == 0 else None
                )
                if sync_parts is not None:
                    _rewrite_sync_copies(
                        sync_parts.copies,
                        enabled=rewrite_token_count_sync_copies,
                    )
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
    schedule_name: str,
    memory_policy: str,
    fallback_warning_keys: set[tuple[object, ...]],
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
            f"ep_overlap expected matching EP token-exchange counts for "
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

    phases: tuple[tuple[fx.Node, ...], ...] | None = None
    if schedule_name != "auto":
        provider = EP_OVERLAP_SCHEDULE_REGISTRY.get(schedule_name)
        if provider is None:
            raise ValueError(f"Unknown EP-overlap schedule {schedule_name!r}.")
        context = _custom_schedule_context(
            region=region,
            exchanges_by_chunk=exchanges_by_chunk,
            memory_policy=memory_policy,
        )
        peer_signature = tuple(
            (exchange.execution, exchange.label) for exchange in exchanges_by_chunk[1]
        )
        if peer_signature != context.exchange_signature:
            raise ValueError(
                f"EP-overlap schedule {schedule_name!r} requires matching "
                f"chunk execution signatures for {region.root_fqn!r} "
                f"({context.direction}); found "
                f"chunk0={context.exchange_signature}, chunk1={peer_signature}"
            )
        anchors = provider(context)
        if anchors is None:
            warning_key = (
                schedule_name,
                context.direction,
                context.exchange_signature,
            )
            if warning_key not in fallback_warning_keys:
                warnings.warn(
                    f"EP-overlap schedule {schedule_name!r} does not match "
                    f"{region.root_fqn!r} ({context.direction}): "
                    "the policy has no variant for "
                    f"signature={context.exchange_signature}, "
                    f"memory_policy={memory_policy!r}. Falling back to 'auto'.",
                    stacklevel=3,
                )
                fallback_warning_keys.add(warning_key)
        else:
            try:
                phases = _build_custom_region_phases(
                    region=region,
                    exchanges_by_chunk=exchanges_by_chunk,
                    anchors=tuple(anchors),
                    order=order,
                    owner_by_node=owner_by_node,
                    pair_first_token_exchange=pair_first_token_exchange,
                    rewrite_token_count_sync_copies=rewrite_token_count_sync_copies,
                )
            except _CustomScheduleMismatch as error:
                raise ValueError(
                    f"Invalid EP-overlap schedule {schedule_name!r} for "
                    f"{region.root_fqn!r} ({context.direction}): {error}."
                ) from error

    if phases is None:
        phases = _build_region_phases(
            region=region,
            exchanges_by_chunk=exchanges_by_chunk,
            order=order,
            owner_by_node=owner_by_node,
            pair_first_token_exchange=pair_first_token_exchange,
            rewrite_token_count_sync_copies=rewrite_token_count_sync_copies,
        )
    return _ScheduledRegion(region, phases) if phases else None


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
    require_token_exchange: bool,
    reorder: bool = True,
    pair_first_token_exchange: bool = False,
    schedule_name: str = "auto",
    memory_policy: str = "default",
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
    fallback_warning_keys: set[tuple[object, ...]] = set()
    scheduled_regions: list[_ScheduledRegion] = []
    for region in chunked_regions:
        planned = _plan_region(
            region,
            order=order,
            owner_by_node=owner_by_node,
            pair_first_token_exchange=pair_first_token_exchange,
            rewrite_token_count_sync_copies=reorder,
            schedule_name=schedule_name,
            memory_policy=memory_policy,
            fallback_warning_keys=fallback_warning_keys,
        )
        if planned is not None:
            scheduled_regions.append(planned)
    logger.debug(
        "ep_overlap discovered %d chunked region(s), scheduled %d: "
        "pattern=%s schedule=%s",
        len(chunked_regions),
        len(scheduled_regions),
        module_pattern,
        schedule_name,
    )

    if scheduled_regions and reorder:
        _apply_schedule(gm, scheduled_regions)
    elif require_token_exchange:
        raise ValueError(
            f"ep_overlap did not find any chunked EP token-exchange regions for "
            f"pattern {module_pattern}."
        )
    return len(scheduled_regions)


def ep_overlap_validate_pass(
    gm: fx.GraphModule,
    example_inputs: tuple[Any, ...] | None = None,
    *,
    module_pattern: str,
    require_token_exchange: bool = False,
    pair_first_token_exchange: bool = False,
) -> fx.GraphModule:
    """Validate the already chunked graph without changing node order."""
    del example_inputs
    validated = _schedule_ep_overlap_regions(
        gm,
        module_pattern=module_pattern,
        require_token_exchange=require_token_exchange,
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
    require_token_exchange: bool = True,
    pair_first_token_exchange: bool = False,
    schedule_name: str = "auto",
    memory_policy: str = "default",
) -> fx.GraphModule:
    """Reorder already chunked regions around EP token exchanges."""
    del example_inputs
    scheduled = _schedule_ep_overlap_regions(
        gm,
        module_pattern=module_pattern,
        require_token_exchange=require_token_exchange,
        pair_first_token_exchange=pair_first_token_exchange,
        schedule_name=schedule_name,
        memory_policy=memory_policy,
    )
    logger.info(
        "Applied ep_overlap scheduling to %d chunked region(s): "
        "module=%s schedule=%s",
        scheduled,
        module_pattern,
        schedule_name,
    )
    return gm

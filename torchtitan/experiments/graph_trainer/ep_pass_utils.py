# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Shared EP chunk/overlap metadata helpers.

Region discovery reads chunk metadata produced by eager chunking and groups
nodes by ``(module_fqn, direction, chunk_id)``.
"""

from __future__ import annotations

import fnmatch
import logging
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.fx as fx

from torchtitan.experiments.graph_trainer.common_utils import _is_backward_node

logger = logging.getLogger(__name__)


aten = torch.ops.aten


def is_c10d_functional_node(node: fx.Node) -> bool:
    """Return whether a node is a distributed functional op (AG/RS/A2A/wait)."""
    return (
        node.op == "call_function"
        and isinstance(node.target, torch._ops.OpOverload)
        and node.target.namespace == "_c10d_functional"
    )


def ordered_nodes(gm: fx.GraphModule) -> dict[fx.Node, int]:
    """Map each graph node to its current topological position."""
    return {node: idx for idx, node in enumerate(gm.graph.nodes)}


@dataclass(frozen=True)
class ChunkOwner:
    """Stable identity for one chunk body in one direction of one module."""

    root_fqn: str
    is_backward: bool
    chunk_id: int


@dataclass(frozen=True)
class ChunkBody:
    """A planned chunk body plus its external graph inputs."""

    owner: ChunkOwner
    nodes: tuple[fx.Node, ...]
    node_set: frozenset[fx.Node]
    live_ins: frozenset[fx.Node]
    producer: str


@dataclass(frozen=True)
class ChunkedRegion:
    """The chunk bodies for one module/direction pair."""

    root_fqn: str
    is_backward: bool
    bodies_by_chunk: dict[int, ChunkBody]


def _chunk_owner(node: fx.Node) -> ChunkOwner | None:
    custom = node.meta.get("custom", {})
    if not isinstance(custom, dict):
        custom = {}

    def get_meta(key: str) -> object:
        return node.meta.get(key, custom.get(key))

    if get_meta("chunked_region_role") != "body":
        return None
    chunk_id = get_meta("chunk_id")
    root = get_meta("chunked_region_fqn")
    is_backward = get_meta("chunked_region_is_backward")
    if chunk_id not in (0, 1) or not isinstance(root, str):
        raise ValueError(f"Chunk body node {node.name} has incomplete chunk metadata.")
    return ChunkOwner(
        root_fqn=root,
        is_backward=bool(
            is_backward if is_backward is not None else _is_backward_node(node)
        ),
        chunk_id=chunk_id,
    )


def _clear_chunk_ownership(nodes: Iterable[fx.Node]) -> None:
    """Mark nodes shared by multiple chunks as outside every chunk body."""
    keys = (
        "chunk_id",
        "chunked_region_fqn",
        "chunked_region_is_backward",
        "chunked_region_producer",
        "chunked_region_role",
    )
    for node in nodes:
        for key in keys:
            node.meta.pop(key, None)
        custom = node.meta.get("custom")
        if isinstance(custom, dict):
            custom = dict(custom)
            for key in keys:
                custom.pop(key, None)
            node.meta["custom"] = custom


def collect_chunked_regions(
    gm: fx.GraphModule, *, module_pattern: str
) -> list[ChunkedRegion]:
    """Collect chunk bodies matching ``module_pattern`` in graph order."""
    nodes_by_owner: dict[ChunkOwner, list[fx.Node]] = defaultdict(list)
    for node in gm.graph.nodes:
        owner = _chunk_owner(node)
        if owner and fnmatch.fnmatchcase(owner.root_fqn, module_pattern):
            nodes_by_owner[owner].append(node)

    grouped: dict[tuple[str, bool], dict[int, ChunkBody]] = defaultdict(dict)
    for owner, nodes in nodes_by_owner.items():
        node_set = frozenset(nodes)
        grouped[(owner.root_fqn, owner.is_backward)][owner.chunk_id] = ChunkBody(
            owner=owner,
            nodes=tuple(nodes),
            node_set=node_set,
            live_ins=frozenset(
                inp
                for node in nodes
                for inp in node.all_input_nodes
                if inp not in node_set
            ),
            producer=str(nodes[0].meta.get("chunked_region_producer", "eager")),
        )

    return [
        ChunkedRegion(root, is_backward, by_chunk)
        for (root, is_backward), by_chunk in grouped.items()
    ]

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""EP process-group isolation for graph_trainer.

Algorithm:
1. Scan collectives and separate EP all-to-alls from all other collectives.
2. If an EP all-to-all resolves to the same process-group object as a non-EP
   collective, create one extra PG with the same ranks for EP traffic.
3. Rewrite the EP all-to-all PG argument. Other collectives keep their original
   PGs and ordering.
"""

from __future__ import annotations

import torch
import torch.fx as fx
from torch.utils._ordered_set import OrderedSet

from torchtitan.tools.logging import logger

c10d = torch.ops._c10d_functional
_EXTRA_EP_PG_REGISTRY: dict[str, str] = {}


def _is_ep_all_to_all(node: fx.Node) -> bool:
    """Return whether ``node`` is a traced all-to-all using the EP PG slot."""
    return (
        node.op == "call_function"
        and node.target == c10d.all_to_all_single.default
        and len(node.args) > 3
    )


def _collective_pg_name(node: fx.Node) -> str | None:
    """Return the process-group name argument for supported c10d collectives."""
    if node.op != "call_function":
        return None
    if node.target == c10d.all_gather_into_tensor.default and len(node.args) > 2:
        return node.args[2]
    if node.target == c10d.reduce_scatter_tensor.default and len(node.args) > 3:
        return node.args[3]
    if node.target == c10d.all_reduce.default and len(node.args) > 2:
        return node.args[2]
    if node.target == c10d.all_to_all_single.default and len(node.args) > 3:
        return node.args[3]
    return None


def _pg_rank_set(pg_name: str) -> frozenset[int]:
    """Resolve ``pg_name`` and return its rank set."""
    import torch.distributed as dist

    pg = dist.distributed_c10d._resolve_process_group(pg_name)
    return frozenset(dist.get_process_group_ranks(pg))


def _same_process_group(lhs_pg_name: str, rhs_pg_name: str) -> bool:
    """Compare process-group object identity after resolving graph PG names."""
    import torch.distributed as dist

    lhs_pg = dist.distributed_c10d._resolve_process_group(lhs_pg_name)
    rhs_pg = dist.distributed_c10d._resolve_process_group(rhs_pg_name)
    return lhs_pg is rhs_pg


def _shares_process_group_with(
    pg_name: str, candidate_pg_names: OrderedSet[str]
) -> bool:
    """Return whether ``pg_name`` resolves to any candidate PG object."""
    for candidate_pg_name in candidate_pg_names:
        if _same_process_group(pg_name, candidate_pg_name):
            if _pg_rank_set(pg_name) != _pg_rank_set(candidate_pg_name):
                raise ValueError(
                    "Resolved the same process group with different rank sets: "
                    f"{pg_name} and {candidate_pg_name}"
                )
            return True
    return False


def _get_or_create_extra_ep_pg(source_pg_name: str) -> str:
    """Create or reuse an EP-only PG with the same ranks as ``source_pg_name``."""
    import torch.distributed as dist

    if source_pg_name in _EXTRA_EP_PG_REGISTRY:
        return _EXTRA_EP_PG_REGISTRY[source_pg_name]

    source_pg = dist.distributed_c10d._resolve_process_group(source_pg_name)
    ranks = dist.get_process_group_ranks(source_pg)
    pg_options = (
        dist.ProcessGroupNCCL.Options(is_high_priority_stream=True)
        if hasattr(dist, "ProcessGroupNCCL")
        else None
    )
    extra_pg = dist.new_group(
        ranks=ranks,
        backend="nccl" if pg_options is not None else None,
        pg_options=pg_options,
        group_desc="ep_extra",
        use_local_synchronization=True,
    )
    _EXTRA_EP_PG_REGISTRY[source_pg_name] = extra_pg.group_name
    logger.info(
        "Created extra EP PG (source: %s, extra: %s, high_priority=%s)",
        source_pg_name,
        extra_pg.group_name,
        pg_options is not None,
    )
    return extra_pg.group_name


def isolate_ep_process_group_pass(
    gm: fx.GraphModule,
    example_inputs: tuple | None = None,
) -> fx.GraphModule:
    """Move EP all-to-alls off any non-EP collective PG they share by object."""
    del example_inputs
    # Step 1: classify process groups by the collectives that use them.
    ep_pg_names: OrderedSet[str] = OrderedSet()
    non_ep_pg_names: OrderedSet[str] = OrderedSet()
    for node in gm.graph.nodes:
        pg_name = _collective_pg_name(node)
        if pg_name is None:
            continue
        if _is_ep_all_to_all(node):
            ep_pg_names.add(pg_name)
        else:
            non_ep_pg_names.add(pg_name)

    logger.debug(
        "EP PG isolation scan: ep_pgs=%s non_ep_pgs=%s",
        list(ep_pg_names),
        list(non_ep_pg_names),
    )

    # Step 2: create replacement PGs only for object-identical sharing. Rank
    # equality alone is not sufficient because distinct PGs can share ranks.
    pg_mapping = {
        pg_name: _get_or_create_extra_ep_pg(pg_name)
        for pg_name in ep_pg_names
        if _shares_process_group_with(pg_name, non_ep_pg_names)
    }
    if not pg_mapping:
        return gm

    # Step 3: rewrite the EP all-to-all PG operand.
    rewrite_counts = {source: 0 for source in pg_mapping}
    for node in gm.graph.nodes:
        if _is_ep_all_to_all(node) and node.args[3] in pg_mapping:
            source = node.args[3]
            node.args = (*node.args[:3], pg_mapping[source], *node.args[4:])
            rewrite_counts[source] += 1
    for source, target in pg_mapping.items():
        logger.info(
            "Rewrote %d EP all-to-all node(s) from PG %s to PG %s",
            rewrite_counts[source],
            source,
            target,
        )
    gm.recompile()
    return gm

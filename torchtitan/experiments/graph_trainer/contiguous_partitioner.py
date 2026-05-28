"""Contiguous partitioner: grow partitions edge-by-edge, merging only
when the result is topologically contiguous.

Algorithm:
  1. Each supported node starts as its own partition (union-find)
  2. Process edges between supported nodes in topological order
  3. For each edge (u, v): if u and v are in different partitions,
     merge iff the combined set has no "crossing" unsupported node
  4. A crossing node N has both an input from the partition and an
     output to the partition — it must execute inside the partition's
     topo range, making single-point contraction impossible

This avoids the CapabilityBasedPartitioner's problems:
  - No merge-order dependence (edges processed in topo order)
  - No heuristic span check (exact crossing detection)
  - No O(V²) dependency viewer (only checks nodes in the merge range)
"""

from __future__ import annotations

from collections import defaultdict

import torch


def partition_contiguous(
    gm: torch.fx.GraphModule,
    is_supported: callable,
) -> list[list[torch.fx.Node]]:
    """Return contiguous partitions of supported nodes.

    Each partition is a list of nodes in topological order. Only
    partitions with at least one supported node are returned.
    """
    graph_nodes = list(gm.graph.nodes)
    node_to_pos = {n: i for i, n in enumerate(graph_nodes)}

    # Step 1: Identify supported nodes
    supported = set()
    for node in graph_nodes:
        if is_supported(node):
            supported.add(node)

    if not supported:
        return []

    # Step 2: Union-Find with per-partition stats.
    #
    # Union-Find tracks which nodes belong to the same partition:
    #   parent[n] — points toward the partition's root representative
    #   uf_rank[n] — tree height at n, used to keep trees balanced
    #
    # Per-partition stats (keyed by root node, updated on merge):
    #   p_min[root] — earliest topo position of any node in the partition
    #   p_max[root] — latest topo position of any node in the partition
    #   p_size[root] — number of nodes in the partition
    parent: dict[torch.fx.Node, torch.fx.Node] = {}
    uf_rank: dict[torch.fx.Node, int] = {}
    p_min: dict[torch.fx.Node, int] = {}
    p_max: dict[torch.fx.Node, int] = {}
    p_size: dict[torch.fx.Node, int] = {}

    for node in supported:
        parent[node] = node
        uf_rank[node] = 0
        pos = node_to_pos[node]
        p_min[node] = pos
        p_max[node] = pos
        p_size[node] = 1

    def find(x: torch.fx.Node) -> torch.fx.Node:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def merge(a: torch.fx.Node, b: torch.fx.Node) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        new_min = min(p_min[ra], p_min[rb])
        new_max = max(p_max[ra], p_max[rb])
        new_size = p_size[ra] + p_size[rb]
        # Union by rank
        if uf_rank[ra] < uf_rank[rb]:
            ra, rb = rb, ra
        parent[rb] = ra
        if uf_rank[ra] == uf_rank[rb]:
            uf_rank[ra] += 1
        p_min[ra] = new_min
        p_max[ra] = new_max
        p_size[ra] = new_size

    # Step 3: Process edges in topological order, merge if contiguous.
    # MAX_SPAN limits the topo range a partition can span. Larger
    # partitions are more likely to be disrupted by the sequential
    # replacement loop (prior replacements shift node positions,
    # breaking the insertion-point check). This is a workaround until
    # batch replacement is implemented.
    MAX_SPAN = 100

    for node in graph_nodes:
        if node not in supported:
            continue
        for inp in node.all_input_nodes:
            if inp not in supported:
                continue
            if find(node) == find(inp):
                continue

            ra, rb = find(node), find(inp)
            if max(p_max[ra], p_max[rb]) - min(p_min[ra], p_min[rb]) > MAX_SPAN:
                continue

            if _can_merge(
                node, inp, find, p_min, p_max,
                supported, graph_nodes, node_to_pos,
            ):
                merge(node, inp)

    # Step 4: Collect partitions in topo order
    groups: dict[torch.fx.Node, list[torch.fx.Node]] = defaultdict(list)
    for node in graph_nodes:
        if node in supported:
            groups[find(node)].append(node)

    return list(groups.values())


def _can_merge(
    a: torch.fx.Node,
    b: torch.fx.Node,
    find: callable,
    p_min: dict,
    p_max: dict,
    supported: set,
    graph_nodes: list,
    node_to_pos: dict,
) -> bool:
    """Check if merging partitions of a and b is contiguous.

    A merge is non-contiguous if there's a PATH through external nodes
    from a partition output back to a partition input:
        partition_node → ext_1 → ext_2 → ... → partition_node
    Contracting the partition would create a cycle.

    We detect this by forward-propagating "reachable from partition"
    through external nodes in [merged_min, merged_max]. If any
    reachable external node feeds back into the partition, there's a
    crossing chain.
    """
    ra, rb = find(a), find(b)
    merged_min = min(p_min[ra], p_min[rb])
    merged_max = max(p_max[ra], p_max[rb])

    def _in_merged(n: torch.fx.Node) -> bool:
        if n not in supported:
            return False
        rn = find(n)
        return rn == ra or rn == rb

    # Collect external nodes in range that are directly reachable from
    # partition outputs (seed set for forward propagation).
    reachable: set[torch.fx.Node] = set()
    for pos in range(merged_min, merged_max + 1):
        node = graph_nodes[pos]
        if _in_merged(node):
            for user in node.users:
                if not _in_merged(user):
                    p = node_to_pos.get(user)
                    if p is not None and merged_min <= p <= merged_max:
                        reachable.add(user)

    # BFS forward through external nodes in range.
    # If any reachable node feeds back into the partition → crossing.
    queue = list(reachable)
    while queue:
        node = queue.pop()
        for user in node.users:
            if _in_merged(user):
                return False  # path from partition → externals → partition
            if user not in reachable:
                p = node_to_pos.get(user)
                if p is not None and merged_min <= p <= merged_max:
                    reachable.add(user)
                    queue.append(user)

    return True

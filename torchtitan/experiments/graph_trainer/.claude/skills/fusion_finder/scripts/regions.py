# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Group the glue nodes of a capture into fusion regions, with their patterns.

    python regions.py <capture dir>

Reads ``graph.jsonl.gz``, ``node_table.csv`` and ``step_budget.json`` (from
``node_table.py``) and ``hbm_bandwidth.json``. Writes ``regions.json`` and prints
the regions above the profit floor.

A region instance is a connected set of glue nodes, plus the kernel-free nodes
(views, allocations) among them, such that no boundary lies on a path between two
members. Boundaries are the nodes a glue kernel cannot absorb: GEMMs, attention,
Dist-MoE fused ops, collectives, nodes whose output is not a GPU tensor (host
values), and nodes that Inductor creates after ``compile.extra_passes``. Paths
follow tensor edges and storage: a write into a storage (an in-place or ``out=``
op, through any view of it) comes after the earlier accesses to that storage and
before the later ones. Instances grow by merging along these edges, in graph
order, while no boundary would lie between their members. A view of a graph input
is like the input: nodes that read it do not merge through it. Hence:

- every glue node is in exactly one instance;
- no path leaves an instance and comes back: an instance can become one
  ``call_module``, and one kernel can replace it.

With each instance as one node the graph must stay acyclic too, so that all can
become calls together (``_break_cycles``).

Forward glue never merges with recompute or backward glue: that edge is a saved
activation, and fusing across it would move backward work into the forward pass.

Instances with the same topology are one region (``Pattern``): the same ops with
the same edges among them and the same arguments, except sizes, scalar operands and
device indices (``match_key`` in ``fusion_finder_capture``). Modules, shapes and
phases do not matter: e.g. the w13 weight gradient of the dense layers and of the
MoE layers' shared experts are one region. Ideal time of an instance: external
input bytes (read once) plus the bytes of its outputs used outside, over the HBM
bandwidth. The gap of a region is its measured time minus its ideal time, summed
over instances. Regions below the profit floor (max(0.5 ms, 0.05% of GPU busy time)) are not worth a kernel: the
gap is what a kernel can save at best.

``regions.json`` lists the regions in the order the lift passes run (by gap,
except that a region comes before the regions whose patterns also match inside
its instances), each with its ``Pattern``.
"""

import argparse
import csv
import gzip
import json
import os
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable
from typing import Any

from node_table import input_nodes, tensors

_FLOOR_MS = 0.5
_FLOOR_PCT_OF_BUSY = 0.05


def _short_op(target: str) -> str:
    return target.removeprefix("aten.").removesuffix(".default")


def _any_layer(fqn: str) -> str:
    return re.sub(r"\blayers\.\d+", "layers.*", fqn)


def key_skeleton(key: Any) -> tuple[str, list[str]]:
    """A ``match_key`` without node names, and the names of its tensor inputs in
    order."""
    inputs: list[str] = []

    def strip(v: Any) -> Any:
        if isinstance(v, dict):
            if v.keys() == {"node"}:
                inputs.append(v["node"])
                return "%"
            return {k: strip(x) for k, x in v.items()}
        if isinstance(v, list):
            return [strip(x) for x in v]
        return v

    return json.dumps(strip(key)), inputs


class Pattern:
    """The topology of a region instance, and its matches in a graph.

    A match is a set of graph nodes, one per member, with the member's ``match_key``
    up to node names and the same tensor edges: each input of a member is the node
    matched to that member, or a node outside the match, the same one wherever the
    instance uses the same input. A member whose output the instance uses only
    inside (not in ``outputs``) must be used only inside the match too.

    A pattern is built from ``nodes``, the instance's members and their match keys
    in graph order, and ``outputs``, the members used outside the instance. A graph
    is given as ``skeleton(n)``, the ``key_skeleton`` of node ``n``'s ``match_key``,
    and ``users(n)``, the nodes that take ``n`` as an argument.
    """

    def __init__(self, nodes: dict[str, Any], outputs: list[str]) -> None:
        self.nodes, self.outputs = nodes, outputs
        self.members = {name: key_skeleton(key) for name, key in nodes.items()}
        self.internal = self.members.keys() - set(outputs)
        # The last member: no other member uses it.
        self.anchor = next(reversed(self.members))
        self.target = nodes[self.anchor][0]
        self.users: dict[str, list[str]] = defaultdict(list)
        for name, (_, inputs) in self.members.items():
            for i in dict.fromkeys(inputs):
                self.users[i].append(name)
        # Matching walks tensor edges, so every member must be reachable through
        # them; members joined only through a storage are not.
        seen, stack = {self.anchor}, [self.anchor]
        while stack:
            name = stack.pop()
            inputs = self.members[name][1] if name in self.members else []
            for near in [*inputs, *self.users[name]]:
                if near not in seen:
                    seen.add(near)
                    stack.append(near)
        if not self.members.keys() <= seen:
            raise ValueError("the instance's members are not connected by tensor edges")

    def to_json(self) -> dict[str, Any]:
        return {"nodes": self.nodes, "outputs": self.outputs}

    def find(
        self,
        anchors: Iterable[str],
        skeleton: Callable[[str], tuple[str, list[str]]],
        users: Callable[[str], list[str]],
    ) -> list[dict[str, str]]:
        """The matches anchored at ``anchors``, tried in order, that share no node
        with an earlier match."""
        matches: list[dict[str, str]] = []
        taken: set[str] = set()
        anchor_skeleton = self.members[self.anchor][0]
        for node in anchors:
            if node in taken or skeleton(node)[0] != anchor_skeleton:
                continue
            bound = self.match(node, skeleton, users)
            if bound is None:
                continue
            members = {bound[m] for m in self.members}
            if taken.isdisjoint(members):
                taken |= members
                matches.append(bound)
        return matches

    def match(
        self,
        node: str,
        skeleton: Callable[[str], tuple[str, list[str]]],
        users: Callable[[str], list[str]],
        within: set[str] | None = None,
    ) -> dict[str, str] | None:
        """A match with ``node`` as the anchor, and only nodes in ``within`` as
        members if given: the graph node of each member and input, or None."""
        return self._extend({self.anchor: node}, set(), skeleton, users, within)

    def _extend(self, bound, checked, skeleton, users, within):
        bound, checked = dict(bound), set(checked)
        taken = set(bound.values())
        # Checking a bound member binds its inputs, without choice.
        todo = [m for m in bound if m in self.members and m not in checked]
        while todo:
            m = todo.pop()
            if within is not None and bound[m] not in within:
                return None
            graph_skeleton, graph_inputs = skeleton(bound[m])
            member_skeleton, member_inputs = self.members[m]
            if graph_skeleton != member_skeleton:
                return None
            # Each user of the member is matched to a distinct user of its node; an
            # internal member's node has no other user.
            num_users = len(users(bound[m]))
            if num_users < len(self.users[m]) or (
                m in self.internal and num_users > len(self.users[m])
            ):
                return None
            for i, graph_input in zip(member_inputs, graph_inputs):
                if i in bound:
                    if bound[i] != graph_input:
                        return None
                    continue
                if graph_input in taken:
                    return None
                bound[i] = graph_input
                taken.add(graph_input)
                if i in self.members:
                    todo.append(i)
            checked.add(m)
        if len(checked) == len(self.members):
            return bound
        # The members still unbound use bound nodes. Branch on the one with the
        # fewest candidates: the users of its bound inputs with its skeleton.
        fewest: tuple[str, list[str]] | None = None
        for u, (member_skeleton, member_inputs) in self.members.items():
            if u in bound:
                continue
            known = [(k, bound[i]) for k, i in enumerate(member_inputs) if i in bound]
            if not known:
                continue
            candidates = [
                c
                for c in users(known[0][1])
                if c not in taken
                and (within is None or c in within)
                and skeleton(c)[0] == member_skeleton
                and all(skeleton(c)[1][k] == g for k, g in known)
            ]
            if fewest is None or len(candidates) < len(fewest[1]):
                fewest = (u, candidates)
                if len(candidates) <= 1:
                    break
        assert fewest is not None, "the members are connected"
        u, candidates = fewest
        for c in candidates:
            found = self._extend({**bound, u: c}, checked, skeleton, users, within)
            if found:
                return found
        return None


def _strongly_connected(succ: dict[Any, set]) -> list[list]:
    """The strongly connected components of a directed graph (Tarjan)."""
    index: dict[Any, int] = {}
    low: dict[Any, int] = {}
    stack: list = []
    on_stack: set = set()
    components = []
    for root in list(succ):
        if root in index:
            continue
        index[root] = low[root] = len(index)
        stack.append(root)
        on_stack.add(root)
        work = [(root, iter(succ.get(root, ())))]
        while work:
            v, it = work[-1]
            for w in it:
                if w not in index:
                    index[w] = low[w] = len(index)
                    stack.append(w)
                    on_stack.add(w)
                    work.append((w, iter(succ.get(w, ()))))
                    break
                if w in on_stack:
                    low[v] = min(low[v], index[w])
            else:
                work.pop()
                if work:
                    low[work[-1][0]] = min(low[work[-1][0]], low[v])
                if low[v] == index[v]:
                    component = []
                    while True:
                        w = stack.pop()
                        on_stack.discard(w)
                        component.append(w)
                        if w == v:
                            break
                    components.append(component)
    return components


def _break_cycles(
    instances: list[list[str]], inputs: dict[str, list[str]], order: dict[str, int]
) -> list[list[str]]:
    """Split instances until the graph with each instance as one node is acyclic:
    the lift passes replace instances by single calls, and an instance on a cycle
    through another (a path from it into the other and back, e.g. through a part of
    the other that joined it without a path between them) cannot become one call.

    Each cycle is cut at the instance where the cut is smallest: a member that the
    cycle enters, and the members after it in the instance."""
    while True:
        owner = {m: k for k, members in enumerate(instances) for m in members}

        def vertex(name: str) -> int | str:
            return owner.get(name, name)

        succ: dict[Any, set] = defaultdict(set)
        for name, ins in inputs.items():
            v = vertex(name)
            for i in ins:
                w = vertex(i)
                if w != v:
                    succ[w].add(v)
        cycles = [set(c) for c in _strongly_connected(succ) if len(c) > 1]
        if not cycles:
            return instances
        cut_now: set[int] = set()
        for cycle in cycles:
            cuts = []
            for k in (v for v in cycle if isinstance(v, int) and v not in cut_now):
                members = instances[k]
                entries = [
                    m
                    for m in members
                    if any(vertex(i) != k and vertex(i) in cycle for i in inputs[m])
                ]
                for entry in entries:
                    cut = {entry}
                    for m in members:  # graph order
                        if m not in cut and any(i in cut for i in inputs[m]):
                            cut.add(m)
                    if len(cut) < len(members):
                        cuts.append((len(cut), -order[entry], k, cut))
            if not cuts:
                continue  # the instances it can cut were cut for another cycle
            _, _, k, cut = min(cuts, key=lambda c: c[:3])
            instances.append([m for m in instances[k] if m in cut])
            instances[k] = [m for m in instances[k] if m not in cut]
            cut_now.add(k)
        assert cut_now, "a cycle through instances with no member to cut"


def find_regions(root: str) -> tuple[float, list[dict]]:
    """The profit floor and the regions of the capture in ``root``, in the order
    the lift passes run, each with its ``Pattern`` (or None), the node names of its
    instances (``members``) and their measured and ideal time
    (``instance_stats``)."""
    with gzip.open(os.path.join(root, "graph.jsonl.gz"), "rt") as f:
        nodes = {r["name"]: r for r in map(json.loads, f)}
    with open(os.path.join(root, "node_table.csv")) as f:
        table = {r["name"]: r for r in csv.DictReader(f)}
    with open(os.path.join(root, "step_budget.json")) as f:
        budget = json.load(f)
    with open(os.path.join(root, "hbm_bandwidth.json")) as f:
        hbm_gbps = json.load(f)["hbm_gbps"]
    order = {name: i for i, name in enumerate(nodes)}

    fx_inputs = {
        name: list(dict.fromkeys(input_nodes([n["args"], n["kwargs"]])))
        for name, n in nodes.items()
    }
    fx_users = defaultdict(list)
    for name in nodes:
        for a in fx_inputs[name]:
            fx_users[a].append(name)
    inputs = {name: list(a) for name, a in fx_inputs.items()}
    # Storage edges: FX has no edge from a write into a buffer to the buffer's other
    # readers and writers. Views (nodes that alias without writing) access no data.
    base = {}
    for name, n in nodes.items():
        base[name] = base[n["aliases"][0]] if n["aliases"] else name
    accesses = defaultdict(list)
    for name, n in nodes.items():
        if n["writes"] or not n["aliases"]:
            for storage in dict.fromkeys(base[a] for a in inputs[name]):
                accesses[storage].append(name)
    for name, n in nodes.items():
        for storage in dict.fromkeys(base[w] for w in n["writes"]):
            for other in accesses[storage]:
                first, then = sorted((name, other), key=order.__getitem__)
                if first != then and first not in inputs[then]:
                    inputs[then].append(first)
    users = defaultdict(list)
    for name in nodes:
        for a in inputs[name]:
            users[a].append(name)

    def category(name: str) -> str:
        return table[name]["category"] if name in table else "input"

    def on_gpu(name: str) -> bool:
        ts = tensors(nodes[name]["val"])
        return bool(ts) and all(t["device"] == "cuda" for t in ts)

    candidates = sorted(
        (
            name
            for name in nodes
            if category(name) in ("glue", "no_kernel")
            and on_gpu(name)
            and nodes[name]["before_inductor"]
        ),
        key=order.__getitem__,
    )
    is_candidate = set(candidates)
    boundaries = [
        name
        for name in nodes
        if name not in is_candidate
        and category(name) != "input"
        and nodes[name]["op"] != "output"
    ]
    bit = {name: 1 << i for i, name in enumerate(boundaries)}
    # Bitsets of the boundaries upstream and downstream of each node (graph order is
    # a topological order).
    upstream, downstream = {}, {}
    for name in nodes:
        mask = bit.get(name, 0)
        for a in inputs[name]:
            mask |= upstream[a]
        upstream[name] = mask
    for name in reversed(nodes):
        mask = bit.get(name, 0)
        for u in users[name]:
            mask |= downstream[u]
        downstream[name] = mask

    # Merge along glue edges unless a boundary would sit on a path between the two
    # sides. A forward value read by recompute or backward is a saved activation:
    # fusing across that edge would move backward work into the forward pass, so
    # those edges are never merged (backward nodes never feed forward ones, so this
    # keeps instances closed under paths).
    parent = {name: name for name in candidates}
    up = {name: upstream[name] for name in candidates}
    down = {name: downstream[name] for name in candidates}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    late = {name: nodes[name]["phase"] != "forward" for name in nodes}
    # A view of a graph input is like the input: it does not join the nodes that read
    # it. (E.g. every MoE layer reads select(arg, 0, 0); joining through it would make
    # one instance of all layers.)
    input_views: set[str] = set()
    for name in candidates:
        n = nodes[name]
        if (
            n["aliases"]
            and not n["writes"]
            and all(
                nodes[a]["op"] in ("placeholder", "get_attr") or a in input_views
                for a in n["aliases"]
            )
        ):
            input_views.add(name)
    for name in candidates:
        for a in inputs[name]:
            if a not in is_candidate or a in input_views or late[a] != late[name]:
                continue
            ra, rb = find(a), find(name)
            if ra == rb or down[ra] & up[rb] or down[rb] & up[ra]:
                continue
            parent[ra] = rb
            up[rb] |= up[ra]
            down[rb] |= down[ra]
    groups = defaultdict(list)
    for name in candidates:
        groups[find(name)].append(name)
    kernel_us = {name: float(table[name]["kernel_us"]) for name in table}
    instances = [
        members for members in groups.values() if any(kernel_us[m] > 0 for m in members)
    ]
    instances = [
        members
        for members in _break_cycles(instances, inputs, order)
        if any(kernel_us[m] > 0 for m in members)
    ]

    def out_bytes(name: str) -> int:
        # Sizes the trace did not resolve count as 0 (node_table.py reports them).
        return int(table[name]["out_bytes"] or 0) if name in table else 0

    def measure(members: list[str]) -> dict:
        mset = set(members)
        nbytes = 0
        for a in dict.fromkeys(
            a for m in members for a in fx_inputs[m] if a not in mset
        ):
            readers = [m for m in fx_users[a] if m in mset]
            # Read through views only: the views' bytes, e.g. one slice of a tensor.
            if all(kernel_us[m] == 0 for m in readers):
                nbytes += min(out_bytes(a), sum(out_bytes(m) for m in readers))
            else:
                nbytes += out_bytes(a)
        written = set()
        for m in members:
            if all(u in mset for u in fx_users[m]):
                continue
            # A view used outside: the kernel that wrote its storage.
            while kernel_us[m] == 0 and fx_inputs[m] and fx_inputs[m][0] in mset:
                m = fx_inputs[m][0]
            if kernel_us[m] > 0 and m not in written:
                written.add(m)
                nbytes += out_bytes(m)
        return {
            "measured_us": sum(kernel_us[m] for m in members),
            "ideal_us": nbytes / (hbm_gbps * 1e3),
            "kernels": sum(int(table[m]["num_kernels"]) for m in members),
        }

    skeletons = {}

    def skeleton(name: str) -> tuple[str, list[str]]:
        if name not in skeletons:
            skeletons[name] = key_skeleton(nodes[name]["match_key"])
        return skeletons[name]

    def pattern_of(members: list[str]) -> Pattern | None:
        mset = set(members)
        try:
            return Pattern(
                {m: nodes[m]["match_key"] for m in members},
                [m for m in members if any(u not in mset for u in fx_users[m])],
            )
        except ValueError:
            return None

    # Instances with the same topology are one region: the first instance's Pattern
    # matches each. Instances whose members are joined only through a storage have
    # no pattern; they are one region when their match keys are the same.
    groups_by_keys = defaultdict(list)  # sorted member skeletons -> [(pattern, group)]
    for members in instances:
        pattern = pattern_of(members)
        within = set(members)
        entries = groups_by_keys[json.dumps(sorted(skeleton(m)[0] for m in members))]
        for other, group in entries:
            if (pattern is None) != (other is None):
                continue
            # Same member count: a match within the instance covers it.
            if other is None or any(
                other.match(m, skeleton, fx_users.__getitem__, within)
                for m in members
                if skeleton(m)[0] == other.members[other.anchor][0]
            ):
                group.append(members)
                break
        else:
            entries.append((pattern, [members]))

    regions = []
    for pattern, group in (e for entries in groups_by_keys.values() for e in entries):
        stats = [measure(members) for members in group]
        measured_ms = sum(s["measured_us"] for s in stats) / 1e3
        ideal_ms = sum(s["ideal_us"] for s in stats) / 1e3
        first = group[0]
        every = [m for members in group for m in members]
        ops = Counter(_short_op(nodes[m]["target"]) for m in first if kernel_us[m] > 0)
        regions.append(
            {
                "gap_ms": round(max(measured_ms - ideal_ms, 0.0), 3),
                "measured_ms": round(measured_ms, 3),
                "ideal_ms": round(ideal_ms, 3),
                "instances": len(group),
                "kernels_per_instance": stats[0]["kernels"],
                "phases": sorted({nodes[m]["phase"] for m in every}),
                "modules": sorted({_any_layer(nodes[m]["module_fqn"]) for m in every}),
                "ops": dict(ops.most_common()),
                # The first instance as it appears in the dumped FX code, one line
                # per node.
                "code": [nodes[m]["code"] for m in first],
                "pattern": pattern,
                "members": group,
                "instance_stats": stats,
            }
        )

    # The lift passes run in this order, and a node an earlier pass lifted is not
    # matched again. So a region comes before every region whose pattern also
    # matches nodes of its instances, unless both match nodes of each other's: then
    # the one with the larger gap goes first. Otherwise regions go by gap.
    regions.sort(key=lambda r: -r["gap_ms"])
    region_of = {m: k for k, r in enumerate(regions) for ms in r["members"] for m in ms}
    by_target = defaultdict(list)
    for name in nodes:
        by_target[nodes[name]["target"]].append(name)
    before = defaultdict(set)  # region -> the regions that must come before it
    for k, r in enumerate(regions):
        if r["pattern"] is None:
            continue
        own = {frozenset(ms) for ms in r["members"]}
        pattern = r["pattern"]
        for bound in pattern.find(
            by_target[pattern.target], skeleton, fx_users.__getitem__
        ):
            matched = frozenset(bound[m] for m in pattern.members)
            if matched not in own:
                before[k] |= {region_of[n] for n in matched if n in region_of} - {k}
    after = defaultdict(set)
    for k, ks in before.items():
        for j in ks:
            after[j].add(k)
    for cycle in _strongly_connected(after):
        for k in cycle:
            before[k] -= {j for j in cycle if j > k}
    ordered: list[int] = []

    def place(k: int) -> None:
        if k not in ordered:
            for j in sorted(before[k]):
                place(j)
            ordered.append(k)

    for k in range(len(regions)):
        place(k)
    regions = [regions[k] for k in ordered]
    for i, r in enumerate(regions):
        r["id"] = f"r{i:03d}"
    floor_ms = max(_FLOOR_MS, _FLOOR_PCT_OF_BUSY / 100 * budget["gpu_busy_ms"])
    return floor_ms, regions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("capture_dir")
    args = parser.parse_args()
    floor_ms, regions = find_regions(args.capture_dir)
    with open(os.path.join(args.capture_dir, "regions.json"), "w") as f:
        json.dump(
            {
                "profit_floor_ms": round(floor_ms, 3),
                "regions": [
                    {
                        **{
                            k: v
                            for k, v in r.items()
                            if k not in ("members", "instance_stats")
                        },
                        "pattern": r["pattern"] and r["pattern"].to_json(),
                    }
                    for r in regions
                ],
            },
            f,
            indent=1,
        )

    above = [r for r in regions if r["gap_ms"] >= floor_ms]
    total_gap = sum(r["gap_ms"] for r in regions)
    print(
        f"{len(regions)} regions ({sum(r['instances'] for r in regions)} instances), "
        f"gap {total_gap:.1f} ms; {len(above)} above the profit floor of "
        f"{floor_ms:.2f} ms hold {sum(r['gap_ms'] for r in above):.1f} ms"
    )
    print(
        f"{'id':5s} {'gap_ms':>8s} {'meas_ms':>8s} {'inst':>5s} {'k/inst':>6s}  phases  modules  ops"
    )
    for r in above:
        ops = ", ".join(f"{op} x{c}" for op, c in list(r["ops"].items())[:4])
        more = f" (+{len(r['ops']) - 4} ops)" if len(r["ops"]) > 4 else ""
        print(
            f"{r['id']:5s} {r['gap_ms']:8.2f} {r['measured_ms']:8.2f} {r['instances']:5d} "
            f"{r['kernels_per_instance']:6d}  {','.join(r['phases'])}  "
            f"{' '.join(r['modules'][:2])}{' ...' if len(r['modules']) > 2 else ''}  {ops}{more}"
        )


if __name__ == "__main__":
    main()

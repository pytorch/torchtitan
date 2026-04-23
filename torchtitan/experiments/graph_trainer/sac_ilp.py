# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ILP-based selective activation checkpointing (SAC) policy.

Formulates save-vs-recompute decisions as an Integer Linear Program (ILP):
minimize total recomputation time subject to a memory budget. The solver
runs once at compile time and produces a mapping from FX node name to
``CheckpointPolicy``.

Requires PuLP (``pip install pulp``) as an optional dependency.

Design:
    1. Scan the forward graph for activation nodes that have backward consumers.
    2. Classify each node: view-like (always recompute), comm ops (always save),
       random ops (always save), and general ops (candidates for ILP).
    3. For candidate nodes, estimate memory cost and recomputation time from
       shape metadata.
    4. Solve per-layer ILP: minimize recomputation time subject to activation
       memory <= budget.
    5. Return a closure ``node_name -> CheckpointPolicy``.
"""

from __future__ import annotations

import operator
from collections import defaultdict

import torch
import torch.fx
from torch._functorch.partitioners import get_default_op_list
from torch.utils.checkpoint import CheckpointPolicy

from torchtitan.experiments.graph_trainer.common_utils import (
    _get_layer_id,
    _is_backward_node,
    _NOT_IN_LAYERS,
)
from torchtitan.tools.logging import logger

# Op classification from upstream partitioner
_OP_LIST = get_default_op_list()

_VIEW_OPS = frozenset(_OP_LIST.view_ops)
_RANDOM_OPS = frozenset(_OP_LIST.random_ops)
_COMPUTE_INTENSIVE_OPS = frozenset(_OP_LIST.compute_intensive_ops)

# Communication ops that should always be saved to avoid re-communication.
_COMM_NAMESPACES = frozenset({"_c10d_functional"})


def _get_overload_packet(target: object) -> object:
    """Return the overloadpacket for overload-agnostic matching."""
    if hasattr(target, "overloadpacket"):
        return target.overloadpacket
    return target


def _is_view_op(node: torch.fx.Node) -> bool:
    return _get_overload_packet(node.target) in _VIEW_OPS


def _is_random_op(node: torch.fx.Node) -> bool:
    return _get_overload_packet(node.target) in _RANDOM_OPS


def _is_comm_op(node: torch.fx.Node) -> bool:
    ns = getattr(node.target, "namespace", None)
    return ns in _COMM_NAMESPACES


def _is_compute_intensive_op(node: torch.fx.Node) -> bool:
    return _get_overload_packet(node.target) in _COMPUTE_INTENSIVE_OPS


def _tensor_bytes(val: torch.Tensor) -> int:
    """Get tensor size in bytes, handling symbolic shapes gracefully."""
    try:
        return int(val.nelement() * val.element_size())
    except Exception:
        # Symbolic shapes (e.g. DSv3 MoE) may fail int() conversion
        return 0


def _estimate_recompute_cost(node: torch.fx.Node) -> float:
    """Estimate relative recomputation cost for a node.

    Uses a simple heuristic based on op type and output size:
    - Compute-intensive ops (mm, bmm, sdpa, etc.): cost proportional to output
      element count, scaled by a large multiplier reflecting their high FLOP cost.
    - Elementwise / reduction ops: cost proportional to output element count.
    - View-like ops: zero cost (they are free to recompute).
    """
    if _is_view_op(node):
        return 0.0

    val = node.meta.get("val")
    if not isinstance(val, torch.Tensor):
        return 0.0

    try:
        numel = int(val.nelement())
    except Exception:
        numel = 0

    if numel == 0:
        return 0.0

    if _is_compute_intensive_op(node):
        # mm/bmm/sdpa are ~100x more expensive per element than pointwise
        return float(numel) * 100.0

    # General elementwise / reduction ops
    return float(numel)


def _has_backward_consumer(node: torch.fx.Node) -> bool:
    """Check if any user of this node is a backward node."""
    return any(_is_backward_node(u) for u in node.users)


# ============================================================
# Node stats collection
# ============================================================


class _NodeStats:
    """Per-node metadata for ILP formulation."""

    __slots__ = ("name", "memory_bytes", "recompute_cost", "forced_policy")

    def __init__(
        self,
        name: str,
        memory_bytes: int,
        recompute_cost: float,
        forced_policy: CheckpointPolicy | None,
    ):
        self.name = name
        self.memory_bytes = memory_bytes
        self.recompute_cost = recompute_cost
        self.forced_policy = forced_policy


def _collect_node_stats(
    gm: torch.fx.GraphModule,
) -> dict[int, list[_NodeStats]]:
    """Collect per-node stats grouped by layer ID.

    Returns a dict mapping layer_id -> list of _NodeStats for nodes that
    are forward call_function nodes with backward consumers (i.e., activations
    that would be saved for backward).

    Nodes with forced policies (views, comms, random ops) have
    ``forced_policy`` set and are excluded from ILP optimization.
    """
    layer_stats: dict[int, list[_NodeStats]] = defaultdict(list)

    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if _is_backward_node(node):
            continue

        # Skip getitem and wait_tensor -- they inherit parent's policy
        if node.target in (
            operator.getitem,
            torch.ops._c10d_functional.wait_tensor.default,
        ):
            continue

        # Only consider nodes whose outputs are consumed by backward
        if not _has_backward_consumer(node):
            continue

        layer_id = _get_layer_id(node)
        val = node.meta.get("val")

        memory_bytes = 0
        if isinstance(val, torch.Tensor):
            memory_bytes = _tensor_bytes(val)

        recompute_cost = _estimate_recompute_cost(node)

        # Determine forced policy
        forced_policy = None
        if _is_view_op(node):
            forced_policy = CheckpointPolicy.PREFER_RECOMPUTE
        elif _is_comm_op(node):
            forced_policy = CheckpointPolicy.MUST_SAVE
        elif _is_random_op(node):
            # Random ops must be saved to preserve determinism
            forced_policy = CheckpointPolicy.MUST_SAVE

        layer_stats[layer_id].append(
            _NodeStats(
                name=node.name,
                memory_bytes=memory_bytes,
                recompute_cost=recompute_cost,
                forced_policy=forced_policy,
            )
        )

    return layer_stats


# ============================================================
# ILP solver
# ============================================================


def _solve_layer_ilp(
    nodes: list[_NodeStats],
    memory_budget: float,
) -> dict[str, CheckpointPolicy]:
    """Solve the ILP for a single layer's activation nodes.

    Minimizes total recomputation time subject to:
        sum(memory_i * save_i) <= memory_budget * total_memory

    where save_i is a binary variable (1 = save, 0 = recompute).

    Args:
        nodes: Per-node stats for this layer.
        memory_budget: Fraction (0.0 to 1.0) of total activation memory
            allowed. 0.0 = recompute everything, 1.0 = save everything.

    Returns:
        Dict mapping node name -> CheckpointPolicy.
    """
    try:
        import pulp
    except ImportError as e:
        raise ImportError(
            "PuLP is required for ILP-based SAC policy. "
            "Install it with: pip install pulp"
        ) from e

    result: dict[str, CheckpointPolicy] = {}

    # Separate forced nodes from candidates
    candidates: list[_NodeStats] = []
    for node_stat in nodes:
        if node_stat.forced_policy is not None:
            result[node_stat.name] = node_stat.forced_policy
        else:
            candidates.append(node_stat)

    if not candidates:
        return result

    # Total memory if all candidates were saved
    total_candidate_memory = sum(n.memory_bytes for n in candidates)

    if total_candidate_memory == 0:
        # No memory cost -- save everything
        for n in candidates:
            result[n.name] = CheckpointPolicy.MUST_SAVE
        return result

    # Memory from forced MUST_SAVE nodes counts against the budget
    forced_save_memory = sum(
        n.memory_bytes for n in nodes if n.forced_policy == CheckpointPolicy.MUST_SAVE
    )
    total_possible_memory = total_candidate_memory + forced_save_memory

    # Available memory budget for candidates (subtract forced saves)
    budget_bytes = memory_budget * total_possible_memory - forced_save_memory
    budget_bytes = max(0.0, budget_bytes)

    # If budget allows saving everything, skip the solver
    if budget_bytes >= total_candidate_memory:
        for n in candidates:
            result[n.name] = CheckpointPolicy.MUST_SAVE
        return result

    # If budget is zero, recompute everything
    if budget_bytes <= 0:
        for n in candidates:
            result[n.name] = CheckpointPolicy.PREFER_RECOMPUTE
        return result

    # Formulate ILP
    prob = pulp.LpProblem("SAC_layer", pulp.LpMinimize)

    # Binary variables: save_i = 1 means save, 0 means recompute
    save_vars = {
        n.name: pulp.LpVariable(f"save_{n.name}", cat=pulp.constants.LpBinary)
        for n in candidates
    }

    # Objective: minimize total recomputation cost
    # recompute cost of node i = cost_i * (1 - save_i)
    # = sum(cost_i) - sum(cost_i * save_i)
    # Since sum(cost_i) is a constant, minimizing recompute cost is
    # equivalent to maximizing sum(cost_i * save_i).
    # PuLP minimizes, so we negate: minimize -sum(cost_i * save_i)
    prob += pulp.lpSum(-n.recompute_cost * save_vars[n.name] for n in candidates)

    # Constraint: total saved memory <= budget
    prob += (
        pulp.lpSum(n.memory_bytes * save_vars[n.name] for n in candidates)
        <= budget_bytes
    )

    # Solve silently
    solver = pulp.PULP_CBC_CMD(msg=0)
    status = prob.solve(solver)

    if status != pulp.constants.LpStatusOptimal:
        logger.warning(
            f"ILP solver did not find optimal solution (status={status}). "
            "Falling back to recompute-all for this layer."
        )
        for n in candidates:
            result[n.name] = CheckpointPolicy.PREFER_RECOMPUTE
        return result

    # Extract solution
    for n in candidates:
        var = save_vars[n.name]
        if pulp.value(var) is not None and pulp.value(var) > 0.5:
            result[n.name] = CheckpointPolicy.MUST_SAVE
        else:
            result[n.name] = CheckpointPolicy.PREFER_RECOMPUTE

    return result


# ============================================================
# Top-level policy factory
# ============================================================


def solve_ilp_policy(
    gm: torch.fx.GraphModule,
    memory_budget: float,
) -> dict[str, CheckpointPolicy]:
    """Run ILP solver on the graph and return per-node policy decisions.

    Solves one ILP per layer. Nodes not in any layer are defaulted to
    PREFER_RECOMPUTE. View-like ops always recompute, comm ops and random ops
    always save.

    Args:
        gm: The joint forward-backward graph module.
        memory_budget: Fraction of total activation memory to allow (0.0-1.0).
            0.0 = recompute everything, 1.0 = save everything.

    Returns:
        Dict mapping node name -> CheckpointPolicy for all forward activation
        nodes with backward consumers.
    """
    if not 0.0 <= memory_budget <= 1.0:
        raise ValueError(
            f"memory_budget must be between 0.0 and 1.0, got {memory_budget}"
        )

    layer_stats = _collect_node_stats(gm)
    all_policies: dict[str, CheckpointPolicy] = {}

    for layer_id, nodes in sorted(layer_stats.items()):
        layer_policies = _solve_layer_ilp(nodes, memory_budget)
        all_policies.update(layer_policies)

        # Log per-layer summary
        num_save = sum(
            1 for p in layer_policies.values() if p == CheckpointPolicy.MUST_SAVE
        )
        num_recompute = sum(
            1 for p in layer_policies.values() if p != CheckpointPolicy.MUST_SAVE
        )
        save_bytes = sum(
            n.memory_bytes
            for n in nodes
            if layer_policies.get(n.name) == CheckpointPolicy.MUST_SAVE
        )
        recompute_bytes = sum(
            n.memory_bytes
            for n in nodes
            if layer_policies.get(n.name) != CheckpointPolicy.MUST_SAVE
        )
        label = "non-layer" if layer_id == _NOT_IN_LAYERS else str(layer_id)
        logger.info(
            f"  ILP layer {label}: "
            f"{num_save} MUST_SAVE ({save_bytes / 1024:.1f} KB), "
            f"{num_recompute} PREFER_RECOMPUTE ({recompute_bytes / 1024:.1f} KB)"
        )

    return all_policies

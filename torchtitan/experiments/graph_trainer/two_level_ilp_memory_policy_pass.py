# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Two-level per-tensor keep/recompute/offload ILP (work in progress).

Plan (mirrors torch's sac_milp two-level decomposition, extended with offload):

  Outer ILP -- budget allocation across transformer blocks. One variable per
    block (a GPU-keep budget), coupled by the global peak-memory constraint.
  Inner ILP -- per-block three-way keep/recompute/offload. Given its allocated
    budget, each block solves a small independent ILP.

This file currently implements the pieces the outer ILP builds on:
  * Step 1: group graph nodes by transformer block (block_of_node /
    group_nodes_by_block).
  * Step 2: the inner ILP (solve_inner_block) and the per-block tradeoff CURVE
    (build_block_curve) -- added_time as a convex piecewise-linear function of
    kept activation bytes. Structurally identical blocks are solved ONCE and
    reused (build_all_block_curves).

The outer budget-allocation ILP and graph tagging are TODO.
"""

import operator
import time
from collections import defaultdict
from dataclasses import dataclass

import torch
import torch.utils._pytree as pytree
from pulp import (
    LpBinary,
    LpMinimize,
    LpProblem,
    LpStatus,
    lpSum,
    LpVariable,
    PULP_CBC_CMD,
)
from torch.fx.node import map_arg
from torch.utils.checkpoint import CheckpointPolicy

# torchinsights splits these: the tensor-category constants live in the
# _fx_utils submodule, while the estimator entry points are re-exported at the
# graph_estimation package top level.
from torchinsights.graph_estimation import (
    estimate_peak_memory,
    MemoryEstimatorResult,
    optimizer_state_bytes,
)
from torchinsights.graph_estimation._fx_utils import ACT, GRAD, INPUT, PARAM, TEMP

# from torchtitan.experiments.graph_trainer.runtime_estimator import (
#     COST_MODEL,
#     INTERPRETER,
#     RuntimeEstimator,
#     RuntimeEstimatorResult,
# )
from torchinsights.graph_estimation.runtime_estimator import (
    COST_MODEL,
    INTERPRETER,
    RuntimeEstimator,
)

from torchtitan.experiments.graph_trainer.common_utils import (
    _get_layer_id,
    _is_backward_node,
    _MODULE_FQN,
)
from torchtitan.experiments.graph_trainer.cpu_offload import (
    _can_offload_node,
    _is_collective_or_wait,
    _is_view,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import TracedResult
from torchtitan.experiments.graph_trainer.transfertime_estimator import get_transfer_bw
from torchtitan.tools.logging import logger

# meta["recompute"] tag for each per-tensor policy decision.
_POLICY_TAG = {
    "keep": CheckpointPolicy.MUST_SAVE,
    "recompute": CheckpointPolicy.MUST_RECOMPUTE,
    "offload": CheckpointPolicy.MUST_CPU_OFFLOAD,
}


# Save-op anchoring policies for get_must_keep_list (which compute-heavy
# save_ops are barred from recompute, i.e. forced to keep-or-offload).
SAVE_OPS_ALL = "all"  # anchor every save_op (whole-graph solvers)
SAVE_OPS_MATMUL_RECOMPUTABLE = "matmul_recomputable"  # inner solver
SAVE_OPS_NONE = "none"  # anchor no save_op

# The matmul family is the ONE class of save_op safe to leave recomputable in
# the per-layer inner solver: BENCHMARK mode times these kernels accurately, so
# the runtime objective can be trusted to keep/offload the costly ones; and they
# are deterministic, so recompute is numerically sound. Every other save_op is
# anchored under SAVE_OPS_MATMUL_RECOMPUTABLE because the estimator prices it
# unreliably and/or it is unsafe or wasteful to recompute:
#   - attention (SDPA family, flex_attention HOP): benchmark can't micro-time a
#     HOP (falls back to the roofline, which under-counts real time; a fused HOP
#     loses its flop entry entirely), and re-running full attention in backward
#     is far costlier than offloading its output.
#   - HOPs / fused inductor code: same costing blind spot.
#   - topk / comm collectives: nondeterministic or re-communicating -> unsafe.
_MATMUL_OVERLOAD_PACKETS = frozenset(
    {
        torch.ops.aten.mm,
        torch.ops.aten.addmm,
        torch.ops.aten.bmm,
        torch.ops.aten.baddbmm,
        torch.ops.aten.linear,
        torch.ops.aten._scaled_mm,
    }
)


# Scale all memory terms to GiB inside the ILP (sac_milp's MEM_MULTIPLIER). Raw
# byte coefficients (~1e10) mixed with runtimes (~1e2) give CBC a coefficient
# spread that breaks its tolerances and yields constraint-violating "Optimal"
# solutions; GiB units keep every coefficient O(1-30).
MEM_MULTIPLIER = 1 << 30

# Offload only tensors at least this large: fewer, larger transfers hold less
# per-tensor overhead at the backward peak (matches the sac_and_offload default).
OFFLOAD_MIN_BYTES = 1 << 20  # 1 MiB

val_of = lambda n: n.meta.get("val", None)  # noqa: E731
INT64_MAX = (1 << 63) - 1


@dataclass(eq=False)
class StorageObject:
    sid: int
    size: int  # in bytes
    producer_node: torch.fx.Node
    produced_index: int
    death_index: int
    last_fwd_use_index: int
    first_bwd_use_index: int
    category: str  # PARAM, GRAD, ACT, TEMP, INPUT


def _is_rng_op(node: torch.fx.Node) -> bool:
    """RNG ops cannot be replayed by the remat pass, so they must never be
    recomputed (they may still be kept or offloaded)."""
    return torch.Tag.nondeterministic_seeded in getattr(node.target, "tags", set())


# ---------------------------------------------------------------------------
# Step 1: group nodes by transformer block
# ---------------------------------------------------------------------------
def block_of_node(node: torch.fx.Node) -> int:
    """For now, this is just a wrapper for _get_layer_id:
    layers.<N> -> N: int
    """
    return _get_layer_id(node)
    # fqn = node.meta.get("custom", {}).get(_MODULE_FQN, "")
    # parts = fqn.split(".")
    # if len(parts) >= 2 and parts[0] == "layers" and parts[1].isdigit():
    #     return f"layers.{parts[1]}"
    # return None


def group_nodes_by_block(gm: torch.fx.GraphModule) -> dict[str, list[torch.fx.Node]]:
    """Map each transformer-block FQN to its forward call_function nodes."""
    blocks: dict[int, list[torch.fx.Node]] = defaultdict(list)
    for node in gm.graph.nodes:
        if node.op != "call_function" or _is_backward_node(node):
            continue
        block = block_of_node(node)
        if block is not None and block != -1:
            blocks[block].append(node)
    return dict(blocks)


def get_must_keep_list(
    gm: torch.fx.GraphModule, *, save_ops_policy: str = SAVE_OPS_ALL
) -> set:
    """Nodes that must NOT be recomputed (they can still keep or offload):
    - RNG ops (cannot be replayed by the remat pass). HARD correctness
      constraint -- the remat pass cannot reproduce the random state.
    - compute-heavy save_ops, selected by ``save_ops_policy``:
        SAVE_OPS_ALL                 -- anchor every save_op (matmul + attention
          + HOPs + comm + topk). Default; used by the whole-graph solvers.
        SAVE_OPS_MATMUL_RECOMPUTABLE -- anchor every save_op EXCEPT the matmul
          family. Used by the per-layer inner solver: matmuls are left
          recomputable because BENCHMARK costs them accurately (the runtime
          objective then keeps/offloads the costly ones), while attention/HOP/
          comm/topk stay anchored because the estimator prices them unreliably
          or they are unsafe/wasteful to recompute. See _save_op_is_anchored.
        SAVE_OPS_NONE                -- anchor no save_op.
    - layer boundaries: a forward node whose output feeds a forward user in a
      HIGHER layer. Anchored for ALL policies -- not recomputing it keeps each
      layer's recompute region self-contained, which is what makes the per-layer
      inner solves independent; recomputing a deep layer would chain back
      through all previous layers and blow up the backward working set.
    """
    from torchtitan.distributed.activation_checkpoint import _get_default_save_ops

    save_ops = _get_default_save_ops()

    must_keep = set()
    for node in gm.graph.nodes:
        if node.op != "call_function" or _is_backward_node(node):
            continue
        if _is_rng_op(node):
            must_keep.add(node)
            continue
        if node.target in save_ops and _save_op_is_anchored(node, save_ops_policy):
            must_keep.add(node)
            continue
        node_layer = _get_layer_id(node)
        for user in node.users:
            if not _is_backward_node(user) and _get_layer_id(user) > node_layer:
                must_keep.add(node)
                break
    return must_keep


def _collect_layer_candidates(gm, mem_est, nodes, node_index):
    """Per-layer activation candidates plus the non-freeable resident baseline.

    Built from ``mem_est.all_tensors`` (the estimator's own saved-for-backward
    ACT classification, already deduped by storage lifetime and attributed to
    each storage's base producer node -- views alias the base and never appear
    as producers). One decision per producer node; multiple storages of a node
    are summed (the node is tagged as a unit).

    Returns (candidates_by_layer: {layer: [(node, size_bytes)]},
             layer_act_bytes: {layer: bytes},
             fixed_bytes: non-freeable resident bytes at the all-keep peak).
    """
    per_layer_node_bytes = defaultdict(lambda: defaultdict(int))
    seen = set()
    total_freeable = 0
    for producer, entries in mem_est.all_tensors.items():
        b = block_of_node(producer)
        if b is None or b == -1:
            continue  # non-block activation -> part of the fixed baseline
        if _is_view(producer):
            continue  # cannot tag a view independently
        for e in entries:
            if e["category"] != ACT or e["first_bwd_use"] is None:
                continue  # not a saved-for-backward activation
            # Dedup on (sid, birth), matching the estimator's own liveness key:
            # a storage id that is freed and later reused starts a fresh
            # interval and is a legitimately distinct activation. Keying on the
            # bare sid would drop that second interval and undercount memory.
            skey: tuple[Unknown, Unknown] = (e["sid"], e["birth"])
            if skey in seen:
                continue
            seen.add(skey)
            per_layer_node_bytes[b][producer] += e["size"]
            total_freeable += e["size"]

    candidates_by_layer = {
        b: list(nb.items()) for b, nb in per_layer_node_bytes.items()
    }
    layer_act = {b: sum(sz for _, sz in c) for b, c in candidates_by_layer.items()}
    fixed_bytes = mem_est.peak_bytes - total_freeable
    return candidates_by_layer, layer_act, fixed_bytes


def _save_op_is_anchored(node: torch.fx.Node, policy: str) -> bool:
    """Whether a save_op node must be barred from recompute under ``policy``.
    Assumes ``node.target`` is already known to be a save_op."""
    if policy == SAVE_OPS_ALL:
        return True
    if policy == SAVE_OPS_NONE:
        return False
    # SAVE_OPS_MATMUL_RECOMPUTABLE: anchor everything except the matmul family.
    target = node.target
    if isinstance(target, torch._ops.OpOverload):
        return target._overloadpacket not in _MATMUL_OVERLOAD_PACKETS
    return True  # HOPs (flex_attention, inductor_compiled_code, ...) always anchored


# ---------------------------------------------------------------------------
# Inner ILP: one independent solve per transformer block.
# ---------------------------------------------------------------------------
def _resolve_fractions(
    keep_fraction: float, recompute_fraction: float, offload_fraction: float
) -> tuple[float, float, float]:
    """Read optional env overrides, validate the keep/recompute/offload split
    sums to 1, and return the (possibly overridden) triple.

    Env overrides (handy for sweeps without editing the trainer call):
    AUTOAC_KEEP_FRAC, AUTOAC_RECOMPUTE_FRAC, AUTOAC_OFFLOAD_FRAC.
    """
    import os

    keep_fraction = float(os.environ.get("AUTOAC_KEEP_FRAC", keep_fraction))
    recompute_fraction = float(
        os.environ.get("AUTOAC_RECOMPUTE_FRAC", recompute_fraction)
    )
    offload_fraction = float(os.environ.get("AUTOAC_OFFLOAD_FRAC", offload_fraction))

    total = keep_fraction + recompute_fraction + offload_fraction
    if abs(total - 1.0) > 1e-6:
        raise ValueError(
            f"keep/recompute/offload fractions must sum to 1, got "
            f"keep={keep_fraction}, recompute={recompute_fraction}, "
            f"offload={offload_fraction} (sum={total})"
        )
    if min(keep_fraction, recompute_fraction, offload_fraction) < 0:
        raise ValueError(
            f"fractions must be non-negative, got keep={keep_fraction}, "
            f"recompute={recompute_fraction}, offload={offload_fraction}"
        )
    return keep_fraction, recompute_fraction, offload_fraction


def _classify_tag(node: torch.fx.Node) -> str:
    """keep/recompute/offload bucket for a node's final ``meta["recompute"]``
    tag. MUST_SAVE / PREFER_SAVE / untagged all mean the storage stays resident
    on the GPU (keep)."""
    pol = node.meta.get("recompute")
    if pol in (CheckpointPolicy.MUST_CPU_OFFLOAD, CheckpointPolicy.PREFER_CPU_OFFLOAD):
        return "offload"
    if pol in (CheckpointPolicy.MUST_RECOMPUTE, CheckpointPolicy.PREFER_RECOMPUTE):
        return "recompute"
    return "keep"


def _audit_tagged_fractions(candidates_by_layer):
    """Measure the keep/recompute/offload split ACTUALLY written to the graph,
    by reading each candidate producer's ``meta["recompute"]`` tag rather than
    trusting the solver's variable values.

    Called after the ILP overrides and the default-SAC base tagging, but before
    apply_cpu_offload_pass / selective_activation_remat_pass materialize -- so it
    reflects the final decision that those passes will act on for every
    candidate, INCLUDING layers the ILP skipped (which keep the default-SAC base
    tag) and any base-tag interaction the per-layer solver totals miss.

    Byte attribution matches _collect_layer_candidates (estimator storage sizes),
    so the fractions are directly comparable to the solver-intent numbers.

    Caveat: this is intent-as-tagged. A downstream pass may still reject a tag
    (e.g. the offload pass skips non-contiguous or last-layer tensors); the
    materialized recompute_dups / offload_ops counts are the post-materialization
    ground truth.

    Returns (per_layer: {b: {keep,recompute,offload}}, total: {keep,recompute,offload}).
    """
    per_layer = {}
    total = {"keep": 0, "recompute": 0, "offload": 0}
    for b, candidates in candidates_by_layer.items():
        ach = {"keep": 0, "recompute": 0, "offload": 0}
        for node, size in candidates:
            ach[_classify_tag(node)] += size
        per_layer[b] = ach
        for kind in total:
            total[kind] += ach[kind]
    return per_layer, total


# ---------------------------------------------------------------------------
# overall ilp - outer + inner ILPs
# ---------------------------------------------------------------------------
def two_level_ilp(
    trace: TracedResult,
    memory_budget: int,
    optimizer,
    model_parts: list[torch.nn.Module],
    runtime_estimation_mode: str = COST_MODEL,
    cpu_offload_budget_gb: float = 100.0,
    interp_ctx: tuple | None = None,  # (model, *run_args) for INTERPRETER mode
    each_layer_separately: bool = True,
) -> torch.fx.GraphModule | None:
    """Two-level keep/recompute/offload solver. Tags ``gm`` with per-node decisions."""
    _t0 = time.perf_counter()
    _timings = {}
    gm = trace.gm

    _t = time.perf_counter()
    mem_est = estimate_peak_memory(gm, num_state_inputs=trace.num_static_inputs)
    opt_bytes = optimizer_state_bytes(optimizer, model_parts[0])
    _timings["mem_estimation"] = time.perf_counter() - _t
    estimated = mem_est.peak_bytes + opt_bytes  # all-keep peak (incl. optimizer state)

    if memory_budget > estimated:
        logger.info(
            "new-autoAC: budget %.2f GB >= estimated peak %.2f GB; nothing to do",
            memory_budget / 1e9,
            estimated / 1e9,
        )
        return gm, None

    logger.info(
        "2-level ILP-based autoAC: runtime estimation mode = %s",
        runtime_estimation_mode,
    )
    _t = time.perf_counter()
    if runtime_estimation_mode == INTERPRETER:
        if interp_ctx is None:
            logger.warning(
                "INTERPRETER runtime mode needs interp_ctx=(model, *run_args). Falling back to `COST_MODEL`."
            )
            runtime_estimation_mode = COST_MODEL
            runtime = RuntimeEstimator()(COST_MODEL).estimate(trace)
        else:
            runtime = RuntimeEstimator()(INTERPRETER).estimate(trace, *interp_ctx)
    else:
        runtime = RuntimeEstimator()(runtime_estimation_mode).estimate(trace)
    _timings["runtime_estimation"] = time.perf_counter() - _t

    _t = time.perf_counter()
    get_fixed_bytes_tuple = get_fixed_bytes(gm, trace.num_static_inputs)
    _timings["get_fixed_bytes"] = time.perf_counter() - _t

    # tuple content:
    # (
    #     live_sids_per_index,  # this is per node index live sids
    #     freeable_sids_per_index,  # this is per node index freeable sids
    #     fixed_bytes_max_by_layer,  # this is per layer fixed byes (that we cannot touch)
    #     accumulative_act_bytes_per_layer,  # this is per layer activation bytes (that we can touch)
    #     act_bytes_per_layer,  # this is per layer activation bytes (that we can touch)
    #     storages_by_prod_node,  # this is per layer storages, the key is the layer of the producer node
    #     candidates_by_layer,
    #     layer_act,
    #     total_freeable,
    # )

    _t = time.perf_counter()
    fractions, _t_construct_ilp, _t_solve = plan_outer(
        mem_est,
        opt_bytes,
        runtime,
        get_fixed_bytes_tuple,
        trace,
        memory_budget,
        optimizer,
        model_parts,
        runtime_estimation_mode,
        cpu_offload_budget_gb,
        interp_ctx,  # (model, *run_args) for INTERPRETER mode
    )
    _timings["outer_ilp"] = time.perf_counter() - _t

    if fractions is None:
        logger.info("ILP: Something went wrong")
        return gm, None

    _t = time.perf_counter()
    gm, block_build_s, block_solve_s, inner_fractions = plan_and_tag_inner(
        mem_est,
        opt_bytes,
        runtime,
        get_fixed_bytes_tuple,
        trace,
        trace.gm,
        runtime_estimation_mode,
        cpu_offload_budget_gb,
        interp_ctx,
        keep_fraction=0.05,
        recompute_fraction=0.90,
        offload_fraction=0.05,
        per_layer_fractions=fractions,
        each_layer_separately=each_layer_separately,
    )
    _timings["inner_ilp"] = time.perf_counter() - _t
    _timings["total"] = time.perf_counter() - _t0
    logger.info(
        "two_level_ilp timing (s): mem_est=%.3f runtime_est=%.3f get_fixed_bytes=%.3f "
        "outer_ilp=%.3f inner_ilp=%.3f | TOTAL=%.3f",
        _timings["mem_estimation"],
        _timings["runtime_estimation"],
        _timings["get_fixed_bytes"],
        _timings["outer_ilp"],
        _timings["inner_ilp"],
        _timings["total"],
    )

    # --- outer-requested fractions, byte-weighted over each layer's owned
    # activation (act_bytes_per_layer = get_fixed_bytes_tuple[4]) ---
    act_by_layer = get_fixed_bytes_tuple[4]
    _tot = sum(act_by_layer.get(L, 0) for L in fractions) or 1
    outer_fractions = {
        "keep": 100
        * sum(fractions[L][0] * act_by_layer.get(L, 0) for L in fractions)
        / _tot,
        "recompute": 100
        * sum(fractions[L][1] * act_by_layer.get(L, 0) for L in fractions)
        / _tot,
        "offload": 100
        * sum(fractions[L][2] * act_by_layer.get(L, 0) for L in fractions)
        / _tot,
    }

    metrics = {
        "num_layers": len(get_fixed_bytes_tuple[6]),  # candidates_by_layer
        "budget_gb": memory_budget / 1e9,
        "each_layer_separately": each_layer_separately,
        "runtime_mode": runtime_estimation_mode,
        # stage times (s)
        "t_mem_est": _timings["mem_estimation"],
        "t_runtime_est": _timings["runtime_estimation"],
        "t_get_fixed_bytes": _timings["get_fixed_bytes"],
        "t_outer_total": _timings["outer_ilp"],
        "t_outer_construct": _t_construct_ilp,
        "t_outer_solve": _t_solve,
        "t_inner_total": _timings["inner_ilp"],
        "t_inner_build": block_build_s,
        "t_inner_solve": block_solve_s,
        "t_total": _timings["total"],
        # fractions (byte-weighted %): outer requested vs inner achieved
        "outer_keep": outer_fractions["keep"],
        "outer_recompute": outer_fractions["recompute"],
        "outer_offload": outer_fractions["offload"],
        "inner_keep": inner_fractions["keep"],
        "inner_recompute": inner_fractions["recompute"],
        "inner_offload": inner_fractions["offload"],
    }
    return gm, metrics


# ---------------------------------------------------------------------------
# outer ilp over all layers + inner ilp per layer
# ---------------------------------------------------------------------------
def plan_outer(
    mem_est: MemoryEstimatorResult,
    opt_bytes,
    runtime,
    get_fixed_bytes_tuple,
    trace: TracedResult,
    memory_budget: int,
    optimizer,
    model_parts: list[torch.nn.Module],
    runtime_estimation_mode: str = COST_MODEL,
    cpu_offload_budget_gb: float = 100.0,
    interp_ctx: tuple | None = None,  # (model, *run_args) for INTERPRETER mode
):
    """Two-level keep/recompute/offload solver. Tags ``gm`` in place and returns
    it (the caller runs apply_cpu_offload_pass + selective_activation_remat_pass).

    Steps: build per-block tradeoff curves -> outer budget allocation -> tag ->
    measure the REAL post-pass peak on a clone. Because the modeled peak is only
    a proxy (recompute working set omitted; peak-node approximation), we sweep the
    outer cap over ``cap_grid`` multipliers of eff_budget, measure each plan's
    real peak, and pick the one whose real peak is CLOSEST TO the budget from
    below -- i.e. keep the most / free the least -> least added runtime. Returns
    the untouched ``gm`` when the budget already fits, or None when no block
    activations exist / no feasible plan is found.
    """
    gm = trace.gm
    # mem_est = estimate_peak_memory(gm, num_state_inputs=trace.num_static_inputs)
    # opt_bytes = optimizer_state_bytes(optimizer, model_parts[0])
    estimated = mem_est.peak_bytes + opt_bytes
    # if memory_budget > estimated:
    #     logger.info(
    #         "new-autoAC: budget %.2f GB >= estimated peak %.2f GB; nothing to do",
    #         memory_budget / 1e9,
    #         estimated / 1e9,
    #     )
    #     return gm

    eff_budget = memory_budget - opt_bytes

    # --- ABLATION KNOBS (env-driven auto-research; all default to the current model) ---
    import os as _abl_os

    _ABL = _abl_os.environ
    _abl_tiebreak = _ABL.get("ABL_TIEBREAK", "1") == "1"  # objective term 2
    _abl_ws = _ABL.get("ABL_WS", "1") == "1"  # recompute working-set term
    _abl_ws_mode = _ABL.get("ABL_WS_MODE", "temp")  # "temp" (fwd temp) | "act" (A_L)
    # Offload eviction-lag multiplier. Default 0: the lag is DISABLED. Budget sweeps
    # across llama3_1b and qwen3_1_7b (floor .. floor+5 GB) showed a nonzero lag was
    # over-conservative -- it made qwen infeasible at its sac-achievable floor while
    # only ever hurting or not helping llama. With lag=0 both models fit every budget
    # from floor to floor+5 with no real-peak violations. Override via env for ablation.
    _abl_evict_mult = "0"  # offload eviction lag mult
    _abl_off_excl = False  # storage rec OR off, not both

    runtime_per_node = runtime.node_runtimes_ms

    blocks = defaultdict(list)
    nodes = list(gm.graph.nodes)
    for node in nodes:
        b = block_of_node(node)
        if b is not None and b != -1:  # skip non-layer nodes (embeddings/loss)
            blocks[b].append(node)

    (
        _,
        freeable_pi,
        fixed_bytes_max_by_layer,
        accumulative_act_bytes_per_layer,
        act_bytes_per_layer,
        storages_by_layer,
        _,
        _,
        _,
    ) = get_fixed_bytes_tuple  # get_fixed_bytes(gm, trace.num_static_inputs)

    # find last layer
    all_layer_ids = list(blocks.keys())

    last_layer_id = max(all_layer_ids) if len(all_layer_ids) > 1 else -1
    if last_layer_id == -1:
        logger.warning(
            "outer: no layers found -- no activation bytes to free, nothing to do"
        )
        return None, None, None

    # the outer ILP should make decision per module for shared resources:
    # such as the GPU peak memory, CPU memory, PCIe bw
    # simply, how much each layer should keep

    block_names = list(blocks)  # <- index i  <->  block_names[i]
    num_blocks = len(block_names)

    _block_act = block_activation_bytes(mem_est)  # freeable activation bytes per block
    block_act = {
        b: _block_act.get(b, 0) for b in block_names
    }  # ensure every block present
    bw = get_transfer_bw()
    bw_d2h, bw_h2d = bw["d2h"] * 1e6, bw["h2d"] * 1e6  # GB/s -> bytes/ms

    fwd_rt_by_block = defaultdict(float)
    bwd_rt_by_block = defaultdict(float)
    for block_id, blk_nodes in blocks.items():
        for n in blk_nodes:
            if _is_backward_node(n):
                bwd_rt_by_block[block_id] += runtime_per_node.get(n.name, 0.0)
            else:
                fwd_rt_by_block[block_id] += runtime_per_node.get(n.name, 0.0)

    # Outer LP: per-layer keep/recompute/offload fractions (continuous).
    # Memory terms scaled to GiB (MEM_MULTIPLIER) so solver coefficients stay O(1-30).
    M = MEM_MULTIPLIER
    layer_ids = sorted(blocks)  # int layer ids in forward order: 0,1,...,L-1
    B_g = eff_budget / M
    Ppeak_g = mem_est.peak_bytes / M
    aG = {b: act_bytes_per_layer[b] / M for b in layer_ids}  # owned act, GiB
    bw_d2h_g, bw_h2d_g = bw_d2h / M, bw_h2d / M  # GiB/ms

    live_bytes = mem_est.live_bytes

    # UNCOUNTED MEMORY probe: layer temp (forward temporaries) is freed within
    # forward in the original graph, so live_bytes never sees it at the peak but
    # recompute needs it during backward: a recompute working set the outer
    # does not count (ws models only ~one layer's A_L).
    _layer_temp = defaultdict(float)
    for _prod, _ents in mem_est.all_tensors.items():
        _b = block_of_node(_prod)
        if isinstance(_b, int) and _b >= 0:
            for _e in _ents:
                if _e["category"] == TEMP:
                    _layer_temp[_b] += _e["size"]
    tempG = {
        b: _layer_temp.get(b, 0.0) / M for b in layer_ids
    }  # recompute WS per layer

    _t_construct_ilp_start = time.perf_counter()

    k, r, o = {}, {}, {}
    prob = LpProblem("outer_ilp", LpMinimize)
    for b in layer_ids:
        k[b] = LpVariable(f"k_{b}", lowBound=0, upBound=1)
        r[b] = LpVariable(f"r_{b}", lowBound=0, upBound=1)
        o[b] = LpVariable(f"o_{b}", lowBound=0, upBound=1)
        prob += k[b] + r[b] + o[b] == 1, f"split_{b}"

    # (Constraint for recompute) rolling recompute working set. Recomputing a layer re-runs its forward
    # in backward, re-creating that layer's forward temps (tempG[b]); memory
    # freed within forward and thus absent from live_bytes, but present during the
    # recompute. Model it as the max single-layer forward temp among
    # recomputed layers (rolling: ~one layer recomputes at a time).
    # (Constraint for recompute) recompute working set = layer forward temporaries re-created in
    # backward (the previously-uncounted memory, ~2.2 GiB/layer for qwen). Near the
    # loss/early-backward peak several layers recompute in a window and their temps
    # coincide, so a single global max (added only at backward positions) is the
    # practical bound; position-specific under-counts the coincidence.
    ws = LpVariable("ws_rec", lowBound=0)
    _ws_src = tempG if _abl_ws_mode == "temp" else aG
    if _abl_ws:
        for b in layer_ids:
            prob += ws >= _ws_src[b] * r[b], f"ws_{b}"
    else:
        prob += ws == 0, "ws_off"  # ablate the recompute working-set term

    # (Constraint for peak) per-position peak from the real live-bytes profile. At each schedule
    # index, its live bytes minus what recompute/offload frees there (only storages
    # in their fwd->bwd gap at that index are freeable), plus the recompute working
    # set, must fit the budget. Using the real profile: (a) captures the true peak
    # location and gradient accumulation, and (b) credits freeing only where the
    # activation is actually live -- the single boundary over-credited freeing 1:1,
    # but at the real (early-backward) peak most freeable activation is already
    # consumed, so freeing it helps far less. Grad-dominated positions (no freeable
    # activation) are covered by the floor check above.
    # eviction lag (DISABLED by default, evict_mult=0): the idea was that an offloaded
    # activation's D2H does not complete instantly, so near the fwd/bwd boundary (the
    # loss peak) the offloaded bytes may still be resident -- so offload freeing was
    # credited only inside a lag-shrunk window while RECOMPUTE freeing spanned the full
    # fwd->bwd gap. Empirically (see the evict_mult sweep) the lag was over-conservative:
    # offload does evict in time in practice, so with evict_lag=0 offload freeing is
    # credited across the whole fwd->bwd gap, same as recompute. The window logic below
    # collapses to [last_fwd_use_index, first_bwd_use_index] when evict_lag == 0.
    nodes_per_layer = max(1, len(nodes) // max(len(layer_ids), 1))
    evict_lag = int(_abl_evict_mult * nodes_per_layer)  # D2H eviction lag (0 = off)
    n_peak_cons = 0
    for t, lb in live_bytes.items():
        if lb / M <= B_g:
            continue  # this position can never bind
        fb_rec = defaultdict(float)  # recompute-freeable at t (erasable storages)
        fb_off = defaultdict(float)  # offload-freeable at t (evicted, not yet reloaded)
        for so in freeable_pi.get(t, ()):
            L = block_of_node(so.producer_node)
            if L not in aG:
                continue
            _off_elig = so.size >= OFFLOAD_MIN_BYTES and (
                so.last_fwd_use_index + evict_lag
                <= t
                <= so.first_bwd_use_index - evict_lag
            )
            if _off_elig and _abl_off_excl:
                fb_off[L] += so.size  # EXCLUSIVE: storage counts for offload only
            else:
                fb_rec[L] += so.size  # recompute frees across the fwd->bwd gap
                if _off_elig:  # non-exclusive: also offload-freeable in its window
                    fb_off[L] += so.size
        if not fb_rec and not fb_off:
            continue  # grad/loss-dominated -> floor handles it
        freed = lpSum(r[L] * (fb_rec[L] / M) for L in fb_rec) + lpSum(
            o[L] * (fb_off[L] / M) for L in fb_off
        )
        # Recompute working set, POSITION-SPECIFIC: at a backward position, only the
        # layer whose backward runs there is being recomputed, so only ITS forward
        # temp is transiently live. Add tempG[L_t]*r[L_t] for that layer (0 at
        # forward positions / non-layer positions). This counts the previously-
        # uncounted recompute memory without the global-max over-constraint.
        ws_here = ws if _is_backward_node(nodes[t]) else 0  # WS only during backward
        prob += lb / M - freed + ws_here <= B_g, f"peak_{t}"
        n_peak_cons += 1
    logger.info("outer: %d per-position peak constraints", n_peak_cons)

    # (C3) offload windows (Option 1: hidden within a next layer's compute window + some offload/reload is hidden
    # behind its own compute).
    for b in layer_ids:
        if b < last_layer_id:
            prob += (
                o[b] * aG[b]
                <= bw_d2h_g * (fwd_rt_by_block[b + 1]),  # + fwd_rt_by_block[b] / 2),
                f"d2h_{b}",
            )
            prob += (
                o[b] * aG[b]
                <= bw_h2d_g * (bwd_rt_by_block[b + 1]),  # + fwd_rt_by_block[b] / 2),
                f"h2d_{b}",
            )
        else:
            prob += o[b] == 0, f"no_off_last_{b}"

    # objective: minimize added runtime = recompute compute (offload is free here),
    # with a tiny tie-break that resolves the degeneracy deterministically: prefer
    # keeping high-index layers (reward grows with layer id), i.e. free LOW layers
    # first. Low layers' backward runs last (all grads resident), so keeping them
    # is the costliest for the peak; freeing them first aligns the (otherwise
    # arbitrary) per-layer choice with the peak, removing the materialized jitter.
    _nl = max(layer_ids) or 1
    _tiebreak = (
        -1e-4 * lpSum((b / _nl) * k[b] for b in layer_ids) if _abl_tiebreak else 0
    )
    prob += (
        lpSum(r[b] * fwd_rt_by_block[b] for b in layer_ids) + _tiebreak
    ), "added_runtime"

    _t_construct_ilp_end = time.perf_counter()
    logger.info(
        "outer_ilp construction: CBC solve took %.3f s",
        _t_construct_ilp_end - _t_construct_ilp_start,
    )
    _t_construct_ilp = _t_construct_ilp_end - _t_construct_ilp_start

    _t_solve_start = time.perf_counter()
    status = prob.solve(PULP_CBC_CMD(msg=0))
    _t_solve_end = time.perf_counter()
    logger.info(
        "outer_ilp: CBC solve took %.3f s (status=%s)",
        _t_solve_end - _t_solve_start,
        LpStatus[status],
    )
    _t_solve = _t_solve_end - _t_solve_start

    if LpStatus[status] != "Optimal":
        logger.warning("outer: LP not optimal (%s)", LpStatus[status])
        return None, None, None

    # --- report keep/recompute/offload ratios ---
    GiB = 1 << 30
    tot_a = sum(act_bytes_per_layer[b] for b in layer_ids)
    kb = rb = ob = 0.0
    alloc = {}
    for b in layer_ids:
        kv = k[b].value() or 0.0
        rv = r[b].value() or 0.0
        ov = o[b].value() or 0.0
        alloc[b] = (kv, rv, ov)
        ab = act_bytes_per_layer.get(b, 0)
        kb += kv * ab
        rb += rv * ab
        ob += ov * ab
    pct = lambda x: 100.0 * x / tot_a if tot_a else 0.0  # noqa: E731
    logger.info(
        "new-autoAC outer LP | budget=%.2f eff=%.2f | act=%.2f GiB -> "
        "keep=%.2f (%.0f%%)  recompute=%.2f (%.0f%%)  offload=%.2f (%.0f%%)",
        memory_budget / GiB,
        eff_budget / GiB,
        tot_a / GiB,
        kb / GiB,
        pct(kb),
        rb / GiB,
        pct(rb),
        ob / GiB,
        pct(ob),
    )
    logger.info(
        "outer ILP decisions for fractions per-layer:\n",
    )
    for b in layer_ids:
        logger.info(
            "outer: layer %d decision: k=%.2f r=%.2f o=%.2f",
            b,
            alloc[b][0],
            alloc[b][1],
            alloc[b][2],
        )

    # Outer LP only: fractions reported above. Inner ILP (node tagging) is separate.
    return alloc, _t_construct_ilp, _t_solve


def plan_and_tag_inner(
    mem_est: MemoryEstimatorResult,
    opt_bytes,
    runtime,
    get_fixed_bytes_tuple,
    trace: TracedResult,
    gm: torch.fx.GraphModule,
    runtime_estimation_mode: str = COST_MODEL,
    cpu_offload_budget_gb: float = 100.0,
    interp_ctx: tuple | None = None,
    keep_fraction: float = 0.05,
    recompute_fraction: float = 0.90,
    offload_fraction: float = 0.05,
    per_layer_fractions: dict | None = None,
    each_layer_separately: bool = True,
) -> torch.fx.GraphModule | None:
    """Per-layer keep/recompute/offload ILP.

    One independent ILP per transformer block. The keep/recompute/offload
    fractions (which MUST sum to 1) are hard constraints on each layer:

      - ``keep_fraction`` bounds the layer's RESIDENT activation, i.e. the
        layer's activation peak becomes at most ``keep_fraction`` of its own
        saved-for-backward activation. keep=0.05 => the layer keeps <= 5% of
        its activation on the GPU; the other 95% is discarded via recompute
        and/or offload. Summed over all layers, the real forward-boundary peak
        drops from ``fixed + sum(act)`` to ``fixed + keep_fraction*sum(act)``.
      - ``offload_fraction`` bounds bytes streamed to CPU (also freed from the
        GPU forward boundary), shared against a global host budget across
        layers.
      - ``recompute_fraction`` is the remainder: since each tensor is exactly
        one of keep/recompute/offload and keep/offload are upper-bounded,
        recompute is forced to at least ``recompute_fraction`` of the layer.

    Objective: minimize recompute runtime (offload is treated as free inside a
    layer). The tiny keep reward fills the keep budget with the
    most-expensive-to-recompute tensors on ties.
    """
    logger.info("naive inner autoAC: starting...")
    _t_start = time.perf_counter()  # whole plan_and_tag_inner wall time
    keep_fraction, recompute_fraction, offload_fraction = _resolve_fractions(
        keep_fraction, recompute_fraction, offload_fraction
    )
    # gm = trace.gm
    # mem_est = estimate_peak_memory(gm, num_state_inputs=trace.num_static_inputs)
    # runtime_per_node = _estimate_runtime(trace, runtime_estimation_mode, interp_ctx)
    runtime_per_node = runtime.node_runtimes_ms

    nodes = list(gm.graph.nodes)
    node_index = {n: i for i, n in enumerate(nodes)}
    # No-recompute set: RNG (correctness) + attention/HOP/comm/topk save_ops (estimator prices them
    # unreliably / unsafe to recompute). The matmul family stays recomputable:
    # with the compile-scratch measurement fixed (gc after the passes), letting
    # matmul recompute makes the requested keep/recompute/offload fractions
    # actually achievable AND drops the real peak to the floor -- the earlier
    # "recomputing save_ops raises the real peak" was the contaminated
    # measurement, not real. Override via AUTOAC_SAVE_OPS_POLICY (all|none).
    import os as _os_sop

    _save_ops_policy = _os_sop.environ.get(
        "AUTOAC_SAVE_OPS_POLICY", SAVE_OPS_MATMUL_RECOMPUTABLE
    )

    must_keep = get_must_keep_list(gm, save_ops_policy=_save_ops_policy)
    # candidates_by_layer, layer_act, fixed_bytes = _collect_layer_candidates(
    #     gm, mem_est, nodes, node_index
    # )
    (
        _,
        _,
        _,
        _,
        _,
        _,
        candidates_by_layer,
        layer_act,
        total_freeable,
    ) = get_fixed_bytes_tuple
    fixed_bytes = mem_est.peak_bytes - total_freeable

    # The remat pass is region based - it tries to find a region of recomputable
    # nodes. We can only decide on actual storages and their node's tags. If a node
    # does not generate a new storage, in other words, it is a view or alias, we need
    # to tag it according to its storage's (or parent node's) tag.
    # Tagging only the saved-activation producers (the candidates) and leaving
    # their intermediate forward inputs untagged does NOT close regions: remat
    # then keeps those untagged inputs alive from forward into backward (to feed
    # the dup) instead of freeing them, so recompute fails to lower the peak.
    # The ILP below overrides only the candidate producers' keep/recompute/
    # offload decision on top of this base.

    def _decision_space(node):
        if _is_rng_op(node) or _is_collective_or_wait(
            node
        ):  # or other nondeterministic
            return "no_recompute"  # k,o allowed; r forbidden
        return "decidable"  # k,r,o all allowed

    # the ILP cannot tag the must save nodes, such as the RNG states and
    # layer boundaries, so we tag them here.
    for node in gm.graph.nodes:
        if node.op != "call_function" or _is_backward_node(node):
            continue
        fqn = node.meta.get("custom", {}).get(_MODULE_FQN, "")
        if fqn.startswith(("lm_head", "loss")):
            continue
        if node.target in (
            operator.getitem,
            torch.ops._c10d_functional.wait_tensor.default,
        ):
            # Propagate the parent's base tag inline (mirrors tag_sac_policy):
            # getitem/wait share the parent's storage, so leaving them untagged
            # would make the remat pass treat them as saved anchors and fail to
            # close the parent's recompute region.
            parent = node.args[0]
            if isinstance(parent, torch.fx.Node) and "recompute" in parent.meta:
                node.meta["recompute"] = parent.meta["recompute"]
            continue
        if isinstance(node.meta.get("val"), torch.SymInt):
            node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
            continue
        cls = _decision_space(node)
        # if cls == "force_keep":
        #     node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
        if cls == "no_recompute":
            node.meta[
                "recompute"
            ] = CheckpointPolicy.MUST_SAVE  # default; ILP may flip to OFFLOAD
        else:
            node.meta[
                "recompute"
            ] = CheckpointPolicy.MUST_RECOMPUTE  # default; ILP may flip to KEEP/OFFLOAD

    M = MEM_MULTIPLIER
    fixed_g = fixed_bytes / M

    # Global host (pinned CPU) budget for offload, tracked across the per-layer
    # loop so independent layer solves don't collectively blow the host budget.

    cpu_cap_gb = float(cpu_offload_budget_gb)
    remaining_off_g = (cpu_cap_gb * 1e9) / M

    logger.info(
        "inner ILP: peak_bytes=%.2f total_freeable=%.2f fixed=%.2f GiB | "
        "fractions keep=%.2f recompute=%.2f offload=%.2f | host_cap=%.2f GiB",
        mem_est.peak_bytes / M,
        sum(layer_act.values()) / M,
        fixed_g,
        keep_fraction,
        recompute_fraction,
        offload_fraction,
        remaining_off_g,
    )

    def _resolve_cand(x, cand_set):
        # Walk up view/alias chains to the underlying offloadable candidate a
        # recomputed dup would actually read (x may be a view of a candidate).
        seen = set()
        while x is not None and x not in seen:
            seen.add(x)
            if x in cand_set:
                return x
            if _is_view(x) and x.all_input_nodes:
                x = x.all_input_nodes[0]  # my assumption: only one input
            else:
                return None
        return None

    total_ach = {"keep": 0, "recompute": 0, "offload": 0}

    if not each_layer_separately:
        time_for_each_budget = {}

        def _tag_of(k, r, o):
            tag = (
                "keep"
                if k.value() > 0.5
                else "offload"
                if o.value() > 0.5
                else "recompute"
            )
            return tag

        def _propagate_to_getitem_view(node, tag):
            for user in node.users:
                is_getitem_or_wait = user.target in (
                    operator.getitem,
                    torch.ops._c10d_functional.wait_tensor.default,
                )
                if is_getitem_or_wait or (_is_view(user) and tag != "offload"):
                    user.meta["recompute"] = _POLICY_TAG[tag]

        def build_inner_ilp(candidates, kf, of, remaining_off_g, act):
            act_g = act / M
            prob = LpProblem(f"inner_layer_{kf}_{of}", LpMinimize)
            k, r, o = {}, {}, {}
            total_keep = total_off = total_rec = 0
            objective = 0
            for node, size in candidates:
                k[node] = LpVariable(f"k_{node.name}", cat=LpBinary)
                r[node] = LpVariable(f"r_{node.name}", cat=LpBinary)
                o[node] = LpVariable(f"o_{node.name}", cat=LpBinary)
                prob += k[node] + r[node] + o[node] == 1
                if node in must_keep:
                    prob += r[node] == 0  # RNG: never recompute
                    # TODO: layer boundaries are excluded
                if size < OFFLOAD_MIN_BYTES or not _can_offload_node(node):
                    # Match the concrete offload pass: views, collectives/waits,
                    # non-contiguous tensors, and tiny tensors should not be tagged
                    # for CPU offload by the ILP.
                    prob += o[node] == 0
                g = size / M
                total_keep += k[node] * g
                total_off += o[node] * g
                total_rec += r[node] * g
                objective += r[node] * runtime_per_node.get(node.name, 0.0)

            # A recomputed node must not read an offloaded input: its dup runs during
            # backward and would need the input on GPU, but an offloaded input is on
            # CPU (the remat/offload reload-hoisting mishandles this and faults at
            # runtime). Resolve view/alias chains to the underlying candidate.
            cand_set = set(k)
            for v in cand_set:
                for x in v.all_input_nodes:
                    u = _resolve_cand(x, cand_set)
                    if u is not None and u is not v:
                        prob += o[u] + r[v] <= 1

            # Fraction constraints (the peak model). The keep target is a hard peak
            # target when feasible, but forced-save nodes can make a layer
            # impossible at very small keep fractions; keep_over records only that
            # unavoidable excess instead of dropping the layer back to base decision. The
            # offload target is capped by remaining host budget and then filled as
            # much as the discrete tensor sizes permit.
            # kf/of are passed in per signature (already resolved from the outer
            # per-layer fractions in the grouping loop); use them directly.
            target_keep_g = kf * act_g
            target_off_g = min(of * act_g, remaining_off_g)
            keep_over = LpVariable(
                f"keep_over_{kf}_{of}_{remaining_off_g}_{act}", lowBound=0
            )
            off_short = LpVariable(
                f"off_short_{kf}_{of}_{remaining_off_g}_{act}", lowBound=0
            )

            prob += total_keep <= target_keep_g + keep_over
            prob += total_off <= target_off_g
            prob += off_short >= target_off_g - total_off

            # Minimize fraction misses first, then recompute time. Offload is free
            # within the requested offload byte budget, so off_short gets a large
            # penalty to make the solver use that budget before paying recompute cost.
            prob += 1e6 * (keep_over + off_short) + objective - 1e-6 * total_keep

            return prob, k, r, o

        # this will group the layers that have the same budget fractions by the outer
        layer_groups = defaultdict(list)  # sig -> [layer_ids]
        sig_meta = {}  # sig -> (candidates_of_representative, kf, of)

        def _node_key(node, size):
            fqn = node.meta.get("custom", {}).get(_MODULE_FQN, "")
            # Strip the "layers.N." prefix to get a layer-relative key. Some nodes
            # are exactly "layers.N" (the block output/boundary) with no submodule
            # suffix, so guard the split -> "" for those.
            if fqn.startswith("layers."):
                parts = fqn.split(".", 2)
                rel = parts[2] if len(parts) >= 3 else ""  # "" = block output/boundary
            else:
                rel = fqn
            return (rel, str(node.target), size)

        def _layer_keys(candidates):
            # Per-layer uniqiue keys: append an ordinal among same-(rel,target,size)
            # nodes so colliding candidates get distinct keys. Order is consistent
            # across identical layers (candidates_by_layer is built the same way per
            # layer), so the i-th collided node maps to the same key in every layer.
            seen = {}
            out = []
            for n, s in candidates:
                base = _node_key(n, s)
                i = seen.get(base, 0)
                out.append(base + (i,))
                seen[base] = i + 1
            return out

        for b, candidates in candidates_by_layer.items():
            if per_layer_fractions is not None:
                _li = int(b.split(".")[1]) if isinstance(b, str) else b
                kf, _rf, of = per_layer_fractions.get(
                    _li, (keep_fraction, recompute_fraction, offload_fraction)
                )
            else:
                kf, of = keep_fraction, offload_fraction

            _keys = _layer_keys(candidates)  # per-layer unique (ordinal-disambiguated)
            sig = (tuple(sorted(_keys)), kf, of)
            layer_groups[sig].append(b)
            sig_meta.setdefault(sig, (candidates, kf, of, layer_act[b]))

        pattern_by_sig = {}
        skipped = []  # grouped solve does not drop layers; kept for the DONE log
        block_build_s = block_solve_s = 0.0
        for sig, (candidates, kf, of, sig_act) in sig_meta.items():
            _t_b = time.perf_counter()
            prob, k, r, o = build_inner_ilp(
                candidates, kf, of, remaining_off_g, sig_act
            )
            _t_s = time.perf_counter()
            prob.solve(PULP_CBC_CMD(msg=0))
            block_build_s += _t_s - _t_b
            block_solve_s += time.perf_counter() - _t_s
            pattern_by_sig[sig] = {
                key: _tag_of(k[n], r[n], o[n])
                for key, (n, s) in zip(_layer_keys(candidates), candidates)
            }

        # apply the plan to each layer separately
        given_offload_budget = cpu_offload_budget_gb

        for sig, layer_id in layer_groups.items():
            pattern = pattern_by_sig[sig]
            for b in layer_id:
                cands = candidates_by_layer[b]
                for key, (node, size) in zip(_layer_keys(cands), cands):
                    if key in pattern:
                        tag = pattern[key]
                        node.meta["recompute"] = _POLICY_TAG[tag]
                        _propagate_to_getitem_view(node, tag)
                        total_ach[tag] += size
                        # TODO accumulate offload bytes here; fall back to recompute or must save if host budget exceeded
    else:

        skipped = []
        block_build_s = 0.0  # summed per-block ILP build time
        block_solve_s = 0.0  # summed per-block CBC solve time
        for b, candidates in candidates_by_layer.items():
            act = layer_act[b]
            act_g = act / M

            _t_blk = time.perf_counter()  # start of this block's ILP build
            prob = LpProblem(f"inner_layer_{b}", LpMinimize)
            k, r, o = {}, {}, {}
            total_keep = total_off = total_rec = 0
            objective = 0
            for node, size in candidates:
                k[node] = LpVariable(f"k_{node.name}", cat=LpBinary)
                r[node] = LpVariable(f"r_{node.name}", cat=LpBinary)
                o[node] = LpVariable(f"o_{node.name}", cat=LpBinary)
                prob += k[node] + r[node] + o[node] == 1
                if node in must_keep:
                    prob += r[node] == 0  # RNG: never recompute
                    # TODO: layer boundaries are excluded
                if size < OFFLOAD_MIN_BYTES or not _can_offload_node(node):
                    # Match the concrete offload pass: views, collectives/waits,
                    # non-contiguous tensors, and tiny tensors should not be tagged
                    # for CPU offload by the ILP.
                    prob += o[node] == 0
                g = size / M
                total_keep += k[node] * g
                total_off += o[node] * g
                total_rec += r[node] * g
                objective += r[node] * runtime_per_node.get(node.name, 0.0)

            # A recomputed node must not read an offloaded input: its dup runs during
            # backward and would need the input on GPU, but an offloaded input is on
            # CPU (the remat/offload reload-hoisting mishandles this and faults at
            # runtime). Resolve view/alias chains to the underlying candidate.
            cand_set = set(k)
            for v in cand_set:
                for x in v.all_input_nodes:
                    u = _resolve_cand(x, cand_set)
                    if u is not None and u is not v:
                        prob += o[u] + r[v] <= 1

            # Fraction constraints (the peak model). The keep target is a hard peak
            # target when feasible, but forced-save nodes can make a layer
            # impossible at very small keep fractions; keep_over records only that
            # unavoidable excess instead of dropping the layer back to base decision. The
            # offload target is capped by remaining host budget and then filled as
            # much as the discrete tensor sizes permit.
            # Per-layer fractions from the outer ILP (keyed by int layer id) take
            # precedence over the global fractions; this makes the inner apply the
            # outer's actual per-layer plan instead of one uniform split.
            if per_layer_fractions is not None:
                _li = int(b.split(".")[1]) if isinstance(b, str) else b
                kf, _rf, of = per_layer_fractions.get(
                    _li, (keep_fraction, recompute_fraction, offload_fraction)
                )
            else:
                kf, of = keep_fraction, offload_fraction
            target_keep_g = kf * act_g
            target_off_g = min(of * act_g, remaining_off_g)
            keep_over = LpVariable(f"keep_over_{b}", lowBound=0)
            off_short = LpVariable(f"off_short_{b}", lowBound=0)

            prob += total_keep <= target_keep_g + keep_over
            prob += total_off <= target_off_g
            prob += off_short >= target_off_g - total_off

            # Minimize fraction misses first, then recompute time. Offload is free
            # within the requested offload byte budget, so off_short gets a large
            # penalty to make the solver use that budget before paying recompute cost.
            prob += 1e6 * (keep_over + off_short) + objective - 1e-6 * total_keep

            _t_solve = time.perf_counter()  # ILP build done; time the solve
            prob.solve(PULP_CBC_CMD(msg=0))
            _blk_build_s = _t_solve - _t_blk
            _blk_solve_s = time.perf_counter() - _t_solve
            block_build_s += _blk_build_s
            block_solve_s += _blk_solve_s
            logger.info(
                "Inner ILP: layer %s timing: %d vars build=%.3fs solve=%.3fs",
                b,
                len(candidates),
                _blk_build_s,
                _blk_solve_s,
            )
            if LpStatus[prob.status] != "Optimal":
                # Non-fatal fallback: save the layer explicitly. This is honest in
                # the verified audit and avoids leaving base-SAC recompute tags in a
                # layer whose ILP did not produce a valid plan.
                for node, _ in candidates:
                    node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
                logger.warning(
                    "Inner ILP: layer %s infeasible (act=%.2f GiB, "
                    "keep<=%.2f offload<=%.2f GiB) -- forced-keep exceeds the keep+"
                    "offload budget; saving this layer",
                    b,
                    act_g,
                    keep_fraction * act_g,
                    offload_fraction * act_g,
                )
                skipped.append(b)
                continue

            ach = {"keep": 0, "recompute": 0, "offload": 0}
            for node, size in candidates:
                tag = (
                    "keep"
                    if k[node].value() > 0.5
                    else "offload"
                    if o[node].value() > 0.5
                    else "recompute"
                )
                node.meta["recompute"] = _POLICY_TAG[tag]
                # Propagate the decision onto the producer's getitem / wait_tensor /
                # view children so the whole storage chain is tagged consistently
                # (mirrors tag_sac_policy). Without this, a producer we tag RECOMPUTE
                # or offload whose getitem/view child kept a stale MUST_SAVE base tag
                # cannot be freed at runtime -- the child pins the producer's storage,
                # which is why the tagged fractions did not lower the real peak.
                # Offloading a view faults (apply_cpu_offload_pass asserts non-view),
                # so for OFFLOAD only the getitem children (real tensors) inherit;
                # keep/recompute propagate to views too.
                for user in node.users:
                    is_getitem_or_wait = user.target in (
                        operator.getitem,
                        torch.ops._c10d_functional.wait_tensor.default,
                    )
                    if is_getitem_or_wait or (_is_view(user) and tag != "offload"):
                        user.meta["recompute"] = _POLICY_TAG[tag]
                ach[tag] += size
            remaining_off_g -= ach["offload"] / M
            for kind in total_ach:
                total_ach[kind] += ach[kind]
            logger.info(
                "Inner ILP: layer %s keep=%.1f%% recompute=%.1f%% offload=%.1f%% "
                "| layer_peak=%.3f GiB (was %.3f) | keep_over=%.3f GiB "
                "off_short=%.3f GiB | host_remaining=%.2f GiB",
                b,
                100 * ach["keep"] / (act or 1),
                100 * ach["recompute"] / (act or 1),
                100 * ach["offload"] / (act or 1),
                ach["keep"] / M,
                act_g,
                (keep_over.value() or 0.0),
                (off_short.value() or 0.0),
                remaining_off_g,
            )

    # Computed for both branches (grouped and per-layer) before the audit.
    total_act = sum(layer_act.values()) or 1

    # Verified from the graph tags (option a): read back what was actually
    # written to node.meta["recompute"], not the solver's variable values.
    verified_by_layer, verified = _audit_tagged_fractions(candidates_by_layer)
    for b in candidates_by_layer:
        vb = verified_by_layer[b]
        b_act = layer_act[b] or 1
        logger.debug(
            "inner ILP: layer %s VERIFIED keep=%.1f%% recompute=%.1f%% "
            "offload=%.1f%%",
            b,
            100 * vb["keep"] / b_act,
            100 * vb["recompute"] / b_act,
            100 * vb["offload"] / b_act,
        )

    # Solver's totals (candidate variables only; excludes skipped layers'
    # base-SAC fallback) -- kept for comparison against the verified numbers.
    logger.info(
        "Inner ILP: solver results keep=%.1f%% recompute=%.1f%% offload=%.1f%%",
        100 * total_ach["keep"] / total_act,
        100 * total_ach["recompute"] / total_act,
        100 * total_ach["offload"] / total_act,
    )

    # Node-count audit: how many candidate producer nodes landed in each bucket
    # (one decision per producer). Complements the byte fractions above.
    node_counts = {"keep": 0, "recompute": 0, "offload": 0}
    for _b, _cands in candidates_by_layer.items():
        for _node, _sz in _cands:
            node_counts[_classify_tag(_node)] += 1
    logger.info(
        "Inner ILP: node counts (tagged candidate producers) "
        "keep=%d recompute=%d offload=%d total=%d",
        node_counts["keep"],
        node_counts["recompute"],
        node_counts["offload"],
        sum(node_counts.values()),
    )

    # Authoritative: fractions read from the graph tags the passes will act on.
    est_fwd_peak_g = (fixed_bytes + verified["keep"]) / M
    logger.info(
        "Inner ILP: done (verified from graph tags) keep=%.1f%% "
        "recompute=%.1f%% offload=%.1f%% | est fwd-boundary peak=%.3f GiB "
        "(all-keep was %.3f) | %d layers skipped",
        100 * verified["keep"] / total_act,
        100 * verified["recompute"] / total_act,
        100 * verified["offload"] / total_act,
        est_fwd_peak_g,
        (fixed_bytes + total_act) / M,
        len(skipped),
    )
    logger.info(
        "Inner ILP total timing: plan_and_tag_inner=%.3fs "
        "(%d blocks: build=%.3fs solve=%.3fs, rest=%.3fs)",
        time.perf_counter() - _t_start,
        len(candidates_by_layer),
        block_build_s,
        block_solve_s,
        (time.perf_counter() - _t_start) - block_build_s - block_solve_s,
    )
    # inner-achieved fractions (byte-weighted %, read back from the graph tags)
    inner_fractions = {
        "keep": 100 * verified["keep"] / total_act,
        "recompute": 100 * verified["recompute"] / total_act,
        "offload": 100 * verified["offload"] / total_act,
    }
    return gm, block_build_s, block_solve_s, inner_fractions


def plan_and_tag_outer(
    trace: TracedResult,
    memory_budget: int,
    optimizer,
    model_parts: list[torch.nn.Module],
    runtime_estimation_mode: str = COST_MODEL,
    cpu_offload_budget_gb: float = 100.0,
    interp_ctx: tuple | None = None,  # (model, *run_args) for INTERPRETER mode
) -> torch.fx.GraphModule | None:
    """Two-level keep/recompute/offload solver. Tags ``gm`` in place and returns
    it (the caller runs apply_cpu_offload_pass + selective_activation_remat_pass).

    Steps: build per-block tradeoff curves -> outer budget allocation -> tag ->
    measure the REAL post-pass peak on a clone. Because the modeled peak is only
    a proxy (recompute working set omitted; peak-node approximation), we sweep the
    outer cap over ``cap_grid`` multipliers of eff_budget, measure each plan's
    real peak, and pick the one whose real peak is CLOSEST TO the budget from
    below -- i.e. keep the most / free the least -> least added runtime. Returns
    the untouched ``gm`` when the budget already fits, or None when no block
    activations exist / no feasible plan is found.
    """
    gm = trace.gm
    mem_est = estimate_peak_memory(gm, num_state_inputs=trace.num_static_inputs)
    opt_bytes = optimizer_state_bytes(optimizer, model_parts[0])
    estimated = mem_est.peak_bytes + opt_bytes
    if memory_budget > estimated:
        logger.info(
            "new-autoAC: budget %.2f GB >= estimated peak %.2f GB; nothing to do",
            memory_budget / 1e9,
            estimated / 1e9,
        )
        return gm

    import os as _os

    runtime_estimation_mode = _os.environ.get(
        "AUTOAC_RT_MODE", runtime_estimation_mode
    )  # diagnostic override: cost-model | benchmark | interpreter
    logger.info("new-autoAC: runtime estimation mode = %s", runtime_estimation_mode)
    if runtime_estimation_mode == INTERPRETER:
        if interp_ctx is None:
            raise ValueError(
                "INTERPRETER runtime mode needs interp_ctx=(model, *run_args); "
                "pass it from the trainer (see AUTOAC_MODE=scratch)."
            )
        runtime = RuntimeEstimator()(INTERPRETER).estimate(trace, *interp_ctx)
    else:
        runtime = RuntimeEstimator()(runtime_estimation_mode).estimate(trace)
    runtime_per_node = runtime.node_runtimes_ms

    blocks = defaultdict(list)
    nodes = list(gm.graph.nodes)
    for node in nodes:
        b = block_of_node(node)
        if b is not None:
            blocks[b].append(node)

    (
        _,
        _,
        fixed_bytes_max_by_layer,
        act_bytes_per_layer,
        storages_by_layer,
    ) = get_fixed_bytes(gm, trace.num_state_inputs)

    # find last layer
    all_layer_ids: set[int] = set()
    for layer_name in blocks.keys():
        parts = layer_name.split(".")
        if parts[0] == "layers" and len(parts) >= 2:
            all_layer_ids.add(int(parts[1]))

    last_layer_id = max(all_layer_ids) if len(all_layer_ids) > 1 else -1
    if last_layer_id == -1:
        logger.warning(
            "new-autoAC: no layers found -- no activation bytes to free, nothing to do"
        )
        return gm

    # the outer ILP should make decision per module for shared resources:
    # such as the GPU peak memory, CPU memory, PCIe bw
    # simply, how much each layer should keep

    block_names = list(blocks)  # <- index i  <->  block_names[i]
    num_blocks = len(block_names)

    _block_act = block_activation_bytes(mem_est)  # freeable activation bytes per block
    block_act = {
        b: _block_act.get(b, 0) for b in block_names
    }  # ensure every block present
    bw = get_transfer_bw()
    bw_d2h, bw_h2d = bw["d2h"] * 1e6, bw["h2d"] * 1e6  # GB/s -> bytes/ms

    # per-block forward (recompute cost) and backward (window) runtimes
    fwd_rt, bwd_rt = defaultdict(float), defaultdict(float)
    for block, blk_nodes in blocks.items():
        for n in blk_nodes:
            if _is_backward_node(n):
                bwd_rt[block] += runtime_per_node.get(n.name, 0.0)
            else:
                fwd_rt[block] += runtime_per_node.get(n.name, 0.0)

    order = sorted(
        block_names, key=lambda b: int(b.split(".")[1])
    )  # layers.0,1,...,L-1
    d2h_window, h2d_window = {}, {}

    suf_fb, suf_b = 0.0, 0.0
    for j in reversed(order):  # last layer -> first
        d2h_window[j] = suf_fb  # blocks AFTER j (fwd only) # (fwd+bwd)
        h2d_window[j] = suf_b  # blocks AFTER j (bwd only)
        suf_fb += fwd_rt[j]  # + bwd_rt[j]
        suf_b += bwd_rt[j]

    blocks_fwd_end_bwd_start = {j: [0, 0] for j in order}
    time_f = 0.0
    for j in order:  # layers.0, 1, ..., L-1
        time_f += fwd_rt[j]
        blocks_fwd_end_bwd_start[j][0] = suf_fb

    time_b = time_f
    for j in reversed(order):  # layers.L-1, ..., 1, 0
        blocks_fwd_end_bwd_start[j][1] = suf_b
        time_b += bwd_rt[j]

    # --- outer three-way ILP: per-block keep/recompute/offload fractions ---
    # Only ACTIVATION bytes are freeable; params/grads/temp/non-block activations
    # form the fixed resident baseline at the peak.
    fixed = mem_est.peak_bytes - sum(block_act.values())
    eff_budget = memory_budget - opt_bytes
    t_fwd_total = sum(fwd_rt[b] for b in block_names)  # D2H overlap capacity (ms)
    t_bwd_total = sum(bwd_rt[b] for b in block_names)  # H2D overlap capacity (ms)

    alloc = solve_outer_three_way(
        mem_est,
        block_act,
        {b: fwd_rt[b] for b in block_names},
        d2h_window,
        h2d_window,
        blocks_fwd_end_bwd_start,
        order,
        time_f,
        time_b,
        bw_d2h,
        bw_h2d,
        t_fwd_total,
        t_bwd_total,
        fixed,
        eff_budget,
        cpu_budget_bytes=100,
    )
    if alloc is None:
        logger.warning(
            "new-autoAC: outer ILP infeasible -- fixed baseline %.2f GiB exceeds "
            "eff_budget %.2f GiB (freeing all activation still cannot fit)",
            fixed / (1 << 30),
            eff_budget / (1 << 30),
        )
        return None

    # --- report the per-block allocation ---
    GiB = 1 << 30
    total_act = sum(block_act.values())
    keep_b = sum(k * block_act[b] for b, (k, _r, _o) in alloc.items())
    rec_b = sum(r * block_act[b] for b, (_k, r, _o) in alloc.items())
    off_b = sum(o * block_act[b] for b, (_k, _r, o) in alloc.items())
    logger.info(
        "new-autoAC outer (3-way): %d blocks | budget=%.2f eff=%.2f fixed=%.2f "
        "opt=%.2f GiB | act=%.2f GiB -> keep=%.2f recompute=%.2f offload=%.2f GiB | "
        "modeled peak=%.2f GiB",
        num_blocks,
        memory_budget / GiB,
        eff_budget / GiB,
        fixed / GiB,
        opt_bytes / GiB,
        total_act / GiB,
        keep_b / GiB,
        rec_b / GiB,
        off_b / GiB,
        (fixed + keep_b) / GiB,
    )
    logger.info(
        "new-autoAC per-block (keep/recompute/offload frac):\n%s",
        "\n".join(
            f"  {b}: k={alloc[b][0]:.2f} r={alloc[b][1]:.2f} o={alloc[b][2]:.2f}  "
            f"(act={block_act[b] / GiB:.3f} GiB, d2h_win={d2h_window[b]:.1f}ms "
            f"h2d_win={h2d_window[b]:.1f}ms)"
            for b in order
        ),
    )

    # Outer ILP only: this returns the per-block fractions via the log above.
    # Materialization (inner ILP -> node.meta["recompute"] tags) is a separate step.
    # Inner ILP is to be implemented later
    return None


# ---------------------------------------------------------------------------
# Outer three-way ILP: per-block keep/recompute/offload FRACTIONS
# ---------------------------------------------------------------------------
def solve_outer_three_way(
    mem_est,
    act_bytes: dict[str, int],  # block -> freeable ACTIVATION bytes (activation only)
    fwd_rt: dict[str, float],  # block -> forward runtime ms (recompute cost)
    d2h_window: dict[str, float],  # block -> D2H overlap window ms (evict gap)
    h2d_window: dict[str, float],  # block -> H2D overlap window ms (reload gap)
    blocks_fwd_end_bwd_start: dict[str, float],  # block -> fwd end -> bwd start ms
    ordered_blocks: list[str],  # order of blocks:  layers.0,1,...,L-1
    time_fwd,  # total forward runtime ms
    time_bwd,  # total backward runtime ms
    bw_d2h: float,  # D2H bandwidth, bytes/ms
    bw_h2d: float,  # H2D bandwidth, bytes/ms
    t_fwd_total: float,  # total forward compute ms (D2H overlap capacity)
    t_bwd_total: float,  # total backward compute ms (H2D overlap capacity)
    fixed_bytes: float,  # non-freeable resident bytes at the peak
    eff_budget_bytes: float,  # memory_budget - opt_bytes
    cpu_budget_bytes: float = 100,
    *,
    keep_eps: float = 1e-3,  # ms/GiB tie-break: prefer keep > offload > recompute
    time_limit: int = 120,
) -> dict[str, tuple[float, float, float]] | None:
    """Allocate a per-block keep/recompute/offload split (fractions summing to 1)
    minimizing added runtime under the global peak budget.

    Decision per block b: r_b (recompute frac), o_b (offload frac); keep is the
    residual k_b = 1 - r_b - o_b. Both recompute and offload evict the activation
    from GPU during its gap, so only KEPT activation is resident at the peak. All
    memory terms are in GiB, all time in ms.

      (C1) peak:   fixed + sum_b k_b*act_b <= eff_budget         (only kept resident)
      (C2) window: all act accumulated <= d2h window             (evict fits its gap)
                   same for h2d window                           (reload fits its gap)
                   -> some penalty for unhidden transfer with stall:
      (C3) shared: d2h_stall >= (sum_b o_b*act_b)/bw_d2h - t_fwd (contention: all
                   h2d_stall >= (sum_b o_b*act_b)/bw_h2d - t_bwd  offloads share one
                      D2H and one H2D engine; bytes past the compute window stall).
      objective:   min sum_b r_b*fwd_rt[b] + d2h_stall + h2d_stall
                       + keep_eps*sum_b (r_b+o_b)*act_b

    Offload that fits both its window (C2) and the shared budget (C3) is modeled as
    fully hidden (zero added time), so the solver prefers keep (if the budget fits),
    then offload, then recompute. The keep_eps term breaks the free-offload tie
    toward keep so a block that already fits the budget is not needlessly freed.

    Returns {block: (keep, recompute, offload)} fractions, or None if infeasible
    (the fixed baseline alone exceeds the budget -- freeing all activation can't fit).
    """
    M = MEM_MULTIPLIER
    blocks = list(act_bytes)
    prob = LpProblem("outer_three_way", LpMinimize)

    def san(b: str) -> str:
        return b.replace(".", "_")

    r = {b: LpVariable(f"r_{san(b)}", 0.0, 1.0) for b in blocks}
    o = {b: LpVariable(f"o_{san(b)}", 0.0, 1.0) for b in blocks}
    for b in blocks:
        prob += r[b] + o[b] <= 1.0, f"split_{san(b)}"  # keep = 1 - r - o >= 0

    act_gib = {b: act_bytes[b] / M for b in blocks}
    bw_d2h_gib = bw_d2h / M  # GiB/ms
    bw_h2d_gib = bw_h2d / M

    # (C1') per-position peak: evaluate memory at each block b's backward step.
    # Block b's activation is ALWAYS present there (kept / recomputed-now / reloaded-
    # now); earlier blocks k<b contribute only their kept part; later blocks are freed.
    # fixed_bytes is the global non-freeable baseline (params+grads+temp+non-block acts,
    # = peak - sum(act)); at full keep m[last] = fixed + total_act = peak, matching (C1).
    fixed_bytes_per_block = block_fixed_bytes(mem_est)
    for idx, b in enumerate(ordered_blocks):
        m_b = (
            fixed_bytes / M
            + act_gib[b]  # block b: ALWAYS present at its own bwd
            + lpSum(
                (1 - o[k] - r[k]) * act_gib[k] for k in ordered_blocks[:idx]
            )  # earlier blocks: only kept part
        )
        prob += m_b <= eff_budget_bytes / M, f"peak_{san(b)}"

    # # (C2) per-block overlap windows (evict + reload must fit this block's own gap).
    # interval = [0, time_fwd + time_bwd]  # start, end
    # old with bugs:
    # for j in ordered_blocks:
    #     start: int | Unknown = max(interval[0], blocks_fwd_end_bwd_start[j][0])
    #     end = min(interval[1], blocks_fwd_end_bwd_start[j][1])
    #     # o[j] * act_gib[j]: how much to offload and reload
    #     prob += o[j] * act_gib[j] <= (time_fwd - start) * bw_d2h_gib, f"d2hwin_{san(j)}"
    #     prob += o[j] * act_gib[j] <= (end - time_fwd) * bw_h2d_gib, f"h2dwin_{san(j)}"
    #     interval[0] = start + o[j] * act_gib[j] / bw_d2h_gib
    #     interval[1] = end - o[j] * act_gib[j] / bw_h2d_gib

    # D2H / H2D stall vars, defined here so (C2') below can reference h2d_stall
    # (H2D is a soft slack in (C2'); the (C3) block adds their lower bounds).
    d2h_stall = LpVariable("d2h_stall", 0.0)
    h2d_stall = LpVariable("h2d_stall", 0.0)

    # (C2) shared-engine timeline: for every block j, the offloaded bytes of
    # blocks j..L-1 must fit the engine time available to them.
    #   D2H: those bytes exist only after block j's fwd -> window = fwd time after j.
    #   H2D: they must arrive before block j's bwd    -> window = bwd time after j.
    # One constraint family, shared o-variables = engine contention, linearly.
    # same but written differently and correctly
    for idx, j in enumerate(ordered_blocks):
        suffix_off = lpSum(o[i] * act_gib[i] for i in ordered_blocks[idx:])
        prob += (
            # here we allow offloading go beyond the fwd-bwd boundary with stall but the memory peak can also increase?
            # suffix_off <= bw_d2h_gib * (d2h_window[j] + d2h_stall),
            suffix_off
            <= bw_d2h_gib
            * (
                d2h_window[j]
            ),  # here we don't allow offloading go beyond the fwd-bwd boundary
            f"d2h_suffix_{san(j)}",
        )
        prob += (
            suffix_off <= bw_h2d_gib * h2d_window[j] + bw_h2d_gib * h2d_stall,
            f"h2d_suffix_{san(j)}",
        )

    prob += (
        lpSum(r[b] * fwd_rt[b] for b in blocks)
        + d2h_stall
        + h2d_stall
        + keep_eps * lpSum((r[b] + o[b]) * act_gib[b] for b in blocks)
    ), "added_runtime"

    prob += (lpSum(o[b] * act_gib[b] for b in blocks) <= cpu_budget_bytes), "cpu_budget"

    status = prob.solve(PULP_CBC_CMD(gapRel=0.02, timeLimit=time_limit, msg=0))
    if LpStatus[status] != "Optimal":
        return None

    alloc: dict[str, tuple[float, float, float]] = {}
    for b in blocks:
        rv = r[b].value() or 0.0
        ov = o[b].value() or 0.0
        alloc[b] = (max(0.0, 1.0 - rv - ov), rv, ov)  # (keep, recompute, offload)
    return alloc


def block_activation_bytes(mem_est):

    totals, seen = defaultdict(int), set()

    for producer, entries in mem_est.all_tensors.items():
        b = block_of_node(producer)
        if b is None or b == -1:
            continue
        for e in entries:
            if e["category"] != ACT or e["first_bwd_use"] is None:
                continue  # not a saved-for-backward activation
            if e["sid"] in seen:  # dedup: one variable per storage id
                continue
            seen.add(e["sid"])
            totals[b] += e["size"]  # bytes (shape x dtype, computed by the estimator)
    return dict(totals)


def block_fixed_bytes(mem_est) -> dict[str, int]:
    totals, seen = defaultdict(int), set()
    for producer, entries in mem_est.all_tensors.items():
        b = block_of_node(producer)
        if b is None or b == -1:
            continue
        for e in entries:
            if e["category"] == ACT:
                continue  # a saved-for-backward activation
            if e["sid"] in seen:  # dedup: one variable per storage id
                continue
            seen.add(e["sid"])
            totals[b] += e["size"]  # bytes (shape x dtype, computed by the estimator)
    return dict(totals)


def _get_size(t: torch.Tensor) -> int:
    return t.untyped_storage().nbytes()


def get_fixed_bytes(
    gm: torch.fx.GraphModule,
    num_state_inputs: int,
):
    """#TODO: explain this function."""
    nodes = list(gm.graph.nodes)
    indices_per_node = {n: i for i, n in enumerate(nodes)}
    num_indices = len(nodes)
    output_inputs = set()
    for node in nodes:
        if node.op == "output":
            output_inputs.update(node.all_input_nodes)

    placeholders = [n for n in nodes if n.op == "placeholder"]
    persistent_state = set(placeholders[:num_state_inputs])

    end = len(nodes)  # "live to the end" sentinel for resident/returned storages

    total_activation_per_blk = 0
    # we have to find each layer's last fwd and first bwd use for each sid
    # the following loop classifies the storages
    live_sids_per_index = {}
    freeable_sids_per_index = {}
    live_key = {}
    death_of = {}
    storages = []
    storages_by_prod_node = defaultdict(list)
    for node in nodes:
        # if node.op != "call_function" or _is_backward_node(node):
        #     continue
        # if sum(1 for u in node.users if _is_backward_node(u)) == 0:
        #     continue

        index = indices_per_node[node]
        for t in pytree.tree_leaves(node.meta.get("val")):
            if not isinstance(t, torch.Tensor) or t.device.type != "cuda":
                continue
            sid = t.untyped_storage()._cdata
            key = live_key.get(sid, None)
            if key is not None and death_of[key] >= index:
                continue  # this means the same sid is still live at this index

            key = (sid, index)
            live_key[sid] = key

            if node in output_inputs or node in persistent_state:
                death_of[key] = num_indices + 1
            else:
                death_of[key] = find_last_use_index(node, sid, indices_per_node)

            last_fwd_index: int = find_last_fwd_use_index(node, sid, indices_per_node)
            if last_fwd_index < 0:
                last_fwd_index = index
            first_bwd_index = find_first_bwd_use_index(node, sid, indices_per_node)

            prod = node
            category: str
            if prod.op in ("placeholder", "get_attr"):
                if prod in persistent_state or prod.op == "get_attr":
                    category = PARAM
                elif "tangent" in prod.name:
                    category = GRAD  # gradient seed
                else:
                    category = INPUT
            elif _is_backward_node(prod):
                category = GRAD if death_of[key] >= end else TEMP
            else:  # forward-produced compute
                last = nodes[min(death_of[key], end - 1)]
                category = ACT if _is_backward_node(last) else TEMP

            new_object = StorageObject(
                sid=sid,
                size=_get_size(t),
                producer_node=node,  # this could also be an int index
                produced_index=index,
                death_index=death_of[key],
                last_fwd_use_index=last_fwd_index,  # for now
                first_bwd_use_index=first_bwd_index,  # for now
                category=category,
            )

            # layer_id = block_of_node(node)
            # node_index = indices_per_node[node]
            # if layer_id is not None and layer_id != -1:
            #     storages_by_layer[layer_id][node_index] = new_object
            storages_by_prod_node[prod].append(new_object)

            storages.append(new_object)
            total_activation_per_blk += _get_size(t)

    """
    per_layer_node_bytes = defaultdict(lambda: defaultdict(int))
    seen = set()
    total_freeable = 0
    for producer, entries in mem_est.all_tensors.items():
        b = block_of_node(producer)
        if b is None or b == -1:
            continue  # non-block activation -> part of the fixed baseline
        if _is_view(producer):
            continue  # cannot tag a view independently
        for e in entries:
            if e["category"] != ACT or e["first_bwd_use"] is None:
                continue  # not a saved-for-backward activation
            # Dedup on (sid, birth), matching the estimator's own liveness key:
            # a storage id that is freed and later reused starts a fresh
            # interval and is a legitimately distinct activation. Keying on the
            # bare sid would drop that second interval and undercount memory.
            skey: tuple[Unknown, Unknown] = (e["sid"], e["birth"])
            if skey in seen:
                continue
            seen.add(skey)
            per_layer_node_bytes[b][producer] += e["size"]
            total_freeable += e["size"]

    candidates_by_layer = {
        b: list(nb.items()) for b, nb in per_layer_node_bytes.items()
    }
    layer_act = {b: sum(sz for _, sz in c) for b, c in candidates_by_layer.items()}
    fixed_bytes = mem_est.peak_bytes - total_freeable
"""

    per_layer_node_bytes: defaultdict[int, defaultdict[Unknown, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    total_freeable = 0

    # live_sids_per_index[index] = set(seen) # to make a copy
    act_add_at, act_remove_at = defaultdict(list), defaultdict(list)
    act_add_at_for_freeable = defaultdict(list)
    act_remove_at_for_freeable = defaultdict(list)
    list_of_fixed_tensors_add_at = defaultdict(list)
    list_of_fixed_tensors_remove_at = defaultdict(list)
    for object in storages:
        if object.category != ACT:
            list_of_fixed_tensors_add_at[object.produced_index].append(object)
            list_of_fixed_tensors_remove_at[object.death_index].append(object)
            continue
        # if object.produced_index == index:
        act_add_at[object.produced_index].append(object)
        act_remove_at[object.death_index].append(object)
        first_backward = object.first_bwd_use_index
        if first_backward is None or first_backward == INT64_MAX:
            continue
        act_add_at_for_freeable[object.last_fwd_use_index + 1].append(
            object
        )  # freeable AFTER last fwd use
        act_remove_at_for_freeable[object.first_bwd_use_index].append(
            object
        )  # resident again from first bwd use
        if object.category == ACT and object.first_bwd_use_index is not None:
            layer_id: int = block_of_node(object.producer_node)
            if layer_id is not None and layer_id != -1:
                per_layer_node_bytes[layer_id][object.producer_node] += object.size
                total_freeable += object.size

    candidates_by_layer = {
        b: list(nb.items()) for b, nb in per_layer_node_bytes.items()
    }
    layer_act = {b: sum(sz for _, sz in c) for b, c in candidates_by_layer.items()}

    fixed_bytes_max = 0
    fixed_bytes = 0
    fixed_bytes_max_by_layer = defaultdict(int)
    for t in range(num_indices):
        fixed_bytes += sum(obj.size for obj in list_of_fixed_tensors_add_at.get(t, ()))
        fixed_bytes_max = max(fixed_bytes_max, fixed_bytes)
        node = nodes[t]
        layer_id = block_of_node(node)
        fixed_bytes_max_by_layer[layer_id] = max(
            fixed_bytes, fixed_bytes_max_by_layer.get(layer_id, 0)
        )
        fixed_bytes -= sum(
            obj.size for obj in list_of_fixed_tensors_remove_at.get(t, ())
        )

    # per-index ACT liveness + freeable sets for the outer peak constraint.
    live = set()
    live_freeable = set()

    # this is just each layer's activation memory
    act_bytes_per_layer = defaultdict(int)
    for so in storages:
        if so.category == ACT:
            act_bytes_per_layer[block_of_node(so.producer_node)] += so.size

    # accumulative activation memory for each layer
    accumulative_act_bytes_per_layer = defaultdict(int)
    cum_act_mem = 0
    for t in range(num_indices):
        for obj in act_add_at.get(t, ()):
            cum_act_mem += obj.size
            live.add(obj)
        for obj in act_add_at_for_freeable.get(t, ()):
            live_freeable.add(obj)
        live_sids_per_index[t] = set(live)  # storage_objects live at index t
        freeable_sids_per_index[t] = set(live_freeable)  # freeable at index t
        layer_id = block_of_node(nodes[t])
        accumulative_act_bytes_per_layer[layer_id] = max(
            accumulative_act_bytes_per_layer[layer_id], cum_act_mem
        )
        for obj in act_remove_at.get(t, ()):
            cum_act_mem -= obj.size
            live.discard(obj)
        for obj in act_remove_at_for_freeable.get(t, ()):
            live_freeable.discard(obj)

    # fixed_bytes_max_by_layer: global non-act baseline during each layer's region.
    # act_bytes_per_layer: activation OWNED by each layer (what the outer splits).
    return (
        live_sids_per_index,  # this is per node index live sids
        freeable_sids_per_index,  # this is per node index freeable sids
        fixed_bytes_max_by_layer,  # this is per layer fixed byes (that we cannot touch)
        accumulative_act_bytes_per_layer,  # this is per layer activation bytes (that we can touch)
        act_bytes_per_layer,  # this is per layer activation bytes (that we can touch)
        storages_by_prod_node,  # this is per layer storages, the key is the layer of the producer node
        candidates_by_layer,
        layer_act,
        total_freeable,
    )


def find_last_fwd_use_index(node, sid, indices, _memo=None) -> int:
    """This method finds the last forward index where the sid is used as input"""
    if _memo is None:
        _memo = {}
    if node in _memo:
        return _memo[node]

    def _has_sid(leaves):
        return any(
            isinstance(x, torch.Tensor)
            and x.device.type == "cuda"
            and x.untyped_storage()._cdata == sid
            for x in leaves
        )

    last_fwd_index = -1
    for user in node.users:
        user_in = pytree.tree_leaves(
            (map_arg(user.args, val_of), map_arg(user.kwargs, val_of))
        )
        if not _has_sid(user_in) or _is_backward_node(user):
            continue  # not a fwd reader of sid
        last_fwd_index = max(last_fwd_index, indices[user])
        # if this user's OUTPUT also carries sid (view / in-place), the storage
        # lives on -- follow the alias chain to its downstream fwd uses.
        if _has_sid(pytree.tree_leaves(user.meta.get("val"))):
            last_fwd_index = max(
                last_fwd_index, find_last_fwd_use_index(user, sid, indices, _memo)
            )
    _memo[node] = last_fwd_index
    return last_fwd_index


def find_first_bwd_use_index(node, sid, indices, _memo=None) -> int:
    """This method finds the first backward index where the sid is used as input"""
    if _memo is None:
        _memo = {}
    if node in _memo:
        return _memo[node]

    def _has_sid(leaves):
        return any(
            isinstance(x, torch.Tensor)
            and x.device.type == "cuda"
            and x.untyped_storage()._cdata == sid
            for x in leaves
        )

    first_bwd_index = INT64_MAX
    for user in node.users:
        user_in = pytree.tree_leaves(
            (map_arg(user.args, val_of), map_arg(user.kwargs, val_of))
        )  # this gives us the inputs of the user node
        if not _has_sid(user_in):
            continue  # user doesn't read sid at all
        if _is_backward_node(user):
            first_bwd_index = min(first_bwd_index, indices[user])
        # Follow the alias chain (view / in-place) through fwd AND bwd nodes:
        # a forward in-place op carries sid forward to a later backward reader.
        if _has_sid(pytree.tree_leaves(user.meta.get("val"))):  # check outputs
            first_bwd_index = min(
                first_bwd_index,
                find_first_bwd_use_index(user, sid, indices, _memo),
            )
    _memo[node] = first_bwd_index
    return first_bwd_index


def find_last_use_index(node, sid, indices, _memo=None):
    """This method finds the last index where the sid is used in the graph."""

    if _memo is None:
        _memo = {}
    if node in _memo:
        return _memo[node]

    def _has_sid(leaves):
        return any(
            isinstance(x, torch.Tensor)
            and x.device.type == "cuda"
            and x.untyped_storage()._cdata == sid
            for x in leaves
        )

    last = -1
    for user in node.users:
        user_in = pytree.tree_leaves(
            (map_arg(user.args, val_of), map_arg(user.kwargs, val_of))
        )
        if not _has_sid(user_in):  # this user does not have sid as input
            continue
        last = max(last, indices[user])  # count ALL readers
        if _has_sid(pytree.tree_leaves(user.meta.get("val"))):
            last = max(last, find_last_use_index(user, sid, indices, _memo))
    _memo[node] = last
    return last


"""

# we have to find each layer's last fwd and first bwd use for each sid
# the following loop classifies the storages
live_sids_per_index = {}
freeable_sids_per_index = {}
live_key = {}
death_of = {}
storages = []
for node in block_nodes:
    if node.op != "call_function" or  _is_backward_node(node):
        continue
    if sum(1 for u in node.users if _is_backward_node(u)) == 0:
        continue

    index = indices_per_block_node[node]
    for t in pytree.tree_leaves(node.meta.get("val")):
        if not isinstance(t, torch.Tensor) or t.device.type != "cuda":
            continue
        sid = t.untyped_storage()._cdata
        key = live_key.get(sid, None)
        if key is not None and death_of[key] >= index:
            continue # this means the same sid is still live at this index

        key = (sid, index)

        live_key[sid] = key
        death_of[key] = find_last_use_index(node, sid, indices_per_block_node)

        last_fwd_index: int = find_last_fwd_use_index(node, sid, indices_per_block_node)
        first_bwd_index = find_first_bwd_use_index(node, sid, indices_per_block_node)
        new_object = StorageObject(
            sid=sid,
            size=_get_size(t),
            producer_node=node, # this could also be an int index
            produced_index=index,
            death_index=death_of[key],
            last_fwd_use_index=last_fwd_index, # for now
            first_bwd_use_index=first_bwd_index, # for now
            )
        storages.append(new_object)
        total_activation_per_blk += _get_size(t)

# live_sids_per_index[index] = set(seen) # to make a copy
add_at, remove_at = defaultdict(list), defaultdict(list)
add_at_for_freeable = defaultdict(list)
remove_at_for_freeable =  defaultdict(list)
for object in storages:
    # if object.produced_index == index:
    add_at[object.produced_index].append(object)
    remove_at[object.death_index].append(object)

    first_backward = obj.first_bwd_use_index
    if first_backward is None or first_backward == INT64_MAX:
        continue
    add_at_for_freeable[object.last_fwd_use_index + 1].append(object)   # freeable AFTER last fwd use
    remove_at_for_freeable[object.first_bwd_use_index].append(object)   # resident again from first bwd use

live = set()
live_freeable = set()
for t in range(num_indices):
    for obj in add_at.get(t, ()):
        live.add(obj)
    for obj in add_at_for_freeable.get(t, ()):
        live_freeable.add(obj)
    live_sids_per_index[t] = set(live)    # storage_objects live at index t
    freeable_sids_per_index[t] = set(live_freeable)    # storage_objects freeable at index t
    for obj in remove_at.get(t, ()):
        live.discard(obj)
    for obj in remove_at_for_freeable.get(t, ()):
        live_freeable.discard(obj)


"""

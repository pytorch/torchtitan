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

import copy
from collections import defaultdict
from dataclasses import dataclass, field

import torch
from pulp import (
    LpBinary,
    LpMinimize,
    LpProblem,
    LpStatus,
    lpSum,
    LpVariable,
    PULP_CBC_CMD,
)
from torch.utils.checkpoint import CheckpointPolicy
from torchtitan.experiments.graph_trainer.common_utils import (
    _is_backward_node,
    _MODULE_FQN,
)
from torchtitan.experiments.graph_trainer.cpu_offload import apply_cpu_offload_pass
from torchtitan.experiments.graph_trainer.make_fx_tracer import TracedResult
from torchtitan.experiments.graph_trainer.memory_estimator import (
    ACT,
    estimate_peak_memory,
    optimizer_state_bytes,
)
from torchtitan.experiments.graph_trainer.runtime_estimator import (
    COST_MODEL,
    INTERPRETER,
    RuntimeEstimator,
    RuntimeEstimatorResult,
)
from torchtitan.experiments.graph_trainer.selective_activation_remat import (
    selective_activation_remat_pass,
)
from torchtitan.experiments.graph_trainer.transfertime_estimator import (
    _transfer_ms,
    get_transfer_bw,
)
from torchtitan.tools.logging import logger

# meta["recompute"] tag for each per-tensor policy decision.
_POLICY_TAG = {
    "keep": CheckpointPolicy.MUST_SAVE,
    "recompute": CheckpointPolicy.MUST_RECOMPUTE,
    "offload": CheckpointPolicy.MUST_CPU_OFFLOAD,
}

# Scale all memory terms to GiB inside the ILP (sac_milp's MEM_MULTIPLIER). Raw
# byte coefficients (~1e10) mixed with runtimes (~1e2) give CBC a coefficient
# spread that breaks its tolerances and yields constraint-violating "Optimal"
# solutions; GiB units keep every coefficient O(1-30).
MEM_MULTIPLIER = 1 << 30

# Offload only tensors at least this large: fewer, larger transfers hold less
# per-tensor overhead at the backward peak (matches the sac_and_offload default).
OFFLOAD_MIN_BYTES = 1 << 20  # 1 MiB


def _is_rng_op(node: torch.fx.Node) -> bool:
    """RNG ops cannot be replayed by the remat pass, so they must never be
    recomputed (they may still be kept or offloaded)."""
    return torch.Tag.nondeterministic_seeded in getattr(node.target, "tags", set())


# ---------------------------------------------------------------------------
# Step 1: group nodes by transformer block
# ---------------------------------------------------------------------------
def block_of_node(node: torch.fx.Node) -> str | None:
    """Return the transformer-block FQN a node belongs to, e.g. "layers.3",
    or None for nodes outside a block (embeddings, norm, lm_head, loss, ...).

    The block is the "layers.<N>" prefix of the node's module FQN tag. Nodes
    without that prefix are not block-owned and are handled as forced-keep
    (they contribute to the fixed baseline, not to any block's decisions).
    """
    fqn = node.meta.get("custom", {}).get(_MODULE_FQN, "")
    parts = fqn.split(".")
    if len(parts) >= 2 and parts[0] == "layers" and parts[1].isdigit():
        return f"layers.{parts[1]}"
    return None


def group_nodes_by_block(gm: torch.fx.GraphModule) -> dict[str, list[torch.fx.Node]]:
    """Map each transformer-block FQN to its forward call_function nodes."""
    blocks: dict[str, list[torch.fx.Node]] = defaultdict(list)
    for node in gm.graph.nodes:
        if node.op != "call_function" or _is_backward_node(node):
            continue
        block = block_of_node(node)
        if block is not None:
            blocks[block].append(node)
    return dict(blocks)


# ---------------------------------------------------------------------------
# outer ilp over all layers + inner ilp per layer
# ---------------------------------------------------------------------------
def plan_and_tag(
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
        if b is None:
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
        if b is None:
            continue
        for e in entries:
            if e["category"] == ACT:
                continue  # a saved-for-backward activation
            if e["sid"] in seen:  # dedup: one variable per storage id
                continue
            seen.add(e["sid"])
            totals[b] += e["size"]  # bytes (shape x dtype, computed by the estimator)
    return dict(totals)

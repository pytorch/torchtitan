# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Two-level per-tensor keep/recompute/offload ILP (work in progress).

The outer ILP splits the memory budget across transformer blocks, then the
inner ILP decides keep, recompute or offload for each tensor inside a block.
The decomposition follows torch's sac_milp, with offload added.

plan_outer does the first step and plan_and_tag_inner does the second and tags
the graph.
"""
import gc
import hashlib
import math
import operator
import os
import statistics
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.utils._pytree as pytree
from pulp import (
    LpBinary,
    LpMinimize,
    LpProblem,
    LpStatus,
    lpSum,
    LpVariable,
    PULP_CBC_CMD,
    value,
)
from torch.fx.node import map_arg
from torch.utils.checkpoint import CheckpointPolicy
from torchinsights.graph_estimation import (
    estimate_peak_memory,
    UNKNOWN_OPTIMIZER_BYTES,
    MemoryEstimatorResult,
    optimizer_state_bytes,
)
from torchinsights.graph_estimation._fx_utils import (
    ACT,
    feeds_grad_collective,
    get_size,
    GRAD,
    INPUT,
    is_pre_bucket_all_gather,
    is_pre_bucket_reduce_scatter,
    PARAM,
    TEMP,
)
from torchinsights.graph_estimation.runtime_estimator import (
    # from torchtitan.experiments.graph_trainer.runtime_estimator import (
    COST_MODEL,
    INTERPRETER,
    RuntimeEstimator,
)
from torchinsights.graph_estimation.transfertime_estimator import get_transfer_bw
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
from torchtitan.experiments.graph_trainer.fsdp_patterns import is_all_gather_into_tensor
from torchtitan.experiments.graph_trainer.make_fx_tracer import TracedResult

# from torchtitan.experiments.graph_trainer.transfertime_estimator import get_transfer_bw
from torchtitan.tools.logging import logger

# Set from --compile.debug_memory_policy_solver. The solver logs around 80
# lines per run, which is handy when tuning a budget and noise otherwise.
_debug_logging = False


def _dbg(msg, *args):
    """Log a solver diagnostic, but only when debug logging is on."""
    if _debug_logging:
        logger.info(msg, *args)


# meta["recompute"] tag for each per-tensor policy decision.
_POLICY_TAG = {
    "keep": CheckpointPolicy.MUST_SAVE,
    "recompute": CheckpointPolicy.MUST_RECOMPUTE,
    "offload": CheckpointPolicy.MUST_CPU_OFFLOAD,
}


# Which compute-heavy save_ops get_must_keep_list bars from recompute, leaving
# them to keep or offload.
SAVE_OPS_ALL = "all"  # anchor every save_op
SAVE_OPS_MATMUL_RECOMPUTABLE = "matmul_recomputable"  # anchor all but matmuls
SAVE_OPS_NONE = "none"  # anchor nothing

# Matmuls are the only save_op we are happy to recompute, because benchmark mode
# times them accurately and they are deterministic. Attention, HOPs, topk and
# collectives are either priced badly by the estimator or unsafe to replay, so
# they stay anchored.
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


# bytes in one GiB
MEM_MULTIPLIER = 1 << 30

# the estimator reads low by up to ~1.3 GiB, so leave a little slack
CALIBRATION_SAFETY_GB = 1.5

# Only offload tensors at least this big, since many small transfers cost more
# overhead than they save.
OFFLOAD_MIN_BYTES = 1 << 20  # 1 MiB

# The modeled peak is only a proxy, so we materialize each plan, measure the
# real peak and correct the cap. Two or three rounds is usually enough.
CALIBRATION_MAX_ITERS = 12
CALIBRATION_TOL_GB = 1.5  # stop once the measured peak is this close under budget

# Every rank pins its own offloaded activations, so they all share whatever host
# memory the node has free.
HOST_MEMORY_FRACTION = 0.8

# Weight on the offload term in the outer objective. Zero prices offload as
# free, which over-offloads by about 15% but always meets the budget. Charging
# the full transfer time is worse, because then the LP stops offloading at all
# and offload is the only way to free activations we cannot recompute. The real
# fix is the bandwidth input rather than this weight.
OFFLOAD_TIME_PRICE = 0.0

# Anchoring attention HOPs cost about 1.4 GiB of extra floor and 4% throughput
# on qwen3-14B for no gain, so anchor nothing by default.
SAVE_OPS_POLICY = SAVE_OPS_NONE

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


# Step 1: group nodes by transformer block
def block_of_node(node: torch.fx.Node) -> int:
    """Return which transformer block a node belongs to, for now just its layer id."""
    return _get_layer_id(node)


def get_must_keep_list(
    gm: torch.fx.GraphModule, *, save_ops_policy: str = SAVE_OPS_ALL
) -> set:
    """Nodes the solver may not recompute, though it may still keep or offload them.

    These are the RNG ops, whose random state the remat pass cannot reproduce,
    the compute-heavy save_ops picked out by save_ops_policy, and the layer
    boundaries. Boundaries are anchored under every policy so that each layer's
    recompute region stays self-contained and the inner solves stay independent.
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


def _sync_plan_from_rank0(gm: torch.fx.GraphModule) -> None:
    """Force every rank to use rank 0's memory policy.

    Each rank solves on its own copy of the graph and the solve is not
    bit-reproducible, so ranks can land on different optima, tag different
    nodes, and then deadlock in NCCL on a mismatched all-gather. Broadcasting
    rank 0's answer is exact under SPMD, since the graph is the same everywhere
    and only the solver's choice differs.
    """
    import torch.distributed as dist

    if not dist.is_available() or not dist.is_initialized():
        return
    local = {
        n.name: n.meta["recompute"] for n in gm.graph.nodes if "recompute" in n.meta
    }
    payload = [local if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    plan = payload[0]

    if set(plan) != set(local):
        # The graphs themselves differ, which broadcasting cannot fix, so fail
        # rather than tag a graph we did not solve for.
        only0 = sorted(set(plan) - set(local))[:5]
        only_here = sorted(set(local) - set(plan))[:5]
        raise RuntimeError(
            f"graph structure differs across ranks: rank {dist.get_rank()} has "
            f"{len(local)} tagged nodes vs rank 0's {len(plan)}. Only on rank 0: "
            f"{only0}; only here: {only_here}."
        )

    changed = 0
    for n in gm.graph.nodes:
        if n.name in plan:
            if n.meta.get("recompute") is not plan[n.name]:
                changed += 1
            n.meta["recompute"] = plan[n.name]
    digest = hashlib.sha256(
        "\n".join(f"{k}={plan[k]}" for k in sorted(plan)).encode()
    ).hexdigest()[:16]
    _dbg(
        "memory policy: adopted rank 0's plan (digest %s over %d tagged nodes); "
        "%d local decision(s) overridden",
        digest,
        len(plan),
        changed,
    )


def _is_recomputable(node: torch.fx.Node, must_keep: set) -> bool:
    """Whether the remat pass may erase and replay this node.

    This mirrors what the tagging path really allows, so both solvers agree. A
    storage that can be neither recomputed nor offloaded is resident under every
    plan, and its bytes have to count as fixed rather than freeable.
    """
    return node not in must_keep  # and not _is_collective_or_wait(node)


def _offload_forbidden(node: torch.fx.Node, size: int, must_keep: set) -> bool:
    """Whether the ILP must not tag this node for CPU offload.

    On top of what _can_offload_node already rejects, this also bars must_keep.
    Offloading a layer boundary hurts badly, because it feeds the whole next
    layer and the o + r <= 1 constraint then blocks recompute for everything
    reading it, which on qwen3-14B moved the achievable peak from 66.92 to
    71.34 GiB. Both solvers must agree here, or the outer plans an offload
    fraction the inner cannot build.
    """
    if size < OFFLOAD_MIN_BYTES or not _can_offload_node(node):
        return True
    return node in must_keep


def _save_op_is_anchored(node: torch.fx.Node, policy: str) -> bool:
    """Whether a save_op is barred from recompute under this policy.

    The node is already known to be a save_op.
    """
    if policy == SAVE_OPS_ALL:
        return True
    if policy == SAVE_OPS_NONE:
        return False
    # SAVE_OPS_MATMUL_RECOMPUTABLE: anchor everything except the matmul family.
    target = node.target
    if isinstance(target, torch._ops.OpOverload):
        return target._overloadpacket not in _MATMUL_OVERLOAD_PACKETS
    return True  # HOPs are always anchored


# ---------------------------------------------------------------------------
# Inner ILP: one independent solve per transformer block.
# ---------------------------------------------------------------------------
def _validate_fractions(
    keep_fraction: float, recompute_fraction: float, offload_fraction: float
) -> tuple[float, float, float]:
    """Check the keep/recompute/offload split is a valid distribution."""
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
    """Bucket a node's final tag into keep, recompute or offload.

    Saved and untagged nodes both count as keep, since either way the storage
    stays on the GPU.
    """
    pol = node.meta.get("recompute")
    if pol in (CheckpointPolicy.MUST_CPU_OFFLOAD, CheckpointPolicy.PREFER_CPU_OFFLOAD):
        return "offload"
    if pol in (CheckpointPolicy.MUST_RECOMPUTE, CheckpointPolicy.PREFER_RECOMPUTE):
        return "recompute"
    return "keep"


def _demote_collective_recompute_tags(gm: torch.fx.GraphModule) -> int:
    """Make sure no collective or wait is left tagged MUST_RECOMPUTE.

    The remat pass duplicates every MUST_RECOMPUTE node, and duplicating a wait
    gives its all-gather a second user, which the bucketing pass rejects.
    Recompute is wrong here anyway, since it would re-issue the communication.

    A tag can reach a collective by several routes, so rather than guard each
    one the invariant is enforced once here, after all tagging. Returns how many
    tags were demoted.
    """
    demoted = 0
    for node in gm.graph.nodes:
        if node.op != "call_function" or _is_backward_node(node):
            continue
        if not _is_collective_or_wait(node):
            continue
        if node.meta.get("recompute") is CheckpointPolicy.MUST_RECOMPUTE:
            node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
            demoted += 1
    if demoted:
        _dbg(
            "tagging: demoted %d collective/wait node(s) from MUST_RECOMPUTE to "
            "MUST_SAVE (recomputing a collective would re-issue the "
            "communication and break all-gather bucketing)",
            demoted,
        )
    return demoted


def _audit_tagged_fractions(candidates_by_layer):
    """Measure the keep/recompute/offload split actually written to the graph.

    This reads the tags instead of trusting the solver's variables, so it also
    covers layers the ILP skipped. It reports intent though, not outcome, since
    a later pass can still reject a tag.

    Returns the split per layer and in total.
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


# the overall ilp, outer plus inner
_TAG_KEY = "recompute"


def _snapshot_tags(gm: torch.fx.GraphModule) -> dict:
    """Record every node's current tag so a plan can be undone later."""
    return {n: n.meta.get(_TAG_KEY, None) for n in gm.graph.nodes}


def _restore_tags(gm: torch.fx.GraphModule, snap: dict) -> None:
    for n in gm.graph.nodes:
        v = snap.get(n, None)
        if v is None:
            n.meta.pop(_TAG_KEY, None)
        else:
            n.meta[_TAG_KEY] = v


def _measure_tagged_peak(
    gm: torch.fx.GraphModule,
    trace: TracedResult,
    opt_bytes: int,
    prefetch_lookahead: int,
    defer_n_layers: int,
):
    """Estimate the peak of the graph the run will really execute.

    The outer LP scores plans with a proxy, and that proxy is what misses the
    budget. So instead we materialize the plan on a throwaway clone, using the
    same two passes as the real pipeline, and ask the peak estimator. The clone
    shares the parent's parameters, so this costs graph-copy time rather than
    weight memory.

    Returns the peak including optimizer state, or (None, None) if the probe
    could not be built.
    """
    import copy as _copy

    from torchtitan.experiments.graph_trainer.cpu_offload import apply_cpu_offload_pass
    from torchtitan.experiments.graph_trainer.selective_activation_remat import (
        selective_activation_remat_pass,
    )

    try:
        probe = torch.fx.GraphModule(gm, _copy.deepcopy(gm.graph))
    except Exception as e:  # noqa: BLE001 fall back to the uncalibrated path
        logger.warning("calibrate: could not clone the graph (%s); skipping", e)
        return None, None
    n_tagged = sum(1 for n in probe.graph.nodes if n.meta.get(_TAG_KEY) is not None)
    apply_cpu_offload_pass(
        probe,
        None,
        prefetch_lookahead=prefetch_lookahead,
        defer_n_layers=defer_n_layers,
    )
    selective_activation_remat_pass(probe, None)
    est = estimate_peak_memory(probe, num_state_inputs=trace.num_static_inputs)
    _pc = est.per_category_at_peak
    _n_nodes = len(list(probe.graph.nodes))
    # Drop the clone before returning. Each probe holds graph state that keeps
    # device memory alive, and the calibration loop builds one per iteration, so
    # leaving them around OOMs before the first step.
    del probe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _dbg(
        "calibrate: probe %d tags -> %d nodes | peak %.2f GiB at node %s | "
        "Activation: %.2f Grad: %.2f PARAM: %.2f TEMP: %.2f",
        n_tagged,
        _n_nodes,
        est.peak_bytes / MEM_MULTIPLIER,
        getattr(est, "peak_node_index", "?"),
        _pc.get(ACT, 0) / MEM_MULTIPLIER,
        _pc.get(GRAD, 0) / MEM_MULTIPLIER,
        _pc.get(PARAM, 0) / MEM_MULTIPLIER,
        _pc.get(TEMP, 0) / MEM_MULTIPLIER,
    )
    return est.peak_bytes + opt_bytes, est


def two_level_ilp(
    trace: TracedResult,
    memory_budget: int,
    optimizer,
    model_parts: list[torch.nn.Module],
    runtime_estimation_mode: str = COST_MODEL,
    cpu_offload_budget_gb: float = 100.0,
    interp_ctx: tuple | None = None,  # (model, *run_args) for INTERPRETER mode
    each_layer_separately: bool = True,
    prefetch_lookahead: int = 1,
    defer_n_layers: int = 1,
    debug_logging: bool = False,
    cpu_offload_bw: int = 10000,
) -> torch.fx.GraphModule | None:
    """Run the two-level solver and tag the graph with per-node decisions.

    Returns the graph and its metrics, or None if the solve failed.
    """
    global _debug_logging
    _debug_logging = debug_logging

    _t0 = time.perf_counter()
    _timings = {}
    gm = trace.gm

    _t = time.perf_counter()
    mem_est = estimate_peak_memory(gm, num_state_inputs=trace.num_static_inputs)
    opt_bytes = optimizer_state_bytes(optimizer, model_parts[0])
    _timings["mem_estimation"] = time.perf_counter() - _t
    estimated = mem_est.peak_bytes + opt_bytes  # all-keep peak, optimizer included

    if memory_budget > estimated:
        _dbg(
            "new-autoAC: budget %.2f GB >= estimated peak %.2f GB; nothing to do",
            memory_budget / 1 << 30,
            estimated / 1 << 30,
        )
        return gm, None

    _dbg(
        "2-level ILP-based autoAC: runtime estimation mode = %s",
        runtime_estimation_mode,
    )

    # All-keep breakdown before the passes run. The trainer logs the same
    # categories afterwards, so diffing the two shows what the passes moved.
    _M = MEM_MULTIPLIER
    _pc = mem_est.per_category_at_peak
    _dbg(
        "PRE-PASS peak %.2f GiB (+opt %.2f = %.2f) | Activation: %.2f Grad: %.2f "
        "INPUT: %.2f PARAM: %.2f TEMP: %.2f",
        mem_est.peak_bytes / _M,
        opt_bytes / _M,
        estimated / _M,
        _pc.get(ACT, 0) / _M,
        _pc.get(GRAD, 0) / _M,
        _pc.get(INPUT, 0) / _M,
        _pc.get(PARAM, 0) / _M,
        _pc.get(TEMP, 0) / _M,
    )

    def _mem_probe(tag):
        if torch.cuda.is_available():
            _dbg(
                "MEMPROBE %-22s allocated=%7.2f GiB reserved=%7.2f GiB",
                tag,
                torch.cuda.memory_allocated() / (1 << 30),
                torch.cuda.memory_reserved() / (1 << 30),
            )

    _mem_probe("before runtime_est")
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
    _mem_probe("after runtime_est")

    # Adopt rank 0's runtimes. Under benchmark mode each rank times its own
    # kernels and the totals land a few percent apart, which matters because the
    # C3 offload windows are bandwidth times a measured forward time. Different
    # windows per rank means the cap rank 0 picks during calibration can be
    # infeasible on another rank, and the one re-solve after the broadcast then
    # raises there alone while rank 0 sails through.
    #
    # This has to happen here rather than in plan_outer: the calibration loop
    # runs on rank 0 only, so a collective inside plan_outer would hang.
    if dist.is_available() and dist.is_initialized():
        _rt_payload = [runtime.node_runtimes_ms if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(_rt_payload, src=0)
        runtime.node_runtimes_ms = _rt_payload[0]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        _mem_probe("after empty_cache")

    _t = time.perf_counter()
    get_fixed_bytes_tuple = get_fixed_bytes(gm, trace.num_static_inputs)
    _timings["get_fixed_bytes"] = time.perf_counter() - _t

    # --- outer plan, calibrated against the materialized graph ---------------
    # Solving the outer LP once does not land on the budget. Its peak model is
    # only a proxy, and since the LP drives its own slack to zero, the proxy
    # error passes straight through to the realized peak.
    #
    # So search the cap rather than trust it. The peak grows with the cap but in
    # jumps, because the inner tagging is discrete, which makes a secant
    # correction oscillate. Bracket and then bisect instead, keeping the largest
    # cap whose materialized peak still fits.
    _cal_tol = CALIBRATION_TOL_GB * MEM_MULTIPLIER
    # Already resolved and validated by resolve_host_offload_cap_gib().
    _host_cap_gib = float(cpu_offload_budget_gb)
    # Transfer rate for the C3 offload window: the idle single-GPU rate divided
    # by log2 of the ranks sharing this node's host path.

    # Ranks on other nodes do not share this path, so use the local count. Only
    # local rank 0 measures, since ranks benchmarking at once overlap each other
    # and the reading swings by a long way.
    # _lrank = int(os.environ.get("LOCAL_RANK", "0"))
    _rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 0
    )
    if cpu_offload_bw == 10000:
        _solo_bw = get_transfer_bw() if _rank == 0 else {"d2h": 0.0, "h2d": 0.0}

        if dist.is_available() and dist.is_initialized():
            # Take the max, since the non-measuring ranks carry 0.0 and a min
            # would hand every rank a bandwidth of zero.
            _t = torch.tensor(
                [_solo_bw["d2h"], _solo_bw["h2d"]],
                dtype=torch.float64,
                device=f"cuda:{torch.cuda.current_device()}",
            )
            dist.all_reduce(_t, op=dist.ReduceOp.MAX)
            _solo_bw = {"d2h": float(_t[0]), "h2d": float(_t[1])}

        # Derate after the reduce so every rank scales the same number.
        _local_ranks = max(1, int(os.environ.get("LOCAL_WORLD_SIZE", "1")))
        _bw_scale = max(1.0, math.log2(_local_ranks))
        _measured_bw = {k: v / _bw_scale for k, v in _solo_bw.items()}
        _dbg(
            "C3 INPUTS: idle d2h=%.1f h2d=%.1f GB/s -> per-rank d2h=%.1f h2d=%.1f "
            "GB/s after /log2(%d)",
            _solo_bw["d2h"],
            _solo_bw["h2d"],
            _measured_bw["d2h"],
            _measured_bw["h2d"],
            _local_ranks,
        )
    else:
        _measured_bw = {"d2h": 0.0, "h2d": 0.0}
        _measured_bw["d2h"] = cpu_offload_bw
        _measured_bw["h2d"] = cpu_offload_bw

    # Only rank 0 searches. When every rank searched, the low-bit differences in
    # their LP coefficients got amplified into structurally different plans that
    # _sync_plan_from_rank0 then rejected. Rank 0 finds the cap and broadcasts
    # it, then every rank does one identical solve at that cap, which is also
    # much cheaper for the other ranks.

    _cap = float(
        memory_budget - opt_bytes
    )  # efficient budget (without optimizer bytes)
    _lo = None  # largest cap whose materialized peak fits
    _hi = None  # smallest cap whose materialized peak overshoots
    _base_tags = _snapshot_tags(gm)
    _best = None  # best FITTING plan: (peak, tags, fractions, cap, inner_fr, bs, ss)
    _floor = None  # lowest peak measured, for the error when nothing fits
    _inf = None  # largest cap the LP called infeasible
    _feas = None  # smallest cap the LP solved
    _t = time.perf_counter()
    _t_construct_ilp = _t_solve = 0.0
    fractions = None
    _n_iter = 0

    NGPU = int(os.environ.get("NGPU", "0"))

    if _rank == 0:
        for _it in range(CALIBRATION_MAX_ITERS if _rank == 0 else 0):
            _n_iter = _it + 1
            _restore_tags(gm, _base_tags)
            # fractions from outer, time for construction, time for solve
            _fr, _tc, _ts = plan_outer(
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
                eff_budget_override=_cap,
                host_cap_gib=_host_cap_gib,
                measured_bw=_measured_bw,
            )
            _t_construct_ilp += _tc or 0.0
            _t_solve += _ts or 0.0
            if _fr is None:
                # The cap sits below the LP's feasibility boundary, so remember it
                # as a lower bound and bisect up toward the cheapest feasible cap.
                # Stepping up blindly made the search oscillate across the boundary
                # and burn every iteration without finding the tightest plan.
                if _best is not None:
                    break
                _inf = _cap if _inf is None else max(_inf, _cap)
                _nxt = (
                    (_inf + _feas) / 2.0
                    if _feas is not None
                    else _cap + max(0.5 * MEM_MULTIPLIER, 0.05 * _cap)
                )
                if _feas is not None and (_feas - _inf) < 0.1 * MEM_MULTIPLIER:
                    break  # boundary located; nothing tighter exists
                _cap = _nxt
                if _cap > memory_budget:
                    break
                continue
            _feas = _cap if _feas is None else min(_feas, _cap)
            fractions = _fr
            _gm_t, _bb, _bs, _ifr = plan_and_tag_inner(
                mem_est,
                opt_bytes,
                runtime,
                get_fixed_bytes_tuple,
                trace,
                gm,
                runtime_estimation_mode,
                cpu_offload_budget_gb,
                interp_ctx,
                keep_fraction=0.05,
                recompute_fraction=0.90,
                offload_fraction=0.05,
                per_layer_fractions=_fr,
                each_layer_separately=each_layer_separately,
            )
            _peak, _est = _measure_tagged_peak(
                gm, trace, opt_bytes, prefetch_lookahead, defer_n_layers
            )
            if _peak is None:
                break  # cloning unavailable; keep this plan uncalibrated
            _err = _peak - memory_budget
            _dbg(
                "calibrate iter %d: cap=%.2f GiB -> materialized peak=%.2f GiB vs "
                "budget %.2f GiB (err %+.2f GiB) [bracket lo=%s hi=%s]",
                _n_iter,
                _cap / MEM_MULTIPLIER,
                _peak / MEM_MULTIPLIER,
                memory_budget / MEM_MULTIPLIER,
                _err / MEM_MULTIPLIER,
                "-" if _lo is None else f"{_lo / MEM_MULTIPLIER:.2f}",
                "-" if _hi is None else f"{_hi / MEM_MULTIPLIER:.2f}",
            )
            _floor = _peak if _floor is None else min(_floor, _peak)
            if _peak <= memory_budget:
                if _best is None or _peak > _best[0]:
                    _best = (_peak, _snapshot_tags(gm), _fr, _cap, _ifr, _bb, _bs)
                _lo = _cap if _lo is None else max(_lo, _cap)
                if memory_budget - _peak <= _cal_tol:
                    break  # just under the budget
                _cap = (
                    (_lo + _hi) / 2.0
                    if _hi is not None
                    else _lo + (memory_budget - _peak)
                )
            else:
                _hi = _cap if _hi is None else min(_hi, _cap)
                if _lo is not None:
                    _cap = (_lo + _hi) / 2.0
                elif _inf is not None:
                    _cap = (_inf + _cap) / 2.0  # push toward the feasible floor
                else:
                    _cap = _cap - _err
            if (
                _lo is not None
                and _hi is not None
                and (_hi - _lo) < 0.25 * MEM_MULTIPLIER
            ):
                break  # bracket collapsed; _best is the answer

        # Share rank 0's verdict: the winning cap, the measured peak and the
        # floor. Everything after this point is identical on every rank.
        _verdict = [
            (
                _best[3] if _best is not None else None,  # winning cap
                _best[0] if _best is not None else None,  # its materialized peak
                _floor,
                _n_iter,
            )
        ]
    else:
        _verdict = [None]

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(_verdict, src=0)

    if _verdict[0] is None:
        raise ValueError("AC solver plan crashed!")
    _won_cap, _won_peak, _floor, _n_iter = _verdict[0]
    if _won_cap is None:
        # No split reaches this budget. In this case, just reaise error instead of a fallback
        _floor_txt = (
            f"the tightest plan measured is {_floor / MEM_MULTIPLIER:.2f} GiB"
            if _floor is not None
            else "no split reached a solvable plan at any cap"
        )
        _need = (
            f"{_floor / MEM_MULTIPLIER:.1f}" if _floor is not None else "a higher value"
        )
        raise ValueError(
            f"memory_budget_gb={memory_budget / MEM_MULTIPLIER:.2f} GiB is "
            f"not achievable for this model and parallelism: {_floor_txt} "
            f"after {_n_iter} calibration iterations. Raise "
            f"--compile.memory_budget_gb to at least {_need}."
        )
    _dbg(
        "calibrate: chose budget %.2f GiB -> materialized peak %.2f GiB "
        "(budget %.2f GiB, headroom %.2f GiB, %d iterations on rank 0)",
        _won_cap / MEM_MULTIPLIER,
        _won_peak / MEM_MULTIPLIER,
        memory_budget / MEM_MULTIPLIER,
        (memory_budget - _won_peak) / MEM_MULTIPLIER,
        _n_iter,
    )

    # One identical solve at the agreed cap on every rank.
    _restore_tags(gm, _base_tags)
    fractions, _tc, _ts = plan_outer(
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
        interp_ctx,
        eff_budget_override=_won_cap,
        host_cap_gib=_host_cap_gib,
        measured_bw=_measured_bw,
    )
    _t_construct_ilp += _tc or 0.0
    _t_solve += _ts or 0.0
    _timings["outer_ilp"] = time.perf_counter() - _t

    # Rank 0's split is the one that counts. Sometimes, every rank might end up
    # having slightly different plans - not intended.
    _fr_payload = [fractions if _rank == 0 else None]
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(_fr_payload, src=0)
    fractions = _fr_payload[0]

    if fractions is None:
        # The budget is unachievable.
        raise ValueError(
            f"memory_budget_gb={memory_budget / MEM_MULTIPLIER:.2f} GiB is not "
            "achievable for this model and parallelism: the outer solver found "
            "no feasible keep/recompute/offload split. Raise "
            "--compile.memory_budget_gb."
        )

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

    _sync_plan_from_rank0(gm)

    _dbg(
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
        "budget_gb": memory_budget / (1 << 30),
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
    eff_budget_override: float | None = None,
    host_cap_gib: float = 0.0,
    measured_bw: dict | None = None,
):
    """Solve the outer LP for per-layer keep/recompute/offload fractions.

    It spreads the memory budget across the transformer blocks and hands the
    resulting fractions to the inner solver. Returns None when the graph has no
    block activations or when no feasible split exists.
    """
    gm = trace.gm
    estimated = mem_est.peak_bytes + opt_bytes
    _dbg(f"memory_budget: {memory_budget}")
    _dbg(f"estimated: {estimated}")
    _dbg(f"opt_bytes: {opt_bytes}")


    # The LP's peak model is only a proxy, calibration can help us
    eff_budget = (
        memory_budget - opt_bytes
        if eff_budget_override is None
        else eff_budget_override
    )

    # --- ablation knobs, all defaulting to the current model ---

    # Offload eviction lag, disabled by default. Only used for debugging
    # might delete later
    _abl_evict_mult = 0.0  # offload eviction lag mult
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
        candidates_by_layer,
        layer_act,
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

    # The outer ILP decides how much each layer keeps, sharing out the resources
    # they compete for: GPU peak memory, host memory and PCIe bandwidth.

    block_names = list(blocks)  # <- index i  <->  block_names[i]
    num_blocks = len(block_names)

    _block_act = block_activation_bytes(mem_est)  # freeable activation bytes per block
    block_act = {
        b: _block_act.get(b, 0) for b in block_names
    }  # ensure every block present
    bw = get_transfer_bw()
    bw_d2h, bw_h2d = bw["d2h"] * 1e6, bw["h2d"] * 1e6  # GB/s -> bytes/ms
    # This benchmark used to be both noisy and optimistic, because the offload
    # path did not always get pinned destination buffers while the benchmark
    # did. With the pinned pool on they use the same buffers, so it now measures
    # what offload will really get and no correction belongs here. Still
    # overridable per machine.
    for _dir in ("d2h", "h2d"):
        _val = (measured_bw or {}).get(_dir, bw[_dir])
        _dbg(
            "C3 INPUTS: %s benchmarked %.1f GB/s -> using %.1f GB/s",
            _dir,
            bw[_dir],
            _val,
        )
        bw[_dir] = float(_val)
    bw_d2h, bw_h2d = bw["d2h"] * 1e6, bw["h2d"] * 1e6

    _dbg(
        "C3 INPUTS: bw_d2h=%.1f GB/s bw_h2d=%.1f GB/s (isolated benchmark)",
        bw["d2h"],
        bw["h2d"],
    )

    fwd_rt_by_block = defaultdict(float)
    bwd_rt_by_block = defaultdict(float)
    total_fwd_time = 0
    total_bwd_time = 0

    for _, blk_nodes in blocks.items():
        for n in blk_nodes:
            if _is_backward_node(n):
                total_bwd_time += runtime_per_node.get(n.name, 0.0)
            else:
                total_fwd_time += runtime_per_node.get(n.name, 0.0)

    for block_id in range(num_blocks):
        bwd_rt_by_block[block_id] = total_bwd_time / num_blocks
        fwd_rt_by_block[block_id] = total_fwd_time / num_blocks

    # The C3 offload windows are bandwidth times these numbers, so log the
    # totals to check them against a profiler trace. From inside the LP an
    # inflated window looks exactly like optimistic bandwidth.
    _fsum = sum(fwd_rt_by_block.values())
    _bsum = sum(bwd_rt_by_block.values())
    _dbg(
        "C3 INPUTS: sum fwd_rt=%.1f ms sum bwd_rt=%.1f ms (total %.1f ms over "
        "%d blocks; per-block fwd=%.2f bwd=%.2f)",
        _fsum,
        _bsum,
        _fsum + _bsum,
        len(fwd_rt_by_block) or 1,
        _fsum / (len(fwd_rt_by_block) or 1),
        _bsum / (len(bwd_rt_by_block) or 1),
    )

    # Outer LP over continuous per-layer keep/recompute/offload fractions.
    # Memory is scaled to GiB so the solver coefficients stay small.
    M = MEM_MULTIPLIER
    layer_ids = sorted(blocks)  # int layer ids in forward order: 0,1,...,L-1
    B_g = eff_budget / M
    Ppeak_g = mem_est.peak_bytes / M
    aG = {b: act_bytes_per_layer[b] / M for b in layer_ids}  # owned act, GiB
    bw_d2h_g, bw_h2d_g = bw_d2h / M, bw_h2d / M  # GiB/ms

    live_bytes = mem_est.live_bytes

    # Recompute working set, meaning a layer's forward temporaries. Still can cause
    # problems - so calibration can help
    _layer_temp = defaultdict(float)
    for _prod, _ents in mem_est.all_tensors.items():
        _b = block_of_node(_prod)
        if isinstance(_b, int) and _b >= 0 and not _is_backward_node(_prod):
            for _e in _ents:
                if _e["category"] == TEMP:
                    _layer_temp[_b] += _e["size"]
    recompute_working_set = {b: _layer_temp.get(b, 0.0) / M for b in layer_ids}

    # Nodes the remat pass may not erase. Same policy the inner solver uses, so
    # both levels agree on which candidate bytes are actually freeable.
    must_keep = get_must_keep_list(gm, save_ops_policy=SAVE_OPS_POLICY)

    _t_construct_ilp_start = time.perf_counter()

    k, r, o = {}, {}, {}
    prob = LpProblem("outer_ilp", LpMinimize)
    for b in layer_ids:
        k[b] = LpVariable(f"k_{b}", lowBound=0, upBound=1)
        r[b] = LpVariable(f"r_{b}", lowBound=0, upBound=1)
        o[b] = LpVariable(f"o_{b}", lowBound=0, upBound=1)
        prob += k[b] + r[b] + o[b] == 1, f"split_{b}"

    # Recompute holds a layer's forward temporaries live while it replays, so the
    # peak carries the largest such working set, not the sum.
    ws = LpVariable("ws_rec", lowBound=0)
    for b in layer_ids:
        prob += ws >= recompute_working_set[b] * r[b], f"ws_{b}"

    # Per-layer freeable ceiling. Freeable here means the inner solver's notion rather.
    _unfree_bytes = defaultdict(float)
    for _b, _cands in candidates_by_layer.items():
        _li = int(_b.split(".")[1]) if isinstance(_b, str) else _b
        for _n, _sz in _cands:
            _no_r = not _is_recomputable(_n, must_keep)
            _no_o = _offload_forbidden(_n, _sz, must_keep)
            if _no_r and _no_o:
                _unfree_bytes[_li] += _sz
    for b in layer_ids:
        _den = max(layer_act.get(b, act_bytes_per_layer[b]), 1)
        _ff = max(0.0, 1.0 - _unfree_bytes.get(b, 0.0) / _den)
        prob += r[b] + o[b] <= _ff, f"freeable_{b}"
    _tot_act = sum(layer_act.get(b, act_bytes_per_layer[b]) for b in layer_ids)
    _tot_unfree = sum(_unfree_bytes.get(b, 0.0) for b in layer_ids)
    _dbg(
        "outer: UNFREEABLE (not recomputable AND not offloadable) = %.2f / %.2f "
        "GiB -> keep floor %.3f%%; no plan can keep less than this. outer act "
        "basis=%.2f GiB, inner act basis=%.2f GiB",
        _tot_unfree / M,
        _tot_act / M,
        100.0 * _tot_unfree / max(_tot_act, 1),
        sum(act_bytes_per_layer[b] for b in layer_ids) / M,
        _tot_act / M,
    )

    nodes_per_layer = max(1, len(nodes) // max(len(layer_ids), 1))
    evict_lag = int(_abl_evict_mult * nodes_per_layer)  # D2H eviction lag (0 = off)
    n_peak_cons = 0

    # Go over all schedule points
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
                fb_off[L] += so.size  # exclusive: storage counts for offload only
            else:
                fb_rec[L] += so.size  # recompute frees across the fwd->bwd gap
                if _off_elig:  # non-exclusive: also offload-freeable in its window
                    fb_off[L] += so.size
        if not fb_rec and not fb_off:
            continue  # grad/loss-dominated -> floor handles it
        freed = lpSum(r[L] * (fb_rec[L] / M) for L in fb_rec) + lpSum(
            o[L] * (fb_off[L] / M) for L in fb_off
        )
        # At a backward position only the layer being recomputed there holds its
        # forward temporaries live, so charge the working set during backward.
        ws_here = ws if _is_backward_node(nodes[t]) else 0
        prob += lb / M - freed + ws_here <= B_g, f"peak_{t}"
        n_peak_cons += 1
    _dbg("outer: %d per-position peak constraints", n_peak_cons)

    # (C3) offload windows: a layer's transfers have to hide inside the compute
    # window beside them.
    for b in layer_ids:
        prob += (
            o[b] * aG[b] <= bw_d2h_g * fwd_rt_by_block[b],
            f"d2h_{b}",
        )
        prob += (
            o[b] * aG[b] <= bw_h2d_g * bwd_rt_by_block[b],
            f"h2d_{b}",
        )

        _dbg(
            f"layer: {b}, bw_d2h_g: {bw_d2h_g}, fwd_rt_by_block[b]: {fwd_rt_by_block[b]}"
        )
        _dbg(
            f"layer: {b}, bw_h2d_g: {bw_h2d_g}, bwd_rt_by_block[b]: {bwd_rt_by_block[b]}"
        )

    # Host-side pinned-memory cap, computed once by the caller and identical on
    # every rank (see resolve_host_offload_cap_gib).
    prob += (
        lpSum(o[b] * aG[b] for b in layer_ids) <= host_cap_gib,
        "host_cap",
    )
    _dbg("outer: host pinned-memory cap %.2f GiB/rank", host_cap_gib)

    _nl = max(layer_ids) or 1
    _tiebreak = (
        # Tiebreak: among equal-cost plans prefer keeping in later layers,
        # whose activations die sooner and so cost less peak.
        -1e-4
        * lpSum((b / _nl) * k[b] for b in layer_ids)
    )
    # debugging purpose:
    _off_price = OFFLOAD_TIME_PRICE
    _off_ms = {b: _off_price * aG[b] / max(bw_d2h_g, 1e-12) for b in layer_ids}
    _dbg(
        "outer: per-layer cost at full fraction -- recompute %.1f ms vs offload "
        "%.1f ms (%.2f GiB at %.1f GB/s)",
        fwd_rt_by_block[layer_ids[0]],
        _off_ms[layer_ids[0]],
        aG[layer_ids[0]],
        bw["d2h"],
    )
    prob += (
        lpSum(r[b] * fwd_rt_by_block[b] for b in layer_ids)
        + lpSum(o[b] * _off_ms[b] for b in layer_ids)
        + _tiebreak
    ), "added_runtime"

    _t_construct_ilp_end = time.perf_counter()
    _dbg(
        "outer_ilp construction: CBC solve took %.3f s",
        _t_construct_ilp_end - _t_construct_ilp_start,
    )
    _t_construct_ilp = _t_construct_ilp_end - _t_construct_ilp_start

    _t_solve_start = time.perf_counter()
    status = prob.solve(PULP_CBC_CMD(msg=0))
    _t_solve_end = time.perf_counter()
    _dbg(
        "outer_ilp: CBC solve took %.3f s (status=%s)",
        _t_solve_end - _t_solve_start,
        LpStatus[status],
    )
    _t_solve = _t_solve_end - _t_solve_start

    if LpStatus[status] != "Optimal":
        # An unreachable budget lands here with no number the user can act on,
        # so re-solve the same constraints with one elastic slack and minimize
        # it. That reports the smallest budget admitting any split. This is
        # diagnostic only and does not produce a plan.
        _diag_slack = LpVariable("diag_budget_slack", lowBound=0)
        for _cname in list(prob.constraints):
            if _cname.startswith("peak_"):
                prob.constraints[_cname].addInPlace(-_diag_slack)
        prob.setObjective(lpSum([_diag_slack]))
        _diag_status = prob.solve(PULP_CBC_CMD(msg=0))
        if LpStatus[_diag_status] == "Optimal":
            _short = value(_diag_slack) or 0.0
            logger.warning(
                "outer: LP INFEASIBLE -- budget UNREACHABLE by %.3f GiB. "
                "Budget=%.2f GiB but the tightest plan needs %.2f GiB, i.e. "
                "--compile.memory_budget_gb >= %.2f. At the binding position "
                "every freeable activation is already freed; the remainder is "
                "unfreeable. Returning no plan (the graph stays untagged).",
                _short,
                B_g,
                B_g + _short,
                B_g + _short + opt_bytes / M,
            )
        else:
            logger.warning(
                "outer: LP not optimal (%s), and the elastic re-solve is %s too "
                "-- the infeasibility is NOT the budget; check the d2h_/h2d_ "
                "windows and the split_ constraints",
                LpStatus[status],
                LpStatus[_diag_status],
            )
        return None, None, None

    # What the LP thinks the post-pass peak will be, so it can be compared with
    # the estimator. Each peak constraint evaluates to predicted_peak - B_g, so
    # the binding position is the largest of them. If this says the plan fits
    # and the real run disagrees, the LP's peak model is the thing at fault.
    _pred, _pred_t = -1e30, None
    for _cn, _c in prob.constraints.items():
        if not _cn.startswith("peak_"):
            continue
        _v = _c.value()
        if _v is not None and _v + B_g > _pred:
            _pred, _pred_t = _v + B_g, _cn
    _ws_val = value(ws) or 0.0
    _dbg(
        "outer: LP PREDICTED post-pass peak = %.2f GiB (graph, excl. optimizer) "
        "at %s; B_g=%.2f GiB; slack=%.2f GiB | ws_rec=%.2f GiB | "
        "with optimizer: %.2f GiB vs budget %.2f GiB",
        _pred,
        _pred_t,
        B_g,
        B_g - _pred,
        _ws_val,
        _pred + opt_bytes / M,
        memory_budget / M,
    )

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
    _dbg(
        "new-autoAC outer LP | budget=%.2f eff=%.2f | act=%.2f GiB -> "
        "keep=%.2f (%.3f%%)  recompute=%.2f (%.3f%%)  offload=%.2f (%.3f%%)",
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
    _dbg(
        "outer ILP decisions for fractions per-layer:\n",
    )
    for b in layer_ids:
        _dbg(
            "outer: layer %2d decision: k=%.4f r=%.4f o=%.4f",
            b,
            alloc[b][0],
            alloc[b][1],
            alloc[b][2],
        )

    # The outer LP only produces fractions; the inner ILP does the tagging.
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
    """Solve one independent keep/recompute/offload ILP per transformer block.

    The three fractions sum to 1 and act as hard constraints on each layer: keep
    bounds what stays resident on the GPU, offload bounds what streams to the
    host against a shared budget, and recompute takes whatever is left. The
    objective minimizes recompute runtime, with a small keep reward that breaks
    ties toward the tensors most expensive to recompute.
    """
    _dbg("inner ILP planner and tagger...")
    _t_start = time.perf_counter()  # whole plan_and_tag_inner wall time
    keep_fraction, recompute_fraction, offload_fraction = _validate_fractions(
        keep_fraction, recompute_fraction, offload_fraction
    )

    runtime_per_node = runtime.node_runtimes_ms

    nodes = list(gm.graph.nodes)
    node_index = {n: i for i, n in enumerate(nodes)}
    # Nodes we refuse to recompute: RNG for correctness, plus the save_ops the
    # estimator prices badly or that are unsafe to replay. Matmuls stay
    # recomputable, which is what makes the requested fractions achievable.
    _save_ops_policy = SAVE_OPS_POLICY
    must_keep = get_must_keep_list(gm, save_ops_policy=_save_ops_policy)
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

    _dbg("inner ILP planner and tagger got candidates")
    # The remat pass works on regions, so every node in a region needs a tag and
    # not just the ones holding a storage. Views and aliases take their parent's
    # tag, and leaving the intermediate forward inputs untagged would keep them
    # alive into backward instead of freeing them. The ILP below only overrides
    # the candidate producers on top of this base tagging.

    def _decision_space(node):
        if _is_rng_op(
            node
        ):  # or _is_collective_or_wait(node):  # or other nondeterministic
            return "no_recompute"  # k,o allowed; r forbidden
        return "decidable"  # k,r,o all allowed

    # The ILP cannot tag the must-save nodes such as RNG states and layer
    # boundaries, so tag them here.
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
            # getitem and wait share the parent's storage, so they take its tag.
            # Leaving them untagged would make the remat pass treat them as
            # saved anchors and fail to close the parent's recompute region.
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
            node.meta["recompute"] = (
                CheckpointPolicy.MUST_SAVE
            )  # default; ILP may flip to OFFLOAD
        else:
            node.meta["recompute"] = (
                CheckpointPolicy.MUST_RECOMPUTE
            )  # default; ILP may flip to KEEP/OFFLOAD

    M = MEM_MULTIPLIER
    fixed_g = fixed_bytes / M

    _dbg("inner ILP planner and tagger made early decisions")

    # Global host (pinned CPU) budget for offload, tracked across the per-layer
    # loop so independent layer solves don't collectively blow the host budget.

    cpu_cap_gb = float(cpu_offload_budget_gb)
    remaining_off_g = (cpu_cap_gb * (1 << 30)) / M

    _dbg(
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
        # Walk up the view and alias chain to the real candidate a recomputed
        # dup would read, since x may only be a view of one.
        seen = set()
        while x is not None and x not in seen:
            seen.add(x)
            if x in cand_set:
                return x
            if _is_view(x) and x.all_input_nodes:
                x = x.all_input_nodes[0]  # assume a view has a single input
            else:
                return None
        return None

    total_ach = {"keep": 0, "recompute": 0, "offload": 0}
    # Solver intent per candidate, so the solver-vs-graph gap can be
    # attributed after the post-solve fixups.
    _dbg_intent: dict = {}
    _dbg_unapplied = [0]

    print(f"each_layer_separately: {each_layer_separately}")
    if not each_layer_separately:
        time_for_each_budget = {}

        def _tag_of(k, r, o):
            tag = (
                "keep"
                if k.value() > 0.5
                else "offload" if o.value() > 0.5 else "recompute"
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

        def _collective_chain(node):
            """Upstream nodes of a wait candidate that must share its decision.

            An FSDP unshard is a cast, then an all-gather, then a wait. Only the
            wait's storage survives into backward so the wait is the candidate,
            but the runtime cost and the remat legality both sit on the
            collective, which therefore needs the same tag. Returns an empty
            list for candidates that are not collectives.
            """
            if node.target is not torch.ops._c10d_functional.wait_tensor.default:
                return []
            coll = node.args[0]
            if not isinstance(coll, torch.fx.Node) or not is_all_gather_into_tensor(
                coll
            ):
                return []
            chain = [coll]
            src = coll.args[0]
            # Take the cast feeding this collective only when it feeds nothing
            # else, since a shared cast must not inherit one wait's decision.
            if (
                isinstance(src, torch.fx.Node)
                and src.op == "call_function"
                and len(src.users) == 1
            ):
                chain.append(src)
            return chain

        def _recompute_ms(node):
            """Recompute cost of a candidate, including its collective chain.

            A wait is only a barrier and costs about 0 ms, so charging it alone
            would price a re-issued all-gather at zero and the solver would buy
            every collective recompute for free.
            """
            ms = runtime_per_node.get(node.name, 0.0)
            for up in _collective_chain(node):
                ms += runtime_per_node.get(up.name, 0.0)
            return ms

        def _tag_collective_chain(node, tag):
            # An offloaded wait still has to be produced in forward before the
            # copy runs, so its collective is kept rather than recomputed.
            up_tag = "keep" if tag == "offload" else tag
            for up in _collective_chain(node):
                up.meta["recompute"] = _POLICY_TAG[up_tag]

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
                if not _is_recomputable(node, must_keep):
                    # RNG, anchored save_ops, layer boundaries and collectives.
                    # A recompute tag on a collective is demoted to save after
                    # the solve anyway, so planning one here just leaks into keep.
                    prob += r[node] == 0
                if _offload_forbidden(node, size, must_keep) or _is_collective_or_wait(
                    node
                ):
                    # Match what the offload pass accepts: views, collectives,
                    # waits, non-contiguous tensors and tiny ones cannot be
                    # offloaded.
                    prob += o[node] == 0
                g = size / M
                total_keep += k[node] * g
                total_off += o[node] * g
                total_rec += r[node] * g
                objective += r[node] * _recompute_ms(node)


            cand_set = set(k)  # membership only

            # A recomputed node must not read an offloaded input, because its
            # dup runs in backward and needs that input on the GPU:
            # handled by remat pass already

            # for v in k:
            #     for x in v.all_input_nodes:
            #         u = _resolve_cand(x, cand_set)
            #         if u is not None and u is not v:
            #             prob += o[u] + r[v] <= 1

            # Keep is a hard
            # target where feasible, but forced-save nodes can make a very small
            # keep fraction impossible, so keep_over records only that
            # unavoidable excess. Offload is capped by the remaining host budget
            # and then filled as far as the discrete tensor sizes allow.
            target_keep_g = kf * act_g
            target_off_g = min(of * act_g, remaining_off_g)
            keep_over = LpVariable(
                f"keep_over_{kf}_{of}_{remaining_off_g}_{act}", lowBound=0
            )
            off_short = LpVariable(
                f"off_short_{kf}_{of}_{remaining_off_g}_{act}", lowBound=0
            )
            off_over = LpVariable(
                f"off_over_{kf}_{of}_{remaining_off_g}_{act}", lowBound=0
            )

            prob += total_keep <= target_keep_g + keep_over

            # Offload aims at its target from either side instead of sitting
            # inside a hard band.
            prob += total_off <= remaining_off_g  # hard: real host limit only
            prob += off_short >= target_off_g - total_off
            prob += off_over >= total_off - target_off_g

            # Priority order: respect the keep cap, then land closest to the
            # offload target, then minimize recompute time. keep_over has to
            # dominate because exceeding keep costs real peak memory, while
            # missing the offload target only pushes those bytes into recompute,

            prob += (
                # This ratio trades 1 GiB of offload error against 0.1 GiB of
                # keep error. A wider ratio let the solver accept a large offload
                # overshoot just to avoid a fraction of a MB of extra keep.
                1e6 * keep_over
                + 1e5 * (off_short + off_over)
                + objective
                - 1e-6 * (total_keep - keep_over)
            )

            return prob, k, r, o, target_keep_g, target_off_g, keep_over, off_short

        # this will group the layers that have the same budget fractions by the outer
        layer_groups = defaultdict(list)  # sig -> [layer_ids]
        sig_meta = {}  # sig -> (candidates_of_representative, kf, of)

        def _node_key(node, size):
            fqn = node.meta.get("custom", {}).get(_MODULE_FQN, "")
            # Strip the "layers.N." prefix to get a layer-relative key. Some
            # nodes are just "layers.N", the block boundary, with no submodule
            # suffix, so guard the split for those.
            if fqn.startswith("layers."):
                parts = fqn.split(".", 2)
                rel = parts[2] if len(parts) >= 3 else ""  # "" = block output/boundary
            else:
                rel = fqn
            return (rel, str(node.target), size)

        def _layer_keys(candidates):
            # Make the keys unique by appending an ordinal, so candidates that
            # collide still get distinct keys. Every layer builds its candidates
            # the same way, so the i-th collision maps to the same key in each.
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

            _rf = max(0.0, 1.0 - kf - of)
            _dbg(f"new ratios: keep: {kf}, recompute: {_rf}, offload: {of}")

            _keys = _layer_keys(candidates)  # per-layer unique (ordinal-disambiguated)
            sig = (tuple(sorted(_keys)), kf, of)
            layer_groups[sig].append(b)
            sig_meta.setdefault(sig, (candidates, kf, of, layer_act[b]))

        _dbg("inner ILP planner and tagger starting the real work...")

        pattern_by_sig = {}
        skipped = []  # grouped solve does not drop layers; kept for the DONE log
        block_build_s = block_solve_s = 0.0
        for sig, (candidates, kf, of, sig_act) in sig_meta.items():
            _t_b = time.perf_counter()
            prob, k, r, o, tgt_k, tgt_o, kover, oshort = build_inner_ilp(
                candidates, kf, of, remaining_off_g, sig_act
            )
            _t_s = time.perf_counter()
            prob.solve(PULP_CBC_CMD(msg=0))
            block_build_s += _t_s - _t_b
            block_solve_s += time.perf_counter() - _t_s
            if LpStatus[prob.status] != "Optimal":
                # Never read variable values from a failed solve. _tag_of still
                # assigns one tag per node, so the counts add up and the result
                # looks plausible while quietly ignoring both fractions.
                raise RuntimeError(
                    f"inner ILP solve failed: status={LpStatus[prob.status]} for "
                    f"kf={kf} of={of} act={sig_act / M:.2f} GiB "
                    f"(target_keep={tgt_k:.3f} target_off={tgt_o:.3f} GiB). "
                    "Refusing to tag from an unsolved problem."
                )
            pattern_by_sig[sig] = {
                key: _tag_of(k[n], r[n], o[n])
                for key, (n, s) in zip(_layer_keys(candidates), candidates)
            }

        # apply the plan to each layer separately
        given_offload_budget = cpu_offload_budget_gb

        pattern_for_layer = {}
        for sig, layer_id in layer_groups.items():
            for b in layer_id:
                pattern_for_layer[b] = pattern_by_sig[sig]

        def _pattern_keep_g(pattern, candidates):
            return (
                sum(
                    s
                    for key, (n, s) in zip(_layer_keys(candidates), candidates)
                    if pattern.get(key) == "keep"
                )
                / M
            )

        # Second pass, redistributing the unused keep allowance.
        #
        # Every layer in a group gets the same pattern, and a keep allowance
        # rarely tiles exactly with the tensor sizes on offer, so each layer
        # leaves a residual and the realized peak lands several GiB under budget
        # for no benefit. Pool those residuals and re-solve per signature with a
        # raised keep target, then hand the boost to as many layers as it covers.
        #
        # Boost the highest layer indices first, since their backward runs
        # earliest and so keeping them costs the peak the least. Total keep never
        # exceeds what the outer asked for, so its peak constraint still holds.
        _layer_idx = lambda b: (
            int(b.split(".")[1]) if isinstance(b, str) else b
        )  # noqa: E731
        leftover_g = 0.0
        base_keep_by_sig = {}
        for sig, (candidates, kf, of, sig_act) in sig_meta.items():
            base_keep_by_sig[sig] = _pattern_keep_g(pattern_by_sig[sig], candidates)
            leftover_g += len(layer_groups[sig]) * max(
                0.0, kf * (sig_act / M) - base_keep_by_sig[sig]
            )

        _dbg(
            "inner ILP planner and tagger solving the patterns for each candidate layer..."
        )

        _boost_log = []
        for sig, (candidates, kf, of, sig_act) in sig_meta.items():
            if leftover_g <= 1e-3:
                break
            # Find the smallest raise that buys another tensor, so the residual
            # spreads over many layers instead of piling onto one. Start from an
            # even share and double until a candidate fits.
            _n_layers = len(layer_groups[sig])
            _act_g = max(sig_act / M, 1e-9)
            boosted, gain = None, 0.0
            step = leftover_g / max(_n_layers, 1)
            while step <= leftover_g + 1e-9:
                _t_b = time.perf_counter()
                prob_b, kb_, rb_, ob_, _tk, _to, _ko, _os_ = build_inner_ilp(
                    candidates,
                    min(1.0, kf + step / _act_g),
                    of,
                    remaining_off_g,
                    sig_act,
                )
                _t_s = time.perf_counter()
                prob_b.solve(PULP_CBC_CMD(msg=0))
                block_build_s += _t_s - _t_b
                block_solve_s += time.perf_counter() - _t_s
                if LpStatus[prob_b.status] == "Optimal":
                    _cand_pattern = {
                        key: _tag_of(kb_[n], rb_[n], ob_[n])
                        for key, (n, s) in zip(_layer_keys(candidates), candidates)
                    }
                    _g = (
                        _pattern_keep_g(_cand_pattern, candidates)
                        - base_keep_by_sig[sig]
                    )
                    if _g > 1e-6:
                        boosted, gain = _cand_pattern, _g
                        break
                step *= 2
            if boosted is None:
                continue  # nothing bigger fits; the residual is unusable here
            n_boost = min(len(layer_groups[sig]), int(leftover_g / gain))
            if n_boost <= 0:
                continue
            for b in sorted(layer_groups[sig], key=_layer_idx, reverse=True)[:n_boost]:
                pattern_for_layer[b] = boosted
            leftover_g -= n_boost * gain
            _boost_log.append((n_boost, len(layer_groups[sig]), gain))
        if _boost_log:
            _dbg(
                "Inner ILP: keep-residual redistribution -> %s | %.3f GiB still "
                "unusable (no candidate fits)",
                " ".join(
                    f"{n}/{tot} layers +{g:.3f} GiB each" for n, tot, g in _boost_log
                ),
                leftover_g,
            )

        # ---- offload-residual redistribution -------------------------------
        # One pattern per signature means the total offload is quantized to what
        # a single layer can reach, times the number of layers. The tensors are
        # coarse, so a target that is unreachable inside one layer gets missed in
        # every layer even though mixing two patterns would reach it.
        #
        # So solve a second pattern with a raised target and apply it to as many
        # layers as the residual pays for, the same trick the keep redistribution
        # above uses.
        def _pattern_off_g(pattern, candidates):
            return (
                sum(
                    s
                    for key, (n, s) in zip(_layer_keys(candidates), candidates)
                    if pattern.get(key) == "offload"
                )
                / M
            )

        _off_log = []
        for sig, (candidates, kf, of, sig_act) in sig_meta.items():
            _n_layers = len(layer_groups[sig])
            _act_g = max(sig_act / M, 1e-9)
            _base_off = _pattern_off_g(pattern_by_sig[sig], candidates)
            _tgt_off = of * _act_g
            _resid = (_tgt_off - _base_off) * _n_layers
            if _resid <= 1e-3:
                continue
            # Find the smallest raise that buys a bigger offload set, doubling
            # until something fits, so the residual spreads over the most layers
            # instead of piling the whole miss onto a few.
            _boost, _gain = None, 0.0
            _step = _resid / max(_n_layers, 1)
            while _step <= _resid + 1e-9:
                _pb, _kb, _rb, _ob, _tk, _to, _ko, _os2 = build_inner_ilp(
                    candidates,
                    kf,
                    min(1.0, of + _step / _act_g),
                    remaining_off_g,
                    sig_act,
                )
                _pb.solve(PULP_CBC_CMD(msg=0))
                if LpStatus[_pb.status] == "Optimal":
                    _cand = {
                        key: _tag_of(_kb[n], _rb[n], _ob[n])
                        for key, (n, s) in zip(_layer_keys(candidates), candidates)
                    }
                    # Only accept a boost that does not buy offload with keep.
                    # Measure against the keep budget rather than the base
                    # pattern, which usually sits just under it, since otherwise
                    # any boost spending the leftover slack gets rejected.
                    _keep_bar = max(
                        kf * _act_g,
                        _pattern_keep_g(pattern_by_sig[sig], candidates),
                    )
                    if _pattern_keep_g(_cand, candidates) <= _keep_bar + 1e-9:
                        _g = _pattern_off_g(_cand, candidates) - _base_off
                        if _g > 1e-6:
                            _boost, _gain = _cand, _g
                            break
                _step *= 2
            if _boost is None:
                continue
            _n = min(_n_layers, int(round(_resid / _gain)))
            if _n <= 0:
                continue
            for b in sorted(layer_groups[sig], key=_layer_idx, reverse=True)[:_n]:
                pattern_for_layer[b] = _boost
            _off_log.append((_n, _n_layers, _gain))
        if _off_log:
            _dbg(
                "Inner ILP: offload-residual redistribution -> %s",
                " ".join(
                    f"{n}/{tot} layers +{g:.3f} GiB each" for n, tot, g in _off_log
                ),
            )

        _dbg("inner ILP planner and tagger applying the solution to each layer...")

        for b, pattern in pattern_for_layer.items():
            cands = candidates_by_layer[b]
            for key, (node, size) in zip(_layer_keys(cands), cands):
                if key in pattern:
                    tag = pattern[key]
                    node.meta["recompute"] = _POLICY_TAG[tag]
                    _propagate_to_getitem_view(node, tag)
                    _tag_collective_chain(node, tag)
                    total_ach[tag] += size
                    _dbg_intent[node] = tag
                else:
                    _dbg_unapplied[0] += size
                    # TODO accumulate offload bytes here, and fall back to
                    # recompute or save once the host budget is exceeded
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
                if not _is_recomputable(node, must_keep):
                    # RNG, anchored save_ops, layer boundaries and collectives.
                    # A recompute tag on a collective is demoted to save after
                    # the solve anyway, so planning one here just leaks into keep.
                    prob += r[node] == 0
                if _offload_forbidden(node, size, must_keep):
                    # Match what the offload pass accepts: views, collectives,
                    # waits, non-contiguous tensors and tiny ones cannot be
                    # offloaded.
                    prob += o[node] == 0
                g = size / M
                total_keep += k[node] * g
                total_off += o[node] * g
                total_rec += r[node] * g
                objective += r[node] * _recompute_ms(node)

            # A recomputed node must not read an offloaded input, because its
            # dup runs in backward and needs that input on the GPU. Resolve view
            # and alias chains to reach the underlying candidate.
            cand_set = set(k)  # membership only
            # Iterate the dict rather than the set. Node hashing is id-based, so
            # set order differs between processes, and emitting the constraints
            # in a per-rank order makes CBC pick different optima and the ranks
            # then deadlock in NCCL. Dicts are insertion-ordered and safe here.
            for v in k:
                for x in v.all_input_nodes:
                    u = _resolve_cand(x, cand_set)
                    if u is not None and u is not v:
                        prob += o[u] + r[v] <= 1

            # Fraction constraints, which are the peak model. Keep is a hard
            # target where feasible, but forced-save nodes can make a very small
            # keep fraction impossible, so keep_over records only that
            # unavoidable excess. Offload is capped by the remaining host budget.
            #
            # The outer ILP's per-layer fractions win over the global ones, so
            # the inner applies the outer's real plan instead of one flat split.
            if per_layer_fractions is not None:
                _li = int(b.split(".")[1]) if isinstance(b, str) else b
                kf, _rf, of = per_layer_fractions.get(
                    _li, (keep_fraction, recompute_fraction, offload_fraction)
                )
            else:
                kf, of = keep_fraction, offload_fraction
            target_keep_g = kf * act_g
            target_off_g = min(of * act_g, remaining_off_g)
            keep_over = 0.01  # LpVariable(f"keep_over_{b}", lowBound=0)
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
            _dbg(
                "Inner ILP: layer %s timing: %d vars build=%.3fs solve=%.3fs",
                b,
                len(candidates),
                _blk_build_s,
                _blk_solve_s,
            )
            if LpStatus[prob.status] != "Optimal":
                # Fall back to saving the whole layer. That shows up honestly in
                # the audit and avoids leaving base tags in a layer that never
                # got a valid plan.
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
                    else "offload" if o[node].value() > 0.5 else "recompute"
                )
                node.meta["recompute"] = _POLICY_TAG[tag]
                _dbg_intent[node] = tag
                _tag_collective_chain(node, tag)
                # Push the decision onto the producer's getitem, wait and view
                # children so the whole storage chain agrees. A child left on a
                # stale save tag pins the producer's storage, and the plan then
                # fails to lower the real peak. Views cannot be offloaded, so
                # only the getitem children inherit an offload tag.
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
            _dbg(
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

    # _demote_collective_recompute_tags(gm)

    # Computed for both branches (grouped and per-layer) before the audit.
    total_act = sum(layer_act.values()) or 1

    # Read the fractions back from the graph tags rather than from the solver's
    # variable values.
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

    # The solver's own totals, kept for comparison against the verified numbers.
    _dbg(
        "Inner ILP: solver results keep=%.1f%% recompute=%.1f%% offload=%.1f%%",
        100 * total_ach["keep"] / total_act,
        100 * total_ach["recompute"] / total_act,
        100 * total_ach["offload"] / total_act,
    )

    # How many producer nodes landed in each bucket, to go alongside the byte
    # fractions above.
    node_counts = {"keep": 0, "recompute": 0, "offload": 0}
    for _b, _cands in candidates_by_layer.items():
        for _node, _sz in _cands:
            node_counts[_classify_tag(_node)] += 1
    _dbg(
        "Inner ILP: node counts (tagged candidate producers) "
        "keep=%d recompute=%d offload=%d total=%d",
        node_counts["keep"],
        node_counts["recompute"],
        node_counts["offload"],
        sum(node_counts.values()),
    )

    # Authoritative: fractions read from the graph tags the passes will act on.
    est_fwd_peak_g = (fixed_bytes + verified["keep"]) / M
    _dbg(
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
    # Audit the all_gather and wait pairs. The policy dump cannot show this,
    # since it prints each wait's tag under its collective's name, so a pair
    # that disagrees is invisible there.
    _pair_ok, _pair_bad = 0, Counter()
    for _n in gm.graph.nodes:
        if _n.op != "call_function" or _is_backward_node(_n):
            continue
        if _n.target is not torch.ops._c10d_functional.wait_tensor.default:
            continue
        _coll = _n.args[0]
        if not isinstance(_coll, torch.fx.Node) or not is_all_gather_into_tensor(_coll):
            continue
        _wt = _n.meta.get("recompute")
        _ct = _coll.meta.get("recompute")
        if _wt == _ct:
            _pair_ok += 1
        else:
            _pair_bad[(_classify_tag(_n), _classify_tag(_coll))] += 1
    _dbg(
        "Inner ILP: {all_gather,wait} pairs consistent=%d inconsistent=%d%s",
        _pair_ok,
        sum(_pair_bad.values()),
        (
            " | "
            + " ".join(f"wait={w}/coll={c}:{n}" for (w, c), n in _pair_bad.items())
            if _pair_bad
            else ""
        ),
    )

    _dbg(
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


def get_fixed_bytes(
    gm: torch.fx.GraphModule,
    num_state_inputs: int,
):
    """Classify every storage in the graph and collect the per-layer byte pools.

    Returns the per-index live and freeable sets, the fixed and activation bytes
    per layer, and the candidate lists that both solvers plan over.
    """
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
    # Find each storage's last forward use and first backward use, and classify
    # it along the way.
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
                continue  # the same storage is still live at this index

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
            elif is_all_gather_into_tensor(prod) or is_pre_bucket_all_gather(
                prod
            ):  # forward and backward
                category = TEMP  # PARAM
            elif is_pre_bucket_reduce_scatter(prod) or (
                _is_backward_node(prod) and feeds_grad_collective(prod)
            ):
                category = GRAD
            elif _is_backward_node(prod):
                category = GRAD if death_of[key] >= end else TEMP
            else:  # forward-produced compute
                last = nodes[min(death_of[key], end - 1)]
                category = ACT if _is_backward_node(last) else TEMP

            new_object = StorageObject(
                sid=sid,
                size=get_size(t),
                producer_node=node,  # this could also be an int index
                produced_index=index,
                death_index=death_of[key],
                last_fwd_use_index=last_fwd_index,  # for now
                first_bwd_use_index=first_bwd_index,  # for now
                category=category,
            )

            # logger.info(f"node name: {node.name}, category: {category}")

            # layer_id = block_of_node(node)
            # node_index = indices_per_node[node]
            # if layer_id is not None and layer_id != -1:
            #     storages_by_layer[layer_id][node_index] = new_object
            storages_by_prod_node[prod].append(new_object)

            storages.append(new_object)
            total_activation_per_blk += get_size(t)

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
            logger.info(f"skipping object.producer_node: {object.producer_node}")
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

    # Per-index activation liveness and freeable sets for the outer peak
    # constraint. These are dicts because a plain set iterates in id-hash order,
    # which differs between processes, and summing sizes in a different order
    # leaves each rank with slightly different LP coefficients and eventually a
    # different plan. Dict insertion order is graph order on every rank.
    live: dict = {}
    live_freeable: dict = {}

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
            live[obj] = None
        for obj in act_add_at_for_freeable.get(t, ()):
            live_freeable[obj] = None
        # Tuples, not sets: consumers only iterate, and the order must be
        # identical on every rank (see the ordered-set note above).
        live_sids_per_index[t] = tuple(live)  # storage_objects live at index t
        freeable_sids_per_index[t] = tuple(live_freeable)  # freeable at index t
        layer_id = block_of_node(nodes[t])
        accumulative_act_bytes_per_layer[layer_id] = max(
            accumulative_act_bytes_per_layer[layer_id], cum_act_mem
        )
        for obj in act_remove_at.get(t, ()):
            cum_act_mem -= obj.size
            live.pop(obj, None)
        for obj in act_remove_at_for_freeable.get(t, ()):
            live_freeable.pop(obj, None)

    # fixed_bytes_max_by_layer is the non-activation baseline during a layer's
    # region, and act_bytes_per_layer is the activation each layer owns, which
    # is what the outer LP splits up.
    return (
        live_sids_per_index,  # live storages at each node index
        freeable_sids_per_index,  # freeable storages at each node index
        fixed_bytes_max_by_layer,  # per layer bytes we cannot touch
        accumulative_act_bytes_per_layer,  # per layer activation, accumulated
        act_bytes_per_layer,  # per layer activation bytes
        storages_by_prod_node,  # storages keyed by their producer node
        candidates_by_layer,
        layer_act,
        total_freeable,
    )


def find_last_fwd_use_index(node, sid, indices, _memo=None) -> int:
    """Find the last forward index where this storage is read as an input."""
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
        # If the user's output carries the same storage, as a view or an
        # in-place op does, follow the alias chain to its later forward uses.
        if _has_sid(pytree.tree_leaves(user.meta.get("val"))):
            last_fwd_index = max(
                last_fwd_index, find_last_fwd_use_index(user, sid, indices, _memo)
            )
    _memo[node] = last_fwd_index
    return last_fwd_index


def find_first_bwd_use_index(node, sid, indices, _memo=None) -> int:
    """Find the first backward index where this storage is read as an input."""
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
        # Follow the alias chain through both forward and backward nodes, since
        # a forward in-place op can carry the storage to a later backward reader.
        if _has_sid(pytree.tree_leaves(user.meta.get("val"))):  # check outputs
            first_bwd_index = min(
                first_bwd_index,
                find_first_bwd_use_index(user, sid, indices, _memo),
            )
    _memo[node] = first_bwd_index
    return first_bwd_index


def find_last_use_index(node, sid, indices, _memo=None):
    """Find the last index anywhere in the graph where this storage is read."""

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
        if not _has_sid(user_in):  # this user does not read the storage
            continue
        last = max(last, indices[user])  # count every reader
        if _has_sid(pytree.tree_leaves(user.meta.get("val"))):
            last = max(last, find_last_use_index(user, sid, indices, _memo))
    _memo[node] = last
    return last

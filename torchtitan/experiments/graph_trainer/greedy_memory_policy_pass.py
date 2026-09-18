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
    MemoryEstimatorResult,
    optimizer_state_bytes,
    UNKNOWN_OPTIMIZER_BYTES,
)
from torchinsights.graph_estimation._fx_utils import (
    ACT,
    BWD_TEMP,
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
    _get_module_fqn,
    _is_backward_node,
    _MODULE_FQN,
    _NOT_IN_LAYERS,
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

    local_tags = {
        n.name: n.meta["recompute"] for n in gm.graph.nodes if "recompute" in n.meta
    }
    local_names = {n.name for n in gm.graph.nodes}

    payload = [(local_tags, local_names) if dist.get_rank() == 0 else None]
    dist.broadcast_object_list(payload, src=0)
    plan, names0 = payload[0]

    if names0 != local_names:
        only0 = sorted(names0 - local_names)[:5]
        only_here = sorted(local_names - names0)[:5]
        raise RuntimeError(
            f"graph structure differs across ranks: rank {dist.get_rank()} has "
            f"{len(local_names)} nodes vs rank 0's {len(names0)}. "
            f"Only on rank 0: {only0}; only here: {only_here}."
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


def greedy_solve(
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
    logger.info(f"mem_est: {mem_est}")
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
    # Only rank 0 solves. The other ranks pick the plan up from the tag
    # broadcast below, so they never set these; keep placeholders so the
    # metrics block at the end works on every rank.
    _fr = None  # per-layer (keep, recompute, offload) from the outer LP
    _bb = _bs = 0.0  # inner ILP build / solve seconds
    _ifr = None  # fractions the inner ILP actually achieved
    _n_iter = 0

    NGPU = int(os.environ.get("NGPU", "0"))

    if _rank == 0:

        gm, _tc, _ts = plan_and_tag_greedy(
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
            eff_budget_override=None,
            host_cap_gib=_host_cap_gib,
            measured_bw=_measured_bw,
            prefetch_lookahead=prefetch_lookahead,
            defer_n_layers=defer_n_layers,
        )
        _t_construct_ilp += _tc or 0.0
        _t_solve += _ts or 0.0

        if gm is None:
            _verdict = [None]

        else:
            _verdict = [True]
    else:
        _verdict = [None]

    # sync the _verdict
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(_verdict, src=0)

    if not _verdict[0]:
        raise ValueError("AC solver plan crashed!")

    _sync_plan_from_rank0(gm)

    # _dbg(
    #     "two_level_ilp timing (s): mem_est=%.3f runtime_est=%.3f get_fixed_bytes=%.3f "
    #     "outer_ilp=%.3f inner_ilp=%.3f | TOTAL=%.3f",
    #     _timings["mem_estimation"],
    #     _timings["runtime_estimation"],
    #     _timings["get_fixed_bytes"],
    #     _timings["outer_ilp"],
    #     _timings["inner_ilp"],
    #     _timings["total"],
    # )

    # --- outer-requested fractions, byte-weighted over each layer's owned
    # activation (act_bytes_per_layer = get_fixed_bytes_tuple[4]) ---
    act_by_layer = get_fixed_bytes_tuple[4]
    _zero_fr = {"keep": 0.0, "recompute": 0.0, "offload": 0.0}
    if _fr:
        _tot = sum(act_by_layer.get(L, 0) for L in _fr) or 1
        outer_fractions = {
            "keep": 100 * sum(_fr[L][0] * act_by_layer.get(L, 0) for L in _fr) / _tot,
            "recompute": 100
            * sum(_fr[L][1] * act_by_layer.get(L, 0) for L in _fr)
            / _tot,
            "offload": 100
            * sum(_fr[L][2] * act_by_layer.get(L, 0) for L in _fr)
            / _tot,
        }
    else:
        outer_fractions = dict(_zero_fr)
    inner_fractions = _ifr or dict(_zero_fr)

    metrics = {
        "num_layers": len(get_fixed_bytes_tuple[6]),  # candidates_by_layer
        "budget_gb": memory_budget / (1 << 30),
        "each_layer_separately": each_layer_separately,
        "runtime_mode": runtime_estimation_mode,
        # stage times (s)
        "t_mem_est": _timings["mem_estimation"],
        "t_runtime_est": _timings["runtime_estimation"],
        "t_get_fixed_bytes": _timings["get_fixed_bytes"],
        # "t_outer_total": _timings["outer_ilp"],
        "t_outer_construct": _t_construct_ilp,
        "t_outer_solve": _t_solve,
        # "t_inner_total": _timings["inner_ilp"],
        "t_inner_build": _bb,
        "t_inner_solve": _bs,
        # "t_total": _timings["total"],
        # fractions (byte-weighted %): outer requested vs inner achieved
        "outer_keep": outer_fractions["keep"],
        "outer_recompute": outer_fractions["recompute"],
        "outer_offload": outer_fractions["offload"],
        "inner_keep": inner_fractions["keep"],
        "inner_recompute": inner_fractions["recompute"],
        "inner_offload": inner_fractions["offload"],
    }
    return gm, metrics


def loss_region_peak(gm: torch.fx.GraphModule | torch.fx.Graph) -> int:
    """Peak bytes of storages ALLOCATED BY the loss/lm_head region.

    Ownership-based, not window-based: a storage counts only if its first
    appearance in the graph is at a loss/lm_head node, so activations and
    parameters belonging to layers 0..N never enter the total. Aliases are
    free -- a node producing only views has no fresh allocations.
    """
    import torch.utils._pytree as pytree
    from torch._inductor.fx_passes.memory_estimator import (
        _is_releasable,
        GraphAliasTracker,
        StorageKey,
    )
    from torch.fx.experimental.symbolic_shapes import optimization_hint

    def _nbytes(sk: StorageKey) -> int:
        # optimization_hint resolves SymInt byte counts to a concrete hint.
        return int(optimization_hint(sk.storage.nbytes()))

    graph = gm.graph if isinstance(gm, torch.fx.GraphModule) else gm
    nodes = list(graph.nodes)
    index = {n: i for i, n in enumerate(nodes)}
    tracker = GraphAliasTracker(nodes)

    def _in_loss(n: torch.fx.Node) -> bool:
        if _get_layer_id(n) != _NOT_IN_LAYERS:
            return False
        fqn = _get_module_fqn(n)
        return "loss" in fqn or "lm_head" in fqn

    # (alloc_idx, death_idx, nbytes, allocating node)
    intervals: list[tuple[int, int, int, torch.fx.Node]] = []
    for node in nodes:
        if not _in_loss(node):
            continue
        for sk in tracker.get_fresh_allocations(node):
            last = tracker.storage_to_last_user.get(sk)
            death = index[last] if last is not None else len(nodes) - 1
            intervals.append((index[node], death, _nbytes(sk), node))

    delta: defaultdict[int, int] = defaultdict(int)
    for lo, hi, nb, _ in intervals:
        delta[lo] += nb
        delta[hi + 1] -= nb

    live = peak = peak_at = 0
    for i in sorted(delta):
        live += delta[i]
        if live > peak:
            peak, peak_at = live, i

    GiB = 1024**3

    logger.info(
        f"loss region peak {peak / GiB:.2f} GiB at "
        f"{nodes[min(peak_at, len(nodes) - 1)].name} "
        f"({len(intervals)} owned storages)"
    )
    for nb, name in sorted(
        ((nb, n.name) for lo, hi, nb, n in intervals if lo <= peak_at <= hi),
        reverse=True,
    )[:12]:
        logger.info(f"    {nb / GiB:7.3f} GiB  {name}")
    return peak


# ---------------------------------------------------------------------------
# outer ilp over all layers + inner ilp per layer
# ---------------------------------------------------------------------------
def plan_and_tag_greedy(
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
    prefetch_lookahead: int = 1,
    defer_n_layers: int = 1,
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
    eff_budget = memory_budget - opt_bytes
    _dbg(f"eff_budget: {eff_budget}")
    # --- ablation knobs, all defaulting to the current model ---

    # Offload eviction lag, disabled by default. Only used for debugging
    # might delete later
    _abl_evict_mult = 0.0  # offload eviction lag mult
    _abl_off_excl = False  # storage rec OR off, not both

    from torchtitan.experiments.graph_trainer.memory_estimator import per_node_memory

    requested_budget = memory_budget  # - opt_bytes

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
        fixed_max_fwd,
        fixed_max_bwd,
        accumulative_act_fwd,
        accumulative_act_bwd,
        grad_size_per_layer,
        bwd_temp_max_per_layer,
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
    M = MEM_MULTIPLIER

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
    bw_d2h_g, bw_h2d_g = bw_d2h / M, bw_h2d / M  # GiB/ms
    _dbg(
        "C3 INPUTS: bw_d2h=%.1f GB/s bw_h2d=%.1f GB/s (isolated benchmark)",
        bw["d2h"],
        bw["h2d"],
    )

    offload_budget_per_layer = defaultdict(float)
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
        offload_budget_per_layer[block_id] = bw_d2h_g * M * fwd_rt_by_block[block_id]

    logger.info(f"fwd_rt_by_block: {fwd_rt_by_block}")
    logger.info(f"bwd_rt_by_block: {bwd_rt_by_block}")
    logger.info(f"bw_d2h_g: {bw_d2h_g}")
    logger.info(f"bw_h2d_g: {bw_h2d_g}")

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

    layer_ids = sorted(blocks)  # int layer ids in forward order: 0,1,...,L-1
    B_g = eff_budget / M
    Ppeak_g = mem_est.peak_bytes / M
    aG = {b: act_bytes_per_layer[b] / M for b in layer_ids}  # owned act, GiB

    live_bytes = mem_est.live_bytes
    logger.info(f"mem_est: {mem_est}")

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
    per_layer_must_keep = {}
    seen_sids = set()
    for node in must_keep:
        layer = _get_layer_id(node)
        if layer not in per_layer_must_keep.keys():
            per_layer_must_keep[layer] = 0

        node_bytes = 0
        for t in pytree.tree_leaves(node.meta.get("val")):
            if not isinstance(t, torch.Tensor) or t.device.type != "cuda":
                continue
            sid = t.untyped_storage()._cdata
            if sid in seen_sids:  # ?: remove_duplicate=False
                continue
            seen_sids.add(sid)
            node_bytes += get_size(t)

        per_layer_must_keep[layer] += node_bytes

    _t_construct_ilp_start = time.perf_counter()

    # parameters
    placeholders = [n for n in nodes if n.op == "placeholder"]
    persistent_state = set(placeholders[: trace.num_static_inputs])
    _local_ranks = max(1, int(os.environ.get("LOCAL_WORLD_SIZE", "1")))
    seen_sids = set()
    param_bytes = 0
    for n in persistent_state:
        for t in pytree.tree_leaves(n.meta.get("val")):
            if not isinstance(t, torch.Tensor) or t.device.type != "cuda":
                continue
            sid = t.untyped_storage()._cdata
            if sid in seen_sids:  # ?: remove_duplicate=False
                continue
            seen_sids.add(sid)
            param_bytes += get_size(t)

    param_bytes_per_layer = param_bytes / len(layer_ids)

    logger.info(f"param_bytes: {param_bytes}")

    # greedy:
    wait_candidates = (
        []
    )  # store only waits: they have very low runtime/GiB ratio, so most of them are chosen to be kept -
    # ideally we want them to be recomputed
    # as they are async ops
    all_candidates_by_size = {}  # node: size
    all_candidates = []
    all_act_size = 0
    decisions_per_act = defaultdict(
        int
    )  # 0: no decision, 1: save, 2: offload, 3: recompute
    for layer_id, candidates in candidates_by_layer.items():
        if layer_id != -1:
            for node, size in candidates:
                # if _is_collective_or_wait(node):
                #     wait_candidates.append(node)
                #     all_act_size += size
                #     all_candidates_by_size[node] = size
                # else:
                all_candidates_by_size[node] = size
                all_candidates.append(node)
                all_act_size += size
                if node in must_keep:
                    decisions_per_act[node] = 1

    logger.info(f"wait_candidates: {wait_candidates}")
    logger.info(f"all_candidates: {all_candidates}")
    # sort in descending order for runtime_per_node[node]/get_size_node(node)
    ranked_activations = sorted(
        (n for n in all_candidates if all_candidates_by_size[n] > 0),
        key=lambda n: (
            -runtime_per_node.get(n.name, 0.0) / (all_candidates_by_size[n] / M),
            n.name,
        ),
    )

    for node in ranked_activations:
        size = all_candidates_by_size[node] / M
        ratio = runtime_per_node.get(node.name, 0.0) / size
        decisions_per_act[node] = 0
        logger.info(
            f"node: {node}, runtime_per_node.get(node.name, 0.0): {runtime_per_node.get(node.name, 0.0)} size: {size:.2f} runtime_per_node[node]/get_size_node(node): {ratio:.2f}"
        )

    logger.info(f"ranked_activations: {ranked_activations}")

    params = (
        len(layer_ids) * param_bytes_per_layer
        + (_local_ranks - 1) * param_bytes_per_layer
    )

    loss_peak = act_bytes_per_layer[-1] + params + all_act_size
    logger.info(f"loss_peak: {loss_peak / (1 << 30):.2f}")
    logger.info(f"params: {params / (1 << 30):.2f}")
    logger.info(f"all_act_size: {all_act_size / (1 << 30):.2f}")
    logger.info(
        f"act_bytes_per_layer.get(-1, 0): {act_bytes_per_layer.get(-1, 0) / (1 << 30):.2f}"
    )
    logger.info(f"offload_budget_per_layer: {offload_budget_per_layer}")

    total_allowed_offload_budget = (
        min(bw_d2h_g * total_fwd_time, bw_h2d_g * total_bwd_time, host_cap_gib) * M
    )
    logger.info(f"total_allowed_offload_budget: {total_allowed_offload_budget}")

    # # first handle the waits as they are free to recompute
    # index = 0
    # logger.info(f"len(wait_candidates): {len(wait_candidates)}")
    # while index < len(wait_candidates) and loss_peak > eff_budget:
    #     act = wait_candidates[index]
    #     layer_id = _get_layer_id(act)
    #     size_act = all_candidates_by_size[act]
    #     logger.info(
    #         f"act: {act.name}, layer_id: {layer_id}, decisions_per_act[act]: {decisions_per_act[act]}"
    #     )
    #     if decisions_per_act[act] == 0:
    #         decisions_per_act[act] = 3
    #         loss_peak -= size_act
    #     index += 1

    offloaded_act = 0
    index = 0
    logger.info(f"len(ranked_activations): {len(ranked_activations)}")
    while (
        index < len(ranked_activations)
        and total_allowed_offload_budget > 0
        and loss_peak > eff_budget
    ):
        act = ranked_activations[index]
        layer_id = _get_layer_id(act)
        size_act = all_candidates_by_size[act]
        logger.info(
            f"act: {act.name}, layer_id: {layer_id}, _can_offload_node(act): {_can_offload_node(act)} offload_budget_per_layer[layer_id] - size_act: {offload_budget_per_layer[layer_id] - size_act} decisions_per_act[act]: {decisions_per_act[act]}"
        )
        if (
            layer_id < last_layer_id
            and _can_offload_node(act)
            and offload_budget_per_layer[layer_id] - size_act >= 0
            and decisions_per_act[act] == 0
        ):

            decisions_per_act[act] = 2
            loss_peak -= size_act
            total_allowed_offload_budget -= size_act
            offloaded_act += size_act
            offload_budget_per_layer[layer_id] -= size_act

        index += 1

    recomputed_act = 0
    index = len(ranked_activations) - 1
    while index >= 0 and loss_peak > eff_budget:
        act = ranked_activations[index]
        size_act = all_candidates_by_size[act]
        if decisions_per_act[act] == 0 and _is_recomputable(act, must_keep):
            decisions_per_act[act] = 3
            loss_peak -= size_act
            recomputed_act += size_act

        index -= 1

    logger.info(f"eff_budget: {eff_budget/M:.2f}")
    logger.info(f"loss_peak: {loss_peak/M:.2f}")
    logger.info(f"loss_peak from different function: {loss_region_peak(gm)/M:.2f}")

    if loss_peak > eff_budget + M:
        return None, None, None

    # logger.info(f"decisions_per_act: {decisions_per_act}")
    logger.info(f"offloaded activations: {offloaded_act/(1 << 30):.2f}")
    logger.info(f"recomputed activations: {recomputed_act/(1 << 30):.2f}")
    logger.info(f"offload_budget_per_layer: {offload_budget_per_layer}")
    logger.info(f"total_allowed_offload_budget: {total_allowed_offload_budget/M:.2f}")
    logger.info(f"loss_peak: {loss_peak/M:.2f}")

    logger.info(
        f"******************************************************************************"
    )
    for layer_id, candidates in candidates_by_layer.items():
        if layer_id != -1:
            logger.info(
                f"==================================================================================="
            )
            logger.info(
                f"layer_id: {layer_id}, offload_budget_per_layer: {offload_budget_per_layer[layer_id]/M:.2f}"
            )
            logger.info(
                f"==================================================================================="
            )
            for act, size_act in candidates:
                decision = ""
                if decisions_per_act[act] == 0 or decisions_per_act[act] == 1:
                    decision = "keep"
                elif decisions_per_act[act] == 2:
                    decision = "offload"
                else:
                    decision = "recompute"

                size = all_candidates_by_size[act] / M
                ratio = runtime_per_node.get(act.name, 0.0) / size

                logger.info(
                    f"act: {act}, can offload: {_can_offload_node(act)}, runtime/size: {ratio:.2f}, size: {size:.2f}, runtime:{runtime_per_node.get(act.name, 0.0):.2f} decision: {decision}"
                )
    logger.info(
        f"******************************************************************************"
    )

    def _tag_of(decision):
        # 0: no decision, 1: save, 2: offload, 3: recompute
        tag = (
            "keep"
            if decision == 0 or decision == 1
            else "offload" if decision == 2 else "recompute"
        )
        return tag

    per_node_decision = {}
    for layer_id, candidates in candidates_by_layer.items():
        for node, size in candidates:
            tag = _tag_of(decisions_per_act[node])
            per_node_decision[node] = tag

    chain_owner = {}
    for node in per_node_decision:
        if node.target is not torch.ops._c10d_functional.wait_tensor.default:
            continue

        stack, seen = [node], set()
        while stack:
            x = stack.pop()
            for up in x.all_input_nodes:

                if (
                    up in seen
                    or up.op != "call_function"
                    or _is_backward_node(up)
                    or up in per_node_decision
                ):
                    continue

                if not (
                    _is_collective_or_wait(up)
                    or _is_view(up)
                    or up.target is torch.ops.bucketing._pre_bucket_all_gather.default
                    or up.target is torch.ops.aten._to_copy.default
                ):
                    continue
                seen.add(up)
                chain_owner[up] = node
                stack.append(up)

    for node in gm.graph.nodes:
        if node.op != "call_function" or _is_backward_node(node):
            continue
        fqn = node.meta.get("custom", {}).get(_MODULE_FQN, "")
        if fqn.startswith(("lm_head", "loss")):
            continue
        if _is_rng_op(node) or isinstance(node.meta.get("val"), torch.SymInt):
            node.meta["recompute"] = CheckpointPolicy.MUST_SAVE
            continue

        if node in per_node_decision:
            tag = per_node_decision[node]
            node.meta["recompute"] = _POLICY_TAG[tag]
            continue

        upstream_collective = chain_owner.get(node, None)
        if upstream_collective is not None and upstream_collective in per_node_decision:
            tag = per_node_decision[upstream_collective]
            node.meta["recompute"] = _POLICY_TAG[tag]
            continue

        # views, getitems and interior plumbing: always recomputable. Rebuilding a
        # view is free, and the walk stops at the base, which carries a real tag.
        node.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE

    _t_construct_ilp_end = time.perf_counter()
    _dbg(
        "outer_ilp construction: CBC solve took %.3f s",
        _t_construct_ilp_end - _t_construct_ilp_start,
    )
    _t_construct_ilp = _t_construct_ilp_end - _t_construct_ilp_start

    _t_solve_start = time.perf_counter()
    _t_solve_end = time.perf_counter()
    _dbg("greedy: fake CBC solve took %.3f s", _t_solve_end - _t_solve_start)
    _t_solve = _t_solve_end - _t_solve_start

    # The outer LP only produces fractions; the inner ILP does the tagging.
    return gm, _t_construct_ilp, _t_solve


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
            # elif is_pre_bucket_reduce_scatter(prod) or (
            #     _is_backward_node(prod) and feeds_grad_collective(prod)
            # ):
            #     category = GRAD
            elif _is_backward_node(prod):
                category = GRAD if death_of[key] >= end else BWD_TEMP
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

    # gradient size:
    grad_size_per_layer = defaultdict(int)
    nonlayer_grad_bytes = 0
    for object in storages:
        if object.category != GRAD:
            continue
        layer_prod = _get_layer_id(object.producer_node)
        grad_size_per_layer[layer_prod] += object.size

    logger.info(f"grad_size_per_layer: {grad_size_per_layer}")

    bwd_temp_add_at, bwd_temp_remove_at = defaultdict(list), defaultdict(list)
    for object in storages:
        if object.category != BWD_TEMP:
            continue
        bwd_temp_add_at[object.produced_index].append(object)
        bwd_temp_remove_at[object.death_index].append(object)

    cur = 0
    bwd_temp_argmax = {}
    bwd_temp_max_per_layer = defaultdict(int)
    for i, node in enumerate(nodes):
        if not _is_backward_node(node):
            continue
        for b in bwd_temp_add_at.get(i, ()):
            cur += b.size  # born at i -> live during i
        L = _get_layer_id(node)
        if _is_backward_node(node) and isinstance(L, int):  # and L >= 0:
            if cur > bwd_temp_max_per_layer[L]:
                bwd_temp_max_per_layer[L] = cur
                bwd_temp_argmax[L] = (i, node.name)
        for b in bwd_temp_remove_at.get(i, ()):
            cur -= b.size  # dies at i -> still live *during* i
    logger.info(f"bwd_temp_max_per_layer: {bwd_temp_max_per_layer}")

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

    # A layer's nodes appear twice in the schedule, once in forward and once in
    # backward, and _get_layer_id keys off module_fqn only, so it cannot tell
    # the two visits apart. Their baselines are very different: backward also
    # carries every gradient accumulated so far. Keep them in separate dicts so
    # a per-layer peak model can charge the right one on each side.
    fixed_bytes_max = 0
    fixed_bytes = 0
    fixed_bytes_max_by_layer = defaultdict(int)
    fixed_max_fwd = defaultdict(int)
    fixed_max_bwd = defaultdict(int)
    for t in range(num_indices):
        fixed_bytes += sum(obj.size for obj in list_of_fixed_tensors_add_at.get(t, ()))
        fixed_bytes_max = max(fixed_bytes_max, fixed_bytes)
        node = nodes[t]
        layer_id = block_of_node(node)
        fixed_bytes_max_by_layer[layer_id] = max(
            fixed_bytes, fixed_bytes_max_by_layer.get(layer_id, 0)
        )
        _by_dir = fixed_max_bwd if _is_backward_node(node) else fixed_max_fwd
        _by_dir[layer_id] = max(fixed_bytes, _by_dir.get(layer_id, 0))
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
            logger.info(
                f"block_of_node(so.producer_node): {block_of_node(so.producer_node)} for {act_bytes_per_layer[block_of_node(so.producer_node)] / MEM_MULTIPLIER} and prod node: {so.producer_node} _get_module_fqn: {_get_module_fqn(so.producer_node)}"
            )
            if block_of_node(so.producer_node) >= 0:
                act_bytes_per_layer[block_of_node(so.producer_node)] += so.size
            elif (
                block_of_node(so.producer_node) == -1
                and _get_module_fqn(so.producer_node) != ""
            ):
                act_bytes_per_layer[block_of_node(so.producer_node)] += so.size
                logger.info(
                    f"counting layer: {block_of_node(so.producer_node)} for {act_bytes_per_layer[block_of_node(so.producer_node)] / MEM_MULTIPLIER} and prod node: {so.producer_node} module fqn: {_get_module_fqn(so.producer_node)}"
                )

    # ================================================================================================================

    non_layer_act_add_at, non_layer_act_remove_at = defaultdict(list), defaultdict(list)
    for object in storages:
        if (
            # object.category == ACT and
            object.category in (ACT, TEMP, BWD_TEMP)
            and block_of_node(object.producer_node) == -1
            and _get_module_fqn(object.producer_node) != ""
        ):
            non_layer_act_add_at[object.produced_index].append(object)
            non_layer_act_remove_at[object.death_index].append(object)

    cur = 0
    non_layer_act_argmax = None
    non_layer_act_peak = 0
    for i, node in enumerate(nodes):
        for b in non_layer_act_add_at.get(i, ()):
            cur += b.size  # born at i -> live during i
        if cur > non_layer_act_peak:
            non_layer_act_peak = cur
            non_layer_act_argmax = (i, node.name)
        for b in non_layer_act_remove_at.get(i, ()):
            cur -= b.size  # dies at i -> still live *during* i
    logger.info(f"non_layer_act_peak: {non_layer_act_peak}")
    logger.info(f"non_layer_act_argmax: {non_layer_act_argmax}")

    # ================================================================================================================

    # accumulative activation memory for each layer, split by direction for the
    # same reason as the fixed pools above
    accumulative_act_bytes_per_layer = defaultdict(int)
    accumulative_act_fwd = defaultdict(int)
    accumulative_act_bwd = defaultdict(int)
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
        _acc_by_dir = (
            accumulative_act_bwd
            if _is_backward_node(nodes[t])
            else accumulative_act_fwd
        )
        _acc_by_dir[layer_id] = max(_acc_by_dir[layer_id], cum_act_mem)
        for obj in act_remove_at.get(t, ()):
            cum_act_mem -= obj.size
            live.pop(obj, None)
        for obj in act_remove_at_for_freeable.get(t, ()):
            live_freeable.pop(obj, None)

    act_bytes_per_layer[-1] = non_layer_act_peak
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
        fixed_max_fwd,  # non-act baseline at each layer's forward high-water
        fixed_max_bwd,  # same, at its backward high-water (grads included)
        accumulative_act_fwd,  # cumulative act at each layer's forward visit
        accumulative_act_bwd,  # same, at its backward visit
        grad_size_per_layer,
        bwd_temp_max_per_layer,
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

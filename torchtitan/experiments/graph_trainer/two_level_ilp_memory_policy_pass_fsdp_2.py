# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
"""Two-level per-tensor keep/recompute/offload ILP (work in progress).

Plan (conceptually similar to torch's sac_milp two-level decomposition,
extended with offload):

  Outer ILP -- budget allocation across transformer blocks. One variable per
    block (a GPU-keep budget), coupled by the global peak-memory constraint.
  Inner ILP -- per-block three-way keep/recompute/offload. Given its allocated
    budget, each block solves a small independent ILP.

This file currently implements the pieces the whole ILP builds on:
  * Step 1: outer ILP (plan_outer) groups graph nodes by transformer block (block_of_node /
    group_nodes_by_block) and makes per-layer k/r/o decisions.
  * Step 2: the inner ILP (plan_and_tag_inner) build per-layer ILP to make per-tensor decisions
    and tags the graph nodes.
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
)
from torchinsights.graph_estimation._fx_utils import (
    ACT,
    feeds_grad_collective,
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

# Set once per solve from --compile.debug_memory_policy_solver. The solver logs
# ~80 diagnostic lines per run (byte pools, per-layer fractions, calibration
# iterations); useful when tuning a budget, noise otherwise.
_debug_logging = False


def _dbg(msg, *args):
    """Solver diagnostic. Silent unless debug_memory_policy_solver is set."""
    if _debug_logging:
        logger.info(msg, *args)


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

# The matmul family is the one class of save_op safe to leave recomputable in
# the per-layer inner solver: benchmark mode times these kernels accurately, so
# the runtime objective can be trusted to keep/offload the costly ones; and they
# are deterministic, so recompute is numerically sound. Every other save_op is
# anchored under SAVE_OPS_MATMUL_RECOMPUTABLE because the estimator prices it
# unreliably and/or it is unsafe or wasteful to recompute:
#   - attention (SDPA family, flex_attention HOP): benchmark can't measure a
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


# memory scaling factor: gb
MEM_MULTIPLIER = 1 << 30

# estimator reads low by up to ~1.3 GiB (from some measurements)
CALIBRATION_SAFETY_GB = 1.5

# Offload only tensors at least this large: fewer, larger transfers hold less
# per-tensor overhead at the backward peak (matches the sac_and_offload default).
OFFLOAD_MIN_BYTES = 1 << 20  # 1 MiB

# The modeled peak is a proxy, so after planning we materialize the plan and
# measure the real peak, then correct the cap and repeat. Converges in 2-3
# iterations on a reachable budget; the cap only runs out near the floor.
CALIBRATION_MAX_ITERS = 12
CALIBRATION_TOL_GB = 1.5  # stop once the measured peak is this close under budget

# Offloaded activations are pinned on the host and every local rank pins its
# own set, so the node's free memory is shared across them.
HOST_MEMORY_FRACTION = 0.8

# Weight on the offload term in the outer objective. 0.0 prices offload as free,
# which over-offloads (~15%, ~347 TFLOPs vs 371 achievable) but always meets the
# budget. Pricing at the full transfer time is worse: the LP then refuses
# offload entirely, and offload is the only lever that frees activations which
# cannot be recomputed. The real fix is the bandwidth input, not this weight.
OFFLOAD_TIME_PRICE = 0.0

# Which save_ops are barred from recompute. SAVE_OPS_NONE bars none; the
# alternative anchors attention HOPs, which measured ~1.4 GiB of extra floor and
# ~4% less throughput on qwen3-14B with no offsetting gain.
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
    """For now, this is just a wrapper for _get_layer_id:
    layers.<N> -> N: int
    """
    return _get_layer_id(node)


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


def _sync_plan_from_rank0(gm: torch.fx.GraphModule) -> None:
    """Force every rank to use rank 0's memory policy.

    Every rank runs this solver independently on its own copy of the graph and
    nothing reconciles the results. The solve is not bit-reproducible across
    processes -- several inputs are accumulated by iterating containers keyed by
    objects whose hash is id-based, and float addition is not associative, so
    ranks build slightly different LP coefficients. This LP is highly degenerate
    (every layer lands on the same k/r/o), so a low-bit coefficient difference
    flips the solver to a different optimum. The ranks then tag different nodes,
    bucket different all-gathers, and deadlock in NCCL -- surfacing five minutes
    later as a watchdog timeout on an all-gather with nothing pointing back here.

    Rather than chase every nondeterminism source, broadcast the decision. This
    is exact under SPMD: the graph is structurally identical on every rank (the
    name-set check below enforces that), only the solver's choice varies.
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
        # Not a solver difference -- the graphs themselves differ. Broadcasting
        # cannot fix that, so fail rather than tag a graph we did not solve for.
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
    """Whether the remat pass may erase and replay ``node``.

    Mirrors what the tagging path actually allows: ``must_keep`` (RNG, anchored
    save_ops, layer boundaries) is barred by the inner ILP, and collectives are
    barred by ``_demote_collective_recompute_tags`` -- a MUST_RECOMPUTE tag on a
    collective is silently rewritten to MUST_SAVE after the solve. A storage that
    is neither recomputable nor offloadable is resident under every plan, so both
    solvers must treat its bytes as fixed rather than freeable.
    """
    return node not in must_keep  # and not _is_collective_or_wait(node)


def _offload_forbidden(node: torch.fx.Node, size: int, must_keep: set) -> bool:
    """Whether the ILP must not tag ``node`` for CPU offload.

    Beyond the concrete pass's own limits (views, collectives/waits,
    non-contiguous and sub-OFFLOAD_MIN_BYTES tensors, all covered by
    ``_can_offload_node``), this bars ``must_keep`` -- RNG, anchored save_ops
    and layer boundaries.

    Offloading a layer boundary is actively harmful. It feeds the whole next
    layer, and ``o[u] + r[v] <= 1`` (a recomputed node may not read an
    offloaded input) then forbids recompute for every candidate that reads it.
    Measured on qwen3-14B, 4 GPUs, bs16, budget 67 GiB: offloading the 40
    boundaries (12.5 GiB) dropped recompute 95.7% -> 93.6% and raised peak
    Activation 16.54 -> 23.70 GiB, moving the achievable peak 66.92 -> 71.34
    and making the budget unreachable. Keeping them instead spends 12.5 GiB of
    keep and fits, at equal or better throughput (4080 vs 4039 tps for full
    recompute). The bandwidth knob does not control this: at cpu_offload_bw=0
    the C3 window is zero yet all 40 were still offloaded, because
    ``r == 0`` is hard for them and the keep budget could not hold them.

    Both solvers must agree on this predicate: the outer counts a storage as
    unfreeable only when it is neither recomputable nor offloadable, so if the
    outer still believed these were offloadable it would plan an offload
    fraction the inner cannot build.
    """
    if size < OFFLOAD_MIN_BYTES or not _can_offload_node(node):
        return True
    return node in must_keep


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
    """keep/recompute/offload bucket for a node's final ``meta["recompute"]``
    tag. MUST_SAVE / PREFER_SAVE / untagged all mean the storage stays resident
    on the GPU (keep)."""
    pol = node.meta.get("recompute")
    if pol in (CheckpointPolicy.MUST_CPU_OFFLOAD, CheckpointPolicy.PREFER_CPU_OFFLOAD):
        return "offload"
    if pol in (CheckpointPolicy.MUST_RECOMPUTE, CheckpointPolicy.PREFER_RECOMPUTE):
        return "recompute"
    return "keep"


def _demote_collective_recompute_tags(gm: torch.fx.GraphModule) -> int:
    """Ensure no collective (or its wait) is tagged MUST_RECOMPUTE.

    ``selective_activation_remat_pass`` duplicates every MUST_RECOMPUTE node.
    Duplicating a ``wait_tensor`` leaves its ``all_gather_into_tensor`` with two
    users, and the downstream bucketing pass requires exactly one
    (``process_collective_bucket``: ``assert len(n.users) == 1``), so the run
    dies with "Expected single user for all_gather_into_tensor_N, got
    {wait_tensor_N, wait_tensor_N_recomputed}". Recomputing a collective is also
    wrong on its own terms: it would re-issue the communication.

    A recompute tag can reach a collective by more than one route -- the
    getitem/wait parent-propagation in the base tagging bypasses
    ``_decision_space``, and a solver override can flip a tag afterwards -- so
    the invariant is enforced once here, after all tagging, rather than at each
    site that writes a tag. Demoting to MUST_SAVE (not offload) is the safe
    direction: the offload pass applies its own eligibility test.

    Returns the number of tags demoted.
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
    """Measure the keep/recompute/offload split actually written to the graph,
    by reading each candidate producer's ``meta["recompute"]`` tag rather than
    trusting the solver's variable values.

    Called after the ILP overrides and the default-SAC base tagging, but before
    apply_cpu_offload_pass / selective_activation_remat_pass materialize -- so it
    reflects the final decision that those passes will act on for every
    candidate, INCLUDING layers the ILP skipped (which keep the default-SAC base
    tag) and any base-tag interaction the per-layer solver totals miss.

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


# overall ilp - outer + inner ILPs
_TAG_KEY = "recompute"


def _snapshot_tags(gm: torch.fx.GraphModule) -> dict:
    """Current ``meta["recompute"]`` for every node, so a plan can be undone."""
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
    """Peak of the graph the run will actually execute, for the current tags.

    The outer LP scores plans with a proxy (original-graph live_bytes minus a
    credit for what recompute/offload should free, plus a working-set term).
    That proxy is what misses the budget. This instead materializes the plan --
    the same two passes the real pipeline applies -- on a throwaway clone and
    asks the peak estimator, which was measured at 0.06% against real CUDA
    (45.49 estimated vs 45.52 real). The clone shares the parent's parameters;
    only the graph is copied, so this costs graph-copy time, not weight memory.

    Returns (peak_bytes_including_optimizer, MemoryEstimatorResult) or
    (None, None) if the probe could not be built.
    """
    import copy as _copy

    from torchtitan.experiments.graph_trainer.cpu_offload import apply_cpu_offload_pass
    from torchtitan.experiments.graph_trainer.selective_activation_remat import (
        selective_activation_remat_pass,
    )

    try:
        probe = torch.fx.GraphModule(gm, _copy.deepcopy(gm.graph))
    except Exception as e:  # noqa: BLE001 - fall back to the uncalibrated path
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
    # Drop the clone before returning. Each probe retains graph state that keeps
    # device memory alive, and the loop builds one per iteration: at 8
    # iterations planning fit, at 12 it reached 83.57 GiB and OOMed before step
    # 0 ("Tried to allocate 640.00 MiB ... 537.94 MiB is free").
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


def _concurrent_transfer_bw() -> dict:
    """Per-rank D2H/H2D bandwidth with every rank transferring at once.

    Each rank times its OWN device while all ranks copy, so the number reflects
    the contention offload actually meets. Three details matter and each was
    established by measurement:

    CUDA EVENTS, NOT WALL CLOCK. Host timing folds in launch and synchronize
    overhead: on an 8-GPU H100 host it read 20.6 GB/s where the copy engines
    were doing 29.5, a systematic 30% underread with a deceptively tight spread.

    RAMP EXCLUDED. Ranks enter and leave the loop at slightly different times,
    and the ends of the window are therefore less contended than the middle. A
    quarter of the iterations run untimed on each side so only the steady state
    is measured. Without this a short burst reads 30-50% high because the ranks
    stop overlapping.

    NO ANALYTIC CONTENTION FACTOR. This used to measure one idle GPU and divide
    by sqrt(local_world_size). That factor was fitted to a scaling curve from an
    earlier, unreliable probe, and it does not transfer: on an 8xH100 host it is
    37% low against profiler traces, and on an 8xB200 host 65% low, because that
    machine shows no D2H contention at all (56.2 GB/s per GPU at 8 GPUs vs 55.6
    at one, 450 GB/s aggregate). Contention is a property of the machine, so it
    has to be measured on the machine.

    Cross-checked against ground truth: profiler traces of a real 8-rank
    qwen3-32B step with offload active give 27.5-31.5 GB/s per rank (18.50 GB
    moved, 587-720 ms of copy-engine time, ~10% duty, mean cross-rank
    concurrency 4.23).

    Collectives inside: call once, outside the calibration loop.
    """
    if not torch.cuda.is_available():
        return {}
    live = dist.is_available() and dist.is_initialized()
    dev = torch.cuda.current_device()

    nbytes = 256 << 20
    iters, ramp, repeats = 20, 5, 3
    src = torch.empty(nbytes, dtype=torch.uint8, device=f"cuda:{dev}")
    dst = torch.empty(nbytes, dtype=torch.uint8, device="cpu", pin_memory=True)
    dst.zero_()  # first-touch the pages before they are timed
    stream = torch.cuda.Stream(device=dev)

    out = {}
    for name, to_host in (("d2h", True), ("h2d", False)):
        samples = []
        for _ in range(repeats):
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            with torch.cuda.stream(stream):
                for _ in range(2):  # warm the path
                    (dst if to_host else src).copy_(
                        src if to_host else dst, non_blocking=True
                    )
            torch.cuda.synchronize()
            if live:
                dist.barrier()  # line the ranks up before the ramp
            with torch.cuda.stream(stream):
                for _ in range(ramp):
                    (dst if to_host else src).copy_(
                        src if to_host else dst, non_blocking=True
                    )
                e0.record(stream)
                for _ in range(iters):
                    (dst if to_host else src).copy_(
                        src if to_host else dst, non_blocking=True
                    )
                e1.record(stream)
                for _ in range(ramp):
                    (dst if to_host else src).copy_(
                        src if to_host else dst, non_blocking=True
                    )
            torch.cuda.synchronize()
            samples.append(nbytes * iters / (e0.elapsed_time(e1) * 1e-3) / 1e9)
        out[name] = statistics.median(samples)
    del src, dst
    torch.cuda.empty_cache()

    if live:
        # Plan against the slowest rank: a rank that cannot move its bytes in
        # time is the one that stalls the step.
        t = torch.tensor(
            [out["d2h"], out["h2d"]], dtype=torch.float64, device=f"cuda:{dev}"
        )
        dist.all_reduce(t, op=dist.ReduceOp.MIN)
        out["d2h"], out["h2d"] = float(t[0]), float(t[1])
    _dbg(
        "C3 INPUTS: contended per-rank d2h=%.1f h2d=%.1f GB/s "
        "(event-timed, ramp excluded, min across ranks)",
        out["d2h"],
        out["h2d"],
    )
    return out


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
    """Two-level keep/recompute/offload solver. Tags ``gm`` with per-node decisions.
    Returns ``gm`` if the solver succeeded, or None if it failed and metrics."""
    global _debug_logging
    _debug_logging = debug_logging

    _t0 = time.perf_counter()
    _timings = {}
    gm = trace.gm

    _t = time.perf_counter()
    mem_est = estimate_peak_memory(gm, num_state_inputs=trace.num_static_inputs)
    opt_bytes = optimizer_state_bytes(optimizer, model_parts[0])
    _timings["mem_estimation"] = time.perf_counter() - _t
    estimated = mem_est.peak_bytes + opt_bytes  # all-keep peak (incl. optimizer state)

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

    # Pre-pass (all-keep) breakdown. The trainer logs the same categories for
    # the POST-pass graph, so diffing the two shows which category the passes
    # actually moved and which one the LP failed to anticipate.
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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        _mem_probe("after empty_cache")

    _t = time.perf_counter()
    get_fixed_bytes_tuple = get_fixed_bytes(gm, trace.num_static_inputs)
    _timings["get_fixed_bytes"] = time.perf_counter() - _t

    # --- outer plan, calibrated against the materialized graph ---------------
    # Solving the outer LP once against (budget - opt_bytes) does not land on
    # the budget. Its peak model is a proxy for the post-pass graph and the
    # error depends on the plan itself, and because the LP drives its own slack
    # to exactly 0 that error passes straight through to the realized peak.
    #
    # Search the effective cap instead of trusting it. peak(cap) rises with cap
    # but does so in jumps (the inner tagging is discrete and the binding
    # position moves), so a secant correction oscillates -- measured cap 34.25
    # -> 45.49 GiB and cap 36.76 -> 51.47 GiB, a 6 GiB step for 2.5 GiB of cap.
    # Bracket instead, then bisect: keep the largest cap whose MATERIALIZED peak
    # still fits, which is the plan that keeps the most and recomputes the least.
    _cal_tol = CALIBRATION_TOL_GB * MEM_MULTIPLIER
    # Already resolved and validated by resolve_host_offload_cap_gib().
    _host_cap_gib = float(cpu_offload_budget_gb)
    # Transfer rate for the C3 offload window: the idle single-GPU rate divided
    # by log2 of the ranks sharing this node's host path.
    #
    # This is a derate, not a contention model. Measuring the true contended
    # rate and using it makes the solver offload roughly twice as much and LOSE
    # throughput: llama3-8B -6.3% at 39.2 GB/s (its real rate is 39.3) and
    # qwen3-14B -11% at 44.9. Every plan that beat full recompute came from a
    # bandwidth well below the truth -- 26.4 at 4 ranks (+7.2%) and 18.0 at 8
    # (+8.9%). log2 reproduces exactly those: 26.1 and 17.4.
    #
    # The gap is duty. C3 assumes the copy engine can run flat out for the whole
    # forward window, but the rate falls as it fills: measured here, 45.5 GB/s at
    # 10% duty down to 19.8 at 100%. Plans built on the idle rate run at ~29%
    # duty and get 28.3, leaving the GPU idle 21% of the step.
    #
    # Ranks on OTHER nodes do not share this path, so this uses the local count.
    # Caveat: on hardware with no D2H contention (8xB200 measured 56.2 GB/s per
    # GPU at 8 GPUs vs 55.6 at one) this is ~67% low and will starve offload.
    # Only local rank 0 measures. The others wait in the all_reduce below, and
    # that is what keeps the reading uncontended: when every rank benchmarked
    # at once they overlapped each other and the draw swung 34.9-52.0 GB/s
    # depending on how much they happened to collide.
    if cpu_offload_bw == 10000:
        _lrank = int(os.environ.get("LOCAL_RANK", "0"))
        _solo_bw = get_transfer_bw() if _lrank == 0 else {"d2h": 0.0, "h2d": 0.0}

        if dist.is_available() and dist.is_initialized():
            # MAX, not MIN: the non-measuring ranks carry 0.0, so MAX selects the
            # measured value. MIN would hand every rank a bandwidth of zero.
            _t = torch.tensor(
                [_solo_bw["d2h"], _solo_bw["h2d"]],
                dtype=torch.float64,
                device=f"cuda:{torch.cuda.current_device()}",
            )
            dist.all_reduce(_t, op=dist.ReduceOp.MAX)
            _solo_bw = {"d2h": float(_t[0]), "h2d": float(_t[1])}

        # Derate AFTER the reduce so every rank scales the same number.
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

    # Only rank 0 searches. Every rank running the search independently made
    # them converge on DIFFERENT caps -- their LP coefficients differ in the low
    # bits (documented in _sync_plan_from_rank0) and the search amplifies that
    # into structurally different plans, which _sync_plan_from_rank0 then
    # rejects: "graph structure differs across ranks: rank 5 has 5335 tagged
    # nodes vs rank 0's 4973". Rank 0 finds the cap, broadcasts it, and every
    # rank does ONE identical solve+tag at that cap; residual per-rank policy
    # differences are still reconciled by the broadcast at the end. This also
    # makes planning ~12x cheaper on the other ranks.
    _rank = (
        torch.distributed.get_rank()
        if torch.distributed.is_available() and torch.distributed.is_initialized()
        else 0
    )
    _cap = float(memory_budget - opt_bytes)
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

    for _it in range(CALIBRATION_MAX_ITERS if _rank == 0 else 0):
        _n_iter = _it + 1
        _restore_tags(gm, _base_tags)
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
            # Cap below the LP's own feasibility boundary. Remember it as a hard
            # lower bound and bisect UP toward the cheapest feasible cap rather
            # than stepping up blindly: stepping up by a fixed 5% and then back
            # down by the budget error made the search oscillate across the
            # boundary (37.66 -> over budget -> 34.35 infeasible -> 36.07
            # infeasible -> 37.87 -> over budget -> ...) and burn every
            # iteration without ever finding the tightest feasible plan.
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
                (_lo + _hi) / 2.0 if _hi is not None else _lo + (memory_budget - _peak)
            )
        else:
            _hi = _cap if _hi is None else min(_hi, _cap)
            if _lo is not None:
                _cap = (_lo + _hi) / 2.0
            elif _inf is not None:
                _cap = (_inf + _cap) / 2.0  # squeeze toward the feasible floor
            else:
                _cap = _cap - _err
        if _lo is not None and _hi is not None and (_hi - _lo) < 0.25 * MEM_MULTIPLIER:
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
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(_verdict, src=0)
    _won_cap, _won_peak, _floor, _n_iter = _verdict[0]
    if _won_cap is None:
        # Budget below what any split can reach. Do NOT fall back to a plan
        # that exceeds it: silently running over the requested budget is the
        # failure this solver exists to prevent, and the old behaviour --
        # returning an untagged graph -- just OOMs at the all-keep peak.
        # Refuse, and name the floor that was actually measured. The outer
        # LP's own "UNREACHABLE by N GiB" number comes from its proxy peak
        # model and is not trustworthy (it claimed >= 48.14 GiB for a case
        # where the materialized graph reached 44.00).
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
        "calibrate: chose cap %.2f GiB -> materialized peak %.2f GiB "
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

    if fractions is None:
        # The budget is unachievable. Returning the graph UNTAGGED here used to
        # be the behaviour, but an untagged graph runs at the full all-keep peak
        # and OOMs several minutes later with a traceback that points at a
        # matmul rather than at the budget. Say so instead.
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
    _dbg(f"memory_budget: {memory_budget}")
    _dbg(f"estimated: {estimated}")
    _dbg(f"opt_bytes: {opt_bytes}")
    # if memory_budget > estimated:
    #     _dbg(
    #         "new-autoAC: budget %.2f GB >= estimated peak %.2f GB; nothing to do",
    #         memory_budget / 1e9,
    #         estimated / 1e9,
    #     )
    #     return gm

    # The LP's peak model is a proxy for the post-pass graph, so solving once
    # against (budget - opt_bytes) lands off-target by an amount that depends
    # on the plan itself. The caller calibrates by re-solving with a corrected
    # cap; eff_budget_override carries that correction.
    eff_budget = (
        memory_budget - opt_bytes
        if eff_budget_override is None
        else eff_budget_override
    )

    # --- ABLATION KNOBS (env-driven auto-research; all default to the current model) ---

    # Offload eviction-lag multiplier. Default 0: the lag is DISABLED. Budget sweeps
    # across llama3_1b and qwen3_1_7b (floor .. floor+5 GB) showed a nonzero lag was
    # over-conservative -- it made qwen infeasible at its sac-achievable floor while
    # only ever hurting or not helping llama. With lag=0 both models fit every budget
    # from floor to floor+5 with no real-peak violations. Override via env for ablation.
    #
    # That sweep was on 1B-class models where the offloaded volume is small. On
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
    # The isolated benchmark is not reproducible: repeated runs on an idle
    # machine returned d2h = 35.5 / 49.5 / 53.0 GB/s, a 1.5x spread, while the
    # runtime inputs it is multiplied against are stable to 0.2%. Since d2h is
    # the binding side of C3, that noise alone moves the offload plan.
    # The startup benchmark is BOTH unreliable and optimistic, so it is not used
    # to size offload by default. Repeated identical runs returned d2h = 35.5 /
    # 37.1 / 49.5 / 53.0 GB/s (a 1.5x spread) while the runtimes it multiplies
    # are stable to 0.2%; and it measures an idle device -- a profiler trace of
    # a real step shows D2H aggregating 32.3 GB/s against ~52 GB/s benchmarked
    # at the same transfer size. Neither 8-way concurrency (-5%) nor compute
    # load (-12%) explains the gap; the offload path's destination buffers are
    # not consistently pre-pinned (pageable copies run at 13-15 GB/s).
    # Defaults are the measured achieved rates; override per machine.
    # With the pinned pool on, the offload path and this benchmark both reuse
    # pinned buffers, so the benchmark measures what offload will actually get
    # and no correction belongs here. Overridable per machine for ablation.
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

    # The C3 offload windows are bandwidth * these times. Print the totals so
    # they can be checked against a profiler trace's real forward/backward
    # split -- an inflated window is indistinguishable from optimistic
    # bandwidth from inside the LP.
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

    # Outer LP: per-layer keep/recompute/offload fractions (continuous).
    # Memory terms scaled to GiB (MEM_MULTIPLIER) so solver coefficients stay O(1-30).
    M = MEM_MULTIPLIER
    layer_ids = sorted(blocks)  # int layer ids in forward order: 0,1,...,L-1
    B_g = eff_budget / M
    Ppeak_g = mem_est.peak_bytes / M
    aG = {b: act_bytes_per_layer[b] / M for b in layer_ids}  # owned act, GiB
    bw_d2h_g, bw_h2d_g = bw_d2h / M, bw_h2d / M  # GiB/ms

    live_bytes = mem_est.live_bytes

    # Recompute working set: the layer's FORWARD temporaries. They are freed
    # inside forward in the original graph, so live_bytes never sees them at a
    # backward position, but re-running the layer's forward during backward
    # re-creates them. Backward-produced TEMP is excluded on purpose -- it is
    # already in live_bytes at exactly the backward positions where ws is added,
    # so counting it here would double-count it (it inflated this term ~4x).
    _layer_temp = defaultdict(float)
    for _prod, _ents in mem_est.all_tensors.items():
        _b = block_of_node(_prod)
        if isinstance(_b, int) and _b >= 0 and not _is_backward_node(_prod):
            for _e in _ents:
                if _e["category"] == TEMP:
                    _layer_temp[_b] += _e["size"]
    # Recompute working set per layer: the forward temporaries a layer must
    # hold live while replaying its own recompute.
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

    # Per-layer freeable ceiling. k+r+o==1 on its own lets the LP set k[b]=0 and
    # claim the layer is fully freed, but storages that are neither recomputable
    # nor offloadable are kept no matter what. r+o therefore cannot exceed the
    # freeable fraction. Without this the outer reports a total keep BELOW the
    # unfreeable floor and hands the inner a keep target it cannot satisfy --
    # the inner then overshoots keep, which reads as "keep is not respected".
    # "Freeable" must be the INNER's policy notion, not liveness. freeable_pi
    # asks whether a storage can be dropped at a position and says yes for 100%
    # of activation bytes. The inner additionally forces r=0 for non-recomputable
    # nodes (RNG, anchored save_ops, layer boundaries, collectives) and o=0 for
    # views / non-offloadable / sub-OFFLOAD_MIN_BYTES tensors; a node hit by both
    # is UNFREEABLE and must be kept. Mirror those exact predicates here so the
    # two levels agree on what a fraction can range over. Denominator is
    # layer_act, the same one the inner divides by.
    _unfree_bytes = defaultdict(float)
    for _b, _cands in candidates_by_layer.items():
        _li = int(_b.split(".")[1]) if isinstance(_b, str) else _b
        for _n, _sz in _cands:
            _no_r = not _is_recomputable(_n, must_keep)
            _no_o = _offload_forbidden(_n, _sz, must_keep)
            if _no_r and _no_o:
                _unfree_bytes[_li] += _sz
    # k+r+o==1 alone lets the LP zero out k[b] and claim the layer is fully
    # freed. Storages that are neither recomputable nor offloadable stay
    # resident regardless, so cap r+o by the share that can actually move.
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
        # Recompute working set, specific to position: at a backward position, only the
        # layer whose backward runs there is being recomputed, so only its forward
        # temp is transiently live. Add recompute_working_set[L_t]*r[L_t] for that layer (0 at
        # forward positions / non-layer positions). This counts the previously-
        # uncounted recompute memory without the global-max over-constraint.
        ws_here = ws if _is_backward_node(nodes[t]) else 0  # WS only during backward
        prob += lb / M - freed + ws_here <= B_g, f"peak_{t}"
        n_peak_cons += 1
    _dbg("outer: %d per-position peak constraints", n_peak_cons)

    # (C3) offload windows (Option 1: hidden within a next layer's compute window + some offload/reload is hidden
    # behind its own compute).
    # The window applies to EVERY layer including the last. Exempting the last
    # layer left o[last] unbounded, and since offload costs nothing in the
    # objective the LP drove it to 1.0 -- the whole layer offloaded at implicitly
    # infinite bandwidth. The last layer is in fact the *least* overlappable one
    # (produced at the end of forward, consumed at the start of backward), so an
    # exemption is backwards.
    # Copy-engine knee. Offload below the knee is close to free; past it, each
    # extra byte costs more throughput than recomputing the same byte would.
    # Measured at bs16/seq2048 on 8xH100, keep pinned at 5%, 60 steps:
    #   offload  0.05    0.10    0.15    0.20
    #   TFLOPs  364.08  370.90  359.20   (over-budget)
    #   peak     57.84   45.34   46.18
    # The optimum is 0.10 of layer activation = 0.728 GiB/layer, which against
    # the modeled window (32.3 GB/s * 40.98 ms = 1.233 GiB) is 0.59 of it. That
    # 0.59 folds two known model errors: the true duty at the optimum is ~0.32,
    # and fwd_rt_by_block understates the real per-block forward window by 1.82x
    # (40.98 ms modeled vs 74.78 ms measured from profiler traces).
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

    # 1e6 GiB^-1 dwarfs any runtime term (fwd_rt_by_block is ms per layer), so
    # the slack stays exactly 0 whenever the budget is reachable and the plan is
    # identical to the non-elastic formulation.
    _nl = max(layer_ids) or 1
    _tiebreak = (
        # Tiebreak: among equal-cost plans prefer keeping in later layers,
        # whose activations die sooner and so cost less peak.
        -1e-4
        * lpSum((b / _nl) * k[b] for b in layer_ids)
    )
    # Offload was priced at exactly zero here, so the LP always drove it to the
    # C3 cap: it is the one lever that both relieves the peak constraint and
    # costs nothing, which is why every solve came back sitting on the window
    # (measured: o = 10.34% against a 10.4% cap, 20.74% against 20.7%). Below
    # the knee offload really is nearly free, so it stays cheap there; above it
    # the measurements say a byte offloaded costs more than a byte recomputed,
    # so price it above recompute and the LP will recompute instead.
    # Price BOTH levers in milliseconds, from the same measured inputs, so they
    # compete on equal terms:
    #   recompute -> fwd_rt_by_block[b], the time to recompute the layer
    #   offload   -> aG[b] / bw_d2h_g,   the time to move the bytes over PCIe
    # Offload used to be priced at exactly zero, which made it the LP's first
    # choice for everything: it both relieves the peak constraint and costs
    # nothing, so every solve sat on whatever offload cap was binding, and
    # LOOSENING the budget made the plan slower because the LP kept trading
    # recompute away for more offload. Measured at bs16/seq2048, keep pinned at
    # 5%, medians over steps >= 30:
    #   offload  0.05    0.10    0.15    0.20
    #   TFLOPs  364.5   371.0   359.7   322.9
    # Offload never beat recompute at the margin, so no overlap discount is
    # applied to the transfer time -- an assumed discount is what produced the
    # over-offloading in the first place. With honest prices the LP reaches for
    # offload only where recompute cannot free the bytes, which is exactly the
    # job offload is uniquely good for: forcing offload to 0 leaves the
    # non-recomputable activations resident and the peak jumps to 59.52 GiB.
    # OFFLOAD_TIME_PRICE weights the offload term. 0.0 is the historical
    # "offload is free" objective, which is WRONG but safe: it
    # over-offloads (~15%, ~347 TFLOPs vs 371 achievable) yet always meets the
    # budget. Pricing at the full transfer time (1.0) is wrong in the other
    # direction and DANGEROUS: the LP then refuses offload, and offload is the
    # only lever that frees non-recomputable activations, so budget 50 came back
    # at a 59.06 GiB real peak with no feasible plan at any cap.
    #
    # The honest reason neither works: the C3 window is sized with an idle
    # copy_ benchmark (49.7 GB/s here) while a real step aggregates 22.8 GB/s of
    # D2H. At the achieved rate the window is 12% of layer activation, which is
    # essentially the measured optimum (10%); at the benchmarked rate it is 27%,
    # so the LP is handed roughly twice the offload headroom that exists. The
    # fix is to feed this the achieved bandwidth rather than to reprice
    # offload -- _concurrent_transfer_bw now supplies it.
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
        # The peak constraints are hard, so an unreachable budget lands here and
        # the caller gets an untagged graph -> OOM with no number to act on.
        # Re-solve the SAME constraints with one elastic slack, minimizing it,
        # to report the smallest budget that admits any split. Diagnostic only:
        # runs on the failure path, does not produce a plan.
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
                "B_g=%.2f GiB but the tightest plan needs %.2f GiB, i.e. "
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

    # What the LP believes the post-pass peak will be, so it can be compared
    # against the estimator's post-pass number. Every peak_ constraint is
    # (lb/M - freed + ws_here) - B_g <= 0, so constraint.value() is
    # predicted_peak - B_g and the binding position is the max. If this says
    # "fits" while the real run does not, the LP's peak MODEL is wrong -- not
    # the solver, not the inner.
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
    _dbg("inner ILP planner and tagger...")
    _t_start = time.perf_counter()  # whole plan_and_tag_inner wall time
    keep_fraction, recompute_fraction, offload_fraction = _validate_fractions(
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
    # measurement, not real.
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
        if _is_rng_op(
            node
        ):  # or _is_collective_or_wait(node):  # or other nondeterministic
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

            An FSDP unshard is ``_to_copy -> all_gather -> wait``. Only the wait's
            storage survives the fwd->bwd gap, so the wait is the ILP candidate --
            but the runtime cost and the remat legality both live on the collective.
            Tag propagation elsewhere is strictly downward (parent -> getitem/wait),
            so without this the collective keeps the pre-tag MUST_RECOMPUTE default
            while its wait is tagged KEEP: a cloned wait then wires to the ORIGINAL
            collective, leaving it two users and tripping the ``len(n.users) == 1``
            assert in bucketing.process_collective_bucket. Returns [] for
            non-collective candidates.
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
            # The bf16 cast feeding this collective, only when it feeds nothing
            # else -- a shared cast must not inherit one wait's decision.
            if (
                isinstance(src, torch.fx.Node)
                and src.op == "call_function"
                and len(src.users) == 1
            ):
                chain.append(src)
            return chain

        def _recompute_ms(node):
            """Recompute cost of a candidate, including its collective chain.

            ``runtime[wait_tensor]`` is a barrier (~0 ms), so charging the wait alone
            prices a re-issued NCCL all-gather at zero and the solver buys every
            collective recompute for free.
            """
            ms = runtime_per_node.get(node.name, 0.0)
            for up in _collective_chain(node):
                ms += runtime_per_node.get(up.name, 0.0)
            return ms

        def _tag_collective_chain(node, tag):
            # An offloaded wait must still be PRODUCED in forward before the D2H
            # copy, so its collective is kept rather than recomputed.
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
                    # RNG / anchored save_ops / layer boundaries, and collectives
                    # (a MUST_RECOMPUTE tag on a collective is rewritten to
                    # MUST_SAVE by _demote_collective_recompute_tags after the
                    # solve, so planning a recompute here just leaks into keep).
                    prob += r[node] == 0
                if _offload_forbidden(node, size, must_keep) or _is_collective_or_wait(
                    node
                ):
                    # Match the concrete offload pass: views, collectives/waits,
                    # non-contiguous tensors, and tiny tensors should not be tagged
                    # for CPU offload by the ILP.
                    prob += o[node] == 0
                g = size / M
                total_keep += k[node] * g
                total_off += o[node] * g
                total_rec += r[node] * g
                objective += r[node] * _recompute_ms(node)

            # A recomputed node must not read an offloaded input: its dup runs during
            # backward and would need the input on GPU, but an offloaded input is on
            # CPU (the remat/offload reload-hoisting mishandles this and faults at
            # runtime). Resolve view/alias chains to the underlying candidate.
            cand_set = set(k)  # membership only
            # Iterate the DICT, not the set. Node.__hash__ is id-based, so
            # set(k) iteration order differs between processes -- emitting these
            # constraints in a per-rank order makes CBC pick different optima of
            # this (highly degenerate) ILP on different ranks. The ranks then
            # tag different nodes, bucket different all-gathers, and deadlock in
            # NCCL. Dicts are insertion-ordered, so this is rank-invariant.

            # for v in k:
            #     for x in v.all_input_nodes:
            #         u = _resolve_cand(x, cand_set)
            #         if u is not None and u is not v:
            #             prob += o[u] + r[v] <= 1

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
            off_over = LpVariable(
                f"off_over_{kf}_{of}_{remaining_off_g}_{act}", lowBound=0
            )

            prob += total_keep <= target_keep_g + keep_over

            # Offload: minimize |total_off - target|, never require membership in
            # a window. The two-sided hard band it replaces was a feasibility
            # test on a binary knapsack whose window (+-1..2% of target, ~20-95
            # MB) is far narrower than the tensor granularity (~0.125-2.1 GiB),
            # AND it interacts badly with the 1e6-scaled objective: CBC reported
            # Infeasible for kf=0.05 of=0.17 on ranks 5/6/7 while another rank
            # solved the identical problem and ran. Same model, different answer
            # per rank and per run -- solver numerics, not achievability. With
            # two penalized slacks the all-recompute plan is always feasible, so
            # Infeasible cannot occur, and the optimum is the closest reachable
            # point from either side.
            prob += total_off <= remaining_off_g  # hard: real host limit only
            prob += off_short >= target_off_g - total_off
            prob += off_over >= total_off - target_off_g

            # Strict priority: (1) keep cap, (2) closest offload, (3) least
            # recompute time. keep_over must dominate: exceeding keep costs real
            # peak memory, while missing the offload target only means those
            # bytes are recomputed -- which frees them just as well and costs
            # only time. 1e3 still dwarfs the per-layer runtime (~30 ms) so
            # offload beats recompute whenever it is free. The tie-break rewards
            # keeping only WITHIN budget.
            prob += (
                # 1e6 : 1e5 means 1 GiB of offload error trades against 0.1 GiB
                # of keep error. At 1e6 : 1e3 the ratio was 1000:1, so the solver
                # accepted a 2.7-point offload overshoot (0.377 GiB/layer, cost
                # 377) rather than spend 0.4 MB of extra keep (cost 400) --
                # technically optimal, practically useless. Recompute is the free
                # residual, so tightening offload pushes the miss into recompute,
                # which is what we want.
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
                # Never read variable values from a failed solve. _tag_of assigns
                # exactly one tag per node regardless of whether the values mean
                # anything, so the result looks plausible (counts still sum) while
                # being only whatever presolve happened to fix -- typically "keep
                # the unfreeable pool, offload the off_only pool, recompute the
                # rest", which silently ignores both requested fractions.
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

            # The knapsack the solver faced: minimize recompute time subject
            # to an offload byte budget", i.e. fill the budget with the highest
            # ms/GiB tensors -- so if a high-density tensor is RECOMPUTED, the
            # only explanations are that o was forced to 0 for it, or the budget
            # was already spent on denser ones. This prints both facts.

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

        # Second pass: redistribute the unused keep allowance.
        #
        # Every layer in a group gets the same pattern, and a layer's keep
        # allowance almost never tiles exactly with the tensor sizes on offer
        # (activations come in ~0.875 GiB granules; measured fill is ~85%). Each
        # layer therefore leaves a residual unused, and 31 identical residuals
        # put the realized peak several GiB under the budget for no runtime
        # benefit. Pool the residuals and hand them to as many layers as they
        # cover, using a second solve per signature with a raised keep target.
        #
        # Boost the highest layer indices first: their backward runs earliest,
        # so keeping them costs the peak the least (same reasoning as the outer
        # tie-break). Total keep never exceeds what the outer requested, so the
        # outer's peak constraint still holds.
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
            # Find the SMALLEST raise that buys another tensor, so the residual
            # spreads over many layers instead of piling onto one. Start from an
            # even share and double until a candidate fits; anything larger just
            # concentrates the same bytes in fewer layers.
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
        # The grouped solve produces ONE pattern per signature and applies it to
        # every layer in that group, so total offload is quantized to
        # n_layers * (per-layer reachable sum). Per-layer sums are coarse (the
        # tensors are ~0.125-2.1 GiB), so a target that is not reachable within
        # a single layer is missed by n_layers * gap even though the TOTAL is
        # reachable by mixing two patterns. Measured: of=0.17 wants 2.371
        # GiB/layer, the best single pattern gives 2.312, and 40 * 0.059 = 2.36
        # GiB of offload is left on the table (16.6% instead of 17.0%).
        #
        # Fix: solve a second pattern with a raised target, then apply it to
        # exactly as many layers as the residual pays for. Same trick the
        # keep-residual redistribution above uses. Layers are heterogeneous only
        # in which pattern they get, so the per-layer solve stays cheap.
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
            # Find the smallest raise that buys a strictly bigger offload set,
            # doubling until something fits. Smallest raise => the residual
            # spreads over the most layers, which keeps every layer close to
            # target instead of piling the whole miss onto a few.
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
                    # The bar is the keep BUDGET, not the base pattern's keep:
                    # the base often sits just under budget, and rejecting any
                    # boost that spends the remaining slack blocks otherwise
                    # legal patterns (measured: of=0.15 found no boost at all
                    # and stayed 0.2 points short).
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
                if not _is_recomputable(node, must_keep):
                    # RNG / anchored save_ops / layer boundaries, and collectives
                    # (a MUST_RECOMPUTE tag on a collective is rewritten to
                    # MUST_SAVE by _demote_collective_recompute_tags after the
                    # solve, so planning a recompute here just leaks into keep).
                    prob += r[node] == 0
                if _offload_forbidden(node, size, must_keep):
                    # Match the concrete offload pass: views, collectives/waits,
                    # non-contiguous tensors, and tiny tensors should not be tagged
                    # for CPU offload by the ILP.
                    prob += o[node] == 0
                g = size / M
                total_keep += k[node] * g
                total_off += o[node] * g
                total_rec += r[node] * g
                objective += r[node] * _recompute_ms(node)

            # A recomputed node must not read an offloaded input: its dup runs during
            # backward and would need the input on GPU, but an offloaded input is on
            # CPU (the remat/offload reload-hoisting mishandles this and faults at
            # runtime). Resolve view/alias chains to the underlying candidate.
            cand_set = set(k)  # membership only
            # Iterate the DICT, not the set. Node.__hash__ is id-based, so
            # set(k) iteration order differs between processes -- emitting these
            # constraints in a per-rank order makes CBC pick different optima of
            # this (highly degenerate) ILP on different ranks. The ranks then
            # tag different nodes, bucket different all-gathers, and deadlock in
            # NCCL. Dicts are insertion-ordered, so this is rank-invariant.
            for v in k:
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
                    else "offload" if o[node].value() > 0.5 else "recompute"
                )
                node.meta["recompute"] = _POLICY_TAG[tag]
                _dbg_intent[node] = tag
                _tag_collective_chain(node, tag)
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
    _dbg(
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
    # Audit the {all_gather, wait} pairs. The policy dump cannot show this: it
    # prints each WAIT's tag under its COLLECTIVE's name (_resolve_producing_op
    # walks through wait_tensor), so a disagreeing pair is invisible there.
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
            " | " + " ".join(f"wait={w}/coll={c}:{n}" for (w, c), n in _pair_bad.items())
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
                size=_get_size(t),
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
            total_activation_per_blk += _get_size(t)

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

    # per-index ACT liveness + freeable sets for the outer peak constraint.
    # Ordered sets (dict keys preserve insertion order). A plain set of
    # StorageObject iterates in id-hash order, which differs between processes.
    # The outer LP sums so.size over that order, and float addition is NOT
    # associative, so each rank builds slightly different LP coefficients; this
    # LP is highly degenerate, so a low-bit coefficient change flips it to a
    # different optimum. Different plans per rank means different tags,
    # different all-gather bucketing, and a NCCL deadlock at the first
    # mismatched collective. Insertion order here is graph order on every rank.
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

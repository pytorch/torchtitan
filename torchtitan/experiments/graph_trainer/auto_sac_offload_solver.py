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
import hashlib
import math
import os
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

from torchtitan.experiments.graph_trainer.auto_sac_offload_solver_helpers import (
    estimate_peak_memory,
    get_per_node_runtime,
    get_transfer_bw,
    optimizer_state_bytes,
)
from torchtitan.experiments.graph_trainer.auto_sac_offload_solver_utils import (
    ACT,
    BENCHMARK,
    BWD_TEMP,
    COST_MODEL,
    get_size,
    GRAD,
    INPUT,
    INTERPRETER,
    PARAM,
    TEMP,
)
from torchtitan.experiments.graph_trainer.common_utils import (
    _get_layer_id,
    _get_module_fqn,
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
from torchtitan.tools.logging import logger

# Set from --compile.debug_memory_policy_solver. The solver logs around 80
# lines per run, which is handy when tuning a budget and noise otherwise.
_debug_logging = False


def solver_debugs(msg, *args):
    """Log a solver diagnostic, but only when debug logging is on."""
    if _debug_logging:
        logger.info(msg, *args)


def _fmt_ranges(values) -> str:
    """Collapse a sorted int list into range notation, e.g. "0-4,7,9-11"."""
    out, vals = [], sorted(values)
    start = prev = vals[0]
    for v in vals[1:] + [None]:
        if v == prev + 1:
            prev = v
            continue
        out.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = v
    return ",".join(out)


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

# Matmuls are the only save_op safe to recompute: timed accurately and
# deterministic. Everything else is priced badly or unsafe to replay.
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

# Only offload tensors at least this big; many small transfers cost more
# overhead than they save.
OFFLOAD_MIN_BYTES = 1 << 20  # 1 MiB

# Every rank pins its own offloaded activations, so they all share whatever host
# memory the node has free.
HOST_MEMORY_FRACTION = 0.8

# Weight on the offload term in the outer objective. Zero over-offloads by
# ~15% but always meets the budget; the full transfer time stops the LP
# offloading at all. The real fix is the bandwidth input, not this weight.
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
    """Nodes the solver may not recompute, though it may still keep or offload
    them: RNG ops, the save_ops save_ops_policy anchors, and the layer
    boundaries (always anchored, so each layer's recompute region is
    self-contained and the inner solves stay independent).
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
    """Broadcast rank 0's tags to every rank.

    The solve is not bit-reproducible, so ranks can pick different optima and
    then deadlock in NCCL on a mismatched all-gather.
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
    solver_debugs(
        "memory policy: adopted rank 0's plan (digest %s over %d tagged nodes); "
        "%d local decision(s) overridden",
        digest,
        len(plan),
        changed,
    )


def _is_recomputable(node: torch.fx.Node, must_keep: set) -> bool:
    """Whether the remat pass may erase and replay this node. Mirrors what the
    tagging path allows, so both solvers agree on which bytes are freeable.
    """
    return node not in must_keep  # and not _is_collective_or_wait(node)


def _offload_forbidden(node: torch.fx.Node, size: int, must_keep: set) -> bool:
    """Whether the ILP must not tag this node for CPU offload.

    On top of what _can_offload_node rejects, this bars must_keep: offloading a
    layer boundary feeds the whole next layer, and the o + r <= 1 constraint
    then blocks recompute for everything reading it.
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
            "keep/recompute/offload fractions must sum to 1, got "
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
    """Bucket a node's final tag into keep, recompute or offload. Untagged
    counts as keep, since the storage stays on the GPU either way.
    """
    pol = node.meta.get("recompute")
    if pol in (CheckpointPolicy.MUST_CPU_OFFLOAD, CheckpointPolicy.PREFER_CPU_OFFLOAD):
        return "offload"
    if pol in (CheckpointPolicy.MUST_RECOMPUTE, CheckpointPolicy.PREFER_RECOMPUTE):
        return "recompute"
    return "keep"


def _audit_tagged_fractions(candidates_by_layer):
    """Measure the split actually written to the graph, per layer and in total.

    Reads the tags rather than the solver's variables, so it covers skipped
    layers too. This is intent, not outcome: a later pass can reject a tag.
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


def two_level_solver(
    trace: TracedResult,
    memory_budget: int,
    optimizer,
    model_parts: list[torch.nn.Module],
    runtime_estimation_mode: str = COST_MODEL,
    cpu_offload_budget_gb: float = 100.0,
    interp_ctx: tuple | None = None,  # (model, *run_args) for INTERPRETER mode
    prefetch_lookahead: int = 1,
    defer_n_layers: int = 1,
    debug_logging: bool = False,
    cpu_offload_bw: int = 10000,
    solver_type: str = "greedy",
) -> torch.fx.GraphModule | None:
    """Run the two-level solver and tag the graph with per-node decisions.

    Returns the graph and its metrics, or None if the solve failed.
    """
    global _debug_logging
    _debug_logging = debug_logging

    _timings = {}
    gm = trace.gm

    _t = time.perf_counter()
    mem_est = estimate_peak_memory(gm, num_state_inputs=trace.num_static_inputs)
    opt_bytes = optimizer_state_bytes(optimizer, model_parts[0])
    _timings["mem_estimation"] = time.perf_counter() - _t
    estimated = mem_est + opt_bytes  # all-keep peak, optimizer included

    if memory_budget > estimated:
        solver_debugs(
            "auto SAC-Offload: budget %.2f GB >= estimated peak %.2f GB; nothing to do",
            memory_budget / 1 << 30,
            estimated / 1 << 30,
        )
        return gm, None

    solver_debugs(
        "2-level ILP-based auto SAC-Offload: runtime estimation mode = %s",
        runtime_estimation_mode,
    )

    def _mem_probe(tag):
        if torch.cuda.is_available():
            solver_debugs(
                "MEMPROBE %-22s allocated=%7.2f GiB reserved=%7.2f GiB",
                tag,
                torch.cuda.memory_allocated() / (1 << 30),
                torch.cuda.memory_reserved() / (1 << 30),
            )

    _mem_probe("before runtime_est")
    _t = time.perf_counter()
    if runtime_estimation_mode == INTERPRETER or runtime_estimation_mode == COST_MODEL:
        logger.warning(
            "INTERPRETER and COST_MODEL modes are currently not supported. Falling back to `BENCHMARK` mode."
        )
        runtime_estimation_mode = BENCHMARK
    _timings["runtime_estimation"] = time.perf_counter() - _t
    _mem_probe("after runtime_est")

    runtime = get_per_node_runtime(gm)
    # Adopt rank 0's runtimes: each rank times its own kernels a few percent
    # apart, which would give each a different C3 offload window. Must happen
    # here, not in plan_outer, which only rank 0 runs.
    if dist.is_available() and dist.is_initialized():
        _rt_payload = [runtime if dist.get_rank() == 0 else None]
        dist.broadcast_object_list(_rt_payload, src=0)
        runtime = _rt_payload[0]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        _mem_probe("after empty_cache")

    _t = time.perf_counter()
    get_fixed_bytes_tuple = get_fixed_bytes(gm, trace.num_static_inputs)
    _timings["get_fixed_bytes"] = time.perf_counter() - _t

    # Already resolved and validated by resolve_host_offload_cap_gib().
    _host_cap_gib = float(cpu_offload_budget_gb)
    # C3 offload window rate: the idle single-GPU rate over log2 of the ranks
    # sharing this host path. Only rank 0 measures; concurrent benchmarks skew.
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
        solver_debugs(
            "Bandwidth inputs: idle d2h=%.1f h2d=%.1f GB/s -> per-rank d2h=%.1f h2d=%.1f "
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

    _t_construct_ilp = _t_solve = 0.0
    # Only rank 0 solves; the rest take the plan from the tag broadcast below.
    # Solving everywhere let low-bit coefficient differences grow into
    # different plans. These placeholders keep the metrics block working.
    _fr = None  # per-layer (keep, recompute, offload) from the outer LP
    _bb = _bs = 0.0  # inner ILP build / solve seconds
    _ifr = None  # fractions the inner ILP actually achieved

    if _rank == 0:

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
            eff_budget_override=None,
            host_cap_gib=_host_cap_gib,
            measured_bw=_measured_bw,
        )
        _t_construct_ilp += _tc or 0.0
        _t_solve += _ts or 0.0

        if _fr is None:
            comm_flag = [None]
        else:
            gm, _bb, _bs, _ifr = plan_and_tag_inner(
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
                solver_type=solver_type,
            )
            comm_flag = [True]
    else:
        comm_flag = [None]

    # sync the comm_flag
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.broadcast_object_list(comm_flag, src=0)

    if not comm_flag[0]:
        raise ValueError("AC solver plan crashed!")

    _sync_plan_from_rank0(gm)

    # Outer-requested fractions, byte-weighted over each layer's owned
    # activation (act_bytes_per_layer = get_fixed_bytes_tuple[4]).
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
        "runtime_mode": runtime_estimation_mode,
        # stage times (s)
        "t_mem_est": _timings["mem_estimation"],
        "t_runtime_est": _timings["runtime_estimation"],
        "t_get_fixed_bytes": _timings["get_fixed_bytes"],
        "t_outer_construct": _t_construct_ilp,
        "t_outer_solve": _t_solve,
        "t_inner_build": _bb,
        "t_inner_solve": _bs,
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
    mem_est: int,
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

    Spreads the memory budget across the transformer blocks. Returns None when
    the graph has no block activations or no feasible split exists.
    """
    gm = trace.gm
    estimated = mem_est + opt_bytes
    solver_debugs(f"memory_budget: {memory_budget}")
    solver_debugs(f"estimated: {estimated}")
    solver_debugs(f"opt_bytes: {opt_bytes}")

    # budget left for the graph once optimizer state is reserved
    eff_budget = memory_budget - opt_bytes
    solver_debugs(f"eff_budget: {eff_budget}")

    runtime_per_node = runtime

    blocks = defaultdict(list)
    nodes = list(gm.graph.nodes)
    for node in nodes:
        b = block_of_node(node)
        if b is not None and b != -1:  # skip non-layer nodes (embeddings/loss)
            blocks[b].append(node)

    (
        _,
        _,
        _,
        _,
        act_bytes_per_layer,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
        bwd_temp_max_per_layer,
    ) = get_fixed_bytes_tuple

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

    bw = get_transfer_bw()
    bw_d2h, bw_h2d = bw["d2h"] * 1e6, bw["h2d"] * 1e6  # GB/s -> bytes/ms
    # With the pinned pool on, the benchmark and the offload path share
    # buffers, so no correction belongs here. Still overridable per machine.
    for _dir in ("d2h", "h2d"):
        _val = (measured_bw or {}).get(_dir, bw[_dir])
        solver_debugs(
            "C3 INPUTS: %s benchmarked %.1f GB/s -> using %.1f GB/s",
            _dir,
            bw[_dir],
            _val,
        )
        bw[_dir] = float(_val)
    bw_d2h, bw_h2d = bw["d2h"] * 1e6, bw["h2d"] * 1e6

    solver_debugs(
        "C3 INPUTS: bw_d2h=%.1f GB/s bw_h2d=%.1f GB/s (isolated benchmark)",
        bw["d2h"],
        bw["h2d"],
    )

    fwd_rt_by_block = defaultdict(float)
    bwd_rt_by_block = defaultdict(float)
    total_fwd_time = 0
    total_bwd_time = 0
    total_loss_bwd_time = 0
    total_loss_fwd_time = 0

    for layer_id, blk_nodes in blocks.items():
        if layer_id != -1:
            for n in blk_nodes:
                if _is_backward_node(n):
                    total_bwd_time += runtime_per_node.get(n.name, 0.0)
                else:
                    total_fwd_time += runtime_per_node.get(n.name, 0.0)
        else:
            for n in blk_nodes:
                if _is_backward_node(n):
                    total_loss_bwd_time += runtime_per_node.get(n.name, 0.0)
                else:
                    total_loss_fwd_time += runtime_per_node.get(n.name, 0.0)

    solver_debugs(
        "outer: runtime fwd %.1f ms bwd %.1f ms (loss fwd %.1f bwd %.1f)",
        total_fwd_time,
        total_bwd_time,
        total_loss_fwd_time,
        total_loss_bwd_time,
    )

    for block_id in range(num_blocks):
        bwd_rt_by_block[block_id] = total_bwd_time / num_blocks
        fwd_rt_by_block[block_id] = total_fwd_time / num_blocks

    # The C3 windows are bandwidth times these, so log them to check against a
    # profiler trace: an inflated window looks just like optimistic bandwidth.
    _fsum = sum(fwd_rt_by_block.values())
    _bsum = sum(bwd_rt_by_block.values())
    solver_debugs(
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
    aG = {b: act_bytes_per_layer[b] / M for b in layer_ids}  # owned act, GiB
    bw_d2h_g, bw_h2d_g = bw_d2h / M, bw_h2d / M  # GiB/ms

    solver_debugs(
        "outer: all-keep peak %.2f GiB + optimizer %.2f GiB vs budget %.2f GiB "
        "-> graph budget %.2f GiB",
        mem_est / M,
        opt_bytes / M,
        memory_budget / M,
        B_g,
    )

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

    def _param_owner_fqn(node):
        """Placeholders never pass through a module forward, so they carry no
        module_fqn -- read it off the first consumer that does. Returns e.g.
        "layers.7.attention.wq" or "tok_embeddings".
        """
        seen, queue = set(), list(node.users)
        while queue:
            u = queue.pop(0)
            if u in seen:
                continue
            seen.add(u)
            if fqn := _get_module_fqn(u):
                return fqn
            queue.extend(u.users)
        return ""

    # Dedup by storage id: weight tying makes the same storage appear twice.
    seen_sids = set()
    param_bytes = 0
    non_block_param_bytes = 0  # embedding / lm_head: not block-shaped
    for n in persistent_state:
        is_block = _param_owner_fqn(n).startswith("layers.")
        for t in pytree.tree_leaves(n.meta.get("val")):
            if not isinstance(t, torch.Tensor) or t.device.type != "cuda":
                continue
            sid = t.untyped_storage()._cdata
            if sid in seen_sids:
                continue
            seen_sids.add(sid)
            sz = get_size(t)
            param_bytes += sz
            if not is_block:
                non_block_param_bytes += sz

    _local_ranks = max(1, int(os.environ.get("LOCAL_WORLD_SIZE", "1")))
    param_bytes_per_layer = param_bytes / len(layer_ids)

    solver_debugs(
        "outer: params %.2f GiB over %d layers (%.2f GiB/layer, %.2f GiB non-block)",
        param_bytes / M,
        len(layer_ids),
        param_bytes_per_layer / M,
        non_block_param_bytes / M,
    )

    k, r, o = {}, {}, {}
    prob = LpProblem("outer_ilp", LpMinimize)
    for b in layer_ids:
        k[b] = LpVariable(f"k_{b}", lowBound=0, upBound=1)
        r[b] = LpVariable(f"r_{b}", lowBound=0, upBound=1)
        o[b] = LpVariable(f"o_{b}", lowBound=0, upBound=1)
        prob += k[b] + r[b] + o[b] == 1, f"split_{b}"

    _min_keep = {}
    for L in layer_ids:
        must_save = per_layer_must_keep.get(L, 0)
        aL = max(act_bytes_per_layer[L], 1)
        _min_keep[L] = min(1.0, must_save / aL)
        prob += k[L] >= _min_keep[L], f"mustkeep_{L}"
    solver_debugs(
        "outer: forced-keep floor per layer: min %.3f max %.3f",
        min(_min_keep.values()),
        max(_min_keep.values()),
    )

    prob += o[last_layer_id] == 0.0

    _fwd_exprs, _bwd_exprs = {}, {}
    for L in layer_ids:
        aL = act_bytes_per_layer[L] / M
        kept_below = lpSum(
            k[l] * (act_bytes_per_layer[l] / M) for l in layer_ids if l < L
        )

        params_total_forward = (
            len(layer_ids) * param_bytes_per_layer
            + (_local_ranks - 1) * param_bytes_per_layer
        )  # all sharded + current layer unsharded

        if L < last_layer_id:
            params_total_forward += (
                _local_ranks - 1
            ) * param_bytes_per_layer  # add next layer's unsharded if not last

        if L == 0:
            prev_layer_offload = 0
        else:
            prev_layer_offload = o[L - 1] * act_bytes_per_layer[L - 1] / M

        prob += (
            prev_layer_offload + kept_below + aL + params_total_forward / M <= B_g
        ), f"fwd_peak_{L}"
        _fwd_exprs[L] = prev_layer_offload + kept_below + aL + params_total_forward / M

        # the previous layer's offload lands back on the GPU here
        prefetch_activation = 0
        if L > 0:
            prefetch_activation += o[L - 1] * act_bytes_per_layer[L - 1] / M

        prob += (
            kept_below
            + aL
            + bwd_temp_max_per_layer[L] / M
            + len(layer_ids) * param_bytes_per_layer / M
            + prefetch_activation
            <= B_g
        ), f"bwd_peak_{L}"
        # same expression minus the prefetch term, kept for the SOLVED report
        bwd_memory_constraint = (
            kept_below
            + aL
            + bwd_temp_max_per_layer[L] / M
            + len(layer_ids) * param_bytes_per_layer / M
        )
        _bwd_exprs[L] = bwd_memory_constraint

    # for loss + lm_head layer/part:
    kept_below = lpSum(k[l] * (act_bytes_per_layer[l] / M) for l in layer_ids)
    all_mem_at_peak = act_bytes_per_layer[-1] / M
    params = (
        len(layer_ids) * param_bytes_per_layer
        + (_local_ranks - 1) * param_bytes_per_layer
    )
    prob += (kept_below + all_mem_at_peak + params / M <= B_g), "loss_peak"
    loss_constraint = kept_below + all_mem_at_peak + params / M

    solver_debugs(
        "outer: loss/lm_head block: act %.2f GiB + params %.2f GiB",
        all_mem_at_peak,
        params / M,
    )

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

        solver_debugs(
            f"layer: {b}, bw_d2h_g: {bw_d2h_g}, fwd_rt_by_block[b]: {fwd_rt_by_block[b]}"
        )
        solver_debugs(
            f"layer: {b}, bw_h2d_g: {bw_h2d_g}, bwd_rt_by_block[b]: {bwd_rt_by_block[b]}"
        )

    # Host-side pinned-memory cap, computed once by the caller and identical on
    # every rank (see resolve_host_offload_cap_gib).
    prob += (
        lpSum(o[b] * aG[b] for b in layer_ids) <= host_cap_gib,
        "host_cap",
    )
    solver_debugs("outer: host pinned-memory cap %.2f GiB/rank", host_cap_gib)

    _off_price = OFFLOAD_TIME_PRICE
    _off_ms = {b: _off_price * aG[b] / max(bw_d2h_g, 1e-12) for b in layer_ids}
    solver_debugs(
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
    ), "added_runtime"

    _t_construct_ilp_end = time.perf_counter()
    solver_debugs(
        "outer_ilp construction: CBC solve took %.3f s",
        _t_construct_ilp_end - _t_construct_ilp_start,
    )
    _t_construct_ilp = _t_construct_ilp_end - _t_construct_ilp_start

    _t_solve_start = time.perf_counter()
    status = prob.solve(PULP_CBC_CMD(msg=0))
    _t_solve_end = time.perf_counter()
    solver_debugs(
        "outer_ilp: CBC solve took %.3f s (status=%s)",
        _t_solve_end - _t_solve_start,
        LpStatus[status],
    )
    _t_solve = _t_solve_end - _t_solve_start

    if LpStatus[status] != "Optimal":
        # Re-solve with one elastic slack to report the smallest budget that
        # admits any split. Diagnostic only; produces no plan.
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

    # Where the LP thinks the peak lands. Compare with the estimator: if this
    # says the plan fits and the real run disagrees, the peak model is at fault.
    rows = sorted(
        [("fwd", L, e.value()) for L, e in _fwd_exprs.items()]
        + [("bwd", L, e.value()) for L, e in _bwd_exprs.items()]
        + [("loss", -1, loss_constraint.value())],
        key=lambda r: -r[2],
    )
    _name = lambda k, L: "loss_peak" if k == "loss" else f"{k}_peak_{L}"  # noqa: E731
    _kind, _L, _peak = rows[0]
    solver_debugs(
        "outer: predicted peak %.2f GiB at %s (budget %.2f, slack %.2f) "
        "-> %.2f GiB with optimizer, vs %.2f GiB requested",
        _peak,
        _name(_kind, _L),
        B_g,
        B_g - _peak,
        _peak + opt_bytes / M,
        memory_budget / M,
    )
    solver_debugs(
        "outer: tightest constraints: %s",
        "  ".join(f"{_name(k, L)}={v:.2f}" for k, L, v in rows[:5]),
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
    solver_debugs(
        "outer: split %.2f GiB of activation -> keep %.2f (%.1f%%)  "
        "recompute %.2f (%.1f%%)  offload %.2f (%.1f%%)",
        tot_a / GiB,
        kb / GiB,
        pct(kb),
        rb / GiB,
        pct(rb),
        ob / GiB,
        pct(ob),
    )
    # Layers usually fall into a handful of distinct splits, so group them
    # instead of printing one line each.
    _by_split = defaultdict(list)
    for b in layer_ids:
        _by_split[tuple(round(v, 3) for v in alloc[b])].append(b)
    for (kv, rv, ov), bs in sorted(_by_split.items(), key=lambda e: -len(e[1])):
        solver_debugs(
            "outer: k=%.3f r=%.3f o=%.3f -> %d layer(s) %s",
            kv,
            rv,
            ov,
            len(bs),
            _fmt_ranges(bs),
        )

    # The outer LP only produces fractions; the inner ILP does the tagging.
    return alloc, _t_construct_ilp, _t_solve


def plan_and_tag_inner(
    mem_est: int,
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
    solver_type: str = "greedy",
) -> torch.fx.GraphModule | None:
    """Pick keep/recompute/offload per tensor, then tag the graph with it.

    Layers sharing a candidate signature and the same outer fractions are solved
    once and the resulting pattern is stamped onto each of them.
    """
    _t_start = time.perf_counter()  # whole plan_and_tag_inner wall time
    keep_fraction, recompute_fraction, offload_fraction = _validate_fractions(
        keep_fraction, recompute_fraction, offload_fraction
    )

    runtime_per_node = runtime

    # Nodes we refuse to recompute: RNG for correctness, plus the save_ops the
    # estimator prices badly or that are unsafe to replay. Matmuls stay
    # recomputable, which is what makes the requested fractions achievable.
    must_keep = get_must_keep_list(gm, save_ops_policy=SAVE_OPS_POLICY)
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
        _,
        _,
        _,
        _,
        _,
        _,
    ) = get_fixed_bytes_tuple
    fixed_bytes = mem_est - total_freeable

    M = MEM_MULTIPLIER

    # Global host (pinned CPU) budget for offload, tracked across the layer loop
    # so independent solves don't collectively blow the host budget.
    remaining_off_g = (float(cpu_offload_budget_gb) * (1 << 30)) / M

    _dbg_unapplied = [0]

    def _tag_of(k, r, o):
        # build_inner_ilp returns plain 0.0/1.0 floats from the greedy
        # ranking; accept LpVariables too so the solver path still works.
        def v(x):
            return x if isinstance(x, (int, float)) else (x.value() or 0.0)

        return "keep" if v(k) > 0.5 else "offload" if v(o) > 0.5 else "recompute"

    def _collective_chain(node):
        """Upstream nodes of a wait candidate that must share its decision.

        An FSDP unshard is a cast, an all-gather, then a wait; only the wait's
        storage survives into backward, so it is the candidate while the cost
        sits upstream. Empty for non-collectives.
        """
        if node.target is not torch.ops._c10d_functional.wait_tensor.default:
            return []
        coll = node.args[0]
        if not isinstance(coll, torch.fx.Node) or not is_all_gather_into_tensor(coll):
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
        """Recompute cost of a candidate, including its collective chain. A
        wait alone costs ~0 ms, which would price a re-issued all-gather free.
        """
        ms = runtime_per_node.get(node.name, 0.0)
        for up in _collective_chain(node):
            ms += runtime_per_node.get(up.name, 0.0)
        return ms

    def build_inner_ilp(candidates, kf, of, remaining_off_g, act):
        """Build one layer pattern's keep/recompute/offload split as an ILP.

        ``kf`` and ``of`` are the outer LP's keep and offload fractions of this
        pattern's ``act`` bytes. Returns the unsolved problem plus its 0/1
        variables, so the caller solves it and reads the decisions back.
        """
        bw_d2h_g = get_transfer_bw()["d2h"] * 1e6 / M  # GiB/ms

        act_g = act / M
        prob = LpProblem(f"inner_layer_{kf}_{of}", LpMinimize)
        k, r, o = {}, {}, {}
        for node, size in candidates:
            k[node] = LpVariable(f"k_{node.name}", cat=LpBinary)
            r[node] = LpVariable(f"r_{node.name}", cat=LpBinary)
            o[node] = LpVariable(f"o_{node.name}", cat=LpBinary)
            prob += k[node] + r[node] + o[node] == 1
            if not _is_recomputable(node, must_keep):
                # RNG, anchored save_ops, layer boundaries and collectives. A
                # recompute tag on a collective is demoted to save after the
                # solve anyway, so planning one here just leaks into keep.
                prob += r[node] == 0
            if _offload_forbidden(node, size, must_keep) or _is_collective_or_wait(
                node
            ):
                # Match what the offload pass accepts: views, collectives,
                # waits, non-contiguous and tiny tensors cannot be offloaded.
                prob += o[node] == 0

        total_keep = lpSum(k[n] * (sz / M) for n, sz in candidates)
        total_off = lpSum(o[n] * (sz / M) for n, sz in candidates)
        objective = lpSum(r[n] * _recompute_ms(n) for n, sz in candidates)

        target_keep_g = kf * act_g
        target_off_g = min(of * act_g, remaining_off_g)
        off_short = LpVariable("off_short", lowBound=0)

        # Keep is a hard constraint: exceeding it costs real peak memory.
        prob += total_keep <= target_keep_g * (1.0 + 1e-6), "must_keep"

        # Offload may run up to 5% past its target, charged at transfer time.
        prob += off_short >= total_off - target_off_g, "offload_uplimit"
        prob += off_short <= target_off_g * 0.05, "offload_over_cap"
        objective += off_short * (1.0 / bw_d2h_g)

        prob.setObjective(objective)
        return prob, k, r, o, target_keep_g, target_off_g

    def build_inner_greedy(candidates, kf, of, remaining_off_g, act):
        """Greedily split one layer's candidates into keep/recompute/offload.

        ``kf`` and ``of`` are the outer LP's keep and offload fractions of this
        layer's ``act`` bytes; recompute takes the rest. Returns one 0/1 entry
        per candidate in each of the k, r, o dicts.
        """
        act_bytes = act  # bytes, not GiB
        k, r, o = {}, {}, {}
        decisions_per_act = defaultdict(
            int
        )  # 0: no decision, 1: save, 2: offload, 3: recompute
        all_candidates = []
        all_candidates_by_size = defaultdict(float)
        for node, size in candidates:
            all_candidates_by_size[node] = size
            all_candidates.append(node)
            decisions_per_act[node] = 0
            if node in must_keep:
                decisions_per_act[node] = 1

        offload_budget = of * act_bytes
        keep_budget = kf * act_bytes
        recompute_budget = (1 - of - kf) * act_bytes

        # most expensive to recompute per byte first
        ranked_activations = sorted(
            (n for n in all_candidates if all_candidates_by_size[n] > 0),
            key=lambda n: (
                -runtime_per_node.get(n.name, 0.0) / (all_candidates_by_size[n] / M),
                n.name,
            ),
        )

        def _fill(gap, ordered, decision, allow_overshoot=False):
            """Move up to `gap` bytes out of keep, preferring `ordered`, and
            return the bytes actually placed.

            `allow_overshoot` takes one extra node to cover the remainder. Only
            the final keep-closing pass wants that, since keeping more than
            planned is what breaks the budget; the offload and recompute passes
            must stop short and let the next pass pick up the rest.
            """
            placed = 0.0
            if gap <= 0:
                return placed
            for n in ordered:
                if placed >= gap:
                    break
                if decisions_per_act[n] != 0:
                    continue
                size_n = all_candidates_by_size[n]
                if size_n <= gap - placed:
                    decisions_per_act[n] = decision
                    placed += size_n
            rem = gap - placed
            if rem > 0 and allow_overshoot:
                over = [
                    n
                    for n in ordered
                    if decisions_per_act[n] == 0
                    and all_candidates_by_size[n] >= rem * 0.95
                ]
                if over:
                    best = min(over, key=lambda n: (all_candidates_by_size[n], n.name))
                    decisions_per_act[best] = decision
                    placed += all_candidates_by_size[best]
            return placed

        # Offload the head of the ranking: the activations the recompute pass
        # below would charge most for.
        offload_gap = offload_budget
        achieved_offload = _fill(
            offload_gap,
            [n for n in ranked_activations if _can_offload_node(n)],
            2,
        )

        # Recompute the cheapest activations per byte: the tail of the
        # ranking, so walk it reversed.
        recompute_order = [
            n for n in reversed(ranked_activations) if _is_recomputable(n, must_keep)
        ]
        achieved_recompute = _fill(recompute_budget, recompute_order, 3)

        # Whatever the two passes could not place stays resident, which can
        # push keep above the outer LP's target and break the budget. Close
        # that gap with recompute.
        kept_act_budget = act_bytes - achieved_recompute - achieved_offload
        if kept_act_budget > keep_budget:
            achieved_recompute += _fill(
                kept_act_budget - keep_budget,
                recompute_order,
                3,
                allow_overshoot=True,
            )

        solver_debugs(
            "inner:   want keep %.2f rec %.2f off %.2f GiB -> got keep %.2f "
            "rec %.2f off %.2f GiB",
            keep_budget / M,
            recompute_budget / M,
            offload_budget / M,
            (act_bytes - achieved_recompute - achieved_offload) / M,
            achieved_recompute / M,
            achieved_offload / M,
        )

        # 0/1 = keep (undecided nodes stay resident), 2 = offload, 3 = recompute
        for node, decision in decisions_per_act.items():
            k[node] = 1.0 if decision in (0, 1) else 0.0
            o[node] = 1.0 if decision == 2 else 0.0
            r[node] = 1.0 if decision == 3 else 0.0

        return k, r, o

    # this will group the layers that have the same budget fractions by the outer
    layer_groups = defaultdict(list)  # sig -> [layer_ids]
    sig_meta = {}  # sig -> (candidates_of_representative, kf, of)

    def _node_key(node, size):
        fqn = node.meta.get("custom", {}).get(_MODULE_FQN, "")
        # Strip the "layers.N." prefix for a layer-relative key; block
        # boundaries are bare "layers.N", so guard the split.
        if fqn.startswith("layers."):
            parts = fqn.split(".", 2)
            rel = parts[2] if len(parts) >= 3 else ""  # "" = block output/boundary
        else:
            rel = fqn
        return (rel, str(node.target), size)

    def _layer_keys(candidates):
        # Append an ordinal so colliding candidates get distinct keys; every
        # layer builds candidates the same way, so the i-th agrees.
        seen = {}
        out = []
        for n, s in candidates:
            base = _node_key(n, s)
            i = seen.get(base, 0)
            out.append(base + (i,))
            seen[base] = i + 1
        return out

    total_keep = 0
    total_recompute = 0
    total_offload = 0
    num_layers = 0
    for b, candidates in candidates_by_layer.items():

        if per_layer_fractions is not None:
            _li = int(b.split(".")[1]) if isinstance(b, str) else b
            kf, _, of = per_layer_fractions.get(
                _li, (keep_fraction, recompute_fraction, offload_fraction)
            )
        else:
            kf, of = keep_fraction, offload_fraction

        total_keep += kf
        total_recompute += 1 - kf - of
        total_offload += of
        num_layers += 1

        _keys = _layer_keys(candidates)  # per-layer unique (ordinal-disambiguated)
        sig = (tuple(sorted(_keys)), kf, of)
        layer_groups[sig].append(b)
        sig_meta.setdefault(sig, (candidates, kf, of, layer_act[b]))

    solver_debugs(
        "inner: target average k=%.3f r=%.3f o=%.3f",
        total_keep / num_layers,
        total_recompute / num_layers,
        total_offload / num_layers,
    )

    pattern_by_sig = {}
    skipped = []  # grouped solve does not drop layers; kept for the DONE log
    block_build_s = block_solve_s = 0.0
    for _sig_i, (sig, (candidates, kf, of, sig_act)) in enumerate(sig_meta.items(), 1):
        solver_debugs(
            "inner: pattern %d/%d (%s) covers %d layer(s) %s, %d candidates, "
            "act %.2f GiB",
            _sig_i,
            len(sig_meta),
            solver_type,
            len(layer_groups[sig]),
            _fmt_ranges(layer_groups[sig]),
            len(candidates),
            sig_act / M,
        )
        _t_b = time.perf_counter()
        if solver_type == "greedy":
            k, r, o = build_inner_greedy(candidates, kf, of, remaining_off_g, sig_act)
            # The pinned host pool is global, but this solves one signature
            # stamped onto every layer sharing it, so charge the pool for all
            # of them or the cap never binds.
            _off_bytes = sum(sz for n, sz in candidates if o.get(n, 0.0) > 0.5)
            remaining_off_g -= _off_bytes * len(layer_groups[sig]) / M
            _t_s = time.perf_counter()
        else:
            prob, k, r, o, tgt_k, tgt_o = build_inner_ilp(
                candidates, kf, of, remaining_off_g, sig_act
            )
            _t_s = time.perf_counter()
            prob.solve(PULP_CBC_CMD(msg=0))
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
        block_build_s += _t_s - _t_b
        block_solve_s += time.perf_counter() - _t_s

        if _debug_logging:
            for node, size in candidates:
                solver_debugs(
                    "inner:   %-22s %7.1f MiB  %6.2f ms  %s",
                    node.name,
                    size / (1 << 20),
                    _recompute_ms(node),
                    _tag_of(k[node], r[node], o[node]),
                )

        pattern_by_sig[sig] = {
            key: _tag_of(k[n], r[n], o[n])
            for key, (n, s) in zip(_layer_keys(candidates), candidates)
        }

    # stamp each signature's pattern onto every layer sharing it
    pattern_for_layer = {}
    for sig, layer_id in layer_groups.items():
        for b in layer_id:
            pattern_for_layer[b] = pattern_by_sig[sig]

    per_node_decision = {}
    for b, pattern in pattern_for_layer.items():
        cands = candidates_by_layer[b]
        for key, (node, size) in zip(_layer_keys(cands), cands):
            if key in pattern:
                per_node_decision[node] = pattern[key]
            else:
                _dbg_unapplied[0] += size
                # TODO accumulate offload bytes here, and fall back to
                # recompute or save once the host budget is exceeded

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

    node_counts = {"keep": 0, "recompute": 0, "offload": 0}
    for _cands in candidates_by_layer.values():
        for _node, _ in _cands:
            node_counts[_classify_tag(_node)] += 1
    solver_debugs(
        "inner: tagged %d candidates -> keep %d recompute %d offload %d",
        sum(node_counts.values()),
        node_counts["keep"],
        node_counts["recompute"],
        node_counts["offload"],
    )

    # Authoritative: fractions read from the graph tags the passes will act on.
    est_fwd_peak_g = (fixed_bytes + verified["keep"]) / M
    solver_debugs(
        "inner: done -- keep %.1f%% recompute %.1f%% offload %.1f%% | "
        "est fwd-boundary peak %.2f GiB (all-keep %.2f) | %d layers skipped",
        100 * verified["keep"] / total_act,
        100 * verified["recompute"] / total_act,
        100 * verified["offload"] / total_act,
        est_fwd_peak_g,
        (fixed_bytes + total_act) / M,
        len(skipped),
    )
    # Audit {all_gather, wait} pairs: the policy dump prints each wait's tag
    # under its collective's name, so a disagreeing pair is invisible there.
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
    solver_debugs(
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

    solver_debugs(
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

    # Find each storage's last forward use and first backward use, and classify
    # it along the way.
    live_sids_per_index = {}
    freeable_sids_per_index = {}
    live_key = {}
    death_of = {}
    storages = []
    storages_by_prod_node = defaultdict(list)
    for node in nodes:
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
            elif _is_backward_node(prod):
                category = BWD_TEMP
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

            storages_by_prod_node[prod].append(new_object)
            storages.append(new_object)

    per_layer_node_bytes: defaultdict[
        int, defaultdict[torch.fx.Node, int]
    ] = defaultdict(lambda: defaultdict(int))
    total_freeable = 0

    # gradient bytes owned by each layer
    grad_size_per_layer = defaultdict(int)
    for object in storages:
        if object.category != GRAD:
            continue
        layer_prod = _get_layer_id(object.producer_node)
        grad_size_per_layer[layer_prod] += object.size

    solver_debugs(
        "graph: gradients %.2f GiB over %d layers",
        sum(grad_size_per_layer.values()) / MEM_MULTIPLIER,
        len(grad_size_per_layer),
    )

    bwd_temp_add_at, bwd_temp_remove_at = defaultdict(list), defaultdict(list)
    for object in storages:
        if object.category != BWD_TEMP:
            continue
        bwd_temp_add_at[object.produced_index].append(object)
        bwd_temp_remove_at[object.death_index].append(object)

    # high-water mark of backward temporaries within each layer
    cur = 0
    bwd_temp_max_per_layer = defaultdict(int)
    for i, node in enumerate(nodes):
        if not _is_backward_node(node):
            continue
        for b in bwd_temp_add_at.get(i, ()):
            cur += b.size  # born at i -> live during i
        L = _get_layer_id(node)
        if isinstance(L, int) and cur > bwd_temp_max_per_layer[L]:
            bwd_temp_max_per_layer[L] = cur
        for b in bwd_temp_remove_at.get(i, ()):
            cur -= b.size  # dies at i -> still live *during* i

    solver_debugs(
        "graph: backward temporaries per layer: min %.2f max %.2f GiB",
        min(bwd_temp_max_per_layer.values(), default=0) / MEM_MULTIPLIER,
        max(bwd_temp_max_per_layer.values(), default=0) / MEM_MULTIPLIER,
    )

    _no_bwd_use = 0  # activations never read by backward
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
        act_add_at[object.produced_index].append(object)
        act_remove_at[object.death_index].append(object)
        first_backward = object.first_bwd_use_index
        if first_backward is None or first_backward == INT64_MAX:
            _no_bwd_use += 1
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

    solver_debugs(
        "graph: %d storages, %.2f GiB freeable activation over %d layers "
        "(%d activations never read by backward)",
        len(storages),
        total_freeable / MEM_MULTIPLIER,
        len(per_layer_node_bytes),
        _no_bwd_use,
    )

    candidates_by_layer = {
        b: list(nb.items()) for b, nb in per_layer_node_bytes.items()
    }
    layer_act = {b: sum(sz for _, sz in c) for b, c in candidates_by_layer.items()}

    # A layer is visited twice and _get_layer_id cannot tell the visits apart,
    # but backward also carries every gradient so far. Separate dicts let the
    # peak model charge the right baseline on each side.
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

    # Per-index activation liveness for the outer peak constraint. Dicts, not
    # sets: set order is id-hash order, which differs per process and would
    # give each rank slightly different LP coefficients.
    live: dict = {}
    live_freeable: dict = {}

    # this is just each layer's activation memory
    act_bytes_per_layer = defaultdict(int)
    for so in storages:
        if so.category == ACT:
            if block_of_node(so.producer_node) >= 0:
                act_bytes_per_layer[block_of_node(so.producer_node)] += so.size
            elif (
                block_of_node(so.producer_node) == -1
                and _get_module_fqn(so.producer_node) != ""
            ):
                act_bytes_per_layer[block_of_node(so.producer_node)] += so.size

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
    solver_debugs(
        "graph: non-layer activation peak %.2f GiB at %s",
        non_layer_act_peak / MEM_MULTIPLIER,
        non_layer_act_argmax[1] if non_layer_act_argmax else "n/a",
    )

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
    # fixed_bytes_max_by_layer is the non-activation baseline per layer;
    # act_bytes_per_layer is the activation the outer LP splits up.
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

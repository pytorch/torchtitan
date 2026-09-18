# Per-node memory estimation for an AOTAutograd joint FX graph
# (e.g. the per-rank graph from aot_export_joint_with_descriptors / autoparallel).
#
# Mirrors the semantics of torch._inductor.fx_passes.memory_estimator.build_memory_profile
# but keeps the per-node attribution instead of flattening to a list of ints.

from __future__ import annotations
from torchtitan.experiments.graph_trainer.make_fx_tracer import TracedResult
import itertools
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from torchinsights.graph_estimation.transfertime_estimator import get_transfer_bw

from torch.utils.checkpoint import CheckpointPolicy
from torchinsights.graph_estimation import (
    estimate_peak_memory,
    MemoryEstimatorResult,
    optimizer_state_bytes,
    UNKNOWN_OPTIMIZER_BYTES,
)
import torch
import torch.fx as fx
import torch.utils._pytree as pytree
from torch._inductor.fx_passes.memory_estimator import (
    _is_releasable,
    GraphAliasTracker,
    StorageKey,
)
from torch.fx.experimental.symbolic_shapes import optimization_hint
from torch.fx.node import map_arg
from torchtitan.experiments.graph_trainer.common_utils import (
    _get_layer_id,
    _get_module_fqn,
    _is_backward_node,
    _MODULE_FQN,
    _NOT_IN_LAYERS,
)
from torchinsights.graph_estimation._fx_utils import (
    _is_backward_node,
    _is_costable_op,
    _VIEW_OP_NAMES,
    ACT,
    feeds_grad_collective,
    get_size,
    GRAD,
    INPUT,
    is_pre_bucket_all_gather,
    is_pre_bucket_reduce_scatter,
    PARAM,
    ROUNDING,
    STATES_PER_PARAM,
    TEMP,
    UNKNOWN_OPTIMIZER_BYTES,
)
from torchtitan.experiments.graph_trainer.cpu_offload import (
    _can_offload_node,
    _get_storage_chain,
    _is_collective_or_wait,
    _is_view,
)
from torchtitan.experiments.graph_trainer.fsdp_patterns import is_all_gather_into_tensor
from torchtitan.experiments.graph_trainer.fsdp_passes import is_wait_tensor_from_fsdp
from torchtitan.tools.logging import logger

from torchinsights.graph_estimation.runtime_estimator import (
    # from torchtitan.experiments.graph_trainer.runtime_estimator import (
    BENCHMARK,
    COST_MODEL,
    INTERPRETER,
    RuntimeEstimator,
)

GiB = 1024**3

# 1: save, 2: offload, 3: recompute
_POLICY_TAG = {
    0: CheckpointPolicy.MUST_SAVE,
    1: CheckpointPolicy.MUST_SAVE,
    2: CheckpointPolicy.MUST_CPU_OFFLOAD,
    3: CheckpointPolicy.MUST_RECOMPUTE,
}

def _nbytes(sk: StorageKey) -> int:
    # optimization_hint resolves SymInt byte counts to a concrete hint.
    return int(optimization_hint(sk.storage.nbytes()))


def get_size_node(node):
    size = 0
    for t in pytree.tree_leaves(node.meta.get("val")):
        if not isinstance(t, torch.Tensor) or t.device.type != "cuda":
            continue
        size += get_size(t)

    return size


@dataclass
class NodeMemory:
    index: int  # index into list(graph.nodes)
    name: str
    target: str
    alloc: int  # fresh storage allocated by this node
    freed: int  # storage whose last use is this node
    live_after_alloc: int  # live bytes at the peak of this node
    live_after_free: int  # live bytes once dead inputs are dropped
    is_bwd: bool
    fqn: str

    @property
    def live_gib(self) -> float:
        return self.live_after_alloc / GiB


def new_storage(node: torch.fx.Node) -> bool:
    # Check for tensor method views
    if node.op == "call_method" and node.target in _VIEW_OP_NAMES:
        return True
    # Check for functions that create views
    if node.op == "call_function" and node.target in {torch.narrow, torch.select}:
        return True
    return False

def _is_rng_op(node: torch.fx.Node) -> bool:
    """RNG ops cannot be replayed by the remat pass, so they must never be
    recomputed (they may still be kept or offloaded)."""
    return torch.Tag.nondeterministic_seeded in getattr(node.target, "tags", set())

def get_must_keep_list(
    gm: torch.fx.GraphModule
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
        # if node.target in save_ops:
        #     must_keep.add(node)
        #     continue
        node_layer = _get_layer_id(node)
        for user in node.users:
            if not _is_backward_node(user) and _get_layer_id(user) > node_layer:
                must_keep.add(node)
                break
    return must_keep


def loss_region_peak(gm: fx.GraphModule | fx.Graph) -> int:
    """Peak bytes of storages ALLOCATED BY the loss/lm_head region.

    Ownership-based, not window-based: a storage counts only if its first
    appearance in the graph is at a loss/lm_head node, so activations and
    parameters belonging to layers 0..N never enter the total. Aliases are
    free -- a node producing only views has no fresh allocations.
    """
    graph = gm.graph if isinstance(gm, fx.GraphModule) else gm
    nodes = list(graph.nodes)
    index = {n: i for i, n in enumerate(nodes)}
    tracker = GraphAliasTracker(nodes)

    def _in_loss(n: fx.Node) -> bool:
        if _get_layer_id(n) != _NOT_IN_LAYERS:
            return False
        fqn = _get_module_fqn(n)
        return "loss" in fqn or "lm_head" in fqn

    # (alloc_idx, death_idx, nbytes, allocating node)
    intervals: list[tuple[int, int, int, fx.Node]] = []
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

def _recompute_ms(node, runtime_per_node):
    """Recompute cost of a candidate, including its collective chain.

    A wait is only a barrier and costs about 0 ms, so charging it alone
    would price a re-issued all-gather at zero and the solver would buy
    every collective recompute for free.
    """
    ms = runtime_per_node.get(node.name, 0.0)
    for up in _collective_chain(node):
        ms += runtime_per_node.get(up.name, 0.0)
    return ms

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
    logger.info(
        "memory policy: adopted rank 0's plan (digest %s over %d tagged nodes); "
        "%d local decision(s) overridden",
        digest,
        len(plan),
        changed,
    )

def per_node_memory(
    gm: fx.GraphModule | fx.Graph,
    runtime_estimation_mode,
    cpu_offload_budget_gb: int,
    requested_budget: int,
    optimizer,
    model_parts: list[torch.nn.Module],
    num_state_inputs: int,
    trace: TracedResult,
    interp_ctx: tuple | None = None,  # (model, *run_args) for INTERPRETER mode
    is_releasable: Callable[[fx.Node], bool] | None = None,
    device_filter: Callable[[torch.device], bool] | None = None,
) -> tuple[int, list[NodeMemory]]:
    """Return (baseline_bytes, rows).

    baseline_bytes is what placeholders/get_attrs (params, buffers, tangents,
    inputs) cost before the first op runs. Each row carries the live-bytes
    watermark at that node index.
    """
    graph = gm.graph if isinstance(gm, fx.GraphModule) else gm
    nodes = list(graph.nodes)
    opt_bytes = optimizer_state_bytes(optimizer, model_parts[0])
    requested_budget = requested_budget - opt_bytes
    logger.info(f"loss_region_peak(gm): {loss_region_peak(gm)/(1 << 30):.2f}")
    logger.info(f"opt_bytes: {opt_bytes/(1 << 30):.2f}")
    logger.info(f"requested_budget: {requested_budget/(1 << 30):.2f}")

    placeholders = [n for n in nodes if n.op == "placeholder"]
    persistent_state = set(placeholders[:num_state_inputs])
    logger.info(f"persistent_state: {persistent_state}")

    forward_ops_per_layer = defaultdict(list)
    backward_ops_per_layer = defaultdict(list)
    activation_ops_per_layer = defaultdict(list)
    activation_size_per_layer = defaultdict(int)
    unsharded_parameter_size_per_layer = defaultdict(int)
    persistent_parameter_size = 0

    for node in persistent_state:
        persistent_parameter_size += get_size_node(node)

    node_index = {n: i for i, n in enumerate(nodes)}

    all_layer_ids: set[int] = set()
    for node in gm.graph.nodes:
        lid = _get_layer_id(node)
        if lid != _NOT_IN_LAYERS:
            all_layer_ids.add(lid)

    last_layer_id = max(all_layer_ids) if len(all_layer_ids) > 1 else _NOT_IN_LAYERS
    # fwd_last = max(node_index[n] for n in forward_ops_per_layer[last_layer_id])
    # bwd_first = min(node_index[n] for n in backward_ops_per_layer[last_layer_id])

    def is_act(node):
        for u in node.users:
            if _is_backward_node(u):
                return True
        return False


    # ---------------------- runtime estimation ------------------------------
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
    # ----------------------–------–--–––––––---–------–-–-–-––-–-–-----------


    runtime_per_node = runtime.node_runtimes_ms

    # runtime = RuntimeEstimator()(BENCHMARK).estimate(trace)
    per_node_ms = runtime.node_runtimes_ms
    logger.info(f"per_node_ms: {per_node_ms}")

    fwd_rt_by_block = defaultdict(float)
    bwd_rt_by_block = defaultdict(float)
    total_fwd_time = 0
    total_bwd_time = 0

    for node in nodes:
        if _is_backward_node(node):
            continue

        _, has_bwd = _get_storage_chain(node)
        _is_act = is_act(node)

        layer = _get_layer_id(node)

        if layer == -1 and not (
            "loss" in _get_module_fqn(node) or "lm_head" in _get_module_fqn(node)
        ):
            # only consider layer 0 ... N and loss + ops without layer tags: inner attention in backward
            continue
        if 0 <= layer <= last_layer_id:
            total_fwd_time += per_node_ms.get(node.name, 0.0)
        # if 0 <= layer <= last_layer_id:  # layer 0 ... N
        forward_ops_per_layer[layer].append(node)

        if is_wait_tensor_from_fsdp(node) or is_pre_bucket_all_gather(node):
            unsharded_parameter_size_per_layer[layer] += get_size_node(node)
        elif _is_act:
            activation_ops_per_layer[layer].append(node)
            activation_size_per_layer[layer] += get_size_node(node)

        # else:  # loss + ops without layer tags: inner attention in backward

    for node in nodes:
        if not _is_backward_node(node):
            continue
        layer = _get_layer_id(node)
        if 0 <= layer <= last_layer_id:
            backward_ops_per_layer[layer].append(node)
            total_bwd_time += per_node_ms.get(node.name, 0.0)
        else:
            if _get_layer_id(node) == -1 and _get_module_fqn(node) != "":
                backward_ops_per_layer[-1].append(node)

    logger.info(f"forward_ops_per_layer: {forward_ops_per_layer}")
    logger.info(f"backward_ops_per_layer: {backward_ops_per_layer}")
    logger.info(f"activation_ops_per_layer: {activation_ops_per_layer}")
    logger.info(f"activation_size_per_layer: {activation_size_per_layer}")
    logger.info(
        f"unsharded_parameter_size_per_layer: {unsharded_parameter_size_per_layer}"
    )
    logger.info(f"persistent_parameter_size: {persistent_parameter_size}")

    total_act_size = 0
    for key, act_size in activation_size_per_layer.items():
        total_act_size += act_size

    logger.info(f"total_act_size: {total_act_size / (1 << 30):.2f}")

    true_activations_per_layer = defaultdict(list)

    def get_true_parent(node):
        all_inputs = node.all_input_nodes
        if len(all_inputs) != 1 or not _is_view(node) :  # not new_storage(node):
            return node
        input_node = all_inputs[0]
        return get_true_parent(input_node)

    # find the candidate activations for each layer:
    for layer, potential_activations in activation_ops_per_layer.items():
        seen = set()
        for node in potential_activations:
            parent_node = get_true_parent(node)
            if "silu" in node.name:
                logger.info(f"node: {node} parent_node: {parent_node}")
            if parent_node not in seen: # and not (is_wait_tensor_from_fsdp(parent_node) or is_pre_bucket_all_gather(parent_node)):
                true_activations_per_layer[layer].append(parent_node)
                seen.add(parent_node)

    logger.info(f"true_activations_per_layer: {true_activations_per_layer}")

    true_activations_size_per_layer = defaultdict(int)
    true_act_size = 0
    for layer, activations in true_activations_per_layer.items():
        for act in activations:
            true_activations_size_per_layer[layer] += get_size_node(act)
            true_act_size += get_size_node(act)

    logger.info(f"true_activations_size_per_layer: {true_activations_size_per_layer}")
    logger.info(f"true_act_size: {true_act_size / (1 << 30):.2f}")

    all_true_activations = []

    must_save = []
    offloadable_candidates = defaultdict(list)
    recomputable_candidates = defaultdict(list)

    bw = get_transfer_bw()
    bw_d2h, bw_h2d = bw["d2h"] * 1e6, bw["h2d"] * 1e6
    total_allowed_offload_budget = min(bw_d2h*total_fwd_time, cpu_offload_budget_gb) * (1 << 30)
    logger.info(f"total_allowed_offload_budget: {total_allowed_offload_budget}")

    decisions_per_act = defaultdict(int) # 0: no decision, 1: save, 2: offload, 3: recompute
    for node in get_must_keep_list(gm):
        decisions_per_act[node] = 1
        must_save.append(node)

    for layer, activations in true_activations_per_layer.items():
        if layer != -1:
            for node in activations:
                all_true_activations.append(node)


    # sort in descending order for per_node_ms[node]/get_size_node(node)
    sizes = {n: get_size_node(n) for n in all_true_activations}
    ranked_activations = sorted(
        (n for n in all_true_activations if sizes[n] > 0),
        key=lambda n: (-per_node_ms.get(n.name, 0.0) / sizes[n], n.name),
    )

    for node in ranked_activations:
        size = get_size_node(node) / (1 << 30)
        ratio = per_node_ms.get(node, 0.0)/size
        logger.info(f"node: {node}, per_node_ms.get(node.name, 0.0): {per_node_ms.get(node.name, 0.0)} size: {size:.2f} per_node_ms[node]/get_size_node(node): {ratio:.2f}")

    loss_peak = loss_region_peak(gm) + persistent_parameter_size + total_act_size - activation_size_per_layer.get(-1, 0)
    logger.info(f"loss_region_peak(gm): {loss_region_peak(gm) / (1 << 30):.2f}")
    logger.info(f"persistent_parameter_size: {persistent_parameter_size / (1 << 30):.2f}")
    logger.info(f"total_act_size: {total_act_size / (1 << 30):.2f}")
    logger.info(f"activation_size_per_layer.get(-1, 0): {activation_size_per_layer.get(-1, 0) / (1 << 30):.2f}")

    offloaded_act = 0
    index = 0
    while index < len(ranked_activations) and offloaded_act < total_allowed_offload_budget and loss_peak > requested_budget:
        act = ranked_activations[index]
        if _can_offload_node(act) and decisions_per_act[act] == 0:
            decisions_per_act[act] = 2
            loss_peak -= sizes[act]
            total_allowed_offload_budget -= sizes[act]
            offloaded_act += sizes[act]

        index += 1

    recomputed_act = 0
    index = 0
    while index < len(ranked_activations) and loss_peak > requested_budget:
        act = ranked_activations[index]
        if decisions_per_act[act] == 0:
            decisions_per_act[act] = 3
            loss_peak -= sizes[act]

            recomputed_act += sizes[act]

        index += 1

    logger.info(f"requested_budget: {requested_budget}")
    logger.info(f"decisions_per_act: {decisions_per_act}")
    logger.info(f"offloaded activations: {offloaded_act/(1 << 30):.2f}")
    logger.info(f"recomputed activations: {recomputed_act/(1 << 30):.2f}")


    chain_owner = {}
    for node in decisions_per_act.keys():
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
                    or up in decisions_per_act
                ):
                    continue

                if not (
                    _is_collective_or_wait(up)
                    or _is_view(up)
                    or up.target
                    is torch.ops.bucketing._pre_bucket_all_gather.default
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

        if node in decisions_per_act:
            tag = decisions_per_act[node]
            node.meta["recompute"] = _POLICY_TAG[tag]
            continue

        upstream_collective = chain_owner.get(node, None)
        if (
            upstream_collective is not None
            and upstream_collective in decisions_per_act
        ):
            tag = decisions_per_act[upstream_collective]
            node.meta["recompute"] = _POLICY_TAG[tag]
            continue

        # views, getitems and interior plumbing: always recomputable. Rebuilding a
        # view is free, and the walk stops at the base, which carries a real tag.
        node.meta["recompute"] = CheckpointPolicy.MUST_RECOMPUTE


    logger.info(f"must_save: {must_save}")

    return gm


def greedy_solve(
    gm: fx.GraphModule | fx.Graph,
    runtime_estimation_mode,
    cpu_offload_budget_gb: int,
    requested_budget: int,
    optimizer,
    model_parts: list[torch.nn.Module],
    num_state_inputs: int,
    trace: TracedResult,
    interp_ctx: tuple | None = None,  # (model, *run_args) for INTERPRETER mode
    is_releasable: Callable[[fx.Node], bool] | None = None,
    device_filter: Callable[[torch.device], bool] | None = None,
):

    gm = per_node_memory(
                gm=gm,
                runtime_estimation_mode=runtime_estimation_mode,
                cpu_offload_budget_gb=cpu_offload_budget_gb,
                requested_budget=requested_budget,
                optimizer=optimizer,
                model_parts=model_parts,
                num_state_inputs=num_state_inputs,
                trace=trace,
                interp_ctx=interp_ctx,  # (model, *run_args) for INTERPRETER mode
                is_releasable=is_releasable,
                device_filter=device_filter,
            )

    if gm is not None:
        _sync_plan_from_rank0(gm)

    return gm

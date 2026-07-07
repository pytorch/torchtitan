# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Static peak-memory estimator for a joint fwd+loss+bwd FX graph.

``estimate_peak_memory`` sweeps the nodes of the joint graph produced by
``minimal_fx_tracer``, tracks per-storage liveness (birth/death over the node
schedule), and reports the peak as the maximum simultaneous live bytes -- plus a
per-category breakdown (parameter / activation / gradient / temporary) of what is
live at that peak.

Liveness is storage-keyed: views/aliases that share an ``untyped_storage`` are
counted once. Parameters and buffers enter as the leading "state" placeholders;
they are pinned live for the whole step because the real allocator keeps them
resident even though the functionalized graph references each only once (a single
bf16 ``_to_copy`` cast). Optimizer state is not traced into this graph and is
accounted separately by ``optimizer_state_bytes``.
"""

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.utils._pytree as pytree
from torch.fx.node import map_arg
from torchtitan.experiments.graph_trainer.common_utils import _is_backward_node
from torchtitan.experiments.graph_trainer.cpu_offload import _is_view
from torchtitan.experiments.graph_trainer.runtime_estimator import _is_costable_op

if TYPE_CHECKING:
    from torchtitan.components.optimizer import OptimizersContainer

ROUNDING = 512  # CUDA caching allocator rounds small allocations to 512B.

# Base optimizer states kept per parameter, by optimizer name (Adam/AdamW keep
# exp_avg + exp_avg_sq; amsgrad adds a third, max_exp_avg_sq -- handled below).
STATES_PER_PARAM = {"Adam": 2, "AdamW": 2}

# ---- storage categories ----
PARAM = "parameter"
GRAD = "gradient"
OPT = "optimizer_state"
ACT = "activation"  # forward intermediate that survives into backward
TEMP = "temporary"  # intermediate freed within its own (fwd or bwd) region
INPUT = "input"
BUFFER = "buffer"


@dataclass
class MemoryEstimatorResult:
    peak_bytes: int
    fwd_peak_bytes: int
    bwd_peak_bytes: int
    peak_node_index: int
    peak_node_name: str
    # category -> bytes live at the global peak point (sums to peak_bytes).
    per_category_at_peak: dict
    # category -> each category's own lifetime maximum (sum != peak_bytes).
    per_category_independent_peak: dict
    # category -> sum of bytes ever attributed to it (for sanity / debugging).
    category_totals: dict
    # storage key -> category (for debugging).
    storage_category: dict
    # live tensor bytes across the whole graph at each node index key: index, val: bytes
    live_bytes: dict
    # (size, birth, last_forward_use, death) for ilp solver
    all_tensors: dict
    # this is a dict for each time index -> live tensors
    live_bytes_per_cat: dict

    def summary(self) -> str:
        """Human-readable report: peak memory, the schedule point (node index +
        name) where it occurs, and the per-category breakdown live at that point."""
        lines = [
            f"peak memory:    {self.peak_bytes / 1e9:.3f} GB",
            f"schedule point: node {self.peak_node_index} ({self.peak_node_name})",
            "per-category at peak:",
        ]
        for cat, nbytes in sorted(
            self.per_category_at_peak.items(), key=lambda kv: -kv[1]
        ):
            lines.append(f"    {cat:16s} {nbytes / 1e9:7.3f} GB")
        return "\n".join(lines)


def find_meaningful_last_fwd_use(node: torch.fx.Node, indeces_of_nodes):
    memo = {}
    lmf = _find_meaningful_last_fwd_use(node, indeces_of_nodes, memo)
    return lmf if lmf >= 0 else indeces_of_nodes[node]


def _find_meaningful_last_fwd_use(node: torch.fx.Node, indeces_of_nodes, memo) -> None:
    """Find the last fwd node that meaningfully uses given node's result.
    Meaningful: not a view, not a no-op cast, not a no-op copy, transpose, etc.
    One way to find this is to use `_is_costable_op` from runtime_estimator.py
    """

    def is_meaningful(node: torch.fx.Node) -> bool:
        # A node "meaningfully" uses its input only if it is a real compute
        # consumer. Views/reshapes/transposes (aliasing, no new allocation) and
        # non-costable bookkeeping ops (getitem, the IGNORE_OPS set) do not pin
        # the input, so we recurse through them. Casts (_to_copy, to.dtype, ...)
        # allocate new storage and are treated as meaningful -- not recursing
        # through them only overshoots the last forward use, which is safe
        # (keeps the activation live slightly longer / offload gap smaller).
        if not _is_costable_op(node) or _is_view(node):
            return False
        return True

    if node in memo:
        return memo[node]
    best = -1  # indeces_of_nodes[node]
    for user in node.users:
        if _is_backward_node(user):
            continue
        elif not is_meaningful(user):
            best = max(
                best, _find_meaningful_last_fwd_use(user, indeces_of_nodes, memo)
            )
        else:  # meaningful user
            best = max(best, indeces_of_nodes[user])  #
    memo[node] = best
    return best


def estimate_peak_memory(
    gm: torch.fx.GraphModule,
    *,
    num_state_inputs: int,
    verbose: bool = False,
) -> MemoryEstimatorResult:
    """Estimate the peak memory of a joint fwd+loss+bwd FX graph.

    Algorithm: for each storage (keyed by ``untyped_storage()._cdata``) compute a
    birth (first node that produces it) and death (last node that uses it), then
    sweep the schedule summing live bytes; the peak is the maximum.

    Categorization (by producer + lifetime):
      - placeholder/get_attr pinned to the end (state) -> ``parameter``
      - backward producer surviving to a graph output  -> ``gradient``
      - backward producer dying within backward        -> ``temporary``
      - forward producer whose last use is backward     -> ``activation``
      - forward producer whose last use is forward      -> ``temporary``

    ``num_state_inputs`` (``TracedResult.num_static_inputs``) is how many leading
    placeholders are persistent state (parameters/buffers); they are pinned live
    for the whole step.
    """
    nodes = list(gm.graph.nodes)
    index = {n: i for i, n in enumerate(nodes)}
    val_of = lambda n: n.meta.get("val", None)  # noqa: E731
    end = len(nodes)  # "live to the end" sentinel for resident/returned storages

    def get_size(t: torch.Tensor) -> int:
        # Count the underlying STORAGE, not the (possibly view) tensor's logical
        # size: a narrow/slice/as_strided view has small numel but can reference a
        # much larger storage. untyped_storage().nbytes() is the real allocation.
        return math.ceil(t.untyped_storage().nbytes() / ROUNDING) * ROUNDING

    # Tensors returned to the caller (loss, grads) must live to the end.
    output_inputs = set()
    for node in nodes:
        if node.op == "output":
            output_inputs.update(node.all_input_nodes)

    # Parameters/buffers are the leading num_state_inputs placeholders; they stay
    # resident on the GPU for the whole step even though the graph references each
    # only once, so pin them live for the entire graph.
    placeholders = [n for n in nodes if n.op == "placeholder"]
    persistent_state = set(placeholders[:num_state_inputs])

    # ---- storage-keyed birth / death / size ----
    # Key by (storage_id, birth_index) so a storage id that is freed and later
    # reused starts a fresh interval instead of merging into one long lifetime.
    birth, death, size, producer_of = {}, {}, {}, {}
    live_key = {}
    last_fwd_use, first_bwd_use = {}, {}
    for i, node in enumerate(nodes):
        # node.meta["val"] holds plain tensors -- minimal_fx_tracer unwraps tensor
        # subclasses (e.g. DTensor) for tracing, so untyped_storage() is valid. For
        # a subclass-carrying graph, use get_untyped_storages from
        # torch.distributed._tools.common_utils instead.
        for t in pytree.tree_leaves(node.meta.get("val")):
            if not isinstance(t, torch.Tensor):
                continue
            # This estimates the GPU peak; CPU-resident tensors (e.g. the pinned
            # host copies produced by ao.offload after apply_cpu_offload_pass) live
            # in host DRAM, not GPU memory. Counting them would inflate the GPU peak
            # by the offloaded byte total, which is exactly the offload savings.
            if t.device.type != "cuda":
                continue
            sid = t.untyped_storage()._cdata

            key = live_key.get(sid)
            if key is None or death.get(key, -1) < i:  # new sid, or it died already
                key = (sid, i)
                live_key[sid] = key
                birth[key] = i
                size[key] = get_size(t)
                producer_of[key] = node

            # Last use: the largest index among users that actually read THIS
            # storage (a multi-output user may consume only some outputs).
            d = i
            for u in node.users:
                u_inputs = pytree.tree_leaves(
                    (map_arg(u.args, val_of), map_arg(u.kwargs, val_of))
                )
                # for u_t in u_inputs:
                #     if (
                #         isinstance(u_t, torch.Tensor)
                #         and u_t.untyped_storage()._cdata == sid
                #     ):
                #         d = max(d, index[u])
                reads_sid = any(  # sid & key both in scope
                    isinstance(u_t, torch.Tensor)
                    and u_t.untyped_storage()._cdata == sid
                    for u_t in u_inputs
                )
                if not reads_sid:
                    continue
                ui = index[u]
                d = max(d, ui)
                if _is_backward_node(u):
                    first_bwd_use[key] = min(first_bwd_use.get(key, ui), ui)
                else:
                    last_fwd_use[key] = max(last_fwd_use.get(key, ui), ui)
            if node in output_inputs or node in persistent_state:
                d = end  # resident params/buffers and returned tensors live to end
            death[key] = max(death.get(key, i), d)

    # ---- categorize each storage from its producer + lifetime ----
    category = {}

    for key in size:
        (sid, i) = key
        prod = producer_of[key]
        # last_forward_use = max(
        #     [index[user] for user in prod.users if not _is_backward_node(user)]
        # )
        # node_details[node] = (size[key], i, last_forward_use, death[key])
        if prod.op in ("placeholder", "get_attr"):
            if prod in persistent_state or prod.op == "get_attr":
                category[key] = PARAM
            elif "tangent" in prod.name:
                category[key] = GRAD  # gradient seed
            else:
                category[key] = INPUT
        elif _is_backward_node(prod):
            category[key] = GRAD if death[key] >= end else TEMP
        else:  # forward-produced compute
            last = nodes[min(death[key], end - 1)]
            category[key] = ACT if _is_backward_node(last) else TEMP

    # (size, birth, last_forward_use, death) for ilp solver
    all_tensors = {}
    _lfu_memo: dict = {}

    def _refined_lfu(prod: torch.fx.Node) -> int:
        lmf = _find_meaningful_last_fwd_use(
            prod, index, _lfu_memo
        )  # internal fn + shared memo
        return lmf if lmf >= 0 else index[prod]  # birth fallback here

    for key in size:
        # if key in first_bwd_use and not _is_backward_node(
        #     producer_of[key]
        # ):  # fwd op read in bwd
        v = producer_of[key]
        all_tensors.setdefault(v, []).append(
            {
                "sid": key[0],
                "category": category[key],
                "size": size[key],
                "birth": birth[key],  # the first index produced this sid
                "last_fwd_use": last_fwd_use.get(
                    key, birth[key]
                ),  # _refined_lfu(v),  
                "first_bwd_use": first_bwd_use.get(key, None),
                # gap = [last_fwd_use, first_bwd_use)
                "death": death[key],  # the last index used this sid
                "producer": v,  # the node that produced this sid
            }
        )

    # ---- O(n) sweep: peak, fwd/bwd peak, at-peak and per-category maxima ----
    add_at, free_at = defaultdict(list), defaultdict(list)
    for key in size:
        add_at[birth[key]].append(key)
        free_at[death[key]].append(key)

    per_cat_live = Counter()
    per_cat_at_peak, per_cat_independent = {}, {}
    current = peak = peak_idx = fwd_peak = bwd_peak = 0
    # live bytes at each node index
    live_bytes = {}
    live_bytes_per_cat_curent = {}
    live_bytes_per_cat = {}
    for i in range(end + 1):  # +1 so end-of-graph frees are processed
        # key mean (sid, i)

        for key in add_at.get(i, ()):
            # live_bytes_dict_of_nodes_curent[cat] += size[key]
            current += size[key]
            per_cat_live[category[key]] += size[key]
            if i < end:
                if _is_backward_node(nodes[i]):
                    bwd_peak = max(bwd_peak, current)
                else:
                    fwd_peak = max(fwd_peak, current)
        if current > peak:
            peak, peak_idx = current, i
            per_cat_at_peak = dict(per_cat_live)

        live_bytes[i] = current
        live_bytes_per_cat[i] = dict(per_cat_live)

        for cat, live in per_cat_live.items():
            per_cat_independent[cat] = max(per_cat_independent.get(cat, 0), live)
        for key in free_at.get(i, ()):
            current -= size[key]
            per_cat_live[category[key]] -= size[key]

    category_totals = Counter()
    for key in size:
        category_totals[category[key]] += size[key]

    result = MemoryEstimatorResult(
        peak_bytes=peak,
        fwd_peak_bytes=fwd_peak,
        bwd_peak_bytes=bwd_peak,
        peak_node_index=peak_idx,
        peak_node_name=nodes[peak_idx].name,
        per_category_at_peak=per_cat_at_peak,
        per_category_independent_peak=per_cat_independent,
        category_totals=dict(category_totals),
        storage_category=category,
        live_bytes=live_bytes,
        all_tensors=all_tensors,
        live_bytes_per_cat=live_bytes_per_cat,
    )

    if verbose:
        print(result.summary())
    return result


def optimizer_state_bytes(
    opt_config: "OptimizersContainer.Config | None", model: torch.nn.Module
) -> int:
    """Persistent optimizer-state bytes, analytically from the optimizer config
    (no optimizer instance / no training step needed). Adam/AdamW keep 2 state
    tensors per param; states are fp32 unless the config requests bf16.
    Returns 0 when there is no optimizer (e.g. the fwd/bwd-only test path)."""
    if opt_config is None or not opt_config.param_groups:
        return 0

    # States-per-param per group: 2 for Adam/AdamW, +1 if amsgrad keeps
    # max_exp_avg_sq. Take the max across groups -- exact for a uniform config;
    # a mix of different per-group state counts would need per-group param
    # attribution (precise FQN->group matching), which we don't do here.
    def _states_per_param(pg) -> int:
        n = STATES_PER_PARAM.get(pg.optimizer_name, 0)
        if n and (getattr(pg, "optimizer_kwargs", None) or {}).get("amsgrad", False):
            n += 1
        return n

    n_states = max((_states_per_param(pg) for pg in opt_config.param_groups), default=0)
    if n_states == 0:
        return 0

    # bf16 optimizer states only under the fused_opt_states_bf16 implementation
    # (Adam/AdamW momentum+variance in bf16); otherwise states are fp32.
    state_dtype = (
        torch.bfloat16
        if opt_config.implementation == "fused_opt_states_bf16"
        else torch.float32
    )
    elt_bytes = torch.finfo(state_dtype).bits // 8
    num_params = sum(p.numel() for p in model.parameters())
    return num_params * n_states * elt_bytes

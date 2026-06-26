# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Static roofline runtime estimator for a joint fwd+loss+bwd FX graph.

This is the runtime analogue of ``memory_estimator.py``. Where the memory
estimator sweeps the graph nodes and tracks storage liveness, this one sweeps
the same nodes and sums a per-op roofline time estimate.

Relation to ``torch.distributed._tools.runtime_estimator.RuntimeEstimator``:
that class is a ``TorchDispatchMode`` -- it intercepts aten ops *as they
execute* (eager) under ``FakeTensorMode``. We do not need that machinery: the
joint graph from ``minimal_fx_tracer`` already materializes every op as a node
carrying its target, its args/kwargs, and its fake output in ``node.meta['val']``
-- exactly the inputs ``__torch_dispatch__`` would see. So we walk the graph
statically and reuse the *same* roofline cost-model helpers the dispatch
estimator uses (``torch.utils._runtime_estimation``).

The roofline model estimates each op as ``max(compute_time, transfer_time)``:
  - compute_time  = op_flops / peak_flops      (compute bound)
  - transfer_time = op_bytes / dram_bandwidth  (memory bound)

``estimate_runtime_original`` is the per-node sum of the graph *as written*.

Limitations (shared with the upstream cost model):
  1. Communication is not modeled. Collectives (all_gather/reduce_scatter/...)
     are not in the flop registry, so this is a compute-only estimate.
  2. Ops compiled into opaque kernels are not costed. aten ops (incl. aten SDPA)
     and the ``flex_attention`` HOP are costed, but once ``regional_inductor``
     fuses a region (e.g. flex_attention) into a single kernel the flops are no
     longer visible to this estimator.
  3. Roofline ignores kernel launch overhead, occupancy, and overlap; it is an
     analytical estimate, not a benchmark.
"""

import operator
from dataclasses import dataclass, field

import torch
import torch.utils._pytree as pytree
from torch.fx.node import map_arg
from torch.utils._runtime_estimation import (
    _FLOAT_TYPES,
    _IGNORE_OPS,
    get_compute_time,
    get_transfer_time,
)
from torch.utils.flop_counter import flop_registry

from torchtitan.experiments.graph_trainer.common_utils import (
    _is_backward_node as _is_bwd_node,
)


@dataclass
class RuntimeEstimatorResult:
    """Per-node roofline runtime of the graph as written."""

    total_runtime_ms: float
    fwd_runtime_ms: float
    bwd_runtime_ms: float
    # node name -> estimated time in ms (only ops that were costed)
    node_runtimes_ms: dict = field(default_factory=dict)

    def summary(self, top_k: int = 10) -> str:
        lines = [
            f"runtime (graph as written): {self.total_runtime_ms:.3f} ms "
            f"(fwd {self.fwd_runtime_ms:.3f} ms, bwd {self.bwd_runtime_ms:.3f} ms)",
            # Attention (flex_attention HOP / aten SDPA) IS costed. Collectives
            # are not, so treat this as a compute-only estimate (no comm/overlap).
            "NOTE: collectives are not costed (compute-only estimate).",
            f"top {top_k} ops by estimated time:",
        ]
        for name, t in sorted(self.node_runtimes_ms.items(), key=lambda kv: -kv[1])[
            :top_k
        ]:
            lines.append(f"    {t:8.4f} ms  {name}")
        return "\n".join(lines)


def _is_costable_op(node: torch.fx.Node) -> bool:
    """A node the roofline model applies to: a call to an aten ``OpOverload``
    that is not a pure view/create op, or a HigherOrderOperator the flop registry
    knows (e.g. ``flex_attention``)."""
    if node.op != "call_function":
        return False
    target = node.target
    # operator.getitem (unpacking multi-output ops) and python builtins carry no
    # kernel cost.
    if target is operator.getitem:
        return False
    if isinstance(target, torch._ops.OpOverload):
        return target._overloadpacket not in _IGNORE_OPS
    # HigherOrderOperators (flex_attention / flex_attention_backward) are not
    # OpOverloads and have no _overloadpacket, but the flop registry keys them by
    # the HOP object itself. Cost them when the registry knows them.
    return target in flop_registry


def _op_time_ms(node: torch.fx.Node) -> float:
    """Roofline time for a single costable node, in milliseconds.

    Reconstructs the (args, kwargs, out) the cost model expects by substituting
    each input ``Node`` with its fake value (``meta['val']``); non-tensor args
    pass through unchanged. Mirrors the upstream dispatch-mode estimator, but
    reads everything from the node instead of a live op invocation.
    """
    val_of = lambda n: n.meta.get("val", None)  # noqa: E731
    args = map_arg(node.args, val_of)
    kwargs = map_arg(node.kwargs, val_of)
    out = node.meta.get("val", None)

    flat_args_kwargs = pytree.tree_leaves((args, kwargs))
    flat_outs = pytree.tree_leaves(out)

    transfer_time = get_transfer_time(flat_args_kwargs, flat_outs)

    # OpOverloads are keyed in the flop registry by their overload packet;
    # HigherOrderOperators (flex_attention) are keyed by the HOP object itself.
    target = node.target
    func_packet = (
        target._overloadpacket if isinstance(target, torch._ops.OpOverload) else target
    )
    # get_compute_time returns 0.0 for ops not in the flop registry; for the ones
    # that are, it asserts exactly one output dtype. Multi-output flop ops (the
    # SDPA family and flex_attention return out=bf16 + logsumexp=fp32) would
    # otherwise trip that assert. The compute dtype is the primary output's dtype,
    # not the fp32 auxiliary stats, so pick the first float output and cost against
    # that. Pass a fresh single-element set because get_compute_time pops from it,
    # and guard the call so one odd node falls back to transfer-only.
    primary_dtype = next(
        (
            t.dtype
            for t in flat_outs
            if isinstance(t, torch.Tensor) and t.dtype in _FLOAT_TYPES
        ),
        None,
    )
    compute_time = 0.0
    if primary_dtype is not None:
        try:
            compute_time = get_compute_time(
                func_packet, args, kwargs, out, {primary_dtype}, node_meta=node.meta
            )
        except Exception:
            compute_time = 0.0

    # transfer/compute are in nanoseconds; convert to ms.
    return max(transfer_time, compute_time) / 1e6


def estimate_runtime_original(
    gm: torch.fx.GraphModule, *, verbose: bool = False
) -> RuntimeEstimatorResult:
    """Roofline runtime of the joint fwd+loss+bwd graph as written.

    Run on the pre-pass graph so ``flex_attention`` is still a costable HOP.
    Returns the total and the forward/backward split, plus a per-node breakdown.
    """
    total = fwd = bwd = 0.0
    node_runtimes_ms: dict = {}

    for node in gm.graph.nodes:
        if not _is_costable_op(node):
            continue
        t = _op_time_ms(node)
        if t <= 0.0:
            continue
        node_runtimes_ms[node.name] = t
        total += t
        if _is_bwd_node(node):
            bwd += t
        else:
            fwd += t

    result = RuntimeEstimatorResult(
        total_runtime_ms=total,
        fwd_runtime_ms=fwd,
        bwd_runtime_ms=bwd,
        node_runtimes_ms=node_runtimes_ms,
    )
    if verbose:
        print(result.summary())
    return result

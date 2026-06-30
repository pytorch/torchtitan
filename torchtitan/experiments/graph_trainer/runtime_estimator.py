# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Static / measured runtime estimator for a joint fwd+loss+bwd FX graph.

This mirrors the API/structure of
``torch.distributed._tools.runtime_estimator.RuntimeEstimator`` -- the mode is
selected via ``__call__``:

  - ``operator-level-cost-model``: a roofline estimate, ``max(compute, HBM transfer)``
    using the same ``torch.utils._runtime_estimation`` helpers the upstream
    estimator uses. Static (no GPU execution).
  - ``operator-level-benchmark``: runs and times each op on real randomly-initialized
   tensors (warmup + timed iters, CUDA events) -- the same per-op micro-benchmark idea
   as upstream's ``_maybe_run_and_benchmark_fallback_kernel``. Each op is benchmarked in
    isolation, so it works even for a model too big to run end-to-end.
  - ``operator-level-interpreter``: executes the *whole* real traced graph via
    ``run_traced`` and times it -- the real end-to-end total (overlap-aware) plus
    a per-node CUDA-event breakdown from an ``fx.Interpreter``. Needs the live
    module and real run inputs, and cudagraph disabled.

It also aggregates per-module forward/backward runtimes and records module
execution order, exposed via ``display_modulewise_stats``.

Difference from upstream: upstream is a ``TorchDispatchMode`` run under
``FakeTensorMode`` that intercepts aten ops as an eager model executes. The
cost-model/benchmark modes here walk the joint graph statically (per FX node,
the granularity our save/recompute/offload solver needs), reading each node's
fake output (``node.meta['val']``) -- exactly the inputs ``__torch_dispatch__``
would see -- and reuse the same cost-model helpers.

Limitations (shared with the upstream cost model):
  1. Communication/collectives are not modeled (compute-only estimate).
  2. Like upstream, we cost every op and eliminate only _IGNORE_OPS (plus
     getitem/structural nodes via _is_costable_op). An op the flop registry
     doesn't know is NOT dropped to zero — its transfer_time is always counted;
     only its compute_time falls back to 0. So a fused/opaque op (e.g.
     flex_attention after regional_inductor) still contributes its tensor I/O,
     just not its hidden compute.
  3. Roofline ignores launch overhead, occupancy, and overlap. Benchmark and
     interpreter modes capture real kernel time; the interpreter total is
     overlap-aware, but its per-node breakdown is serialized.

Note: The interpreter mode is the only one that can capture the real end-to-end.
"""

import operator
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import median

import torch
import torch.utils._pytree as pytree
from torch.fx.node import map_arg
from torch.utils._runtime_estimation import (
    _FLOAT_TYPES,
    _IGNORE_OPS,
    _VIEW_OPS,
    get_compute_time,
    get_transfer_time,
)
from typing_extensions import Self

from torchtitan.experiments.graph_trainer.common_utils import (
    _is_backward_node,
    _MODULE_FQN,
)
from torchtitan.experiments.graph_trainer.make_fx_tracer import run_traced, TracedResult

# Estimate modes, named to match the upstream RuntimeEstimator.
COST_MODEL = "operator-level-cost-model"
BENCHMARK = "operator-level-benchmark"
INTERPRETER = "operator-level-interpreter"


@dataclass
class RuntimeEstimatorResult:
    """Per-node + per-module runtime of the graph as written."""

    total_runtime_ms: float
    fwd_runtime_ms: float
    bwd_runtime_ms: float
    estimate_mode_type: str = COST_MODEL
    # node name -> estimated/measured time in ms
    node_runtimes_ms: dict = field(default_factory=dict)
    # module fqn -> {"fw": ms, "bw": ms}
    mod_runtimes_ms: dict = field(default_factory=dict)

    def summary(self, top_k: int = 10) -> str:
        lines = [
            f"runtime ({self.estimate_mode_type}): {self.total_runtime_ms:.3f} ms "
            f"(fwd {self.fwd_runtime_ms:.3f} ms, bwd {self.bwd_runtime_ms:.3f} ms)",
            # Attention (flex_attention HOP / aten SDPA) IS costed. Collectives
            # are not, so treat this as a compute-only estimate (no comm/overlap).
            "NOTE: collectives are not costed (compute-only estimate).",
            f"top {top_k} ops by time:",
        ]
        for name, t in sorted(self.node_runtimes_ms.items(), key=lambda kv: -kv[1])[
            :top_k
        ]:
            lines.append(f"    {t:8.4f} ms  {name}")
        return "\n".join(lines)


class _TimingInterpreter(torch.fx.Interpreter):
    """Times each compute node with CUDA events (serialized; per-op cost table)."""

    def __init__(self, gm: torch.fx.GraphModule):
        super().__init__(gm)
        self.events: dict = {}
        # self._ignored_time = 0.0

    def run_node(self, n):
        if n.op in ("placeholder", "output", "get_attr"):
            return super().run_node(n)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        out = super().run_node(n)
        # torch.cuda.synchronize()
        end.record()
        self.events[n.name] = (n, start, end)
        return out

    def timings_ms(self) -> dict:
        # One sync, THEN read every elapsed_time (events must be completed first).
        torch.cuda.synchronize()

        # for name, (_, s, e) in self.events.items():
        #     if (
        #         "squeeze" in name
        #         or "transpose" in name
        #         or "expand" in name
        #         or "t_" in name
        #         or "split" in name
        #         or "zeros_like" in name
        #         or "ones_like" in name
        #     ):
        #         self._ignored_time += s.elapsed_time(e)
        # print(f"ignored time: {self._ignored_time} ms")
        return {name: s.elapsed_time(e) for name, (_, s, e) in self.events.items()}


def _is_costable_op(node: torch.fx.Node) -> bool:
    """Whether to apply the roofline to this node.

    Mirrors the upstream RuntimeEstimator: cost *every* op, eliminating only
    ``_IGNORE_OPS`` (and ``getitem``, which just unpacks multi-output ops). We do
    NOT drop an op to zero merely because it is "uncostable": ``_roofline_estimate``
    always counts ``transfer_time`` from the node's tensors, and ``get_compute_time``
    falls back to 0 for targets the flop registry doesn't know. So an unknown op
    (a non-registry HOP, a fused inductor kernel, a custom op) still contributes
    its memory-transfer cost rather than nothing. Known compute ops (aten GEMMs,
    the ``flex_attention`` HOP pre-fusion, etc.) additionally get ``compute_time``.
    """
    if node.op != "call_function":
        return False
    target = node.target
    # operator.getitem just indexes a multi-output op's result; no kernel cost.
    if target is operator.getitem:
        return False
    if isinstance(target, torch._ops.OpOverload):
        return target._overloadpacket not in _IGNORE_OPS
    # HOPs / other call_functions: cost their transfer time (compute falls back
    # to 0 inside get_compute_time if the flop registry doesn't know them).
    return True


def _module_fqn(node: torch.fx.Node) -> str:
    return node.meta.get("custom", {}).get(_MODULE_FQN, "")


def _capture(traced: TracedResult, module, args, build):
    """Run the traced graph with a custom interpreter built by ``build(gm)`` and
    return that interpreter instance (run_traced doesn't expose it otherwise)."""
    holder: dict = {}

    def factory(gm):
        inst = build(gm)
        holder["inst"] = inst
        return inst

    run_traced(traced, module=module, interpreter_cls=factory)(*args)
    return holder["inst"]


class RuntimeEstimator:
    """FX-graph-based runtime estimator mirroring the upstream
    ``torch.distributed._tools.runtime_estimator.RuntimeEstimator`` API.

    Static modes (cost-model / benchmark) need only the graph::

        est = RuntimeEstimator()(estimate_mode_type="operator-level-cost-model")
        result = est.estimate(gm)            # gm or TracedResult

    The interpreter mode runs the real graph, so it needs the live module and
    real inputs (and cudagraph disabled)::

        est = RuntimeEstimator()(estimate_mode_type="operator-level-interpreter")
        result = est.estimate(traced, module, *run_args)

    Then ``est.display_modulewise_stats(depth=2)``.
    """

    def __init__(self, warmup_iters: int = 2, bench_iters: int = 3) -> None:
        self._estimate = self._roofline_estimate
        self._estimate_mode_type = COST_MODEL
        self._warmup_iters = warmup_iters
        self._bench_iters = bench_iters
        # module fqn -> {"fw": ms, "bw": ms}
        self.mod_runtimes: dict[str, dict[str, float]] = defaultdict(
            lambda: defaultdict(float)
        )
        self.mod_fw_order: list[str] = []
        self.mod_bw_order: list[str] = []
        self.total_runtime: float = 0.0
        self.fwd_runtime: float = 0.0
        self.bwd_runtime: float = 0.0

    # ---- mode selection (mirrors upstream __call__) ----
    def __call__(self, estimate_mode_type: str) -> Self:
        if estimate_mode_type == COST_MODEL:
            self._estimate = self._roofline_estimate
        elif estimate_mode_type == BENCHMARK:
            self._estimate = self._benchmark_estimate
        elif estimate_mode_type == INTERPRETER:
            # Interpreter times the whole graph at once, not per node, so there
            # is no per-node estimate function for it.
            self._estimate = None
        else:
            raise NotImplementedError(
                f"estimate_mode_type {estimate_mode_type} not supported. Supported: "
                f"{COST_MODEL}, {BENCHMARK}, {INTERPRETER}"
            )
        self._estimate_mode_type = estimate_mode_type
        return self

    # Adapted from: https://github.com/pytorch/pytorch/blob/main/torch/distributed/_tools/runtime_estimator.py
    # ---- per-node estimate methods (static modes) ----
    def _roofline_estimate(self, node: torch.fx.Node) -> float:
        """Roofline time for a single node, in ms: ``max(compute, transfer)``.

        Reconstructs (args, kwargs, out) the cost model expects by substituting
        each input ``Node`` with its fake value (``meta['val']``)."""

        if not torch.accelerator.is_available():
            raise AssertionError(
                "Roofline estimation needs to access CUDA capabilities to make estimations"
            )

        val_of = lambda n: n.meta.get("val", None)  # noqa: E731
        args = map_arg(node.args, val_of)
        kwargs = map_arg(node.kwargs, val_of)
        out = node.meta.get("val", None)

        flat_args_kwargs = pytree.tree_leaves((args, kwargs))
        flat_outs = pytree.tree_leaves(out)
        transfer_time = get_transfer_time(flat_args_kwargs, flat_outs)

        target = node.target
        func_packet = (
            target._overloadpacket
            if isinstance(target, torch._ops.OpOverload)
            else target
        )

        if func_packet in _IGNORE_OPS:
            # print(f"ignoring {node.name} {func_packet}")
            return 0.0

        # get_compute_time asserts exactly one output dtype; multi-output flop ops
        # (SDPA family, flex_attention return out=bf16 + logsumexp=fp32) would trip
        # it, so cost against the primary (first float) output dtype.
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
        return max(transfer_time, compute_time) / 1e6

    def _benchmark_estimate(self, node: torch.fx.Node) -> float:
        """Real per-op time (ms): materialize the node's inputs as real random
        tensors and time the kernel (warmup + timed iters, CUDA events).

        Mirrors upstream's ``_maybe_run_and_benchmark_fallback_kernel``. Falls
        back to the roofline for HOPs, view/inplace-view ops, on CPU, or when the
        op errors on random inputs."""
        target = node.target
        # HOPs (flex_attention) cannot be micro-benchmarked in isolation.
        if not isinstance(target, torch._ops.OpOverload):
            return self._roofline_estimate(node)
        # Views/create ops are ~free and would not run cleanly on random inputs.
        if target._overloadpacket in _VIEW_OPS:
            return 0.0
        if torch.Tag.inplace_view in getattr(target, "tags", ()):
            return 0.0
        if not torch.cuda.is_available():
            return self._roofline_estimate(node)
        try:
            real_args = map_arg(node.args, self._real_leaf)
            real_kwargs = map_arg(node.kwargs, self._real_leaf)
            target(*real_args, **(real_kwargs or {}))  # correctness + allocate
            for _ in range(self._warmup_iters):
                target(*real_args, **(real_kwargs or {}))
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(self._bench_iters):
                target(*real_args, **(real_kwargs or {}))
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) / self._bench_iters
        except Exception:
            # Random inputs can violate an op's constraints; fall back analytically.
            return self._roofline_estimate(node)

    def _real_leaf(self, n: torch.fx.Node):
        """map_arg callback: real random tensor from an input node's fake value.
        rand for floats, ones otherwise (matches upstream ``to_real_tensor``)."""
        return self._to_real(n.meta.get("val", None))

    @staticmethod
    def _to_real(v):
        if isinstance(v, torch.Tensor):
            shape = tuple(v.shape)
            if v.dtype in _FLOAT_TYPES:
                return torch.rand(shape, dtype=v.dtype, device=v.device)
            return torch.ones(shape, dtype=v.dtype, device=v.device)
        return v

    # ---- entry point: dispatch on the selected mode ----
    def estimate(
        self,
        traced,
        module=None,
        *args,
        warmup: int = 1,
        reps: int = 3,
    ) -> RuntimeEstimatorResult:
        """Run the selected mode.

        Static modes (cost-model / benchmark) accept a ``GraphModule`` or a
        ``TracedResult`` and ignore ``module``/``args``. The interpreter mode
        requires a ``TracedResult``, the live ``module``, and the real run
        ``args`` (the same inputs ``run_traced`` takes).
        """
        if self._estimate_mode_type in (COST_MODEL, BENCHMARK):
            gm = traced.gm if isinstance(traced, TracedResult) else traced
            return self._estimate_static(gm)
        elif self._estimate_mode_type == INTERPRETER:
            if not isinstance(traced, TracedResult) or module is None:
                raise ValueError(
                    "interpreter mode needs a TracedResult and the live module: "
                    "estimate(traced, module, *run_args). Run with cudagraph "
                    "disabled."
                )
            return self._estimate_interpreter(
                traced, module, *args, warmup=warmup, reps=reps
            )
        else:
            raise NotImplementedError(
                f"estimate_mode_type {self._estimate_mode_type} not supported."
            )

    # ---- static run (mirrors upstream __torch_dispatch__ accumulation) ----
    def _estimate_static(self, gm: torch.fx.GraphModule) -> RuntimeEstimatorResult:
        """Walk the graph, time each costable node via the selected per-node
        estimate, and aggregate total / fwd / bwd / per-module runtimes."""
        self.total_runtime = self.fwd_runtime = self.bwd_runtime = 0.0
        self.mod_runtimes = defaultdict(lambda: defaultdict(float))
        self.mod_fw_order, self.mod_bw_order = [], []
        node_runtimes: dict = {}
        seen_fw: set = set()
        seen_bw: set = set()

        for node in gm.graph.nodes:
            if not _is_costable_op(node):
                continue
            t = self._estimate(node)
            if t <= 0.0:
                continue
            node_runtimes[node.name] = t
            self._accumulate(node, t, seen_fw, seen_bw)

        return self._build_result(node_runtimes)

    # ---- interpreter run (real execution of the whole graph) ----
    def _estimate_interpreter(
        self,
        traced: TracedResult,
        module,
        *args,
        warmup: int = 1,
        reps: int = 3,
    ) -> RuntimeEstimatorResult:
        """Execute the real traced graph via ``run_traced`` and measure runtime.

        ``total_runtime_ms`` is the real end-to-end step time (whole-graph timed
        run, overlap-aware, median over reps). The per-node times come from a
        serialized ``fx.Interpreter`` and so do NOT sum to the total. Memory is
        out of scope here -- use ``measure_traced`` for peak memory. Run with
        cudagraph disabled (the per-node interpreter cannot replay a cudagraph).
        """
        gm = traced.gm

        # warmup (tracing/autotune/allocator settling)
        for _ in range(warmup):
            run_traced(traced, module=module)(*args)
        torch.cuda.synchronize()

        # total runtime: time the whole traced run once per rep (kernels overlap
        # as in a real step), median over reps. We do NOT sum per-op times here.
        totals_ms = []
        for _ in range(reps):
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            run_traced(traced, module=module)(*args)
            end.record()
            torch.cuda.synchronize()
            totals_ms.append(start.elapsed_time(end))
        total_runtime_ms = median(totals_ms)

        # per-node times via the timing interpreter (serialized cost table).
        timing = _capture(traced, module, args, _TimingInterpreter)
        per_node = timing.timings_ms()

        self.total_runtime = self.fwd_runtime = self.bwd_runtime = 0.0
        self.mod_runtimes = defaultdict(lambda: defaultdict(float))
        self.mod_fw_order, self.mod_bw_order = [], []
        name_to_node = {n.name: n for n in gm.graph.nodes}
        seen_fw: set = set()
        seen_bw: set = set()
        for nm, t in per_node.items():
            node = name_to_node.get(nm)
            if node is not None:
                self._accumulate(node, t, seen_fw, seen_bw)

        # The total is the real overlap-aware end-to-end time, not the per-op sum.
        self.total_runtime = total_runtime_ms
        return self._build_result(per_node)

    # ---- shared accumulation / result helpers ----
    def _accumulate(
        self, node: torch.fx.Node, t: float, seen_fw: set, seen_bw: set
    ) -> None:
        self.total_runtime += t
        fqn = _module_fqn(node)
        if _is_backward_node(node):
            self.bwd_runtime += t
            self.mod_runtimes[fqn]["bw"] += t
            if fqn and fqn not in seen_bw:
                seen_bw.add(fqn)
                self.mod_bw_order.append(fqn)
        else:
            self.fwd_runtime += t
            self.mod_runtimes[fqn]["fw"] += t
            if fqn and fqn not in seen_fw:
                seen_fw.add(fqn)
                self.mod_fw_order.append(fqn)

    def _build_result(self, node_runtimes: dict) -> RuntimeEstimatorResult:
        return RuntimeEstimatorResult(
            total_runtime_ms=self.total_runtime,
            fwd_runtime_ms=self.fwd_runtime,
            bwd_runtime_ms=self.bwd_runtime,
            estimate_mode_type=self._estimate_mode_type,
            node_runtimes_ms=node_runtimes,
            mod_runtimes_ms={k: dict(v) for k, v in self.mod_runtimes.items()},
        )

    def display_modulewise_stats(self, depth: int = 2) -> None:
        """Print module forward/backward execution orders and per-module
        runtimes, up to ``depth`` levels (mirrors upstream)."""
        print("Forward Execution Order:")
        for fqn in self.mod_fw_order:
            if fqn.count(".") + 1 <= depth:
                print(f"  {fqn}")
        print("Backward Execution Order:")
        for fqn in self.mod_bw_order:
            if fqn.count(".") + 1 <= depth:
                print(f"  {fqn}")
        for fqn, rt in self.mod_runtimes.items():
            if fqn and fqn.count(".") + 1 <= depth:
                print(
                    f"{fqn} fw: {rt.get('fw', 0.0):.3f} ms bw: {rt.get('bw', 0.0):.3f} ms"
                )


def estimate_runtime(
    gm: torch.fx.GraphModule,
    *,
    mode: str = COST_MODEL,
) -> RuntimeEstimatorResult:
    """Convenience wrapper for the static modes: build a ``RuntimeEstimator`` in
    ``mode`` and run it on ``gm``.

    Defaults to the roofline cost model (no GPU execution). Pass
    ``mode="operator-level-benchmark"`` to micro-benchmark each op on real
    tensors. The interpreter mode needs the live module + run inputs, so use
    ``RuntimeEstimator()(INTERPRETER).estimate(traced, module, *args)`` directly.
    """
    if mode == INTERPRETER:
        raise ValueError(
            "interpreter mode needs the live module and run inputs; use "
            "RuntimeEstimator()(INTERPRETER).estimate(traced, module, *args)."
        )
    return RuntimeEstimator()(mode).estimate(gm)


def _op_label(node: torch.fx.Node | None) -> str:
    """Short op-type label for a node (overload packet / function name)."""
    if node is None:
        return "<absent>"
    t = node.target
    if isinstance(t, torch._ops.OpOverload):
        return str(t._overloadpacket)
    return getattr(t, "__name__", str(t))


def compare_node_breakdown(
    cm_result: RuntimeEstimatorResult,
    bm_result: RuntimeEstimatorResult,
    it_result: RuntimeEstimatorResult,
    gm: torch.fx.GraphModule,
    top_k: int = 15,
) -> None:
    """compares the cost-model, benchmark, and interpreter.

    Joins the three modes' per-node times by node name and reports:
      1. the biggest individual missed nodes (and their op type);
      2. misses grouped by op type;
      3. per-op-type interpreter-vs-cost-model totals + cm/it ratio (where the
         roofline under-counts even ops it does cost).

    The post-regional graph: the interpreter reflects the real fused execution.
    The pre-regional: the interpreter runs eager flex -> not representative.
    """
    name_to_node = {n.name: n for n in gm.graph.nodes}
    cmn = cm_result.node_runtimes_ms  # roofline per-node (costable, >0)
    bmn = bm_result.node_runtimes_ms  # benchmark per-node
    itn = it_result.node_runtimes_ms  # interpreter per-node (ALL compute ops)

    print(
        "\n[Per-OP-Comparison] === why cost-model / benchmark underestimate interpreter ==="
    )
    print(
        f"[Per-OP-Comparison] node counts: interpreter={len(itn)}  "
        f"cost-model={len(cmn)}  benchmark={len(bmn)}"
    )

    # 2. biggest individual nodes the interpreter timed but cost-model missed.
    misses = sorted(
        ((nm, t) for nm, t in itn.items() if cmn.get(nm, 0.0) <= 0.0),
        key=lambda kv: -kv[1],
    )
    print(
        f"[Per-OP-Comparison] top {top_k} nodes timed by interpreter but ~0 in cost-model:"
    )
    for nm, t in misses[:top_k]:
        print(f"    {t:8.3f} ms  {nm:40s} {_op_label(name_to_node.get(nm))}")

    # 3. misses grouped by op type.
    miss_by_op: dict = {}
    for nm, t in misses:
        a = miss_by_op.setdefault(_op_label(name_to_node.get(nm)), [0, 0.0])
        a[0] += 1
        a[1] += t
    print("[Per-OP-Comparison] misses grouped by op type (by total interpreter ms):")
    for lbl, (cnt, tot) in sorted(miss_by_op.items(), key=lambda kv: -kv[1][1])[:top_k]:
        print(f"    {tot:8.3f} ms  x{cnt:<5} {lbl}")

    # 4. per-op-type interpreter vs cost-model vs benchmark (shows both the
    #    missed ops and the under-estimated ones).
    by_op: dict = {}
    for nm, t in itn.items():
        lbl = _op_label(name_to_node.get(nm))
        a = by_op.setdefault(lbl, [0, 0.0, 0.0, 0.0])  # cnt, it, cm, bm
        a[0] += 1
        a[1] += t
        a[2] += cmn.get(nm, 0.0)
        a[3] += bmn.get(nm, 0.0)
    print(f"[Per-OP-Comparison] per-op-type (top {top_k} by interpreter ms):")
    print(
        f"    {'op':30s} {'cnt':>4} {'it_ms':>9} {'cm_ms':>9} {'bm_ms':>9} "
        f"{'cm/it':>6} {'bm/it':>6}"
    )
    for lbl, (cnt, itt, cmt, bmt) in sorted(by_op.items(), key=lambda kv: -kv[1][1])[
        :top_k
    ]:
        rc = cmt / itt if itt else float("nan")
        rb = bmt / itt if itt else float("nan")
        print(
            f"    {lbl:30s} {cnt:>4} {itt:>9.3f} {cmt:>9.3f} {bmt:>9.3f} "
            f"{rc:>6.2f} {rb:>6.2f}"
        )

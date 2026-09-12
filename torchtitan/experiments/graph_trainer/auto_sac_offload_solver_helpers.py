# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import time
import warnings
from collections import defaultdict

import torch
import torch.utils._pytree as pytree
from torch.fx.node import map_arg
from torch.utils._runtime_estimation import _FLOAT_TYPES, _VIEW_OPS

from torchtitan.experiments.graph_trainer.auto_sac_offload_solver_utils import (
    _FREE_MEM_FRACTION,
    _is_costable_op,
    _MIN_SATURATING_BYTES,
    _VIEW_OP_NAMES,
    get_size,
    STATES_PER_PARAM,
    UNKNOWN_OPTIMIZER_BYTES,
)
from torchtitan.tools.logging import logger


def estimate_peak_memory(
    gm: torch.fx.GraphModule,
    *,
    num_state_inputs: int,
) -> int:
    """Estimate the peak GPU bytes of a joint fwd+loss+bwd FX graph.

    Gives every storage a birth and a death index, then sweeps the schedule
    summing live bytes and returns the largest total.

    ``num_state_inputs`` is how many leading placeholders are persistent state
    (parameters/buffers), which stay resident for the whole step.
    """
    nodes = list(gm.graph.nodes)
    index = {n: i for i, n in enumerate(nodes)}
    val_of = lambda n: n.meta.get("val", None)  # noqa: E731
    end = len(nodes)  # "live to the end" sentinel

    # tensors returned to the caller (loss, grads) live to the end
    output_inputs = set()
    for node in nodes:
        if node.op == "output":
            output_inputs.update(node.all_input_nodes)

    # params/buffers are referenced once but stay resident all step, so pin them
    placeholders = [n for n in nodes if n.op == "placeholder"]
    persistent_state = set(placeholders[:num_state_inputs])

    # ---- view aliasing repair ----
    # A view shares its base's storage and must not add bytes. Some transforms
    # rebuild node.meta["val"] without that aliasing (seen after FSDP bucketing:
    # aten.t outputs get a fresh fake storage), which would count every
    # transposed weight twice. Trust the op over the metadata and remap view
    # outputs onto the storage of their first tensor input.
    alias_of = {}

    def _resolve_sid(sid):
        seen = set()
        while sid in alias_of and sid not in seen:
            seen.add(sid)
            sid = alias_of[sid]
        return sid

    for node in nodes:
        if node.op != "call_function":
            continue
        packet = getattr(node.target, "_overloadpacket", node.target)
        if (
            str(getattr(packet, "__name__", packet)).split(".")[-1]
            not in _VIEW_OP_NAMES
        ):
            continue
        out = node.meta.get("val")
        if not isinstance(out, torch.Tensor):
            continue
        base = next(
            (
                a.meta.get("val")
                for a in node.all_input_nodes
                if isinstance(a.meta.get("val"), torch.Tensor)
            ),
            None,
        )
        if base is None:
            continue
        out_sid = out.untyped_storage()._cdata
        base_sid = base.untyped_storage()._cdata
        if out_sid != base_sid:
            alias_of[out_sid] = base_sid

    def _matching_wait(node):
        """The wait_tensor that completes this async collective, or None."""
        for u in node.users:
            if (
                getattr(u.target, "namespace", None) == "_c10d_functional"
                and getattr(u.target, "_opname", "") == "wait_tensor"
            ):
                return u
        return None

    def _is_functional_collective(node) -> bool:
        return getattr(node.target, "namespace", None) == "_c10d_functional"

    # ---- storage-keyed birth / death / size ----
    # Keyed by (storage_id, birth_index), so an id the allocator reuses starts a
    # fresh interval instead of merging into one long lifetime.
    birth, death, size = {}, {}, {}
    live_key = {}
    for i, node in enumerate(nodes):
        # node.meta["val"] holds plain tensors: make_fx unwraps subclasses (e.g.
        # DTensor) while tracing, so untyped_storage() is valid here.
        for t in pytree.tree_leaves(node.meta.get("val")):
            if not isinstance(t, torch.Tensor):
                continue
            # This is the GPU peak. Host-resident tensors (the pinned copies an
            # offload pass makes) would inflate it by exactly the savings.
            if t.device.type != "cuda":
                continue
            sid = _resolve_sid(t.untyped_storage()._cdata)

            key = live_key.get(sid)
            if key is None or death.get(key, -1) < i:  # new sid, or it died already
                key = (sid, i)
                live_key[sid] = key
                birth[key] = i
                size[key] = get_size(t)

            # Last use: the largest index among users that read THIS storage (a
            # multi-output user may consume only some outputs).
            d = i
            for u in node.users:
                u_inputs = pytree.tree_leaves(
                    (map_arg(u.args, val_of), map_arg(u.kwargs, val_of))
                )
                reads_sid = any(
                    isinstance(u_t, torch.Tensor)
                    and u_t.untyped_storage()._cdata == sid
                    for u_t in u_inputs
                )
                if not reads_sid:
                    continue
                d = max(d, index[u])

                if _is_functional_collective(u):
                    # all_gather_into_tensor_out writes into its own input, so
                    # that storage is already live as the collective's output
                    # and extending it would double count. reduce_scatter has no
                    # out= aliasing, so its input really dies at the collective.
                    if not str(u.target).endswith("_out.default"):
                        w = _matching_wait(u)
                        if w is not None:
                            d = max(d, index[w])

            if node in output_inputs or node in persistent_state:
                d = end  # resident params/buffers and returned tensors

            death[key] = max(death.get(key, i), d)

    # ---- O(n) sweep for the peak ----
    add_at, free_at = defaultdict(list), defaultdict(list)
    for key in size:
        add_at[birth[key]].append(key)
        free_at[death[key]].append(key)

    current = peak = 0
    for i in range(end + 1):  # +1 so end-of-graph frees are processed
        for key in add_at.get(i, ()):
            current += size[key]
        peak = max(peak, current)
        for key in free_at.get(i, ()):
            current -= size[key]

    return peak


def optimizer_state_bytes(opt_config: object | None, model: torch.nn.Module) -> int:
    """Per-rank persistent optimizer-state bytes, derived from the optimizer
    config without needing an optimizer instance or a training step.

    Returns 0 when there is no optimizer, and ``UNKNOWN_OPTIMIZER_BYTES`` (-1)
    when any param group names an optimizer missing from ``STATES_PER_PARAM`` --
    callers subtract this from a budget, so an under-count would silently OOM.
    """
    if opt_config is None or not opt_config.param_groups:
        return 0

    # Max across groups: exact for a uniform config, and the conservative side
    # of the approximation for a mixed one.
    def _states_per_param(pg) -> int:
        name = getattr(pg, "optimizer_name", None)
        kwargs = getattr(pg, "optimizer_kwargs", None) or {}
        if name not in STATES_PER_PARAM:
            logger.warning(
                "optimizer_state_bytes: cannot model optimizer %r, so its state "
                "size is reported as unknown rather than guessed. Add %r to "
                "STATES_PER_PARAM with its per-parameter state count to fix.",
                name,
                name,
            )
            return UNKNOWN_OPTIMIZER_BYTES  # negative marks the group unmodelled
        n = STATES_PER_PARAM[name]
        if n and kwargs.get("amsgrad", False):
            n += 1  # max_exp_avg_sq
        if name == "SGD" and kwargs.get("momentum", 0):
            n += 1  # momentum_buffer
        return n

    per_group = [_states_per_param(pg) for pg in opt_config.param_groups]
    if any(n < 0 for n in per_group):
        # Unknown anywhere means unknown overall; taking the max first would
        # hide an unmodelled group behind a known one.
        return UNKNOWN_OPTIMIZER_BYTES
    n_states = max(per_group, default=0)
    if n_states == 0:
        return 0

    # states are fp32 except under the bf16 fused implementation
    state_dtype = (
        torch.bfloat16
        if opt_config.implementation == "fused_opt_states_bf16"
        else torch.float32
    )
    elt_bytes = torch.finfo(state_dtype).bits // 8
    # count the local shard of a DTensor, so the result is per rank
    num_params = sum(
        (p.to_local() if hasattr(p, "to_local") else p).numel()
        for p in model.parameters()
    )

    return num_params * n_states * elt_bytes


def get_per_node_runtime(
    gm,
    *,
    warmup_iters: int = 2,
    bench_iters: int = 3,
) -> dict:
    """Benchmark every costable node of a joint FX graph on random inputs and
    return its time in ms, keyed by node name.

    Requires CUDA. Nodes that cannot be timed in isolation are priced at 0.0:
    HOPs (flex_attention), view and inplace-view ops, ``_c10d_functional``
    collectives, and anything that errors on random inputs.
    """

    def to_real(v):
        if isinstance(v, torch.Tensor):
            shape = tuple(v.shape)
            if v.dtype in _FLOAT_TYPES:
                return torch.rand(shape, dtype=v.dtype, device=v.device)
            return torch.ones(shape, dtype=v.dtype, device=v.device)
        return v

    def benchmark(node) -> float:
        target = node.target
        # HOPs (flex_attention) cannot be micro-benchmarked in isolation
        if not isinstance(target, torch._ops.OpOverload):
            return 0
        # views and create ops are ~free and misbehave on random inputs
        if target._overloadpacket in _VIEW_OPS:
            return 0.0
        if torch.Tag.inplace_view in getattr(target, "tags", ()):
            return 0.0
        if not torch.cuda.is_available():
            raise ValueError("No cuda device found for runtime benchmarking")
        # Collectives must never be benchmarked: they are async, so firing one
        # per iteration without waiting strands every output, and issuing
        # communication the other ranks are not in can hang or corrupt state.
        if getattr(target, "namespace", None) == "_c10d_functional":
            return 0  # assume async
        # This runs once per costable node, so anything retained accumulates.
        # no_grad keeps an autograd graph from pinning tensors, and the del
        # drops the last references before the next node runs.
        real_args = real_kwargs = out = None
        try:
            with torch.no_grad():
                real_args = map_arg(node.args, lambda n: to_real(n.meta.get("val")))
                real_kwargs = map_arg(node.kwargs, lambda n: to_real(n.meta.get("val")))
                out = target(*real_args, **(real_kwargs or {}))
                for _ in range(warmup_iters):
                    out = target(*real_args, **(real_kwargs or {}))
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(bench_iters):
                    out = target(*real_args, **(real_kwargs or {}))
                end.record()
                torch.cuda.synchronize()
                return start.elapsed_time(end) / bench_iters
        except Exception:
            return 0  # random inputs can violate an op's constraints
        finally:
            del real_args, real_kwargs, out

    graph = gm.gm.graph if hasattr(gm, "gm") else gm.graph
    benchmark_ms: dict = {}
    for node in graph.nodes:
        if not _is_costable_op(node):
            continue
        benchmark_ms[node.name] = benchmark(node)

    if torch.cuda.is_available():
        # the per-node tensors are freed but still sit in the allocator pool,
        # inflating the peak whatever runs next has to fit alongside
        torch.cuda.empty_cache()

    return benchmark_ms


def get_transfer_bw(
    dev: str | None = None,
    nbytes: int = 512 * 1024 * 1024,
    iters: int = 20,
    pinned: bool = True,
) -> dict:
    """Measured ``{h2d, d2h}`` GB/s for ``dev``, defaulting to the current device.

    Under torchrun each rank drives its own GPU over its own host link, so a
    hard-coded "cuda:0" would make every rank report rank 0's bandwidth.
    """
    if dev is None:
        dev = f"cuda:{torch.cuda.current_device()}"
    bw = _measure_transfer_bw(nbytes=nbytes, iters=iters, pinned=pinned, dev=dev)
    return {"h2d": bw["h2d"], "d2h": bw["d2h"]}


def _measure_transfer_bw(
    nbytes: int = 512 * 1024 * 1024,
    iters: int = 20,
    pinned: bool = True,
    dev: str = "cuda:0",
) -> dict:
    """Measure H2D and D2H bandwidth (GB/s) with pinned host memory, which is
    the regime an offload path actually uses. Call ``get_transfer_bw`` instead.

    The default 512 MB saturates the link; a smaller tensor may not reach this
    rate, so the roofline over-predicts throughput for tiny offloads.
    """
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"Measuring host<->device bandwidth requires a CUDA device, but "
            f"CUDA is not available (requested {dev!r})."
        )
    count = torch.cuda.device_count()
    # DeviceIndex is int8_t, so "cuda:999" wraps negative rather than staying
    # large; reject anything outside [0, count).
    index = torch.device(dev).index
    if index is not None and not (0 <= index < count):
        available = [
            f"cuda:{i} ({torch.cuda.get_device_name(i)})" for i in range(count)
        ]
        raise RuntimeError(
            f"requested device {dev!r} does not exist. "
            f"Available CUDA devices: {available}"
        )

    # Fit the buffer to what is free on the device, keeping some headroom.
    # Shrink if it does not fit, and refuse if that would stop saturating.
    free, _ = torch.cuda.mem_get_info(torch.device(dev))
    budget = int(free * _FREE_MEM_FRACTION)
    if nbytes > budget:
        if budget < _MIN_SATURATING_BYTES:
            raise RuntimeError(
                f"{dev} has {free / 1e6:.0f} MB free ({budget / 1e6:.0f} MB usable), "
                f"too little to measure host<->device bandwidth "
                f"reliably (need >= {_MIN_SATURATING_BYTES / 1e6:.0f} MB)."
            )
        warnings.warn(
            f"host<->device bandwidth benchmark: {dev} has {free / 1e6:.0f} MB free; "
            f"shrinking the "
            f"benchmark buffer from {nbytes / 1e6:.0f} MB to {budget / 1e6:.0f} MB. "
            f"The reported bandwidth may under-estimate the link's peak.",
            stacklevel=2,
        )
        nbytes = budget

    host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=pinned)
    gpu = torch.empty(nbytes, dtype=torch.uint8, device=dev)

    # Synchronize the benchmarked device, not the caller's current one.
    def time_copy(dst, src):
        torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        for _ in range(iters):
            dst.copy_(src, non_blocking=True)
        torch.cuda.synchronize(dev)
        return (time.perf_counter() - t0) / iters

    # warmup
    gpu.copy_(host, non_blocking=True)
    host.copy_(gpu, non_blocking=True)
    torch.cuda.synchronize(dev)

    h2d_s = time_copy(gpu, host)  # host -> device
    d2h_s = time_copy(host, gpu)  # device -> host
    return {"h2d": nbytes / h2d_s / 1e9, "d2h": nbytes / d2h_s / 1e9}

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Entry point: torchtitan.train.main plus per-step allocator measurement.

``MEMPLAN_ALLOCATOR``:
  - ``caching`` (default): PyTorch caching allocator. Each step prints one
    ``MEMSTAT`` JSON line (peak allocated / reserved). On
    ``MEMPLAN_TRACE_STEP`` it dumps ``state_before.pkl`` and ``trace.pkl``
    (allocator history with FX node attribution and phase metadata).
  - ``trace``: planned allocator (``memplan_alloc.so``) with no plan; dumps the
    alloc/free sequence of ``MEMPLAN_TRACE_STEP`` to ``window_trace.txt``.
  - ``plan``: planned allocator serving ``MEMPLAN_PLAN`` from
    ``MEMPLAN_PLAN_START_STEP`` on.

One train step is one planning window.
"""

import ctypes
import json
import logging
import os
import pickle
import time
from typing import cast

import torch

_MEM = torch.cuda.memory
_SO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "memplan_alloc.so")
_MODE = os.environ.get("MEMPLAN_ALLOCATOR", "caching")
_LIB = None
_STAT_KEYS = [
    "cur_alloc",
    "peak_alloc",
    "arena_size",
    "arena_live_peak",
    "pool_reserved_cur",
    "pool_reserved_high",
    "pool_used_cur",
    "pool_used_high",
    "hits",
    "misses",
    "overlaps",
    "window_fallbacks",
    "window_allocs",
    "diverged",
]


def _install_planned_allocator() -> None:
    """Swap in the planned allocator before any CUDA allocation happens."""
    global _LIB
    alloc = torch.cuda.memory.CUDAPluggableAllocator(
        _SO, "memplan_malloc", "memplan_free"
    )
    torch.cuda.memory.change_current_allocator(alloc)
    _LIB = ctypes.CDLL(_SO)
    _LIB.memplan_begin_window.argtypes = [ctypes.c_uint64]
    _LIB.memplan_set_trace_window.argtypes = [ctypes.c_int32, ctypes.c_int32]
    _LIB.memplan_dump_trace.argtypes = [ctypes.c_char_p]
    _LIB.memplan_load_plan.argtypes = [ctypes.c_char_p, ctypes.c_int]
    _LIB.memplan_set_plan_active.argtypes = [ctypes.c_int]
    _LIB.memplan_stats.argtypes = [ctypes.POINTER(ctypes.c_int64)]

    # The pluggable allocator has no device stats; serve torchtitan's memory
    # monitor from the planned allocator's counters instead.
    def memory_stats(device=None):
        s = _planned_stats()
        reserved = s["arena_size"] + s["pool_reserved_high"]
        return {
            "active_bytes.all.peak": s["peak_alloc"],
            "allocated_bytes.all.peak": s["peak_alloc"],
            "reserved_bytes.all.peak": reserved,
            "num_alloc_retries": 0,
            "num_ooms": 0,
        }

    def reset_peak_memory_stats(device=None):
        _LIB.memplan_reset_peak()

    for mod in (torch.cuda, torch.cuda.memory):
        mod.memory_stats = memory_stats
        mod.reset_peak_memory_stats = reset_peak_memory_stats


def _planned_stats() -> dict:
    buf = (ctypes.c_int64 * len(_STAT_KEYS))()
    _LIB.memplan_stats(buf)
    return dict(zip(_STAT_KEYS, list(buf)))


if _MODE in ("trace", "plan"):
    _install_planned_allocator()
elif _MODE != "caching":
    raise ValueError(f"unknown MEMPLAN_ALLOCATOR={_MODE}")

from torchtitan.config import ConfigLoader  # noqa: E402
from torchtitan.observability.logging import init_logger  # noqa: E402
from torchtitan.trainer import Trainer  # noqa: E402

logger = logging.getLogger(__name__)


def _out_dir() -> str:
    d = os.environ.get("MEMPLAN_OUT", "/tmp/memplan")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        d = os.path.join(d, f"rank{os.environ['RANK']}")
    os.makedirs(d, exist_ok=True)
    return d


def _with_phase(fn, phase: str):
    def wrapped(*args, **kwargs):
        prev = _MEM._get_memory_metadata()
        _MEM._set_memory_metadata(phase)
        try:
            return fn(*args, **kwargs)
        finally:
            _MEM._set_memory_metadata(prev)

    return wrapped


def _caching_step(step, run_step, trace_step, out):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    start = torch.cuda.memory_stats()
    tracing = step == trace_step
    if tracing:
        with open(os.path.join(out, "state_before.pkl"), "wb") as f:
            pickle.dump(_MEM._snapshot(), f, protocol=4)
        _MEM._record_memory_history(
            enabled="all", context="alloc", stacks="python", max_entries=5_000_000
        )
    t0 = time.perf_counter()
    _MEM._set_memory_metadata("step")
    try:
        result = run_step()
    finally:
        _MEM._set_memory_metadata("")
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    s = torch.cuda.memory_stats()
    rec = {
        "step": step,
        "time_s": round(dt, 3),
        "start_allocated": start["allocated_bytes.all.current"],
        "start_reserved": start["reserved_bytes.all.current"],
        "peak_allocated": s["allocated_bytes.all.peak"],
        "peak_requested": s["requested_bytes.all.peak"],
        "peak_active": s["active_bytes.all.peak"],
        "peak_reserved": s["reserved_bytes.all.peak"],
        "end_allocated": s["allocated_bytes.all.current"],
        "end_reserved": s["reserved_bytes.all.current"],
        "num_alloc_retries": s["num_alloc_retries"],
        "num_ooms": s["num_ooms"],
        "num_segments": s["segment.all.current"],
        "num_allocs_in_step": s["allocation.all.allocated"]
        - start["allocation.all.allocated"],
    }
    if tracing:
        t1 = time.perf_counter()
        snap = _MEM._snapshot(augment_with_fx_traces=True)
        _MEM._record_memory_history(enabled=None)
        with open(os.path.join(out, "trace.pkl"), "wb") as f:
            pickle.dump(snap, f, protocol=4)
        rec["trace_dump_s"] = round(time.perf_counter() - t1, 1)
    return result, rec


def _planned_step(step, run_step, trace_step, out):
    device = torch.cuda.current_device()
    plan_start = int(os.environ.get("MEMPLAN_PLAN_START_STEP", "3"))
    if _MODE == "plan" and step == plan_start:
        # Release fallback-pool memory cached by the unplanned warmup steps,
        # then reserve the arena and switch the plan on.
        _LIB.memplan_trim_pool()
        plan_path = os.environ["MEMPLAN_PLAN"].format(rank=os.environ.get("RANK", "0"))
        n = _LIB.memplan_load_plan(plan_path.encode(), device)
        if n < 0:
            raise RuntimeError(f"memplan_load_plan failed: {n}")
        _LIB.memplan_set_plan_active(1)
        print(f"[memplan] plan active from step {step}: {n} entries", flush=True)
    if _MODE == "plan" and step == plan_start + 1:
        # The previous (unplanned) step's gradients lived in the fallback pool
        # until the first planned step's zero_grad; release that memory too.
        _LIB.memplan_trim_pool()
    if _MODE == "trace" and step == trace_step:
        # Window ids count from 0 at step 1. Trace two consecutive steps so
        # the planner sees when step-escaping tensors (gradients) are freed.
        _LIB.memplan_set_trace_window(step - 1, 2)
    torch.cuda.synchronize()
    _LIB.memplan_reset_peak()
    start = _planned_stats()
    _LIB.memplan_begin_window(torch.cuda.current_stream().cuda_stream)
    t0 = time.perf_counter()
    try:
        result = run_step()
    finally:
        _LIB.memplan_end_window()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    s = _planned_stats()
    rec = {
        "step": step,
        "time_s": round(dt, 3),
        "start_allocated": start["cur_alloc"],
        "peak_allocated": s["peak_alloc"],
        "peak_reserved": s["arena_size"] + s["pool_reserved_high"],
        "end_allocated": s["cur_alloc"],
        **{k: s[k] for k in _STAT_KEYS if k not in ("cur_alloc", "peak_alloc")},
    }
    if _MODE == "trace" and step == trace_step + 1:
        n = _LIB.memplan_dump_trace(os.path.join(out, "window_trace.txt").encode())
        rec["trace_events"] = n
    return result, rec


def _instrument(trainer: Trainer) -> None:
    engine = trainer.engine
    captured = {}
    orig_fb, orig_opt = engine.forward_backward, engine.optim_step

    def forward_backward(*args, **kwargs):
        result = orig_fb(*args, **kwargs)
        captured["loss"] = result.loss
        return result

    def optim_step(*args, **kwargs):
        grad_norm = orig_opt(*args, **kwargs)
        captured["grad_norm"] = grad_norm
        return grad_norm

    engine.forward_backward = forward_backward
    engine.optim_step = optim_step
    if _MODE == "caching":
        engine.forward_backward = _with_phase(engine.forward_backward, "fwdbwd")
        engine.optim_step = _with_phase(engine.optim_step, "optim")
    trace_step = int(os.environ.get("MEMPLAN_TRACE_STEP", "-1"))
    out = _out_dir()
    orig_train_step = trainer.train_step
    measure = _caching_step if _MODE == "caching" else _planned_step

    def train_step(data_iterator):
        step = engine.num_completed_steps + 1
        result, rec = measure(
            step, lambda: orig_train_step(data_iterator), trace_step, out
        )
        rec["allocator"] = _MODE
        # Exact values for bitwise numerics comparison across allocators.
        for key in ("loss", "grad_norm"):
            v = captured.pop(key, None)
            if v is not None:
                rec[key] = float(v.detach().float().cpu().item()).hex()
        print("MEMSTAT " + json.dumps(rec), flush=True)
        with open(os.path.join(out, "memstat.jsonl"), "a") as f:
            f.write(json.dumps(rec) + "\n")
        return result

    trainer.train_step = train_step


def main() -> None:
    init_logger()
    config = cast(Trainer.Config, ConfigLoader().load())
    trainer = config.build()
    try:
        _instrument(trainer)
        trainer.train()
    finally:
        trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

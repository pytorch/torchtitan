#!/usr/bin/env python3
"""Offline check: the GPU_MAX_HW_QUEUES formula vs. real profiler traces.

Counts distinct GPU stream lanes in each trace and asserts the config-computed
Q removes collisions for them (Q >= next_pow2(observed_lanes)). Trace lanes can
be fewer than the logical estimate (HIP-graph capture folds lanes), so the bound
is >=, not ==. Loads hw_queues.py by path so it runs on bare python3 (no torch).

Usage: python tools/verify_hw_queues.py <trace.json[.gz]> ...
"""
import gzip
import importlib.util
import json
import sys
from pathlib import Path

_path = Path(__file__).resolve().parents[1] / "torchtitan/experiments/graph_trainer/hw_queues.py"
_spec = importlib.util.spec_from_file_location("hw_queues", _path)
hwq = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hwq)  # module top is stdlib-only; torch is imported lazily

# Parallelism config behind the calibration traces: EP=2 2-node DeepSeek-V3 with
# dense FSDP AG/RS overlap on (dedicated PGs -> their own streams).
EP2_DSV3 = dict(
    dp_shard_active=True, is_moe=True, ep=2, tp=1, cp=1,
    fsdp_ag_rs_overlap=True, cudagraph=True,
)


def observed_lanes(path):
    opener = None
    if path.suffix == ".gz":
        opener = gzip.open
    else:
        opener = open
    with opener(path) as f:
        d = json.load(f)
    events = None 
    if isinstance(d, dict):
        events = d["traceEvents"]
    else:
        events = d
    return len(
        {
            e["args"]["name"]
            for e in events
            if e.get("ph") == "M"
            and e.get("name") == "thread_name"
            and str(e.get("args", {}).get("name", "")).startswith("stream")
        }
    )


def main(traces):
    if not traces:
        print("usage: verify_hw_queues.py <trace.json[.gz]> ...")
        return 1
    q = hwq._next_pow2(len(hwq._stream_lanes(**EP2_DSV3)))
    print(f"formula (EP=2 DSv3): Q={q}")
    rc = 0
    for t in map(Path, traces):
        obs = observed_lanes(t)
        need = hwq._next_pow2(obs)
        ok = q >= need
        rc |= not ok
        print(f"  [{'OK' if ok else 'FAIL'}] {t.name}: observed={obs}, need Q>={need}, have {q}")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

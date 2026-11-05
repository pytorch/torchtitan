# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Check the KernelGen problems that make_problems.py wrote, on local GPUs.

Driver:  python validate_problems.py <capture dir> [--gpus 0,1,2,3] [--only g003_rope,...]
Worker:  python validate_problems.py --worker <region folder>   (one problem per process)

Writes ``validation.json`` into each region folder. Nothing contacts the KernelGen service.
Per problem:
  * payload.json matches the create schema and reference_src == problem.py;
  * problem.py parses and imports; Model(*get_init_inputs()).cuda().eval() builds;
  * get_inputs() matches the recorded input shapes/strides/dtypes;
  * forward under no_grad returns the recorded output shapes/dtypes;
  * run A == runs B1..B3 bitwise (run-to-run determinism; B1..B3 are timed, the fastest
    is reported);
  * run A == run C bitwise, where run C enables torch.use_deterministic_algorithms(True)
    globally as the training run did (catches an op whose deterministic kernel the
    reference did not select);
  * inputs are not mutated;
  * non-finite / all-zero output statistics, forward time, peak memory.
"""

import argparse
import ast
import importlib.util
import json
import math
import os
import queue
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

MAX_SAFE_INT = 2**53 - 1


def check_payload(pdir):
    errs = []
    p = json.loads((pdir / "payload.json").read_text())
    src = (pdir / "problem.py").read_text()
    for k in ("op_type", "shape", "num_flops", "num_bytes", "reference_src"):
        if k not in p:
            errs.append(f"missing {k}")
    for k, lim in (("op_type", 255), ("shape", 255), ("dtype", 64)):
        v = p.get(k)
        if v is not None and (
            not isinstance(v, str) or not v.strip() or v != v.strip() or len(v) > lim
        ):
            errs.append(f"bad {k}")
    for k in ("num_flops", "num_bytes", "num_gpus", "num_kernels"):
        v = p.get(k)
        if v is not None and (
            not isinstance(v, int) or isinstance(v, bool) or v <= 0 or v > MAX_SAFE_INT
        ):
            errs.append(f"bad {k}={v!r}")
    for k in (
        "arithmetic_intensity",
        "max_valid_abs_diff",
        "max_valid_ulp",
        "comm_bytes",
    ):
        v = p.get(k)
        if v is not None and (
            not isinstance(v, (int, float))
            or isinstance(v, bool)
            or v < 0
            or not math.isfinite(v)
        ):
            errs.append(f"bad {k}={v!r}")
    if p.get("bound_type", "compute") not in (
        "compute",
        "memory",
        "mixed",
        "communication",
    ):
        errs.append("bad bound_type")
    if p.get("usage", "inference") not in (
        "inference",
        "training",
        "inference/training",
    ):
        errs.append("bad usage")
    d = p.get("description")
    if d is not None and (not isinstance(d, str) or len(d) > 10000):
        errs.append("bad description")
    rs = p.get("reference_src", "")
    if not isinstance(rs, str) or not rs.strip() or len(rs) > 2_000_000:
        errs.append("bad reference_src")
    if rs != src:
        errs.append("reference_src != problem.py")
    if not src.isascii():
        errs.append("problem.py is not ASCII")
    return p, errs


def worker(pdir):
    import torch

    pdir = Path(pdir)
    res = {"folder": pdir.name, "errors": [], "warnings": []}
    meta = json.loads((pdir / "problem_meta.json").read_text())
    _, perrs = check_payload(pdir)
    res["errors"] += perrs
    src = (pdir / "problem.py").read_text()
    ast.parse(src, filename="problem.py")
    spec = importlib.util.spec_from_file_location(
        f"kg_{pdir.name}", pdir / "problem.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    def dedup_broadcast(x):
        for d in range(x.dim()):
            if x.stride(d) == 0 and x.size(d) > 1:
                x = x.narrow(d, 0, 1)
        return x

    def thash(x):
        """Order-sensitive int64 hash of the tensor's bits (broadcast dims counted once)."""
        x = dedup_broadcast(x)
        if x.is_complex():
            x = torch.view_as_real(x)
        word = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[
            x.element_size()
        ]
        if x.dtype == torch.bool:
            x = x.view(torch.uint8)
        x = x.reshape(1, -1) if x.dim() == 0 else x
        row = max(1, x[0].numel())
        step = max(1, (1 << 27) // row)
        acc = torch.zeros((), dtype=torch.int64, device=x.device)
        off = 0
        for i in range(0, x.shape[0], step):
            c = x[i : i + step].reshape(-1).view(word).to(torch.int64)
            w = (
                torch.arange(off, off + c.numel(), device=x.device, dtype=torch.int64)
                * 0x9E3779B1
                + 0x7F4A7C15
            )
            acc += ((c ^ (c >> 7)) * w).sum()
            off += c.numel()
        return int(acc.item())

    def stats(x):
        x = dedup_broadcast(x)
        n = x.numel()
        zeros = nonfinite = 0
        x2 = x.reshape(1, -1) if x.dim() == 0 else x
        row = max(1, x2[0].numel())
        step = max(1, (1 << 27) // row)
        for i in range(0, x2.shape[0], step):
            c = x2[i : i + step]
            zeros += int((c == 0).sum().item())
            if c.is_floating_point() or c.is_complex():
                nonfinite += int((~torch.isfinite(c)).sum().item())
        return {"numel": n, "zero_frac": zeros / max(n, 1), "nonfinite": nonfinite}

    def as_tuple(o):
        return tuple(o) if isinstance(o, (tuple, list)) else (o,)

    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    model = mod.Model(*mod.get_init_inputs()).cuda().eval()
    inputs = mod.get_inputs()
    torch.cuda.synchronize()
    res["input_gen_s"] = round(time.time() - t0, 2)
    exp_in = meta["inputs"]
    if len(inputs) != len(exp_in):
        res["errors"].append(f"{len(inputs)} inputs, expected {len(exp_in)}")
    for x, e in zip(inputs, exp_in):
        if (
            list(x.shape) != e["shape"]
            or str(x.dtype) != e["dtype"]
            or list(x.stride()) != e["stride"]
        ):
            res["errors"].append(
                f"input {e['var']}: got {list(x.shape)} {list(x.stride())} {x.dtype}, "
                f"expected {e['shape']} {e['stride']} {e['dtype']}"
            )
    h_in0 = [thash(x) for x in inputs]

    with torch.no_grad():
        out = as_tuple(model(*inputs))
        exp_out = meta["expected_outputs"]
        if len(out) != len(exp_out):
            res["errors"].append(f"{len(out)} outputs, expected {len(exp_out)}")
        res["outputs"] = []
        for k, (o, e) in enumerate(zip(out, exp_out)):
            if list(o.shape) != e["shape"] or str(o.dtype) != e["dtype"]:
                res["errors"].append(
                    f"output {k}: got {list(o.shape)} {o.dtype}, expected {e['shape']} {e['dtype']}"
                )
            st = stats(o)
            st.update(shape=list(o.shape), dtype=str(o.dtype).replace("torch.", ""))
            if list(o.stride()) != e["stride"]:
                st["stride_differs"] = [list(o.stride()), e["stride"]]
            res["outputs"].append(st)
            if st["numel"] > 1 and st["zero_frac"] == 1.0:
                res["warnings"].append(f"output {k} is all zeros")
            if st["nonfinite"]:
                res["warnings"].append(
                    f"output {k} has {st['nonfinite']} non-finite values"
                )
        h_a = [thash(o) for o in out]
        del out

        times = []
        for _ in range(3):  # run A was the warmup; report the fastest of 3 timed runs
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(
                enable_timing=True
            )
            start.record()
            out = as_tuple(model(*inputs))
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) * 1000)
            h_b = [thash(o) for o in out]
            del out
            if h_b != h_a:
                break
        res["forward_us"] = round(min(times), 1)

        torch.use_deterministic_algorithms(True)
        torch.utils.deterministic.fill_uninitialized_memory = (
            False  # as torchtitan's set_determinism
        )
        try:
            out = as_tuple(model(*inputs))
            h_c = [thash(o) for o in out]
            del out
        except RuntimeError as e:
            h_c = None
            res["errors"].append(
                f"global deterministic mode raised: {str(e).splitlines()[0][:300]}"
            )
        finally:
            torch.use_deterministic_algorithms(False)

    if h_a != h_b:
        res["errors"].append(
            "run-to-run outputs differ (reference not deterministic) at outputs "
            f"{[k for k, (a, b) in enumerate(zip(h_a, h_b)) if a != b]}"
        )
    if h_c is not None and h_a != h_c:
        res["errors"].append(
            "outputs differ from a globally deterministic run at outputs "
            f"{[k for k, (a, c) in enumerate(zip(h_a, h_c)) if a != c]}"
        )
    if [thash(x) for x in inputs] != h_in0:
        res["errors"].append("forward mutated its inputs")
    res["peak_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 1)
    res["ok"] = not res["errors"]
    print("RESULT " + json.dumps(res))


def run_one(pdir, gpu, timeout):
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=str(gpu),
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        PYTHONDONTWRITEBYTECODE="1",
    )
    t0 = time.time()
    try:
        p = subprocess.run(
            [sys.executable, __file__, "--worker", str(pdir)],
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return {
            "folder": pdir.name,
            "ok": False,
            "errors": [f"timeout after {timeout}s"],
            "warnings": [],
        }
    line = next(
        (l for l in reversed(p.stdout.splitlines()) if l.startswith("RESULT ")), None
    )
    if line is None:
        tail = [l for l in p.stderr.strip().splitlines() if l.strip()][-3:]
        return {
            "folder": pdir.name,
            "ok": False,
            "errors": ["crashed: " + " | ".join(tail)[-600:]],
            "warnings": [],
        }
    res = json.loads(line[len("RESULT ") :])
    res["wall_s"] = round(time.time() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "root", nargs="?", help="capture directory (after make_problems.py)"
    )
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    ap.add_argument("--gpus", default="0,1,2,3")
    ap.add_argument("--only", help="comma-separated region folder names")
    ap.add_argument("--timeout", type=int, default=1200)
    a = ap.parse_args()
    if a.worker:
        return worker(a.worker)
    pdirs = sorted(
        p.parent for p in Path(a.root, "fusion_regions").glob("*/problem.py")
    )
    if a.only:
        pdirs = [p for p in pdirs if p.name in a.only.split(",")]
    gpus = a.gpus.split(",")
    free = queue.Queue()
    for g in gpus:
        free.put(g)

    def task(pdir):
        g = free.get()
        try:
            r = run_one(pdir, g, a.timeout)
        finally:
            free.put(g)
        (pdir / "validation.json").write_text(json.dumps(r, indent=1) + "\n")
        print(
            f"{'OK  ' if r['ok'] else 'FAIL'} {pdir.name} {r.get('forward_us', '-')}us "
            f"peak={r.get('peak_gib', '-')}GiB {'; '.join(r['errors'] + r['warnings'])[:300]}",
            flush=True,
        )
        return r

    with ThreadPoolExecutor(len(gpus)) as ex:
        results = list(ex.map(task, pdirs))
    print(f"ok {sum(r['ok'] for r in results)}/{len(results)}")


if __name__ == "__main__":
    main()

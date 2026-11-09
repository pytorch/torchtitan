"""Exactness gate and timing for substitute-kernels problems (Qwen3 235B-A22B, autokernel run 1; copied from the Qwen3 30B-A3B loop).

    python harness.py <pat_dir>/specialization_<l2>.json <candidate.py> [--seeds 0 1 2]
        [--real inputs.pt] [--time] [--baseline <file.py>]

Inputs follow the spec (shape, stride, dtype, device). int64 tensors with a last dim k
<= NUM_EXPERTS are top-k style indices: k distinct values per row in [0, NUM_EXPERTS);
longer int64 rows are token positions in [0, MAX_POSITION). Reference and candidate
get identical inputs; every output is compared byte for byte (autokernel's gate), plus
structure, shape, stride, dtype, device, and which outputs alias each other.
--time reports median us over CUDA-event timed eager calls and CUDA-graph replays.
"""

import argparse
import importlib.util
import json
import os
import statistics
import sys

sys.dont_write_bytecode = True  # never write __pycache__ under kernel_cache/

import torch

# Qwen3 235B-A22B (as 30B-A3B): 128 routed experts, 4096-token context.
NUM_EXPERTS = int(os.environ.get("HARNESS_NUM_EXPERTS", "128"))
MAX_POSITION = int(os.environ.get("HARNESS_MAX_POSITION", "4096"))

DT = {"float32": torch.float32, "bfloat16": torch.bfloat16, "int64": torch.int64,
      "int32": torch.int32, "bool": torch.bool, "float16": torch.float16,
      "float8_e4m3fn": torch.float8_e4m3fn, "float8_e8m0fnu": torch.float8_e8m0fnu}


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def make_input(d, g):
    if "shape" not in d:
        return d["value"]
    shape, stride, dtype = d["shape"], d["stride"], DT[d["dtype"]]
    base = [1 if st == 0 else s for s, st in zip(shape, stride)]
    if dtype.is_floating_point:
        v = torch.randn(base, generator=g, device="cuda").to(dtype)
    elif dtype == torch.bool:
        v = torch.rand(base, generator=g, device="cuda") < 0.3
    elif len(shape) == 0:
        v = torch.randint(1, 1 << 20, (), generator=g, device="cuda", dtype=dtype)
    elif base[-1] <= NUM_EXPERTS:
        # Router top-k indices: distinct experts per row.
        rows = int(torch.tensor(base[:-1]).prod()) if len(base) > 1 else 1
        v = torch.rand(rows, NUM_EXPERTS, generator=g, device="cuda").argsort(-1)[:, : base[-1]]
        v = v.reshape(base).to(dtype)
    else:
        # Token positions: 0..MAX_POSITION-1 repeating, as the RoPE tables are indexed.
        v = (torch.arange(base[-1], device="cuda") % MAX_POSITION).expand(base).to(dtype)
    out = torch.empty_strided(base, [st if st else 1 for st in stride], dtype=dtype, device="cuda")
    out.copy_(v)
    return out.expand(shape) if base != shape else out


def flat(o):
    if isinstance(o, (tuple, list)):
        return [x for v in o for x in flat(v)]
    return [o]


def alias_groups(outs):
    ptrs = [o.untyped_storage().data_ptr() if isinstance(o, torch.Tensor) else None for o in outs]
    return [ptrs.index(p) if p is not None else None for p in ptrs]


def check(spec, ref_out, cand_out):
    errs = []
    ref_out, cand_out = flat(ref_out), flat(cand_out)
    want = [w for w in flat(spec["outputs"]) if "shape" in w or "value" in w]
    if len(cand_out) != len(want):
        return [f"{len(cand_out)} outputs, spec has {len(want)}"]
    for k, (r, c, w) in enumerate(zip(ref_out, cand_out, want)):
        if not isinstance(c, torch.Tensor):
            errs.append(f"out{k}: not a tensor")
            continue
        meta = (list(c.shape), list(c.stride()), str(c.dtype).removeprefix("torch."), str(c.device))
        if meta != (w["shape"], w["stride"], w["dtype"], w["device"]):
            errs.append(f"out{k}: metadata {meta} != spec {(w['shape'], w['stride'], w['dtype'], w['device'])}")
            continue
        rb = r.detach().reshape(-1).contiguous().view(torch.uint8)
        cb = c.detach().reshape(-1).contiguous().view(torch.uint8)
        if not torch.equal(rb, cb):
            n = int((r.reshape(-1).contiguous().view(torch.uint8) != cb).sum())
            errs.append(f"out{k}: {n} bytes differ")
    if alias_groups(ref_out) != alias_groups(cand_out):
        errs.append(f"alias structure {alias_groups(cand_out)} != reference {alias_groups(ref_out)}")
    return errs


def median_us(fn, n=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record(); fn(); e.record(); torch.cuda.synchronize()
        ts.append(s.elapsed_time(e) * 1e3)
    return statistics.median(ts), (min(ts), max(ts))


def graph_us(fn, n=50):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    return median_us(g.replay, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("spec")
    ap.add_argument("candidate")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--real", help="torch.save'd list of real inputs captured from training")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--baseline", help="file whose run() to time against (default reference.py)")
    a = ap.parse_args()
    spec = json.load(open(a.spec))
    pdir = os.path.dirname(os.path.abspath(a.spec))
    ref = load(os.path.join(pdir, spec["reference"]), "reference_mod")
    cand = load(a.candidate, "candidate_mod")
    torch.backends.cuda.matmul.allow_tf32 = False
    # As the training run (torchtitan/distributed/utils.py set_determinism): deterministic
    # algorithms without the NaN fill of torch.empty, which would inflate reference timings.
    torch.use_deterministic_algorithms(True)
    torch.utils.deterministic.fill_uninitialized_memory = False
    cases = []
    for seed in a.seeds:
        g = torch.Generator(device="cuda").manual_seed(seed)
        cases.append((f"seed{seed}", [make_input(d, g) for d in spec["inputs"]]))
    if a.real:
        cases.append(("real", [x.cuda() if isinstance(x, torch.Tensor) else x for x in torch.load(a.real)]))
    def poisoned_run(snap, ref_out, byte):
        """Run the candidate on fresh input copies with its future output memory poisoned.

        Fresh cudaMalloc memory reads as zero, and a reused block may hold an earlier correct
        output, so a kernel that skips bytes whose correct value is 0 (e.g. zero padding) would
        pass. Blocks of the reference outputs' sizes are filled with ``byte`` and freed so the
        caching allocator hands them to the candidate, and torch.empty NaN-fills meanwhile.
        """
        inputs = [x.clone() if isinstance(x, torch.Tensor) else x for x in snap]
        bait = [torch.empty(t.numel() * t.element_size(), dtype=torch.uint8, device=t.device).fill_(byte)
                for t in flat(ref_out) if isinstance(t, torch.Tensor) and t.numel()]
        del bait
        prev = torch.utils.deterministic.fill_uninitialized_memory
        torch.utils.deterministic.fill_uninitialized_memory = True
        try:
            return cand.run(*inputs)
        finally:
            torch.utils.deterministic.fill_uninitialized_memory = prev

    ok = True
    for name, args in cases:
        snap = [x.clone() if isinstance(x, torch.Tensor) else x for x in args]
        r = ref.run(*args)
        c1 = poisoned_run(snap, r, 0xFF)
        c2 = poisoned_run(snap, r, 0x5A)
        errs = check(spec, r, c1) + [f"rerun: {e}" for e in check(spec, c1, c2)]
        print(f"{name}: {'EXACT' if not errs else 'MISMATCH ' + '; '.join(errs[:6])}")
        ok &= not errs
    if a.time:
        args = cases[0][1]
        base = load(a.baseline, "baseline_mod") if a.baseline else ref
        for label, mod in (("baseline", base), ("candidate", cand)):
            e, espread = median_us(lambda: mod.run(*args))
            gr, gspread = graph_us(lambda: mod.run(*args))
            print(f"{label}: eager {e:.1f} us (min {espread[0]:.1f} max {espread[1]:.1f}), "
                  f"cuda-graph {gr:.1f} us (min {gspread[0]:.1f} max {gspread[1]:.1f})")
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()

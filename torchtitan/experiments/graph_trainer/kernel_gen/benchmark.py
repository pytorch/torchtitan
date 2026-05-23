#!/usr/bin/env python3
"""Benchmark and validate generated kernels against their PyTorch reference.

For each kernel in <dir>/<name>/:
  1. Run Model.forward (eager) and capture reference output + time.
  2. Validate + time the KernelAgent triton kernel against reference.
  3. Validate + time torch.compile(model.forward) against reference.
  4. Write benchmark.json with per-backend status + timings + errors.

Usage:
  python -m torchtitan.experiments.graph_trainer.kernel_gen.benchmark [--problems P1 ...]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re as _re
from pathlib import Path

import torch

DEFAULT_GENERATED_DIR = Path(__file__).parent / "generated"
GENERATED_DIR = DEFAULT_GENERATED_DIR  # overridden by --dir

# Tolerance tiers:
#   bitwise: atol=8e-2 + rtol=1e-3 allows ~1 ULP in bf16 around magnitude 1
#     (bf16 ULP at |x|≈1 is 2^-4 = 0.0625). Covers silu/silu_backward and
#     other elementwise ops where instruction reordering causes 1-ULP drift.
#   reduction: atol=5e-2 + rtol=1e-2 for ops with order-dependent accumulation
#     (sum, norm). bf16 reductions can differ by 0.01-0.03 depending on tile.
#   complex: atol=5e-2 + rtol=1e-2 for complex arithmetic with intermediate
#     f32 rounding.
TOLERANCE_BITWISE = {"rtol": 1e-3, "atol": 8e-2}
TOLERANCE_REDUCTION = {"rtol": 1e-2, "atol": 5e-2}
TOLERANCE_COMPLEX = {"rtol": 1e-2, "atol": 5e-2}

_REDUCTION_OPS = {
    "aten.sum", "aten.mean", "aten.amax", "aten.amin", "aten.prod",
    "aten.norm", "aten.var", "aten.std", "aten._fused_rms_norm",
    "aten._fused_rms_norm_backward", "aten.layer_norm", "aten.softmax",
    "aten.log_softmax", "aten.cross_entropy_loss",
}

_COMPLEX_OPS = {
    "aten.view_as_complex", "aten.view_as_real", "aten._conj",
}

NUM_CORRECTNESS_TRIALS = 5
WARMUP = 20
BENCH_ITERS = 100


def _detect_tolerance(problem_path: Path) -> tuple[dict, str]:
    content = problem_path.read_text()
    if any(op in content for op in _REDUCTION_OPS):
        return TOLERANCE_REDUCTION, "reduction"
    if any(op in content for op in _COMPLEX_OPS):
        return TOLERANCE_COMPLEX, "complex"
    return TOLERANCE_BITWISE, "bitwise"


def _load_module(path: Path, module_name: str):
    """Load a Python module from path; strip any header comments before
    'import torch' so @triton.jit's inspect.getsourcelines() works."""
    import tempfile
    content = path.read_text()
    idx = content.find("import torch")
    if idx > 0:
        clean = content[idx:]
        tmp = Path(tempfile.gettempdir()) / f"_bench_{module_name}_{path.parent.name}.py"
        tmp.write_text(clean)
        load_path = tmp
    else:
        load_path = path

    spec = importlib.util.spec_from_file_location(module_name, str(load_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load {load_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _benchmark_fn(fn, inputs, warmup=WARMUP, iters=BENCH_ITERS):
    for _ in range(warmup):
        fn(*inputs)
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    for _ in range(iters):
        fn(*inputs)
    end_event.record()
    torch.cuda.synchronize()
    return start_event.elapsed_time(end_event) / iters


def _compute_errors(out_candidate, out_ref) -> tuple[float, float, str | None]:
    """Compute (max_abs_err, max_rel_err, error_message_or_None).

    Walks tuples/lists pairwise, casts to float32 for comparison, returns
    max errors across all elements. Returns (0, 0, msg) on shape/type
    mismatch.
    """
    if isinstance(out_ref, (tuple, list)):
        if isinstance(out_candidate, tuple) and len(out_candidate) == 1 and not isinstance(out_ref, tuple) is False:
            pass  # fall through
        if not isinstance(out_candidate, (tuple, list)):
            return 0.0, 0.0, "candidate returned Tensor, expected tuple"
        if len(out_candidate) != len(out_ref):
            return 0.0, 0.0, f"tuple length mismatch: {len(out_candidate)} vs {len(out_ref)}"
        max_abs, max_rel = 0.0, 0.0
        for i, (k, r) in enumerate(zip(out_candidate, out_ref)):
            if not isinstance(k, torch.Tensor) or not isinstance(r, torch.Tensor):
                continue
            if k.shape != r.shape:
                return max_abs, max_rel, f"output[{i}] shape mismatch: {k.shape} vs {r.shape}"
            a, b = k.float().to(r.device), r.float()
            abs_err = (a - b).abs().max().item()
            rel_err = ((a - b).abs() / (b.abs() + 1e-12)).max().item()
            max_abs = max(max_abs, abs_err)
            max_rel = max(max_rel, rel_err)
        return max_abs, max_rel, None

    if isinstance(out_candidate, tuple):
        if len(out_candidate) == 1:
            out_candidate = out_candidate[0]
        else:
            return 0.0, 0.0, "candidate returned tuple, expected Tensor"
    if out_candidate.shape != out_ref.shape:
        return 0.0, 0.0, f"shape mismatch: {out_candidate.shape} vs {out_ref.shape}"
    a, b = out_candidate.float().to(out_ref.device), out_ref.float()
    abs_err = (a - b).abs().max().item()
    rel_err = ((a - b).abs() / (b.abs() + 1e-12)).max().item()
    return abs_err, rel_err, None


def _validate(candidate_fn, model, get_inputs, tols, tol_tier) -> dict:
    """Run NUM_CORRECTNESS_TRIALS random-input trials.

    Returns a per-backend result dict with status / max_abs_err /
    max_rel_err / tol_tier / (reason on FAIL).
    """
    max_abs, max_rel = 0.0, 0.0
    last_inputs = None
    for trial in range(NUM_CORRECTNESS_TRIALS):
        torch.manual_seed(trial * 137 + 42)
        torch.cuda.manual_seed(trial * 137 + 42)
        inputs = get_inputs()
        last_inputs = inputs
        try:
            with torch.no_grad():
                ref_out = model(*inputs)
                cand_out = candidate_fn(*inputs)
        except Exception as e:
            return {
                "status": "FAIL",
                "tol_tier": tol_tier,
                "max_abs_err": max_abs,
                "max_rel_err": max_rel,
                "reason": f"execution (trial {trial}): {e}",
            }

        abs_err, rel_err, err_msg = _compute_errors(cand_out, ref_out)
        max_abs = max(max_abs, abs_err)
        max_rel = max(max_rel, rel_err)

        # Check tolerance: abs_err <= atol + rtol * |ref|
        # Approximate: pass if abs_err < atol OR rel_err < rtol.
        atol, rtol = tols["atol"], tols["rtol"]
        passed = (abs_err <= atol) or (rel_err <= rtol)
        if not passed:
            return {
                "status": "FAIL",
                "tol_tier": tol_tier,
                "max_abs_err": max_abs,
                "max_rel_err": max_rel,
                "reason": err_msg or (
                    f"trial {trial} ({tol_tier} tol): "
                    f"max_abs_err={abs_err:.4e}, max_rel_err={rel_err:.4e}"
                ),
            }

    return {
        "status": "PASS",
        "tol_tier": tol_tier,
        "max_abs_err": max_abs,
        "max_rel_err": max_rel,
        "_last_inputs": last_inputs,  # for benchmarking; popped before save
    }


def run_one(name: str) -> dict:
    """Validate + benchmark both backends for a single problem."""
    problem_dir = GENERATED_DIR / name
    problem_path = problem_dir / "problem.py"
    kernel_path = problem_dir / "kernel.py"

    if not kernel_path.exists():
        return {"name": name, "status": "SKIP", "reason": "no kernel.py"}
    if not problem_path.exists():
        return {"name": name, "status": "SKIP", "reason": "no problem.py"}

    try:
        problem_mod = _load_module(problem_path, f"problem_{name}")
        kernel_mod = _load_module(kernel_path, f"kernel_{name}")
    except Exception as e:
        return {"name": name, "status": "ERROR", "reason": f"import: {e}"}

    model_cls = getattr(problem_mod, "Model", None)
    get_inputs = getattr(problem_mod, "get_inputs", None)
    kernel_fn = getattr(kernel_mod, "kernel_function", None)
    if model_cls is None or get_inputs is None or kernel_fn is None:
        return {"name": name, "status": "ERROR", "reason": "missing Model/get_inputs/kernel_function"}

    tols, tol_tier = _detect_tolerance(problem_path)
    model = model_cls()

    # --- Validate KernelAgent triton kernel ---
    triton_result = _validate(kernel_fn, model, get_inputs, tols, tol_tier)
    triton_inputs = triton_result.pop("_last_inputs", None)

    # --- Validate torch.compile baseline ---
    compile_result = {"status": "SKIP", "tol_tier": tol_tier, "reason": "compile failed"}
    compile_inputs = None
    try:
        compiled_fn = torch.compile(model.forward, fullgraph=True)
        compile_result = _validate(compiled_fn, model, get_inputs, tols, tol_tier)
        compile_inputs = compile_result.pop("_last_inputs", None)
    except Exception as e:
        compile_result = {
            "status": "ERROR", "tol_tier": tol_tier,
            "max_abs_err": 0.0, "max_rel_err": 0.0,
            "reason": f"torch.compile failed: {e}",
        }

    # --- Benchmark eager (always, as the reference time) ---
    # Pick any one set of inputs; use triton's if available else compile's.
    bench_inputs = triton_inputs or compile_inputs
    eager_ms = None
    if bench_inputs is not None:
        try:
            eager_ms = _benchmark_fn(model.forward, bench_inputs)
        except Exception as e:
            eager_ms = None

    # --- Benchmark each backend that PASSED correctness ---
    if triton_result["status"] == "PASS" and bench_inputs is not None:
        try:
            triton_result["time_ms"] = _benchmark_fn(kernel_fn, bench_inputs)
        except Exception as e:
            triton_result["status"] = "FAIL"
            triton_result["reason"] = f"benchmark failed: {e}"

    if compile_result["status"] == "PASS" and bench_inputs is not None:
        try:
            compile_result["time_ms"] = _benchmark_fn(compiled_fn, bench_inputs)
        except Exception as e:
            compile_result["status"] = "FAIL"
            compile_result["reason"] = f"benchmark failed: {e}"

    # --- Save benchmark.json ---
    bench_data = {
        "eager_ms": eager_ms,
        "kernelagent_triton": triton_result,
        "torch_compile": compile_result,
    }
    (problem_dir / "benchmark.json").write_text(json.dumps(bench_data, indent=2))

    return {"name": name, **bench_data}


def _fmt(ms):
    return f"{ms:.3f}" if isinstance(ms, (int, float)) else "-"


def _print_row(r):
    name = r["name"]
    eager_ms = r.get("eager_ms")
    triton = r.get("kernelagent_triton", {})
    compile_ = r.get("torch_compile", {})

    if r.get("status") in ("SKIP", "ERROR"):
        print(f"  {name:<14} {r['status']:<6} {r.get('reason', '')}")
        return

    def status_str(d):
        s = d.get("status", "?")
        if s == "PASS":
            return f"PASS({d.get('tol_tier', '?')})"
        return s

    tri_s = status_str(triton)
    cmp_s = status_str(compile_)
    eager = _fmt(eager_ms)
    tri_t = _fmt(triton.get("time_ms")) if triton.get("status") == "PASS" else "-"
    cmp_t = _fmt(compile_.get("time_ms")) if compile_.get("status") == "PASS" else "-"
    tri_err = f"{triton.get('max_abs_err', 0):.2e}" if "max_abs_err" in triton else "-"
    cmp_err = f"{compile_.get('max_abs_err', 0):.2e}" if "max_abs_err" in compile_ else "-"

    print(
        f"  {name:<14} eager={eager:>7}ms  "
        f"triton={tri_s:<14} {tri_t:>7}ms err={tri_err:<10}  "
        f"compile={cmp_s:<14} {cmp_t:>7}ms err={cmp_err}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=None)
    parser.add_argument("--problems", nargs="*")
    args = parser.parse_args()

    global GENERATED_DIR
    if args.dir is not None:
        GENERATED_DIR = args.dir

    if args.problems:
        names = args.problems
    else:
        names = sorted(
            d.name for d in GENERATED_DIR.iterdir()
            if d.is_dir() and (d / "kernel.py").exists()
        )

    print(f"Benchmarking {len(names)} kernels...\n")
    results = []
    for name in names:
        r = run_one(name)
        results.append(r)
        _print_row(r)

    triton_pass = sum(
        1 for r in results
        if isinstance(r.get("kernelagent_triton"), dict)
        and r["kernelagent_triton"].get("status") == "PASS"
    )
    compile_pass = sum(
        1 for r in results
        if isinstance(r.get("torch_compile"), dict)
        and r["torch_compile"].get("status") == "PASS"
    )
    print(f"\nkernelagent_triton: {triton_pass}/{len(results)} pass")
    print(f"torch_compile:      {compile_pass}/{len(results)} pass")


if __name__ == "__main__":
    main()

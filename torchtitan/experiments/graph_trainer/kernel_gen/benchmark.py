#!/usr/bin/env python3
"""Benchmark and validate all generated kernels against their PyTorch reference.

For each kernel in generated/<name>/:
  1. Correctness: compare kernel_function output to Model.forward from problem.py
  2. Benchmark: measure wall-clock time for both (CUDA events, 100 iterations)
  3. Report: speedup, max error, pass/fail

Usage:
  python -m torchtitan.experiments.graph_trainer.kernel_gen.benchmark [--problems PROB1 PROB2 ...]
  python -m torchtitan.experiments.graph_trainer.kernel_gen.benchmark  # runs all
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path

import torch

DEFAULT_GENERATED_DIR = Path(__file__).parent / "generated"
GENERATED_DIR = DEFAULT_GENERATED_DIR  # overridden by --dir

# Tolerance tiers:
#   bitwise: atol=0, rtol=0 — for pure elementwise/view ops (no reductions)
#   reduction: rtol=1e-3, atol=1e-5 — for ops involving reductions (sum, mean, norm)
#     Reductions accumulate floating point in different orders, so bitwise is impossible.
#   complex: rtol=1e-3, atol=1e-4 — for complex arithmetic (mul, view_as_complex)
TOLERANCE_BITWISE = {"rtol": 0, "atol": 0}
TOLERANCE_REDUCTION = {"rtol": 1e-3, "atol": 1e-5}
TOLERANCE_COMPLEX = {"rtol": 1e-3, "atol": 1e-4}

# Ops that involve reduction (order-dependent accumulation)
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
    """Auto-detect the right tolerance from the ops in problem.py."""
    content = problem_path.read_text()
    has_reduction = any(op in content for op in _REDUCTION_OPS)
    has_complex = any(op in content for op in _COMPLEX_OPS)
    if has_reduction:
        return TOLERANCE_REDUCTION, "reduction"
    if has_complex:
        return TOLERANCE_COMPLEX, "complex"
    return TOLERANCE_BITWISE, "bitwise"


def _load_module(path: Path, module_name: str):
    """Load a Python module from path.

    For problem.py files with a description preamble, writes a clean
    temp copy then imports it via importlib (not exec) — this is required
    because @triton.jit needs inspect.getsourcelines() to find the source.
    """
    import importlib.util
    import tempfile

    content = path.read_text()
    idx = content.find("import torch")
    if idx > 0:
        # Has preamble — write a clean temp file
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
    """Benchmark a callable with CUDA event timing."""
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


def _compare_outputs(out_kernel, out_ref, tols):
    """Compare kernel output to reference, handling tuples/lists."""
    if isinstance(out_ref, (tuple, list)):
        if not isinstance(out_kernel, (tuple, list)):
            return False, "kernel returned Tensor, expected tuple"
        if len(out_kernel) != len(out_ref):
            return False, f"tuple length mismatch: {len(out_kernel)} vs {len(out_ref)}"
        max_err = 0.0
        for i, (k, r) in enumerate(zip(out_kernel, out_ref)):
            if not isinstance(k, torch.Tensor) or not isinstance(r, torch.Tensor):
                continue
            if k.shape != r.shape:
                return False, f"output[{i}] shape mismatch: {k.shape} vs {r.shape}"
            try:
                k_f = k.float().to(r.device)
                r_f = r.float()
            except Exception:
                k_f = k.float().cpu()
                r_f = r.float().cpu()
            err = (k_f - r_f).abs().max().item()
            max_err = max(max_err, err)
            if not torch.allclose(k_f, r_f, **tols):
                return False, f"output[{i}] numerical mismatch (max_err={err:.4e})"
        return True, max_err
    else:
        if isinstance(out_kernel, tuple):
            if len(out_kernel) == 1:
                out_kernel = out_kernel[0]
            else:
                return False, "kernel returned tuple, expected Tensor"
        if out_kernel.shape != out_ref.shape:
            return False, f"shape mismatch: {out_kernel.shape} vs {out_ref.shape}"
        if out_kernel.dtype != out_ref.dtype:
            # Allow f32 vs bf16 if close
            pass
        try:
            k_f = out_kernel.float().to(out_ref.device)
            r_f = out_ref.float()
        except Exception:
            k_f = out_kernel.float().cpu()
            r_f = out_ref.float().cpu()
        max_err = (k_f - r_f).abs().max().item()
        if not torch.allclose(k_f, r_f, **tols):
            return False, f"numerical mismatch (max_err={max_err:.4e})"
        return True, max_err


def run_one(name: str) -> dict:
    """Validate and benchmark a single kernel."""
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

    # --- Correctness (multiple trials, auto-detected tolerance) ---
    tols, tol_tier = _detect_tolerance(problem_path)
    model = model_cls()
    max_err = 0.0
    for trial in range(NUM_CORRECTNESS_TRIALS):
        try:
            torch.manual_seed(trial * 137 + 42)
            torch.cuda.manual_seed(trial * 137 + 42)
            inputs = get_inputs()
            with torch.no_grad():
                ref_out = model(*inputs)
                kernel_out = kernel_fn(*inputs)
        except Exception as e:
            return {"name": name, "status": "FAIL", "reason": f"execution (trial {trial}): {e}"}

        ok, detail = _compare_outputs(kernel_out, ref_out, tols)
        if not ok:
            return {
                "name": name, "status": "FAIL",
                "reason": f"trial {trial} ({tol_tier} tol): {detail}",
            }
        if isinstance(detail, (int, float)):
            max_err = max(max_err, detail)

    # --- Benchmark ---
    try:
        ref_fn = model.forward
        ref_ms = _benchmark_fn(ref_fn, inputs)
        kern_ms = _benchmark_fn(kernel_fn, inputs)
        speedup = ref_ms / kern_ms if kern_ms > 0 else float("inf")

        # torch.compile baseline
        try:
            compiled_fn = torch.compile(model.forward, fullgraph=True)
            # warmup compile
            compiled_fn(*inputs)
            compiled_fn(*inputs)
            compile_ms = _benchmark_fn(compiled_fn, inputs)
        except Exception:
            compile_ms = None
    except Exception as e:
        return {
            "name": name, "status": "PASS", "max_err": max_err,
            "reason": f"correctness OK, benchmark failed: {e}",
        }

    # Save benchmark.json for the fused op registry
    import json
    bench_data = {
        "eager_ms": ref_ms,
        "triton_ms": kern_ms,
    }
    if compile_ms is not None:
        bench_data["compile_ms"] = compile_ms
    (problem_dir / "benchmark.json").write_text(json.dumps(bench_data, indent=2))

    return {
        "name": name,
        "status": "PASS",
        "max_err": max_err,
        "tol_tier": tol_tier,
        "ref_ms": ref_ms,
        "compile_ms": compile_ms,
        "kern_ms": kern_ms,
        "speedup": speedup,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=None,
                        help="Directory with problem subdirs (default: torchtitan/experiments/graph_trainer/kernel_gen/generated)")
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
        print(f"  {name}...", end=" ", flush=True)
        r = run_one(name)
        results.append(r)
        if r["status"] == "PASS" and "kern_ms" in r:
            comp_str = f"  compile={r['compile_ms']:.3f}ms" if r.get("compile_ms") else ""
            tier = r.get("tol_tier", "?")
            print(
                f"PASS [{tier}]  ref={r['ref_ms']:.3f}ms{comp_str}  kern={r['kern_ms']:.3f}ms  "
                f"speedup={r['speedup']:.2f}x  max_err={r['max_err']:.2e}"
            )
        elif r["status"] == "PASS":
            print(f"PASS  max_err={r['max_err']:.2e}  (no benchmark)")
        else:
            print(f"{r['status']}  {r.get('reason', '')}")

    # Summary table
    print("\n" + "=" * 115)
    print(f"{'Kernel':<16} {'Status':<6} {'Tol':<10} {'Eager':>8} {'Compile':>8} {'Triton':>8} {'vs Eager':>9} {'vs Compile':>11} {'Max Err':>10}")
    print("-" * 115)
    for r in results:
        ref = f"{r['ref_ms']:.3f}" if "ref_ms" in r else "-"
        comp = f"{r['compile_ms']:.3f}" if r.get("compile_ms") else "-"
        kern = f"{r['kern_ms']:.3f}" if "kern_ms" in r else "-"
        spd = f"{r['speedup']:.2f}x" if "speedup" in r else "-"
        if r.get("compile_ms") and r.get("kern_ms") and r["kern_ms"] > 0:
            vs_comp = f"{r['compile_ms'] / r['kern_ms']:.2f}x"
        else:
            vs_comp = "-"
        err = f"{r['max_err']:.2e}" if "max_err" in r else "-"
        tier = r.get("tol_tier", "-")
        reason = f"  ({r['reason'][:60]})" if r["status"] != "PASS" else ""
        print(f"{r['name']:<16} {r['status']:<6} {tier:<10} {ref:>8} {comp:>8} {kern:>8} {spd:>9} {vs_comp:>11} {err:>10}{reason}")
    print("=" * 115)

    passed = sum(1 for r in results if r["status"] == "PASS")
    print(f"\n{passed}/{len(results)} passed")


if __name__ == "__main__":
    main()

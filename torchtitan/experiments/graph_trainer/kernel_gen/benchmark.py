#!/usr/bin/env python3
"""Benchmark and validate generated kernels against eager.

Eager is truth. torch.compile is trusted as "good enough" — we measure
its drift from eager but always mark it PASS. The triton kernel is
accepted if it satisfies an effective allclose against eager:

    |triton - eager| <= atol_eff + rtol * |eager|

where ``atol_eff = max(tol_tier_atol, compile_max_abs_err)`` — the tier
floor expanded to whatever compile already drifts to. This collapses
"within tolerance" and "no worse than compile" into a single test and
avoids the rel_err blow-up at the bf16 noise floor.

Errors are reported as ``max_tol_units = max(|a-b| / (atol+rtol*|b|))``:
unitless, ``<= 1.0`` means PASS.

For each kernel in <dir>/<name>/:
  1. Measure torch.compile vs eager → record max_abs / max_tol_units, PASS.
  2. Measure triton vs eager under atol_eff → PASS iff max_tol_units <= 1.
  3. Benchmark eager, triton (if PASS), and compile (if PASS).
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


def _pairwise_errors(
    a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float
) -> tuple[float, float]:
    """Return (max_abs_err, max_tol_units) for a vs reference b.

    Uses the standard allclose formula: an element passes iff
    ``|a-b| <= atol + rtol * |b|``. ``max_tol_units`` is the worst-case
    ``|a-b| / (atol + rtol*|b|)`` across all elements — a unitless
    "how far over budget" number. ``<= 1.0`` means every element
    satisfies allclose. Naturally handles near-zero |b| because the
    ``atol`` term floors the denominator.
    """
    diff = (a - b).abs()
    max_abs = diff.max().item() if diff.numel() else 0.0
    budget = atol + rtol * b.abs()
    units = diff / budget  # budget >= atol > 0 so no division-by-zero
    max_units = units.max().item() if units.numel() else 0.0
    return max_abs, max_units


def _compute_errors(
    out_candidate, out_ref, atol: float, rtol: float
) -> tuple[float, float, str | None]:
    """Compute (max_abs_err, max_tol_units, error_message_or_None) over
    a possibly-nested tuple of tensors. Casts to float32 for comparison.
    """
    if isinstance(out_ref, (tuple, list)):
        if not isinstance(out_candidate, (tuple, list)):
            return 0.0, 0.0, "candidate returned Tensor, expected tuple"
        if len(out_candidate) != len(out_ref):
            return 0.0, 0.0, f"tuple length mismatch: {len(out_candidate)} vs {len(out_ref)}"
        max_abs, max_units = 0.0, 0.0
        for i, (k, r) in enumerate(zip(out_candidate, out_ref)):
            if not isinstance(k, torch.Tensor) or not isinstance(r, torch.Tensor):
                continue
            if k.shape != r.shape:
                return max_abs, max_units, f"output[{i}] shape mismatch: {k.shape} vs {r.shape}"
            a, b = k.float().to(r.device), r.float()
            abs_err, units = _pairwise_errors(a, b, atol, rtol)
            max_abs = max(max_abs, abs_err)
            max_units = max(max_units, units)
        return max_abs, max_units, None

    if isinstance(out_candidate, tuple):
        if len(out_candidate) == 1:
            out_candidate = out_candidate[0]
        else:
            return 0.0, 0.0, "candidate returned tuple, expected Tensor"
    if out_candidate.shape != out_ref.shape:
        return 0.0, 0.0, f"shape mismatch: {out_candidate.shape} vs {out_ref.shape}"
    a, b = out_candidate.float().to(out_ref.device), out_ref.float()
    abs_err, units = _pairwise_errors(a, b, atol, rtol)
    return abs_err, units, None


def _measure_diff(
    candidate_fn, eager_fn, get_inputs, atol: float, rtol: float
) -> dict:
    """Run NUM_CORRECTNESS_TRIALS trials comparing candidate to eager.

    Returns max_abs_err / max_tol_units across trials plus the last input
    tuple. Does NOT decide pass/fail — caller does. On execution or
    shape error, sets ``_exec_error`` and stops early.
    """
    max_abs, max_units = 0.0, 0.0
    last_inputs = None
    for trial in range(NUM_CORRECTNESS_TRIALS):
        torch.manual_seed(trial * 137 + 42)
        torch.cuda.manual_seed(trial * 137 + 42)
        inputs = get_inputs()
        last_inputs = inputs
        try:
            with torch.no_grad():
                ref_out = eager_fn(*inputs)
                cand_out = candidate_fn(*inputs)
        except Exception as e:
            return {
                "max_abs_err": max_abs,
                "max_tol_units": max_units,
                "_last_inputs": last_inputs,
                "_exec_error": f"execution (trial {trial}): {e}",
            }
        abs_err, units, err_msg = _compute_errors(cand_out, ref_out, atol, rtol)
        if err_msg:
            return {
                "max_abs_err": max_abs,
                "max_tol_units": max_units,
                "_last_inputs": last_inputs,
                "_exec_error": err_msg,
            }
        max_abs = max(max_abs, abs_err)
        max_units = max(max_units, units)
    return {
        "max_abs_err": max_abs,
        "max_tol_units": max_units,
        "_last_inputs": last_inputs,
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
    atol, rtol = tols["atol"], tols["rtol"]
    model = model_cls()

    # --- Measure torch.compile drift from eager (trusted, always PASS) ---
    # Reported with the tier (atol, rtol); status is PASS regardless.
    compiled_fn = None
    compile_abs = 0.0
    compile_result: dict
    try:
        compiled_fn = torch.compile(model.forward, fullgraph=True)
        res = _measure_diff(compiled_fn, model, get_inputs, atol, rtol)
        compile_inputs = res.pop("_last_inputs", None)
        exec_err = res.pop("_exec_error", None)
        compile_abs = res["max_abs_err"]
        compile_units = res["max_tol_units"]
        if exec_err is not None:
            compile_result = {
                "status": "ERROR", "tol_tier": tol_tier,
                "max_abs_err": compile_abs, "max_tol_units": compile_units,
                "reason": exec_err,
            }
        else:
            compile_result = {
                "status": "PASS", "tol_tier": tol_tier,
                "max_abs_err": compile_abs, "max_tol_units": compile_units,
                "note": "trusted baseline",
            }
    except Exception as e:
        compile_inputs = None
        compile_result = {
            "status": "ERROR", "tol_tier": tol_tier,
            "max_abs_err": 0.0, "max_tol_units": 0.0,
            "reason": f"torch.compile failed: {e}",
        }

    # --- Measure triton drift from eager; accept via effective allclose ---
    # Effective atol is loosened to whatever compile already drifts to, so
    # "no worse than compile" passes automatically. rtol stays as the tier
    # value. PASS iff every element satisfies |triton-eager| <= atol_eff +
    # rtol*|eager|, i.e. max_tol_units <= 1.0.
    atol_eff = max(atol, compile_abs)
    res = _measure_diff(kernel_fn, model, get_inputs, atol_eff, rtol)
    triton_inputs = res.pop("_last_inputs", None)
    exec_err = res.pop("_exec_error", None)
    triton_abs = res["max_abs_err"]
    triton_units = res["max_tol_units"]
    if exec_err is not None:
        triton_result = {
            "status": "FAIL", "tol_tier": tol_tier,
            "max_abs_err": triton_abs, "max_tol_units": triton_units,
            "reason": exec_err,
        }
    elif triton_units <= 1.0:
        triton_result = {
            "status": "PASS", "tol_tier": tol_tier,
            "max_abs_err": triton_abs, "max_tol_units": triton_units,
            "atol_eff": atol_eff, "rtol": rtol,
        }
    else:
        triton_result = {
            "status": "FAIL", "tol_tier": tol_tier,
            "max_abs_err": triton_abs, "max_tol_units": triton_units,
            "atol_eff": atol_eff, "rtol": rtol,
            "reason": (
                f"max_tol_units={triton_units:.3f} > 1.0 "
                f"(atol_eff={atol_eff:.4e}, rtol={rtol:.4e}, "
                f"max_abs_err={triton_abs:.4e})"
            ),
        }
    bench_inputs = triton_inputs or compile_inputs

    # --- Benchmark eager (always, as the reference time) ---
    eager_ms = None
    if bench_inputs is not None:
        try:
            eager_ms = _benchmark_fn(model.forward, bench_inputs)
        except Exception:
            eager_ms = None

    # --- Benchmark each backend that PASSED correctness ---
    if triton_result["status"] == "PASS" and bench_inputs is not None:
        try:
            triton_result["time_ms"] = _benchmark_fn(kernel_fn, bench_inputs)
        except Exception as e:
            triton_result["status"] = "FAIL"
            triton_result["reason"] = f"benchmark failed: {e}"

    if compile_result["status"] == "PASS" and bench_inputs is not None and compiled_fn is not None:
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

    def err_str(d):
        if "max_tol_units" not in d:
            return "-"
        u = d["max_tol_units"]
        a = d.get("max_abs_err", 0.0)
        return f"{u:.2f}u/{a:.2e}"

    print(
        f"  {name:<14} eager={eager:>7}ms  "
        f"triton={tri_s:<14} {tri_t:>7}ms err={err_str(triton):<18}  "
        f"compile={cmp_s:<14} {cmp_t:>7}ms err={err_str(compile_)}"
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

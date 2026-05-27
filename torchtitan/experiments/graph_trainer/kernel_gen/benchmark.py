#!/usr/bin/env python3
"""Benchmark and validate generated kernels against eager.

Eager is truth. torch.compile is trusted as "good enough" — we measure
its drift from eager but always mark it PASS. Triton kernels are
accepted if they satisfy an effective allclose against eager:

    |triton - eager| <= atol_eff + rtol * |eager|

where ``atol_eff = max(tol_tier_atol, compile_max_abs_err)`` — the tier
floor expanded to whatever compile already drifts to. This collapses
"within tolerance" and "no worse than compile" into a single test and
avoids the rel_err blow-up at the bf16 noise floor.

Errors are reported as ``max_tol_units = max(|a-b| / (atol+rtol*|b|))``:
unitless, ``<= 1.0`` means PASS.

Four backends are evaluated per problem in <dir>/<name>/:
  - torch_compile              : torch.compile(model.forward), trusted baseline.
  - torch_compile_max_autotune : torch.compile with mode="max-autotune".
  - kernelagent_triton         : initial kernel.py from KernelAgent.
  - kernelagent_triton_opt     : NCU-optimized optimized_kernel.py (if present).

Each backend gets a status / max_abs_err / max_tol_units / time_ms entry
in benchmark.json.

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
    the first import so @triton.jit's inspect.getsourcelines() works."""
    import tempfile
    content = path.read_text()
    # Find the first 'import' statement (could be 'import triton' or 'import torch').
    idx = min(
        (content.find(s) for s in ("import triton", "import torch") if content.find(s) >= 0),
        default=-1,
    )
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


def _validate_triton(
    kernel_fn, model, get_inputs, atol_eff: float, rtol: float, tol_tier: str
) -> tuple[dict, object]:
    """Run triton vs eager with effective atol; return (result_dict, last_inputs).

    PASS iff every element satisfies the effective allclose, equivalently
    ``max_tol_units <= 1.0`` with ``atol=atol_eff``.
    """
    res = _measure_diff(kernel_fn, model, get_inputs, atol_eff, rtol)
    last_inputs = res.pop("_last_inputs", None)
    exec_err = res.pop("_exec_error", None)
    abs_err = res["max_abs_err"]
    units = res["max_tol_units"]
    if exec_err is not None:
        return {
            "status": "FAIL", "tol_tier": tol_tier,
            "max_abs_err": abs_err, "max_tol_units": units,
            "reason": exec_err,
        }, last_inputs
    if units <= 1.0:
        return {
            "status": "PASS", "tol_tier": tol_tier,
            "max_abs_err": abs_err, "max_tol_units": units,
            "atol_eff": atol_eff, "rtol": rtol,
        }, last_inputs
    return {
        "status": "FAIL", "tol_tier": tol_tier,
        "max_abs_err": abs_err, "max_tol_units": units,
        "atol_eff": atol_eff, "rtol": rtol,
        "reason": (
            f"max_tol_units={units:.3f} > 1.0 "
            f"(atol_eff={atol_eff:.4e}, rtol={rtol:.4e}, "
            f"max_abs_err={abs_err:.4e})"
        ),
    }, last_inputs


def run_one(name: str) -> dict:
    """Validate + benchmark all backends for a single problem.

    Backends: torch_compile, kernelagent_triton (kernel.py),
    kernelagent_triton_opt (optimized_kernel.py, if present).
    """
    problem_dir = GENERATED_DIR / name
    problem_path = problem_dir / "problem.py"
    kernel_path = problem_dir / "kernel.py"
    opt_kernel_path = problem_dir / "optimized_kernel.py"

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

    opt_kernel_fn = None
    opt_load_error = None
    if opt_kernel_path.exists():
        try:
            opt_mod = _load_module(opt_kernel_path, f"kernel_opt_{name}")
            opt_kernel_fn = getattr(opt_mod, "kernel_function", None)
            if opt_kernel_fn is None:
                opt_load_error = "missing kernel_function"
        except Exception as e:
            opt_load_error = f"import: {e}"

    tols, tol_tier = _detect_tolerance(problem_path)
    atol, rtol = tols["atol"], tols["rtol"]
    model = model_cls()

    # --- Measure torch.compile drift from eager (trusted, always PASS) ---
    # Two compile variants: default and max-autotune. Both treated as trusted
    # baselines (status=PASS regardless of drift).
    def _measure_compile(mode: str | None) -> tuple[object, dict, object]:
        try:
            kwargs = {"fullgraph": True}
            if mode is not None:
                kwargs["mode"] = mode
            fn = torch.compile(model.forward, **kwargs)
            res = _measure_diff(fn, model, get_inputs, atol, rtol)
            inputs = res.pop("_last_inputs", None)
            exec_err = res.pop("_exec_error", None)
            if exec_err is not None:
                result = {
                    "status": "ERROR", "tol_tier": tol_tier,
                    "max_abs_err": res["max_abs_err"], "max_tol_units": res["max_tol_units"],
                    "reason": exec_err,
                }
            else:
                result = {
                    "status": "PASS", "tol_tier": tol_tier,
                    "max_abs_err": res["max_abs_err"], "max_tol_units": res["max_tol_units"],
                    "note": "trusted baseline",
                }
            return fn, result, inputs
        except Exception as e:
            return None, {
                "status": "ERROR", "tol_tier": tol_tier,
                "max_abs_err": 0.0, "max_tol_units": 0.0,
                "reason": f"torch.compile(mode={mode!r}) failed: {e}",
            }, None

    compiled_fn, compile_result, compile_inputs = _measure_compile(None)
    compiled_ma_fn, compile_ma_result, compile_ma_inputs = _measure_compile("max-autotune")
    compile_abs = compile_result.get("max_abs_err", 0.0)

    # --- Validate triton kernels against eager via effective allclose ---
    # atol_eff = max(tier_atol, compile_drift); rtol stays the tier value.
    atol_eff = max(atol, compile_abs)
    triton_result, triton_inputs = _validate_triton(
        kernel_fn, model, get_inputs, atol_eff, rtol, tol_tier
    )

    if opt_kernel_fn is not None:
        opt_result, opt_inputs = _validate_triton(
            opt_kernel_fn, model, get_inputs, atol_eff, rtol, tol_tier
        )
    elif opt_load_error is not None:
        opt_result = {
            "status": "ERROR", "tol_tier": tol_tier,
            "max_abs_err": 0.0, "max_tol_units": 0.0,
            "reason": f"load optimized_kernel.py: {opt_load_error}",
        }
        opt_inputs = None
    else:
        opt_result = {"status": "SKIP", "reason": "no optimized_kernel.py"}
        opt_inputs = None

    bench_inputs = triton_inputs or opt_inputs or compile_inputs or compile_ma_inputs

    # --- Benchmark eager (always, as the reference time) ---
    eager_ms = None
    if bench_inputs is not None:
        try:
            eager_ms = _benchmark_fn(model.forward, bench_inputs)
        except Exception:
            eager_ms = None

    # --- Benchmark each backend that PASSED correctness ---
    def _maybe_bench(result: dict, fn) -> None:
        if result.get("status") == "PASS" and bench_inputs is not None and fn is not None:
            try:
                result["time_ms"] = _benchmark_fn(fn, bench_inputs)
            except Exception as e:
                result["status"] = "FAIL"
                result["reason"] = f"benchmark failed: {e}"

    _maybe_bench(triton_result, kernel_fn)
    _maybe_bench(opt_result, opt_kernel_fn)
    _maybe_bench(compile_result, compiled_fn)
    _maybe_bench(compile_ma_result, compiled_ma_fn)

    # --- Save benchmark.json ---
    bench_data = {
        "eager_ms": eager_ms,
        "kernelagent_triton": triton_result,
        "kernelagent_triton_opt": opt_result,
        "torch_compile": compile_result,
        "torch_compile_max_autotune": compile_ma_result,
    }
    (problem_dir / "benchmark.json").write_text(json.dumps(bench_data, indent=2))

    return {"name": name, **bench_data}


def _fmt(ms):
    return f"{ms:.3f}" if isinstance(ms, (int, float)) else "-"


def _print_row(r):
    name = r["name"]
    eager_ms = r.get("eager_ms")
    triton = r.get("kernelagent_triton", {})
    triton_opt = r.get("kernelagent_triton_opt", {})
    compile_ = r.get("torch_compile", {})
    compile_ma = r.get("torch_compile_max_autotune", {})

    if r.get("status") in ("SKIP", "ERROR"):
        print(f"  {name:<14} {r['status']:<6} {r.get('reason', '')}")
        return

    def short_status(d):
        s = d.get("status", "?")
        if s == "PASS":
            return "PASS"
        if s == "FAIL":
            return "FAIL"
        if s == "ERROR":
            return "ERR"
        if s == "SKIP":
            return "-"
        return s

    def cell(d):
        t = d.get("time_ms")
        if d.get("status") == "PASS" and t is not None:
            return f"{t:>8.3f}"
        return f"{short_status(d):>8}"

    eager = _fmt(eager_ms)
    print(
        f"  {name:<14} eager={eager:>7}ms  "
        f"triton={cell(triton)}  "
        f"opt={cell(triton_opt)}  "
        f"compile={cell(compile_)}  "
        f"compile_ma={cell(compile_ma)}"
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

    def _count_pass(key):
        return sum(
            1 for r in results
            if isinstance(r.get(key), dict) and r[key].get("status") == "PASS"
        )

    n = len(results)
    n_opt = sum(
        1 for r in results
        if isinstance(r.get("kernelagent_triton_opt"), dict)
        and r["kernelagent_triton_opt"].get("status") != "SKIP"
    )
    print(f"\nkernelagent_triton:         {_count_pass('kernelagent_triton')}/{n} pass")
    print(f"kernelagent_triton_opt:     {_count_pass('kernelagent_triton_opt')}/{n_opt} pass")
    print(f"torch_compile:              {_count_pass('torch_compile')}/{n} pass")
    print(f"torch_compile_max_autotune: {_count_pass('torch_compile_max_autotune')}/{n} pass")


if __name__ == "__main__":
    main()

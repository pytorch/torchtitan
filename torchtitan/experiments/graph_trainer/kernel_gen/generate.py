#!/usr/bin/env python3
"""Run KernelAgent on all problem files: generate + NCU-guided optimize.

For each problem in generated/<name>/:
  1. Generate a correct Triton kernel (if kernel.py doesn't exist)
  2. Optimize with NCU profiling (if optimized_kernel.py doesn't exist)

Usage:
  python -m torchtitan.experiments.graph_trainer.kernel_gen.generate [--problems PROB1 PROB2 ...]
  python -m torchtitan.experiments.graph_trainer.kernel_gen.generate  # runs all
  python -m torchtitan.experiments.graph_trainer.kernel_gen.generate --skip-optimize  # generation only
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


DEFAULT_GENERATED_DIR = Path(__file__).parent / "generated"
GENERATED_DIR = DEFAULT_GENERATED_DIR  # overridden by --dir


def _setup_env():
    """Set up API key and proxy in current process (inherited by forks)."""
    import os
    os.environ.setdefault("LOG_LEVEL", "WARNING")
    sys.path.insert(0, os.path.expanduser("~/local/KernelAgent"))
    from torchtitan.experiments.graph_trainer.kernel_gen.kernelagent_bridge import _ensure_api_key, _ensure_proxy
    _ensure_api_key()
    _ensure_proxy()


def generate_one(name: str) -> dict:
    """Generate a correct Triton kernel for a single problem."""
    import os
    os.environ.setdefault("LOG_LEVEL", "WARNING")
    sys.path.insert(0, os.path.expanduser("~/local/KernelAgent"))

    problem_dir = GENERATED_DIR / name
    problem_file = problem_dir / "problem.py"
    if not problem_file.exists():
        return {"name": name, "success": False, "message": f"No problem.py in {problem_dir}"}

    problem_text = problem_file.read_text()
    log_dir = str(problem_dir / "logs")

    from torchtitan.experiments.graph_trainer.kernel_gen.kernelagent_bridge import generate_kernel

    print(f"[{name}] Generating kernel...", flush=True)
    try:
        result = generate_kernel(
            problem_text,
            num_workers=4,
            max_rounds=10,
            output_dir=log_dir,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"name": name, "success": False, "message": str(e)}

    if result.get("success"):
        kernel_code = result["kernel_code"]
        kernel_path = problem_dir / "kernel.py"
        kernel_path.write_text(kernel_code)
        print(f"[{name}] Generated — saved to {kernel_path}")
        return {"name": name, "success": True, "kernel_path": str(kernel_path)}
    else:
        print(f"[{name}] Generation FAILED — {result.get('message', 'unknown error')}")
        return {"name": name, "success": False, "message": result.get("message", "")}


def optimize_one(name: str, opt_rounds: int = 5) -> dict:
    """NCU-guided optimization of an existing kernel."""
    import os
    os.environ.setdefault("LOG_LEVEL", "WARNING")
    sys.path.insert(0, os.path.expanduser("~/local/KernelAgent"))

    problem_dir = GENERATED_DIR / name
    kernel_path = problem_dir / "kernel.py"
    problem_path = problem_dir / "problem.py"

    if not kernel_path.exists():
        return {"name": name, "success": False, "message": "no kernel.py"}
    if not problem_path.exists():
        return {"name": name, "success": False, "message": "no problem.py"}

    from torchtitan.experiments.graph_trainer.kernel_gen.kernelagent_bridge import optimize_kernel

    kernel_code = kernel_path.read_text()
    opt_dir = str(problem_dir / "opt_logs")

    print(f"[{name}] Optimizing with NCU ({opt_rounds} rounds)...", flush=True)
    try:
        result = optimize_kernel(
            kernel_code=kernel_code,
            problem_path=problem_path,
            strategy="greedy",
            num_workers=1,
            max_rounds=opt_rounds,
            output_dir=opt_dir,
        )
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"name": name, "success": False, "message": str(e)}

    out = {
        "name": name,
        "success": result.get("success", False),
        "best_time_ms": result.get("best_time_ms"),
        "pytorch_baseline_ms": result.get("pytorch_baseline_ms"),
        "pytorch_compile_ms": result.get("pytorch_compile_ms"),
        "initial_kernel_time_ms": result.get("initial_kernel_time_ms"),
    }

    if result.get("success") and result.get("kernel_code"):
        opt_path = problem_dir / "optimized_kernel.py"
        opt_path.write_text(result["kernel_code"])
        out["optimized_path"] = str(opt_path)

        best = result["best_time_ms"]
        initial = result.get("initial_kernel_time_ms", float("inf"))
        baseline = result.get("pytorch_baseline_ms", float("inf"))
        compile_ms = result.get("pytorch_compile_ms", float("inf"))

        parts = [f"best={best:.4f}ms"]
        if initial != float("inf"):
            parts.append(f"vs_initial={initial / best:.2f}x")
        if baseline != float("inf"):
            parts.append(f"vs_eager={baseline / best:.2f}x")
        if compile_ms != float("inf"):
            parts.append(f"vs_compile={compile_ms / best:.2f}x")
        print(f"[{name}] Optimized — {', '.join(parts)}")
    else:
        msg = result.get("error", result.get("message", ""))
        print(f"[{name}] Optimization FAILED — {msg}")

    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", type=Path, default=None,
                        help="Directory with problem subdirs (default: torchtitan/experiments/graph_trainer/kernel_gen/generated)")
    parser.add_argument("--problems", nargs="*")
    parser.add_argument("--max-parallel", type=int, default=5)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--skip-optimize", action="store_true",
                        help="Skip NCU optimization (generation only)")
    parser.add_argument("--optimize-only", action="store_true",
                        help="Skip generation, only run NCU optimization")
    parser.add_argument("--opt-rounds", type=int, default=5,
                        help="Max NCU optimization rounds per kernel (default: 5)")
    args = parser.parse_args()

    global GENERATED_DIR
    if args.dir is not None:
        GENERATED_DIR = args.dir

    if args.problems:
        names = args.problems
    else:
        names = sorted(
            d.name for d in GENERATED_DIR.iterdir()
            if d.is_dir() and (d / "problem.py").exists()
        )

    _setup_env()

    # --- Phase 1: Generate kernels ---
    gen_results = []
    if not args.optimize_only:
        to_generate = [n for n in names if not (GENERATED_DIR / n / "kernel.py").exists()]
        already = [n for n in names if (GENERATED_DIR / n / "kernel.py").exists()]
        for n in already:
            print(f"[{n}] Skipping generation — kernel.py exists")

        if to_generate:
            print(f"\nGenerating {len(to_generate)} kernels: {', '.join(to_generate)}\n")
            if args.sequential:
                gen_results = [generate_one(n) for n in to_generate]
            else:
                with ProcessPoolExecutor(max_workers=min(args.max_parallel, len(to_generate))) as pool:
                    futures = {pool.submit(generate_one, n): n for n in to_generate}
                    for future in as_completed(futures):
                        try:
                            gen_results.append(future.result())
                        except Exception as e:
                            n = futures[future]
                            gen_results.append({"name": n, "success": False, "message": str(e)})

        gen_ok = sum(1 for r in gen_results if r["success"])
        if gen_results:
            print(f"\nGeneration: {gen_ok}/{len(gen_results)} succeeded\n")

    # --- Phase 2: NCU-guided optimization ---
    opt_results = []
    if not args.skip_optimize:
        to_optimize = [
            n for n in names
            if (GENERATED_DIR / n / "kernel.py").exists()
            and not (GENERATED_DIR / n / "optimized_kernel.py").exists()
        ]
        already_opt = [
            n for n in names
            if (GENERATED_DIR / n / "optimized_kernel.py").exists()
        ]
        for n in already_opt:
            print(f"[{n}] Skipping optimization — optimized_kernel.py exists")

        if to_optimize:
            print(f"\nOptimizing {len(to_optimize)} kernels: {', '.join(to_optimize)}\n")
            # NCU requires exclusive GPU access, so run sequentially
            for n in to_optimize:
                opt_results.append(optimize_one(n, args.opt_rounds))

        opt_ok = sum(1 for r in opt_results if r["success"])
        if opt_results:
            print(f"\nOptimization: {opt_ok}/{len(opt_results)} succeeded\n")

    # --- Phase 3: Benchmark all kernels (eager vs compile vs triton) ---
    import torchtitan.experiments.graph_trainer.kernel_gen.benchmark as bench_mod
    bench_mod.GENERATED_DIR = GENERATED_DIR
    benchmark_one = bench_mod.run_one

    to_benchmark = [
        n for n in names
        if (GENERATED_DIR / n / "kernel.py").exists()
        and not (GENERATED_DIR / n / "benchmark.json").exists()
    ]
    bench_results = []
    if to_benchmark:
        print(f"\nBenchmarking {len(to_benchmark)} kernels...\n")
        for n in to_benchmark:
            print(f"  {n}...", end=" ", flush=True)
            r = benchmark_one(n)
            bench_results.append(r)
            if r["status"] == "PASS" and "kern_ms" in r:
                comp_str = f"  compile={r['compile_ms']:.3f}ms" if r.get("compile_ms") else ""
                print(
                    f"PASS  eager={r['ref_ms']:.3f}ms{comp_str}  triton={r['kern_ms']:.3f}ms  "
                    f"speedup={r['speedup']:.2f}x"
                )
            elif r["status"] == "PASS":
                print(f"PASS  (no benchmark)")
            else:
                print(f"{r['status']}  {r.get('reason', '')[:60]}")

    # --- Summary ---
    print("\n" + "=" * 100)
    print("SUMMARY")
    print("=" * 100)

    if gen_results:
        gen_ok = sum(1 for r in gen_results if r["success"])
        print(f"\nGeneration: {gen_ok}/{len(gen_results)}")
        for r in gen_results:
            status = "OK" if r["success"] else "FAIL"
            msg = f"  ({r.get('message', '')[:60]})" if not r["success"] else ""
            print(f"  {status:>4}  {r['name']}{msg}")

    if bench_results:
        print(f"\n{'Kernel':<16} {'Status':<6} {'Eager':>8} {'Compile':>8} {'Triton':>8} {'vs Eager':>9} {'vs Compile':>11}")
        print("-" * 80)
        for r in bench_results:
            ref = f"{r['ref_ms']:.3f}" if "ref_ms" in r else "-"
            comp = f"{r['compile_ms']:.3f}" if r.get("compile_ms") else "-"
            kern = f"{r['kern_ms']:.3f}" if "kern_ms" in r else "-"
            spd = f"{r['speedup']:.2f}x" if "speedup" in r else "-"
            if r.get("compile_ms") and r.get("kern_ms") and r["kern_ms"] > 0:
                vs_comp = f"{r['compile_ms'] / r['kern_ms']:.2f}x"
            else:
                vs_comp = "-"
            reason = f"  ({r.get('reason', '')[:40]})" if r["status"] != "PASS" else ""
            print(f"  {r['name']:<14} {r['status']:<6} {ref:>8} {comp:>8} {kern:>8} {spd:>9} {vs_comp:>11}{reason}")

    if opt_results:
        print(f"\n{'Kernel':<28} {'Status':<6} {'Eager':>9} {'Compile':>9} {'Initial':>9} {'Best':>9} {'vs Eager':>9}")
        print("-" * 100)
        for r in opt_results:
            status = "OK" if r["success"] else "FAIL"
            eager = f"{r['pytorch_baseline_ms']:.4f}" if r.get("pytorch_baseline_ms") else "-"
            comp = f"{r['pytorch_compile_ms']:.4f}" if r.get("pytorch_compile_ms") else "-"
            initial = f"{r['initial_kernel_time_ms']:.4f}" if r.get("initial_kernel_time_ms") else "-"
            best = f"{r['best_time_ms']:.4f}" if r.get("best_time_ms") and r["best_time_ms"] != float("inf") else "-"
            if r.get("success") and r.get("best_time_ms") and r.get("pytorch_baseline_ms"):
                speedup = f"{r['pytorch_baseline_ms'] / r['best_time_ms']:.2f}x"
            else:
                speedup = "-"
            print(f"  {r['name']:<26} {status:<6} {eager:>9} {comp:>9} {initial:>9} {best:>9} {speedup:>9}")

    print("=" * 100)


if __name__ == "__main__":
    main()

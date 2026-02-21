#!/usr/bin/env python3
"""Benchmark activation checkpointing modes with and without cpu_offload.

Runs 6 configurations of llama3_8b training and collects metrics:
  1. mode=none
  2. mode=full
  3. mode=selective
  4. mode=none      + cpu_offload=true
  5. mode=full      + cpu_offload=true
  6. mode=selective + cpu_offload=true

Usage:
    NGPU=8 python benchmark_ac_modes.py [--steps 20] [--output-dir ./ac_benchmark]
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


CONFIGS = [
    # {"mode": "none", "cpu_offload": False},
    # {"mode": "full", "cpu_offload": False},
    # {"mode": "selective", "cpu_offload": False},
    {"mode": "none", "cpu_offload": True},
    # {"mode": "full", "cpu_offload": True},
    # {"mode": "selective", "cpu_offload": True},
]

BASE_CONFIG = "./torchtitan/models/llama3/train_configs/llama3_8b.toml"


def run_label(cfg: dict) -> str:
    offload = "+cpu_offload" if cfg["cpu_offload"] else ""
    return f"ac_{cfg['mode']}{offload}"


def run_training(cfg: dict, output_dir: str, steps: int, ngpu: int) -> dict:
    label = run_label(cfg)
    run_dir = os.path.join(output_dir, label)
    os.makedirs(run_dir, exist_ok=True)

    env = os.environ.copy()
    env["NGPU"] = str(ngpu)
    env["CONFIG_FILE"] = BASE_CONFIG
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    # Show all ranks' output so errors from any rank are visible
    env["LOG_RANK"] = ",".join(str(i) for i in range(ngpu))

    cpu_offload_flag = (
        "--activation_checkpoint.cpu_offload"
        if cfg["cpu_offload"]
        else "--activation_checkpoint.no-cpu_offload"
    )

    cmd = [
        "./run_train.sh",
        f"--job.dump_folder={run_dir}",
        f"--training.steps={steps}",
        f"--activation_checkpoint.mode={cfg['mode']}",
        cpu_offload_flag,
        "--profiling.enable_profiling",
        f"--profiling.save_traces_folder=profile_trace",
        "--profiling.enable_memory_snapshot",
        f"--profiling.save_memory_snapshot_folder=memory_snapshot",
        "--metrics.enable_wandb",
        "--metrics.no-enable_tensorboard",
    ]

    env["WANDB_PROJECT"] = os.environ.get("WANDB_PROJECT", "torchtitan-ac-benchmark")
    env["WANDB_RUN_NAME"] = label
    env["WANDB_RUN_GROUP"] = os.path.basename(os.path.abspath(output_dir))

    print(f"\n{'='*70}")
    print(f"Running: {label}")
    print(f"  mode={cfg['mode']}, cpu_offload={cfg['cpu_offload']}")
    print(f"  output: {run_dir}")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"{'='*70}\n")

    log_file = os.path.join(run_dir, "train.log")
    proc = subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    with open(log_file, "w") as log_f:
        log_f.write("=== STDOUT ===\n")
        log_f.write(proc.stdout)
        log_f.write("\n=== STDERR ===\n")
        log_f.write(proc.stderr)

    # Print combined output to console, show more on failure
    combined = proc.stdout + proc.stderr
    if proc.returncode != 0:
        print(combined[-5000:] if len(combined) > 5000 else combined)
    else:
        print(proc.stdout[-3000:] if len(proc.stdout) > 3000 else proc.stdout)

    result = {
        "label": label,
        "returncode": proc.returncode,
        "log_file": log_file,
        "metrics": {},
    }

    if proc.returncode != 0:
        print(f"*** {label} FAILED (exit code {proc.returncode}) ***")
        return result

    # Parse metrics from log output
    result["metrics"] = parse_metrics(proc.stdout)
    return result


def parse_metrics(log_output: str) -> dict:
    """Extract key metrics from training log output."""
    metrics = {
        "peak_memory_gib": None,
        "wps": [],          # words per second
        "mfu": [],          # model flops utilization
        "loss": [],
        "cuda_mem_pct": None,
    }

    for line in log_output.splitlines():
        # CUDA memory usage for model
        m = re.search(r"CUDA memory usage for model:\s*([\d.]+)GiB\(([\d.]+)%\)", line)
        if m:
            metrics["cuda_mem_pct"] = float(m.group(2))

        # Training step metrics: wps, mfu, loss
        m = re.search(r"wps:\s*([\d,]+)", line)
        if m:
            metrics["wps"].append(int(m.group(1).replace(",", "")))

        m = re.search(r"mfu:\s*([\d.]+)%", line)
        if m:
            metrics["mfu"].append(float(m.group(1)))

        m = re.search(r"loss:\s*([\d.]+)", line)
        if m:
            metrics["loss"].append(float(m.group(1)))

        # Peak memory
        m = re.search(r"peak_memory_active.*?:\s*([\d.]+)\s*GiB", line)
        if m:
            metrics["peak_memory_gib"] = float(m.group(1))

    return metrics


def print_summary(results: list[dict]):
    print(f"\n{'='*70}")
    print("BENCHMARK SUMMARY")
    print(f"{'='*70}")

    header = f"{'Config':<30} {'Status':<8} {'Avg WPS':>10} {'Avg MFU':>10} {'Peak Mem':>12}"
    print(header)
    print("-" * len(header))

    for r in results:
        status = "OK" if r["returncode"] == 0 else "FAIL"
        m = r["metrics"]

        avg_wps = ""
        if m.get("wps"):
            # Skip first step (warmup)
            vals = m["wps"][1:] if len(m["wps"]) > 1 else m["wps"]
            avg_wps = f"{sum(vals) / len(vals):,.0f}" if vals else ""

        avg_mfu = ""
        if m.get("mfu"):
            vals = m["mfu"][1:] if len(m["mfu"]) > 1 else m["mfu"]
            avg_mfu = f"{sum(vals) / len(vals):.2f}%" if vals else ""

        peak_mem = ""
        if m.get("peak_memory_gib") is not None:
            peak_mem = f"{m['peak_memory_gib']:.2f} GiB"
        elif m.get("cuda_mem_pct") is not None:
            peak_mem = f"{m['cuda_mem_pct']:.2f}% cap"

        print(f"{r['label']:<30} {status:<8} {avg_wps:>10} {avg_mfu:>10} {peak_mem:>12}")

    print(f"\nLogs saved to each run's directory.")


def main():
    parser = argparse.ArgumentParser(description="Benchmark AC modes for llama3 8B")
    parser.add_argument(
        "--steps", type=int, default=20, help="Number of training steps per run"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./ac_benchmark",
        help="Base output directory for all runs",
    )
    parser.add_argument(
        "--ngpu",
        type=int,
        default=int(os.environ.get("NGPU", "8")),
        help="Number of GPUs (default: $NGPU or 8)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for cfg in CONFIGS:
        result = run_training(cfg, args.output_dir, args.steps, args.ngpu)
        results.append(result)

    print_summary(results)


if __name__ == "__main__":
    main()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Launch local checks or prepare scheduler-neutral 256-GPU job specifications."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE = "scripts.graph_trainer_runs.configs"


def training_command(
    topology: str,
    implementation: str,
    output_dir: Path,
    *,
    num_nodes: int,
    node_rank: str = "0",
    master_addr: str = "127.0.0.1",
    master_port: str = "29500",
) -> list[str]:
    num_gpus = num_nodes * 4
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nnodes={num_nodes}",
        "--nproc-per-node=4",
        f"--node-rank={node_rank}",
        f"--master-addr={master_addr}",
        f"--master-port={master_port}",
        "--log-dir",
        str(output_dir / "launcher_logs" / f"node-{node_rank}"),
        "--tee=3",
        "--local-ranks-filter=0",
        "-m",
        "torchtitan.train",
        "--module",
        MODULE,
        "--config",
        f"{topology}_{num_gpus}gpu_{implementation}",
        "--output-dir",
        str(output_dir),
    ]


def require_idle_gpus(*, wait: bool) -> None:
    """Require four idle physical GPUs; never stop another workload."""
    stable_checks = 0
    while True:
        gpu_rows = list(
            csv.reader(
                io.StringIO(
                    subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=uuid,utilization.gpu,memory.used",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                    )
                )
            )
        )
        selection = os.environ.get("CUDA_VISIBLE_DEVICES", "0,1,2,3").split(",")
        if len(selection) != 4 or len(set(selection)) != 4:
            raise ValueError(
                "CUDA_VISIBLE_DEVICES must select exactly four distinct GPUs"
            )
        selected = []
        for value in selection:
            row = (
                gpu_rows[int(value)]
                if value.isdigit()
                else next(row for row in gpu_rows if row[0] == value)
            )
            selected.append(row)
        busy = any(int(row[1]) > 5 or int(row[2]) >= 2048 for row in selected)
        apps = list(
            csv.reader(
                io.StringIO(
                    subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-compute-apps=gpu_uuid",
                            "--format=csv,noheader",
                        ],
                        text=True,
                    )
                )
            )
        )
        busy = busy or bool(
            {row[0] for row in selected} & {row[0] for row in apps if row}
        )
        if not busy:
            stable_checks += 1
            if not wait or stable_checks == 3:
                return
        elif not wait:
            raise RuntimeError(
                "selected GPUs are busy; use --wait-for-idle or reserve idle GPUs"
            )
        else:
            stable_checks = 0
        print(
            "Waiting for four idle GPUs; no existing processes will be stopped.",
            flush=True,
        )
        time.sleep(30)


def job_spec(
    topology: str, implementation: str, output_dir: Path, *, mode: str, steps: int
) -> dict:
    source_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
    ).strip()
    env = {
        "GT_RUN_MODE": mode,
        "GT_RUN_STEPS": str(steps),
        "GT_RUN_SOURCE_COMMIT": source_commit,
    }
    for name in ("GT_RUN_C4_SOURCE", "GT_RUN_TOKENIZER_PATH"):
        if name in os.environ:
            env[name] = os.environ[name]
    return {
        "name": f"gt-{topology}-{implementation}-256",
        "topology": topology,
        "implementation": implementation,
        "num_nodes": 64,
        "gpus_per_node": 4,
        "priority": "normal",
        "source_commit": source_commit,
        "cwd": str(REPO_ROOT),
        "argv": training_command(
            topology,
            implementation,
            output_dir,
            num_nodes=64,
            node_rank="${NODE_RANK}",
            master_addr="${MASTER_ADDR}",
            master_port="${MASTER_PORT}",
        ),
        "env": env,
        "output_dir": str(output_dir),
    }


def slurm_script(spec: dict) -> str:
    worker = [
        sys.executable,
        "-m",
        "scripts.graph_trainer_runs.launch",
        "worker",
        "--topology",
        spec["topology"],
        "--implementation",
        spec["implementation"],
        "--output-dir",
        spec["output_dir"],
    ]
    env_lines = [
        f"export {key}={shlex.quote(value)}" for key, value in spec["env"].items()
    ]
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            f"cd {shlex.quote(spec['cwd'])}",
            *env_lines,
            'export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | sed -n "1p")',
            "export MASTER_PORT=${MASTER_PORT:-29500}",
            "srun --nodes=64 --ntasks=64 --ntasks-per-node=1 --gpus-per-task=4 --gpu-bind=none "
            + shlex.join(worker),
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    local = subparsers.add_parser("local")
    local.add_argument("topology", choices=("spmd", "pp"))
    local.add_argument("--mode", choices=("verify", "benchmark"), default="verify")
    local.add_argument(
        "--implementation", choices=("eager", "graph_trainer", "both"), default="both"
    )
    local.add_argument("--steps", type=int, default=12)
    local.add_argument("--output-dir", type=Path, required=True)
    local.add_argument("--wait-for-idle", action="store_true")
    local.add_argument("--dry-run", action="store_true")
    submit = subparsers.add_parser("submit")
    submit.add_argument("topology", choices=("spmd", "pp", "all"))
    submit.add_argument(
        "--implementation", choices=("eager", "graph_trainer", "both"), default="both"
    )
    submit.add_argument("--mode", choices=("verify", "benchmark"), default="benchmark")
    submit.add_argument("--steps", type=int, default=60)
    submit.add_argument("--output-dir", type=Path, required=True)
    submit.add_argument("--submit", action="store_true")
    submit.add_argument("--submit-command", nargs="+")
    submit.add_argument("--account")
    submit.add_argument("--partition")
    submit.add_argument("--constraint")
    worker = subparsers.add_parser("worker")
    worker.add_argument("--topology", choices=("spmd", "pp"), required=True)
    worker.add_argument(
        "--implementation", choices=("eager", "graph_trainer"), required=True
    )
    worker.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    if args.action == "worker":
        expected_commit = os.environ["GT_RUN_SOURCE_COMMIT"]
        actual_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
        if actual_commit != expected_commit:
            raise RuntimeError(
                "worker checkout differs from the submitted source commit"
            )
        node_rank = os.environ.get("NODE_RANK", os.environ.get("SLURM_NODEID"))
        if node_rank is None:
            raise ValueError("worker requires NODE_RANK or SLURM_NODEID")
        subprocess.run(
            training_command(
                args.topology,
                args.implementation,
                output_dir,
                num_nodes=64,
                node_rank=node_rank,
                master_addr=os.environ["MASTER_ADDR"],
                master_port=os.environ["MASTER_PORT"],
            ),
            check=True,
        )
        return
    if args.steps < 10:
        parser.error("--steps must be at least ten")
    implementations = (
        ("eager", "graph_trainer")
        if args.implementation == "both"
        else (args.implementation,)
    )
    if args.action == "local":
        if args.mode == "verify" and args.implementation != "both":
            parser.error("verification requires both implementations")
        if not args.dry_run:
            if output_dir.exists():
                raise FileExistsError("use a new output directory for every run")
            require_idle_gpus(wait=args.wait_for_idle)
        env = os.environ.copy()
        env.update(
            {
                "GT_RUN_MODE": args.mode,
                "GT_RUN_STEPS": str(args.steps),
                "TORCHTITAN_PAIRED_NUMERICS_SPMD_ACCUMULATION_STEPS": "4",
                "TORCHTITAN_PAIRED_NUMERICS_CAPTURE_STEPS": f"1,2,{args.steps}",
                "TORCHTITAN_PAIRED_NUMERICS_CUDA_GRAPHS": "1",
            }
        )
        if args.mode == "verify":
            commands = [
                [
                    sys.executable,
                    "-m",
                    "scripts.graph_trainer_runs.run_paired_numerics",
                    args.topology,
                    "--cuda-graphs",
                    "--capture-steps",
                    f"1,2,{args.steps}",
                    "--output-dir",
                    str(output_dir),
                ]
            ]
        else:
            commands = [
                training_command(
                    args.topology,
                    implementation,
                    output_dir / implementation,
                    num_nodes=1,
                )
                for implementation in implementations
            ]
        for command in commands:
            print(shlex.join(command), flush=True)
            if not args.dry_run:
                require_idle_gpus(wait=False)
                subprocess.run(command, env=env, cwd=REPO_ROOT, check=True)
        return
    if args.submit:
        if not args.submit_command and (not args.account or not args.partition):
            parser.error("Slurm submission requires --account and --partition")
        if subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO_ROOT, text=True
        ).strip():
            parser.error("commit or isolate working-tree changes before submission")
        if output_dir.exists():
            raise FileExistsError("use a new output directory for every submission")
        output_dir.mkdir(parents=True)
    topologies = ("spmd", "pp") if args.topology == "all" else (args.topology,)
    for topology in topologies:
        for implementation in implementations:
            spec = job_spec(
                topology,
                implementation,
                output_dir / f"{topology}-{implementation}",
                mode=args.mode,
                steps=args.steps,
            )
            print(json.dumps(spec, indent=2), flush=True)
            if not args.submit:
                continue
            if args.submit_command:
                subprocess.run(
                    args.submit_command, input=json.dumps(spec), text=True, check=True
                )
            else:
                command = [
                    "sbatch",
                    "--parsable",
                    f"--job-name={spec['name']}",
                    "--nodes=64",
                    "--ntasks-per-node=1",
                    "--gpus-per-node=4",
                    "--exclusive",
                    "--nice=0",
                    f"--account={args.account}",
                    f"--partition={args.partition}",
                    f"--output={output_dir / (spec['name'] + '-%j.log')}",
                ]
                if args.constraint:
                    command.append(f"--constraint={args.constraint}")
                subprocess.run(command, input=slurm_script(spec), text=True, check=True)


if __name__ == "__main__":
    main()

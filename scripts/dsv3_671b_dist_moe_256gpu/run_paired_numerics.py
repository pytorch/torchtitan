#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Run and enforce the four-GPU DistMoE MTP1 paired numerics gate."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.dsv3_671b_dist_moe_256gpu.paired_numerics_runtime import (
    parse_capture_steps,
)


_MODULE = "scripts.dsv3_671b_dist_moe_256gpu.paired_numerics_configs"
_CONFIGS = {
    "pp1": (
        "eager_mtp1_pp1_paired_numerics_4gpu",
        "graph_trainer_mtp1_pp1_paired_numerics_4gpu",
    ),
    "pp2": (
        "eager_mtp1_pp2_paired_numerics_4gpu",
        "graph_trainer_mtp1_pp2_paired_numerics_4gpu",
    ),
}
_SCENARIOS = ("eager", "graph_trainer")


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def compare_config_contracts(
    output_root: Path,
    *,
    world_size: int = 4,
) -> dict[str, Any]:
    """Compare selected workload fields across all ranks and both engines."""
    contracts: dict[str, list[dict[str, Any]]] = {}
    errors: list[str] = []
    for scenario in _SCENARIOS:
        scenario_contracts = []
        for rank in range(world_size):
            path = output_root / "config_contracts" / scenario / f"rank_{rank}.json"
            if not path.is_file():
                errors.append(f"missing config contract: {path}")
                continue
            scenario_contracts.append(_load_json(path))
        contracts[scenario] = scenario_contracts
        if scenario_contracts:
            rank_zero_common = scenario_contracts[0]["common"]
            for rank, contract in enumerate(scenario_contracts[1:], start=1):
                if contract["common"] != rank_zero_common:
                    errors.append(
                        f"{scenario} rank {rank} has a different common config"
                    )

    if all(contracts.values()):
        eager_common = contracts["eager"][0]["common"]
        graph_common = contracts["graph_trainer"][0]["common"]
        if eager_common != graph_common:
            errors.append("eager and GraphTrainer common config contracts differ")

    return {
        "equal": not errors,
        "errors": errors,
        "eager": contracts["eager"][0] if contracts["eager"] else None,
        "graph_trainer": (
            contracts["graph_trainer"][0] if contracts["graph_trainer"] else None
        ),
    }


def _parameter_records(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = manifest.get("parameters")
    if not isinstance(records, list):
        raise TypeError("gradient manifest 'parameters' must be a list")
    by_fqn = {record["fqn"]: record for record in records}
    if len(by_fqn) != len(records):
        raise ValueError("gradient manifest contains duplicate parameter FQNs")
    return by_fqn


def compare_gradient_manifests(
    output_root: Path,
    *,
    capture_steps: tuple[int, ...],
    world_size: int = 4,
) -> dict[str, Any]:
    """Compare every local pre-optimizer gradient digest by step and rank."""
    comparisons = []
    total_parameters = 0
    total_matches = 0
    total_mismatches = 0
    for step in capture_steps:
        for rank in range(world_size):
            paths = {
                scenario: (
                    output_root
                    / "gradient_sha256"
                    / scenario
                    / f"step_{step}"
                    / f"rank_{rank}.json"
                )
                for scenario in _SCENARIOS
            }
            missing = [str(path) for path in paths.values() if not path.is_file()]
            if missing:
                comparisons.append(
                    {
                        "step": step,
                        "rank": rank,
                        "equal": False,
                        "missing": missing,
                        "mismatches": [],
                    }
                )
                total_mismatches += 1
                continue
            manifests = {scenario: _load_json(path) for scenario, path in paths.items()}
            eager = _parameter_records(manifests["eager"])
            graph = _parameter_records(manifests["graph_trainer"])
            fqns = sorted(set(eager) | set(graph))
            mismatches = []
            for fqn in fqns:
                eager_record = eager.get(fqn)
                graph_record = graph.get(fqn)
                if eager_record is None or graph_record is None:
                    mismatches.append(
                        {
                            "fqn": fqn,
                            "kind": "missing_parameter",
                            "eager_present": eager_record is not None,
                            "graph_trainer_present": graph_record is not None,
                        }
                    )
                    continue
                differing_fields = [
                    field
                    for field in (
                        "index",
                        "dtype",
                        "global_shape",
                        "local_shape",
                        "placements",
                        "gradient_sha256",
                    )
                    if eager_record[field] != graph_record[field]
                ]
                if differing_fields:
                    mismatches.append(
                        {
                            "fqn": fqn,
                            "kind": "different_gradient",
                            "differing_fields": differing_fields,
                            "eager": eager_record,
                            "graph_trainer": graph_record,
                        }
                    )
            num_parameters = len(fqns)
            num_mismatches = len(mismatches)
            num_matches = num_parameters - num_mismatches
            total_parameters += num_parameters
            total_matches += num_matches
            total_mismatches += num_mismatches
            comparisons.append(
                {
                    "step": step,
                    "rank": rank,
                    "equal": not mismatches,
                    "num_parameters": num_parameters,
                    "num_matches": num_matches,
                    "num_mismatches": num_mismatches,
                    "mismatches": mismatches,
                }
            )
    return {
        "equal": total_mismatches == 0,
        "total_parameters": total_parameters,
        "total_matches": total_matches,
        "total_mismatches": total_mismatches,
        "comparisons": comparisons,
    }


def compare_scalar_metrics(job_dump_folder: Path) -> dict[str, Any]:
    """Read full-precision TensorBoard scalars through loss_compare helpers."""
    from scripts.loss_compare import extract_metrics_from_tensorboard

    metric_names = ("loss", "grad_norm")
    metrics = {
        "eager": extract_metrics_from_tensorboard(
            str(job_dump_folder), "tb_baseline", metric_names
        ),
        "graph_trainer": extract_metrics_from_tensorboard(
            str(job_dump_folder), "tb_test", metric_names
        ),
    }
    comparisons: dict[str, Any] = {}
    equal = True
    for metric_name in metric_names:
        eager = metrics["eager"][metric_name]
        graph = metrics["graph_trainer"][metric_name]
        metric_equal = eager == graph
        equal = equal and metric_equal
        comparisons[metric_name] = {
            "equal": metric_equal,
            "eager": {str(step): value for step, value in sorted(eager.items())},
            "graph_trainer": {
                str(step): value for step, value in sorted(graph.items())
            },
        }
    return {"equal": equal, "metrics": comparisons}


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare eager and GraphTrainer MTP1 loss, grad norm, and every "
            "rank-local parameter gradient on four GPUs."
        )
    )
    parser.add_argument("topology", choices=sorted(_CONFIGS))
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New directory for logs, TensorBoard events, and SHA-256 manifests.",
    )
    parser.add_argument(
        "--capture-steps",
        default="1,2",
        help="Comma-separated optimizer steps to hash (default: 1,2).",
    )
    parser.add_argument(
        "--pp1-accumulation-steps",
        type=int,
        default=4,
        help="PP1 local gradient accumulation depth (default: 4).",
    )
    parser.add_argument(
        "--cuda-graphs",
        action="store_true",
        help="Also exercise the existing outer full-step CUDA graph path.",
    )
    parser.add_argument(
        "--print-command",
        action="store_true",
        help="Print the loss_compare.py command and exit without running it.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    capture_steps = parse_capture_steps(args.capture_steps)
    if args.pp1_accumulation_steps <= 0:
        raise ValueError("--pp1-accumulation-steps must be positive")

    output_root = args.output_dir.resolve()
    if output_root.exists() and not args.print_command:
        raise FileExistsError(
            f"output directory already exists: {output_root}; use a new path"
        )
    eager_config, graph_config = _CONFIGS[args.topology]
    job_dump_folder = output_root / "jobs"
    command = [
        sys.executable,
        str(_REPO_ROOT / "scripts" / "loss_compare.py"),
        ".",
        ".",
        f"--baseline-module={_MODULE}",
        f"--test-module={_MODULE}",
        f"--baseline-config={eager_config}",
        f"--test-config={graph_config}",
        f"--steps={max(capture_steps)}",
        "--no-seed-checkpoint",
        "--metrics=loss,grad_norm",
        "--baseline-ngpus=4",
        "--test-ngpus=4",
        f"--output-folder={output_root}",
        f"--job-dump-folder={job_dump_folder}",
    ]
    print(shlex.join(command), flush=True)
    if args.print_command:
        return

    env = os.environ.copy()
    env.update(
        {
            "TORCHTITAN_PAIRED_NUMERICS_OUTPUT_ROOT": str(output_root),
            "TORCHTITAN_PAIRED_NUMERICS_CAPTURE_STEPS": ",".join(
                str(step) for step in capture_steps
            ),
            "TORCHTITAN_PAIRED_NUMERICS_CUDA_GRAPHS": (
                "1" if args.cuda_graphs else "0"
            ),
            "TORCHTITAN_PAIRED_NUMERICS_PP1_ACCUMULATION_STEPS": str(
                args.pp1_accumulation_steps
            ),
        }
    )
    completed = subprocess.run(command, cwd=_REPO_ROOT, env=env, check=False)
    if completed.returncode != 0:
        raise subprocess.CalledProcessError(completed.returncode, command)

    result = {
        "format_version": 1,
        "topology": args.topology,
        "capture_steps": list(capture_steps),
        "cuda_graphs_enabled": args.cuda_graphs,
        "pp1_accumulation_steps": args.pp1_accumulation_steps,
        "config_contract": compare_config_contracts(output_root),
        "scalar_metrics": compare_scalar_metrics(job_dump_folder),
        "gradient_sha256": compare_gradient_manifests(
            output_root,
            capture_steps=capture_steps,
        ),
    }
    result["equal"] = all(
        result[key]["equal"]
        for key in ("config_contract", "scalar_metrics", "gradient_sha256")
    )
    result_path = output_root / "paired_numerics_result.json"
    _write_json(result_path, result)
    print(
        "Paired numerics: "
        f"config={result['config_contract']['equal']} "
        f"scalars={result['scalar_metrics']['equal']} "
        f"gradients={result['gradient_sha256']['total_matches']}/"
        f"{result['gradient_sha256']['total_parameters']} exact",
        flush=True,
    )
    print(f"Result: {result_path}", flush=True)
    if not result["equal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

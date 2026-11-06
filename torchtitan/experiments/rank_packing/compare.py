# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fail-closed comparison of complete same-topology training replays."""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


def _compare(left: Any, right: Any, path: str, rows: list[dict]) -> None:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        if left.shape != right.shape or left.dtype != right.dtype:
            raise ValueError(f"Metadata mismatch at {path}")
        finite = bool(torch.isfinite(left).all() and torch.isfinite(right).all())
        exact = finite and torch.equal(left, right)
        row = {"path": path, "numel": left.numel(), "exact": exact, "finite": finite}
        if left.is_floating_point() or left.is_complex():
            if finite:
                dtype = torch.complex128 if left.is_complex() else torch.float64
                delta = (left.to(dtype) - right.to(dtype)).abs()
                scale = right.to(dtype).abs()
                denominator = max(scale.max().item() if scale.numel() else 0, 1e-12)
                row.update(
                    max_abs=delta.max().item() if delta.numel() else 0,
                    max_abs_over_reference_max=(delta.max().item() / denominator)
                    if delta.numel()
                    else 0,
                    allclose_1e_5_1e_6=torch.allclose(
                        left, right, rtol=1e-5, atol=1e-6
                    ),
                )
            else:
                row.update(max_abs=None, allclose_1e_5_1e_6=False)
        rows.append(row)
    elif isinstance(left, dict) and isinstance(right, dict):
        if left.keys() != right.keys():
            raise ValueError(f"Key mismatch at {path}")
        for key in left:
            _compare(left[key], right[key], f"{path}/{key}", rows)
    elif isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            raise ValueError(f"Length mismatch at {path}")
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            _compare(a, b, f"{path}/{index}", rows)
    else:
        finite = all(
            not isinstance(value, (float, complex))
            or (math.isfinite(value.real) and math.isfinite(value.imag))
            for value in (left, right)
        )
        rows.append({"path": path, "exact": finite and left == right, "finite": finite})


def _load_manifest(directory: Path) -> tuple[dict, int]:
    outcome = json.loads((directory / "outcome.json").read_text())
    if (
        outcome["exit_code"] != 0
        or outcome["timed_out"]
        or not outcome.get("process_group_gone", False)
    ):
        raise ValueError(f"Incomplete or failed run: {directory}")
    launch = json.loads((directory / "launch.json").read_text())
    options = launch["options"]
    for rank in range(8):
        metadata = json.loads((directory / f"metadata_rank_{rank}.json").read_text())
        visible = launch["env"]["CUDA_VISIBLE_DEVICES"].split(",")
        expected_device = 0 if len(visible) == 1 else rank
        if (
            metadata["cuda_visible_devices"] != launch["env"]["CUDA_VISIBLE_DEVICES"]
            or metadata["device"] != f"cuda:{expected_device}"
        ):
            raise ValueError(
                f"Physical rank mapping mismatch: {directory}, rank {rank}"
            )
    if options["world_size"] != 8:
        raise ValueError("Expected eight logical ranks")
    num_steps = options["steps"]
    expected = {f"initial_rank_{rank}.pt" for rank in range(8)} | {
        f"step_{step}_rank_{rank}.pt"
        for step in range(1, num_steps + 1)
        for rank in range(8)
    }
    observed = {path.name for path in directory.glob("*.pt")}
    if observed != expected:
        raise ValueError(f"Incomplete or extra snapshot manifest at {directory}")
    return launch, num_steps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    left_launch, num_steps = _load_manifest(args.left)
    right_launch, right_steps = _load_manifest(args.right)
    if (
        num_steps != right_steps
        or left_launch["source_commit"] != right_launch["source_commit"]
    ):
        raise ValueError("Incompatible source or step count")
    for key in ("world_size", "steps", "seq_len", "dist_moe"):
        if left_launch["options"][key] != right_launch["options"][key]:
            raise ValueError(f"Recipe mismatch: {key}")
    left_env, right_env = left_launch["env"], right_launch["env"]
    for key in set(left_env) | set(right_env):
        if key.startswith(("NCCL_", "TORCH_SYMM")) and left_env.get(
            key
        ) != right_env.get(key):
            raise ValueError(f"Communication configuration mismatch: {key}")
    masks = [
        launch["env"]["CUDA_VISIBLE_DEVICES"].split(",")
        for launch in (left_launch, right_launch)
    ]
    if sorted(map(len, masks)) != [1, 8]:
        raise ValueError(
            "This comparison requires one packed GPU and eight physical GPUs."
        )
    if any(len(set(mask)) != len(mask) for mask in masks):
        raise ValueError("Repeated physical GPU in visible-device mask.")
    results = []
    metadata_keys = [
        "rank",
        "local_rank",
        "world_size",
        "gpu",
        "torch",
        "nccl",
        "parallelism",
        "model",
        "model_parameters",
        "seq_len",
        "tokens_per_dp_rank",
        "compute_regions",
        "modules",
        "parameter_layout",
        "requested_steps",
        "dist_moe",
        "attention_impl",
    ]
    for rank in range(8):
        metadata = [
            json.loads((directory / f"metadata_rank_{rank}.json").read_text())
            for directory in (args.left, args.right)
        ]
        if metadata[0]["rank"] != rank or metadata[1]["rank"] != rank:
            raise ValueError("Rank metadata mismatch")
        for key in metadata_keys:
            if metadata[0][key] != metadata[1][key]:
                raise ValueError(f"Rank {rank} incompatible metadata: {key}")
        files = [
            f"initial_rank_{rank}.pt",
            *[f"step_{step}_rank_{rank}.pt" for step in range(1, num_steps + 1)],
        ]
        for name in files:
            snapshots = [
                torch.load(directory / name, map_location="cpu", weights_only=False)
                for directory in (args.left, args.right)
            ]
            if name.startswith("step_"):
                for snapshot in snapshots:
                    gradients = snapshot["gradients_pre_clip"]
                    required = {
                        key
                        for key, entry in metadata[0]["parameter_layout"].items()
                        if entry["requires_grad"]
                    }
                    if gradients.keys() != snapshot["parameters"].keys() or any(
                        gradients[key] is None for key in required
                    ):
                        raise ValueError(f"Missing gradients: {name}")
            rows = []
            _compare(*snapshots, name, rows)
            failures = [row for row in rows if not row["exact"]]
            tensors = [row for row in rows if "numel" in row]
            result = {
                "file": name,
                "compared_values": len(rows),
                "tensors": len(tensors),
                "tensor_elements": sum(row["numel"] for row in tensors),
                "exact": not failures,
                "nonexact_values": len(failures),
                "max_abs": max((row.get("max_abs") or 0 for row in rows), default=0),
                "largest_differences": sorted(
                    failures, key=lambda row: row.get("max_abs") or 0, reverse=True
                )[:15],
            }
            results.append(result)
    report = {
        "left": str(args.left),
        "right": str(args.right),
        "all_exact": all(result["exact"] for result in results),
        "files": results,
    }
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "files"}, indent=2
        )
    )
    for result in results:
        print(
            result["file"],
            "exact",
            result["exact"],
            "nonexact",
            result["nonexact_values"],
            "max_abs",
            result["max_abs"],
        )
    raise SystemExit(0 if report["all_exact"] else 1)


if __name__ == "__main__":
    main()

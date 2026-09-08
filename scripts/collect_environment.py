#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Collect source and live runtime provenance for a TorchTitan run.

Run this script with the same Python interpreter and execution environment used
to launch training. Otherwise, the recorded packages and runtime may not match
the training process.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import socket
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEDULER_ENV_VARS = (
    "SLURM_CLUSTER_NAME",
    "SLURM_CPUS_PER_TASK",
    "SLURM_JOB_ACCOUNT",
    "SLURM_JOB_ID",
    "SLURM_JOB_NAME",
    "SLURM_JOB_NODELIST",
    "SLURM_JOB_PARTITION",
    "SLURM_JOB_QOS",
    "SLURM_NNODES",
    "SLURM_NTASKS",
)


def _run_command(command: list[str], *, cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.rstrip()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git_source(repo_root: Path) -> dict[str, Any]:
    revision = _run_command(["git", "rev-parse", "HEAD"], cwd=repo_root)
    if revision is None:
        return {
            "repository_root": str(repo_root.resolve()),
            "revision": None,
            "branch": None,
            "is_dirty": None,
            "tracked_diff_sha256": None,
            "status": None,
        }

    branch = _run_command(["git", "branch", "--show-current"], cwd=repo_root)
    status = _run_command(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repo_root,
    )
    diff = _run_command(["git", "diff", "--binary", "HEAD"], cwd=repo_root)
    return {
        "repository_root": str(repo_root.resolve()),
        "revision": revision,
        "branch": branch or None,
        "is_dirty": bool(status),
        "tracked_diff_sha256": _sha256(diff.encode()) if diff else None,
        "status": status.splitlines() if status else [],
    }


def _distribution_direct_url(distribution: importlib.metadata.Distribution) -> dict:
    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        return {}
    try:
        value = json.loads(direct_url_text)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, dict) else {}


def _wheel_digest(direct_url: dict) -> str | None:
    archive_info = direct_url.get("archive_info", {})
    if not isinstance(archive_info, dict):
        return None
    hashes = archive_info.get("hashes", {})
    if isinstance(hashes, dict) and isinstance(hashes.get("sha256"), str):
        return hashes["sha256"]
    archive_hash = archive_info.get("hash")
    if isinstance(archive_hash, str) and archive_hash.startswith("sha256="):
        return archive_hash.removeprefix("sha256=")
    return None


def _canonicalize_distribution_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _distribution_info(
    distribution: importlib.metadata.Distribution,
) -> dict[str, Any]:
    direct_url = _distribution_direct_url(distribution)
    vcs_info = direct_url.get("vcs_info", {})
    revision = vcs_info.get("commit_id") if isinstance(vcs_info, dict) else None
    return {
        "installed": True,
        "version": distribution.version,
        "revision": revision,
        "location": str(distribution.locate_file("")),
        "wheel_sha256": _wheel_digest(direct_url),
    }


def _declared_distribution_names(repo_root: Path) -> set[str]:
    pyproject_path = repo_root / "pyproject.toml"
    if not pyproject_path.is_file():
        return set()
    with pyproject_path.open("rb") as pyproject_file:
        project = tomllib.load(pyproject_file).get("project", {})

    requirement_strings = list(project.get("dependencies", []))
    for optional_requirements in project.get("optional-dependencies", {}).values():
        requirement_strings.extend(optional_requirements)

    names = {_canonicalize_distribution_name(project.get("name", "torchtitan"))}
    for requirement in requirement_strings:
        match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)", requirement)
        if match:
            names.add(_canonicalize_distribution_name(match.group(1)))
    return names


def _installed_packages(repo_root: Path) -> dict[str, dict[str, Any]]:
    packages: dict[str, dict[str, Any]] = {}
    for distribution in importlib.metadata.distributions():
        distribution_name = distribution.metadata.get("Name")
        if not distribution_name:
            continue
        name = _canonicalize_distribution_name(distribution_name)
        packages[name] = _distribution_info(distribution)

    for name in _declared_distribution_names(repo_root) - packages.keys():
        packages[name] = {
            "installed": False,
            "version": None,
            "revision": None,
            "location": None,
            "wheel_sha256": None,
        }
    return dict(sorted(packages.items()))


def _torch_runtime() -> dict[str, Any]:
    try:
        import torch
    except Exception as error:
        return {"probe_error": f"{type(error).__name__}: {error}"}

    cuda: dict[str, Any] = {
        "is_available": torch.cuda.is_available(),
        "runtime_version": torch.version.cuda,
        "device_count": 0,
        "devices": [],
    }
    nccl: dict[str, Any] = {"version": None, "is_available": False}
    try:
        nccl["is_available"] = torch.distributed.is_nccl_available()
        if nccl["is_available"]:
            nccl["version"] = torch.cuda.nccl.version()
    except Exception as error:
        nccl["probe_error"] = f"{type(error).__name__}: {error}"

    if torch.cuda.is_available():
        try:
            cuda["device_count"] = torch.cuda.device_count()
            cuda["devices"] = [
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "compute_capability": list(torch.cuda.get_device_capability(index)),
                    "total_memory_bytes": torch.cuda.get_device_properties(
                        index
                    ).total_memory,
                }
                for index in range(torch.cuda.device_count())
            ]
        except Exception as error:
            cuda["probe_error"] = f"{type(error).__name__}: {error}"

    return {
        "version": torch.__version__,
        "revision": getattr(torch.version, "git_version", None),
        "path": getattr(torch, "__file__", None),
        "cuda": cuda,
        "nccl": nccl,
    }


def _selected_environment() -> dict[str, str]:
    prefixes = ("NCCL_", "TORCH_NCCL_")
    exact_names = {
        "CUDA_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "PYTORCH_ALLOC_CONF",
    }
    return {
        name: value
        for name, value in sorted(os.environ.items())
        if name in exact_names or name.startswith(prefixes)
    }


def _accelerator_topology() -> dict[str, str | None]:
    return {
        "nvidia_smi": _run_command(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,memory.total,driver_version",
                "--format=csv,noheader",
            ]
        ),
        "nvidia_smi_topology": _run_command(["nvidia-smi", "topo", "-m"]),
    }


def collect_environment(repo_root: Path) -> dict[str, Any]:
    packages = _installed_packages(repo_root)
    source = _git_source(repo_root)
    if source["revision"] is None:
        source["revision"] = packages.get("torchtitan", {}).get("revision")

    runtime = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "wheel_digests": {
            name: package["wheel_sha256"] for name, package in packages.items()
        },
        "packages": packages,
        "torch": _torch_runtime(),
        "accelerator_topology": _accelerator_topology(),
        "environment": _selected_environment(),
        "scheduler": {
            name: os.environ[name]
            for name in SCHEDULER_ENV_VARS
            if os.environ.get(name)
        },
    }
    return {"source": source, "runtime": runtime}


def write_environment(output_path: Path, environment: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    with temporary_path.open("w") as output_file:
        json.dump(environment, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    temporary_path.replace(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record source and live runtime provenance as JSON."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="TorchTitan Git checkout used for source provenance.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_environment(args.output, collect_environment(args.repo_root))


if __name__ == "__main__":
    main()

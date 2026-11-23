# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Bind an indexed GraphTrainer schedule to verified kernel problem specializations."""

from __future__ import annotations

import ast
import copy
import json
import logging
import os
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path

import torch
from torch import fx

from torchtitan.experiments.graph_trainer.auto_fuser.schedule import (
    IndexedGraph,
    SchedulePlan,
)
from torchtitan.experiments.graph_trainer.auto_fuser.substitute_kernels import (
    normalize_names,
    tensor_contract,
    VerifiedKernel,
)
from torchtitan.experiments.graph_trainer.registry import register_graph_pass

logger = logging.getLogger(__name__)


@dataclass(frozen=True, kw_only=True)
class KernelSource:
    path: Path
    sha256: str
    entrypoint: str = "kernel_function"


def _current_target() -> dict:
    return {
        "torch_git_version": torch.version.git_version,
        "cuda": torch.version.cuda,
        "device_capability": list(torch.cuda.get_device_capability()),
        "nvidia-cutlass-dsl": metadata.version("nvidia-cutlass-dsl"),
        "TORCH_DISABLE_NATIVE_JIT": os.environ.get("TORCH_DISABLE_NATIVE_JIT"),
    }


def _reference_body(problem: Path) -> str:
    source = problem.read_text()
    ref = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "reference"
    )
    # FX normalizes both local names and device literals. Keep its argument
    # order, constants, output order and in-place operations in the lookup.
    return "\n".join(source.splitlines()[ref.lineno : ref.end_lineno]).strip()


def substitute_with_kernels(
    gm: fx.GraphModule,
    calls: list[fx.Node],
    *,
    problem_dir: Path,
    manifest: dict,
    kernels: dict[str, KernelSource],
) -> fx.GraphModule:
    """Match each scheduled call to a problem by normalized FX code and tensor contract.

    Problem directory names retain the original capture's AOTAutograd hashes. Those
    hashes include compiler and host settings. The problem's saved reference and
    input/output descriptions let us check the kernel ABI directly on another host.
    """
    expected_regions = Counter()
    for info in manifest["problems"].values():
        expected_regions[info["region"]] += info["calls"]
    region_counts = Counter(
        call.meta["fusion_region"].split("_", 1)[0] for call in calls
    )
    if region_counts != expected_regions:
        raise ValueError(
            f"{problem_dir.name}: region calls differ "
            f"(missing={expected_regions - region_counts}, extra={region_counts - expected_regions})"
        )

    problems = {}
    solved_regions = set()
    for key in kernels:
        region_id = manifest["problems"][key]["region"]
        solved_regions.add(region_id)
        directory = problem_dir / "kernels" / key
        body = _reference_body(directory / "problem.py")
        spec = json.loads((directory / "spec.json").read_text())
        env = spec["env"]
        if (
            env["deterministic"] != torch.are_deterministic_algorithms_enabled()
            or env["fill_uninitialized_memory"]
            != torch.utils.deterministic.fill_uninitialized_memory
            or env["gpu"] != torch.cuda.get_device_name()
        ):
            raise ValueError(
                f"{problem_dir.name} requires the environment in spec.json"
            )
        signature = (
            region_id,
            body,
            json.dumps([spec["inputs"], spec["outputs"]], sort_keys=True),
        )
        if signature in problems:
            raise ValueError(
                f"Ambiguous kernel problems: {problems[signature]} and {key}"
            )
        problems[signature] = key

    hits = Counter()
    replacements = {}
    for call in calls:
        region_id = call.meta["fusion_region"].split("_", 1)[0]
        if region_id not in solved_regions:
            continue
        region = gm.get_submodule(call.target)
        code = normalize_names(region).code.strip().split("\n", 1)[1].strip()
        inputs = [arg.meta["val"] for arg in call.args]
        outputs = call.meta["val"]
        contract = tensor_contract(inputs, outputs)
        key = problems.get((region_id, code, json.dumps(contract, sort_keys=True)))
        if key is None:
            continue
        if key not in replacements:
            source = kernels[key]
            replacements[key] = VerifiedKernel(
                source.path,
                problem_id=key,
                sha256=source.sha256,
                entrypoint=source.entrypoint,
                impure=region._is_impure,
            )
        gm.set_submodule(call.target, replacements[key])
        hits[key] += 1

    expected = Counter({key: manifest["problems"][key]["calls"] for key in kernels})
    if hits != expected:
        raise ValueError(
            f"{problem_dir.name}: verified kernel calls differ (missing={expected - hits}, extra={hits - expected})"
        )
    logger.info(
        "%s: %d scheduled calls, %d kernel hits across %d specializations, %d reference calls",
        problem_dir.name,
        len(calls),
        hits.total(),
        len(hits),
        len(calls) - hits.total(),
    )
    return gm


def register_schedule(
    configure_schedule: Callable[[SchedulePlan], None],
    *,
    problem_dir: Path,
    kernels: Path | dict[str, KernelSource],
) -> str:
    """Register a recipe's plan and a kernel pack directory or verified source mapping."""
    problem_dir = problem_dir.resolve()
    manifest = json.loads((problem_dir / "manifest.json").read_text())
    if isinstance(kernels, Path):
        pack = json.loads((kernels / "manifest.json").read_text())
        if pack["target"] != manifest["target"]:
            raise ValueError(
                f"{kernels.name} and {problem_dir.name} target different environments"
            )
        sources = {}
        for key, kernel in pack["kernels"].items():
            if kernel["calls"] != manifest["problems"][key]["calls"]:
                raise ValueError(f"Pack and problem disagree on call count for {key}")
            sources[key] = KernelSource(
                path=kernels / "kernels" / key / "kernel.py", sha256=kernel["sha256"]
            )
    else:
        sources = kernels

    def apply_schedule(gm: fx.GraphModule, example_inputs=None) -> fx.GraphModule:
        target = _current_target()
        if target != manifest["target"]:
            raise ValueError(
                f"{problem_dir.name} requires {manifest['target']}, found {target}"
            )
        # Stage the rewrite on a graph copy. A bad region or a missing kernel
        # must leave the input graph intact for another pass or a retry.
        working = fx.GraphModule(gm, copy.deepcopy(gm.graph))
        plan = SchedulePlan(IndexedGraph(working))
        configure_schedule(plan)
        calls = plan.apply()
        substitute_with_kernels(
            working, calls, problem_dir=problem_dir, manifest=manifest, kernels=sources
        )
        for call in calls:
            gm.add_submodule(call.target, working.get_submodule(call.target))
        gm.graph = working.graph
        return gm

    apply_schedule.__name__ = (
        apply_schedule.__qualname__
    ) = f"schedule_{problem_dir.name}"
    register_graph_pass(apply_schedule)
    return apply_schedule.__name__

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pre-optimizer gradient hashing for the DistMoE paired numerics gate.

This module is benchmark-only instrumentation. A paired recipe installs it
before ``Optim`` is built. The wrapper hashes each rank-local gradient shard at
the entry to ``Optim.step``. At that point pipeline gradient finalization has
completed, but clipping and the parameter update have not started.

The digest is over the tensor's contiguous local byte representation. DTensor
gradients are never gathered, so the manifest describes both the global tensor
metadata and the rank-local shard that was hashed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import torch
from torch.distributed.tensor import DTensor


_CAPTURE_STEPS_ENV = "TORCHTITAN_PAIRED_NUMERICS_CAPTURE_STEPS"
_OUTPUT_ROOT_ENV = "TORCHTITAN_PAIRED_NUMERICS_OUTPUT_ROOT"
_INSTALLED_SCENARIO: str | None = None


def parse_capture_steps(raw_steps: str) -> tuple[int, ...]:
    """Parse a comma-separated, nonempty set of positive training steps."""
    try:
        steps = tuple(int(value.strip()) for value in raw_steps.split(","))
    except ValueError as error:
        raise ValueError(
            f"{_CAPTURE_STEPS_ENV} must contain comma-separated integers"
        ) from error
    if not steps or any(step <= 0 for step in steps):
        raise ValueError(f"{_CAPTURE_STEPS_ENV} must contain positive steps")
    if len(set(steps)) != len(steps):
        raise ValueError(f"{_CAPTURE_STEPS_ENV} must not contain duplicate steps")
    return tuple(sorted(steps))


def _normalize_fqn(fqn: str) -> str:
    return re.sub(r"\.parametrizations\.([^.]+)\.original", r".\1", fqn)


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def tensor_sha256(tensor: torch.Tensor, *, chunk_size: int = 16 << 20) -> str:
    """Hash a tensor's contiguous local bytes without a DTensor all-gather."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    local_bytes = (
        _local_tensor(tensor).detach().contiguous().view(torch.uint8).reshape(-1)
    )
    digest = hashlib.sha256()
    for start in range(0, local_bytes.numel(), chunk_size):
        chunk = local_bytes[start : start + chunk_size].cpu()
        digest.update(chunk.numpy().tobytes())
    return digest.hexdigest()


def _placements(tensor: torch.Tensor) -> list[str]:
    if not isinstance(tensor, DTensor):
        return []
    return [repr(placement) for placement in tensor.placements]


def _parameter_fqns(
    model_parts: list[torch.nn.Module], parameters: list[torch.nn.Parameter]
) -> list[str]:
    fqns: list[str] = []
    for part_index, model_part in enumerate(model_parts):
        name_by_id = {
            id(parameter): f"part{part_index}.{_normalize_fqn(name)}"
            for name, parameter in model_part.named_parameters()
        }
        for parameter in model_part.parameters():
            try:
                fqns.append(name_by_id[id(parameter)])
            except KeyError as error:
                raise RuntimeError(
                    "an optimizer parameter has no model FQN in "
                    f"pipeline part {part_index}"
                ) from error
    if len(fqns) != len(parameters):
        raise RuntimeError(
            "optimizer parameter order does not match the concatenated model parts: "
            f"{len(parameters)} parameters, {len(fqns)} FQNs"
        )
    if len(set(fqns)) != len(fqns):
        raise RuntimeError("paired numerics requires unique rank-local parameter FQNs")
    return fqns


def _gradient_record(
    index: int,
    fqn: str,
    parameter: torch.nn.Parameter,
) -> dict[str, Any]:
    gradient = parameter.grad
    if gradient is None:
        raise RuntimeError(f"parameter {fqn!r} has no gradient at optimizer step entry")
    local_gradient = _local_tensor(gradient)
    return {
        "index": index,
        "fqn": fqn,
        "dtype": str(gradient.dtype),
        "global_shape": list(gradient.shape),
        "local_shape": list(local_gradient.shape),
        "placements": _placements(gradient),
        "gradient_sha256": tensor_sha256(gradient),
    }


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary_path.replace(path)


def _capture_gradients(
    optim: Any,
    *,
    scenario: str,
    current_step: int,
) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()
    fqns: list[str] = optim._paired_numerics_parameter_fqns
    records = [
        _gradient_record(index, fqn, parameter)
        for index, (fqn, parameter) in enumerate(
            zip(fqns, optim.parameters, strict=True)
        )
    ]
    output_root = Path(os.environ[_OUTPUT_ROOT_ENV])
    output_path = (
        output_root
        / "gradient_sha256"
        / scenario
        / f"step_{current_step}"
        / f"rank_{rank}.json"
    )
    _write_json_atomic(
        output_path,
        {
            "format_version": 1,
            "scenario": scenario,
            "step": current_step,
            "rank": rank,
            "world_size": world_size,
            "parameters": records,
        },
    )


def install_gradient_sha256_capture(scenario: str) -> None:
    """Install one process-wide pre-optimizer gradient capture wrapper."""
    global _INSTALLED_SCENARIO
    if scenario not in {"eager", "graph_trainer"}:
        raise ValueError(f"unknown paired numerics scenario: {scenario!r}")
    if _INSTALLED_SCENARIO is not None:
        if _INSTALLED_SCENARIO != scenario:
            raise RuntimeError(
                "gradient capture is already installed for "
                f"{_INSTALLED_SCENARIO!r}, not {scenario!r}"
            )
        return
    if _OUTPUT_ROOT_ENV not in os.environ:
        raise RuntimeError(
            f"{_OUTPUT_ROOT_ENV} is required; use run_paired_numerics.py"
        )
    if _CAPTURE_STEPS_ENV not in os.environ:
        raise RuntimeError(
            f"{_CAPTURE_STEPS_ENV} is required; use run_paired_numerics.py"
        )
    capture_steps = set(parse_capture_steps(os.environ[_CAPTURE_STEPS_ENV]))

    from torchtitan.components.optim import Optim

    original_init = Optim.__init__
    original_step = Optim.step

    def init_with_fqns(self, config, **kwargs):
        original_init(self, config, **kwargs)
        self._paired_numerics_parameter_fqns = _parameter_fqns(
            kwargs["model_parts"], self.parameters
        )

    def step_with_capture(self, loss, *, current_step):
        if current_step in capture_steps:
            _capture_gradients(
                self,
                scenario=scenario,
                current_step=current_step,
            )
        return original_step(self, loss, current_step=current_step)

    Optim.__init__ = init_with_fqns
    Optim.step = step_with_capture
    _INSTALLED_SCENARIO = scenario

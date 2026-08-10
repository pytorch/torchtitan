# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import random
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 only
    import tomli as tomllib

from torchtitan.config import Configurable
from torchtitan.experiments.rl.sandbox import SandboxSpec

_CONVENTIONAL_ARTIFACT_PATH = "/logs/artifacts"


@dataclass(frozen=True, kw_only=True, slots=True)
class TerminalBenchSample:
    task_name: str
    instruction: str
    artifact_paths: tuple[str, ...]
    work_sandbox: SandboxSpec
    verifier_sandbox: SandboxSpec
    verifier_timeout_s: float


@dataclass(frozen=True, kw_only=True, slots=True)
class TerminalBenchArtifact:
    """One task artifact copied from the work sandbox to the host."""

    remote_path: str
    local_path: Path


class TerminalBenchDataset(Configurable):
    """Endless stream of single-container tasks from a Terminal-Bench checkout."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        tasks_dir: str = "terminal-bench/tasks"
        task_names: list[str] = field(default_factory=lambda: ["cli-2ph-simplex"])
        image_prefix: str = "torchtitan-terminal-bench"
        seed: int = 42
        shuffle: bool = True

        def __post_init__(self) -> None:
            if not self.task_names:
                raise ValueError("TerminalBenchDataset.task_names must not be empty")
            if not self.image_prefix:
                raise ValueError("TerminalBenchDataset.image_prefix must not be empty")

    def __init__(self, config: Config) -> None:
        self._samples = tuple(
            _load_sample(
                Path(config.tasks_dir) / task_name,
                image_prefix=config.image_prefix,
            )
            for task_name in config.task_names
        )
        self._rng = random.Random(config.seed)
        self._shuffle = config.shuffle
        self._order = list(range(len(self._samples)))
        if self._shuffle:
            self._rng.shuffle(self._order)
        self._pos = 0

    def __iter__(self) -> Iterator[TerminalBenchSample]:
        return self

    def __next__(self) -> TerminalBenchSample:
        if self._pos == len(self._order):
            if self._shuffle:
                self._rng.shuffle(self._order)
            self._pos = 0
        index = self._order[self._pos]
        self._pos += 1
        return self._samples[index]

    def state_dict(self) -> dict:
        return {
            "rng_state": self._rng.getstate(),
            "order": list(self._order),
            "pos": self._pos,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self._rng.setstate(state_dict["rng_state"])
        self._order = list(state_dict["order"])
        self._pos = state_dict["pos"]


def _load_sample(task_dir: Path, *, image_prefix: str) -> TerminalBenchSample:
    task_toml = task_dir / "task.toml"
    instruction_path = task_dir / "instruction.md"
    work_dockerfile = task_dir / "environment" / "Dockerfile"
    verifier_dockerfile = task_dir / "tests" / "Dockerfile"
    for path in (
        task_toml,
        instruction_path,
        work_dockerfile,
        verifier_dockerfile,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if (task_dir / "environment" / "docker-compose.yaml").exists() or (
        task_dir / "environment" / "docker-compose.yml"
    ).exists():
        raise ValueError(
            f"Terminal-Bench v1 supports only single-container tasks; got {task_dir}"
        )

    with task_toml.open("rb") as file:
        config = tomllib.load(file)
    environment = config.get("environment", {})
    verifier = config.get("verifier", {})
    if verifier.get("environment_mode") != "separate":
        raise ValueError(f"{task_toml} must use verifier.environment_mode='separate'")
    if int(environment.get("gpus", 0)) != 0:
        raise ValueError(f"Terminal-Bench v1 supports CPU tasks only; got {task_dir}")

    raw_artifacts = config.get("artifacts", [])
    if not all(isinstance(path, str) for path in raw_artifacts):
        raise ValueError(
            f"Terminal-Bench v1 supports string artifact paths only; got {task_toml}"
        )
    artifact_paths = tuple(dict.fromkeys([*raw_artifacts, _CONVENTIONAL_ARTIFACT_PATH]))
    for artifact_path in artifact_paths:
        _validate_artifact_path(artifact_path, task_toml=task_toml)

    task_name = str(config.get("task", {}).get("name") or task_dir.name)
    image_name = _image_name(image_prefix, task_dir.name)
    work_timeout_s = float(config.get("agent", {}).get("timeout_sec", 1800.0))
    verifier_timeout_s = float(verifier.get("timeout_sec", 600.0))
    work_spec = _sandbox_spec(
        environment,
        image=f"{image_name}-work:latest",
        timeout_s=work_timeout_s,
    )
    verifier_environment = {**environment, **verifier.get("environment", {})}
    verifier_spec = _sandbox_spec(
        verifier_environment,
        image=f"{image_name}-verifier:latest",
        timeout_s=verifier_timeout_s,
    )
    return TerminalBenchSample(
        task_name=task_name,
        instruction=instruction_path.read_text(encoding="utf-8"),
        artifact_paths=artifact_paths,
        work_sandbox=work_spec,
        verifier_sandbox=verifier_spec,
        verifier_timeout_s=verifier_timeout_s,
    )


def _sandbox_spec(
    config: dict,
    *,
    image: str,
    timeout_s: float,
) -> SandboxSpec:
    return SandboxSpec(
        image=image,
        num_cpus=int(config.get("cpus", 1)),
        memory_mb=int(config.get("memory_mb", 2048)),
        storage_mb=int(config["storage_mb"]) if "storage_mb" in config else None,
        timeout_s=timeout_s,
    )


def _image_name(prefix: str, task_name: str) -> str:
    slug = re.sub(r"[^a-z0-9_.-]+", "-", task_name.lower()).strip("-.")
    if not slug:
        raise ValueError(f"cannot derive an image name from task {task_name!r}")
    return f"{prefix.rstrip('-')}-{slug}"


def _validate_artifact_path(path: str, *, task_toml: Path) -> None:
    parsed = Path(path)
    if not parsed.is_absolute() or ".." in parsed.parts:
        raise ValueError(
            f"artifact paths in {task_toml} must be absolute and may not contain '..'; "
            f"got {path!r}"
        )

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc
import asyncio
from dataclasses import dataclass, field
from pathlib import Path

from torchtitan.config import Configurable


@dataclass(frozen=True, kw_only=True, slots=True)
class SandboxSpec:
    """Provider-neutral description of one isolated execution environment."""

    image: str
    num_cpus: int = 1
    memory_mb: int = 2048
    storage_mb: int | None = None
    timeout_s: float = 1800.0
    env: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.image:
            raise ValueError("SandboxSpec.image must not be empty")
        if self.num_cpus <= 0:
            raise ValueError(
                f"SandboxSpec.num_cpus must be positive; got {self.num_cpus}"
            )
        if self.memory_mb <= 0:
            raise ValueError(
                f"SandboxSpec.memory_mb must be positive; got {self.memory_mb}"
            )
        if self.storage_mb is not None and self.storage_mb <= 0:
            raise ValueError(
                "SandboxSpec.storage_mb must be positive when set; "
                f"got {self.storage_mb}"
            )
        if self.timeout_s <= 0:
            raise ValueError(
                f"SandboxSpec.timeout_s must be positive; got {self.timeout_s}"
            )


@dataclass(frozen=True, kw_only=True, slots=True)
class SandboxExecResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""


class SandboxPathNotFoundError(FileNotFoundError):
    """The requested path does not exist inside a sandbox."""


class SandboxSession(abc.ABC):
    """One stateful sandbox owned by a rollout or verifier operation."""

    @property
    @abc.abstractmethod
    def id(self) -> str:
        """Provider-assigned identifier for logging and cleanup."""

    @abc.abstractmethod
    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout_s: float | None = None,
    ) -> SandboxExecResult:
        """Execute a foreground command and return its exit code and output."""

    @abc.abstractmethod
    async def download(self, remote_path: str, local_path: Path) -> None:
        """Download one file or directory to ``local_path``."""

    @abc.abstractmethod
    async def upload(self, local_path: Path, remote_path: str) -> None:
        """Upload one file or directory to ``remote_path``."""

    @abc.abstractmethod
    async def close(self) -> None:
        """Destroy the sandbox. Implementations must be idempotent."""

    async def __aenter__(self) -> SandboxSession:
        return self

    async def __aexit__(self, exc_type, exc, traceback) -> None:
        await self.close()


class SandboxClient(Configurable, abc.ABC):
    """Creates sandbox sessions while bounding provider-side concurrency."""

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        max_concurrent_sessions: int = 32

        def __post_init__(self) -> None:
            if self.max_concurrent_sessions < 1:
                raise ValueError(
                    "max_concurrent_sessions must be >= 1; "
                    f"got {self.max_concurrent_sessions}"
                )

    def __init__(self, config: Config) -> None:
        self._session_slots = asyncio.Semaphore(config.max_concurrent_sessions)

    async def _acquire_session_slot(self) -> None:
        await self._session_slots.acquire()

    def _release_session_slot(self) -> None:
        self._session_slots.release()

    @abc.abstractmethod
    async def create(
        self,
        spec: SandboxSpec,
        *,
        owner_id: str,
    ) -> SandboxSession:
        """Provision a new session for ``owner_id``."""

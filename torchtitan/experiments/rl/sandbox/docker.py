# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import io
import logging
import shlex
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from torchtitan.experiments.rl.sandbox._transfer import (
    archive_local_path,
    materialize_archive,
)
from torchtitan.experiments.rl.sandbox.protocol import (
    SandboxClient,
    SandboxExecResult,
    SandboxPathNotFoundError,
    SandboxSession,
    SandboxSpec,
)

logger = logging.getLogger(__name__)


def _load_docker() -> Any:
    try:
        import docker
    except ImportError as error:
        raise ImportError(
            "DockerSandboxClient requires the optional 'docker' package. "
            "Install TorchTitan with the 'sandbox-docker' extra."
        ) from error
    return docker


class DockerSandboxClient(SandboxClient):
    """Local sandbox backend using the host Docker daemon."""

    @dataclass(kw_only=True, slots=True)
    class Config(SandboxClient.Config):
        pids_limit: int = 512

        def __post_init__(self) -> None:
            SandboxClient.Config.__post_init__(self)
            if self.pids_limit < 1:
                raise ValueError(f"pids_limit must be >= 1; got {self.pids_limit}")

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._pids_limit = config.pids_limit

    async def create(
        self,
        spec: SandboxSpec,
        *,
        owner_id: str,
    ) -> SandboxSession:
        await self._acquire_session_slot()
        docker_client = None
        try:
            docker_module = _load_docker()
            docker_client = await asyncio.to_thread(docker_module.from_env)
            kwargs: dict[str, Any] = {
                "image": spec.image,
                "entrypoint": "",
                "command": ["sh", "-c", "while :; do sleep 3600; done"],
                "detach": True,
                "remove": False,
                "environment": spec.env or None,
                "labels": {
                    "torchtitan.sandbox": "true",
                    "torchtitan.owner_id": owner_id,
                },
                "network_disabled": False,
                "nano_cpus": int(spec.num_cpus * 1_000_000_000),
                "mem_limit": f"{spec.memory_mb}m",
                "pids_limit": self._pids_limit,
                "security_opt": ["no-new-privileges"],
                "cap_drop": ["ALL"],
                "cap_add": ["SETGID", "SETPCAP", "SETUID"],
                "init": True,
            }
            container = await asyncio.to_thread(
                docker_client.containers.run,
                **kwargs,
            )
        except BaseException:
            if docker_client is not None:
                try:
                    await asyncio.to_thread(docker_client.close)
                finally:
                    self._release_session_slot()
            else:
                self._release_session_slot()
            raise

        if spec.storage_mb is not None:
            logger.warning(
                "DockerSandboxClient cannot portably enforce storage_mb=%d; "
                "the Docker daemon's storage policy applies",
                spec.storage_mb,
            )
        return _DockerSandboxSession(
            container=container,
            docker_client=docker_client,
            default_timeout_s=spec.timeout_s,
            release_slot=self._release_session_slot,
        )


class _DockerSandboxSession(SandboxSession):
    def __init__(
        self,
        *,
        container: Any,
        docker_client: Any,
        default_timeout_s: float,
        release_slot: Callable[[], None],
    ) -> None:
        self._container = container
        self._docker_client = docker_client
        self._default_timeout_s = default_timeout_s
        self._release_slot = release_slot
        self._closed = False
        self._close_lock = asyncio.Lock()

    @property
    def id(self) -> str:
        return str(self._container.id)

    async def exec(
        self,
        command: str,
        *,
        cwd: str | None = None,
        timeout_s: float | None = None,
    ) -> SandboxExecResult:
        self._ensure_open()
        effective_timeout = self._default_timeout_s if timeout_s is None else timeout_s
        if effective_timeout <= 0:
            raise ValueError(f"timeout_s must be positive; got {effective_timeout}")

        shell_command = command
        if cwd is not None:
            shell_command = f"cd {shlex.quote(cwd)} && {shell_command}"
        wrapped = (
            f"timeout -s KILL {effective_timeout}s "
            f"sh -c {shlex.quote(shell_command)}"
        )
        exit_code, output = await asyncio.to_thread(
            self._container.exec_run,
            ["sh", "-c", wrapped],
            demux=True,
        )
        stdout_raw, stderr_raw = output
        return SandboxExecResult(
            exit_code=int(exit_code),
            stdout=(stdout_raw or b"").decode("utf-8", errors="replace"),
            stderr=(stderr_raw or b"").decode("utf-8", errors="replace"),
        )

    async def download(self, remote_path: str, local_path: Path) -> None:
        self._ensure_open()
        docker_module = _load_docker()
        try:
            stream, _ = await asyncio.to_thread(
                self._container.get_archive,
                remote_path,
            )
            data = await asyncio.to_thread(lambda: b"".join(stream))
        except docker_module.errors.NotFound as error:
            raise SandboxPathNotFoundError(remote_path) from error
        await asyncio.to_thread(materialize_archive, data, local_path)

    async def upload(self, local_path: Path, remote_path: str) -> None:
        self._ensure_open()
        if not local_path.exists() and not local_path.is_symlink():
            raise FileNotFoundError(local_path)

        parent = PurePosixPath(remote_path).parent.as_posix()
        if not parent:
            parent = "/"
        mkdir_result = await self.exec(f"mkdir -p {shlex.quote(parent)}")
        if mkdir_result.exit_code != 0:
            raise RuntimeError(
                f"failed to create {parent!r} in Docker sandbox: "
                f"{mkdir_result.stderr}"
            )
        data = await asyncio.to_thread(
            archive_local_path,
            local_path,
            archive_name=PurePosixPath(remote_path).name,
        )
        accepted = await asyncio.to_thread(
            self._container.put_archive,
            parent,
            io.BytesIO(data),
        )
        if not accepted:
            raise RuntimeError(f"Docker rejected upload to {remote_path!r}")

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            try:
                await asyncio.to_thread(self._container.remove, force=True)
            finally:
                try:
                    await asyncio.to_thread(self._docker_client.close)
                finally:
                    self._closed = True
                    self._release_slot()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("sandbox session is closed")

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import asyncio
import math
import shlex
import tarfile
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any
from uuid import uuid4

from torchtitan.experiments.rl.sandbox._transfer import materialize_archive
from torchtitan.experiments.rl.sandbox.protocol import (
    SandboxClient,
    SandboxExecResult,
    SandboxPathNotFoundError,
    SandboxSession,
    SandboxSpec,
)


def _load_daytona_sdk() -> dict[str, Any]:
    try:
        from daytona import AsyncDaytona, CreateSandboxFromImageParams, Image, Resources
    except ImportError as error:
        raise ImportError(
            "DaytonaSandboxClient requires the optional 'daytona' package. "
            "Install TorchTitan with the 'sandbox-daytona' extra."
        ) from error
    return {
        "AsyncDaytona": AsyncDaytona,
        "CreateSandboxFromImageParams": CreateSandboxFromImageParams,
        "Image": Image,
        "Resources": Resources,
    }


class DaytonaSandboxClient(SandboxClient):
    """Remote sandbox backend using Daytona's async Python SDK."""

    @dataclass(kw_only=True, slots=True)
    class Config(SandboxClient.Config):
        create_timeout_s: float = 300.0
        auto_stop_interval_mins: int = 30
        auto_delete_interval_mins: int = 0

        def __post_init__(self) -> None:
            SandboxClient.Config.__post_init__(self)
            if self.create_timeout_s <= 0:
                raise ValueError(
                    f"create_timeout_s must be positive; got {self.create_timeout_s}"
                )
            if self.auto_stop_interval_mins < 0:
                raise ValueError("auto_stop_interval_mins must be non-negative")
            if self.auto_delete_interval_mins < 0:
                raise ValueError("auto_delete_interval_mins must be non-negative")

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._create_timeout_s = config.create_timeout_s
        self._auto_stop_interval_mins = config.auto_stop_interval_mins
        self._auto_delete_interval_mins = config.auto_delete_interval_mins

    async def create(
        self,
        spec: SandboxSpec,
        *,
        owner_id: str,
    ) -> SandboxSession:
        await self._acquire_session_slot()
        sdk = _load_daytona_sdk()
        client = sdk["AsyncDaytona"]()
        try:
            resources = sdk["Resources"](
                cpu=spec.num_cpus,
                memory=max(1, math.ceil(spec.memory_mb / 1024)),
                **(
                    {"disk": max(1, math.ceil(spec.storage_mb / 1024))}
                    if spec.storage_mb is not None
                    else {}
                ),
            )
            params = sdk["CreateSandboxFromImageParams"](
                image=sdk["Image"].base(spec.image),
                resources=resources,
                env_vars=spec.env,
                labels={
                    "torchtitan.sandbox": "true",
                    "torchtitan.owner_id": owner_id,
                },
                network_block_all=False,
                auto_stop_interval=self._auto_stop_interval_mins,
                auto_delete_interval=self._auto_delete_interval_mins,
                ephemeral=True,
            )
            sandbox = await client.create(params, timeout=self._create_timeout_s)
        except BaseException:
            try:
                await client.close()
            finally:
                self._release_session_slot()
            raise

        return _DaytonaSandboxSession(
            sandbox=sandbox,
            client=client,
            default_timeout_s=spec.timeout_s,
            release_slot=self._release_session_slot,
        )


class _DaytonaSandboxSession(SandboxSession):
    def __init__(
        self,
        *,
        sandbox: Any,
        client: Any,
        default_timeout_s: float,
        release_slot: Callable[[], None],
    ) -> None:
        self._sandbox = sandbox
        self._client = client
        self._default_timeout_s = default_timeout_s
        self._release_slot = release_slot
        self._closed = False
        self._close_lock = asyncio.Lock()

    @property
    def id(self) -> str:
        return str(self._sandbox.id)

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
        if cwd is not None:
            command = f"cd {shlex.quote(cwd)} && {command}"
        response = await self._sandbox.process.exec(
            f"sh -c {shlex.quote(command)}",
            timeout=math.ceil(effective_timeout),
        )
        # Daytona's foreground exec API currently returns one merged output
        # stream, so preserve it as stdout and leave stderr empty.
        return SandboxExecResult(
            exit_code=int(response.exit_code),
            stdout=response.result or "",
            stderr="",
        )

    async def download(self, remote_path: str, local_path: Path) -> None:
        self._ensure_open()
        try:
            info = await self._sandbox.fs.get_file_info(remote_path)
        except Exception as error:
            if _is_not_found(error):
                raise SandboxPathNotFoundError(remote_path) from error
            raise

        if not info.is_dir:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            await self._sandbox.fs.download_file(remote_path, str(local_path))
            return

        remote_archive = f"/tmp/torchtitan-download-{uuid4().hex}.tar"
        parent = PurePosixPath(remote_path).parent.as_posix() or "/"
        name = PurePosixPath(remote_path).name
        try:
            result = await self.exec(
                f"tar -C {shlex.quote(parent)} -cf {shlex.quote(remote_archive)} "
                f"{shlex.quote(name)}"
            )
            if result.exit_code != 0:
                raise RuntimeError(
                    f"failed to archive Daytona directory {remote_path!r}: "
                    f"{result.stdout}"
                )
            with tempfile.NamedTemporaryFile(suffix=".tar") as archive:
                await self._sandbox.fs.download_file(remote_archive, archive.name)
                data = await asyncio.to_thread(Path(archive.name).read_bytes)
            await asyncio.to_thread(materialize_archive, data, local_path)
        finally:
            await self.exec(f"rm -f {shlex.quote(remote_archive)}", timeout_s=30)

    async def upload(self, local_path: Path, remote_path: str) -> None:
        self._ensure_open()
        if not local_path.exists() and not local_path.is_symlink():
            raise FileNotFoundError(local_path)
        parent = PurePosixPath(remote_path).parent.as_posix() or "/"
        mkdir_result = await self.exec(f"mkdir -p {shlex.quote(parent)}")
        if mkdir_result.exit_code != 0:
            raise RuntimeError(
                f"failed to create {parent!r} in Daytona sandbox: "
                f"{mkdir_result.stdout}"
            )
        if local_path.is_file():
            await self._sandbox.fs.upload_file(str(local_path), remote_path)
            return

        remote_archive = f"/tmp/torchtitan-upload-{uuid4().hex}.tar"
        with tempfile.NamedTemporaryFile(suffix=".tar") as archive:
            with tarfile.open(archive.name, mode="w") as tar:
                tar.add(
                    local_path,
                    arcname=PurePosixPath(remote_path).name,
                    recursive=True,
                )
            await self._sandbox.fs.upload_file(archive.name, remote_archive)
        try:
            result = await self.exec(
                f"tar -C {shlex.quote(parent)} -xf {shlex.quote(remote_archive)}"
            )
            if result.exit_code != 0:
                raise RuntimeError(
                    f"failed to extract Daytona upload at {remote_path!r}: "
                    f"{result.stdout}"
                )
        finally:
            await self.exec(f"rm -f {shlex.quote(remote_archive)}", timeout_s=30)

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            try:
                await self._sandbox.delete()
            finally:
                try:
                    await self._client.close()
                finally:
                    self._closed = True
                    self._release_slot()

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("sandbox session is closed")


def _is_not_found(error: Exception) -> bool:
    status = getattr(error, "status", None) or getattr(error, "status_code", None)
    return status == 404 or type(error).__name__ == "DaytonaNotFoundError"

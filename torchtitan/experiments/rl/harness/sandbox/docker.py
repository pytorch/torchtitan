# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Local docker/podman sandbox backend.

Intended for single-box smoke tests of the coding-agent loop without any cloud
provider. Implements the same ``Sandbox`` Protocol as ``DaytonaSandbox`` so
example code that boots sandboxes via ``make_sandbox`` works unchanged.

Ported from THUDM/slime ``slime/agent/sandbox.py``.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import tempfile
from pathlib import Path

from torchtitan.experiments.rl.harness.sandbox.base import (
    _getenv,
    ExecResult,
    FileContent,
)

logger = logging.getLogger(__name__)


class DockerSandbox:
    """Async sandbox over a local docker/podman container.

    Env knobs:
      ``TT_DOCKER_RUNTIME``  -- ``docker`` (default) or ``podman``.
      ``TT_DOCKER_NETWORK``  -- container network (default ``host`` so the
                                in-container agent can dial the adapter on the
                                host loopback).
      ``TT_DOCKER_RUN_ARGS`` -- extra args appended to ``run`` (shlex-split).
    """

    runtime_env = ("TT_DOCKER_RUNTIME",)
    network_env = ("TT_DOCKER_NETWORK",)
    run_args_env = ("TT_DOCKER_RUN_ARGS",)

    def __init__(self, image: str, *, timeout: int | None = None, **_ignored) -> None:
        self.image = image
        self.timeout = timeout
        self._runtime = _getenv(*self.runtime_env, default="docker")
        self._network = _getenv(*self.network_env, default="host")
        self._extra_run = shlex.split(_getenv(*self.run_args_env, default=""))
        self._cid = ""
        self.sandbox_id = ""

    async def _cli(
        self,
        *args: str,
        input_bytes: bytes | None = None,
        timeout: int = 120,
    ) -> ExecResult:
        proc = await asyncio.create_subprocess_exec(
            self._runtime,
            *args,
            stdin=(
                asyncio.subprocess.PIPE
                if input_bytes is not None
                else asyncio.subprocess.DEVNULL
            ),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(
                proc.communicate(input=input_bytes), timeout=timeout
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
            raise
        return (
            proc.returncode or 0,
            out.decode("utf-8", "replace"),
            err.decode("utf-8", "replace"),
        )

    async def __aenter__(self) -> DockerSandbox:
        run = [
            "run",
            "-d",
            "--network",
            self._network,
            *self._extra_run,
            "--entrypoint",
            "sleep",
            self.image,
            "infinity",
        ]
        rc, out, err = await self._cli(*run, timeout=300)
        if rc != 0:
            raise RuntimeError(f"{self._runtime} run failed (rc={rc}): {err[:400]}")
        self._cid = out.strip().splitlines()[-1].strip()
        self.sandbox_id = self._cid[:12]
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._cid:
            try:
                await self._cli("rm", "-f", self._cid, timeout=60)
            except Exception as e:
                logger.warning("docker rm %s failed: %s", self.sandbox_id, e)

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
    ) -> ExecResult:
        args = ["exec", "-u", user]
        for k, v in (env or {}).items():
            args += ["-e", f"{k}={v}"]
        args += [self._cid, "bash", "-c", cmd]
        rc, out, err = await self._cli(*args, timeout=timeout)
        if check and rc != 0:
            raise RuntimeError(
                f"docker exec failed (exit={rc}): {cmd[:120]}\n{err[:400]}"
            )
        return rc, out, err

    async def write_file(
        self, sandbox_path: str, content: FileContent, *, user: str = "root"
    ) -> None:
        parent = os.path.dirname(sandbox_path) or "/"
        await self.exec(f"mkdir -p {shlex.quote(parent)}", user="root", check=False)

        cleanup = False
        if isinstance(content, Path):
            host_path = str(content)
        else:
            data = content.encode("utf-8") if isinstance(content, str) else content
            fd, host_path = tempfile.mkstemp(prefix="tt_docker_sb_")
            with os.fdopen(fd, "wb") as fp:
                fp.write(data)
            cleanup = True
        try:
            rc, _, err = await self._cli(
                "cp", host_path, f"{self._cid}:{sandbox_path}", timeout=600
            )
            if rc != 0:
                raise RuntimeError(f"{self._runtime} cp failed (rc={rc}): {err[:400]}")
        finally:
            if cleanup:
                try:
                    os.unlink(host_path)
                except OSError:
                    pass
        if user and user != "root":
            await self.exec(
                f"chown {shlex.quote(user)}:{shlex.quote(user)} {shlex.quote(sandbox_path)}",
                user="root",
                check=False,
            )

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        rc, out, _ = await self.exec(
            f"cat {shlex.quote(sandbox_path)}", user=user, timeout=60
        )
        return out if rc == 0 else ""

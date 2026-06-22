# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Daytona cloud sandbox backend.

Boots a remote cloud container from a Docker image. Meant for boxes that cannot
expose an inbound port to the cloud (an internal dev box): the agent inside the
sandbox reaches the on-box Anthropic adapter through a file-relay bridge over the
Daytona ``fs`` API (see ``bridge.py``), not a direct dial-back.

Ported from THUDM/slime ``slime/agent/sandbox.py``.
"""

from __future__ import annotations

import logging
import shlex
from pathlib import Path

from torchtitan.experiments.rl.harness.sandbox.base import (
    _getenv,
    ExecResult,
    FileContent,
)

logger = logging.getLogger(__name__)

# Label stamped on every sandbox we create, so cleanup can target ONLY our
# sandboxes (never another tenant's) when sharing a Daytona account.
HARNESS_LABELS = {"owner": "titan_swe_r2e"}


class DaytonaSandbox:
    """Async sandbox over a Daytona cloud sandbox (https://daytona.io).

    Daytona builds a snapshot wrapping any public image on first ``create`` and
    runs ``process.exec`` as the image's default user. R2E-Gym images default to
    ``root`` with the repo interpreter on ``PATH``, so ``exec`` runs as root and
    drops to an unprivileged user via ``runuser`` when asked. ``fs`` writes as
    root, so ``write_file`` uploads then chowns.

    Env knobs (``SLIME_AGENT_DAYTONA_*`` accepted as aliases):
      ``DAYTONA_API_KEY``                  -- API key (required).
      ``DAYTONA_API_URL`` / ``DAYTONA_TARGET`` -- override cloud endpoint/region.
      ``TT_DAYTONA_CPU``                   -- vCPUs per sandbox (default 2).
      ``TT_DAYTONA_MEM_GB``                -- memory GiB per sandbox (default 4).
      ``TT_DAYTONA_DISK_GB``               -- disk GiB per sandbox (default 10).
      ``TT_DAYTONA_CREATE_TIMEOUT``        -- snapshot-build/boot wait (default 900s).
    """

    api_key_env = ("DAYTONA_API_KEY",)
    api_url_env = ("DAYTONA_API_URL",)
    target_env = ("DAYTONA_TARGET",)

    def __init__(self, image: str, *, timeout: int | None = None, **_ignored) -> None:
        self.image = image
        self.timeout = timeout
        self._client = None
        self._sb = None
        self.sandbox_id = ""

    @property
    def daytona(self):
        """Underlying ``daytona.AsyncSandbox`` (for the fs-relay bridge)."""
        return self._sb

    async def __aenter__(self) -> DaytonaSandbox:
        from daytona import (  # type: ignore
            AsyncDaytona,
            CreateSandboxFromImageParams,
            DaytonaConfig,
            Resources,
        )

        api_key = _getenv(*self.api_key_env)
        if not api_key:
            raise RuntimeError(
                "DAYTONA_API_KEY is not set; required for the daytona sandbox backend."
            )
        cfg = DaytonaConfig(
            api_key=api_key,
            api_url=_getenv(*self.api_url_env) or None,
            target=_getenv(*self.target_env) or None,
        )
        self._client = AsyncDaytona(cfg)
        cpu = int(_getenv("TT_DAYTONA_CPU", "SLIME_AGENT_DAYTONA_CPU", default="2"))
        mem = int(
            _getenv("TT_DAYTONA_MEM_GB", "SLIME_AGENT_DAYTONA_MEM_GB", default="4")
        )
        disk = int(
            _getenv("TT_DAYTONA_DISK_GB", "SLIME_AGENT_DAYTONA_DISK_GB", default="10")
        )
        create_timeout = float(
            _getenv(
                "TT_DAYTONA_CREATE_TIMEOUT",
                "SLIME_AGENT_DAYTONA_CREATE_TIMEOUT",
                default="900",
            )
        )
        # Cloud-side TTL so an orphan (left by a SIGKILL'd run that never reached
        # __aexit__, e.g. MAST preemption) self-reaps: once it goes idle it
        # auto-stops after auto_stop minutes, then auto-deletes immediately
        # (auto_delete=0). A live rollout keeps the sandbox active (the host polls
        # its fs continuously via the bridge), so it is never stopped mid-run.
        auto_stop = int(_getenv("TT_DAYTONA_AUTO_STOP_MIN", default="30"))
        auto_delete = int(_getenv("TT_DAYTONA_AUTO_DELETE_MIN", default="0"))
        params = CreateSandboxFromImageParams(
            image=self.image,
            resources=Resources(cpu=cpu, memory=mem, disk=disk),
            labels=HARNESS_LABELS,
            auto_stop_interval=auto_stop,
            auto_delete_interval=auto_delete,
        )
        self._sb = await self._client.create(params, timeout=create_timeout)
        self.sandbox_id = self._sb.id
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            if self._sb is not None:
                await self._client.delete(self._sb)
        except Exception as e:
            logger.warning("daytona delete %s failed: %s", self.sandbox_id[:8], e)
        finally:
            try:
                if self._client is not None:
                    await self._client.close()
            except Exception:
                pass

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
    ) -> ExecResult:
        # exec lands as root on R2E images; drop privileges with runuser when a
        # non-root user is requested (keeps env via --whitelist-environment, the
        # same contract DockerSandbox uses).
        if user and user != "root":
            keys = ",".join((env or {}).keys())
            wl = f"--whitelist-environment={keys} " if keys else ""
            full = f"runuser -u {shlex.quote(user)} {wl}-- bash -c {shlex.quote(cmd)}"
        else:
            full = f"bash -c {shlex.quote(cmd)}"
        r = await self._sb.process.exec(full, env=env or None, timeout=timeout)
        rc = int(getattr(r, "exit_code", 0) or 0)
        out = getattr(r, "result", "") or ""
        if check and rc != 0:
            raise RuntimeError(
                f"daytona exec failed (exit={rc}): {cmd[:120]}\n{out[:400]}"
            )
        return rc, out, ""

    async def write_file(
        self, sandbox_path: str, content: FileContent, *, user: str = "root"
    ) -> None:
        import os

        parent = os.path.dirname(sandbox_path) or "/"
        await self.exec(f"mkdir -p {shlex.quote(parent)}", user="root", check=False)
        if isinstance(content, Path):
            data = content.read_bytes()
        elif isinstance(content, bytes):
            data = content
        else:
            data = str(content).encode("utf-8")
        # fs.upload_file writes as root; chown afterwards for non-root owners.
        await self._sb.fs.upload_file(data, sandbox_path)
        if user and user != "root":
            await self.exec(
                f"chown {shlex.quote(user)}:{shlex.quote(user)} {shlex.quote(sandbox_path)}",
                user="root",
                check=False,
            )

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        # root can read any file; the user arg is accepted for protocol parity.
        rc, out, _ = await self.exec(
            f"cat {shlex.quote(sandbox_path)}", user="root", timeout=60
        )
        return out if rc == 0 else ""

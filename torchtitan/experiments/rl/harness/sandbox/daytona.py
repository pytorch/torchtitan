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


def _eager_rebuild_daytona_models() -> None:
    """Resolve Daytona's Pydantic models at import (single-threaded) so concurrent
    sandbox creates never race their lazy rebuild.

    ``CreateSandboxFromImageParams`` has a forward-ref field (``StrictStr``) that
    Pydantic resolves lazily on first instantiation. Under a wide concurrent boot
    fanout (e.g. rollout_concurrency=48 in one controller) many first-uses race the
    rebuild and raise ``PydanticUserError: ... is not fully defined``. Calling
    ``model_rebuild`` once here, before any rollout, makes every later create safe.
    Best-effort: a no-op if Daytona isn't installed (e.g. CPU-only environments).
    """
    try:
        from pydantic import StrictStr  # noqa: F401 -- in scope for the rebuild

        from daytona import CreateSandboxFromImageParams  # type: ignore

        CreateSandboxFromImageParams.model_rebuild()
    except Exception as e:  # noqa: BLE001 -- import/rebuild is best-effort
        logger.warning("[daytona] eager model_rebuild skipped: %s", e)


_eager_rebuild_daytona_models()


# Process-wide AsyncDaytona client, shared by every sandbox in this worker: one
# client = one pooled TLS session reused across all concurrent rollouts. A
# per-sandbox client instead opens its own pool, and the handshake storm under a
# many-way boot fanout over a high-latency link times out the exec requests.
_SHARED_CLIENT = None
_SHARED_CLIENT_LOCK = None


async def _get_shared_client(*, api_key: str | None, api_url, target):
    global _SHARED_CLIENT, _SHARED_CLIENT_LOCK
    import asyncio

    from daytona import AsyncDaytona, DaytonaConfig  # type: ignore

    if not api_key:
        raise RuntimeError(
            "DAYTONA_API_KEY is not set; required for the daytona sandbox backend."
        )
    if _SHARED_CLIENT_LOCK is None:
        _SHARED_CLIENT_LOCK = asyncio.Lock()
    async with _SHARED_CLIENT_LOCK:
        if _SHARED_CLIENT is None:
            cfg = DaytonaConfig(api_key=api_key, api_url=api_url, target=target)
            _SHARED_CLIENT = AsyncDaytona(cfg)
    return _SHARED_CLIENT


class DaytonaSandbox:
    """Async sandbox over a Daytona cloud sandbox (https://daytona.io).

    Daytona builds a snapshot wrapping any public image on first ``create`` and
    runs ``process.exec`` as the image's default user. R2E-Gym images default to
    ``root`` with the repo interpreter on ``PATH``, so ``exec`` runs as root and
    drops to an unprivileged user via ``runuser`` when asked. ``fs`` writes as
    root, so ``write_file`` uploads then chowns.

    Env knobs:
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
        import asyncio
        import random

        from daytona import CreateSandboxFromImageParams, Resources  # type: ignore

        self._client = await _get_shared_client(
            api_key=_getenv(*self.api_key_env),
            api_url=_getenv(*self.api_url_env) or None,
            target=_getenv(*self.target_env) or None,
        )
        cpu = int(_getenv("TT_DAYTONA_CPU", default="2"))
        mem = int(_getenv("TT_DAYTONA_MEM_GB", default="4"))
        disk = int(_getenv("TT_DAYTONA_DISK_GB", default="10"))
        create_timeout = float(_getenv("TT_DAYTONA_CREATE_TIMEOUT", default="900"))
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
        # Daytona create transiently 401s a valid key under a concurrent boot burst.
        # Retry with jittered backoff so a wide fanout does not retry in lockstep.
        retries = int(_getenv("TT_DAYTONA_CREATE_RETRIES", default="5"))
        backoff = 5.0
        for attempt in range(retries + 1):
            try:
                self._sb = await self._client.create(params, timeout=create_timeout)
                break
            except Exception as e:
                if attempt >= retries:
                    raise
                logger.warning(
                    "daytona create failed (attempt %d/%d): %s",
                    attempt + 1,
                    retries + 1,
                    e,
                )
                await asyncio.sleep(backoff * (0.5 + random.random()))
                backoff = min(backoff * 2, 60.0)
        self.sandbox_id = self._sb.id
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        # Delete only this sandbox; never close the process-wide shared client --
        # other concurrent rollouts are still using its pooled connections.
        try:
            if self._sb is not None:
                await self._client.delete(self._sb)
        except Exception as e:
            logger.warning("daytona delete %s failed: %s", self.sandbox_id[:8], e)

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
    ) -> ExecResult:
        # A high-latency host->daytona link (e.g. cross-region MAST -> daytona-US)
        # can make even a fast command's exec request time out. Raise every exec's
        # timeout floor via TT_DAYTONA_EXEC_TIMEOUT_MIN (default 0 = unchanged).
        import os

        timeout = max(timeout, int(os.environ.get("TT_DAYTONA_EXEC_TIMEOUT_MIN", "0")))
        # exec lands as root on R2E images; drop privileges with runuser when a
        # non-root user is requested. Inline env as `env K=V` (not the SDK env=)
        # so the whole command is self-contained for the session API below.
        env_prefix = (
            "env " + " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + " "
            if env
            else ""
        )
        if user and user != "root":
            keys = ",".join((env or {}).keys())
            wl = f"--whitelist-environment={keys} " if keys else ""
            inner = f"runuser -u {shlex.quote(user)} {wl}-- bash -c {shlex.quote(cmd)}"
        else:
            inner = f"bash -c {shlex.quote(cmd)}"
        rc, out = await self._session_exec(env_prefix + inner, timeout=timeout)
        if check and rc != 0:
            raise RuntimeError(
                f"daytona exec failed (exit={rc}): {cmd[:120]}\n{out[:400]}"
            )
        return rc, out, ""

    async def _session_exec(self, full: str, *, timeout: int) -> tuple[int, str]:
        """Run ``full`` via the session API (create_session +
        execute_session_command(run_async) + poll) so a slow host->daytona link
        does not trip the exec-call timeout. Do NOT delete the session -- Daytona
        kills its child processes, which would kill the backgrounded claude.
        """
        import asyncio
        import os
        import random
        import uuid

        from daytona import SessionExecuteRequest

        retries = int(os.environ.get("TT_DAYTONA_EXEC_RETRIES", "5"))
        backoff = 5.0
        sid = cid = ""
        for attempt in range(retries + 1):
            try:
                sid = uuid.uuid4().hex
                await self._sb.process.create_session(sid)
                resp = await self._sb.process.execute_session_command(
                    sid,
                    SessionExecuteRequest(command=full, run_async=True),
                    timeout=timeout,
                )
                cid = resp.cmd_id
                if not cid:
                    raise RuntimeError("daytona session exec returned no cmd_id")
                break
            except Exception:
                try:
                    await self._sb.process.delete_session(sid)
                except Exception:
                    pass
                if attempt >= retries:
                    raise
                # Jittered backoff: spread retries over [0.5, 1.5)x so a wide
                # fanout does not retry in lockstep against the same rate limit.
                await asyncio.sleep(backoff * (0.5 + random.random()))
                backoff = min(backoff * 2, 60.0)

        loop = asyncio.get_event_loop()
        deadline = loop.time() + timeout + 120.0

        async def poll():
            # Daytona briefly returns an empty exit_code mid-command; the SDK then
            # raises "convert exit code to int". Treat only that as "still running".
            try:
                return await self._sb.process.get_session_command(sid, cid)
            except Exception as e:
                if "convert exit code" in str(e):
                    return None
                raise

        cmd = await poll()
        polls = 0
        while cmd is None or cmd.exit_code is None:
            if loop.time() > deadline:
                raise TimeoutError(
                    f"daytona exec poll exceeded {timeout + 120:.0f}s "
                    f"(sandbox likely stopped/deleted); cmd={full[:80]}"
                )
            await asyncio.sleep(0.1 if polls < 5 else 1.0)
            cmd = await poll()
            polls += 1

        logs = await self._sb.process.get_session_command_logs(sid, cid)
        out = getattr(logs, "stdout", "") or ""
        err = getattr(logs, "stderr", "") or ""
        return int(cmd.exit_code), out + err

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

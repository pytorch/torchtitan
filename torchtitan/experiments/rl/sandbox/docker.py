# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Docker/podman backend for the sandbox contract.

A container started from a per-instance image (repo + deps baked in) is a
faithful, isolated work + eval environment. The same ``SandboxFactory`` base
admits a future remote (microVM / snapshot) backend with no change to envs,
graders, or rollouters.

Design choices, on purpose:
- One-shot ``exec`` (no long-lived shell/REPL): no shell-state bugs, naturally
  parallel. Each ``exec`` starts at ``repo_root``; shell ``cd`` does not persist.
- Network is a provision-time container setting (default ``none``) rather than a
  per-call flag: the default blocks egress (an anti-cheat lock) and tasks whose
  images bake in their dependencies need no network to run.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import tempfile
from dataclasses import dataclass

from torchtitan.experiments.rl.sandbox.base import (
    ExecResult,
    Sandbox,
    SandboxFactory,
    SweProvisionError,
    SweProvisionUserError,
)

logger = logging.getLogger(__name__)

# Container label so leftover sandboxes from a crashed run (where the normal
# close()/finally teardown never ran) can be found and reaped. See reap_orphans.
SANDBOX_LABEL = "ttrl-sandbox"


class DockerSandbox:
    """A ``Sandbox`` backed by one long-lived docker/podman container.

    The container is started detached with ``sleep infinity`` as PID 1 so each
    ``exec`` is an independent ``<runtime> exec`` into it, with ``repo_root`` as
    the working directory.
    """

    def __init__(
        self,
        *,
        image: str,
        runtime: str,
        network: str,
        repo_root: str,
        run_id: str,
    ) -> None:
        self._image = image
        self._runtime = runtime
        self._network = network
        self._repo_root = repo_root
        self._run_id = run_id
        self._cid = ""

    async def start(self) -> None:
        """Create and start the container. Raises on failure (no half-open handle)."""
        rc, out, err, _ = await self._cli(
            "run",
            "-d",
            "--network",
            self._network,
            "--label",
            SANDBOX_LABEL,
            "--label",
            f"{SANDBOX_LABEL}-run={self._run_id}",
            "--entrypoint",
            "sleep",
            self._image,
            "infinity",
            timeout_s=600.0,
        )
        if rc != 0:
            # A bad image reference is non-retryable (user) input; pull/daemon
            # flakes are transient.
            msg = f"{self._runtime} run failed (rc={rc}): {err[:400]}"
            if "no such image" in err.lower() or "manifest unknown" in err.lower():
                raise SweProvisionUserError(msg)
            raise SweProvisionError(msg)
        self._cid = out.strip().splitlines()[-1].strip()

    async def exec(self, cmd: str, *, timeout_s: float) -> ExecResult:
        if not self._cid:
            raise RuntimeError("DockerSandbox.exec called before start()")
        rc, out, err, timed_out = await self._cli(
            "exec",
            "-w",
            self._repo_root,
            self._cid,
            "bash",
            "-c",
            cmd,
            timeout_s=timeout_s,
        )
        output = out
        if err:
            if output and not output.endswith("\n"):
                output += "\n"
            output += err
        return ExecResult(
            output=output,
            exit_code=None if timed_out else rc,
            timed_out=timed_out,
        )

    async def write_file(self, path: str, content: str) -> None:
        if not self._cid:
            raise RuntimeError("DockerSandbox.write_file called before start()")
        parent = os.path.dirname(path) or "/"
        await self.exec(f"mkdir -p {shlex.quote(parent)}", timeout_s=30.0)
        fd, host_path = tempfile.mkstemp(prefix="ttrl_sandbox_")
        try:
            with os.fdopen(fd, "w") as fp:
                fp.write(content)
            rc, _, err, _ = await self._cli(
                "cp", host_path, f"{self._cid}:{path}", timeout_s=300.0
            )
            if rc != 0:
                raise RuntimeError(f"{self._runtime} cp failed (rc={rc}): {err[:400]}")
        finally:
            try:
                os.unlink(host_path)
            except OSError:
                pass

    async def read_file(self, path: str) -> str:
        res = await self.exec(f"cat {shlex.quote(path)}", timeout_s=60.0)
        return res.output if res.exit_code == 0 else ""

    async def close(self) -> None:
        # Idempotent: a second close (or close on a never-started handle) is a
        # no-op. The rollouter closes every env in a finally, so this must not raise.
        if not self._cid:
            return
        cid, self._cid = self._cid, ""
        try:
            await self._cli("rm", "-f", cid, timeout_s=60.0)
        except Exception as e:
            logger.warning("[sandbox] %s rm %s failed: %s", self._runtime, cid[:12], e)

    async def _cli(self, *args: str, timeout_s: float) -> tuple[int, str, str, bool]:
        """Run the container CLI; return (rc, stdout, stderr, timed_out)."""
        proc = await asyncio.create_subprocess_exec(
            self._runtime,
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            out, err = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
        except (asyncio.TimeoutError, TimeoutError):
            try:
                proc.kill()
                await proc.wait()
            except ProcessLookupError:
                pass
            return -1, "", "", True
        return (
            proc.returncode or 0,
            out.decode("utf-8", "replace"),
            err.decode("utf-8", "replace"),
            False,
        )


class DockerSandboxFactory(SandboxFactory):
    """Provisions ``DockerSandbox`` instances over docker/podman."""

    @dataclass(kw_only=True, slots=True)
    class Config(SandboxFactory.Config):
        runtime: str = "podman"
        """Container CLI: ``podman`` or ``docker``."""

        network: str = "none"
        """Container network mode. ``none`` (default) blocks egress -- the
        anti-cheat lock; images that bake in their deps need no network. Set
        ``host`` only if a task genuinely needs network."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        # PID scopes this run's containers for reaping; a run is one process.
        self._run_id = str(os.getpid())

    async def _provision_one(self, *, image: str) -> Sandbox:
        sb = DockerSandbox(
            image=image,
            runtime=self._config.runtime,
            network=self._config.network,
            repo_root=self._config.repo_root,
            run_id=self._run_id,
        )
        try:
            await sb.start()
        except BaseException:
            await sb.close()
            raise
        return sb


async def reap_orphans(*, runtime: str = "podman") -> int:
    """Force-remove every sandbox container left over from a prior run.

    Best-effort cleanup for the crash path where ``close()`` never ran (SIGKILL,
    OOM). Matches on ``SANDBOX_LABEL`` only. Returns the number removed. Safe to
    call at startup on a single-box dev setup; do not call while another run's
    sandboxes are live.
    """
    proc = await asyncio.create_subprocess_exec(
        runtime,
        "ps",
        "-aq",
        "--filter",
        f"label={SANDBOX_LABEL}",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    out, _ = await proc.communicate()
    cids = out.decode("utf-8", "replace").split()
    if not cids:
        return 0
    rm = await asyncio.create_subprocess_exec(
        runtime,
        "rm",
        "-f",
        *cids,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    await rm.wait()
    logger.info("[sandbox] reaped %d orphan sandbox container(s)", len(cids))
    return len(cids)

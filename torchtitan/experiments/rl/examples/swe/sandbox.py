# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Sandbox execution surface for SWE coding-agent rollouts.

A ``Sandbox`` is the thinnest useful runtime handle: a stateless one-shot
``exec`` plus file read/write and an idempotent ``close``. It is a RUNTIME
object held in env state, never a config field. Backend selection is the one
config seam, exposed through ``SandboxFactory`` (a ``Configurable``): swapping
backends is a one-line config change.

Only a docker/podman backend ships today. The R2E-Gym task images ship the repo
at ``/testbed``, the interpreter on ``PATH``, and all dependencies baked in, so a
container started from the per-instance image is a faithful, isolated work + eval
environment. The same protocol admits a future remote (microVM / snapshot)
backend with no change to the env, rubric, or rollouter.

Design choices, on purpose:
- One-shot ``exec`` (no long-lived shell/REPL): no shell-state bugs, naturally
  parallel. Each ``exec`` starts at ``repo_root``; shell ``cd`` does not persist
  across calls -- the agent is told to use absolute paths or chain with ``&&``.
- Network is a provision-time container setting (default ``none``) rather than a
  per-call flag: R2E-Gym deps are baked into the image, so the eval needs no
  egress, and ``--network none`` enforces the anti-cheat egress block for free.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex
import tempfile
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from torchtitan.config import Configurable

logger = logging.getLogger(__name__)

# Container label so leftover sandboxes from a crashed run (where the normal
# close()/finally teardown never ran) can be found and reaped. See reap_orphans.
SANDBOX_LABEL = "ttrl-swe-sandbox"


@dataclass(frozen=True, slots=True)
class ExecResult:
    """Result of one ``Sandbox.exec`` call.

    ``output`` is stdout and stderr merged (stdout first), which is what the
    coding agent sees as a tool observation and what the test runner writes its
    pass/fail report to. ``exit_code`` is ``None`` when the command was killed
    for exceeding its timeout (``timed_out`` is then ``True``).
    """

    output: str
    exit_code: int | None
    timed_out: bool = False


@runtime_checkable
class Sandbox(Protocol):
    """Minimal async execution surface: one-shot exec + files + close.

    A RUNTIME handle held in env state; it never enters a Config. ``exec`` runs
    with ``repo_root`` as the working directory and is stateless across calls.
    """

    async def exec(self, cmd: str, *, timeout_s: float) -> ExecResult:
        ...

    async def write_file(self, path: str, content: str) -> None:
        ...

    async def read_file(self, path: str) -> str:
        ...

    async def close(self) -> None:
        """Release the sandbox. Idempotent and safe on a half-initialized handle."""
        ...


class SweProvisionError(RuntimeError):
    """A transient provisioning failure (image pull, daemon hiccup) worth retrying."""


class SweProvisionUserError(ValueError):
    """A non-retryable provisioning failure caused by a bad sample (e.g. bad image
    reference). Fail fast instead of burning the retry budget."""


class DockerSandbox:
    """A ``Sandbox`` backed by one long-lived docker/podman container.

    The container is started detached with ``sleep infinity`` as PID 1 so that
    each ``exec`` is an independent ``<runtime> exec`` into it. ``repo_root`` is
    the working directory for every ``exec``.
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
            # A bad image reference is a sample-level (user) error; pull/daemon
            # flakes are transient. Both surface here; treat unknown-image as
            # non-retryable so we fail fast on bad data.
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
        fd, host_path = tempfile.mkstemp(prefix="ttrl_swe_sb_")
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
        # Idempotent: a second close (or a close on a never-started handle) is a
        # no-op. The rollouter closes every env in a finally, so this must not raise.
        if not self._cid:
            return
        cid, self._cid = self._cid, ""
        try:
            await self._cli("rm", "-f", cid, timeout_s=60.0)
        except Exception as e:
            logger.warning(
                "[swe.sandbox] %s rm %s failed: %s", self._runtime, cid[:12], e
            )

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


class SandboxFactory(Configurable):
    """The backend config seam: builds a started ``Sandbox`` per rollout.

    Concurrency is capped by a per-factory semaphore (``max_concurrent_provision``)
    so a burst of rollouts cannot overwhelm the container daemon / disk. Share one
    factory across a run so the cap is global (the rollouter injects a single
    factory into every env).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        runtime: str = "podman"
        """Container CLI: ``podman`` or ``docker``."""

        network: str = "none"
        """Container network mode. ``none`` (default) blocks egress -- the
        anti-cheat egress lock; R2E-Gym images need no network to run their
        tests. Set ``host`` only if a task's eval genuinely needs network."""

        repo_root: str = "/testbed"
        """Working directory for every ``exec`` (the R2E-Gym repo root)."""

        max_concurrent_provision: int = 8
        """Max sandboxes provisioned at once across this factory."""

        provision_retries: int = 3
        """Attempts for a transient provision failure (image pull / daemon flake)."""

        provision_retry_delay_s: float = 5.0
        """Delay between provision retries."""

        provision_timeout_s: float = 600.0
        """Per-attempt provision timeout (first-use image pull can be slow)."""

    def __init__(self, config: Config) -> None:
        self._config = config
        self._sem = asyncio.Semaphore(config.max_concurrent_provision)
        # PID is enough to scope this run's containers for reaping; a run is a
        # single process. (Date/random are intentionally avoided here.)
        self._run_id = str(os.getpid())

    @property
    def repo_root(self) -> str:
        return self._config.repo_root

    async def provision(self, *, image: str) -> Sandbox:
        """Start and return a fresh sandbox for ``image``, under the concurrency cap."""
        async with self._sem:
            last: Exception | None = None
            for attempt in range(self._config.provision_retries):
                sb = DockerSandbox(
                    image=image,
                    runtime=self._config.runtime,
                    network=self._config.network,
                    repo_root=self._config.repo_root,
                    run_id=self._run_id,
                )
                try:
                    await asyncio.wait_for(
                        sb.start(), timeout=self._config.provision_timeout_s
                    )
                    return sb
                except SweProvisionUserError:
                    # Bad sample/image: fail fast, do not retry.
                    await sb.close()
                    raise
                except Exception as e:
                    last = e
                    await sb.close()
                    logger.warning(
                        "[swe.sandbox] provision attempt %d/%d for %s failed: %s",
                        attempt + 1,
                        self._config.provision_retries,
                        image,
                        str(e)[:200],
                    )
                    await asyncio.sleep(self._config.provision_retry_delay_s)
            raise SweProvisionError(
                f"provision exhausted {self._config.provision_retries} retries for {image}"
            ) from last


async def reap_orphans(*, runtime: str = "podman") -> int:
    """Force-remove every sandbox container left over from a prior run.

    Best-effort cleanup for the crash path where ``close()`` never ran (SIGKILL,
    OOM). Matches on ``SANDBOX_LABEL``, so it only touches this example's
    containers. Returns the number removed. Safe to call at startup on a
    single-box dev setup; do not call while another run's sandboxes are live.
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
    logger.info("[swe.sandbox] reaped %d orphan sandbox container(s)", len(cids))
    return len(cids)

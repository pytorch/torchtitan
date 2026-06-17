# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Task-agnostic sandbox contract for RL rollouts.

A ``Sandbox`` is the thinnest useful runtime handle for executing an agent's
actions in an isolated environment: a stateless one-shot ``exec`` plus file I/O
and an idempotent ``close``. It is a RUNTIME object held in env state, never a
config field.

This surface is shared, not SWE-specific: any "run commands in an isolated
environment" RL task (SWE repo-repair, terminal-bench-style shells, etc.) builds
its env, grader, and dataset on top of the same protocol. Backend selection is
the one config seam: ``SandboxFactory`` is an abstract ``Configurable`` whose
subclasses (e.g. ``DockerSandboxFactory``) each implement one backend, so
swapping backends is a one-line config change.

The base factory owns the cross-backend provisioning POLICY -- a global
concurrency cap and transient-failure retries -- so every backend gets it for
free and only implements ``_provision_one`` ("create one started sandbox").

Richer surfaces (e.g. a GUI/computer-use sandbox needing screenshots and input
events) should be a sibling protocol that reuses ``SandboxFactory``'s
provisioning lifecycle, not an overload of this minimal exec/file contract.
"""

from __future__ import annotations

import abc
import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from torchtitan.config import Configurable

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ExecResult:
    """Result of one ``Sandbox.exec`` call.

    ``output`` is stdout and stderr merged (stdout first), which is what an agent
    sees as a tool observation and what a test runner writes its report to.
    ``exit_code`` is ``None`` when the command was killed for exceeding its
    timeout (``timed_out`` is then ``True``).
    """

    output: str
    exit_code: int | None
    timed_out: bool = False


@runtime_checkable
class Sandbox(Protocol):
    """Minimal async execution surface: one-shot exec + files + close.

    A RUNTIME handle held in env state; it never enters a Config. ``exec`` runs
    with the factory's ``repo_root`` as the working directory and is stateless
    across calls (shell ``cd``/vars do not persist between calls).
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
    """A non-retryable provisioning failure caused by bad input (e.g. a bad image
    reference). Fail fast instead of burning the retry budget."""


class SandboxFactory(Configurable, abc.ABC):
    """Abstract backend seam: builds a started ``Sandbox`` per rollout.

    Subclass per backend and implement ``_provision_one``. The base owns the
    cross-backend policy: a per-factory semaphore caps concurrent provisions
    (share one factory across a run so the cap is global -- the rollouter injects
    a single factory into every env), and transient failures are retried while
    ``SweProvisionUserError`` fails fast.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        repo_root: str = "/testbed"
        """Working directory for every ``exec`` (the task repo root inside the sandbox)."""

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

    @property
    def repo_root(self) -> str:
        return self._config.repo_root

    async def provision(self, *, image: str) -> Sandbox:
        """Return a started sandbox for ``image`` under the concurrency cap, with retries."""
        async with self._sem:
            last: Exception | None = None
            for attempt in range(self._config.provision_retries):
                try:
                    return await asyncio.wait_for(
                        self._provision_one(image=image),
                        timeout=self._config.provision_timeout_s,
                    )
                except SweProvisionUserError:
                    raise  # bad input: do not retry
                except Exception as e:
                    last = e
                    logger.warning(
                        "[sandbox] provision attempt %d/%d for %s failed: %s",
                        attempt + 1,
                        self._config.provision_retries,
                        image,
                        str(e)[:200],
                    )
                    await asyncio.sleep(self._config.provision_retry_delay_s)
            raise SweProvisionError(
                f"provision exhausted {self._config.provision_retries} retries for {image}"
            ) from last

    @abc.abstractmethod
    async def _provision_one(self, *, image: str) -> Sandbox:
        """Create and return ONE started sandbox.

        Backends must clean up any partially started resources before raising,
        and should raise ``SweProvisionUserError`` for non-retryable bad input.
        """

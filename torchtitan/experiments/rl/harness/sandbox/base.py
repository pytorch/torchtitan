# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Provider-agnostic sandbox contract + backend factory.

The contract is intentionally small: async context management, command execution,
and file read/write. A coding-agent example builds its task-specific setup, agent
runner, and grader on top of this without depending on one sandbox provider.
The backend lives in a sibling module (``daytona.py``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Protocol, runtime_checkable


ExecResult = tuple[int, str, str]  # (exit_code, stdout, stderr)
FileContent = str | bytes | Path


@runtime_checkable
class Sandbox(Protocol):
    """Minimal async sandbox interface used by agent rollouts.

    ``write_file`` accepts either in-memory content (``str``/``bytes``) or a
    host ``Path`` to stream into the sandbox.
    """

    sandbox_id: str

    async def __aenter__(self) -> Sandbox:
        ...

    async def __aexit__(self, exc_type, exc, tb) -> None:
        ...

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
    ) -> ExecResult:
        ...

    async def write_file(
        self, sandbox_path: str, content: FileContent, *, user: str = "root"
    ) -> None:
        ...

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        ...


def _getenv(*names: str, default: str = "") -> str:
    """Return the first non-empty value among ``names`` (alias support)."""
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value
    return default


def make_sandbox(image: str, **kwargs) -> Sandbox:
    """Factory: build the sandbox backend selected by ``TT_SANDBOX_BACKEND``.

    Only ``daytona`` -> ``DaytonaSandbox`` is bundled; the factory is the seam for
    adding another provider as a new ``sandbox`` backend. The backend is imported
    lazily so a missing optional SDK (``daytona``) only errors when it is selected.
    """
    backend = _getenv("TT_SANDBOX_BACKEND", default="daytona").lower()
    if backend != "daytona":
        raise ValueError(
            f"unknown sandbox backend {backend!r}; only 'daytona' is bundled"
        )
    from torchtitan.experiments.rl.harness.sandbox.daytona import DaytonaSandbox

    return DaytonaSandbox(image, **kwargs)

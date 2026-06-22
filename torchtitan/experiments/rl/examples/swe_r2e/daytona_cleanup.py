# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Delete this harness's Daytona sandboxes, then confirm none remain.

A SIGKILL'd run never executes ``DaytonaSandbox.__aexit__``, so its cloud
sandboxes linger and consume the account disk quota (causing later runs to fail
with "Total disk limit exceeded"). The launcher calls this once at startup (clear
orphans from a prior killed run) and once at exit (clean up + confirm 0 remain).

Only sandboxes carrying our ``HARNESS_LABELS`` are deleted, so this is safe on a
Daytona account shared with other workloads. Defense in depth: every sandbox is
also created with a cloud-side auto-delete TTL (see ``DaytonaSandbox``), so an
orphan self-reaps even if this script never runs (e.g. the whole host dies).

Usage:
    python -m torchtitan.experiments.rl.examples.swe_r2e.daytona_cleanup
"""

from __future__ import annotations

import asyncio
import os

from torchtitan.experiments.rl.harness.sandbox.daytona import HARNESS_LABELS


async def _cleanup() -> int:
    from daytona import AsyncDaytona, DaytonaConfig, ListSandboxesQuery

    client = AsyncDaytona(DaytonaConfig(api_key=os.environ["DAYTONA_API_KEY"]))
    query = ListSandboxesQuery(labels=HARNESS_LABELS)
    try:
        sandboxes = [sb async for sb in client.list(query)]
        for sb in sandboxes:
            try:
                await client.delete(sb)
            except Exception as e:
                print(
                    f"[daytona_cleanup] delete {getattr(sb, 'id', '?')[:8]} failed: {e}"
                )
        # delete is async (the sandbox sits in "destroying" briefly), so poll the
        # count down to 0 before reporting -- otherwise the confirmation shows a
        # transient nonzero right after a delete.
        remaining = len(sandboxes)
        for _ in range(15):
            remaining = len([sb async for sb in client.list(query)])
            if remaining == 0:
                break
            await asyncio.sleep(1)
        print(
            f"[daytona_cleanup] deleted {len(sandboxes)} sandbox(es); remaining={remaining}"
        )
        return remaining
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(_cleanup())

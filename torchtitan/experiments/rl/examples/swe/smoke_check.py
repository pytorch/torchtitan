# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU-only end-to-end check of the sandbox + grading path against a real R2E-Gym
container (no GPU, no trainer).

Provisions a sandbox from the first bundled smoke instance, runs a couple of
``bash`` commands through the exec surface, then grades the UNMODIFIED (buggy)
repo. Expected outcome: the hidden tests run (``eval_ran=1``) but the issue is
unfixed, so ``resolved=0``. This proves provisioning, exec, file I/O, and the
junit grading harness work before spending GPUs on a full RL run.

Run:
    python -m torchtitan.experiments.rl.examples.swe.smoke_check
"""

from __future__ import annotations

import asyncio
import os

from torchtitan.experiments.rl.examples.swe import grading
from torchtitan.experiments.rl.examples.swe.data import R2EGymDataset
from torchtitan.experiments.rl.sandbox import DockerSandboxFactory

_SMOKE_DATA = os.path.join(os.path.dirname(__file__), "data", "r2e_smoke.jsonl")


async def _main() -> int:
    ds = R2EGymDataset(R2EGymDataset.Config(data_path=_SMOKE_DATA, shuffle=False))
    sample = next(iter(ds))
    print(f"instance: {sample.instance_id}  image: {sample.image}")

    factory = DockerSandboxFactory(DockerSandboxFactory.Config(runtime="podman"))
    print("provisioning sandbox (first-use image pull can take a while) ...")
    sandbox = await factory.provision(image=sample.image)
    try:
        res = await sandbox.exec("pwd && ls | head", timeout_s=60.0)
        print(f"\n$ pwd && ls | head  (exit={res.exit_code})\n{res.output}")

        print("grading the unmodified (buggy) repo ...")
        grade = await grading.grade_r2e(
            sandbox,
            sample=sample,
            repo_root=factory.repo_root,
            timeout_s=600.0,
        )
        print(f"grade: {grade}")
    finally:
        await sandbox.close()

    ok = grade["eval_ran"] == 1.0 and grade["resolved"] == 0.0
    print(
        "\nSANDBOX + GRADING OK (tests ran on buggy repo, resolved=0)"
        if ok
        else "\nFAIL: expected eval_ran=1, resolved=0"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))

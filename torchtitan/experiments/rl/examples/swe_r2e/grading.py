# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""R2E-Gym grading: score an agent's patch in a fresh sandbox.

No-test-cheating guarantee: the eval sandbox is built from the SAME image but
starts CLEAN (the agent never touched it), so only the model-produced diff affects
reward. We apply the diff, inject the dataset's hidden tests, run pytest with
junit-xml, and compare each test's status to ``expected_output_json`` -- the exact
per-test match that r2egym's ``_calculate_reward_r2e`` uses.

Ported from THUDM/slime ``examples/coding_agent_rl/sandbox.py`` (R2E paths only).
"""

from __future__ import annotations

import json
import logging
import os

from torchtitan.experiments.rl.harness import apply_pre_commands, make_sandbox, Sandbox

logger = logging.getLogger(__name__)

_PATCH = "/workspace/__cagent_patch__.diff"


async def evaluate_r2e(
    *,
    image: str,
    workdir: str,
    diff_text: str,
    r2e: dict,
    pre_commands: list[str] | str | None = None,
    timeout_sec: int = 600,
) -> tuple[float, bool, bool]:
    """Returns ``(reward, solved, applied_cleanly)`` for an R2E task.

    A fresh sandbox is booted from ``image``; ``pre_commands`` (e.g.
    ``git checkout <base_sha> -f``) re-align the baseline, the agent's diff is
    applied as root (R2E images run everything as root with the repo interpreter
    on PATH), then the hidden tests run and are compared to the expected statuses.
    """
    async with make_sandbox(image) as ev:
        if pre_commands:
            await apply_pre_commands(ev, workdir, pre_commands, user="root")
        applied = await _apply_diff(ev, workdir, diff_text, user="root")
        if not applied:
            return 0.0, False, False
        reward, solved = await _run_r2e(ev, workdir, r2e, timeout_sec)
        return reward, solved, True


async def _apply_diff(
    ev: Sandbox, workdir: str, diff_text: str, *, user: str = "root"
) -> bool:
    if not diff_text.strip():
        return True
    await ev.write_file(_PATCH, diff_text, user=user)
    for cmd in [
        f"cd {workdir} && git apply --3way --whitespace=nowarn {_PATCH}",
        f"cd {workdir} && git apply --whitespace=nowarn {_PATCH}",
        f"cd {workdir} && patch -p1 --no-backup-if-mismatch < {_PATCH}",
    ]:
        ec, _, _ = await ev.exec(cmd, user=user, check=False, timeout=120)
        if ec == 0:
            return True
    return False


def _parse_junit(xml_text: str) -> dict[str, str]:
    """junit-xml -> ``{test_name: PASSED|FAILED|SKIPPED}``. Indexes each case under
    both its bare name and ``Class.name`` so R2E expected keys (either form) match.
    """
    import xml.etree.ElementTree as ET

    out: dict[str, str] = {}
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return out
    for tc in root.iter("testcase"):
        name = tc.get("name") or ""
        cls = tc.get("classname") or ""
        status = "PASSED"
        for ch in tc:
            tag = ch.tag.lower()
            if tag in ("failure", "error"):
                status = "FAILED"
            elif tag == "skipped":
                status = "SKIPPED"
        if name:
            out[name] = status
            if cls:
                out[f"{cls.split('.')[-1]}.{name}"] = status
    return out


def _r2e_match(parsed: dict[str, str], expected: dict[str, str]) -> bool:
    """All expected ``(test -> status)`` must be reproduced. Falls back to substring
    matching (like r2egym) when keys don't line up exactly."""
    if not expected:
        return False
    for tname, estatus in expected.items():
        key = (
            tname
            if tname in parsed
            else next((k for k in parsed if tname in k or k in tname), None)
        )
        if key is None or parsed[key] != estatus:
            return False
    return True


async def _run_r2e(
    ev: Sandbox, workdir: str, r2e: dict, timeout: int
) -> tuple[float, bool]:
    names = r2e.get("test_file_names") or []
    codes = r2e.get("test_file_codes") or []
    expected_raw = r2e.get("expected_output_json") or "{}"
    expected = (
        json.loads(expected_raw)
        if isinstance(expected_raw, str)
        else dict(expected_raw)
    )

    for name, code in zip(names, codes):
        await ev.write_file(f"{workdir}/{name}", code, user="root")
    # conftest.py / fixtures are auto-loaded; only run the test_* modules.
    run_files = " ".join(n for n in names if os.path.basename(n).startswith("test"))
    if not run_files:
        return 0.0, False

    xml = f"{workdir}/.r2e_junit.xml"
    await ev.exec(
        f"cd {workdir} && python -m pytest -p no:cacheprovider -q {run_files} "
        f"--junit-xml={xml} > /tmp/r2e_pytest.log 2>&1 || true",
        user="root",
        env={"QT_QPA_PLATFORM": "offscreen", "MPLBACKEND": "Agg", "DISPLAY": ""},
        check=False,
        timeout=timeout,
    )
    parsed = _parse_junit(await ev.read_file(xml, user="root"))
    solved = _r2e_match(parsed, expected)
    return (1.0 if solved else 0.0), solved

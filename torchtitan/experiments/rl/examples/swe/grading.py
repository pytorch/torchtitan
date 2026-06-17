# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""R2E-Gym grading: run the hidden tests in the live sandbox and score.

Mirrors r2egym's reward: inject the instance's hidden test files, run them under
pytest with a junit-xml report, parse per-test PASSED/FAILED/SKIPPED, and require
that every entry in ``expected_output_json`` is reproduced exactly. The tests are
written fresh at grade time (overwriting anything the agent may have placed at
those paths), so the agent cannot tamper with the grading tests -- it can only
influence the score through its edits to the repo source.

Grading runs in the SAME sandbox the agent worked in. This is forced by the
rollout lifecycle: the rollouter tears every env (and its sandbox) down before
scoring, so the only place a live sandbox exists is the terminal env step. The
parse/compare functions are pure so they are unit-testable without a sandbox.
"""

from __future__ import annotations

import json
import logging
import os
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torchtitan.experiments.rl.examples.swe.data import R2EGymSample
    from torchtitan.experiments.rl.examples.swe.sandbox import Sandbox

logger = logging.getLogger(__name__)

# junit report path inside the sandbox; under /tmp to avoid clobbering the repo.
_JUNIT_XML = "/tmp/ttrl_swe_junit.xml"
# Headless env so GUI-importing test suites (e.g. orange3) do not hang/fail.
_HEADLESS_ENV = "QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg DISPLAY="


def parse_junit(xml_text: str) -> dict[str, str]:
    """Parse a junit-xml report into ``{test_name: PASSED|FAILED|SKIPPED}``.

    Each case is indexed under both its bare ``name`` and ``Class.name`` so an
    expected key in either form can be matched.
    """
    out: dict[str, str] = {}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out
    for tc in root.iter("testcase"):
        name = tc.get("name") or ""
        cls = tc.get("classname") or ""
        status = "PASSED"
        for child in tc:
            tag = child.tag.lower()
            if tag in ("failure", "error"):
                status = "FAILED"
            elif tag == "skipped":
                status = "SKIPPED"
        if name:
            out[name] = status
            if cls:
                out[f"{cls.split('.')[-1]}.{name}"] = status
    return out


def _match_key(parsed: dict[str, str], expected_name: str) -> str | None:
    """Resolve an expected test name to a parsed key, falling back to substring
    matching (as r2egym does) when class/name forms do not line up exactly."""
    if expected_name in parsed:
        return expected_name
    return next((k for k in parsed if expected_name in k or k in expected_name), None)


def matched_fraction(parsed: dict[str, str], expected: dict[str, str]) -> float:
    """Fraction of expected (test -> status) entries reproduced in ``parsed``."""
    if not expected:
        return 0.0
    hits = 0
    for name, status in expected.items():
        key = _match_key(parsed, name)
        if key is not None and parsed[key] == status:
            hits += 1
    return hits / len(expected)


def is_resolved(parsed: dict[str, str], expected: dict[str, str]) -> bool:
    """True only if EVERY expected (test -> status) entry is reproduced exactly."""
    if not expected:
        return False
    return all(
        (key := _match_key(parsed, name)) is not None and parsed[key] == status
        for name, status in expected.items()
    )


def _load_expected(sample: R2EGymSample) -> dict[str, str]:
    raw = sample.expected_output_json
    return json.loads(raw) if isinstance(raw, str) else dict(raw)


async def grade_r2e(
    sandbox: Sandbox, *, sample: R2EGymSample, repo_root: str, timeout_s: float
) -> dict[str, float]:
    """Run the hidden tests in ``sandbox`` and return per-rollout reward signals.

    Returns a dict consumed by ``RewardR2EGym``:
      ``resolved``      1.0 if all expected test statuses are reproduced, else 0.0.
      ``passed_frac``   fraction of expected test statuses reproduced (partial signal).
      ``eval_ran``      1.0 if the test run completed, 0.0 if it timed out / produced
                        no report (lets stale/failed grades be filtered out).
    """
    failed = {"resolved": 0.0, "passed_frac": 0.0, "eval_ran": 0.0}

    # Inject the canonical hidden tests fresh (anti-tamper: overwrite the agent's).
    for name, code in zip(sample.test_file_names, sample.test_file_codes):
        await sandbox.write_file(f"{repo_root}/{name}", code)

    run_files = [
        n for n in sample.test_file_names if os.path.basename(n).startswith("test")
    ]
    if not run_files:
        logger.warning(
            "[swe.grading] %s has no runnable test files", sample.instance_id
        )
        return failed

    joined = " ".join(run_files)
    res = await sandbox.exec(
        f"{_HEADLESS_ENV} python -m pytest -p no:cacheprovider -q {joined} "
        f"--junit-xml={_JUNIT_XML}",
        timeout_s=timeout_s,
    )
    if res.timed_out:
        logger.warning("[swe.grading] %s eval timed out", sample.instance_id)
        return failed

    xml_text = await sandbox.read_file(_JUNIT_XML)
    parsed = parse_junit(xml_text)
    if not parsed:
        logger.warning("[swe.grading] %s produced no junit report", sample.instance_id)
        return failed

    expected = _load_expected(sample)
    return {
        "resolved": 1.0 if is_resolved(parsed, expected) else 0.0,
        "passed_frac": matched_fraction(parsed, expected),
        "eval_ran": 1.0,
    }

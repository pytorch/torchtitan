# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Pytest wrapper for the bitwise parity torchrun script.

The pytest is responsible for launching a script with torchrun, loading the results
and running the assertions. The script runs the actual test and saves the results.

This is done this way so that the yaml is agnostic to what the test is doing, i.e.
the test executes torchrun, not the yaml."""

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from torchtitan.experiments.rl.tests.utils import h100_test

EXPECTED_CHECKS = {
    "batch_invariance",
    "trainer_vs_vllm_prefill",
    "vllm_decode_vs_prefill",
}


def _load_summary(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with path.open() as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        return {
            "status": "failed",
            "error": {
                "type": type(exc).__name__,
                "message": f"could not parse {path}: {exc}",
            },
        }


def _format_failure(
    *,
    result: subprocess.CompletedProcess[str],
    summary: dict[str, Any] | None,
    result_path: Path,
) -> str:
    parts = [f"bitwise parity script failed with exit code {result.returncode}"]
    if summary is None:
        parts.append(f"result file was not written: {result_path}")
    else:
        parts.append("result summary:")
        parts.append(json.dumps(summary, indent=2, sort_keys=True))

    parts.extend(
        [
            "stdout:",
            result.stdout,
            "stderr:",
            result.stderr,
        ]
    )
    return "\n".join(parts)


class TestBitwiseParityIntegration:
    @h100_test(num_gpus=2)
    def test_bitwise_parity(self, tmp_path):
        result_path = tmp_path / "bitwise_parity_result.json"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=2",
                "--module",
                "torchtitan.experiments.rl.tests.scripts.bitwise_parity",
                "--result-path",
                str(result_path),
            ],
            text=True,
            capture_output=True,
            env=os.environ.copy(),
        )

        summary = _load_summary(result_path)
        assert result.returncode == 0 and summary is not None, _format_failure(
            result=result,
            summary=summary,
            result_path=result_path,
        )
        assert summary.get("status") == "passed", _format_failure(
            result=result,
            summary=summary,
            result_path=result_path,
        )

        completed = {
            check.get("name")
            for check in summary.get("checks", [])
            if check.get("status") == "passed"
        }
        assert completed == EXPECTED_CHECKS, (
            "bitwise parity script did not report the expected checks\n"
            f"expected: {sorted(EXPECTED_CHECKS)}\n"
            f"actual: {sorted(completed)}\n"
            f"summary:\n{json.dumps(summary, indent=2, sort_keys=True)}"
        )

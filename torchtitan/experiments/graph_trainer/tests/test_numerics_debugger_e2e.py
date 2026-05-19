# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import sys

import pytest
import torch

from torchtitan.tools.compare_numerics import match_entries, parse_log


def _run_cmd(cmd: list[str], *, env: dict[str, str]) -> None:
    try:
        result = subprocess.run(
            cmd,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=600,
        )
    except subprocess.TimeoutExpired as e:
        stdout = e.stdout or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode("utf-8", errors="replace")
        tail = "\n".join(stdout.splitlines()[-80:])
        raise RuntimeError(
            f"Command timed out: {' '.join(cmd)}\nLast 80 lines:\n{tail}"
        ) from e
    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-80:])
        raise RuntimeError(
            f"Command failed with rc={result.returncode}: {' '.join(cmd)}\n"
            f"Last 80 lines:\n{tail}"
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_llama3_dump_numerics_eager_vs_aot_fx_trace(tmp_path):
    """E2E smoke test for ``--profiler.dump_numerics`` on Llama3.

    Runs a small two-step eager job and a small two-step graph-trainer
    ``aot_fx_trace`` replay job with the same deterministic settings, then
    verifies that both activation logs parse and that ``compare_numerics``
    writes HTML.
    """
    env = os.environ.copy()
    env.update(
        {
            "NGPU": "1",
            "LOG_RANK": "0",
        }
    )

    common_args = [
        "--training.steps",
        "2",
        "--profiler.dump_numerics",
        "--profiler.profile_freq",
        "2",
        "--debug.seed",
        "42",
        "--debug.deterministic",
        # Keep this E2E within smaller CI GPU memory. DebugMode records every
        # dispatch, so the default debugmodel batch=8/seq=2048 is too large.
        "--training.local_batch_size",
        "1",
        "--training.seq_len",
        "128",
        "--training.mixed_precision_param",
        "float32",
    ]

    eager_dir = tmp_path / "eager"
    eager_env = env | {"MODULE": "llama3", "CONFIG": "llama3_debugmodel"}
    _run_cmd(
        [
            "./run_train.sh",
            "--dump_folder",
            str(eager_dir),
            *common_args,
        ],
        env=eager_env,
    )

    traced_dir = tmp_path / "traced"
    traced_env = env | {
        "MODULE": "graph_trainer.llama3",
        "CONFIG": "graph_trainer_llama3_debugmodel",
    }
    _run_cmd(
        [
            "./run_train.sh",
            "--dump_folder",
            str(traced_dir),
            *common_args,
            "--compile.mode",
            "aot_fx_trace",
        ],
        env=traced_env,
    )

    eager_log = eager_dir / "numerics" / "rank_0_activations.log"
    traced_log = traced_dir / "numerics" / "rank_0_activations.log"
    eager_entries, _ = parse_log(str(eager_log))
    traced_entries, _ = parse_log(str(traced_log))
    assert len(eager_entries) > 0
    assert len(traced_entries) > 0
    assert len(eager_entries) > 400
    assert len(traced_entries) > 350

    results = match_entries(eager_entries, traced_entries)
    paired = [r for r in results if r[0] is not None and r[1] is not None]
    exact = [r for r in paired if r[4] == "exact"]
    fuzzy = [r for r in paired if r[4] == "fuzzy"]
    stats = [r for r in paired if r[4] == "stats"]
    numeric_matches = [r for r in paired if r[2] == "match"]
    diffs = [r for r in paired if r[2] == "diff"]
    eager_only = [r for r in results if r[2] == "eager_only"]
    traced_only = [r for r in results if r[2] == "traced_only"]
    match_summary = {
        "eager_entries": len(eager_entries),
        "traced_entries": len(traced_entries),
        "results": len(results),
        "paired": len(paired),
        "exact": len(exact),
        "fuzzy": len(fuzzy),
        "stats": len(stats),
        "numeric_matches": len(numeric_matches),
        "diffs": len(diffs),
        "eager_only": len(eager_only),
        "traced_only": len(traced_only),
    }
    print(f"numerics debugger e2e match summary: {match_summary}")
    assert len(paired) >= int(
        0.65 * min(len(eager_entries), len(traced_entries))
    ), match_summary
    assert len(exact) >= 100, match_summary
    assert len(numeric_matches) >= 250, match_summary

    html_path = tmp_path / "diff.html"
    _run_cmd(
        [
            sys.executable,
            "-m",
            "torchtitan.tools.compare_numerics",
            str(eager_log),
            str(traced_log),
            "--name1",
            "eager_debug_mode",
            "--name2",
            "traced_debug_mode",
            "--output",
            str(html_path),
        ],
        env=env,
    )

    html = html_path.read_text()
    assert "Numerics Comparison" in html
    assert "eager_debug_mode" in html
    assert "traced_debug_mode" in html

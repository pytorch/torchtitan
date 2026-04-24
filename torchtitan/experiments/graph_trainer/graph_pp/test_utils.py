# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Common testing infrastructure for graph PP integration and numerical parity tests."""

import subprocess
import sys
from dataclasses import dataclass


@dataclass
class GraphPPTestConfig:
    """Test configuration for graph PP integration tests."""

    pp_degree: int = 2
    dp_shard_degree: int = 2
    ep_degree: int = 2
    seq_len: int = 2048
    local_batch_size: int = 8
    microbatch_size: int = 2
    schedule: str = "Interleaved1F1B"
    seed: int = 42
    steps: int = 10
    ngpu: int = 8


def run_graph_pp_numerical_parity_test(
    *,
    baseline_module: str,
    baseline_config: str,
    test_module: str,
    test_config: str,
    schedule: str = "Interleaved1F1B",
    pp_degree: int = 2,
    dp_shard_degree: int = 2,
    ep_degree: int = 2,
    ngpu: int = 8,
    steps: int = 10,
    seed: int = 42,
    extra_baseline_options: str = "",
    extra_test_options: str = "",
) -> bool:
    """Run loss_compare.py to verify graph PP produces identical numerics to regular PP.

    Both runs use --debug.seed and --debug.deterministic for reproducibility.
    Returns True if assertion passed, raises on failure.
    """
    pp_opts = (
        f"--parallelism.pipeline_parallel_degree {pp_degree} "
        f"--parallelism.data_parallel_shard_degree {dp_shard_degree} "
        f"--parallelism.expert_parallel_degree {ep_degree} "
        f"--parallelism.pipeline_parallel_schedule {schedule}"
    )

    baseline_options = f"{pp_opts} {extra_baseline_options}".strip()
    test_options = f"{pp_opts} {extra_test_options}".strip()

    cmd = [
        sys.executable,
        "scripts/loss_compare.py",
        ".",
        ".",
        f"--baseline-module={baseline_module}",
        f"--baseline-config={baseline_config}",
        f"--test-module={test_module}",
        f"--test-config={test_config}",
        "--assert-equal",
        f"--steps={steps}",
        f"--ngpu={ngpu}",
    ]
    if baseline_options:
        cmd.append(f"--baseline-options={baseline_options}")
    if test_options:
        cmd.append(f"--test-options={test_options}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
    )

    if result.returncode != 0:
        raise AssertionError(
            f"Graph PP numerical parity test FAILED for schedule={schedule}\n"
            f"stdout: {result.stdout}\n"
            f"stderr: {result.stderr}"
        )

    return True

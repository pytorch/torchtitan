# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for the per-proc launch-env thread cap (rl/train.py)."""

import os

from torchtitan.experiments.rl.train import default_thread_cap, thread_cap_env


def test_thread_cap_env_sets_all_blas_backends() -> None:
    # All three backends must be capped together; missing one still oversubscribes.
    assert thread_cap_env(8) == {
        "OMP_NUM_THREADS": "8",
        "MKL_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "8",
    }


def test_default_thread_cap_partitions_cores_without_oversubscription() -> None:
    cpu_count = os.cpu_count() or 1
    for procs in (1, 2, 4, 8, cpu_count):
        n = default_thread_cap(procs)
        assert n >= 1  # never zero threads
        assert n * procs <= cpu_count  # thread pools sum to <= host cores

    # Degenerate case: more procs than cores -> floor at 1 thread/proc (can't do better).
    assert default_thread_cap(cpu_count * 4) == 1

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


@pytest.mark.parametrize(
    "recipe,num_gpus",
    [
        ("single_gpu", 1),
        pytest.param("tensor_parallel", 2, marks=pytest.mark.multi_gpu),
    ],
)
def test_trainer_smoke(recipe, num_gpus, tmp_path):
    if torch.cuda.device_count() < num_gpus:
        pytest.skip(f"requires {num_gpus} CUDA devices")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            f"--nproc_per_node={num_gpus}",
            "-m",
            "torchtitan.train",
            "--module",
            "torchtitan.experiments.peri_ln.tests.gpu_configs",
            "--config",
            recipe,
            "--output-dir",
            str(tmp_path),
        ],
        cwd=Path(__file__).resolve().parents[4],
        env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8"},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stdout

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

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

# Runs in a fresh process: allocator settings are process-wide.
_PIN_EXACTLY = """
import torch
from torch.distributed._state_dict_utils import _create_cpu_state_dict
from torchtitan.rl.generator.vllm.model_wrapper import _pin_large_blocks_exactly

torch.cuda.init()
_pin_large_blocks_exactly()
print(torch._C._accelerator_getAllocatorSettings())
# Just over a power of two above the threshold, where rounding nearly doubles a block.
weight = torch.empty(65 << 20, dtype=torch.uint8, device="cuda")
_create_cpu_state_dict({"weight": weight}, pin_memory=True)
print(torch.cuda.host_memory_stats()["allocated_bytes.current"])
"""


def test_weight_sync_buffers_are_pinned_at_their_exact_size():
    result = subprocess.run(
        [sys.executable, "-c", _PIN_EXACTLY],
        env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"},
        capture_output=True,
        text=True,
        check=True,
    )
    settings, pinned_bytes = result.stdout.split()

    assert settings == "max_split_size_mb:512,pinned_max_round_threshold_mb:64"
    assert int(pinned_bytes) == 65 << 20  # 128 MiB without the threshold

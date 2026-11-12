# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import subprocess
import sys


def test_graph_trainer_imports_without_triton() -> None:
    """GraphTrainer imports where Triton has no wheels; its kernels fail at launch."""
    script = r"""
import importlib
import importlib.abc
import sys


class BlockTriton(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if fullname == "triton" or fullname.startswith("triton."):
            raise ModuleNotFoundError(f"No module named {fullname!r}", name=fullname)
        return None


sys.meta_path.insert(0, BlockTriton())

from torchtitan.experiments.graph_trainer import paged_stash_memory_policy as policy
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer  # noqa: F401

assert "triton" not in sys.modules

# Only the kernel module needs Triton.
try:
    importlib.import_module("torchtitan.experiments.graph_trainer.paged_stash_kernels")
except ModuleNotFoundError as error:
    assert error.name == "triton", error
else:
    raise AssertionError("paged_stash_kernels imported without triton")

# A launch reports the missing Triton, before it touches the buffer.
buffer = object.__new__(policy.PagedStashBuffer)
try:
    buffer.launch_copy(None, None, None, None, None, hidden_size=0, num_blocks=0)
except ModuleNotFoundError as error:
    assert error.name == "triton", error
else:
    raise AssertionError("launch_copy ran without triton")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

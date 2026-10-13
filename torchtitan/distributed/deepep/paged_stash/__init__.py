# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.


"""MoE paged stashing for CUDA-graphable MoE: a port of Megatron-Core's.

* ``config``: ``PagedStashConfig``, shared by the eager trainer and
  GraphTrainer. Importing this package imports only the config.
* ``kernels``: the Triton page-copy kernels, verbatim from Megatron-Core.
* ``runtime``: the page buffers, the stash manager, the paged-layer schedule
  and the overflow runner, shared by both trainers.
* ``eager``: the eager trainer's frontend, which pages what a module region
  saves for backward.

GraphTrainer's frontend is
``torchtitan/experiments/graph_trainer/paged_stash_memory_policy.py``.
"""

from torchtitan.distributed.deepep.paged_stash.config import (
    OverflowCheckMode,
    PagedStashConfig,
)

__all__ = ["OverflowCheckMode", "PagedStashConfig"]

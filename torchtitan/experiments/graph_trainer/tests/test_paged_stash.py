# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""TorchTitan optimizer integration for GraphTrainer paged stashing."""

import pytest
import torch
from graph_trainer.paged_stash_memory_policy import PagedStashManager, PagedStashRunner
from torch import nn

from torchtitan.components.optim.optimizer import AdamW, OptimizersContainer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_paged_stash_runner_accepts_torchtitan_fused_adamw() -> None:
    PagedStashManager.reset_instance()
    manager = PagedStashManager.get_instance()
    manager.enabled = True
    manager.device = torch.device("cuda:0")
    manager.overflow = torch.zeros(1, dtype=torch.int64, device=manager.device)
    manager.host_spill = torch.zeros(1, dtype=torch.int64, device=manager.device)
    model = nn.Linear(8, 8, device=manager.device)
    optimizers = OptimizersContainer.Config(
        optimizers=[
            AdamW.Config(
                pattern=r".*",
                lr=0.1,
                fused=True,
                moment_dtype="bfloat16",
            )
        ]
    ).build(model_parts=[model])

    try:
        runner = PagedStashRunner(optimizers, overflow_check="deferred")
        assert runner._optimizers is optimizers
    finally:
        PagedStashManager.reset_instance()

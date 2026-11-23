# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""One-step recipes paired with fusion schedules for the GB300 kernel problem sets."""

import os
from pathlib import Path

from torchtitan.experiments.graph_trainer.trainer import GraphTrainer


def kernel_packs_root() -> Path:
    """Locate the kernel-packs checkout next to TorchTitan or at the supplied path."""
    sibling = Path(__file__).resolve().parents[4] / "graph-trainer-kernel-packs"
    return Path(os.environ.get("GRAPH_TRAINER_KERNEL_PACKS", sibling)).resolve()


def one_microbatch(config: GraphTrainer.Config) -> GraphTrainer.Config:
    config.training.num_tokens_per_train_step = -1
    config.training.steps = 1
    config.debug.seed = 42
    config.debug.deterministic = True
    return config

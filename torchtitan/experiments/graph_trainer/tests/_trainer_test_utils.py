# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import torch
import torch.nn as nn

from torchtitan.components.loss import cross_entropy_loss
from torchtitan.config import ActivationCheckpointConfig
from torchtitan.distributed.utils import get_train_context
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.trainer import Trainer


def build_minimal_trainer(
    model: nn.Module,
    model_config,
    trainer_cls: type[Trainer],
    *,
    activation_checkpoint_mode: str = "none",
    compile_enable_passes: bool = True,
    compile_passes: list[str] | None = None,
    compile_joint_passes: list[str] | None = None,
    tokenizer=None,
) -> Trainer:
    """Build the minimal Trainer/GraphTrainer needed for single-GPU test steps."""
    trainer = object.__new__(trainer_cls)
    trainer.model_parts = [model]
    trainer.loss_fn = cross_entropy_loss
    trainer.parallel_dims = SimpleNamespace(pp_enabled=False, cp_enabled=False)
    trainer.train_context = get_train_context(False)
    trainer.model_config = model_config
    trainer.device = torch.device("cuda")
    trainer.tokenizer = tokenizer

    if trainer_cls is GraphTrainer:
        trainer.config = SimpleNamespace(
            compile=SimpleNamespace(
                mode="aot_fx_trace",
                enable_passes=compile_enable_passes,
                passes=[] if compile_passes is None else list(compile_passes),
                joint_passes=[]
                if compile_joint_passes is None
                else list(compile_joint_passes),
                precompile_artifact_dir="",
            ),
            activation_checkpoint=ActivationCheckpointConfig(
                mode=activation_checkpoint_mode
            ),
        )
        trainer._fwd_bwd_step_module = None
        trainer._traced_step = None
    else:
        trainer.config = SimpleNamespace()

    return trainer

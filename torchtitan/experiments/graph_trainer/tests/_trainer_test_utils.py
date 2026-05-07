# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import torch
import torch.nn as nn

from torchtitan.components.loss import ChunkedCELoss, CrossEntropyLoss
from torchtitan.config import ActivationCheckpointConfig
from torchtitan.distributed.utils import get_train_context
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer
from torchtitan.trainer import Trainer


def build_minimal_trainer(
    model: nn.Module,
    model_config,
    trainer_cls: type[Trainer],
    *,
    loss_fn=None,
    activation_checkpoint_mode: str = "none",
    compile_enable_passes: bool = True,
    compile_passes: list[str] | None = None,
    compile_joint_passes: list[str] | None = None,
    compile_numerics_changing_optim: bool = False,
    tokenizer=None,
    fsdp_reshard_after_forward: str = "default",
) -> Trainer:
    """Build the minimal Trainer/GraphTrainer needed for single-GPU test steps."""
    trainer = object.__new__(trainer_cls)
    trainer.model_parts = [model]
    trainer.loss_fn = (
        loss_fn if loss_fn is not None else CrossEntropyLoss.Config().build()
    )
    if isinstance(trainer.loss_fn, ChunkedCELoss):
        lm_head = model.lm_head
        assert lm_head is not None, "Model must have lm_head for ChunkedCELoss"
        trainer.loss_fn.set_lm_head(lm_head)
        model._skip_lm_head = True
    trainer.parallel_dims = SimpleNamespace(pp_enabled=False, cp_enabled=False)
    trainer.train_context = get_train_context(False)
    trainer.model_config = model_config
    trainer.device = torch.device("cuda")
    trainer.tokenizer = tokenizer
    trainer.ntokens_seen = 0

    if trainer_cls is GraphTrainer:
        trainer.config = SimpleNamespace(
            compile=SimpleNamespace(
                mode="aot_fx_trace",
                enable_passes=compile_enable_passes,
                passes=[] if compile_passes is None else list(compile_passes),
                joint_passes=(
                    [] if compile_joint_passes is None else list(compile_joint_passes)
                ),
                precompile_artifact_dir="",
                memory_policy="default",
                inductor_compilation="regional",
                numerics_changing_optim=compile_numerics_changing_optim,
                enable_cudagraph=True,
                debug_graph_passes=False,
                cpu_offload_prefetch_n_layers=1,
                cpu_offload_defer_n_layers=1,
                cpu_offload_budget_gb=100.0,
            ),
            model_spec=SimpleNamespace(model=model_config),
            activation_checkpoint=ActivationCheckpointConfig(
                mode=activation_checkpoint_mode
            ),
            parallelism=SimpleNamespace(
                pipeline_parallel_degree=1,
                fsdp_reshard_after_forward=fsdp_reshard_after_forward,
                enable_async_tensor_parallel=False,
            ),
        )
        trainer._fwd_bwd_step_module = None
        trainer._traced_step = None
    else:
        trainer.config = SimpleNamespace()

    return trainer

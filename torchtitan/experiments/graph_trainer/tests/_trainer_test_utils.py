# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import torch
import torch.nn as nn

from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.distributed.utils import get_train_context
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
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
    compile_ep_overlap_chunk_dim: str = "batch",
    compile_ep_overlap_chunk_strategy: str = "graph",
    compile_ep_overlap_module_fqn: str = "layers.*",
    compile_ep_overlap_disable_early_grad_accumulation: bool = False,
    compile_inductor_compilation: str = "regional",
    compile_disable_passes: list[str] | None = None,
    compile_numerics_changing_optim: bool = False,
    tokenizer=None,
    fsdp_reshard_after_forward: str = "default",
) -> Trainer:
    """Build the minimal Trainer/GraphTrainer needed for single-GPU test steps."""
    trainer = object.__new__(trainer_cls)
    trainer.model_parts = [model]
    trainer.loss_fn = CrossEntropyLoss.Config().build()
    trainer.parallel_dims = SimpleNamespace(pp_enabled=False, cp_enabled=False)
    trainer.train_context = get_train_context(enable_loss_parallel=False)
    trainer.model_config = model_config
    trainer.device = torch.device("cuda")
    trainer.tokenizer = tokenizer
    trainer.ntokens_seen = 0

    if trainer_cls is GraphTrainer:
        trainer.config = SimpleNamespace(
            compile=GraphTrainerCompileConfig(
                enable=True,
                mode="aot_fx_trace",
                enable_passes=compile_enable_passes,
                passes=[] if compile_passes is None else list(compile_passes),
                disable_passes=(
                    []
                    if compile_disable_passes is None
                    else list(compile_disable_passes)
                ),
                inductor_compilation=compile_inductor_compilation,
                numerics_changing_optim=compile_numerics_changing_optim,
                ep_overlap_chunk_dim=compile_ep_overlap_chunk_dim,
                ep_overlap_chunk_strategy=compile_ep_overlap_chunk_strategy,
                ep_overlap_module_fqn=compile_ep_overlap_module_fqn,
                ep_overlap_disable_early_grad_accumulation=(
                    compile_ep_overlap_disable_early_grad_accumulation
                ),
            ),
            model_spec=SimpleNamespace(model=model_config),
            activation_checkpoint={
                "none": None,
                "selective": SelectiveAC.Config(),
                "full": FullAC.Config(),
            }[activation_checkpoint_mode],
            parallelism=SimpleNamespace(
                pipeline_parallel_degree=1,
                fsdp_reshard_after_forward=fsdp_reshard_after_forward,
                enable_async_tensor_parallel=False,
                spmd_backend="default",
            ),
        )
        trainer._fwd_bwd_step_module = None
        trainer._traced_step = None
    else:
        trainer.config = SimpleNamespace(
            parallelism=SimpleNamespace(spmd_backend="default"),
        )

    return trainer

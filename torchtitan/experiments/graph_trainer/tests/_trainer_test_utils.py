# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from collections.abc import Iterator
from contextlib import contextmanager
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.nn as nn

from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.config import DebugConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.experiments.graph_trainer.configs import (
    EpOverlapConfig,
    GraphTrainerCompileConfig,
)
from torchtitan.experiments.graph_trainer.trainer import (
    GraphTrainer,
    GraphTrainingEngine,
)
from torchtitan.trainer import Trainer
from torchtitan.training_engine import TrainingEngine


@contextmanager
def single_device_parallel_dims() -> Iterator[ParallelDims]:
    """Provide a real rank-1 mesh for tests that exercise model preprocessing."""
    owns_process_group = not dist.is_initialized()
    if owns_process_group:
        dist.init_process_group(
            backend="gloo",
            store=dist.HashStore(),
            rank=0,
            world_size=1,
        )

    try:
        parallel_dims = ParallelDims(
            dp_replicate=1,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            world_size=1,
            enable_sequence_parallel=False,
        )
        parallel_dims.build_mesh()
        yield parallel_dims
    finally:
        if owns_process_group:
            dist.destroy_process_group()


def build_minimal_trainer(
    model: nn.Module,
    model_config,
    trainer_cls: type[Trainer],
    *,
    activation_checkpoint_mode: str = "none",
    compile_enable_passes: bool = True,
    compile_enable_inplace_graph_gradient_accumulation: bool = False,
    compile_passes: list[str] | None = None,
    compile_ep_overlap_enabled: bool = False,
    compile_ep_overlap_chunk_dim: str = "batch",
    compile_ep_overlap_chunk_strategy: str = "graph",
    compile_ep_overlap_module_fqn: str = "layers.*",
    compile_ep_overlap_disable_early_grad_accumulation: bool = False,
    compile_inductor_compilation: str = "regional",
    compile_disable_passes: list[str] | None = None,
    compile_numerics_changing_optim: bool = False,
    tokenizer=None,
    fsdp_reshard_after_forward: str = "default",
    parallel_dims: ParallelDims,
) -> Trainer:
    """Build the minimal Trainer/GraphTrainer needed for single-GPU test steps."""
    trainer = object.__new__(trainer_cls)
    engine_cls = GraphTrainingEngine if trainer_cls is GraphTrainer else TrainingEngine
    trainer.engine = engine = object.__new__(engine_cls)
    engine.model_parts = [model]
    engine.loss_fn = CrossEntropyLoss.Config().build()
    engine.parallel_dims = parallel_dims
    engine.forward_backward_body_fn = engine._non_pp_forward_backward_body
    engine.model_config = model_config
    engine.device = torch.device("cuda")
    engine.preprocess_inputs_kwargs = {}
    trainer.tokenizer = tokenizer
    trainer.dataloader = SimpleNamespace(max_num_documents=None)
    engine.max_num_documents = None
    engine.ntokens_seen = 0
    engine.num_completed_steps = 0
    engine.sdc_replayer = None

    if trainer_cls is GraphTrainer:
        trainer.config = SimpleNamespace(
            compile=GraphTrainerCompileConfig(
                mode="aot_fx_trace",
                enable_passes=compile_enable_passes,
                enable_inplace_graph_gradient_accumulation=(
                    compile_enable_inplace_graph_gradient_accumulation
                ),
                passes=[] if compile_passes is None else list(compile_passes),
                disable_passes=(
                    []
                    if compile_disable_passes is None
                    else list(compile_disable_passes)
                ),
                inductor_compilation=compile_inductor_compilation,
                numerics_changing_optim=compile_numerics_changing_optim,
                ep_overlap=EpOverlapConfig(
                    enabled=compile_ep_overlap_enabled,
                    chunk_dim=compile_ep_overlap_chunk_dim,
                    strategy=compile_ep_overlap_chunk_strategy,
                    module_fqn=compile_ep_overlap_module_fqn,
                    disable_early_grad_accumulation=(
                        compile_ep_overlap_disable_early_grad_accumulation
                    ),
                ),
            ),
            model=model_config,
            activation_checkpoint={
                "none": None,
                "selective": SelectiveAC.Config(),
                "full": FullAC.Config(),
            }[activation_checkpoint_mode],
            dataloader=SimpleNamespace(max_num_documents=None),
            debug=DebugConfig(),
            training=TrainingConfig(),
            parallelism=SimpleNamespace(
                enable_sequence_parallel=False,
                pipeline_parallel_degree=1,
                fsdp_reshard_after_forward=fsdp_reshard_after_forward,
            ),
        )
        engine._traced_step = None
        engine._graph_runner = None
        engine._trainable_params = None
        engine._graph_gradient_state = None
        engine._pinned_pool_ctx = None
    else:
        trainer.config = SimpleNamespace(
            dataloader=SimpleNamespace(max_num_documents=None),
            training=TrainingConfig(),
            parallelism=SimpleNamespace(enable_sequence_parallel=False),
        )

    engine.config = trainer.config

    return trainer

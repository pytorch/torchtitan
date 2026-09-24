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
from torchtitan.distributed.utils import get_spmd_context
from torchtitan.experiments.graph_trainer.common_utils import (
    accumulate_param_grads_,
    maybe_register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.configs import (
    EpOverlapConfig,
    GraphTrainerCompileConfig,
    trace_input_preparer_keys,
)
from torchtitan.experiments.graph_trainer.graph_builder import make_fwd_bwd_step
from torchtitan.experiments.graph_trainer.make_fx_tracer import (
    minimal_fx_tracer,
    run_traced,
)
from torchtitan.experiments.graph_trainer.passes import (
    apply_graph_passes,
    construct_default_graph_passes,
    construct_mandatory_graph_passes,
)
from torchtitan.experiments.graph_trainer.registry import (
    PASS_PIPELINE_REGISTRY,
    TRACE_CALL_INPUT_PREPARERS,
    TRACE_INPUT_PREPARERS,
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
        engine._pinned_pool_ctx = None
        engine._traced_step = None
        engine._test_graph_call = None

        def prepare_trace_inputs(args, kwargs) -> None:
            for pass_name in trace_input_preparer_keys(trainer.config.compile):
                prepare = TRACE_INPUT_PREPARERS.get(pass_name)
                if prepare is not None:
                    prepare(trainer.config.compile, args, kwargs)

        def prepare_trace_call_inputs(args, kwargs):
            for pass_name in trace_input_preparer_keys(trainer.config.compile):
                prepare = TRACE_CALL_INPUT_PREPARERS.get(pass_name)
                if prepare is not None:
                    prepared = prepare(trainer.config.compile, args, kwargs)
                    if prepared is not None:
                        args, kwargs = prepared
            return args, kwargs

        def run_direct_graph_step(*, inputs, labels, model_kwargs, loss_kwargs):
            # This test adapter traces one unsplit joint graph. Unwrap the
            # runtime's single-microbatch calling convention at its boundary.
            (input_args,) = inputs
            (inputs,) = input_args
            (labels,) = labels
            (model_kwargs,) = model_kwargs

            if engine._traced_step is None:
                maybe_register_blockmask_pytree_node()
                fwd_bwd_fn = make_fwd_bwd_step(model, engine.loss_fn)
                with get_spmd_context(
                    parallel_dims=parallel_dims,
                    spmd_typechecking=False,
                ):
                    engine._traced_step = minimal_fx_tracer(
                        fwd_bwd_fn,
                        module=model,
                        prepare_inputs=prepare_trace_inputs,
                        prepare_call_inputs=prepare_trace_call_inputs,
                    )(
                        inputs,
                        labels,
                        loss_kwargs["global_valid_tokens"],
                        model_kwargs,
                    )
                if trainer.config.compile.enable_passes:
                    pipeline_fn = PASS_PIPELINE_REGISTRY.get(
                        trainer.config.compile.pass_pipeline,
                        construct_default_graph_passes,
                    )
                    passes = pipeline_fn(
                        engine._traced_step,
                        trainer.config,
                        parallel_dims=engine.parallel_dims,
                    )
                else:
                    passes = construct_mandatory_graph_passes()
                engine._traced_step.gm = apply_graph_passes(
                    engine._traced_step.gm,
                    engine._traced_step.example_inputs,
                    passes,
                    compile_config=trainer.config.compile,
                    respect_disable_passes=trainer.config.compile.enable_passes,
                )
                engine._test_graph_call = run_traced(
                    engine._traced_step,
                    module=model,
                )

            outputs = engine._test_graph_call(
                inputs,
                labels,
                loss_kwargs["global_valid_tokens"],
                model_kwargs,
            )
            params = tuple(
                parameter for parameter in model.parameters() if parameter.requires_grad
            )
            accumulate_param_grads_(
                params,
                outputs[1:],
                clone_grads_to_initialize_param_grad=True,
            )
            return outputs[0]

        # Compiler component tests inspect the unsplit graph directly.
        engine.forward_backward_body_fn = run_direct_graph_step
    else:
        trainer.config = SimpleNamespace(
            dataloader=SimpleNamespace(max_num_documents=None),
            training=TrainingConfig(),
            parallelism=SimpleNamespace(enable_sequence_parallel=False),
        )

    engine.config = trainer.config

    return trainer

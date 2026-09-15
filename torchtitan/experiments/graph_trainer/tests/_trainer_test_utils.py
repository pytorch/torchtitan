# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from types import SimpleNamespace

import torch
import torch.nn as nn

from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.config import TrainingConfig
from torchtitan.distributed.activation_checkpoint import FullAC, SelectiveAC
from torchtitan.distributed.utils import get_spmd_context
from torchtitan.experiments.graph_trainer.common_utils import accumulate_param_grads_
from torchtitan.experiments.graph_trainer.configs import (
    EpOverlapConfig,
    GraphTrainerCompileConfig,
    trace_input_preparer_keys,
)
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
from torchtitan.experiments.graph_trainer.trainer import GraphTrainer, make_fwd_bwd_step
from torchtitan.trainer import Trainer


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
) -> Trainer:
    """Build the minimal Trainer/GraphTrainer needed for single-GPU test steps."""
    trainer = object.__new__(trainer_cls)
    trainer.model_parts = [model]
    trainer.loss_fn = CrossEntropyLoss.Config().build()
    trainer.parallel_dims = SimpleNamespace(
        pp_enabled=False,
        cp_enabled=False,
        spmd_backend="partial_dtensor",
    )
    trainer.train_context = get_spmd_context()
    trainer.fwd_bwd_fn = trainer._forward_backward_body
    trainer.model_config = model_config
    trainer.device = torch.device("cuda")
    trainer.tokenizer = tokenizer
    trainer.dataloader = SimpleNamespace(max_num_documents=None)
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
            model_spec=SimpleNamespace(model=model_config),
            activation_checkpoint={
                "none": None,
                "selective": SelectiveAC.Config(),
                "full": FullAC.Config(),
            }[activation_checkpoint_mode],
            dataloader=SimpleNamespace(max_num_documents=None),
            training=TrainingConfig(),
            parallelism=SimpleNamespace(
                pipeline_parallel_degree=1,
                fsdp_reshard_after_forward=fsdp_reshard_after_forward,
                spmd_backend="partial_dtensor",
            ),
        )
        trainer._traced_step = None
        trainer._test_graph_call = None

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

        def run_direct_graph_step(microbatches, global_valid_tokens):
            assert len(microbatches) == 1
            inputs, labels, extra_kwargs = microbatches[0]
            if trainer._traced_step is None:
                fwd_bwd_fn = make_fwd_bwd_step(model, trainer.loss_fn)
                with trainer.train_context():
                    trainer._traced_step = minimal_fx_tracer(
                        fwd_bwd_fn,
                        module=model,
                        prepare_inputs=prepare_trace_inputs,
                        prepare_call_inputs=prepare_trace_call_inputs,
                    )(inputs, labels, global_valid_tokens, extra_kwargs)
                if trainer.config.compile.enable_passes:
                    pipeline_fn = PASS_PIPELINE_REGISTRY.get(
                        trainer.config.compile.pass_pipeline,
                        construct_default_graph_passes,
                    )
                    passes = pipeline_fn(
                        trainer._traced_step,
                        trainer.config,
                        parallel_dims=trainer.parallel_dims,
                    )
                else:
                    passes = construct_mandatory_graph_passes()
                trainer._traced_step.gm = apply_graph_passes(
                    trainer._traced_step.gm,
                    trainer._traced_step.example_inputs,
                    passes,
                    compile_config=trainer.config.compile,
                    respect_disable_passes=trainer.config.compile.enable_passes,
                )
                trainer._test_graph_call = run_traced(
                    trainer._traced_step,
                    module=model,
                )

            outputs = trainer._test_graph_call(
                inputs,
                labels,
                global_valid_tokens,
                extra_kwargs,
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

        # Compiler component tests use a direct traced callable so they can
        # inspect the unsplit graph.
        trainer.fwd_bwd_fn = run_direct_graph_step
    else:
        trainer.config = SimpleNamespace(
            dataloader=SimpleNamespace(max_num_documents=None),
            training=TrainingConfig(),
            parallelism=SimpleNamespace(spmd_backend="partial_dtensor"),
        )

    return trainer

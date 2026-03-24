# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unified JIT/AOT compilation dispatcher for graph_trainer training.

Supports two compilation modes via --compile.mode:
- JIT: standard torch.compile() with custom backend
- AOT: explicit joint graph export + custom graph passes
"""

import functools
from collections.abc import Callable
from typing import Any

import torch
import torch.nn as nn

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy
from torchtitan.experiments.graph_trainer.common_utils import (
    get_transformer_block_buckets,
    parallelize_inputs,
    register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig
from torchtitan.experiments.graph_trainer.graph_utils import (
    CompiledModule,
    get_compiler_passes_from_config,
    get_joint_custom_passes_from_config,
    joint_graph_builder,
    make_compiler_with_passes,
    pp_joint_graph_builder,
)
from torchtitan.experiments.graph_trainer.jit_backend import (
    get_compile_backend_with_passes,
)
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.protocols.model_spec import ParallelizeFunction
from torchtitan.tools.logging import logger


def _apply_jit_compile(
    model: nn.Module,
    compile_config: GraphTrainerCompileConfig,
    fsdp_reshard_after_forward: bool,
) -> nn.Module:
    """Apply JIT compilation (torch.compile with custom backend)."""
    transformer_block_buckets = get_transformer_block_buckets(model)
    backend = get_compile_backend_with_passes(
        compile_config,
        fsdp_reshard_after_forward,
        transformer_block_buckets,
    )
    model = torch.compile(
        model,
        backend=backend,
        fullgraph=True,
    )
    logger.info("Applied JIT compilation (torch.compile) to the model")
    return model


def _apply_aot_compile(
    model: nn.Module,
    parallel_dims: ParallelDims,
    compile_config: GraphTrainerCompileConfig,
    dump_folder: str,
    fsdp_reshard_after_forward: bool,
    joint_passes: list,
) -> CompiledModule:
    """Apply AOT compilation (joint graph export + pass pipeline)."""
    register_blockmask_pytree_node()

    # Get joint custom passes from config
    joint_custom_passes = get_joint_custom_passes_from_config(
        parallel_dims, compile_config, fsdp_reshard_after_forward
    )
    # Prepend any user-configured joint passes
    joint_custom_passes = joint_passes + joint_custom_passes

    # Get compiler passes from config
    compiler_passes = get_compiler_passes_from_config(
        model, compile_config, parallel_dims
    )

    # Create compilers with specified passes
    fw_compiler, bw_compiler = make_compiler_with_passes(
        compiler_passes, dump_folder=dump_folder
    )

    # Create custom joint_graph_builder with compilers
    model_joint_graph_builder = functools.partial(
        joint_graph_builder,
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        joint_custom_passes=joint_custom_passes,
        dump_folder=dump_folder,
        compile_config=compile_config,
    )

    model = CompiledModule(
        model, parallel_dims, model_joint_graph_builder, parallelize_inputs
    )
    logger.info("Applied AOT compilation (joint graph export) to the model")
    return model


def apply_compile(
    model: nn.Module,
    *,
    compile_config: GraphTrainerCompileConfig,
    parallelism: ParallelismConfig,
    parallel_dims: ParallelDims,
    dump_folder: str,
) -> nn.Module:
    """
    Apply compilation to the model based on the configured mode.

    Note: Graph PP compilation is handled separately by graph_pp_pipeline_llm,
    not by this function. This only handles per-stage or non-PP compilation.

    Args:
        model: The model to compile
        compile_config: Compilation configuration with mode and passes
        parallelism: Parallelism configuration
        parallel_dims: Parallel dimensions
        dump_folder: Folder for dumping debug graphs
    """
    if not compile_config.enable:
        return model

    mode = compile_config.mode
    if mode is None:
        logger.info("No compile mode set, skipping compilation")
        return model

    torch._inductor.config.reorder_for_peak_memory = False
    torch._dynamo.config.capture_scalar_outputs = True

    fsdp_reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        parallelism.fsdp_reshard_after_forward, parallel_dims.pp_enabled
    )

    if mode == "jit":
        if "model" not in compile_config.components:
            return model
        return _apply_jit_compile(
            model,
            compile_config,
            fsdp_reshard_after_forward,
        )
    elif mode == "aot":
        return _apply_aot_compile(
            model,
            parallel_dims,
            compile_config,
            dump_folder,
            fsdp_reshard_after_forward,
            joint_passes=[],
        )
    else:
        raise ValueError(f"Unknown compile mode: {mode}. Must be 'jit' or 'aot'.")


class _ModelWithLoss(nn.Module):
    """Wraps a model to include loss computation in the forward graph.

    GraphPPRunner._prepare_fwd_args adds targets as a forward input for
    the last stage, so the model graph must include loss computation.
    Delegates attribute access to the inner model so optimizer hooks
    and other code can access model submodules.
    """

    def __init__(self, model: nn.Module, loss_fn: Callable) -> None:
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def forward(self, x, targets):
        output = self.model(x)
        return self.loss_fn(output, targets)

    def init_weights(self, **kwargs):
        self.model.init_weights(**kwargs)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


class _GraphPPScheduleAdapter:
    """Adapter that wraps a GraphPPRunner to match the trainer's pp_schedule interface.

    The trainer calls ``self.pp_schedule.step(inputs, target=..., losses=..., ...)``
    in the PP branch of ``forward_backward_step``. This adapter delegates to
    ``GraphPPRunner.step()`` which manages stage states and runs the schedule.
    """

    def __init__(self, runner: Any) -> None:
        self.runner = runner

    def step(self, *args: Any, **kwargs: Any) -> None:
        self.runner.step(*args, **kwargs)


def graph_pp_pipeline_llm(
    model: nn.Module,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
    device: torch.device,
    model_config: BaseModel.Config,
    parallelize_fn: ParallelizeFunction,
    loss_fn: Callable,
) -> tuple[Any, list[nn.Module], bool, bool]:
    """Pipeline function for graph-based PP.

    Splits the model into per-stage chunks (like ``pipeline_llm``), applies
    SPMD parallelisms, then for each stage captures the joint graph with AOT
    and applies graph PP passes (split_fsdp_collectives, split_dI_dW).
    Creates ``GraphPipelineStage`` objects and a ``GraphPPRunner`` for
    execution.

    Returns the same 4-tuple as ``pipeline_llm`` so the trainer's PP path
    works unchanged.
    """
    import dataclasses

    from torchtitan.distributed.pipeline_parallel import (
        generate_llm_fqn_per_model_part,
        pipeline_module_split,
    )

    assert isinstance(compile_config, GraphTrainerCompileConfig)

    # Infer graph PP passes from parallelism config
    graph_pp_passes = []
    if parallel_dims.fsdp_enabled:
        graph_pp_passes.append("split_fsdp_collectives")

    pp_mesh = parallel_dims.get_mesh("pp")

    # Step 1: Split model into per-stage chunks (same as pipeline_llm)
    num_layers = model_config.n_layers
    input_weight = parallelism.pipeline_parallel_first_stage_less_layers
    output_weight = parallelism.pipeline_parallel_last_stage_less_layers
    from torch.distributed.pipelining.schedules import (
        get_schedule_class,
        ScheduleDualPipeV,
        ScheduleZBVZeroBubble,
    )

    schedule_class = get_schedule_class(parallelism.pipeline_parallel_schedule)
    is_v_schedule = schedule_class in (ScheduleDualPipeV, ScheduleZBVZeroBubble)
    stages_per_rank = 2 if is_v_schedule else 1
    num_virtual_stages = parallel_dims.pp * stages_per_rank

    # V-schedules (DualPipeV, ZBV) benefit from split_dI_dW to enable
    # zero-bubble scheduling where dW is interleaved with the next F.
    if is_v_schedule:
        graph_pp_passes.append("split_dI_dW")

    logger.info("Graph PP passes: %s", graph_pp_passes)

    module_names_per_stage = parallelism.module_fqns_per_model_part
    if module_names_per_stage is None:
        module_names_per_stage = generate_llm_fqn_per_model_part(
            num_virtual_stages, num_layers, input_weight, output_weight
        )

    stages, model_parts = pipeline_module_split(
        model,
        pp_mesh,
        parallelism.pipeline_parallel_schedule,
        device,
        module_names_per_stage,
    )

    # Step 2: Apply SPMD parallelisms to each part (without compile)
    no_compile_config = dataclasses.replace(compile_config, enable=False)
    for i, m in enumerate(model_parts):
        m = parallelize_fn(
            m,
            parallel_dims=parallel_dims,
            training=training,
            model_converters=model_converters,
            parallelism=parallelism,
            compile_config=no_compile_config,
            ac_config=ac_config,
            dump_folder=dump_folder,
        )
        model_parts[i] = m
        stages[i].submod = m

    # Step 3: Store the info needed for deferred graph capture.
    # The trainer will call to_empty() + init_weights() after this returns.
    # Graph capture happens lazily on first GraphPPRunner.step() call.
    microbatch_size = parallelism.pipeline_parallel_microbatch_size
    seq_len = training.seq_len

    # Build input/output args for each stage (for P2P buffer shapes)
    stage_input_args = []
    stage_output_args = []
    for stage in stages:
        if stage.is_first:
            inp = (
                torch.randint(
                    0,
                    model_config.vocab_size,
                    (microbatch_size, seq_len),
                    device="meta",
                ),
            )
        elif stage.is_last:
            # Last stage receives activations + targets (for _ModelWithLoss)
            inp = (
                torch.empty(
                    microbatch_size,
                    seq_len,
                    model_config.dim,
                    dtype=torch.bfloat16,
                    device="meta",
                ),
            )
        else:
            inp = (
                torch.empty(
                    microbatch_size,
                    seq_len,
                    model_config.dim,
                    dtype=torch.bfloat16,
                    device="meta",
                ),
            )
        stage_input_args.append(inp)

        if stage.is_last:
            # Model returns loss scalar (0-dimensional)
            out = (torch.tensor(0.0, dtype=torch.float32, device="meta"),)
        else:
            out = (
                torch.empty(
                    microbatch_size,
                    seq_len,
                    model_config.dim,
                    dtype=torch.bfloat16,
                    device="meta",
                ),
            )
        stage_output_args.append(out)

    has_first_stage = any(s.is_first for s in stages)
    has_last_stage = any(s.is_last for s in stages)

    # Create a lazy adapter that builds GraphPPRunner on first step()
    adapter = _LazyGraphPPAdapter(
        stages=stages,
        model_parts=model_parts,
        stage_input_args=stage_input_args,
        stage_output_args=stage_output_args,
        parallel_dims=parallel_dims,
        compile_config=compile_config,
        parallelism=parallelism,
        training=training,
        dump_folder=dump_folder,
        loss_fn=loss_fn,
        device=device,
        graph_pp_passes=graph_pp_passes,
    )

    return adapter, model_parts, has_first_stage, has_last_stage


class _LazyGraphPPAdapter:
    """Lazily builds GraphPPRunner on first step() call.

    Graph capture requires initialized model weights, which aren't
    available until after the trainer calls to_empty() + init_weights().
    """

    def __init__(
        self,
        stages,
        model_parts,
        stage_input_args,
        stage_output_args,
        parallel_dims,
        compile_config,
        parallelism,
        training,
        dump_folder,
        loss_fn,
        device,
        graph_pp_passes,
    ):
        self._stages = stages
        self._model_parts = model_parts
        self._stage_input_args = stage_input_args
        self._stage_output_args = stage_output_args
        self._parallel_dims = parallel_dims
        self._compile_config = compile_config
        self._parallelism = parallelism
        self._training = training
        self._dump_folder = dump_folder
        self._loss_fn = loss_fn
        self._device = device
        self._graph_pp_passes = graph_pp_passes
        self._runner = None

    def _build(self):
        from autoparallel.graph_passes.graph_pp_runner import (
            _get_stage_from_action,
            _run_reduce_grad_module,
            get_multiplexed_graph_callables,
            GraphPipelineStage,
            GraphPPRunner,
            overlap_fw_bw,
            stage_backward_input,
            stage_backward_weight,
            stage_forward,
            stage_full_backward,
            stage_reshard,
            stage_unshard,
        )
        from torch.distributed.pipelining.schedules import (
            BACKWARD_INPUT,
            BACKWARD_WEIGHT,
            FORWARD,
            FULL_BACKWARD,
            OVERLAP_F_B,
            REDUCE_GRAD,
            RESHARD,
            UNSHARD,
        )

        compile_config = self._compile_config
        graph_pp_passes = self._graph_pp_passes
        parallel_dims = self._parallel_dims

        register_blockmask_pytree_node()
        fsdp_reshard_after_forward = get_fsdp_reshard_after_forward_policy(
            self._parallelism.fsdp_reshard_after_forward,
            parallel_dims.pp_enabled,
        )
        joint_custom_passes = get_joint_custom_passes_from_config(
            parallel_dims, compile_config, fsdp_reshard_after_forward
        )
        compiler_passes = get_compiler_passes_from_config(
            self._model_parts[0], compile_config, parallel_dims
        )
        fw_compiler, bw_compiler = make_compiler_with_passes(
            compiler_passes, dump_folder=self._dump_folder
        )

        pp_mesh = parallel_dims.get_mesh("pp")
        num_stages = self._stages[0].num_stages
        graph_stages = []

        for i, (stage, m) in enumerate(
            zip(self._stages, self._model_parts, strict=True)
        ):
            # For the last stage, wrap the model with loss so the graph
            # includes loss computation and accepts targets as input.
            # This is needed because GraphPPRunner._prepare_fwd_args adds
            # targets to composite_args for the last stage.
            if stage.is_last and self._loss_fn is not None:
                m = _ModelWithLoss(m, self._loss_fn)
                self._model_parts[i] = m

            # Create example inputs on device for graph capture
            microbatch_size = self._parallelism.pipeline_parallel_microbatch_size
            seq_len = self._training.seq_len
            if stage.is_first:
                example_args = (
                    torch.randint(
                        0,
                        seq_len,  # safe vocab upper bound
                        (microbatch_size, seq_len),
                        device=self._device,
                    ),
                )
            elif stage.is_last and self._loss_fn is not None:
                dim = m.model.config.dim if hasattr(m, "model") else 256
                example_args = (
                    torch.randn(
                        microbatch_size,
                        seq_len,
                        dim,
                        device=self._device,
                        dtype=torch.bfloat16,
                        requires_grad=True,
                    ),
                    # Target tokens for loss computation
                    torch.randint(
                        0,
                        seq_len,
                        (microbatch_size, seq_len),
                        device=self._device,
                    ),
                )
            else:
                dim = m.config.dim if hasattr(m, "config") else 256
                example_args = (
                    torch.randn(
                        microbatch_size,
                        seq_len,
                        dim,
                        device=self._device,
                        dtype=torch.bfloat16,
                        requires_grad=True,
                    ),
                )

            dt_args, dt_kwargs = parallelize_inputs(parallel_dims, example_args, {})

            graph_callables, graph_meta = pp_joint_graph_builder(
                m,
                dt_args,
                dt_kwargs,
                fw_compiler=fw_compiler,
                bw_compiler=bw_compiler,
                joint_custom_passes=list(joint_custom_passes),
                dump_folder=self._dump_folder,
                compile_config=compile_config,
                graph_pp_passes=graph_pp_passes,
                parallel_dims=parallel_dims,
            )

            graph_stage = GraphPipelineStage(
                submodule=m,
                graph_callables=graph_callables,
                graph_meta=graph_meta,
                stage_index=stage.stage_index,
                num_stages=num_stages,
                device=self._device,
                input_args=self._stage_input_args[i],
                output_args=self._stage_output_args[i],
                group=pp_mesh.get_group(),
            )
            graph_stages.append(graph_stage)

        from torch.distributed.pipelining.schedules import (
            _PipelineScheduleRuntime,
            get_schedule_class,
            PipelineScheduleMulti,
        )

        microbatch_size = self._parallelism.pipeline_parallel_microbatch_size
        n_microbatches = self._training.local_batch_size // microbatch_size
        schedule_class = get_schedule_class(
            self._parallelism.pipeline_parallel_schedule
        )
        is_looped = issubclass(schedule_class, PipelineScheduleMulti)
        if not is_looped:
            # Graph PP needs register_custom_function which requires a
            # _PipelineScheduleRuntime subclass. Single-stage schedules
            # don't support this, so we use Interleaved1F1B instead.
            schedule_class = get_schedule_class("Interleaved1F1B")
            is_looped = True
        # loss_fn=None because loss is embedded in the last stage's model
        # (via _ModelWithLoss wrapper). The schedule doesn't need to
        # compute loss separately.
        schedule = schedule_class(
            graph_stages if is_looped else graph_stages[0],
            n_microbatches=n_microbatches,
            loss_fn=None,
            scale_grads=False,
            backward_requires_autograd=False,
        )
        assert isinstance(schedule, _PipelineScheduleRuntime)

        # Check if the schedule generates UNSHARD/REDUCE_GRAD actions.
        # If not (e.g., Interleaved1F1B), we need to inline unshard
        # before each forward and reduce_grad after each backward.
        has_fsdp_actions = any(
            a is not None and a.computation_type in (UNSHARD, REDUCE_GRAD)
            for actions in schedule.pipeline_order.values()
            for a in actions
        )

        if has_fsdp_actions:
            # Schedule handles UNSHARD/REDUCE_GRAD explicitly
            schedule.register_custom_function(FORWARD, stage_forward)
            schedule.register_custom_function(FULL_BACKWARD, stage_full_backward)
            schedule.register_custom_function(UNSHARD, stage_unshard)
            schedule.register_custom_function(RESHARD, stage_reshard)
        else:
            # Schedule doesn't generate UNSHARD/REDUCE_GRAD actions
            # (e.g. Interleaved1F1B). Run unshard once before the first
            # forward of each step, keep params unsharded for all
            # microbatches. Reduce_grad + reshard run after all backwards
            # complete, via a monkey-patched _accumulate_stage_sharded_grads.
            _unshard_done: dict[int, bool] = {}

            def _forward_with_unshard(action, ctx):
                _, _, stage = _get_stage_from_action(action, ctx)
                if stage.stage_index not in _unshard_done:
                    stage_unshard(action, ctx)
                    _unshard_done[stage.stage_index] = True
                stage_forward(action, ctx)

            schedule.register_custom_function(FORWARD, _forward_with_unshard)
            schedule.register_custom_function(FULL_BACKWARD, stage_full_backward)

            # Override _accumulate_stage_sharded_grads to run reduce_grad
            # after all microbatches' backwards complete, then reshard.
            runner_ref = [None]  # Will be set after GraphPPRunner is created

            def _accumulate_with_reduce_grad(stage):
                # Fix None grads
                device = self._device
                fixed_grads = []
                for i, g in enumerate(stage.state["unsharded_grads"]):
                    if g is None:
                        rg = stage.graph_callables.reduce_grad
                        if rg is not None:
                            ph = rg.graph.find_nodes(op="placeholder")[i]
                            val = ph.meta.get("val")
                            if val is not None:
                                fixed_grads.append(torch.zeros_like(val, device=device))
                                continue
                        fixed_grads.append(g)
                    else:
                        fixed_grads.append(g)
                stage.state["unsharded_grads"] = fixed_grads

                # Run reduce_grad to get sharded grads
                if stage.graph_callables.reduce_grad is not None:
                    sharded_grads = _run_reduce_grad_module(
                        stage.graph_callables.reduce_grad,
                        stage.graph_meta,
                        stage.state["unsharded_grads"],
                        inductor=stage.inductor,
                    )
                    stage.state["sharded_grads"] = sharded_grads
                else:
                    stage.state["sharded_grads"] = stage.state["unsharded_grads"]

                # Now accumulate onto param.grad
                runner_ref[0]._orig_accumulate(stage)

                # Clear unshard state for next step
                _unshard_done.pop(stage.stage_index, None)

        # Always register these for schedules that use them
        def _patched_stage_reduce_grad(action, ctx):
            _, _, stage = _get_stage_from_action(action, ctx)
            if stage.graph_callables.reduce_grad is None:
                stage.state["sharded_grads"] = stage.state["unsharded_grads"]
            else:
                device = self._device
                fixed_grads = []
                for i, g in enumerate(stage.state["unsharded_grads"]):
                    if g is None:
                        rg = stage.graph_callables.reduce_grad
                        ph = rg.graph.find_nodes(op="placeholder")[i]
                        val = ph.meta.get("val")
                        if val is not None:
                            fixed_grads.append(torch.zeros_like(val, device=device))
                        else:
                            fixed_grads.append(torch.zeros(1, device=device))
                    else:
                        fixed_grads.append(g)
                sharded_grads = _run_reduce_grad_module(
                    stage.graph_callables.reduce_grad,
                    stage.graph_meta,
                    fixed_grads,
                    inductor=stage.inductor,
                )
                stage.state["sharded_grads"] = sharded_grads

        schedule.register_custom_function(REDUCE_GRAD, _patched_stage_reduce_grad)

        # DualPipeV emits BACKWARD_INPUT/BACKWARD_WEIGHT actions in its
        # zero-bubble tail, but without split_dI_dW graph pass, stages
        # don't have separate bw_dI/bw_dW graphs. Wrap the handlers to
        # fall back to stage_full_backward when split graphs are missing.
        def _backward_input_with_fallback(action, ctx):
            _, _, bw_stage = _get_stage_from_action(action, ctx)
            if bw_stage.graph_callables.bw_dI is None:
                from torch.distributed.pipelining.schedules import _Action

                new_action = _Action(
                    action.stage_index,
                    FULL_BACKWARD,
                    action.microbatch_index,
                    action.sub_actions,
                )
                stage_full_backward(new_action, ctx)
                return
            stage_backward_input(action, ctx)

        def _backward_weight_with_fallback(action, ctx):
            _, _, bw_stage = _get_stage_from_action(action, ctx)
            if bw_stage.graph_callables.bw_dW is None:
                # Full backward already ran during BACKWARD_INPUT, skip.
                return
            stage_backward_weight(action, ctx)

        schedule.register_custom_function(BACKWARD_INPUT, _backward_input_with_fallback)
        schedule.register_custom_function(
            BACKWARD_WEIGHT, _backward_weight_with_fallback
        )

        use_inductor = not any(
            "_local_scalar_dense" in str(n)
            for stage in graph_stages
            for gm in [stage.graph_callables.fw, stage.graph_callables.full_bw]
            if gm is not None
            for n in gm.graph.nodes
        )
        if not use_inductor:
            logger.warning(
                "Detected _local_scalar_dense in PP graphs (likely from EP), "
                "falling back to interpreter execution"
            )

        # DualPipeV wraps FORWARD+FULL_BACKWARD pairs in OVERLAP_F_B actions.
        # Use multiplexed graphs that fuse F+B into a single graph with
        # interleaved compute/comm for true overlap when possible.
        has_overlap_actions = any(
            a is not None and a.computation_type == OVERLAP_F_B
            for actions in schedule.pipeline_order.values()
            for a in actions
        )
        if has_overlap_actions:
            from functools import partial

            from autoparallel.graph_passes.graph_multiplex import (
                multiplex_fw_bw_graph,
            )

            stage_graphs = {
                gs.stage_index: gs.graph_callables for gs in graph_stages
            }
            multiplexed_graph_callables = get_multiplexed_graph_callables(
                stage_graphs,
                partial(multiplex_fw_bw_graph, overlap_with_annotations=True),
            )
            schedule.register_custom_function(
                OVERLAP_F_B,
                partial(
                    overlap_fw_bw,
                    multiplexed_graph_callables,
                ),
            )
        else:
            # No OVERLAP_F_B in schedule — register sequential fallback
            def _overlap_f_b_sequential(action, ctx):
                assert (
                    action.sub_actions is not None
                ), "OVERLAP_F_B requires sub_actions"
                for sub_a in action.sub_actions:
                    custom_fn = schedule._comp_type_to_function_map.get(
                        sub_a.computation_type
                    )
                    if custom_fn is not None:
                        custom_fn(sub_a, ctx)
                    else:
                        raise ValueError(
                            f"No custom function registered for "
                            f"{sub_a.computation_type} inside OVERLAP_F_B"
                        )

            schedule.register_custom_function(OVERLAP_F_B, _overlap_f_b_sequential)
        self._runner = GraphPPRunner(schedule, inductor=use_inductor)

        # Wire the monkey-patch for schedules without UNSHARD/REDUCE_GRAD
        if not has_fsdp_actions:
            self._runner._orig_accumulate = self._runner._accumulate_stage_sharded_grads
            self._runner._accumulate_stage_sharded_grads = _accumulate_with_reduce_grad
            runner_ref[0] = self._runner

        logger.info(
            "Built GraphPPRunner with %d stages (inductor=%s, inline_fsdp=%s)",
            len(graph_stages),
            use_inductor,
            not has_fsdp_actions,
        )

    def step(self, *args, **kwargs):
        if self._runner is None:
            self._build()
        self._runner.step(*args, **kwargs)

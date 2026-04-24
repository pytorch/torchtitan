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

Additionally supports pre-compile via --compile.precompile_artifact_dir:
- When set during training, loads a precompiled artifact and skips compilation
  entirely
- Generate artifacts with precompile_main.py
"""

import dataclasses
import functools
import warnings

import torch
import torch.nn as nn

from torchtitan.config import ParallelismConfig
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
)
from torchtitan.experiments.graph_trainer.jit_backend import (
    get_compile_backend_with_passes,
)
from torchtitan.experiments.graph_trainer.precompile import (
    _ARTIFACT_KEY,
    ConfigFingerprint,
)
from torchtitan.experiments.graph_trainer.storage import (
    DiskStorageAdapter,
    StorageAdapter,
)
from torchtitan.tools.logging import logger

# Compiler passes whose output supports serialization for precompile.
# full_inductor_compilation produces OutputCode via compile_fx_inner;
# regional_inductor produces RegionalOutputCode (an OutputCode subclass).
_SERIALIZABLE_PASSES: frozenset[str] = frozenset(
    ("full_inductor_compilation", "regional_inductor")
)


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
    model.compile(
        backend=backend,
        fullgraph=True,
    )
    logger.info("Applied JIT compilation (torch.compile) to the model")
    return model


def _make_precompile_callback(
    model: nn.Module,
    compile_config: GraphTrainerCompileConfig,
    parallel_dims: ParallelDims,
    storage: StorageAdapter | None = None,
    config_fingerprint: ConfigFingerprint | None = None,
):
    """Build the on_compile callback that saves the compiled artifact to disk."""
    from .precompile import compute_config_fingerprint, precompile_save

    if storage is None:
        storage = DiskStorageAdapter(compile_config.precompile_artifact_dir)
    if config_fingerprint is None:
        config_fingerprint = compute_config_fingerprint(
            model, compile_config, parallel_dims
        )

    def on_compile(compiled_fn, out_spec):
        precompile_save(
            model,
            compiled_fn,
            storage,
            out_spec=out_spec,
            metadata={
                "world_size": torch.distributed.get_world_size(),
            },
            config_fingerprint=config_fingerprint,
        )

    return on_compile


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

    # When loading a precompiled artifact, compute storage/fingerprint
    # once before checking for existence and deserializing the artifact.
    if compile_config.precompile_artifact_dir:
        from .precompile import compute_config_fingerprint

        storage = DiskStorageAdapter(compile_config.precompile_artifact_dir)
        config_fingerprint = compute_config_fingerprint(
            model, compile_config, parallel_dims
        )

        if not storage.exists(_ARTIFACT_KEY):
            raise ValueError(
                f"Precompiled artifact not found at "
                f"'{compile_config.precompile_artifact_dir}/{_ARTIFACT_KEY}'. "
                f"Run precompile_main first to generate the artifact."
            )

        return _apply_aot_compile_load(
            model, parallel_dims, storage, config_fingerprint
        )

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


def _apply_aot_compile_load(
    model: nn.Module,
    parallel_dims: ParallelDims,
    storage: StorageAdapter,
    config_fingerprint: ConfigFingerprint,
) -> CompiledModule:
    """Load a precompiled artifact and wrap the model with it."""
    from .precompile import precompile_load

    # BlockMask must be registered as a pytree node before unpickling
    # the artifact, which may contain BlockMask objects in its specs.
    register_blockmask_pytree_node()

    precompiled_fn = precompile_load(
        model,
        storage,
        expected_fingerprint=config_fingerprint,
    )

    def _unused_graph_builder(*args, **kwargs):
        raise RuntimeError(
            "joint_graph_builder should not be called when using a precompiled artifact"
        )

    compiled_model = CompiledModule(
        model,
        parallel_dims,
        joint_graph_builder=_unused_graph_builder,
        parallelize_inputs=parallelize_inputs,
        precompiled_fn=precompiled_fn,
    )
    logger.info("Applied precompiled artifact (precompile load) to the model")
    return compiled_model


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

    if mode in ("aot", "jit"):
        warnings.warn(
            f"compile.mode='{mode}' is deprecated and will be removed in the "
            "future. Please use --compile.mode='aot_fx_trace' instead.",
            FutureWarning,
            stacklevel=2,
        )

    torch._inductor.config.reorder_for_peak_memory = False
    torch._dynamo.config.capture_scalar_outputs = True

    fsdp_reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        parallelism.fsdp_reshard_after_forward, parallel_dims.pp_enabled
    )

    if compile_config.precompile_artifact_dir and mode not in ("aot", "aot_fx_trace"):
        logger.warning(
            "--compile.precompile_artifact_dir is only supported with "
            f"--compile.mode=aot or aot_fx_trace, but mode is '{mode}'. "
            "Ignoring precompile."
        )
        compile_config = dataclasses.replace(compile_config, precompile_artifact_dir="")

    if (
        compile_config.precompile_artifact_dir
        and mode == "aot"
        and not (_SERIALIZABLE_PASSES & set(compile_config.passes))
    ):
        raise ValueError(
            "--compile.precompile_artifact_dir requires at least one pass that "
            "produces serializable output "
            f"({', '.join(sorted(_SERIALIZABLE_PASSES))}) in --compile.passes."
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
    elif mode == "aot_fx_trace":
        # aot_fx_trace traces fwd+loss+bwd together inside forward_backward_step,
        # so no model-level wrapping is needed here. If precompile_artifact_dir
        # is set, the precompiled artifact will be loaded lazily in
        # GraphTrainer._make_fx_forward_backward_step.
        if compile_config.precompile_artifact_dir:
            logger.info(
                "aot_fx_trace compile mode: precompiled artifact will be loaded "
                f"from {compile_config.precompile_artifact_dir}"
            )
        else:
            logger.info(
                "aot_fx_trace compile mode: graph capture will happen at training time"
            )
        return model
    else:
        raise ValueError(
            f"Unknown compile mode: {mode}. Must be 'jit', 'aot', or 'aot_fx_trace'."
        )


def graph_pp_pipeline_llm(
    model: nn.Module,
    *,
    parallel_dims,
    training,
    model_converters,
    parallelism,
    compile_config,
    ac_config,
    dump_folder: str,
    device: torch.device,
    model_config,
    parallelize_fn,
    loss_fn,
):
    """Unified graph PP pipeline for both manual SPMD and autoparallel.

    Splits the model into per-stage chunks, applies parallelisms via
    parallelize_fn (which may be manual SPMD or AutoParallelGraph), exports
    joint graphs for each stage, and creates a GraphPPRunner.

    Returns the same 4-tuple as pipeline_llm:
        (runner, model_parts, has_first_stage, has_last_stage)
    """
    from torch.distributed.pipelining.schedules import (
        get_schedule_class,
        ScheduleDualPipeV,
        ScheduleZBVZeroBubble,
    )

    from torchtitan.distributed.pipeline_parallel import (
        build_pipeline_schedule,
        generate_llm_fqn_per_model_part,
        get_pipeline_metadata,
        pipeline_module_split,
    )
    from torchtitan.experiments.graph_trainer.graph_pp import (
        GraphPipelineStage,
        GraphPPRunner,
        ModelWithLoss,
    )
    from torchtitan.experiments.graph_trainer.graph_pp.common import (
        get_shape_inference_fns,
    )
    from torchtitan.experiments.graph_trainer.graph_utils import export_joint_for_pp

    pp_mesh = parallel_dims.get_mesh("pp")

    num_virtual_stages, num_layers, input_weight, output_weight = (
        get_pipeline_metadata(parallel_dims, parallelism, model_config)
    )
    module_names_per_stage = parallelism.module_fqns_per_model_part
    if module_names_per_stage is None:
        module_names_per_stage = generate_llm_fqn_per_model_part(
            num_virtual_stages, num_layers, input_weight, output_weight,
        )

    stages, model_parts = pipeline_module_split(
        model,
        pp_mesh,
        parallelism.pipeline_parallel_schedule,
        device,
        module_names_per_stage,
    )

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

    # Determine graph PP passes
    graph_pp_passes = []
    if parallel_dims.fsdp_enabled:
        graph_pp_passes.append("split_fsdp_collectives")
    schedule_class = get_schedule_class(parallelism.pipeline_parallel_schedule)
    if schedule_class in (ScheduleDualPipeV, ScheduleZBVZeroBubble):
        graph_pp_passes.append("split_dI_dW")

    # Build compilers + joint custom passes
    fsdp_reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        parallelism.fsdp_reshard_after_forward, parallel_dims.pp_enabled
    )
    joint_custom_passes = get_joint_custom_passes_from_config(
        parallel_dims, compile_config, fsdp_reshard_after_forward
    )
    compiler_passes = get_compiler_passes_from_config(
        model_parts[0], compile_config, parallel_dims
    )
    fw_compiler, bw_compiler = make_compiler_with_passes(
        compiler_passes, dump_folder=dump_folder
    )

    num_stages = len(module_names_per_stage)
    (
        shape_fn_first_stage,
        shape_fn_intermediate,
        shape_fn_last_stage_output,
    ) = get_shape_inference_fns(model_config, training, parallelism, has_loss=True)

    register_blockmask_pytree_node()

    microbatch_size = parallelism.pipeline_parallel_microbatch_size
    dp_degree = parallel_dims.dp_replicate * parallel_dims.dp_shard
    spmd_batch_size = microbatch_size * dp_degree

    graph_stages = []
    for i, (stage, m) in enumerate(zip(stages, model_parts)):
        is_last = stage.is_last

        if is_last and loss_fn is not None:
            m = ModelWithLoss(m, loss_fn)
            model_parts[i] = m

        # Get joint graph: either pre-built (autoparallel) or trace now (manual SPMD)
        jwd = getattr(m, "_joint_with_descriptors", None)
        if jwd is None:
            if stage.is_first:
                example_args = (
                    torch.randint(
                        0,
                        model_config.vocab_size,
                        (spmd_batch_size, training.seq_len),
                        device=device,
                    ),
                )
            elif is_last:
                example_args = (
                    torch.randn(
                        spmd_batch_size,
                        training.seq_len,
                        model_config.dim,
                        device=device,
                        dtype=torch.bfloat16,
                        requires_grad=True,
                    ),
                    torch.randint(
                        0,
                        model_config.vocab_size,
                        (spmd_batch_size, training.seq_len),
                        device=device,
                    ),
                )
            else:
                example_args = (
                    torch.randn(
                        spmd_batch_size,
                        training.seq_len,
                        model_config.dim,
                        device=device,
                        dtype=torch.bfloat16,
                        requires_grad=True,
                    ),
                )

            dt_args, dt_kwargs = parallelize_inputs(parallel_dims, example_args, {})
            jwd = export_joint_for_pp(
                m, dt_args, dt_kwargs,
                joint_custom_passes=list(joint_custom_passes),
                compile_config=compile_config,
            )

        graph_stage = GraphPipelineStage(
            submodule=m,
            graph_callables=None,
            graph_meta=None,
            stage_index=stage.stage_index,
            num_stages=num_stages,
            device=device,
            input_args=(
                shape_fn_first_stage()
                if stage.is_first
                else shape_fn_intermediate()
            ),
            output_args=(
                shape_fn_last_stage_output()
                if is_last
                else shape_fn_intermediate()
            ),
            group=pp_mesh.get_group("pp"),
        )
        graph_stage.joint_graph = jwd
        graph_stages.append(graph_stage)

        logger.info(
            "PP stage_idx %d: joint graph ready for %s",
            stage.stage_index,
            module_names_per_stage[stage.stage_index],
        )

    pp_schedule = build_pipeline_schedule(
        parallelism=parallelism,
        local_batch_size=training.local_batch_size,
        stages=graph_stages,
        loss_fn=None,
        backward_requires_autograd=False,
        scale_grads=False,
    )

    runner = GraphPPRunner(pp_schedule)

    has_first_stage = any(s.is_first for s in graph_stages)
    has_last_stage = any(s.is_last for s in graph_stages)

    return runner, model_parts, has_first_stage, has_last_stage

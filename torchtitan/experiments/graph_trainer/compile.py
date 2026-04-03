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

    torch._inductor.config.reorder_for_peak_memory = False
    torch._dynamo.config.capture_scalar_outputs = True

    fsdp_reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        parallelism.fsdp_reshard_after_forward, parallel_dims.pp_enabled
    )

    if compile_config.precompile_artifact_dir and mode != "aot":
        logger.warning(
            "--compile.precompile_artifact_dir is only supported with "
            f"--compile.mode=aot, but mode is '{mode}'. Ignoring precompile."
        )
        compile_config = dataclasses.replace(compile_config, precompile_artifact_dir="")

    if compile_config.precompile_artifact_dir and not (
        _SERIALIZABLE_PASSES & set(compile_config.passes)
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
        # so no model-level wrapping is needed here.
        logger.info(
            "aot_fx_trace compile mode: graph capture will happen at training time"
        )
        return model
    else:
        raise ValueError(
            f"Unknown compile mode: {mode}. Must be 'jit', 'aot', or 'aot_fx_trace'."
        )

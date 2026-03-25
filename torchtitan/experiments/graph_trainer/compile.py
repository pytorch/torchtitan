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

Additionally supports pre-compile via --compile.precompile:
- On first run: compiles with serializable=True, saves the artifact, continues training
- On subsequent runs: detects existing artifact, loads it, skips compilation
"""

import dataclasses
import functools
import pickle

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


def _get_precompile_storage_and_key(
    compile_config: GraphTrainerCompileConfig,
) -> tuple[StorageAdapter, str]:
    storage = DiskStorageAdapter(compile_config.precompile_artifact_dir)
    rank = torch.distributed.get_rank()
    # Per-rank artifact keys will go away once the Compile on one Rank
    # (CooR) project lands — at that point we will precompile once and
    # reuse that artifact across all ranks.
    artifact_key = f"default_rank{rank}"
    return storage, artifact_key


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


def _make_precompile_callback(
    model: nn.Module,
    compile_config: GraphTrainerCompileConfig,
    parallel_dims: ParallelDims,
    storage: StorageAdapter | None = None,
    artifact_key: str | None = None,
):
    """Build the on_compile callback that saves the compiled artifact to disk."""
    from .precompile import compute_config_fingerprint, precompile_save

    if storage is None or artifact_key is None:
        storage, artifact_key = _get_precompile_storage_and_key(compile_config)
    config_fingerprint = compute_config_fingerprint(
        model, compile_config, parallel_dims
    )

    def on_compile(compiled_fn, out_spec):
        precompile_save(
            model,
            compiled_fn,
            storage,
            artifact_key,
            out_spec=out_spec,
            metadata={
                "world_size": torch.distributed.get_world_size(),
                "rank": torch.distributed.get_rank(),
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

    # When precompile is enabled, compute storage/key once and reuse them
    # for both the load attempt and the save callback to avoid duplicate
    # DiskStorageAdapter construction and fingerprint computation.
    storage: StorageAdapter | None = None
    artifact_key: str | None = None
    if compile_config.precompile:
        storage, artifact_key = _get_precompile_storage_and_key(compile_config)

        if storage.exists(artifact_key):
            from .precompile import compute_config_fingerprint

            config_fingerprint = compute_config_fingerprint(
                model, compile_config, parallel_dims
            )
            try:
                return _apply_aot_compile_load(
                    model, parallel_dims, storage, artifact_key, config_fingerprint
                )
            except (ValueError, pickle.UnpicklingError, RuntimeError) as e:
                # ValueError: fingerprint/param/buffer mismatches from our
                # validation. pickle.UnpicklingError: corrupted or
                # incompatible serialized data. RuntimeError: intentionally
                # broad to catch remaining deserialization failures (e.g.
                # shape mismatches in torch.load) that surface as
                # RuntimeError. We log the exception type so unrelated
                # errors (CUDA OOM, NCCL) are distinguishable in logs.
                logger.warning(
                    f"Stale precompile artifact detected ({type(e).__name__}), "
                    f"recompiling: {e}"
                )
                storage.delete(artifact_key)

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

    serializable = compile_config.precompile
    on_compile = (
        _make_precompile_callback(
            model,
            compile_config,
            parallel_dims,
            storage=storage,
            artifact_key=artifact_key,
        )
        if serializable
        else None
    )

    # Create custom joint_graph_builder with compilers
    model_joint_graph_builder = functools.partial(
        joint_graph_builder,
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        joint_custom_passes=joint_custom_passes,
        dump_folder=dump_folder,
        compile_config=compile_config,
        serializable=serializable,
        on_compile=on_compile,
    )

    model = CompiledModule(
        model, parallel_dims, model_joint_graph_builder, parallelize_inputs
    )
    msg = "Applied AOT compilation (joint graph export) to the model"
    if serializable:
        msg += " with serializable=True (precompile save)"
    logger.info(msg)
    return model


def _apply_aot_compile_load(
    model: nn.Module,
    parallel_dims: ParallelDims,
    storage: StorageAdapter,
    artifact_key: str,
    config_fingerprint: str,
) -> CompiledModule:
    """Load a precompiled artifact and wrap the model with it."""
    from .precompile import precompile_load

    # BlockMask must be registered as a pytree node before unpickling
    # the artifact, which may contain BlockMask objects in its specs.
    register_blockmask_pytree_node()

    precompiled_fn = precompile_load(
        model, storage, artifact_key, expected_fingerprint=config_fingerprint
    )

    def _unused_graph_builder(*args, **kwargs):
        raise RuntimeError(
            "joint_graph_builder should not be called when "
            "using a precompiled artifact"
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

    if compile_config.precompile and mode != "aot":
        logger.warning(
            "--compile.precompile is only supported with --compile.mode=aot, "
            f"but mode is '{mode}'. Ignoring precompile."
        )
        compile_config = dataclasses.replace(compile_config, precompile=False)

    if compile_config.precompile and not (
        _SERIALIZABLE_PASSES & set(compile_config.passes)
    ):
        raise ValueError(
            "--compile.precompile requires at least one serializable pass "
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

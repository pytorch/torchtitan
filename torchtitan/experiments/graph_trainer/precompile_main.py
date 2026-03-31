# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Single-process precompile entry point for graph_trainer.

Uses compile-on-one-rank (CooR) to generate a rank-agnostic compiled
artifact from a single process, which can then be loaded by all ranks
during torchrun training. This avoids the need to run torchrun with N
GPUs just for precompilation.

Usage:
    python -m torchtitan.experiments.graph_trainer.precompile_main \
        --module graph_trainer.llama3 \
        --config graph_trainer_llama3_debugmodel \
        --compile.passes full_inductor_compilation \
        --compile.joint_passes inductor_decomposition \
        --compile.precompile_artifact_dir /tmp/precompile_artifacts
"""

import dataclasses
import functools

import torch
import torch.distributed as dist

from torchtitan.config import ConfigManager, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.common_utils import (
    apply_graph_ac,
    parallelize_inputs,
    register_blockmask_pytree_node,
)
from torchtitan.experiments.graph_trainer.compile import (
    _make_precompile_callback,
    _SERIALIZABLE_PASSES,
)
from torchtitan.experiments.graph_trainer.graph_utils import (
    CompiledModule,
    get_compiler_passes_from_config,
    get_joint_custom_passes_from_config,
    joint_graph_builder,
    make_compiler_with_passes,
)
from torchtitan.experiments.graph_trainer.precompile import _ARTIFACT_KEY
from torchtitan.experiments.graph_trainer.storage import DiskStorageAdapter
from torchtitan.tools import utils
from torchtitan.tools.logging import logger


def main():
    config_manager = ConfigManager()
    config = config_manager.parse_args()

    compile_config = config.compile

    if not compile_config.precompile_artifact_dir:
        raise ValueError(
            "precompile_main requires --compile.precompile_artifact_dir to be set."
        )
    # Only one pass in the pipeline needs to produce serializable OutputCode.
    # Earlier graph passes like bucketing can still run as long as the final
    # lowering pass is one of the serializable passes below.
    if not (_SERIALIZABLE_PASSES & set(compile_config.passes)):
        raise ValueError(
            "precompile_main requires at least one pass that produces "
            "serializable output "
            f"({', '.join(sorted(_SERIALIZABLE_PASSES))}) in --compile.passes."
        )

    parallelism = config.parallelism
    dp_replicate = parallelism.data_parallel_replicate_degree
    dp_shard = parallelism.data_parallel_shard_degree
    cp = parallelism.context_parallel_degree
    tp = parallelism.tensor_parallel_degree
    pp = parallelism.pipeline_parallel_degree

    # dp_shard=-1 means "use remaining ranks" which can't be inferred
    # in single-process mode. The compiled graph bakes in tensor shapes
    # that depend on dp_shard, so the exact value must match training.
    if dp_shard < 0:
        raise ValueError(
            "precompile_main requires an explicit "
            "--parallelism.data_parallel_shard_degree (not -1). "
            "Set it to the value you will use during torchrun training."
        )
    world_size = dp_replicate * dp_shard * cp * tp * pp

    logger.info(f"Initializing single-process precompile with world_size={world_size}")

    # rank must be 0 because --virtual-local-rank maps every torchrun rank
    # to local rank 0, so the precompiled artifact needs to match that setup.
    # Fake backend produces correct collective output shapes without real
    # communication, letting us trace distributed ops on a single process.
    dist.init_process_group("fake", rank=0, world_size=world_size)

    # CooR must be enabled globally (not just during tracing) so that the
    # parallelization phase (TP, FSDP mesh setup) also uses symbolic
    # coordinates rather than hardcoding rank-specific values.
    import torch.distributed.config as dist_config

    dist_config.compile_on_one_rank = True

    # Match the deterministic mode that the training loop will use.
    # The backward graph captures use_deterministic_algorithms() at
    # compile time and asserts it matches at runtime.
    if config.debug.deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    parallel_dims = ParallelDims(
        dp_shard=dp_shard,
        dp_replicate=dp_replicate,
        cp=cp,
        tp=tp,
        pp=pp,
        ep=parallelism.expert_parallel_degree,
        etp=parallelism.expert_tensor_parallel_degree,
        world_size=world_size,
    )
    parallel_dims.build_mesh()

    model_spec = config.model_spec
    if model_spec is None:
        raise ValueError(
            "model_spec must be set. Pass --module to specify the model "
            "(e.g. --module graph_trainer.llama3)."
        )

    # TODO: Factor the model setup below with the training path so precompile
    # and training share a single implementation of build/parallelize/init.
    model_config = model_spec.model
    model_config.update_from_config(trainer_config=config)

    logger.info(f"Building {model_spec.name} {model_spec.flavor} on meta device")
    with (
        torch.device("meta"),
        utils.set_default_dtype(TORCH_DTYPE_MAP[config.training.dtype]),
    ):
        model = model_config.build()

    model.verify_module_protocol()

    no_compile_config = dataclasses.replace(compile_config, enable=False)
    model = model_spec.parallelize_fn(
        model,
        parallel_dims=parallel_dims,
        training=config.training,
        model_converters=config.model_converters,
        parallelism=parallelism,
        compile_config=no_compile_config,
        ac_config=config.activation_checkpoint,
        dump_folder=config.dump_folder,
    )

    # CooR must be disabled during init_weights because DTensor RNG ops
    # (weight initialization seeding) raise NotImplementedError under
    # compile_on_one_rank=True. Re-enable for the tracing phase after.
    device_type = utils.device_type
    model.to_empty(device=device_type)
    dist_config.compile_on_one_rank = False
    try:
        with torch.no_grad():
            model.init_weights(buffer_device=None)
    finally:
        dist_config.compile_on_one_rank = True
    model.train()

    logger.info("Model parallelized and materialized, starting AOT compile")

    # Augment compile_config with AC joint passes to match the training
    # path, which calls apply_graph_ac during parallelization. Without
    # this the SAC pass won't run and the config fingerprint will differ.
    if config.activation_checkpoint.mode != "none":
        apply_graph_ac(compile_config, config.activation_checkpoint)

    register_blockmask_pytree_node()

    from torchtitan.distributed.fsdp import get_fsdp_reshard_after_forward_policy

    fsdp_reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        parallelism.fsdp_reshard_after_forward, parallel_dims.pp_enabled
    )

    joint_custom_passes = get_joint_custom_passes_from_config(
        parallel_dims, compile_config, fsdp_reshard_after_forward
    )
    compiler_passes = get_compiler_passes_from_config(
        model, compile_config, parallel_dims
    )
    fw_compiler, bw_compiler = make_compiler_with_passes(
        compiler_passes, dump_folder=config.dump_folder
    )

    from .precompile import compute_config_fingerprint

    storage = DiskStorageAdapter(compile_config.precompile_artifact_dir)
    config_fingerprint = compute_config_fingerprint(
        model, compile_config, parallel_dims
    )

    on_compile = _make_precompile_callback(
        model,
        compile_config,
        parallel_dims,
        storage=storage,
        config_fingerprint=config_fingerprint,
    )

    model_joint_graph_builder = functools.partial(
        joint_graph_builder,
        fw_compiler=fw_compiler,
        bw_compiler=bw_compiler,
        joint_custom_passes=joint_custom_passes,
        dump_folder=config.dump_folder,
        compile_config=compile_config,
        serializable=True,
        on_compile=on_compile,
    )

    compiled_model = CompiledModule(
        model, parallel_dims, model_joint_graph_builder, parallelize_inputs
    )

    # Forward pass triggers AOT compilation; the backward graph is compiled
    # eagerly (not lazily) because serializable=True sets
    # force_non_lazy_backward_lowering=True in aot_compile_joint.
    seq_len = config.training.seq_len
    local_batch_size = config.training.local_batch_size
    vocab_size = model_config.vocab_size

    dummy_input = torch.randint(
        0, vocab_size, (local_batch_size, seq_len), device=device
    )
    logger.info("Running forward pass to trigger AOT compilation...")
    compiled_model(dummy_input)

    logger.info(
        f"Precompile complete. Artifact saved to "
        f"{compile_config.precompile_artifact_dir}/{_ARTIFACT_KEY}.bin"
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    main()

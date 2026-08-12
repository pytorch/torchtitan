# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.experiments.graph_trainer.common_utils import (
    annotate_module_fqns,
    apply_simple_fsdp,
)
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.configs import GraphTrainerCompileConfig

from .model import GraphTrainerMuseGlimmerModel


def parallelize_muse_glimmer(
    model: GraphTrainerMuseGlimmerModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: GraphTrainerCompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
):
    if parallel_dims.cp_enabled:
        raise ValueError(
            "Context parallelism is not supported for GraphTrainer MuseGlimmer."
        )

    assert training.seq_len % parallel_dims.seq_len_divisor == 0, (
        f"Sequence length {training.seq_len} must be divisible by the product "
        f"of TP degree ({parallel_dims.tp}) and 2 * CP degree "
        f"({parallel_dims.cp}), i.e. {parallel_dims.seq_len_divisor}."
    )

    annotate_module_fqns(model)

    if parallel_dims.tp_enabled:
        model.parallelize(parallel_dims)
        maybe_enable_async_tp(parallelism, compile_config, parallel_dims.get_mesh("tp"))

    parallelized_model = apply_simple_fsdp(
        model, parallel_dims=parallel_dims, training=training
    )
    parallelized_model = apply_compile(
        parallelized_model,
        compile_config=compile_config,
        parallelism=parallelism,
        parallel_dims=parallel_dims,
        dump_folder=dump_folder,
    )

    return parallelized_model

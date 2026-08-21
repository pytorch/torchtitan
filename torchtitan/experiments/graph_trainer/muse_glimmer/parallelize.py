# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
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

    annotate_module_fqns(model)

    if parallelism.spmd_backend == "spmd_types" or parallel_dims.tp_enabled:
        model.parallelize(parallel_dims)

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

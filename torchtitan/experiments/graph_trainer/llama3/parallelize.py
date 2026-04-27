# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.experiments.graph_trainer.common_utils import (
    annotate_module_fqns,
    apply_simple_fsdp,
)
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.llama3.model import GraphTrainerLlama3Model


def annotate_llama(model: GraphTrainerLlama3Model) -> None:
    """Attach module FQN annotations to FX graph nodes.

    Tags each submodule's forward with its fully-qualified name via
    ``torch.fx.traceback.annotate_fn`` for downstream passes (bucketing,
    SAC region boundaries, etc.).
    """
    annotate_module_fqns(model)


def parallelize_llama(
    model: GraphTrainerLlama3Model,
    *,
    parallel_dims: ParallelDims,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig | None = None,
    ac_config: ActivationCheckpointConfig | None = None,
    dump_folder: str = "",
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    annotate_llama(model)

    if parallel_dims.tp_enabled:
        tp_mesh = parallel_dims.get_mesh("tp")
        model.parallelize(tp_mesh)

    # Apply simple_fsdp unconditionally. The `fsdp` mesh always exists with a
    # real backend (see ParallelDims._mesh_exist), even at degree 1, so that
    # MixedPrecisionPolicy's param_dtype cast still applies in single-GPU runs.
    model = apply_simple_fsdp(model, parallel_dims=parallel_dims, parallelism=parallelism)

    # Apply compilation based on mode
    model = apply_compile(
        model,
        compile_config=compile_config,
        parallelism=parallelism,
        parallel_dims=parallel_dims,
        dump_folder=dump_folder,
    )

    return model

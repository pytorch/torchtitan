# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3 parallelization: minimal FSDP-only baseline.

This is the starting configuration for autoresearch experiments. It applies
composable FSDP (``fully_shard``) over the data-parallel mesh and nothing else:
no tensor/context/pipeline/expert parallel, no quantization, no compile, no
custom kernels. It is intentionally simple and high-precision so it can serve as
the clean base recipe (and the quality golden) that experiments build on.
"""

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.fsdp import (
    disable_fsdp_gradient_division,
    get_fsdp_reshard_after_forward_policy,
)
from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan.tools.logging import logger


def parallelize_qwen3(
    model: Qwen3Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
    skip_dp: bool = False,
):
    """Apply the baseline FSDP-only parallelization for the Qwen3 14B workload.

    Wraps each transformer block, ``lm_head``, and the root model with composable
    FSDP on the ``fsdp`` mesh, using the configured mixed-precision policy. This
    baseline is DP-only; the experiment search is expected to extend it.
    """
    if parallel_dims.tp_enabled:
        raise NotImplementedError("Qwen3 baseline FSDP does not support TP.")
    if parallel_dims.cp_enabled:
        raise NotImplementedError("Qwen3 baseline FSDP does not support CP.")
    if parallel_dims.pp_enabled:
        raise NotImplementedError("Qwen3 baseline FSDP does not support PP.")
    if parallel_dims.ep_enabled:
        raise NotImplementedError("Qwen3 baseline FSDP does not support EP.")

    if skip_dp or not parallel_dims.dp_enabled:
        return model

    if parallel_dims.dp_replicate != 1:
        raise NotImplementedError("Qwen3 baseline FSDP does not support HSDP.")
    if training.enable_cpu_offload:
        raise NotImplementedError("Qwen3 baseline FSDP does not support CPU offload.")

    fsdp_mesh = parallel_dims.get_mesh("fsdp")
    mp_policy = MixedPrecisionPolicy(
        param_dtype=getattr(torch, training.mixed_precision_param),
        reduce_dtype=getattr(torch, training.mixed_precision_reduce),
        cast_forward_inputs=False,
    )
    reshard_after_forward = get_fsdp_reshard_after_forward_policy(
        parallelism.fsdp_reshard_after_forward,
        parallel_dims.pp_enabled,
    )
    fsdp_config = {
        "mesh": fsdp_mesh,
        "mp_policy": mp_policy,
        "reshard_after_forward": reshard_after_forward,
    }

    # Shard each transformer block, then lm_head, then the root (which also covers
    # tok_embeddings). The last block keeps its parameters after forward because
    # they are needed immediately at the start of backward.
    layers = list(model.layers.values())
    for idx, layer in enumerate(layers):
        layer_config = fsdp_config
        if idx == len(layers) - 1:
            layer_config = {**fsdp_config, "reshard_after_forward": False}
        fully_shard(layer, **layer_config)
    fully_shard(model.lm_head, **fsdp_config)
    fully_shard(model, **fsdp_config)

    # TorchTitan scales gradients by the token count in the loss, so FSDP must not
    # additionally divide gradients by the data-parallel size.
    disable_fsdp_gradient_division(model)

    logger.info(
        "Applied baseline Qwen3 FSDP with dp_shard=%s, reshard_after_forward=%s",
        parallel_dims.dp_shard,
        reshard_after_forward,
    )
    return model

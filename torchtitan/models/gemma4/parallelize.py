# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
#
# Gemma-4 Model Parallelization
# Applies TP, activation checkpointing, torch.compile, and FSDP to the model

import torch.nn as nn

from torchtitan.config import (
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import ActivationCheckpointingConfig
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.fsdp import (
    apply_fsdp_to_decoder,
    resolve_fsdp_mesh,
    resolve_sparse_fsdp_mesh,
)
from torchtitan.models.gemma4.model import Gemma4Model
from torchtitan.tools.logging import logger


def parallelize_gemma4(
    model: Gemma4Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointingConfig,
    dump_folder: str,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the Gemma-4 model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    # Untie tok_embeddings / lm_head weights before FSDP wrapping.
    #
    # Gemma-4 uses ``enable_weight_tying=True``, which makes
    # ``tok_embeddings.weight`` and ``lm_head.weight`` the *same* Parameter
    # object. When ChunkedLossWrapper is used, lm_head is invoked per-chunk
    # inside the loss function and its FSDP group is explicitly resharded via
    # ``lm_head.reshard()`` after the chunked loop — before the decoder
    # backward propagates back through norm → tok_embeddings.
    #
    # If the three modules share one FSDP param group (the tied path in
    # ``apply_fsdp_to_decoder``), that ``reshard()`` call also reshards
    # tok_embeddings and norm, so FSDP's backward hooks fire on an already-
    # resharded group and produce zero gradients for the entire decoder.
    #
    # The fix (same approach as the transformers_modeling_backend) is to clone
    # lm_head.weight into an independent Parameter before FSDP wraps the model.
    # FSDP then takes the non-tied branch: tok_embeddings in its own group,
    # [norm, lm_head] in a separate group. ``lm_head.reshard()`` only touches
    # that second group, leaving tok_embeddings unaffected during backward.
    #
    # Note: this causes tok_embeddings and lm_head to diverge during training,
    # which is expected — the model learns independent embedding and output
    # projections. Checkpoint loading via Gemma4StateDictAdapter correctly
    # re-ties the weights at load time; they become independent again here.
    if model.enable_weight_tying and model.tok_embeddings is not None and model.lm_head is not None:
        model.lm_head.weight = nn.Parameter(
            model.lm_head.weight.clone(),
            requires_grad=model.lm_head.weight.requires_grad,
        )
        model.enable_weight_tying = False
        logger.info(
            "Untied tok_embeddings/lm_head weights for FSDP + ChunkedLossWrapper "
            "compatibility. Parameters will train independently."
        )

    if parallelism.spmd_backend == "spmd_types" or parallel_dims.tp_enabled:
        model.parallelize(parallel_dims)
    model_compile_enabled = (
        compile_config.enable and "model" in compile_config.components
    )

    if ac_config is not None:
        ac_config.build(dump_folder=dump_folder).apply(model)

    # turn on per-TransformerBlock compile after AC wrapping and before FSDP
    if model_compile_enabled:
        apply_compile(
            model,
            compile_config=compile_config,
            parallel_dims=parallel_dims,
        )

    # Always run apply_fsdp_to_decoder -- with shard_degree=1 it is a no-op for
    # the all-gather but still installs the MixedPrecisionPolicy.
    if parallelism.spmd_backend == "spmd_types":
        dp_mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims)
        edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallel_dims)
    else:
        names = (
            ["dp_replicate", "fsdp"] if parallel_dims.dp_replicate_enabled else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(names)
        dp_mesh_dims = None
        edp_mesh = None
        edp_mesh_dims = None
        if parallel_dims.ep_enabled:
            edp_mesh_names = (
                ["dp_replicate", "efsdp"]
                if parallel_dims.dp_replicate_enabled
                else ["efsdp"]
            )
            edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

    apply_fsdp_to_decoder(
        model,
        dp_mesh,
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        pp_enabled=parallel_dims.pp_enabled,
        cpu_offload=training.enable_cpu_offload,
        reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
        ep_degree=parallel_dims.ep,
        edp_mesh=edp_mesh,
        dp_mesh_dims=dp_mesh_dims,
        edp_mesh_dims=edp_mesh_dims,
        enable_symm_mem=parallelism.enable_fsdp_symm_mem,
    )

    return model

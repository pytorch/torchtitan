# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Qwen3 parallelization scaffold.

This file is intentionally strategy-free. Autoresearch is expected to replace
``parallelize_qwen3`` with an implementation specialized to the exact train
command, model flavor, and cluster/system it is optimizing for.
"""

import torch
import torch.nn.functional as F
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.compile import apply_compile
from torchtitan.distributed.fsdp import (
    disable_fsdp_gradient_division,
    get_fsdp_reshard_after_forward_policy,
)
from torchtitan.components.quantization.mx import mxfp8_shared_input_gate_up
from torchtitan.models.qwen3.model import Qwen3Model
from torchtitan.tools.logging import logger


def _enable_shared_mxfp8_gate_up_input_cast(model: Qwen3Model) -> None:
    patched_count = 0

    def _shared_mxfp8_feed_forward(x: torch.Tensor, *, feed_forward):
        w1_out, w3_out = mxfp8_shared_input_gate_up(
            x,
            feed_forward.w1.weight,
            feed_forward.w3.weight,
        )
        return feed_forward.w2(F.silu(w1_out) * w3_out)

    for layer in model.layers.values():
        feed_forward = getattr(layer, "feed_forward", None)
        if feed_forward is None:
            continue
        if not (
            hasattr(feed_forward.w1.weight, "config")
            and hasattr(feed_forward.w3.weight, "config")
        ):
            continue
        feed_forward.forward = lambda x, feed_forward=feed_forward: (
            _shared_mxfp8_feed_forward(x, feed_forward=feed_forward)
        )
        patched_count += 1

    logger.info(
        "Enabled shared MXFP8 input casts for %s Qwen3 feed-forward modules",
        patched_count,
    )


def _compile_qwen3_feed_forward(
    model: Qwen3Model, compile_config: CompileConfig
) -> None:
    compiled_count = 0
    for layer in model.layers.values():
        feed_forward = getattr(layer, "feed_forward", None)
        if feed_forward is None:
            continue
        feed_forward.compile(backend=compile_config.backend, fullgraph=True)
        compiled_count += 1

    logger.info(
        "Compiling %s Qwen3 feed-forward modules with torch.compile",
        compiled_count,
    )


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
    """Generated machine-specific Qwen3 parallelization entry point.

    A valid implementation may make narrow assumptions about the train command,
    mesh shape, hardware topology, memory budget, model flavor, and enabled
    TorchTitan features. It does not need to be a universal implementation.
    """
    if parallel_dims.tp_enabled:
        raise NotImplementedError("Qwen3 DP-only parallelize does not support TP.")
    if parallel_dims.cp_enabled:
        raise NotImplementedError("Qwen3 DP-only parallelize does not support CP.")
    if parallel_dims.pp_enabled:
        raise NotImplementedError("Qwen3 DP-only parallelize does not support PP.")
    if parallel_dims.ep_enabled:
        raise NotImplementedError("Qwen3 DP-only parallelize does not support EP.")

    if skip_dp or not parallel_dims.dp_enabled:
        return model

    if parallel_dims.dp_replicate != 1:
        raise NotImplementedError("Qwen3 baseline FSDP bootstrap does not support HSDP.")
    if training.enable_cpu_offload:
        raise NotImplementedError(
            "Qwen3 baseline FSDP bootstrap does not support CPU offload."
        )

    if compile_config.enable and "model" in compile_config.components:
        apply_compile(model, compile_config)
    _enable_shared_mxfp8_gate_up_input_cast(model)
    if compile_config.enable and "feed_forward" in compile_config.components:
        _compile_qwen3_feed_forward(model, compile_config)

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

    layers = list(model.layers.values())
    for idx, layer in enumerate(layers):
        layer_fsdp_config = fsdp_config
        if idx == len(layers) - 1:
            layer_fsdp_config = {
                **fsdp_config,
                "reshard_after_forward": False,
            }
        fully_shard(layer, **layer_fsdp_config)
    fully_shard(model.lm_head, **fsdp_config)
    fully_shard(model, **fsdp_config)
    if layers:
        for layer, next_layer in zip(layers, layers[1:]):
            layer.set_modules_to_forward_prefetch([next_layer])
        layers[-1].set_modules_to_forward_prefetch([model.lm_head])
        model.lm_head.set_modules_to_backward_prefetch([layers[-1]])
        for layer, prev_layer in zip(reversed(layers[1:]), reversed(layers[:-1])):
            layer.set_modules_to_backward_prefetch([prev_layer])

    disable_fsdp_gradient_division(model)
    logger.info(
        "Applied baseline Qwen3 FSDP with dp_shard=%s, reshard_after_forward=%s",
        parallel_dims.dp_shard,
        reshard_after_forward,
    )

    return model

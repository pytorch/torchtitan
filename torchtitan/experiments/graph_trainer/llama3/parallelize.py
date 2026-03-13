# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch.fx.traceback import annotate_fn

from torchtitan.components.quantization.float8 import find_float8_linear_config
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TORCH_DTYPE_MAP,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.experiments.graph_trainer.common_utils import maybe_disable_eager_ac
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    data_parallel,
    MixedPrecisionPolicy,
)
from torchtitan.models.llama3.model import Llama3Model
from torchtitan.models.llama3.parallelize import apply_tp
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools.logging import logger


# for selective op activation checkpointing
_op_sac_save_list = {
    torch.ops.aten.mm.default,
    torch.ops.aten.linear.default,
    torch.ops.aten._scaled_dot_product_efficient_attention.default,
    torch.ops.aten._scaled_dot_product_flash_attention.default,
    torch.ops.aten._scaled_dot_product_cudnn_attention.default,
    torch.ops.aten._scaled_dot_product_attention_math.default,
    torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
    torch.ops._c10d_functional.reduce_scatter_tensor.default,
    # for low precision training, it's useful to always save
    # the result of max, since the absolute maximum is
    # used to compute the scaling factor for quantization.
    torch.ops.aten.max.default,
    torch._higher_order_ops.flex_attention,
    torch.ops.torch_attn._varlen_attn.default,
    torch._higher_order_ops.inductor_compiled_code,
}


def annotate_llama() -> None:
    """Attach annotations to FX graph nodes with ``torch.fx.traceback.annotate_fn``

    - Flex attention annotation: Tags FlexAttentionWrapper.forward with
      {"compile_with_inductor": "flex_attention"} so the compiler can apply
      regional inductor pass based on the annotation. Regional inductor is now only
      supported in AOT mode.

    """
    from torchtitan.models.common.attention import FlexAttentionWrapper

    FlexAttentionWrapper.forward = annotate_fn(
        {"compile_with_inductor": "flex_attention"}
    )(FlexAttentionWrapper.forward)


def parallelize_llama(
    model: Llama3Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    """
    Apply tensor parallelism, activation checkpointing, torch.compile, and data
    parallelism to the model.

    NOTE: The passed-in model preferably should be on meta device. Otherwise,
    the model must fit on GPU or CPU memory.
    """
    # TODO: TP currently cannot handle uneven seq_len because we set
    #       `use_local_output=True` to use plain Tensors for legacy reasons.
    #       Need to revisit this.
    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}).
        """

    annotate_llama()

    maybe_disable_eager_ac(compile_config, ac_config)

    if parallel_dims.tp_enabled:
        float8_config = find_float8_linear_config(model_converters.converters)
        enable_float8_linear = float8_config is not None
        float8_is_rowwise = float8_config is not None and float8_config.recipe_name in (
            "rowwise",
            "rowwise_with_gw_hp",
        )

        # For now, float8 all-gather with TP is only supported for tensorwise
        # float8 scaling recipes. For rowwise recipes, we use regular TP and
        # all-gather happens in high precision.
        enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise

        tp_mesh = parallel_dims.get_mesh("tp")
        apply_tp(
            model,
            tp_mesh,
            loss_parallel=not parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=enable_float8_tensorwise_tp,
        )
        maybe_enable_async_tp(parallelism, compile_config, tp_mesh)

    if ac_config.mode != "none":
        model_compile_enabled = (
            compile_config.enable and "model" in compile_config.components
        )
        apply_ac(
            model,
            ac_config,
            model_compile_enabled=model_compile_enabled,
            op_sac_save_list=_op_sac_save_list,
            base_folder=dump_folder,
        )

    # apply data parallel
    if (
        parallel_dims.dp_replicate_enabled
        or parallel_dims.dp_shard_enabled
        or parallel_dims.cp_enabled
    ):
        if parallel_dims.dp_replicate_enabled:
            if parallel_dims.dp_shard_enabled or parallel_dims.cp_enabled:
                dp_mesh_dim_names = ["dp_replicate", "fsdp"]
                dp_mode = "hybrid_shard"
            else:
                dp_mesh_dim_names = ["dp_replicate"]
                dp_mode = "replicate"
        else:
            dp_mesh_dim_names = ["fsdp"]
            dp_mode = "fully_shard"

        mp_policy = MixedPrecisionPolicy(
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
        )

        model = data_parallel(
            model,
            parallel_dims.get_mesh(dp_mesh_dim_names),
            mode=dp_mode,
            mp_policy=mp_policy,
        )
        logger.info(
            "Applied Data Parallel (simple_fsdp) (dp mode=%s) to the model", dp_mode
        )

    # Apply compilation based on mode
    model = apply_compile(
        model,
        compile_config=compile_config,
        parallelism=parallelism,
        parallel_dims=parallel_dims,
        dump_folder=dump_folder,
    )

    return model

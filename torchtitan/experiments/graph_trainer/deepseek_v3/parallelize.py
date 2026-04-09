# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.distributed.device_mesh import DeviceMesh
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

from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.experiments.graph_trainer.common_utils import (
    annotate_ac_regions,
    apply_graph_ac,
)
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.deepseek_v3.model import (
    GraphTrainerDeepSeekV3Model,
)

from torchtitan.experiments.graph_trainer.simple_fsdp import (
    data_parallel,
    MixedPrecisionPolicy,
)
from torchtitan.models.deepseek_v3.parallelize import apply_moe_ep_tp, apply_non_moe_tp
from torchtitan.protocols.model_converter import ModelConvertersContainer
from torchtitan.tools.logging import logger


def annotate_deepseekv3(model: GraphTrainerDeepSeekV3Model) -> None:
    """Attach annotations to FX graph nodes with ``torch.fx.traceback.annotate_fn``

    - Expert Parallel (EP) annotations: Tags "dispatch", "combine", and "compute"
      regions in MoE for debugging purposes.
    - Flex attention annotation: Tags FlexAttention.forward with
      {"compile_with_inductor": "flex_attention"} so the compiler can apply
      regional inductor pass based on the annotation. Regional inductor is now only
      supported in AOT mode.
    - AC region annotation: Tags each transformer block's forward with a unique
      ac_region_id so that apply_sac_pass can assign per-block ac_graph_id
      boundaries for the min-cut partitioner.

    """
    from torchtitan.distributed.expert_parallel import ExpertParallel
    from torchtitan.models.common.attention import FlexAttention
    from torchtitan.models.common.moe import MoE

    ExpertParallel._token_dispatch = annotate_fn({"EP": "dispatch"})(
        ExpertParallel._token_dispatch
    )
    ExpertParallel._token_combine = annotate_fn({"EP": "combine"})(
        ExpertParallel._token_combine
    )
    MoE.forward = annotate_fn({"EP": "compute"})(MoE.forward)

    FlexAttention.forward = annotate_fn({"compile_with_inductor": "flex_attention"})(
        FlexAttention.forward
    )

    annotate_ac_regions(model)


# Adapted from llama4/infra/parallelize.py
def parallelize_deepseekv3(
    model: GraphTrainerDeepSeekV3Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    # TODO: TP currently cannot handle uneven seq_len because we set
    #       `use_local_output=True` to use plain Tensors for legacy reasons.
    #       Need to revisit this.
    assert (
        training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}), i.e. {parallel_dims.seq_len_divisor}.
        """

    from torchtitan.models.common.attention import ScaledDotProductAttention

    if parallelism.context_parallel_degree > 1 and not isinstance(
        model.config.layers[0].attention.inner_attention,
        ScaledDotProductAttention.Config,
    ):
        raise NotImplementedError("CP support is only supported for SDPA.")

    annotate_deepseekv3(model)

    if parallel_dims.tp_enabled:
        float8_config = find_float8_linear_config(model_converters.converters)
        enable_float8_linear = float8_config is not None
        float8_is_rowwise = float8_config is not None and float8_config.recipe_name in (
            "rowwise",
            "rowwise_with_gw_hp",
        )

        enable_float8_tensorwise_tp = enable_float8_linear and not float8_is_rowwise
        if enable_float8_tensorwise_tp:
            # TODO(jianiw): This branch needs to be tested and enabled
            raise NotImplementedError(
                "Currently, float8 tensorwise TP is not tested for deepseekv3"
            )

        apply_non_moe_tp(
            model,
            parallel_dims.get_mesh("tp"),
            enable_loss_parallel=not parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=False,
            enable_cp=parallel_dims.cp_enabled,
            enable_sp=parallelism.enable_sequence_parallel,
        )
        maybe_enable_async_tp(parallelism, compile_config, parallel_dims.get_mesh("tp"))

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        from torchtitan.components.quantization import find_pad_multiple

        pad_multiple = find_pad_multiple(model_converters.converters)

        apply_moe_ep_tp(
            model,
            tp_mesh=parallel_dims.get_optional_mesh("tp"),
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
            etp_mesh=parallel_dims.get_optional_mesh("etp"),
            ep_etp_mesh=parallel_dims.get_optional_mesh(["ep", "etp"]),
            pad_multiple=pad_multiple,
        )

    if ac_config.mode != "none":
        apply_graph_ac(compile_config, ac_config)

    mp_policy = MixedPrecisionPolicy(
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
    )

    # apply data parallel
    dp_mesh: DeviceMesh | None = None
    if (
        parallel_dims.fsdp_enabled
        or parallel_dims.ep_enabled
        or parallel_dims.dp_replicate_enabled
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

        dp_mesh = parallel_dims.get_mesh(dp_mesh_dim_names)

        # the mesh dim names of which the MoE params are sharded on via FSDP/HSDP
        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

        for _, transformer_block in model.layers.items():
            if transformer_block.moe_enabled and parallel_dims.ep_enabled:
                experts_shard_dim = 0
                assert edp_mesh is not None
                assert hasattr(transformer_block, "moe")
                if (
                    edp_mesh["efsdp"].size() * parallel_dims.ep
                    > transformer_block.moe.experts.num_experts
                ):
                    experts_shard_dim = 1

                transformer_block.moe.experts = data_parallel(
                    transformer_block.moe.experts,
                    edp_mesh,
                    dp_mode,
                    mp_policy=mp_policy,
                    shard_dim=experts_shard_dim,
                )

        model = data_parallel(
            model,
            dp_mesh,
            dp_mode,
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

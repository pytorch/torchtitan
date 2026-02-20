# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.fx.traceback import annotate_fn

from torchtitan.config import JobConfig, TORCH_DTYPE_MAP
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.models.deepseek_v3.infra.parallelize import (
    apply_moe_ep_tp,
    apply_non_moe_tp,
)
from torchtitan.tools.logging import logger

from ..common_utils import register_blockmask_pytree_node
from ..compilation import apply_compilation
from ..simple_fsdp import data_parallel, MixedPrecisionPolicy


def get_transformer_block_buckets(model) -> list[list[str] | str]:
    module_list = [
        model.tok_embeddings,
        [model.norm, model.output],
    ]
    for layer_id, transformer_block in model.layers.items():
        # [TODO](ruisizhang123) add EP support for transformer block bucketing
        module_list.append(transformer_block)

    def convert_modules_to_fqns(modules, module_to_fqn_mapping):
        """Convert a (possibly nested) list of modules to FQN strings."""
        result = []
        for m in modules:
            if isinstance(m, list):
                result.append(convert_modules_to_fqns(m, module_to_fqn_mapping))
            else:
                result.append(module_to_fqn_mapping.get(m, None))
        return result

    module_to_name = {m: n for n, m in model.named_modules()}
    module_fqns = convert_modules_to_fqns(module_list, module_to_name)
    return module_fqns


def annotate_deepseekv3() -> None:
    from torchtitan.distributed.expert_parallel import ExpertParallel
    from torchtitan.models.attention import FlexAttentionWrapper
    from torchtitan.models.moe.moe import MoE

    # annotate the MoE with dispatch, compute and combine
    ExpertParallel._token_dispatch = annotate_fn({"EP": "dispatch"})(
        ExpertParallel._token_dispatch
    )
    ExpertParallel._token_combine = annotate_fn({"EP": "combine"})(
        ExpertParallel._token_combine
    )
    MoE.forward = annotate_fn({"EP": "compute"})(MoE.forward)

    # annotate flex_attention with compile_with_inductor
    FlexAttentionWrapper.forward = annotate_fn(
        {"compile_with_inductor": "flex_attention"}
    )(FlexAttentionWrapper.forward)


# Adapted from llama4/infra/parallelize.py
def parallelize_deepseekv3(
    model: nn.Module,
    parallel_dims: ParallelDims,
    job_config: JobConfig,
):
    compile_mode = getattr(job_config.compile, "mode", None)

    # Annotations for AOT mode (must happen before tracing)
    if compile_mode == "aot":
        annotate_deepseekv3()
        register_blockmask_pytree_node()

    # TODO: TP currently cannot handle uneven seq_len because we set
    #       `use_local_output=True` to use plain Tensors for legacy reasons.
    #       Need to revisit this.
    assert (
        job_config.training.seq_len % parallel_dims.seq_len_divisor == 0
    ), f"""
        Sequence length {job_config.training.seq_len} must be divisible by the product of TP degree
        ({parallel_dims.tp}) and 2 * CP degree ({parallel_dims.cp}), i.e. {parallel_dims.seq_len_divisor}.
        """

    if (
        job_config.parallelism.context_parallel_degree > 1
        and model.model_args.attn_type != "sdpa"
    ):
        raise NotImplementedError("CP support is only supported for SDPA.")

    if parallel_dims.tp_enabled:
        enable_float8_linear = "float8" in job_config.model.converters
        float8_is_rowwise = job_config.quantize.linear.float8.recipe_name in (
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
            loss_parallel=not job_config.parallelism.disable_loss_parallel,
            enable_float8_tensorwise_tp=False,
            cp_enabled=parallel_dims.cp_enabled,
        )
        maybe_enable_async_tp(job_config, parallel_dims.get_mesh("tp"))

    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        apply_moe_ep_tp(
            model,
            tp_mesh=parallel_dims.get_optional_mesh("tp"),
            ep_mesh=parallel_dims.get_optional_mesh("ep"),
            etp_mesh=parallel_dims.get_optional_mesh("etp"),
            ep_etp_mesh=parallel_dims.get_optional_mesh(["ep", "etp"]),
        )

    if job_config.activation_checkpoint.mode != "none":
        apply_ac(model, job_config.activation_checkpoint)

    mp_policy = MixedPrecisionPolicy(
        param_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[job_config.training.mixed_precision_reduce],
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

    # apply compilation
    if compile_mode:
        torch._dynamo.config.capture_scalar_outputs = True
        model = apply_compilation(
            model,
            job_config,
            parallel_dims,
            compile_mode,
            transformer_block_buckets=get_transformer_block_buckets(model),
        )

    return model

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torch.fx.traceback import annotate_fn

from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.tensor_parallel import maybe_enable_async_tp
from torchtitan.experiments.graph_trainer.common_utils import (
    annotate_module_fqns,
    apply_simple_fsdp,
)
from torchtitan.experiments.graph_trainer.compile import apply_compile
from torchtitan.experiments.graph_trainer.deepseek_v3.model import (
    GraphTrainerDeepSeekV3Model,
)
from torchtitan.tools.logging import logger


def annotate_deepseekv3(model: GraphTrainerDeepSeekV3Model) -> None:
    """Attach annotations to FX graph nodes with ``torch.fx.traceback.annotate_fn``

    - Expert Parallel (EP) annotations: Tags "dispatch", "combine", and "compute"
      regions in MoE for debugging purposes.
    - Module FQN annotation: Tags each submodule's forward with its
      fully-qualified name for downstream passes (bucketing, SAC region
      boundaries, etc.).
    """
    from torchtitan.models.common.moe import MoE
    from torchtitan.models.common.token_dispatcher import LocalTokenDispatcher

    LocalTokenDispatcher.dispatch = annotate_fn({"EP": "dispatch"})(
        LocalTokenDispatcher.dispatch
    )
    LocalTokenDispatcher.combine = annotate_fn({"EP": "combine"})(
        LocalTokenDispatcher.combine
    )
    MoE.forward = annotate_fn({"EP": "compute"})(MoE.forward)

    annotate_module_fqns(model)


# Adapted from llama4/infra/parallelize.py
def parallelize_deepseekv3(
    model: GraphTrainerDeepSeekV3Model,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
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

    # Read comm_backend from the model's token dispatcher config
    # (set by moe_comm_backend in model_registry / config factories).
    moe_config = next((l.moe for l in model.config.layers if l.moe is not None), None)
    comm_backend = (
        getattr(moe_config.experts.token_dispatcher, "comm_backend", "standard")
        if moe_config is not None
        else "standard"
    )
    if comm_backend == "hybridep":
        from torchtitan.distributed.deepep import hybridep  # noqa: F401

    # Config-based sharding: ShardingConfig is populated on the model config
    # in ``DeepSeekV3Model.update_from_config`` (dense + MoE);
    # ``Module.parallelize`` applies it. Replaces the imperative
    # ``apply_moe_ep_tp`` call.
    if parallel_dims.tp_enabled or parallel_dims.ep_enabled:
        model.parallelize(parallel_dims)
    if parallel_dims.tp_enabled:
        maybe_enable_async_tp(parallelism, compile_config, parallel_dims.get_mesh("tp"))

    # Apply simple_fsdp unconditionally. The `fsdp` mesh always exists with a
    # real backend (see ParallelDims._mesh_exist), even at degree 1, so that
    # MixedPrecisionPolicy's param_dtype cast still applies in single-GPU runs.
    model = apply_simple_fsdp(model, parallel_dims=parallel_dims, training=training)

    # TODO: HybridEP applies to other sparse models (e.g. Qwen3). Refactor
    # the common HybridEP buffer init into a shared utility.
    if comm_backend == "hybridep" and parallel_dims.ep_enabled:
        from torchtitan.distributed.deepep.hybridep import get_buffer

        ep_group = parallel_dims.get_mesh("ep").get_group()
        moe_config = next(l.moe for l in model.config.layers if l.moe is not None)
        num_local_experts = moe_config.num_experts // parallel_dims.ep
        hidden_dim = model.config.dim
        num_tokens = training.local_batch_size * training.seq_len
        get_buffer(
            group=ep_group,
            hidden_dim=hidden_dim,
            num_tokens=num_tokens,
            num_local_experts=num_local_experts,
        )
        logger.info(
            f"Pre-initialized HybridEP buffer (hidden_dim={hidden_dim}, "
            f"num_tokens={num_tokens}, num_local_experts={num_local_experts})"
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

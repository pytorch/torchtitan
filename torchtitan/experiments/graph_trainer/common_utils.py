# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from collections.abc import Callable
from typing import Any

import torch.nn as nn

from graph_trainer import common_utils as _graph_trainer_common
from graph_trainer.common_utils import annotate_module_fqns, get_simple_fsdp_mesh
from torch.distributed.device_mesh import DeviceMesh
from torch.fx.traceback import annotate_fn

from torchtitan.config import TORCH_DTYPE_MAP, TrainingConfig
from torchtitan.distributed import ParallelismContext
from torchtitan.experiments.graph_trainer.simple_fsdp import (
    data_parallel,
    MixedPrecisionPolicy,
)
from torchtitan.models.base.module import Module
from torchtitan.models.common.decoder import Decoder, TransformerBlock
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.moe import MoE


logger = logging.getLogger(__name__)


def __getattr__(name: str) -> Any:
    """Delegate reusable helpers to the standalone GraphTrainer package."""
    return getattr(_graph_trainer_common, name)


def disable_inplace_wgrad_accum(model: Module.Config) -> None:
    """Turn off in-place WGrad accumulation for GraphTrainer tracing."""
    for _, linear_config, _, _ in model.traverse(Linear.Config):
        linear_config.inplace_wgrad_accum = False


def build_decoder_config_for_backend(
    config_builder: Callable, attn_backend: str, **builder_kwargs
):
    """Build a Decoder config with a supported backend and traceable linears."""
    config = config_builder(attn_backend=attn_backend, **builder_kwargs)
    disable_inplace_wgrad_accum(config)
    return config


def annotate_graph_trainer_model(model: Decoder) -> None:
    """Attach the annotations consumed by GraphTrainer passes."""
    if model.config.first_moe is not None:
        annotate_moe_ep_regions()
    annotate_module_fqns(model)


_MOE_EP_REGIONS_ANNOTATED = False


def annotate_moe_ep_regions() -> None:
    """Annotate MoE EP compute, dispatch, and combine regions for FX passes."""
    global _MOE_EP_REGIONS_ANNOTATED
    if _MOE_EP_REGIONS_ANNOTATED:
        return

    from torchtitan.models.common.moe.token_dispatcher import (
        AllToAllTokenDispatcher,
        LocalTokenDispatcher,
    )

    LocalTokenDispatcher.dispatch = annotate_fn({"EP": "dispatch"})(
        LocalTokenDispatcher.dispatch
    )
    LocalTokenDispatcher.combine = annotate_fn({"EP": "combine"})(
        LocalTokenDispatcher.combine
    )
    AllToAllTokenDispatcher.dispatch = annotate_fn({"EP": "dispatch"})(
        AllToAllTokenDispatcher.dispatch
    )
    AllToAllTokenDispatcher.combine = annotate_fn({"EP": "combine"})(
        AllToAllTokenDispatcher.combine
    )
    AllToAllTokenDispatcher._token_count_exchange = annotate_fn(
        {_graph_trainer_common._EP_TOKEN_COUNT_EXCHANGE: "dispatch"}
    )(AllToAllTokenDispatcher._token_count_exchange)
    AllToAllTokenDispatcher._sync_token_count_exchange = annotate_fn(
        {_graph_trainer_common._EP_TOKEN_COUNT_SYNC: "dispatch"}
    )(AllToAllTokenDispatcher._sync_token_count_exchange)
    AllToAllTokenDispatcher._dispatch_token_exchange = annotate_fn(
        {_graph_trainer_common._EP_TOKEN_EXCHANGE: "dispatch"}
    )(AllToAllTokenDispatcher._dispatch_token_exchange)
    AllToAllTokenDispatcher._combine_token_exchange = annotate_fn(
        {_graph_trainer_common._EP_TOKEN_EXCHANGE: "combine"}
    )(AllToAllTokenDispatcher._combine_token_exchange)
    MoE.forward = annotate_fn({"EP": "compute"})(MoE.forward)
    _MOE_EP_REGIONS_ANNOTATED = True


def apply_simple_fsdp(
    model: nn.Module,
    *,
    parallelism_context: ParallelismContext,
    training: TrainingConfig,
) -> nn.Module:
    """Wrap the model and any MoE experts with GraphTrainer SimpleFSDP."""
    fsdp_mesh = get_simple_fsdp_mesh(parallelism_context)

    if parallelism_context.dp_replicate_enabled:
        if parallelism_context.dp_shard_enabled or parallelism_context.cp_enabled:
            dp_replicate_mesh = parallelism_context.get_optional_mesh(
                "dp_replicate", include_singleton_axes=True
            )
            assert dp_replicate_mesh is not None
            dp_mesh = DeviceMesh._concatenate([dp_replicate_mesh, fsdp_mesh])
            dp_mode = "hybrid_shard"
        else:
            dp_mesh = parallelism_context.get_mesh("dp_replicate")
            dp_mode = "replicate"
    else:
        dp_mesh = fsdp_mesh
        dp_mode = "fully_shard"

    mp_policy = MixedPrecisionPolicy(
        param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
        reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
    )

    if parallelism_context.ep_enabled and isinstance(model, Decoder):
        edp_mesh_names = (
            ["dp_replicate", "edp_shard"]
            if parallelism_context.dp_replicate_enabled
            else ["edp_shard"]
        )
        edp_mesh = parallelism_context.get_optional_mesh(edp_mesh_names)
        assert edp_mesh is not None

        for transformer_block in model.modules():
            if not isinstance(transformer_block, TransformerBlock):
                continue
            moe = transformer_block.feed_forward
            if not isinstance(moe, MoE):
                continue
            routed_experts = moe.routed_experts
            experts_shard_dim = 0
            if (
                edp_mesh["edp_shard"].size() * parallelism_context.ep
                > routed_experts.w13.group_size
            ):
                experts_shard_dim = 1

            if experts_shard_dim == 0:
                data_parallel(
                    routed_experts,
                    edp_mesh,
                    dp_mode,
                    mp_policy=mp_policy,
                    shard_dim=0,
                    non_dp_mesh=parallelism_context.get_optional_mesh("ep"),
                    param_shard_placements={},
                )
            else:
                data_parallel(
                    routed_experts.w13,
                    edp_mesh,
                    dp_mode,
                    mp_policy=mp_policy,
                    shard_dim=2,
                    non_dp_mesh=parallelism_context.get_optional_mesh("ep"),
                )
                data_parallel(
                    routed_experts.w2,
                    edp_mesh,
                    dp_mode,
                    mp_policy=mp_policy,
                    shard_dim=1,
                    non_dp_mesh=parallelism_context.get_optional_mesh("ep"),
                )

    model = data_parallel(
        model,
        dp_mesh,
        dp_mode,
        mp_policy=mp_policy,
        non_dp_mesh=parallelism_context.get_optional_mesh("tp"),
    )
    logger.info(
        "Applied Data Parallel (simple_fsdp) (dp mode=%s) to the model", dp_mode
    )
    return model

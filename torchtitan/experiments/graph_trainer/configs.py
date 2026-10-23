# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields

from graph_trainer.configs import (
    EpOverlapChunkDim,
    EpOverlapConfig,
    GraphTrainerCompileConfig,
    MOE_BLOCK_FQN,
    PagedStashConfig,
    SPMDGradientAccumulationConfig,
    SUPPORTED_EP_OVERLAP_MODULE_FQNS,
    TRANSFORMER_BLOCK_FQN,
    validate_ep_overlap_config,
)

from torchtitan.components.loss import ChunkedLossWrapper
from torchtitan.distributed.activation_remat import DefaultActivationRematConfig
from torchtitan.experiments.graph_trainer.chunked_loss import (
    ChunkedLossWrapperWithParamGrads,
)
from torchtitan.experiments.graph_trainer.common_utils import (
    disable_inplace_wgrad_accum,
)
from torchtitan.models.base.model import BaseModel
from torchtitan.trainer import Trainer


def to_graph_trainer_config(
    base_config: Trainer.Config,
    model_config_cls: type[BaseModel.Config],
) -> "GraphTrainer.Config":
    """Convert a base Trainer.Config to a GraphTrainer.Config.

    Copies all fields from the base config and converts its model config to the
    GraphTrainer model config class, without local compile regions or in-place
    WGRAD accumulation because GraphTrainer traces the whole step. The
    ``compile`` field keeps the GraphTrainer.Config default; callers should
    explicitly set it.
    """
    from .trainer import GraphTrainer

    d = {f.name: getattr(base_config, f.name) for f in fields(base_config)}
    graph_model = model_config_cls(
        **{
            f.name: getattr(base_config.model, f.name)
            for f in fields(base_config.model)
        }
    )
    # GraphTrainer compiles the whole step, so it drops local compile regions.
    graph_model.local_compile_regions = []
    disable_inplace_wgrad_accum(graph_model)
    d["model"] = graph_model

    # GraphTrainer uses graph-based SAC instead of eager activation remat.
    ac = d.get("activation_remat")
    if ac is not None:
        d["activation_remat"] = DefaultActivationRematConfig()

    # The tracer requires explicit autograd outputs for lm_head parameters.
    loss_cfg = d.get("loss")
    if isinstance(loss_cfg, ChunkedLossWrapper.Config):
        d["loss"] = ChunkedLossWrapperWithParamGrads.Config(
            **{f.name: getattr(loss_cfg, f.name) for f in fields(loss_cfg)}
        )

    return GraphTrainer.Config(**d)


__all__ = [
    "EpOverlapChunkDim",
    "EpOverlapConfig",
    "GraphTrainerCompileConfig",
    "MOE_BLOCK_FQN",
    "PagedStashConfig",
    "SPMDGradientAccumulationConfig",
    "SUPPORTED_EP_OVERLAP_MODULE_FQNS",
    "TRANSFORMER_BLOCK_FQN",
    "to_graph_trainer_config",
    "validate_ep_overlap_config",
]

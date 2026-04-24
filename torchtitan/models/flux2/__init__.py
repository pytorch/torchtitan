# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.components.loss import build_mse_loss
from torchtitan.protocols.model_spec import ModelSpec

from .flux2_datasets import Flux2DataLoader
from .model import Flux2Model
from .parallelize import parallelize_flux2

from torchtitan.models.flux2.src.flux2.model import Flux2Params, Klein4BParams, Klein9BParams

__all__ = [
    "Flux2Model",
    "Flux2DataLoader",
    "flux2_configs",
    "parallelize_flux2",
]


def _config_from_params(params) -> Flux2Model.Config:
    return Flux2Model.Config(
        in_channels=params.in_channels,
        context_in_dim=params.context_in_dim,
        hidden_size=params.hidden_size,
        num_heads=params.num_heads,
        depth=params.depth,
        depth_single_blocks=params.depth_single_blocks,
        axes_dim=tuple(params.axes_dim),
        theta=params.theta,
        mlp_ratio=params.mlp_ratio,
        use_guidance_embed=params.use_guidance_embed,
    )


flux2_configs = {
    "flux2-dev": _config_from_params(Flux2Params()),
    "flux2-klein-4b": _config_from_params(Klein4BParams()),
    "flux2-klein-9b": _config_from_params(Klein9BParams()),
    "flux2-klein-base-4b": _config_from_params(Klein4BParams()),
    "flux2-klein-base-9b": _config_from_params(Klein9BParams()),
}


def model_registry(flavor: str) -> ModelSpec:
    return ModelSpec(
        name="flux2",
        flavor=flavor,
        model=flux2_configs[flavor],
        parallelize_fn=parallelize_flux2,
        pipelining_fn=None,
        build_loss_fn=build_mse_loss,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )

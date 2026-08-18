# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields

from torchtitan.models.muse_glimmer import model_registry as muse_glimmer_model_registry
from torchtitan.protocols.model_spec import ModelSpec

from .model import GraphTrainerMuseGlimmerModel
from .parallelize import parallelize_muse_glimmer


def model_registry(
    flavor: str,
    attn_backend: str = "flex",
) -> ModelSpec:
    base = muse_glimmer_model_registry(flavor, attn_backend=attn_backend)
    config = GraphTrainerMuseGlimmerModel.Config(
        **{f.name: getattr(base.model, f.name) for f in fields(base.model)}
    )
    return ModelSpec(
        name="graph_trainer/muse_glimmer",
        flavor=flavor,
        model=config,
        parallelize_fn=parallelize_muse_glimmer,
        pipelining_fn=None,
        post_optimizer_build_fn=None,
        state_dict_adapter=None,
    )

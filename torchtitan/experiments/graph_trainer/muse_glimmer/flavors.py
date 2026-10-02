# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GraphTrainer Muse Glimmer model flavors."""

from dataclasses import fields

from torchtitan.models.muse_glimmer import (
    build_model_config as build_muse_glimmer_model_config,
)
from .model import GraphTrainerMuseGlimmerModel


def build_model_config(
    flavor: str,
    *,
    seq_len: int | None = None,
    attn_backend: str = "flex",
) -> GraphTrainerMuseGlimmerModel.Config:
    base = build_muse_glimmer_model_config(
        flavor, seq_len=seq_len, attn_backend=attn_backend
    )
    config = GraphTrainerMuseGlimmerModel.Config(
        **{f.name: getattr(base, f.name) for f in fields(base)}
    )
    return config

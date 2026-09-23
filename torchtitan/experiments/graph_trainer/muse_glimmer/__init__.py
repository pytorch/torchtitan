# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import fields

from torchtitan.models.muse_glimmer import model_registry as muse_glimmer_model_registry
from .model import GraphTrainerMuseGlimmerModel


def model_registry(
    flavor: str,
    *,
    seq_len: int | None = None,
    attn_backend: str = "flex",
) -> GraphTrainerMuseGlimmerModel.Config:
    base = muse_glimmer_model_registry(
        flavor, seq_len=seq_len, attn_backend=attn_backend
    )
    config = GraphTrainerMuseGlimmerModel.Config(
        **{f.name: getattr(base, f.name) for f in fields(base)}
    )
    return config

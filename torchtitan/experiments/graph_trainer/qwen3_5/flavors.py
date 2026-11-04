# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GraphTrainer Qwen3.5 model flavors."""

from dataclasses import fields

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.qwen3_5 import build_model_config as build_qwen35_model_config

from ..common_utils import disable_inplace_wgrad_accum
from .model import GraphTrainerQwen35Model


def build_model_config(
    flavor: str,
    *,
    parallelism: ParallelismConfig,
    seq_len: int | None = None,
    attn_backend: str = "varlen",
) -> GraphTrainerQwen35Model.Config:
    base = build_qwen35_model_config(
        flavor, parallelism=parallelism, seq_len=seq_len, attn_backend=attn_backend
    )
    config = GraphTrainerQwen35Model.Config(
        **{f.name: getattr(base, f.name) for f in fields(base)}
    )
    # GraphTrainer traces the backward.
    disable_inplace_wgrad_accum(config)
    return config

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GraphTrainer Kimi K2.7 model flavors."""

from dataclasses import fields

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.kimi_k2_7 import (
    build_model_config as build_kimi_k2_7_model_config,
    KimiK25Model,
)
from torchtitan.models.kimi_k2_7.qk_clip import QKClipFlexInnerAttention

from ..common_utils import disable_inplace_wgrad_accum
from .model import GraphTrainerKimiK25Model


def without_qk_clip(config: KimiK25Model.Config) -> KimiK25Model.Config:
    """Replace QK-clip attention with plain flex attention in place.

    QK clipping records each step's attention logits in a Python list, which a
    traced step does not replay (see ``GraphTrainerKimiK25Model.Config``).
    """
    for _, attention, parent, field_name in config.traverse(
        QKClipFlexInnerAttention.Config
    ):
        assert parent is not None and isinstance(field_name, str)
        setattr(
            parent,
            field_name,
            FlexInnerAttention.Config(
                **{f.name: getattr(attention, f.name) for f in fields(attention)}
            ),
        )
    return config


def build_model_config(
    flavor: str,
    *,
    parallelism: ParallelismConfig,
    seq_len: int | None = None,
) -> GraphTrainerKimiK25Model.Config:
    base = without_qk_clip(
        build_kimi_k2_7_model_config(
            flavor, parallelism=parallelism, seq_len=seq_len, attn_backend="flex"
        )
    )
    config = GraphTrainerKimiK25Model.Config(
        **{f.name: getattr(base, f.name) for f in fields(base)}
    )
    # GraphTrainer traces the backward.
    disable_inplace_wgrad_accum(config)
    return config

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""HuggingFace checkpoint adapter for Kimi K3.1 weights."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torchtitan.models.kimi_k3.state_dict_adapter import KimiK3StateDictAdapter

if TYPE_CHECKING:
    from .model import KimiK31Model


class KimiK31StateDictAdapter(KimiK3StateDictAdapter):
    def __init__(
        self,
        model_config: KimiK31Model.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)
        self.mla_from_hf_map.update(
            {
                "language_model.model.layers.{}.self_attn.indexer.wq_b.weight": "layers.{}.attention.indexer.wq_b.weight",
                "language_model.model.layers.{}.self_attn.indexer.wk.weight": "layers.{}.attention.indexer.wk.weight",
                "language_model.model.layers.{}.self_attn.indexer.k_norm.weight": (
                    "layers.{}.attention.indexer.k_norm.weight"
                ),
                "language_model.model.layers.{}.self_attn.indexer.k_norm.bias": (
                    "layers.{}.attention.indexer.k_norm.bias"
                ),
                "language_model.model.layers.{}.self_attn.indexer.weights_proj.weight": (
                    "layers.{}.attention.indexer.weights_proj.weight"
                ),
            }
        )

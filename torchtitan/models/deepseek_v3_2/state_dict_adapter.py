# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.models.deepseek_v3.state_dict_adapter import (
    DeepSeekV3StateDictAdapter,
)


class DeepSeekV32StateDictAdapter(DeepSeekV3StateDictAdapter):
    """StateDictAdapter for DeepSeek-V3.2 — extends V3 with indexer mappings."""

    def __init__(self, model_config, hf_assets_path):
        super().__init__(model_config, hf_assets_path)
        self.from_hf_map.update(
            {
                "model.layers.{}.self_attn.indexer.wq_b.weight": (
                    "layers.{}.attention.indexer.wq_b.weight"
                ),
                "model.layers.{}.self_attn.indexer.wk.weight": (
                    "layers.{}.attention.indexer.wk.weight"
                ),
                "model.layers.{}.self_attn.indexer.k_norm.weight": (
                    "layers.{}.attention.indexer.k_norm.weight"
                ),
                "model.layers.{}.self_attn.indexer.k_norm.bias": (
                    "layers.{}.attention.indexer.k_norm.bias"
                ),
                "model.layers.{}.self_attn.indexer.weights_proj.weight": (
                    "layers.{}.attention.indexer.weights_proj.weight"
                ),
            }
        )

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan.models.deepseek_v3.state_dict_adapter import DeepSeekV3StateDictAdapter

from .model import DeepSeekV32Model


class DeepSeekV32StateDictAdapter(DeepSeekV3StateDictAdapter):
    """StateDictAdapter for DeepSeekV3.2 model.

    Extends the V3 adapter with Lightning Indexer weight mappings.
    ``to_hf`` and ``from_hf`` are inherited unchanged — the Indexer
    keys follow the same ``layers.{}.`` pattern as all V3 layer keys
    and are handled by the parent's transform logic.
    """

    def __init__(
        self,
        model_config: DeepSeekV32Model.Config,
        hf_assets_path: str | None,
    ):
        super().__init__(model_config, hf_assets_path)
        self.from_hf_map.update(
            {
                "model.layers.{}.self_attn.dsa_indexer.q_b_proj.weight": (
                    "layers.{}.attention.indexer.wq_b.weight"
                ),
                "model.layers.{}.self_attn.dsa_indexer.k_proj.weight": (
                    "layers.{}.attention.indexer.wk.weight"
                ),
                "model.layers.{}.self_attn.dsa_indexer.k_norm.weight": (
                    "layers.{}.attention.indexer.k_norm.weight"
                ),
                "model.layers.{}.self_attn.dsa_indexer.k_norm.bias": (
                    "layers.{}.attention.indexer.k_norm.bias"
                ),
                "model.layers.{}.self_attn.dsa_indexer.weights_proj.weight": (
                    "layers.{}.attention.indexer.weights_proj.weight"
                ),
            }
        )

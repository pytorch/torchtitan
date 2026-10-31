# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import copy

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.deepseek_v4 import build_model_config
from torchtitan.models.deepseek_v4.state_dict_adapter import DeepSeekV4StateDictAdapter
from torchtitan.models.llama3 import build_model_config as build_llama_config


def test_hash_routing_keys_follow_each_layers_feed_forward_config():
    config = build_model_config(
        "debugmodel", seq_len=128, parallelism=ParallelismConfig()
    )
    dense_config = (
        build_llama_config("debugmodel", seq_len=128, parallelism=ParallelismConfig())
        .layers[0]
        .feed_forward
    )
    config.layers[0].feed_forward = copy.deepcopy(dense_config)
    config.layers[2].feed_forward = copy.deepcopy(config.layers[1].feed_forward)
    adapter = DeepSeekV4StateDictAdapter(config, hf_assets_path=None)

    assert "layers.0.ffn.gate.tid2eid" not in adapter.from_hf_map
    assert adapter.from_hf_map["layers.1.ffn.gate.tid2eid"] == (
        "layers.1.feed_forward.router.tid2eid"
    )
    assert "layers.2.ffn.gate.tid2eid" not in adapter.from_hf_map

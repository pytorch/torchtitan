# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.models.glm5 import build_model_config
from torchtitan.models.glm5.state_dict_adapter import Glm5StateDictAdapter
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan_recipes.tests.models.glm5 import glm5_debugmodel, glm5_2_scaled_debugmodel
from torchtitan_recipes.tests.models.glm5_configs import resolve_model_config


@pytest.mark.parametrize("name,shape,sources", [
    ("debugmodel", (2048, 256), tuple(range(8))),
    ("glm5_2_scaled_debugmodel", (32768, 768), (0, 1, 2, 2, 2, 2, 6, 6, 6, 6, 10, 10, 10, 10, 14, 14)),
])
def test_moved_configs_keep_checkpoint_shapes_and_hf_roundtrip(name, shape, sources):
    config = resolve_model_config(name, seq_len=128)
    with torch.device("meta"):
        model = config.build()
    state = model.state_dict()
    # These shapes and sharing groups are the pre-migration checkpoint ABI.
    assert state["tok_embeddings.weight"].shape == shape
    assert state["lm_head.weight"].shape == shape
    assert config.index_sources == sources
    assert len(model.layers) == len(sources)
    layer_shapes = (
        {"wq_a": (128, 256), "wq_b": (512, 128), "wkv_a": (96, 256),
         "inner_attention.wkv_b": (768, 64), "wo": (256, 512),
         "indexer.weights_proj": (4, 256)}
        if name == "debugmodel" else
        {"wq_a": (384, 768), "wq_b": (2048, 384), "wkv_a": (256, 768),
         "inner_attention.wkv_b": (3584, 192), "wo": (768, 2048),
         "indexer.weights_proj": (8, 768)}
    )
    for projection, expected_shape in layer_shapes.items():
        assert state[f"layers.0.attention.{projection}.weight"].shape == expected_shape
    adapter = Glm5StateDictAdapter(config, hf_assets_path=None)
    restored = adapter.from_hf(adapter.to_hf(state))
    model.load_state_dict(restored, strict=True)
    assert {key: value.shape for key, value in restored.items()} == {
        key: value.shape for key, value in state.items()
    }
    recipe = (glm5_debugmodel(flavor=name, seq_len=128) if name == "debugmodel"
              else glm5_2_scaled_debugmodel(seq_len=128))
    with torch.device("meta"):
        recipe.model.build().load_state_dict(state, strict=True)
    with pytest.raises(ValueError, match="Unknown glm5 flavor"):
        build_model_config(name, parallelism=ParallelismConfig())


def test_reference_config_retains_historical_default_context():
    # Parity serializes this field into its scenario identity even for T=128.
    reference = resolve_model_config("debugmodel")
    assert reference.max_context_length == 2048
    assert reference.layers[0].attention.rope.max_context_length == 2048
    assert resolve_model_config("glm5_2_scaled_debugmodel").max_context_length == 4096

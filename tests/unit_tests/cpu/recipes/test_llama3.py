# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan_recipes.models import llama3 as recipes


def test_activation_offload_recipe_keeps_the_last_block_on_gpu():
    config = recipes.llama3_8b_activation_offload()
    remat = config.activation_remat
    num_layers = len(config.model.layers)
    assert remat.block_policy("layers.0").offloads_activations
    assert remat.block_policy(f"layers.{num_layers - 2}").offloads_activations
    assert not remat.block_policy(f"layers.{num_layers - 1}").offloads_activations

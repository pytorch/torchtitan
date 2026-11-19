# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from torchtitan_recipes.models import llama3 as recipes


def test_activation_offload_recipe_offloads_every_block():
    # The offloader itself keeps the block that runs last on GPU.
    config = recipes.llama3_8b_activation_offload()
    remat = config.activation_remat
    for i in range(len(config.model.layers)):
        assert remat.block_policy(f"layers.{i}").offloads_activations

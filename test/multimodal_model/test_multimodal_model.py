# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from torchtitan.models.llama_multimodal import ModelArgs, VisionEncoder

from test.multimodal_model.test_utils import fixed_init_model, fixed_init_tensor


@pytest.fixture
def model_config():
    return ModelArgs(
        dim=32,
        num_layers=2,
        num_heads=4,
        tile_size=49,
        patch_size=9,
        max_num_tiles=4,
        in_channels=3,
        return_intermediates=[0, 1],
        num_layers_learnable_head=2,
        decoder_embed_dim=128,
    )


class TestMultimodalModelVisionEncoder:
    @pytest.fixture(autouse=True)
    def setup_class(self, model_config):
        self.model_args = model_config
        self.batch_size = 1
        self.num_imgs = 2
        self.num_tiles = 4
        self.aspect_ratio = torch.tensor([[1, 3], [2, 2]]).reshape(
            self.batch_size, self.num_imgs, 2
        )
        image = torch.rand(
            (
                self.batch_size,
                self.num_imgs,
                self.num_tiles,
                self.model_args.in_channels,
                self.model_args.tile_size,
                self.model_args.tile_size,
            )
        )
        self.image = fixed_init_tensor(image.shape, min_val=-1, max_val=1)

    def test_llama_mm_vision_encoder(self):
        model = VisionEncoder(self.model_args)
        fixed_init_model(model, min_val=-1, max_val=1)
        # call model
        output = model(self.image, self.aspect_ratio)

        # assertion
        expected_shape = (
            self.batch_size,
            self.num_imgs * self.num_tiles * (model.vit.patches_per_tile + 1),
            self.model_args.decoder_embed_dim,
        )
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, but got {output.shape}"

        # TODO: Need to ensure numerical stability before doing convergence test.
        # output.mean() = 3.994, we need to debug why it is not close to 5.28800, which is
        # the test value from the original torch tune test
        # assert torch.allclose(
        #     output.mean(), torch.tensor(5.28800), atol=1e-3, rtol=1e-3
        # )

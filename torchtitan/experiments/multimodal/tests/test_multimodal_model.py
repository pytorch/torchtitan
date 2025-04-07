# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch

from torchtitan.experiments.llama_multimodal import (
    ModelArgs,
    MultimodalDecoder,
    VisionEncoder,
)

from .test_utils import fixed_init_model, fixed_init_tensor


@pytest.fixture
def encoder_config():
    return ModelArgs(
        encoder_embed_dim=32,
        encoder_num_layers=2,
        encoder_num_heads=4,
        tile_size=49,
        patch_size=9,
        max_num_tiles=4,
        in_channels=3,
        return_intermediates=[0, 1],
        num_layers_projection=2,
        decoder_embed_dim=128,
    )


@pytest.fixture
def decoder_config():
    return ModelArgs(
        decoder_embed_dim=512,
        vocab_size=10000,
        fusion_interval=2,
        num_special_tokens=3,
        decoder_num_layers=6,
        decoder_num_heads=8,
        decoder_num_kv_heads=4,
        max_seq_len=512,
        rope_theta=50000.0,
    )


class TestMultimodalModelVisionEncoder:
    @pytest.fixture(autouse=True)
    def setup_class(self, encoder_config):
        self.model_args = encoder_config
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
        output = model(self.image, self.aspect_ratio)
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


class TestMultimodalModelDecoder:
    @pytest.fixture(autouse=True)
    def setup_class(self, decoder_config):
        self.model_args = decoder_config
        self.batch_size = 1
        self.decoder_embed_dim = self.model_args.decoder_embed_dim
        self.vocab_size = self.model_args.vocab_size
        self.seq_len = 128
        self.input = {
            "tokens": torch.arange(self.batch_size * self.seq_len).reshape(
                self.batch_size, self.seq_len
            ),
            "encoder_input": fixed_init_tensor(
                (self.batch_size, self.seq_len, self.decoder_embed_dim),
                min_val=-1,
                max_val=1,
            ),
            "encoder_mask": None,
        }

    @torch.no_grad()
    def test_llama_mm_decoder(self):
        model = MultimodalDecoder(self.model_args)
        fixed_init_model(model, min_val=-1, max_val=1)
        output = model(**self.input)
        expected_shape = (self.batch_size, self.seq_len, self.vocab_size)
        assert (
            output.shape == expected_shape
        ), f"Expected shape {expected_shape}, but got {output.shape}"

        # TODO: Need to ensure numerical stability before doing convergence test.
        # output.mean() = -0.0134, we need to debug why it is not close to -9.47548e-5, which is
        # the test value from the original torch tune test
        # assert torch.allclose(
        #     output.mean(), torch.tensor(-9.47548e-5), atol=1e-3, rtol=1e-3
        # )

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.muse_glimmer import model_registry


class TestMuseGlimmerConditionalVision(unittest.TestCase):
    def test_image_free_path_gives_all_vision_parameters_zero_gradients(self):
        def cpu_flex_attention(q, k, v, **kwargs):
            return (
                F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    scale=kwargs["scale"],
                    enable_gqa=kwargs["enable_gqa"],
                ),
                SimpleNamespace(),
            )

        model = model_registry("debugmodel_mm", seq_len=8).build()
        model.init_states()
        hidden_TD = torch.randn(4, model.config.dim, requires_grad=True)
        encoder_outputs = []
        assert model.vision_encoder is not None
        with patch.object(
            FlexInnerAttention,
            "compiled_flex_attn",
            cpu_flex_attention,
        ):
            handle = model.vision_encoder.register_forward_hook(
                lambda _module, _args, output: encoder_outputs.append(output)
            )
            try:
                result_TD = model._prepare_multimodal_embeds(
                    hidden_TD,
                    pixel_values=None,
                    grid_thw=None,
                    vision_bank_indices_T=None,
                )
                result_TD.sum().backward()
            finally:
                handle.remove()

        torch.testing.assert_close(result_TD, hidden_TD, rtol=0, atol=0)
        self.assertEqual(len(encoder_outputs), 1)
        self.assertEqual(encoder_outputs[0].shape, (1, model.vision_encoder.output_dim))
        self.assertTrue(encoder_outputs[0].requires_grad)

        assert model.vision_adapter is not None
        assert model.vision_projection is not None
        assert model.perception_emb_norm is not None
        for module in (
            model.vision_encoder,
            model.vision_adapter,
            model.vision_projection,
            model.perception_emb_norm,
        ):
            for parameter in module.parameters():
                self.assertIsNotNone(parameter.grad)
                torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter))


if __name__ == "__main__":
    unittest.main()

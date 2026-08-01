# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the model-agnostic vision<->text fusion helpers."""

import unittest

import torch

from torchtitan.models.common.multimodal import scatter_vision_embeds


class TestScatterVisionEmbeds(unittest.TestCase):
    def test_scatter_routes_gradients_to_both_streams(self):
        # The scatter is in place, so the graph has to be built from tensors
        # that are not themselves leaves requiring grad.
        inputs_source = torch.arange(30.0).view(2, 5, 3).requires_grad_()
        vision_embeds = torch.arange(12.0).view(2, 2, 3).requires_grad_()
        inputs_embeds = inputs_source * 1.0
        inputs_before = inputs_source.detach().clone()

        actual = scatter_vision_embeds(
            inputs_embeds,
            vision_embeds=vision_embeds,
            vision_positions=[
                (0, 0, 1, 1),
                (1, 1, 2, 2),
            ],
        )
        expected = torch.stack(
            (
                torch.cat(
                    (
                        inputs_before[0, :1],
                        vision_embeds.detach()[0, :1],
                        inputs_before[0, 2:],
                    )
                ),
                torch.cat(
                    (
                        inputs_before[1, :2],
                        vision_embeds.detach()[1],
                        inputs_before[1, 4:],
                    )
                ),
            )
        )

        torch.testing.assert_close(actual, expected)

        actual.sum().backward()
        # Overwritten positions must not propagate to the text embeddings, and
        # every vision token that was scattered must receive gradient.
        expected_input_grad = torch.tensor(
            [[1, 0, 1, 1, 1], [1, 1, 0, 0, 1]],
            dtype=inputs_source.dtype,
        ).unsqueeze(-1)
        expected_vision_grad = torch.tensor(
            [[1, 0], [1, 1]],
            dtype=vision_embeds.dtype,
        ).unsqueeze(-1)
        torch.testing.assert_close(
            inputs_source.grad,
            expected_input_grad.expand_as(inputs_source),
        )
        torch.testing.assert_close(
            vision_embeds.grad,
            expected_vision_grad.expand_as(vision_embeds),
        )


if __name__ == "__main__":
    unittest.main()

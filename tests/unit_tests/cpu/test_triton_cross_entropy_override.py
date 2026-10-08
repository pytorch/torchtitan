# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from unittest import TestCase

from torchtitan.components.loss import CrossEntropyLoss

from torchtitan_recipes.overrides.triton_cross_entropy import (
    triton_cross_entropy,
    TritonCrossEntropyLoss,
)


class TestTritonCrossEntropyOverride(TestCase):
    def test_preserves_cross_entropy_config(self) -> None:
        config = CrossEntropyLoss.Config(global_vocab_size=128256)

        overridden = triton_cross_entropy(config)

        self.assertIsInstance(overridden, TritonCrossEntropyLoss.Config)
        self.assertEqual(overridden.global_vocab_size, config.global_vocab_size)

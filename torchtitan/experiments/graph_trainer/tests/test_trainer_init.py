#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch
import torch.nn as nn

from torchtitan.experiments.graph_trainer.trainer import GraphTrainer


class _FlexShardLikeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self._dstorages = [object()]
        self.param = nn.Parameter(torch.ones(1, device="cpu"))
        self.register_buffer("cache", torch.zeros(1, device="cpu"))
        self.seen_buffer_device = None

    def init_weights(self, *, buffer_device=None):
        self.seen_buffer_device = buffer_device
        device = buffer_device or self.cache.device
        self.cache = torch.ones(1, device=device)


class TestGraphTrainerInit(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_flex_shard_init_moves_buffers_without_model_to(self):
        trainer = GraphTrainer.__new__(GraphTrainer)
        model = _FlexShardLikeModel()

        trainer._init_model_weights(model, "cuda", None)

        self.assertEqual(model.seen_buffer_device, torch.device("cuda"))
        self.assertEqual(model.cache.device.type, "cuda")
        self.assertEqual(model.param.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()

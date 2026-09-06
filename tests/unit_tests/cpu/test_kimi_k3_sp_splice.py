# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The multimodal splice on a sequence-parallel shard, through the real
collective on a one-rank gloo group (the gather is an identity there, so the
whole sequence it returns must equal the whole-sequence splice)."""

import unittest

import torch
import torch.distributed as dist

from torchtitan.models.common.multimodal import (
    get_vision_positions,
    scatter_vision_embeds,
)
from torchtitan.models.kimi_k3.model import _splice_under_sequence_parallel


class TestSequenceParallelSplice(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._owns_pg = not dist.is_initialized()
        if cls._owns_pg:
            dist.init_process_group(
                backend="gloo",
                init_method="tcp://localhost:12363",
                world_size=1,
                rank=0,
            )

    @classmethod
    def tearDownClass(cls):
        if cls._owns_pg and dist.is_initialized():
            dist.destroy_process_group()

    def test_matches_the_whole_sequence_splice_and_carries_the_gradient(self):
        T, D, image_id = 12, 8, 7
        tokens = torch.tensor([1, 2, 7, 7, 7, 7, 3, 7, 7, 4, 5, 6])
        num_tokens_per_item = torch.tensor([4, 2])
        embeddings = torch.randn(T, D, requires_grad=True)
        vision = torch.randn(6, D, requires_grad=True)

        group = dist.group.WORLD
        assert group is not None
        out = _splice_under_sequence_parallel(
            embeddings,
            tokens,
            vision_embeds=vision,
            num_tokens_per_item=num_tokens_per_item,
            image_id=image_id,
            group=group,
        )
        reference = scatter_vision_embeds(
            embeddings.detach().clone(),
            vision_embeds=vision.detach(),
            vision_positions=get_vision_positions(
                tokens, num_tokens_per_item, image_id
            ),
        )
        self.assertTrue(torch.equal(out, reference))

        out.sum().backward()
        assert embeddings.grad is not None and vision.grad is not None
        # Text rows keep their gradient, placeholder rows hand it to the tower.
        placeholder = tokens == image_id
        self.assertTrue(torch.equal(embeddings.grad[~placeholder], torch.ones(6, D)))
        self.assertTrue(torch.equal(embeddings.grad[placeholder], torch.zeros(6, D)))
        self.assertTrue(torch.equal(vision.grad, torch.ones(6, D)))


if __name__ == "__main__":
    unittest.main()

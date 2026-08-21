# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CP shard selection for visual features, on a NON-identity partition.

Every CP rank encodes every image (``prepare_context_parallel_input`` shards the
sequence but leaves ``pixel_values`` whole) while holding only a slice of the
sentinels, so each rank has to keep its own contiguous slice of the features. The
configurations this was exercised on until now gave a partition where the slice
happened to be the whole thing, which is a test that cannot fail -- these pin the
cases where it can: an uneven split, a rank holding nothing, and a partition that
disagrees with the encode.

``_select_cp_shard`` only reads ``self._cp_group``, and only to ask its rank, so a
stub plus a patched ``get_rank`` covers it on CPU with no process group.
"""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from torchtitan.models.kimi_k3.multimodal_model import KimiK3MultimodalModel


_SELECT = KimiK3MultimodalModel._select_cp_shard


def select(features, num_rows, counts, *, rank):
    stub = SimpleNamespace(_cp_group=object())
    with mock.patch("torch.distributed.get_rank", return_value=rank):
        return _SELECT(stub, features, num_rows, counts)


class TestSelectCPShard(unittest.TestCase):
    def test_uneven_split_gives_each_rank_its_own_contiguous_slice(self):
        features = torch.arange(8, dtype=torch.float32).unsqueeze(1)
        counts = torch.tensor([3, 5])
        got0 = select(features, 8, counts, rank=0)
        got1 = select(features, 8, counts, rank=1)
        self.assertEqual(got0.flatten().tolist(), [0.0, 1.0, 2.0])
        self.assertEqual(got1.flatten().tolist(), [3.0, 4.0, 5.0, 6.0, 7.0])
        # Together they reconstruct the encode exactly once, in order.
        self.assertEqual(
            torch.cat([got0, got1]).flatten().tolist(), features.flatten().tolist()
        )

    def test_a_rank_holding_no_sentinels_gets_nothing(self):
        """The case the missing call got wrong. Without the selection this rank
        splices ALL the features into a shard with no sentinel positions."""
        features = torch.arange(6, dtype=torch.float32).unsqueeze(1)
        got = select(features, 6, torch.tensor([6, 0]), rank=1)
        self.assertEqual(got.shape[0], 0)

    def test_a_list_of_per_image_features_is_concatenated_in_order(self):
        features = [torch.full((2, 1), 1.0), torch.full((3, 1), 2.0)]
        got = select(features, 5, torch.tensor([1, 4]), rank=1)
        self.assertEqual(got.flatten().tolist(), [1.0, 2.0, 2.0, 2.0])

    def test_middle_rank_of_three_starts_after_the_lower_ranks(self):
        features = torch.arange(9, dtype=torch.float32).unsqueeze(1)
        got = select(features, 9, torch.tensor([2, 4, 3]), rank=1)
        self.assertEqual(got.flatten().tolist(), [2.0, 3.0, 4.0, 5.0])

    def test_counts_that_disagree_with_the_encode_raise(self):
        features = torch.zeros(8, 1)
        with self.assertRaises(ValueError) as caught:
            select(features, 8, torch.tensor([3, 4]), rank=0)
        self.assertIn("disagree", str(caught.exception))

    def test_no_counts_means_cp_is_off_and_nothing_is_dropped(self):
        features = torch.zeros(8, 1)
        self.assertIs(select(features, 8, None, rank=0), features)


if __name__ == "__main__":
    unittest.main()

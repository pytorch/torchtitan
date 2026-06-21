# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchtitan.models.common.attention import create_attention_mask, get_causal_mask_mod


class TestCreateAttentionMask(unittest.TestCase):
    def test_separate_full_blocks_kwarg_is_runtime_compatible(self):
        mask = create_attention_mask(
            get_causal_mask_mod(),
            1,
            None,
            4,
            4,
            device="cpu",
            BLOCK_SIZE=2,
            separate_full_blocks=True,
        )

        self.assertEqual(mask.shape, (1, 1, 4, 4))


if __name__ == "__main__":
    unittest.main()

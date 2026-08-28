# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch

from torchtitan.models.flux import flux_configs
from torchtitan.models.flux.model.model import FluxModel
from torchtitan.models.utils import quadratic_attention_flops_per_token


def _expected_attention_flops(config, seq_len: int) -> int:
    """Attention FLOPs per token on the convention the other models share."""
    total = 0
    for block, depth in (
        (config.single_blocks[0], config.depth_single_blocks),
        (config.double_blocks[0], config.depth),
    ):
        head_dim = block.hidden_size // block.num_heads
        total += (
            quadratic_attention_flops_per_token(
                num_heads=block.num_heads,
                qk_head_dim=head_dim,
                v_head_dim=head_dim,
                seq_len=seq_len,
            )
            * depth
        )
    return total


class TestFluxFlops(unittest.TestCase):
    def test_flops_per_token_matches_the_documented_decomposition(self):
        """Rebuild the value from the terms the docstring names. Counting a
        single head_dim for attention instead of (qk_head_dim + v_head_dim)
        halves the attention term and shows up here."""
        seq_len = 512
        config = flux_configs["flux-debug"]()
        with torch.device("meta"):
            model = FluxModel(config)

        nparams, flops_per_token = config.get_nparams_and_flops(model, seq_len)

        db_h = config.double_blocks[0].hidden_size
        db_r = config.double_blocks[0].mlp_ratio
        sb_h = config.single_blocks[0].hidden_size
        fl_h = config.final_layer_config.hidden_size

        # one side of each symmetric double-stream block is not on a token's path
        one_side = int(db_h * db_h * (4 + 2 * db_r))
        # modulation runs once per sample, not per token
        per_sample_mod = (
            12 * db_h * db_h * config.depth
            + 3 * sb_h * sb_h * config.depth_single_blocks
            + 2 * fl_h * fl_h
        )

        expected = (
            6 * nparams
            - 6 * one_side * config.depth
            - 6 * per_sample_mod * (seq_len - 1) // seq_len
            + _expected_attention_flops(config, seq_len)
        )

        self.assertEqual(flops_per_token, expected)


if __name__ == "__main__":
    unittest.main()

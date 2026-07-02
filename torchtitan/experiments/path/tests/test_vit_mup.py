# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import re
import unittest

from torchtitan.experiments.path.config_registry import (
    _vit_model_config,
    _vit_optimizer_config,
    MUP_PATTERN,
    VIT_BASE_WIDTH,
    VIT_HEAD_DIM,
    VIT_WIDTHS,
)


class TestViTMuPParamGroups(unittest.TestCase):
    def test_mup_group_scales_lr_and_wd_across_widths(self):
        """muP scales the hidden-matmul group by 1/m and its wd by m; catch-all unscaled."""
        base_lr, base_wd = 3e-4, 0.0125
        for flavor, width in (("w256", 256), ("w512", 512)):
            config = _vit_optimizer_config(flavor, mup=True, lr=base_lr, wd=base_wd)
            mup_group, catch_all = config.param_groups[0], config.param_groups[-1]

            self.assertEqual(mup_group.pattern, MUP_PATTERN)
            self.assertAlmostEqual(mup_group.lr_mult, VIT_BASE_WIDTH / width)
            self.assertAlmostEqual(
                mup_group.optimizer_kwargs["weight_decay"],
                base_wd * width / VIT_BASE_WIDTH,
            )

            self.assertEqual(catch_all.pattern, r".*")
            self.assertAlmostEqual(catch_all.lr_mult, 1.0)
            self.assertAlmostEqual(catch_all.optimizer_kwargs["weight_decay"], base_wd)

    def test_mup_pattern_matches_hidden_matmuls_exactly(self):
        """MUP_PATTERN selects exactly the per-block attention/mlp matmul weights."""
        model = _vit_model_config("w256", mup=True).build()
        param_names = {name for name, _ in model.named_parameters()}
        expected = {
            f"blocks.{i}.{submodule}.{leaf}.weight"
            for i in range(len(model.blocks))
            for submodule, leaf in (
                ("attention", "c_attn"),
                ("attention", "c_proj"),
                ("mlp", "c_fc"),
                ("mlp", "c_proj"),
            )
        }
        self.assertEqual(len(expected), 4 * len(model.blocks))
        self.assertTrue(
            expected <= param_names,
            f"hidden matmul weights missing from model: {expected - param_names}",
        )

        matched = {name for name in param_names if re.search(MUP_PATTERN, name)}
        self.assertEqual(matched, expected)

    def test_widths_are_head_dim_multiples(self):
        for flavor in VIT_WIDTHS:
            _vit_model_config(flavor, mup=True)
        VIT_WIDTHS["w100"] = 100
        try:
            with self.assertRaises(ValueError):
                _vit_model_config("w100", mup=True)
        finally:
            del VIT_WIDTHS["w100"]
        self.assertTrue(all(d % VIT_HEAD_DIM == 0 for d in VIT_WIDTHS.values()))


if __name__ == "__main__":
    unittest.main()

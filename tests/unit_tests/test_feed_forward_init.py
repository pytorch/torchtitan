# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from functools import partial

import torch
import torch.nn as nn

from torchtitan.models.common.config_utils import make_ffn_config


class TestFeedForwardInit(unittest.TestCase):
    """Weight-init wiring for the shared SwiGLU FeedForward (``make_ffn_config``)."""

    def _build(self, w1w3_init, w2_init):
        cfg = make_ffn_config(
            dim=64,
            hidden_dim=128,
            w1w3_param_init=w1w3_init,
            w2_param_init=w2_init,
        )
        ff = cfg.build()
        ff.to_empty(device="cpu")
        ff.init_states()
        return ff

    def test_w1_and_w3_share_input_init_not_w2(self):
        """w1 and w3 use ``w1w3_param_init``; only w2 gets ``w2_param_init``."""
        wide = {"weight": partial(nn.init.trunc_normal_, std=0.02, a=-0.06, b=0.06)}
        narrow = {
            "weight": partial(nn.init.trunc_normal_, std=0.005, a=-0.015, b=0.015)
        }

        torch.manual_seed(0)
        ff = self._build(w1w3_init=wide, w2_init=narrow)

        # w1/w3 follow the wide bound, not the narrow (depth-scaled) w2 bound.
        for name in ("w1", "w3"):
            w = getattr(ff, name).weight
            self.assertLessEqual(w.abs().max().item(), 0.06 + 1e-6)
            self.assertGreater(w.abs().max().item(), 0.015)

        self.assertLessEqual(ff.w2.weight.abs().max().item(), 0.015 + 1e-6)

    def test_truncation_bounds_are_effective(self):
        """Explicit +-3 sigma bounds actually clip samples (default a/b do not)."""
        std = 0.02
        init = {
            "weight": partial(nn.init.trunc_normal_, std=std, a=-3 * std, b=3 * std)
        }
        torch.manual_seed(0)
        ff = self._build(w1w3_init=init, w2_init=init)
        for name in ("w1", "w2", "w3"):
            self.assertLessEqual(
                getattr(ff, name).weight.abs().max().item(), 3 * std + 1e-6
            )


if __name__ == "__main__":
    unittest.main()

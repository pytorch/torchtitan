# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest
from types import SimpleNamespace

import torch.nn as nn

from torchtitan.config import ParallelismConfig
from torchtitan.models.common.param_init import skip_param_init
from torchtitan.models.llama3 import llama3_configs
from torchtitan.models.llama3.model import Llama3Model


def _make_config(enable_weight_tying: bool = False) -> Llama3Model.Config:
    # Start from the standard debugmodel config and adjust weight tying.
    config = llama3_configs["debugmodel"](attn_backend="flex")
    # Replace tok_embeddings param_init based on weight tying flag.
    import dataclasses
    from functools import partial

    tok_init = (
        {"weight": skip_param_init}
        if enable_weight_tying
        else {"weight": partial(nn.init.normal_, std=1.0)}
    )
    config = dataclasses.replace(
        config,
        enable_weight_tying=enable_weight_tying,
        tok_embeddings=dataclasses.replace(config.tok_embeddings, param_init=tok_init),
    )
    return config


class TestLlama3WeightTying(unittest.TestCase):
    def test_weights_are_shared_when_tying_enabled(self):
        """tok_embeddings.weight and output.weight should share the same storage."""
        model = Llama3Model(_make_config(enable_weight_tying=True))
        self.assertIs(
            model.tok_embeddings.weight,
            model.lm_head.weight,
            "tok_embeddings.weight and output.weight must be the same tensor object",
        )

    def test_weights_are_independent_when_tying_disabled(self):
        """Without weight tying, tok_embeddings and output have separate weights."""
        model = Llama3Model(_make_config(enable_weight_tying=False))
        self.assertIsNot(
            model.tok_embeddings.weight,
            model.lm_head.weight,
            "tok_embeddings.weight and output.weight must be distinct tensor objects",
        )

    def test_weights_remain_tied_after_init_states(self):
        """Weights must still be shared after calling init_states."""
        config = _make_config(enable_weight_tying=True)
        model = Llama3Model(config)
        model.init_states()
        self.assertIs(
            model.tok_embeddings.weight,
            model.lm_head.weight,
            "tok_embeddings.weight and output.weight must remain tied after init_states",
        )

    def test_tied_parameter_count_matches_unique_parameters(self):
        """Tied embeddings should be counted once, not subtracted away."""
        config = _make_config(enable_weight_tying=True)
        model = Llama3Model(config)
        unique_param_count = sum(
            p.numel() for p in {id(p): p for p in model.parameters()}.values()
        )
        reported_param_count, _ = config.get_nparams_and_flops(model, seq_len=512)

        self.assertEqual(reported_param_count, unique_param_count)

    def test_pp_guard_raises_when_weight_tying_and_pp_enabled(self):
        """update_from_config must raise NotImplementedError when PP > 1 and weight tying is on."""
        config = _make_config(enable_weight_tying=True)

        runtime_config = SimpleNamespace(
            parallelism=ParallelismConfig(pipeline_parallel_degree=2),
            is_inference=False,
        )

        with self.assertRaises(NotImplementedError):
            config.update_from_config(config=runtime_config)

    def test_pp_guard_does_not_raise_without_weight_tying(self):
        """update_from_config must NOT raise when PP > 1 and weight tying is off."""
        config = _make_config(enable_weight_tying=False)

        runtime_config = SimpleNamespace(
            parallelism=ParallelismConfig(pipeline_parallel_degree=2),
            is_inference=False,
        )

        # Should not raise
        config.update_from_config(config=runtime_config)


if __name__ == "__main__":
    unittest.main()

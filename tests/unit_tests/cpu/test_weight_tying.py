# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import torch.nn as nn
from torchtitan.config import TrainingConfig

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.param_init import skip_param_init
from torchtitan.models.llama3 import MODEL_FLAVORS
from torchtitan.models.llama3.model import Llama3Model
from torchtitan.trainer import Trainer


def _make_config(enable_weight_tying: bool = False) -> Llama3Model.Config:
    # Start from the standard debugmodel config and adjust weight tying.
    build_config, max_context_length = MODEL_FLAVORS["debugmodel"]
    config = build_config(attn_backend="flex", seq_len=max_context_length)
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
        """Trainer config rejects weight tying when PP is enabled."""
        config = _make_config(enable_weight_tying=True)

        with self.assertRaises(NotImplementedError):
            Trainer.Config(
                model=config,
                training=TrainingConfig(
                    max_context_length=config.max_context_length,
                    disable_cuda_graphs=True,
                ),
                parallelism=ParallelismConfig(pipeline_parallel_degree=2),
            )

    def test_pp_guard_does_not_raise_without_weight_tying(self):
        """Trainer config allows PP when weight tying is off."""
        config = _make_config(enable_weight_tying=False)

        Trainer.Config(
            model=config,
            training=TrainingConfig(
                max_context_length=config.max_context_length,
                disable_cuda_graphs=True,
            ),
            parallelism=ParallelismConfig(pipeline_parallel_degree=2),
        )


if __name__ == "__main__":
    unittest.main()

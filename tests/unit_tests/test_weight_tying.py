# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import unittest

from torchtitan.models.common.attention import GQAttention
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.feed_forward import compute_ffn_hidden_dim, FeedForward
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.models.common.rope import RoPE
from torchtitan.models.llama3.model import Llama3Model, Llama3TransformerBlock


def _make_config(enable_weight_tying: bool = False):
    return Llama3Model.Config(
        dim=64,
        n_layers=2,
        vocab_size=256,
        enable_weight_tying=enable_weight_tying,
        tok_embeddings=Embedding.Config(),
        norm=RMSNorm.Config(),
        output=Linear.Config(),
        layer=Llama3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(),
            ffn_norm=RMSNorm.Config(),
            feed_forward=FeedForward.Config(
                hidden_dim=compute_ffn_hidden_dim(64, multiple_of=64),
            ),
            attention=GQAttention.Config(
                n_heads=4,
                attn_backend="sdpa",
                rope_backend="complex",
            ),
        ),
        rope=RoPE.Config(
            dim=64 // 4,
            max_seq_len=512,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
    )


class TestLlama3WeightTying(unittest.TestCase):
    def test_weights_are_shared_when_tying_enabled(self):
        """tok_embeddings.weight and output.weight should share the same storage."""
        model = Llama3Model(_make_config(enable_weight_tying=True))
        self.assertIs(
            model.tok_embeddings.weight,
            model.output.weight,
            "tok_embeddings.weight and output.weight must be the same tensor object",
        )

    def test_weights_are_independent_when_tying_disabled(self):
        """Without weight tying, tok_embeddings and output have separate weights."""
        model = Llama3Model(_make_config(enable_weight_tying=False))
        self.assertIsNot(
            model.tok_embeddings.weight,
            model.output.weight,
            "tok_embeddings.weight and output.weight must be distinct tensor objects",
        )

    def test_weights_remain_tied_after_init_weights(self):
        """Weights must still be shared after calling init_weights."""
        model = Llama3Model(_make_config(enable_weight_tying=True))
        model.init_weights()
        self.assertIs(
            model.tok_embeddings.weight,
            model.output.weight,
            "tok_embeddings.weight and output.weight must remain tied after init_weights",
        )

    def test_pp_guard_raises_when_weight_tying_and_pp_enabled(self):
        """update_from_config must raise NotImplementedError when PP > 1 and weight tying is on."""
        from unittest.mock import MagicMock

        config = _make_config(enable_weight_tying=True)

        trainer_config = MagicMock()
        trainer_config.training.seq_len = 512
        trainer_config.parallelism.pipeline_parallel_degree = 2
        trainer_config.parallelism.context_parallel_degree = 1
        trainer_config.parallelism.tensor_parallel_degree = 1

        with self.assertRaises(NotImplementedError):
            config.update_from_config(trainer_config=trainer_config)

    def test_pp_guard_does_not_raise_without_weight_tying(self):
        """update_from_config must NOT raise when PP > 1 and weight tying is off."""
        from unittest.mock import MagicMock

        config = _make_config(enable_weight_tying=False)

        trainer_config = MagicMock()
        trainer_config.training.seq_len = 512
        trainer_config.parallelism.pipeline_parallel_degree = 2
        trainer_config.parallelism.context_parallel_degree = 1
        trainer_config.parallelism.tensor_parallel_degree = 1

        # Should not raise
        config.update_from_config(trainer_config=trainer_config)


if __name__ == "__main__":
    unittest.main()

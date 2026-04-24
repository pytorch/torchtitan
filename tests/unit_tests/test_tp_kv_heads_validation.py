# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for TP-degree / n_kv_heads divisibility validation in model configs.

Covers Issue #2574: models with GQA (n_kv_heads < n_heads) would crash deep in
the forward pass when tensor_parallel_degree > n_kv_heads. The fix adds an
early ValueError in update_from_config for llama3, llama4, qwen3, and gpt_oss.
"""

import unittest
from types import SimpleNamespace

try:
    import sys
    from unittest.mock import MagicMock

    # triton is Linux-only; stub it so MoE kernel imports don't crash on Windows
    if "triton" not in sys.modules:
        sys.modules["triton"] = MagicMock()
        sys.modules["triton.language"] = MagicMock()

    from torchtitan.models.common import (
        compute_ffn_hidden_dim,
        Embedding,
        Linear,
        RMSNorm,
        RoPE,
    )
    from torchtitan.models.llama3 import Llama3Model, Llama3TransformerBlock

    _IMPORTS_OK = True
except Exception:
    _IMPORTS_OK = False

_DIM = 256
_N_LAYERS = 2
_VOCAB_SIZE = 2048


def _make_trainer_config(tp: int, seq_len: int = 2048):
    """Minimal trainer_config stub with just the fields update_from_config reads."""
    training = SimpleNamespace(seq_len=seq_len)
    parallelism = SimpleNamespace(
        tensor_parallel_degree=tp,
        context_parallel_degree=1,
        pipeline_parallel_degree=1,
    )
    return SimpleNamespace(training=training, parallelism=parallelism)


def _make_llama3_config(n_heads: int, n_kv_heads: int | None) -> "Llama3Model.Config":
    """Build a minimal Llama3Model.Config with the given head counts."""
    from torchtitan.models.common.attention import ScaledDotProductAttention
    from torchtitan.models.common.config_utils import make_ffn_config, make_gqa_config

    _LINEAR_INIT = {"weight": lambda t: t}
    _NORM_INIT = {"weight": lambda t: t}

    layers = []
    for layer_id in range(_N_LAYERS):
        layers.append(
            Llama3TransformerBlock.Config(
                attention_norm=RMSNorm.Config(
                    normalized_shape=_DIM, param_init=_NORM_INIT
                ),
                ffn_norm=RMSNorm.Config(normalized_shape=_DIM, param_init=_NORM_INIT),
                attention=make_gqa_config(
                    dim=_DIM,
                    n_heads=n_heads,
                    n_kv_heads=n_kv_heads,
                    wqkv_param_init=_LINEAR_INIT,
                    wo_param_init=_LINEAR_INIT,
                    inner_attention=ScaledDotProductAttention.Config(),
                    rope_backend="complex",
                ),
                feed_forward=make_ffn_config(
                    dim=_DIM,
                    hidden_dim=compute_ffn_hidden_dim(_DIM, multiple_of=256),
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_LINEAR_INIT,
                ),
            )
        )

    return Llama3Model.Config(
        dim=_DIM,
        vocab_size=_VOCAB_SIZE,
        tok_embeddings=Embedding.Config(num_embeddings=_VOCAB_SIZE, embedding_dim=_DIM),
        norm=RMSNorm.Config(normalized_shape=_DIM),
        output=Linear.Config(in_features=_DIM, out_features=_VOCAB_SIZE),
        rope=RoPE.Config(
            dim=_DIM // n_heads,
            max_seq_len=4096,
            theta=500000,
            backend="complex",
            scaling="llama",
        ),
        layers=layers,
    )


@unittest.skipUnless(_IMPORTS_OK, "torchtitan model imports not available")
class TestTPKVHeadsValidation(unittest.TestCase):
    """Validate that update_from_config rejects configs where n_heads or
    n_kv_heads are not divisible by tensor_parallel_degree."""

    # ------------------------------------------------------------------
    # n_kv_heads not divisible by TP  →  should raise
    # ------------------------------------------------------------------

    def test_llama3_kv_heads_not_divisible_raises(self):
        """n_kv_heads=2, tp=4 → fractional KV heads per rank → ValueError."""
        cfg = _make_llama3_config(n_heads=8, n_kv_heads=2)
        with self.assertRaises(ValueError):
            cfg.update_from_config(trainer_config=_make_trainer_config(tp=4))

    # ------------------------------------------------------------------
    # n_heads not divisible by TP  →  should raise
    # ------------------------------------------------------------------

    def test_llama3_n_heads_not_divisible_raises(self):
        """n_heads=6, tp=4 → fractional Q heads per rank → ValueError."""
        cfg = _make_llama3_config(n_heads=6, n_kv_heads=6)
        with self.assertRaises(ValueError):
            cfg.update_from_config(trainer_config=_make_trainer_config(tp=4))

    # ------------------------------------------------------------------
    # Valid configs  →  should not raise
    # ------------------------------------------------------------------

    def test_llama3_valid_gqa_does_not_raise(self):
        """n_kv_heads=8, n_heads=16, tp=4 → both divisible → no error."""
        cfg = _make_llama3_config(n_heads=16, n_kv_heads=8)
        cfg.update_from_config(trainer_config=_make_trainer_config(tp=4))

    def test_llama3_mha_none_kv_heads_does_not_raise(self):
        """n_kv_heads=None (MHA, falls back to n_heads=16), tp=4 → no error."""
        cfg = _make_llama3_config(n_heads=16, n_kv_heads=None)
        cfg.update_from_config(trainer_config=_make_trainer_config(tp=4))

    def test_llama3_tp1_skips_check(self):
        """tp=1 → check skipped even with indivisible n_kv_heads."""
        cfg = _make_llama3_config(n_heads=8, n_kv_heads=3)
        cfg.update_from_config(trainer_config=_make_trainer_config(tp=1))


if __name__ == "__main__":
    unittest.main()

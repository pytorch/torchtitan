# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Unit tests for TP-degree / attention-head divisibility validation.

Covers Issue #2574: models with GQA (n_kv_heads < n_heads) would crash deep in
the forward pass when tensor_parallel_degree > n_kv_heads. Validation rejects
such configs early unless the attention uses ``PaddedQKVLinear``, which pads
its heads up to a multiple of the TP degree when built. The padding itself is
tested in test_tp_padding.py.
"""

import unittest
from dataclasses import dataclass
from functools import partial

import torch.nn as nn
from torchtitan.config import DebugConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import convert_config_type
from torchtitan.config.validation import validate_model_training_config
from torchtitan.models.common import compute_ffn_hidden_dim, Embedding, Linear, RMSNorm
from torchtitan.models.common.attention import (
    GQAttention,
    ScaledDotProductInnerAttention,
    validate_tp_head_sharding,
)
from torchtitan.models.common.config_utils import make_ffn_config, make_gqa_config
from torchtitan.models.llama3 import Llama3Model
from torchtitan.models.llama3.model import Llama3TransformerBlock

_DIM = 64
_HEAD_DIM = 8
_N_LAYERS = 2
_VOCAB_SIZE = 2048
_INIT = {"weight": partial(nn.init.trunc_normal_, std=0.02)}


class _SubclassedAttention(GQAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(GQAttention.Config):
        pass


def _attention_config(
    n_heads: int, n_kv_heads: int | None, *, pad_heads_for_tp: bool = False
) -> GQAttention.Config:
    return make_gqa_config(
        dim=_DIM,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=_HEAD_DIM,
        wqkv_param_init=_INIT,
        wo_param_init=_INIT,
        inner_attention=ScaledDotProductInnerAttention.Config(),
        rope=None,
        pad_heads_for_tp=pad_heads_for_tp,
    )


def _make_llama3_config(
    n_heads: int, n_kv_heads: int | None, *, pad_heads_for_tp: bool = False
) -> Llama3Model.Config:
    """Build a minimal Llama3Model.Config with the given head counts."""
    layers = [
        Llama3TransformerBlock.Config(
            attention_norm=RMSNorm.Config(normalized_shape=_DIM),
            ffn_norm=RMSNorm.Config(normalized_shape=_DIM),
            attention=_attention_config(
                n_heads, n_kv_heads, pad_heads_for_tp=pad_heads_for_tp
            ),
            feed_forward=make_ffn_config(
                dim=_DIM,
                hidden_dim=compute_ffn_hidden_dim(_DIM, multiple_of=256),
                w13_param_init=_INIT,
                w2_param_init=_INIT,
            ),
        )
        for _ in range(_N_LAYERS)
    ]
    return Llama3Model.Config(
        max_context_length=4096,
        dim=_DIM,
        vocab_size=_VOCAB_SIZE,
        tok_embeddings=Embedding.Config(num_embeddings=_VOCAB_SIZE, embedding_dim=_DIM),
        norm=RMSNorm.Config(normalized_shape=_DIM),
        lm_head=Linear.Config(in_features=_DIM, out_features=_VOCAB_SIZE),
        layers=layers,
    )


def _validate(config: Llama3Model.Config, tp: int) -> None:
    validate_model_training_config(
        config,
        parallelism=ParallelismConfig(tensor_parallel_degree=tp),
        training=TrainingConfig(
            max_context_length=config.max_context_length,
            disable_cuda_graphs=True,
        ),
        debug=DebugConfig(),
        activation_checkpoint=None,
        max_num_documents=None,
    )


class TestTPKVHeadsValidation(unittest.TestCase):
    """Validate that config checking rejects models where n_heads or
    n_kv_heads are not divisible by tensor_parallel_degree, unless the
    attention allows TP head padding."""

    def test_llama3_kv_heads_not_divisible_raises(self):
        """n_kv_heads=2, tp=4 -> fractional KV heads per rank -> ValueError."""
        cfg = _make_llama3_config(n_heads=8, n_kv_heads=2)
        with self.assertRaisesRegex(
            ValueError, "must divide n_kv_heads .* PaddedQKVLinear"
        ):
            _validate(cfg, tp=4)

    def test_llama3_n_heads_not_divisible_raises(self):
        """n_heads=2, tp=4 -> fractional Q heads per rank -> ValueError."""
        cfg = _make_llama3_config(n_heads=2, n_kv_heads=2)
        with self.assertRaisesRegex(ValueError, "must divide n_heads"):
            _validate(cfg, tp=4)

    def test_llama3_padding_allowed_does_not_raise(self):
        """n_kv_heads=2, tp=4, pad_heads_for_tp -> no error."""
        cfg = _make_llama3_config(n_heads=8, n_kv_heads=2, pad_heads_for_tp=True)
        _validate(cfg, tp=4)

    def test_llama3_valid_gqa_does_not_raise(self):
        """n_kv_heads=8, n_heads=16, tp=4 -> both divisible -> no error."""
        _validate(_make_llama3_config(n_heads=16, n_kv_heads=8), tp=4)

    def test_llama3_mha_none_kv_heads_does_not_raise(self):
        """n_kv_heads=None (MHA, falls back to n_heads=16), tp=4 -> no error."""
        _validate(_make_llama3_config(n_heads=16, n_kv_heads=None), tp=4)

    def test_llama3_tp1_skips_check(self):
        """tp=1 -> valid GQA head counts do not raise."""
        _validate(_make_llama3_config(n_heads=8, n_kv_heads=2), tp=1)

    def test_padding_rejects_attention_subclasses(self):
        config = _attention_config(n_heads=6, n_kv_heads=3)
        with self.assertRaisesRegex(
            ValueError, "PaddedQKVLinear supports only GQAttention"
        ):
            convert_config_type(
                _attention_config(n_heads=6, n_kv_heads=3, pad_heads_for_tp=True),
                _SubclassedAttention,
            )
        # Without the flag, subclasses are accepted as before.
        convert_config_type(config, _SubclassedAttention)

    def test_unpaddable_attention_raises_without_padding_hint(self):
        config = convert_config_type(
            _attention_config(n_heads=8, n_kv_heads=2), _SubclassedAttention
        )
        with self.assertRaisesRegex(ValueError, "must divide n_kv_heads") as ctx:
            validate_tp_head_sharding(config, tp=4)
        self.assertNotIn("PaddedQKVLinear", str(ctx.exception))

    def test_mixed_padded_and_unpadded_layers_validate_every_layer(self):
        padded = _attention_config(n_heads=8, n_kv_heads=2, pad_heads_for_tp=True)
        unpadded = _attention_config(n_heads=8, n_kv_heads=2)
        config = _make_llama3_config(n_heads=8, n_kv_heads=2)
        config.layers[0].attention = padded
        config.layers[1].attention = unpadded
        with self.assertRaisesRegex(ValueError, "must divide n_kv_heads"):
            _validate(config, tp=4)


if __name__ == "__main__":
    unittest.main()

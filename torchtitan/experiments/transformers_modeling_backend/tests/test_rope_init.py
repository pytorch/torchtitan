# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests that rotary buffers are populated by the backend's meta-init path.

HF rotary modules compute ``inv_freq`` in ``__init__``, and register it as a
non-persistent buffer -- so it is absent from the state dict and a checkpoint
load cannot restore it. Under the backend's meta init + ``to_empty()``,
``_init_weights`` is the only code that writes real values into it.

A failure here is silent: nothing raises, and RoPE rotates by uninitialized
memory. It has already regressed once that way (#3775 was fixed by #3772
against transformers 4.x, then the fix became dead code when the CI pin moved
to 5.x), so these tests compare against a directly constructed module instead
of relying on any code path in the backend.

Run with:
    python -m pytest torchtitan/experiments/transformers_modeling_backend/tests/test_rope_init.py -x -v
"""

import unittest

import torch
from transformers import LlamaConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.models.llama import modeling_llama
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM,
    LlamaRotaryEmbedding,
)

from torchtitan.experiments.transformers_modeling_backend.model import (
    HFTransformerModel,
)

_LLAMA3_SCALING = {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 256,
}


def _tiny_config(rope_type: str = "default") -> LlamaConfig:
    """A small Llama config with the rope settings grouped under
    ``rope_parameters``, the way transformers 5.x expects them."""
    config = LlamaConfig(
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_hidden_layers=2,
        vocab_size=128,
        max_position_embeddings=256,
    )
    params = {"rope_type": rope_type, "rope_theta": 10000.0}
    if rope_type != "default":
        params.update(_LLAMA3_SCALING)
    config.rope_parameters = params
    return config


class TestRotaryEmbeddingInit(unittest.TestCase):
    def setUp(self):
        # _patch_hf_llama_like rebinds attributes on PreTrainedModel itself, so
        # the patch stays in place after the model that installed it is gone.
        # Save and restore them so it does not affect other tests in the same
        # session.
        self._saved = (
            PreTrainedModel._init_weights,
            PreTrainedModel._initialize_weights,
            modeling_llama.LlamaDecoderLayer.__init__,
        )

    def tearDown(self):
        (
            PreTrainedModel._init_weights,
            PreTrainedModel._initialize_weights,
            modeling_llama.LlamaDecoderLayer.__init__,
        ) = self._saved

    def _materialize(self, config: LlamaConfig) -> LlamaForCausalLM:
        """Reproduce the backend's build path: patch, build on meta, to_empty,
        then run every submodule through the patched ``_init_weights`` the way
        ``HFTransformerModel.init_states`` does."""
        # The patch only rebinds the classes passed to it, not the instance, so
        # it can be installed without constructing an HFTransformerModel.
        HFTransformerModel._patch_hf_llama_like(
            None,
            decoder_layer_cls=modeling_llama.LlamaDecoderLayer,
            attention_cls=modeling_llama.LlamaAttention,
            mlp_cls=modeling_llama.LlamaMLP,
            experts_cls=None,
            router_cls=None,
        )
        with torch.device("meta"):
            model = LlamaForCausalLM(config)
        self.assertEqual(model.model.rotary_emb.inv_freq.device.type, "meta")
        model.to_empty(device=torch.device("cpu"))
        # ``to_empty()`` leaves the rotary buffers as arbitrary memory, which in
        # practice is usually zeros -- finite, and so indistinguishable from a
        # value written deliberately. Fill them with NaN instead, so that "was
        # never populated" is deterministic rather than dependent on what the
        # allocator returned.
        for buffer in model.model.rotary_emb.buffers():
            buffer.fill_(float("nan"))
        model.apply(model._init_weights)
        return model

    def _assert_matches_direct_construction(self, rope_type: str):
        config = _tiny_config(rope_type)
        # Expected values: the same module built directly on CPU, where HF's own
        # __init__ computes inv_freq normally. Deriving them that way keeps them
        # independent of the population under test.
        expected = LlamaRotaryEmbedding(config).inv_freq

        rotary = self._materialize(config).model.rotary_emb
        self.assertEqual(rotary.rope_type, rope_type)
        torch.testing.assert_close(rotary.inv_freq, expected)

    def test_inv_freq_default_rope(self):
        self._assert_matches_direct_construction("default")

    def test_inv_freq_llama3_rope(self):
        self._assert_matches_direct_construction("llama3")

    def test_original_inv_freq_is_materialized(self):
        """The copy dynamic/longrope restore from must hold real values too.

        It is not read for ``default``/``llama3``, but when it is read, all of
        it is copied over ``inv_freq``, so leaving it uninitialized produces a
        corrupted ``inv_freq`` a few forwards later.
        """
        config = _tiny_config("llama3")
        expected = LlamaRotaryEmbedding(config).inv_freq

        rotary = self._materialize(config).model.rotary_emb
        torch.testing.assert_close(rotary.original_inv_freq, expected)

    def test_no_rotary_buffer_is_left_uninitialized(self):
        """Catch buffers that this branch does not handle yet.

        The 4.x -> 5.x bump added ``original_inv_freq`` next to ``inv_freq``;
        a future one can add another. Asserting on names only covers what is
        already handled, so check that no buffer the rotary module registers is
        still NaN after materialization.
        """
        rotary = self._materialize(_tiny_config("llama3")).model.rotary_emb
        still_nan = [
            name for name, buffer in rotary.named_buffers() if torch.isnan(buffer).any()
        ]
        self.assertEqual(still_nan, [])


if __name__ == "__main__":
    unittest.main()

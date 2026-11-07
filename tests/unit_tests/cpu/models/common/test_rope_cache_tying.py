# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""``Decoder`` shares one RoPE cache across layers with equal RoPE configs."""

import dataclasses
import unittest

import torch

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.rope import RoPE
from torchtitan.models.deepseek_v4 import build_model_config as build_dsv4_config
from torchtitan.models.llama3 import MODEL_FLAVORS
from torchtitan.models.llama3.model import Llama3Model


def _make_config() -> Llama3Model.Config:
    build_config, max_context_length = MODEL_FLAVORS["debugmodel"]
    return build_config(
        attn_backend="flex", seq_len=max_context_length, parallelism=ParallelismConfig()
    )


def _with_layer_theta(
    config: Llama3Model.Config, layer: int, theta: float
) -> Llama3Model.Config:
    layers = list(config.layers)
    block = layers[layer]
    rope = dataclasses.replace(block.attention.rope, theta=theta)
    attention = dataclasses.replace(block.attention, rope=rope)
    layers[layer] = dataclasses.replace(block, attention=attention)
    return dataclasses.replace(config, layers=layers)


def _ropes(model: torch.nn.Module) -> list[RoPE]:
    return [m for m in model.modules() if isinstance(m, RoPE)]


class TestRoPECacheTying(unittest.TestCase):
    def test_caches_are_shared_after_build(self):
        ropes = _ropes(Llama3Model(_make_config()))
        self.assertGreater(len(ropes), 1)
        for rope in ropes[1:]:
            self.assertIs(rope.cache, ropes[0].cache)

    def test_caches_are_shared_after_to_empty_and_init_states(self):
        """Mirror the trainer: build on meta, ``to_empty()``, then ``init_states()``."""
        with torch.device("meta"):
            model = Llama3Model(_make_config())
        model.to_empty(device="cpu")
        model.init_states()

        ropes = _ropes(model)
        expected = ropes[0]._precompute_cache()
        for rope in ropes:
            self.assertIs(rope.cache, ropes[0].cache)
        torch.testing.assert_close(ropes[0].cache, expected, rtol=0, atol=0)

    def test_different_configs_keep_separate_caches(self):
        config = _with_layer_theta(_make_config(), layer=1, theta=10000.0)
        model = Llama3Model(config)
        model.init_states()

        ropes = _ropes(model)
        odd = model.layers["1"].attention.rope
        shared = [rope for rope in ropes if rope is not odd]
        for rope in shared[1:]:
            self.assertIs(rope.cache, shared[0].cache)
        self.assertIsNot(odd.cache, shared[0].cache)
        torch.testing.assert_close(odd.cache, odd._precompute_cache(), rtol=0, atol=0)
        self.assertFalse(torch.equal(odd.cache, shared[0].cache))


class TestDeepSeekV4RoPECacheTying(unittest.TestCase):
    """dsv4 mixes RoPE configs: ``rope`` on uncompressed layers and
    ``rope_compress`` on compressed ones, which is also used by the RoPEs
    inside ``Compressor`` and ``Indexer``."""

    def test_one_cache_per_rope_config(self):
        config = build_dsv4_config(
            "debugmodel", seq_len=256, parallelism=ParallelismConfig()
        )
        with torch.device("meta"):
            model = config.build()
        model.to_empty(device="cpu")
        model.init_states()

        caches = {}
        for name, module in model.named_modules():
            if isinstance(module, RoPE):
                caches.setdefault(id(module.cache), []).append((name, module))
        self.assertEqual(len(caches), 2)

        groups = list(caches.values())
        self.assertNotEqual(groups[0][0][1].config, groups[1][0][1].config)
        for group in groups:
            rep = group[0][1]
            torch.testing.assert_close(
                rep.cache, rep._precompute_cache(), rtol=0, atol=0
            )


if __name__ == "__main__":
    unittest.main()

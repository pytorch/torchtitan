# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Graft-suffix decomposition, and the swallow that hid 37 flavors.

Finding 36 called the suffix parsing "magic" and noted it had already hidden 37 flavors
once. Two separate hazards were behind that:

* the flags were derived by a chain of ``endswith``/``elif`` whose correctness depended on
  ``_gated_lora`` being tested before ``_gated`` -- true only because of the order the
  branches happened to be written in;
* ``_model_registry_accepts`` caught bare ``Exception``, so ANY bug inside
  ``model_registry`` reported as "not one of our flavors" and the name silently
  disappeared from discovery.

The first is now a table sorted by suffix length, so the ordering is structural. The
second is narrowed to the exceptions that actually mean "not a flavor". These pin both.
"""

import unittest

from torchtitan.models.kimi_k3 import (
    _decompose_graft,
    _model_registry_accepts,
    flavor_names,
    model_registry,
)


class TestGraftDecomposition(unittest.TestCase):
    def test_the_longer_suffix_wins_regardless_of_table_order(self):
        got = _decompose_graft("kimi_k3_k3mini_block_attn_res_gated_lora")
        self.assertEqual(got.base_flavor, "kimi_k3_k3mini_block_attn_res")
        self.assertTrue(got.gated)
        self.assertEqual(got.lora_rank, 16)

    def test_the_shorter_suffix_still_matches_on_its_own(self):
        got = _decompose_graft("kimi_k3_k3mini_block_attn_res_gated")
        self.assertEqual(got.base_flavor, "kimi_k3_k3mini_block_attn_res")
        self.assertTrue(got.gated)
        self.assertIsNone(got.lora_rank)

    def test_no_suffix_leaves_the_name_alone(self):
        got = _decompose_graft("kimi_k3_k3mini_block_attn_res")
        self.assertEqual(got.base_flavor, "kimi_k3_k3mini_block_attn_res")
        self.assertFalse(got.gated)
        self.assertIsNone(got.lora_rank)

    def test_flags_reach_the_built_spec(self):
        """Decomposition is only useful if the spec ends up carrying it."""
        for name, gated, rank in (
            ("kimi_k3_k3mini_block_attn_res", False, None),
            ("kimi_k3_k3mini_block_attn_res_gated", True, None),
            ("kimi_k3_k3mini_block_attn_res_gated_lora", True, 16),
        ):
            spec = model_registry(name).model
            with self.subTest(flavor=name):
                self.assertEqual(bool(spec.attn_res_gated), gated)
                self.assertEqual(spec.lora_rank, rank)


class TestAcceptDoesNotSwallowBugs(unittest.TestCase):
    def test_an_unexpected_error_is_not_reported_as_not_a_flavor(self):
        """The failure mode that hid 37 flavors: any bug reading as 'unknown name'."""
        import torchtitan.models.kimi_k3 as pkg

        original = pkg.model_registry

        def explodes(flavor, attn_backend=None):
            raise RuntimeError("a bug inside model_registry, not a bad flavor name")

        pkg.model_registry = explodes
        try:
            with self.assertRaises(RuntimeError):
                _model_registry_accepts("kimi_k3_k3mini_block_attn_res")
        finally:
            pkg.model_registry = original

    def test_a_genuinely_unknown_name_is_still_rejected_quietly(self):
        self.assertFalse(_model_registry_accepts("not_a_kimi_flavor_at_all"))

    def test_every_discovered_flavor_is_actually_buildable(self):
        """The round trip, which is the property that broke when 37 went missing.

        flavor_names() is the SCALING_LAW_TABLE cross product by design -- not every
        buildable size, so k3mini is legitimately absent from it. What must hold is that
        nothing it advertises fails to build.
        """
        names = flavor_names()
        self.assertGreater(len(names), 10, "flavor discovery collapsed")
        for name in names:
            with self.subTest(flavor=name):
                self.assertTrue(
                    _model_registry_accepts(name),
                    f"{name} is advertised by flavor_names() but does not build",
                )


if __name__ == "__main__":
    unittest.main()

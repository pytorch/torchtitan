# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""The sweep-size flavor stubs must agree with SCALING_LAW_TABLE.

Finding 27 called the twelve copy-paste ``kimi_linear_<size>_<variant>`` stubs a
candidate for loop generation. They are kept explicit on purpose -- torchtitan's config
registries are looked up with ``getattr(module, name)``, and a module-level ``def`` is
greppable, completable and shows up in a traceback, none of which a name injected into
``globals()`` does.

What the finding was right about is that NOTHING checked the stubs against the table, so
a row added to ``SCALING_LAW_TABLE`` silently had no flavor, a stub for a removed row
silently kept building, and a stub could name one row while building another. Those are
the gaps here.

The invariant is NOT "table cross product == stub set", which the first version of this
file asserted and which is false by design: ``2p8t`` is exposed under the ``kimi_k3_``
prefix, ``447m_aligned`` only with the ``_n4`` suffix, and ``528m_l16`` is a hand-built
16-layer variant with no row at all. What holds is narrower and is what is pinned below:
a row is either absent from the ``kimi_linear_`` namespace or present in ALL THREE
variants, every stub builds the row it names, and any stub naming no row is a listed
exception rather than a surprise.
"""

import unittest

from torchtitan.models.kimi_k3 import config_registry
from torchtitan.models.kimi_k3.model_configs import SCALING_LAW_TABLE


_VARIANTS = ("baseline", "block_attn_res", "full_attn_res")

# Stubs that deliberately do not correspond to a SCALING_LAW_TABLE row. Listed rather
# than pattern-matched so that a NEW unexplained stub fails this file.
_KNOWN_NON_TABLE_SIZES = frozenset({"528m_l16"})


def _stub_sizes() -> dict[str, set[str]]:
    """``{size: {variant, ...}}`` over the ``kimi_linear_<size>_<variant>`` stubs."""
    found: dict[str, set[str]] = {}
    for name in dir(config_registry):
        if not name.startswith("kimi_linear_") or not callable(
            getattr(config_registry, name)
        ):
            continue
        for variant in _VARIANTS:
            if name.endswith(f"_{variant}"):
                size = name[len("kimi_linear_") : -len(f"_{variant}")]
                found.setdefault(size, set()).add(variant)
                break
    return found


class TestScalingLawStubs(unittest.TestCase):
    def test_a_size_is_either_absent_or_covered_by_all_three_variants(self):
        partial = {
            size: sorted(variants)
            for size, variants in _stub_sizes().items()
            if len(variants) != len(_VARIANTS)
        }
        self.assertFalse(partial, f"sizes with only some variants: {partial}")

    def test_every_stub_names_a_table_row_or_a_listed_exception(self):
        table_sizes = {row.name for row in SCALING_LAW_TABLE}
        unexplained = set(_stub_sizes()) - table_sizes - _KNOWN_NON_TABLE_SIZES
        self.assertFalse(
            unexplained,
            f"stub sizes with no SCALING_LAW_TABLE row and no listed reason: "
            f"{sorted(unexplained)}",
        )

    def test_each_stub_builds_the_row_it_names(self):
        """A stub pointing at the wrong row is what a name-only check cannot see."""
        by_name = {row.name: row for row in SCALING_LAW_TABLE}
        checked = 0
        for size, variants in _stub_sizes().items():
            row = by_name.get(size)
            if row is None:
                continue
            for variant in sorted(variants):
                name = f"kimi_linear_{size}_{variant}"
                kc = getattr(config_registry, name)().model_spec.model.kimi_config
                self.assertEqual(
                    kc.num_hidden_layers,
                    row.n_layers,
                    f"{name} built {kc.num_hidden_layers} layers, table says "
                    f"{row.n_layers}",
                )
                self.assertEqual(kc.hidden_size, row.d_model, name)
                self.assertEqual(kc.num_attention_heads, row.num_heads, name)
                checked += 1
        # Guard the guard: a bug in _stub_sizes would make this vacuously pass.
        self.assertGreaterEqual(checked, 12, "expected at least the twelve sweep stubs")


if __name__ == "__main__":
    unittest.main()

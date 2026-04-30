# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for compare_numerics.py.

Run with:
    python -m unittest tests.unit_tests.test_compare_numerics -v
"""

import os
import tempfile
import unittest

import torch


class TestCompareNumerics(unittest.TestCase):
    """Tests for compare_numerics.py: log parsing, matching, HTML generation."""

    def test_parse_log(self):
        from torchtitan.tools.compare_numerics import parse_log

        content = """\
Total captured activations: 2
================================================================================

[mod_a/op_0_mm]
  Shape: torch.Size([4, 8]), Dtype: torch.float32
  L1 norm:  1.000000e+00
  L2 norm:  2.000000e+00
  Min:      -1.000000e+00
  Max:      1.000000e+00
  Location: test.py:10

[mod_b/op_0_add]
  Shape: torch.Size([4, 8]), Dtype: torch.float32
  L1 norm:  3.000000e+00
  L2 norm:  4.000000e+00
  Min:      0.000000e+00
  Max:      2.000000e+00
  Phase: backward
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", delete=False
        ) as f:
            f.write(content)
            path = f.name
        try:
            entries = parse_log(path)
            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0].key, "mod_a/op_0_mm")
            self.assertEqual(entries[0].phase, "forward")
            self.assertEqual(entries[0].stats["L1 norm"], "1.000000e+00")
            self.assertEqual(entries[1].key, "mod_b/op_0_add")
            self.assertEqual(entries[1].phase, "backward")
        finally:
            os.unlink(path)

    def test_exact_match(self):
        from torchtitan.tools.compare_numerics import OpEntry, match_entries

        eager = [OpEntry(key="mod/op_0_mm", stats={"L1 norm": "1.0"})]
        traced = [OpEntry(key="mod/op_0_mm", stats={"L1 norm": "1.0"})]
        results = match_entries(eager, traced)
        matched = [(e, t, s, d) for e, t, s, d in results if s == "match"]
        self.assertEqual(len(matched), 1)

    def test_fuzzy_match_ignores_counter(self):
        from torchtitan.tools.compare_numerics import OpEntry, match_entries

        eager = [
            OpEntry(key="mod/op_0_mm", stats={"L1 norm": "1.0"}),
            OpEntry(key="mod/op_3_mm", stats={"L1 norm": "2.0"}, phase="backward"),
        ]
        traced = [
            OpEntry(key="mod/op_0_mm", stats={"L1 norm": "1.0"}),
            OpEntry(key="mod/op_1_mm", stats={"L1 norm": "2.0"}, phase="backward"),
        ]
        results = match_entries(eager, traced)
        matched = [(e, t, s, d) for e, t, s, d in results if e and t]
        self.assertEqual(len(matched), 2)

    def test_numeric_match_by_shape_and_l1(self):
        from torchtitan.tools.compare_numerics import OpEntry, match_entries

        eager = [
            OpEntry(
                key="<none>/op_5_add",
                stats={"Shape": "[4, 8]", "L1 norm": "1.5"},
                phase="backward",
            ),
        ]
        traced = [
            OpEntry(
                key="layers.0/op_2_add",
                stats={"Shape": "[4, 8]", "L1 norm": "1.5"},
                phase="backward",
            ),
        ]
        results = match_entries(eager, traced)
        paired = [(e, t, s, d) for e, t, s, d in results if e and t]
        self.assertEqual(len(paired), 1)

    def test_shape_diff_detected(self):
        from torchtitan.tools.compare_numerics import OpEntry, match_entries

        eager = [OpEntry(key="mod/op_0_mm", stats={
            "Shape": "[2048, 2048]", "L1 norm": "1.0",
        })]
        traced = [OpEntry(key="mod/op_0_mm", stats={
            "Shape": "[16384, 2048]", "L1 norm": "8.0",
        })]
        results = match_entries(eager, traced)
        diffs = results[0][3]
        self.assertIn("Shape", diffs)

    def test_generate_html(self):
        from torchtitan.tools.compare_numerics import (
            OpEntry,
            generate_html,
            match_entries,
        )

        eager = [OpEntry(key="mod/op_0_mm", stats={"L1 norm": "1.0"})]
        traced = [OpEntry(key="mod/op_0_mm", stats={"L1 norm": "1.0"})]
        results = match_entries(eager, traced)
        html_out = generate_html(results, "eager.log", "traced.log")
        self.assertIn("mod/op_0_mm", html_out)
        self.assertIn("Forward", html_out)

    def test_fuzzy_match_shows_both_keys(self):
        from torchtitan.tools.compare_numerics import (
            OpEntry,
            generate_html,
            match_entries,
        )

        eager = [OpEntry(key="mod/op_3_mm", stats={"L1 norm": "1.0"}, phase="backward")]
        traced = [OpEntry(key="mod/op_1_mm", stats={"L1 norm": "1.0"}, phase="backward")]
        results = match_entries(eager, traced)
        html_out = generate_html(results, "eager.log", "traced.log")
        self.assertIn("mod/op_3_mm", html_out)
        self.assertIn("mod/op_1_mm", html_out)
        self.assertIn("key-eager", html_out)
        self.assertIn("key-traced", html_out)

    def test_near_zero_diff_very_faint(self):
        """Negligible diffs relative to L1 norm should have near-zero
        intensity (very faint color), not full red."""
        from torchtitan.tools.compare_numerics import OpEntry, _compute_diffs

        e = OpEntry(key="test", stats={
            "L1 norm": "1.000000e+03", "Mean": "1.293876e-20",
        })
        t = OpEntry(key="test", stats={
            "L1 norm": "1.000000e+03", "Mean": "-1.284860e-20",
        })
        diffs = _compute_diffs(e, t)
        self.assertIn("Mean", diffs)
        # Intensity should be negligible (< 1e-15)
        self.assertLess(diffs["Mean"], 1e-15)

    def test_real_diff_flagged(self):
        """Meaningful diffs should be flagged."""
        from torchtitan.tools.compare_numerics import OpEntry, _compute_diffs

        e = OpEntry(key="test", stats={
            "L1 norm": "1.000000e+03", "Mean": "1.000000e-02",
        })
        t = OpEntry(key="test", stats={
            "L1 norm": "1.000000e+03", "Mean": "2.000000e-02",
        })
        diffs = _compute_diffs(e, t)
        self.assertIn("Mean", diffs)

    def test_shape_diff_always_flagged(self):
        """Shape mismatches are always flagged at max intensity."""
        from torchtitan.tools.compare_numerics import OpEntry, _compute_diffs

        e = OpEntry(key="test", stats={
            "L1 norm": "1.0", "Shape": "[2048, 2048]",
        })
        t = OpEntry(key="test", stats={
            "L1 norm": "1.0", "Shape": "[16384, 2048]",
        })
        diffs = _compute_diffs(e, t)
        self.assertIn("Shape", diffs)
        self.assertEqual(diffs["Shape"], 1.0)


if __name__ == "__main__":
    unittest.main()

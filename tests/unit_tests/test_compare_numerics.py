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
            entries, skipped = parse_log(path)
            self.assertEqual(len(entries), 2)
            self.assertEqual(entries[0].key, "mod_a/op_0_mm")
            self.assertEqual(entries[0].phase, "forward")
            self.assertEqual(entries[0].stats["L1 norm"], "1.000000e+00")
            self.assertEqual(entries[1].key, "mod_b/op_0_add")
            self.assertEqual(entries[1].phase, "backward")
            # Missing header line → empty set, not an error.
            self.assertEqual(skipped, set())
        finally:
            os.unlink(path)

    def test_parse_log_reads_skipped_ops(self):
        """``Excluded ops dispatched:`` header is parsed into the set."""
        from torchtitan.tools.compare_numerics import parse_log

        content = """\
Total captured activations: 1
Excluded ops dispatched: wait_tensor, _pre_bucket_all_gather, all_gather_into_tensor_out
================================================================================

[mod/op_0_mm]
  Shape: torch.Size([4, 8])
  L1 norm:  1.0
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", delete=False
        ) as f:
            f.write(content)
            path = f.name
        try:
            _, skipped = parse_log(path)
            self.assertEqual(
                skipped,
                {"wait_tensor", "_pre_bucket_all_gather", "all_gather_into_tensor_out"},
            )
        finally:
            os.unlink(path)

    def test_parse_log_empty_skipped(self):
        """``Excluded ops dispatched: (none)`` parses as empty set."""
        from torchtitan.tools.compare_numerics import parse_log

        content = """\
Total captured activations: 0
Excluded ops dispatched: (none)
================================================================================
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".log", delete=False
        ) as f:
            f.write(content)
            path = f.name
        try:
            _, skipped = parse_log(path)
            self.assertEqual(skipped, set())
        finally:
            os.unlink(path)

    def test_html_renders_skipped_ops_section(self):
        """The skipped-ops section appears in the HTML for both sides,
        showing one chip per op name and the run names."""
        from torchtitan.tools.compare_numerics import (
            OpEntry,
            generate_html,
            match_entries,
        )

        eager = [OpEntry(key="mod/op_0_mm", stats={"L1 norm": "1.0"})]
        traced = [OpEntry(key="mod/op_0_mm", stats={"L1 norm": "1.0"})]
        html_out = generate_html(
            match_entries(eager, traced),
            "e.log",
            "t.log",
            name1="eager",
            name2="traced",
            skipped1=set(),
            skipped2={"wait_tensor", "_pre_bucket_all_gather"},
        )
        self.assertIn("Excluded ops dispatched", html_out)
        # The skipped chip should carry the op name, not any FQN.
        self.assertIn("wait_tensor", html_out)
        self.assertIn("_pre_bucket_all_gather", html_out)
        self.assertIn("skipped-chip", html_out)
        # Side 1 has no skipped ops → renders "(none)".
        self.assertIn("(none)", html_out)

    def test_exact_match(self):
        from torchtitan.tools.compare_numerics import OpEntry, match_entries

        eager = [OpEntry(key="mod/op_0_mm", stats={"L1 norm": "1.0"})]
        traced = [OpEntry(key="mod/op_0_mm", stats={"L1 norm": "1.0"})]
        results = match_entries(eager, traced)
        matched = [r for r in results if r[2] == "match"]
        self.assertEqual(len(matched), 1)
        # Strategy should be "exact" for same-key matches.
        self.assertEqual(matched[0][4], "exact")

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
        matched = [r for r in results if r[0] and r[1]]
        self.assertEqual(len(matched), 2)
        strategies = {r[4] for r in matched}
        self.assertEqual(strategies, {"exact", "fuzzy"})

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
        paired = [r for r in results if r[0] and r[1]]
        self.assertEqual(len(paired), 1)
        self.assertEqual(paired[0][4], "stats")

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

    def test_per_stat_scaling(self):
        """Each stat is compared on its own scale, not L1-normalized.

        L1 norm itself differs by 0.1% — should register as ~1e-3 diff.
        Min differs by 100% — should register as ~1.0 diff. The two
        intensities must not be tied together (the old behavior scaled
        Min by L1, which buried meaningful Min diffs).
        """
        from torchtitan.tools.compare_numerics import OpEntry, _compute_diffs

        e = OpEntry(key="test", stats={
            "L1 norm": "1.000e+03", "Min": "-1.0",
        })
        t = OpEntry(key="test", stats={
            "L1 norm": "1.001e+03", "Min": "-2.0",
        })
        diffs = _compute_diffs(e, t)
        # L1 norm: |1003 - 1000| / 1003 ≈ 0.001
        self.assertAlmostEqual(diffs["L1 norm"], 1e-3, places=3)
        # Min: |1 - 2| / 2 = 0.5 — judged on Min's own magnitude, not L1
        self.assertAlmostEqual(diffs["Min"], 0.5, places=2)

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

    def test_results_preserve_eager_order(self):
        """Eager-side rows must appear in eager-log order regardless of
        which matching pass claimed each entry."""
        from torchtitan.tools.compare_numerics import OpEntry, match_entries

        # Eager: [exact, fuzzy, only, exact]. Traced has matching entries
        # for indices 0, 1, 3 — index 2 is eager_only. Without
        # order-preservation the result would interleave by pass
        # (exact-then-fuzzy), shuffling indices 1 and 3.
        eager = [
            OpEntry(key="mod/op_0_mm", stats={"L1 norm": "1.0"}),
            OpEntry(key="mod/op_3_mm", stats={"L1 norm": "2.0"}),
            OpEntry(key="mod/op_5_only", stats={"L1 norm": "9.0"}),
            OpEntry(key="mod/op_7_silu", stats={"L1 norm": "4.0"}),
        ]
        traced = [
            OpEntry(key="mod/op_0_mm", stats={"L1 norm": "1.0"}),
            OpEntry(key="mod/op_1_mm", stats={"L1 norm": "2.0"}),
            OpEntry(key="mod/op_7_silu", stats={"L1 norm": "4.0"}),
        ]
        results = match_entries(eager, traced)
        eager_keys = [r[0].key for r in results if r[0] is not None]
        self.assertEqual(eager_keys, [e.key for e in eager])

    def test_only_side_has_empty_strategy(self):
        """eager_only / traced_only rows must have empty match_strategy."""
        from torchtitan.tools.compare_numerics import OpEntry, match_entries

        eager = [OpEntry(key="mod/op_0_mm", stats={"L1 norm": "1.0"})]
        traced = [OpEntry(key="mod/op_0_other", stats={"L1 norm": "2.0"})]
        results = match_entries(eager, traced)
        statuses = {r[2]: r[4] for r in results}
        self.assertEqual(statuses["eager_only"], "")
        self.assertEqual(statuses["traced_only"], "")

    def test_html_uses_default_names(self):
        """Without --name1/--name2 the page uses 'run1' and 'run2'."""
        from torchtitan.tools.compare_numerics import (
            OpEntry,
            generate_html,
            match_entries,
        )

        eager = [OpEntry(key="mod/op_0_mm", stats={"L1 norm": "1.0"})]
        traced = [OpEntry(key="mod/op_0_mm", stats={"L1 norm": "1.0"})]
        html_out = generate_html(match_entries(eager, traced), "e.log", "t.log")
        self.assertIn("run1 vs run2", html_out)
        self.assertIn(">run1<", html_out)
        self.assertIn(">run2<", html_out)

    def test_html_uses_custom_names(self):
        """Custom names appear in the title, side labels, only-side
        counts, and filter checkboxes."""
        from torchtitan.tools.compare_numerics import (
            OpEntry,
            generate_html,
            match_entries,
        )

        eager = [OpEntry(key="mod/op_0_mm", stats={"L1 norm": "1.0"})]
        traced = [OpEntry(key="mod/op_0_other", stats={"L1 norm": "1.0"})]
        html_out = generate_html(
            match_entries(eager, traced),
            "e.log",
            "t.log",
            name1="bf16",
            name2="fp32",
        )
        self.assertIn("bf16 vs fp32", html_out)
        self.assertIn("bf16 only", html_out)
        self.assertIn("fp32 only", html_out)
        self.assertIn("Show bf16-only", html_out)
        self.assertIn("Show fp32-only", html_out)
        self.assertNotIn("run1", html_out)

    def test_html_escapes_names(self):
        """Names are HTML-escaped to prevent injection."""
        from torchtitan.tools.compare_numerics import (
            OpEntry,
            generate_html,
            match_entries,
        )

        eager = [OpEntry(key="mod/op_0_mm", stats={"L1 norm": "1.0"})]
        traced = [OpEntry(key="mod/op_0_mm", stats={"L1 norm": "1.0"})]
        html_out = generate_html(
            match_entries(eager, traced),
            "e.log",
            "t.log",
            name1="<img src=x>",
            name2="A&B",
        )
        # Name must be escaped, not injected as raw markup.
        self.assertNotIn("<img src=x>", html_out)
        self.assertIn("&lt;img src=x&gt;", html_out)
        self.assertIn("A&amp;B", html_out)

    def test_html_renders_match_method_column(self):
        """The HTML must include the Match method column header and
        chip labels for each strategy."""
        from torchtitan.tools.compare_numerics import (
            OpEntry,
            generate_html,
            match_entries,
        )

        eager = [
            OpEntry(key="mod/op_0_mm", stats={"L1 norm": "1.0"}),
            OpEntry(key="mod/op_5_mm", stats={"L1 norm": "2.0"}, phase="backward"),
            OpEntry(
                key="<none>/op_5_add",
                stats={"Shape": "[4, 8]", "L1 norm": "1.5"},
                phase="backward",
            ),
        ]
        traced = [
            OpEntry(key="mod/op_0_mm", stats={"L1 norm": "1.0"}),
            OpEntry(key="mod/op_1_mm", stats={"L1 norm": "2.0"}, phase="backward"),
            OpEntry(
                key="layers.0/op_2_add",
                stats={"Shape": "[4, 8]", "L1 norm": "1.5"},
                phase="backward",
            ),
        ]
        html_out = generate_html(match_entries(eager, traced), "e.log", "t.log")
        self.assertIn("Match method", html_out)
        # Each pass produces its corresponding chip label and CSS class.
        self.assertIn("strategy-exact", html_out)
        self.assertIn("same op key", html_out)
        self.assertIn("strategy-fuzzy", html_out)
        self.assertIn("fuzzy op key", html_out)
        self.assertIn("strategy-stats", html_out)

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

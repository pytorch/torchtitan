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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
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
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
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
            generate_html,
            match_entries,
            OpEntry,
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
        from torchtitan.tools.compare_numerics import match_entries, OpEntry

        eager = [OpEntry(key="mod/op_0_mm", stats={"L1 norm": "1.0"})]
        traced = [OpEntry(key="mod/op_0_mm", stats={"L1 norm": "1.0"})]
        results = match_entries(eager, traced)
        matched = [r for r in results if r[2] == "match"]
        self.assertEqual(len(matched), 1)
        # Strategy should be "exact" for same-key matches.
        self.assertEqual(matched[0][4], "exact")

    def test_parse_log_reads_input_producers(self):
        from torchtitan.tools.compare_numerics import parse_log

        content = """\
Total captured activations: 1
================================================================================

[mod/op_1_add]
  Output hash: 9.9e+02
  Input hashes: 1.0e+02, 2.0e+02
  Input producers: mod/op_0_mm;
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(content)
            path = f.name
        try:
            entries, _ = parse_log(path)
            self.assertEqual(entries[0].input_producers, "mod/op_0_mm;")
        finally:
            os.unlink(path)

    def test_html_wraps_input_with_producer_tooltip(self):
        from torchtitan.tools.compare_numerics import (
            generate_html,
            match_entries,
            OpEntry,
        )

        eager = [
            OpEntry(
                key="mod/op_1_add",
                stats={"L1 norm": "1.0"},
                input_hashes="1.0e+02, 2.0e+02",
                input_producers="mod/op_0_mm;",
            ),
        ]
        traced = [
            OpEntry(
                key="mod/op_1_add",
                stats={"L1 norm": "1.0"},
                input_hashes="1.0e+02, 2.0e+02",
                input_producers="mod/op_0_mm;",
            ),
        ]
        results = match_entries(eager, traced)
        html_out = generate_html(results, "e.log", "t.log")
        # First input value wrapped with producer tooltip span; carries
        # both data-tooltip (for the CSS ::after bubble) and the native
        # title attribute (fallback).
        self.assertIn('class="prod-link"', html_out)
        self.assertIn('data-tooltip="produced by mod/op_0_mm"', html_out)
        self.assertIn('title="produced by mod/op_0_mm"', html_out)
        # Second input has no producer — appears as bare text, not in a span.
        self.assertNotIn(">2.0e+02</span>", html_out)
        # CSS for hover affordance is present.
        self.assertIn(".prod-link", html_out)

    def test_parse_log_reads_hashes(self):
        from torchtitan.tools.compare_numerics import parse_log

        content = """\
Total captured activations: 1
================================================================================

[mod/op_0_mm]
  Shape: torch.Size([8, 8]), Dtype: torch.float32
  L1 norm:  1.0
  Output hash: 1.234e+02
  Input hashes: 5.0e+01, 6.0e+01
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(content)
            path = f.name
        try:
            entries, _ = parse_log(path)
            self.assertEqual(entries[0].output_hash, "1.234e+02")
            self.assertEqual(entries[0].input_hashes, "5.0e+01, 6.0e+01")
        finally:
            os.unlink(path)

    def test_html_flags_large_hash_mismatch_red(self):
        """An ~order-of-magnitude diff in output_hash gets full-red styling."""
        from torchtitan.tools.compare_numerics import (
            generate_html,
            match_entries,
            OpEntry,
        )

        eager = [
            OpEntry(
                key="mod/op_0_mm",
                stats={"L1 norm": "1.0"},
                output_hash="1.234e+02",
                input_hashes="5.0e+01",
            ),
        ]
        traced = [
            OpEntry(
                key="mod/op_0_mm",
                stats={"L1 norm": "1.0"},
                output_hash="9.999e+02",
                input_hashes="5.0e+01",
            ),
        ]
        results = match_entries(eager, traced)
        html_out = generate_html(results, "e.log", "t.log")
        self.assertIn("Output L1 norm", html_out)
        self.assertIn("Input L1 norm", html_out)
        self.assertIn("1.234e+02", html_out)
        self.assertIn("9.999e+02", html_out)
        # Full-red text for ~1.0 relative diff.
        self.assertIn("color: rgb(248,80,80)", html_out)

    def test_html_hash_last_ulp_drift_renders_faint(self):
        """1-ULP-level last-digit drift in hashes (e.g. float32 reduction
        noise) renders with low intensity, not full red.  This is the
        pattern we see in _fused_adamw_ rows where 22/285 inputs drift
        by ~1e-7 between eager and traced."""
        from torchtitan.tools.compare_numerics import (
            generate_html,
            match_entries,
            OpEntry,
        )

        eager = [
            OpEntry(
                key="<none>/op_0__fused_adamw_",
                stats={},
                input_hashes="1.260219e+01, 5.743664e+00",
            ),
        ]
        traced = [
            OpEntry(
                key="<none>/op_0__fused_adamw_",
                stats={},
                input_hashes="1.260218e+01, 5.743663e+00",
            ),
        ]
        results = match_entries(eager, traced)
        html_out = generate_html(results, "e.log", "t.log")
        # Mismatched values are present
        self.assertIn("1.260219e+01", html_out)
        self.assertIn("1.260218e+01", html_out)
        # Full-red (rgb(248,80,80)) text should NOT appear for ~1e-7 diff.
        # Faint text is rgb(220-something, 220-something, 220-something).
        # Specifically check: the worst diff here is ~7.9e-8, which is
        # below the 1e-8 floor → intensity 0 → no styling at all.
        # If the diff were larger (e.g., 1e-5), text would be faint pink
        # but not full red.
        self.assertNotIn(
            'style="color: rgb(248,80,80); background: rgba(248,80,80,0.15)',
            html_out,
        )

    def test_overrides_force_match(self):
        """An override pair forces matching even when keys/stats diverge."""
        from torchtitan.tools.compare_numerics import match_entries, OpEntry

        eager = [
            OpEntry(
                key="feed_forward/op_7_mul",
                stats={"Shape": "[4,8]"},
                output_hash="9.99e+02",
            ),
        ]
        traced = [
            OpEntry(
                key="layers.2.feed_forward/op_3_mul",
                stats={"Shape": "[4,8]"},
                output_hash="1.23e+05",
            ),
        ]
        overrides = {
            "feed_forward/op_7_mul": "layers.2.feed_forward/op_3_mul",
        }
        results = match_entries(eager, traced, overrides=overrides)
        # Single paired row, strategy is "override".
        paired = [r for r in results if r[0] and r[1]]
        self.assertEqual(len(paired), 1)
        self.assertEqual(paired[0][4], "override")
        # Output hashes differ, so status is "diff" (not "match").
        self.assertEqual(paired[0][2], "diff")

    def test_overrides_silently_skip_missing(self):
        """Override entries whose target keys don't exist on either side
        are ignored (no error)."""
        from torchtitan.tools.compare_numerics import match_entries, OpEntry

        eager = [OpEntry(key="mod/op_0_mm", stats={"L1 norm": "1.0"})]
        traced = [OpEntry(key="mod/op_0_mm", stats={"L1 norm": "1.0"})]
        overrides = {"nonexistent/op_0_x": "also_missing/op_0_y"}
        results = match_entries(eager, traced, overrides=overrides)
        # Falls through to exact pass — the real ops still pair.
        paired = [r for r in results if r[0] and r[1]]
        self.assertEqual(len(paired), 1)
        self.assertEqual(paired[0][4], "exact")

    def test_strip_root_class_prefix(self):
        from torchtitan.tools.compare_numerics import _strip_root_class_prefix

        # Strips wrapped-model class names regardless of which model.
        self.assertEqual(
            _strip_root_class_prefix("FSDPLlama3Model.layers.0.wq/op_0_mm"),
            "layers.0.wq/op_0_mm",
        )
        self.assertEqual(
            _strip_root_class_prefix("FSDPQwenModel.layers.0.wq/op_0_mm"),
            "layers.0.wq/op_0_mm",
        )
        # Doesn't strip already-unprefixed keys.
        self.assertEqual(
            _strip_root_class_prefix("layers.0.wq/op_0_mm"),
            "layers.0.wq/op_0_mm",
        )
        # Doesn't strip the special <none> root.
        self.assertEqual(
            _strip_root_class_prefix("<none>/op_0_mm"),
            "<none>/op_0_mm",
        )
        # Doesn't strip a lowercase-only segment (e.g. "layers.0...").
        self.assertEqual(
            _strip_root_class_prefix("layers.0/op_0_mm"),
            "layers.0/op_0_mm",
        )

    def test_exact_match_strips_root_class_prefix(self):
        """Eager DebugMode keys (rooted at model class) must match
        traced keys (unrooted) under exact-pass matching."""
        from torchtitan.tools.compare_numerics import match_entries, OpEntry

        eager = [
            OpEntry(
                key="FSDPLlama3Model.layers.2.attention.wo/op_0_mm",
                stats={"L1 norm": "1.0"},
            ),
        ]
        traced = [
            OpEntry(
                key="layers.2.attention.wo/op_0_mm",
                stats={"L1 norm": "1.0"},
            ),
        ]
        results = match_entries(eager, traced)
        matched = [r for r in results if r[0] and r[1]]
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0][4], "exact")
        # Display keys preserved (not normalized).
        self.assertTrue(matched[0][0].key.startswith("FSDPLlama3Model."))
        self.assertFalse(matched[0][1].key.startswith("FSDPLlama3Model."))

    def test_fuzzy_match_strips_root_class_prefix(self):
        from torchtitan.tools.compare_numerics import match_entries, OpEntry

        eager = [
            OpEntry(
                key="FSDPLlama3Model.layers.0.wq/op_3_mm",
                stats={"L1 norm": "1.0"},
                phase="backward",
            ),
        ]
        traced = [
            OpEntry(
                key="layers.0.wq/op_1_mm",
                stats={"L1 norm": "1.0"},
                phase="backward",
            ),
        ]
        results = match_entries(eager, traced)
        matched = [r for r in results if r[0] and r[1]]
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0][4], "fuzzy")

    def test_fuzzy_match_ignores_counter(self):
        from torchtitan.tools.compare_numerics import match_entries, OpEntry

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

    def test_numeric_match_by_shape_and_output_hash(self):
        """Pass 3 keys ops by (op_type, Shape, output_hash) — same data,
        different FQN should still pair."""
        from torchtitan.tools.compare_numerics import match_entries, OpEntry

        eager = [
            OpEntry(
                key="<none>/op_5_add",
                stats={"Shape": "[4, 8]"},
                output_hash="1.5",
                phase="backward",
            ),
        ]
        traced = [
            OpEntry(
                key="layers.0/op_2_add",
                stats={"Shape": "[4, 8]"},
                output_hash="1.5",
                phase="backward",
            ),
        ]
        results = match_entries(eager, traced)
        paired = [r for r in results if r[0] and r[1]]
        self.assertEqual(len(paired), 1)
        self.assertEqual(paired[0][4], "stats")

    def test_match_status_ignores_l2_min_max_mean(self):
        """Match status is driven only by Shape + output_hash +
        input_hashes.  A row whose L2/Min/Max/Mean differ but whose
        Shape and hashes agree should still classify as ``match``."""
        from torchtitan.tools.compare_numerics import match_entries, OpEntry

        eager = [
            OpEntry(
                key="m/op_0_mm",
                stats={"Shape": "[4, 8]", "L2 norm": "1.0", "Min": "-1.0"},
                output_hash="9.99e+01",
                input_hashes="5.0e+01",
            ),
        ]
        traced = [
            OpEntry(
                key="m/op_0_mm",
                stats={"Shape": "[4, 8]", "L2 norm": "2.0", "Min": "-7.0"},
                output_hash="9.99e+01",
                input_hashes="5.0e+01",
            ),
        ]
        results = match_entries(eager, traced)
        self.assertEqual(results[0][2], "match")

    def test_match_status_flags_output_hash_diff(self):
        """Differing output_hash → diff status, regardless of stats."""
        from torchtitan.tools.compare_numerics import match_entries, OpEntry

        eager = [
            OpEntry(key="m/op_0_mm", stats={"Shape": "S"}, output_hash="1.0"),
        ]
        traced = [
            OpEntry(key="m/op_0_mm", stats={"Shape": "S"}, output_hash="2.0"),
        ]
        results = match_entries(eager, traced)
        self.assertEqual(results[0][2], "diff")

    def test_match_status_flags_shape_diff(self):
        from torchtitan.tools.compare_numerics import match_entries, OpEntry

        eager = [
            OpEntry(key="m/op_0_mm", stats={"Shape": "[4,8]"}, output_hash="1.0"),
        ]
        traced = [
            OpEntry(key="m/op_0_mm", stats={"Shape": "[8,4]"}, output_hash="1.0"),
        ]
        results = match_entries(eager, traced)
        self.assertEqual(results[0][2], "diff")

    def test_shape_diff_detected(self):
        from torchtitan.tools.compare_numerics import match_entries, OpEntry

        eager = [
            OpEntry(
                key="mod/op_0_mm",
                stats={
                    "Shape": "[2048, 2048]",
                    "L1 norm": "1.0",
                },
            )
        ]
        traced = [
            OpEntry(
                key="mod/op_0_mm",
                stats={
                    "Shape": "[16384, 2048]",
                    "L1 norm": "8.0",
                },
            )
        ]
        results = match_entries(eager, traced)
        diffs = results[0][3]
        self.assertIn("Shape", diffs)

    def test_generate_html(self):
        from torchtitan.tools.compare_numerics import (
            generate_html,
            match_entries,
            OpEntry,
        )

        eager = [OpEntry(key="mod/op_0_mm", stats={"L1 norm": "1.0"})]
        traced = [OpEntry(key="mod/op_0_mm", stats={"L1 norm": "1.0"})]
        results = match_entries(eager, traced)
        html_out = generate_html(results, "eager.log", "traced.log")
        self.assertIn("mod/op_0_mm", html_out)
        self.assertIn("Forward", html_out)

    def test_fuzzy_match_shows_both_keys(self):
        from torchtitan.tools.compare_numerics import (
            generate_html,
            match_entries,
            OpEntry,
        )

        eager = [OpEntry(key="mod/op_3_mm", stats={"L1 norm": "1.0"}, phase="backward")]
        traced = [
            OpEntry(key="mod/op_1_mm", stats={"L1 norm": "1.0"}, phase="backward")
        ]
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
        from torchtitan.tools.compare_numerics import _compute_diffs, OpEntry

        e = OpEntry(
            key="test",
            stats={
                "L2 norm": "1.000e+03",
                "Min": "-1.0",
            },
        )
        t = OpEntry(
            key="test",
            stats={
                "L2 norm": "1.001e+03",
                "Min": "-2.0",
            },
        )
        diffs = _compute_diffs(e, t)
        # L2 norm: |1003 - 1000| / 1003 ≈ 0.001
        self.assertAlmostEqual(diffs["L2 norm"], 1e-3, places=3)
        # Min: |1 - 2| / 2 = 0.5 — judged on Min's own magnitude, not L2
        self.assertAlmostEqual(diffs["Min"], 0.5, places=2)

    def test_real_diff_flagged(self):
        """Meaningful diffs should be flagged."""
        from torchtitan.tools.compare_numerics import _compute_diffs, OpEntry

        e = OpEntry(
            key="test",
            stats={
                "L2 norm": "1.000000e+03",
                "Mean": "1.000000e-02",
            },
        )
        t = OpEntry(
            key="test",
            stats={
                "L2 norm": "1.000000e+03",
                "Mean": "2.000000e-02",
            },
        )
        diffs = _compute_diffs(e, t)
        self.assertIn("Mean", diffs)

    def test_results_preserve_eager_order(self):
        """Eager-side rows must appear in eager-log order regardless of
        which matching pass claimed each entry."""
        from torchtitan.tools.compare_numerics import match_entries, OpEntry

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
        from torchtitan.tools.compare_numerics import match_entries, OpEntry

        eager = [OpEntry(key="mod/op_0_mm", stats={"L1 norm": "1.0"})]
        traced = [OpEntry(key="mod/op_0_other", stats={"L1 norm": "2.0"})]
        results = match_entries(eager, traced)
        statuses = {r[2]: r[4] for r in results}
        self.assertEqual(statuses["eager_only"], "")
        self.assertEqual(statuses["traced_only"], "")

    def test_html_uses_default_names(self):
        """Without --name1/--name2 the page uses 'run1' and 'run2'."""
        from torchtitan.tools.compare_numerics import (
            generate_html,
            match_entries,
            OpEntry,
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
            generate_html,
            match_entries,
            OpEntry,
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
            generate_html,
            match_entries,
            OpEntry,
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
            generate_html,
            match_entries,
            OpEntry,
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
        from torchtitan.tools.compare_numerics import _compute_diffs, OpEntry

        e = OpEntry(
            key="test",
            stats={
                "L1 norm": "1.0",
                "Shape": "[2048, 2048]",
            },
        )
        t = OpEntry(
            key="test",
            stats={
                "L1 norm": "1.0",
                "Shape": "[16384, 2048]",
            },
        )
        diffs = _compute_diffs(e, t)
        self.assertIn("Shape", diffs)
        self.assertEqual(diffs["Shape"], 1.0)


if __name__ == "__main__":
    unittest.main()

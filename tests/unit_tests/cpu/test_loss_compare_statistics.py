# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU-only tests for paired statistics in loss_compare.py."""

import contextlib
import io
import math
import unittest

from scripts.loss_compare import (
    assert_losses_equal,
    compare_series,
    generate_summary_statistics,
    perform_loss_analysis,
)


class TestCompareSeries(unittest.TestCase):
    def test_reports_paired_statistics(self):
        comparison = compare_series(
            {1: 1.0, 2: 2.0, 3: 3.0},
            {1: 1.0, 2: 2.5, 3: 2.0},
        )

        self.assertEqual(comparison.baseline_step_count, 3)
        self.assertEqual(comparison.test_step_count, 3)
        self.assertEqual(comparison.paired_step_count, 3)
        self.assertEqual(comparison.matched_count, 1)
        self.assertEqual(comparison.exact_match_count, 1)
        self.assertEqual(comparison.first_divergent_step, 2)
        self.assertAlmostEqual(comparison.mean_absolute_error, 0.5)
        self.assertAlmostEqual(
            comparison.root_mean_square_error,
            math.sqrt(1.25 / 3),
        )
        self.assertEqual(comparison.max_absolute_diff, 1.0)
        self.assertEqual(comparison.max_absolute_diff_step, 3)
        self.assertEqual(comparison.final_step, 3)
        self.assertEqual(comparison.final_diff, -1.0)
        self.assertEqual(
            comparison.differences,
            ((1, 0.0), (2, 0.5), (3, -1.0)),
        )

    def test_only_common_steps_are_paired(self):
        comparison = compare_series({1: 1.0, 3: 3.0}, {2: 2.0, 3: 4.0})

        self.assertEqual(comparison.baseline_step_count, 2)
        self.assertEqual(comparison.test_step_count, 2)
        self.assertEqual(comparison.paired_step_count, 1)
        self.assertEqual(comparison.exact_match_count, 0)
        self.assertEqual(comparison.first_divergent_step, 3)
        self.assertEqual(comparison.max_absolute_diff_step, 3)
        self.assertEqual(comparison.final_step, 3)
        self.assertEqual(comparison.final_diff, 1.0)

    def test_comparator_can_be_shared_with_assertions(self):
        comparison = compare_series(
            {1: 1.0},
            {1: 1.0001},
            comparator=lambda baseline, test: abs(test - baseline) <= 1e-3,
        )

        self.assertEqual(comparison.exact_match_count, 0)
        self.assertEqual(comparison.matched_count, 1)
        self.assertIsNone(comparison.first_divergent_step)


class TestLossCompareReporting(unittest.TestCase):
    def test_summary_contains_paired_statistics(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            generate_summary_statistics(
                {1: 1.0, 2: 2.0},
                {1: 1.0, 2: 3.0},
                None,
                metric_name="grad_norm",
            )

        report = output.getvalue()
        self.assertIn("metric: grad_norm", report)
        self.assertIn("steps: 2/2", report)
        self.assertIn("paired_steps: 2", report)
        self.assertIn("exact_match: 1/2", report)
        self.assertIn("first_divergent_step: 2", report)
        self.assertIn("mae: 5.000000e-01", report)
        self.assertIn("rmse: 7.071068e-01", report)
        self.assertIn("max_abs_diff: 1.000000e+00 at step 2", report)
        self.assertIn("final_diff: 1.000000e+00 (step 2)", report)

    def test_analysis_reports_non_loss_metrics(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            perform_loss_analysis(
                {1: 1.0},
                {1: 2.0},
                None,
                metric_name="grad_norm",
            )

        report = output.getvalue()
        self.assertIn("GRAD NORM COMPARISON ANALYSIS", report)
        self.assertIn("Step-by-step grad norm comparison:", report)


class TestLossCompareAssertions(unittest.TestCase):
    def test_default_assertion_remains_exact(self):
        output = io.StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(
            output
        ), contextlib.redirect_stderr(output):
            assert_losses_equal({1: 1.0}, {1: 1.0001})

    def test_assertion_accepts_shared_comparator(self):
        assert_losses_equal(
            {1: 1.0},
            {1: 1.0001},
            comparator=lambda baseline, test: abs(test - baseline) <= 1e-3,
        )


if __name__ == "__main__":
    unittest.main()

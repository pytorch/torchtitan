# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""CPU-only tests for loss comparison diagnostics."""

import contextlib
import io
import unittest

from scripts.loss_compare import (
    assert_losses_equal,
    generate_summary_statistics,
    perform_loss_analysis,
)


class TestLossCompareReporting(unittest.TestCase):
    def test_summary_contains_exact_divergence_diagnostics(self):
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
        self.assertIn("common_steps: 2", report)
        self.assertIn("exact_matches: 1/2", report)
        self.assertIn("first_divergent_step: 2", report)
        self.assertIn("max_abs_diff: 1.000000e+00 at step 2", report)

    def test_summary_handles_no_common_steps(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            generate_summary_statistics({1: 1.0}, {2: 2.0}, None)

        report = output.getvalue()
        self.assertIn("common_steps: 0", report)
        self.assertIn("exact_matches: 0/0", report)
        self.assertIn("first_divergent_step: N/A", report)
        self.assertIn("max_abs_diff: N/A", report)

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
    def test_assertion_remains_exact(self):
        output = io.StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(
            output
        ), contextlib.redirect_stderr(output):
            assert_losses_equal({1: 1.0}, {1: 1.0001})


if __name__ == "__main__":
    unittest.main()

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Unit tests for local snapshot comparison semantics."""

import unittest

import torch

from .compare import _compare


class SnapshotComparisonTest(unittest.TestCase):
    def test_exact_nested_state(self):
        state = {"weights": torch.arange(8).float(), "optimizer": [torch.ones(2), 3]}
        rows = []
        _compare(state, state, "root", rows)
        self.assertTrue(all(row["exact"] for row in rows))
        self.assertEqual(sum(row.get("numel", 0) for row in rows), 10)

    def test_gradient_corruption_is_reported(self):
        rows = []
        _compare(torch.ones(3), torch.tensor([1.0, 1.0, 2.0]), "grad", rows)
        self.assertFalse(rows[0]["exact"])
        self.assertFalse(rows[0]["allclose_1e_5_1e_6"])
        self.assertEqual(rows[0]["max_abs"], 1.0)

    def test_nan_is_not_equal(self):
        rows = []
        _compare(
            torch.tensor([float("nan")]), torch.tensor([float("nan")]), "grad", rows
        )
        self.assertFalse(rows[0]["exact"])
        self.assertFalse(rows[0]["allclose_1e_5_1e_6"])

    def test_matching_inf_is_not_equal(self):
        rows = []
        _compare(
            torch.tensor([float("inf")]), torch.tensor([float("inf")]), "grad", rows
        )
        self.assertFalse(rows[0]["exact"])
        self.assertFalse(rows[0]["finite"])

    def test_complex_imaginary_difference_is_reported(self):
        rows = []
        _compare(torch.tensor([1 + 2j]), torch.tensor([1 + 3j]), "grad", rows)
        self.assertFalse(rows[0]["exact"])
        self.assertEqual(rows[0]["max_abs"], 1.0)

    def test_metadata_mismatch_raises(self):
        for left, right in [
            (torch.ones(2), torch.ones(3)),
            (torch.ones(2), torch.ones(2, dtype=torch.float64)),
            ({"x": 1}, {"y": 1}),
            ([1, 2], [1]),
        ]:
            with self.subTest(left=left, right=right), self.assertRaises(ValueError):
                _compare(left, right, "root", [])


if __name__ == "__main__":
    unittest.main()

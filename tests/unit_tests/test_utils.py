# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import gc
import unittest
from unittest.mock import MagicMock, patch

import torch

from torchtitan.tools.utils import (
    _round_up,
    check_if_feature_in_pytorch,
    Color,
    GarbageCollection,
    get_peak_flops,
    NoColor,
    set_default_dtype,
)


class TestRoundUp(unittest.TestCase):
    """Tests for the _round_up utility function."""

    def test_exact_multiple(self):
        """x that is already a multiple of y should be returned as-is."""
        self.assertEqual(_round_up(10, 5), 10)
        self.assertEqual(_round_up(16, 8), 16)
        self.assertEqual(_round_up(0, 7), 0)

    def test_non_multiple_rounds_up(self):
        """x that is not a multiple of y should be rounded up."""
        self.assertEqual(_round_up(7, 5), 10)
        self.assertEqual(_round_up(1, 5), 5)
        self.assertEqual(_round_up(9, 8), 16)
        self.assertEqual(_round_up(13, 4), 16)

    def test_x_equals_one(self):
        """x=1 with various y values."""
        self.assertEqual(_round_up(1, 1), 1)
        self.assertEqual(_round_up(1, 2), 2)
        self.assertEqual(_round_up(1, 100), 100)

    def test_large_values(self):
        """Large values should work correctly."""
        self.assertEqual(_round_up(1000001, 1000000), 2000000)
        self.assertEqual(_round_up(1000000, 1000000), 1000000)


class TestGetPeakFlops(unittest.TestCase):
    """Tests for the get_peak_flops function — all GPU name branch coverage."""

    @patch("torchtitan.tools.utils.subprocess.run")
    def test_a100(self, mock_run):
        """A100 should return 312 TFLOPS."""
        mock_run.side_effect = FileNotFoundError("lspci not found")
        result = get_peak_flops("NVIDIA A100-SXM4-80GB")
        self.assertEqual(result, 312e12)

    @patch("torchtitan.tools.utils.subprocess.run")
    def test_h100_sxm(self, mock_run):
        """H100 SXM should return 989 TFLOPS."""
        mock_run.side_effect = FileNotFoundError("lspci not found")
        result = get_peak_flops("NVIDIA H100 SXM")
        self.assertEqual(result, 989e12)

    @patch("torchtitan.tools.utils.subprocess.run")
    def test_h100_nvl(self, mock_run):
        """H100 NVL should return 835 TFLOPS."""
        mock_run.side_effect = FileNotFoundError("lspci not found")
        result = get_peak_flops("NVIDIA H100 NVL")
        self.assertEqual(result, 835e12)

    @patch("torchtitan.tools.utils.subprocess.run")
    def test_h100_pcie(self, mock_run):
        """H100 PCIe should return 756 TFLOPS."""
        mock_run.side_effect = FileNotFoundError("lspci not found")
        result = get_peak_flops("NVIDIA H100 PCIe")
        self.assertEqual(result, 756e12)

    @patch("torchtitan.tools.utils.subprocess.run")
    def test_h200(self, mock_run):
        """H200 should return 989 TFLOPS."""
        mock_run.side_effect = FileNotFoundError("lspci not found")
        result = get_peak_flops("NVIDIA H200")
        self.assertEqual(result, 989e12)

    @patch("torchtitan.tools.utils.subprocess.run")
    def test_h20(self, mock_run):
        """H20 should return 148 TFLOPS."""
        mock_run.side_effect = FileNotFoundError("lspci not found")
        result = get_peak_flops("NVIDIA H20")
        self.assertEqual(result, 148e12)

    @patch("torchtitan.tools.utils.subprocess.run")
    def test_b200(self, mock_run):
        """B200 should return 2.25 PFLOPS."""
        mock_run.side_effect = FileNotFoundError("lspci not found")
        result = get_peak_flops("NVIDIA B200")
        self.assertEqual(result, 2.25e15)

    @patch("torchtitan.tools.utils.subprocess.run")
    def test_mi355x(self, mock_run):
        """MI355X should return 2500 TFLOPS."""
        mock_run.side_effect = FileNotFoundError("lspci not found")
        result = get_peak_flops("AMD MI355X")
        self.assertEqual(result, 2500e12)

    @patch("torchtitan.tools.utils.subprocess.run")
    def test_mi300x(self, mock_run):
        """MI300X should return 1300 TFLOPS."""
        mock_run.side_effect = FileNotFoundError("lspci not found")
        result = get_peak_flops("AMD MI300X")
        self.assertEqual(result, 1300e12)

    @patch("torchtitan.tools.utils.subprocess.run")
    def test_mi325x(self, mock_run):
        """MI325X should return 1300 TFLOPS (same as MI300X)."""
        mock_run.side_effect = FileNotFoundError("lspci not found")
        result = get_peak_flops("AMD MI325X")
        self.assertEqual(result, 1300e12)

    @patch("torchtitan.tools.utils.subprocess.run")
    def test_mi250x(self, mock_run):
        """MI250X should return 191.5 TFLOPS."""
        mock_run.side_effect = FileNotFoundError("lspci not found")
        result = get_peak_flops("AMD MI250X")
        self.assertEqual(result, 191.5e12)

    @patch("torchtitan.tools.utils.subprocess.run")
    def test_l40s(self, mock_run):
        """L40S should return 362 TFLOPS."""
        mock_run.side_effect = FileNotFoundError("lspci not found")
        result = get_peak_flops("NVIDIA l40s")
        self.assertEqual(result, 362e12)

    @patch("torchtitan.tools.utils.subprocess.run")
    def test_unknown_gpu_fallback(self, mock_run):
        """Unknown GPU should fallback to A100 (312 TFLOPS)."""
        mock_run.side_effect = FileNotFoundError("lspci not found")
        result = get_peak_flops("Some Unknown GPU")
        self.assertEqual(result, 312e12)

    @patch("torchtitan.tools.utils.subprocess.run")
    def test_lspci_output_overrides_device_name_for_h100(self, mock_run):
        """When lspci finds H100 lines, those override the device_name argument."""
        mock_result = MagicMock()
        mock_result.stdout = (
            "06:00.0 3D controller: NVIDIA Corporation H100 NVL (rev a1)\n"
        )
        mock_run.return_value = mock_result
        result = get_peak_flops("Some Other GPU")
        # lspci output contains "H100" and "NVL" → 835 TFLOPS
        self.assertEqual(result, 835e12)

    @patch("torchtitan.tools.utils.subprocess.run")
    def test_lspci_no_nvidia_h100_lines(self, mock_run):
        """When lspci runs but has no NVIDIA H100 lines, device_name is used."""
        mock_result = MagicMock()
        mock_result.stdout = "06:00.0 VGA compatible controller: AMD Radeon\n"
        mock_run.return_value = mock_result
        result = get_peak_flops("NVIDIA A100-SXM4-80GB")
        self.assertEqual(result, 312e12)


class TestColorDataclasses(unittest.TestCase):
    """Tests for Color and NoColor dataclasses."""

    def test_color_has_ansi_codes(self):
        """All Color fields should contain ANSI escape codes (non-empty strings)."""
        color = Color()
        for field_name in Color.__dataclass_fields__:
            value = getattr(color, field_name)
            self.assertIsInstance(value, str)
            self.assertTrue(len(value) > 0, f"Color.{field_name} should be non-empty")
            self.assertIn(
                "\033[", value, f"Color.{field_name} should contain ANSI escape"
            )

    def test_nocolor_has_empty_strings(self):
        """All NoColor fields should be empty strings."""
        no_color = NoColor()
        for field_name in NoColor.__dataclass_fields__:
            value = getattr(no_color, field_name)
            self.assertEqual(value, "", f"NoColor.{field_name} should be empty")

    def test_field_parity(self):
        """Color and NoColor must have the same set of fields."""
        self.assertEqual(
            set(Color.__dataclass_fields__.keys()),
            set(NoColor.__dataclass_fields__.keys()),
        )

    def test_frozen_immutability(self):
        """Color and NoColor are frozen dataclasses — attribute assignment should fail."""
        color = Color()
        with self.assertRaises(AttributeError):
            color.red = "changed"

        no_color = NoColor()
        with self.assertRaises(AttributeError):
            no_color.red = "changed"


class TestSetDefaultDtype(unittest.TestCase):
    """Tests for the set_default_dtype context manager."""

    def test_sets_dtype_inside_context(self):
        """Default dtype should be changed inside the context manager."""
        original = torch.get_default_dtype()
        with set_default_dtype(torch.bfloat16):
            self.assertEqual(torch.get_default_dtype(), torch.bfloat16)
        # Restored after exit
        self.assertEqual(torch.get_default_dtype(), original)

    def test_restores_dtype_on_exception(self):
        """Default dtype should be restored even if an exception occurs."""
        original = torch.get_default_dtype()
        with self.assertRaises(ValueError):
            with set_default_dtype(torch.float16):
                self.assertEqual(torch.get_default_dtype(), torch.float16)
                raise ValueError("Test exception")
        self.assertEqual(torch.get_default_dtype(), original)

    def test_nested_contexts(self):
        """Nested contexts should independently restore dtypes."""
        original = torch.get_default_dtype()
        with set_default_dtype(torch.bfloat16):
            self.assertEqual(torch.get_default_dtype(), torch.bfloat16)
            with set_default_dtype(torch.float16):
                self.assertEqual(torch.get_default_dtype(), torch.float16)
            self.assertEqual(torch.get_default_dtype(), torch.bfloat16)
        self.assertEqual(torch.get_default_dtype(), original)


class TestCheckIfFeatureInPytorch(unittest.TestCase):
    """Tests for check_if_feature_in_pytorch warning behavior."""

    @patch("torchtitan.tools.utils.logger")
    def test_warns_for_source_build(self, mock_logger):
        """Should warn when PyTorch is built from source (version contains 'git')."""
        with patch("torchtitan.tools.utils.torch.__version__", "2.6.0a0+git1234abc"):
            check_if_feature_in_pytorch("cool_feature", "https://pr_url")
        mock_logger.warning.assert_called_once()

    @patch("torchtitan.tools.utils.logger")
    def test_warns_for_old_nightly(self, mock_logger):
        """Should warn when version is older than min_nightly_version."""
        with patch("torchtitan.tools.utils.torch.__version__", "2.4.0"):
            check_if_feature_in_pytorch(
                "cool_feature",
                "https://pr_url",
                min_nightly_version="2.5.0",
            )
        mock_logger.warning.assert_called_once()

    @patch("torchtitan.tools.utils.logger")
    def test_no_warn_for_new_version(self, mock_logger):
        """Should not warn when version meets min_nightly_version."""
        with patch("torchtitan.tools.utils.torch.__version__", "2.6.0"):
            check_if_feature_in_pytorch(
                "cool_feature",
                "https://pr_url",
                min_nightly_version="2.5.0",
            )
        mock_logger.warning.assert_not_called()

    @patch("torchtitan.tools.utils.logger")
    def test_no_warn_without_min_version(self, mock_logger):
        """No warning when min_nightly_version is None and not a git build."""
        with patch("torchtitan.tools.utils.torch.__version__", "2.6.0"):
            check_if_feature_in_pytorch("cool_feature", "https://pr_url")
        mock_logger.warning.assert_not_called()


class TestGarbageCollection(unittest.TestCase):
    """Tests for the GarbageCollection utility class."""

    def setUp(self):
        """Ensure GC is enabled before each test."""
        gc.enable()

    def tearDown(self):
        """Always re-enable GC after each test, even on failure."""
        gc.enable()

    def test_init_disables_gc(self):
        """GarbageCollection.__init__ should disable automatic garbage collection."""
        self.assertTrue(gc.isenabled())

        gc_handler = GarbageCollection(gc_freq=100)
        self.assertFalse(gc.isenabled())

    def test_init_invalid_gc_freq(self):
        """gc_freq must be positive."""
        with self.assertRaises(AssertionError):
            GarbageCollection(gc_freq=0)
        with self.assertRaises(AssertionError):
            GarbageCollection(gc_freq=-1)

    def test_run_collects_at_correct_frequency(self):
        """GC.run() should collect at multiples of gc_freq, skipping step 1."""
        gc_handler = GarbageCollection(gc_freq=10)

        with patch.object(GarbageCollection, "collect") as mock_collect:
            # Step 1 should not trigger (because step_count > 1 check fails)
            gc_handler.run(1)
            mock_collect.assert_not_called()

            # Steps 2-9 should not trigger
            for step in range(2, 10):
                gc_handler.run(step)
            mock_collect.assert_not_called()

            # Step 10 should trigger
            gc_handler.run(10)
            mock_collect.assert_called_once()

    def test_collect_static_method(self):
        """GarbageCollection.collect is a static method that calls gc.collect."""
        with patch("torchtitan.tools.utils.gc.collect") as mock_gc_collect:
            GarbageCollection.collect("Test reason", generation=2)
            mock_gc_collect.assert_called_once_with(2)

        # Also test default generation=1
        with patch("torchtitan.tools.utils.gc.collect") as mock_gc_collect:
            GarbageCollection.collect("Another reason")
            mock_gc_collect.assert_called_once_with(1)


if __name__ == "__main__":
    unittest.main()

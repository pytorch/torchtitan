#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Compare RL training losses against a committed reference file for regression
detection.  Follows the same pattern as ``scripts/loss_compare.py`` for
pre-training, but adapted for the Monarch-based RL launcher.

The script runs GRPO training with deterministic settings (seed=42,
deterministic ops) and extracts ``loss/mean`` from TensorBoard events.
It then compares the extracted losses step-by-step against a reference
file using exact floating-point equality.

The reference-file I/O helpers are reused from ``scripts/loss_compare.py``;
only the TensorBoard extraction is RL-specific (see below).  Exactly one of
``--export-result`` / ``--import-result`` must be given -- training is always
a means to either export a new reference or assert against an existing one.

Example usages:

1. Export reference losses (run once on a known-good setup):
   python torchtitan/experiments/rl/scripts/loss_compare.py \\
       --hf-assets-path /path/to/Qwen3-0.6B \\
       --dump-folder /tmp/rl_loss_guard \\
       --export-result tests/assets/losses/rl_grpo_cuda.txt

2. Assert losses match reference (CI mode):
   python torchtitan/experiments/rl/scripts/loss_compare.py \\
       --hf-assets-path /path/to/Qwen3-0.6B \\
       --dump-folder /tmp/rl_loss_guard \\
       --import-result tests/assets/losses/rl_grpo_cuda.txt \\
       --assert-equal
"""

import argparse
import os
import subprocess
import sys
import unittest

LOG_PREFIX = "[RL_LOSS_COMPARE]"

TB_LOSS_TAG = "loss/mean"


def log_print(message: str = "") -> None:
    if message:
        print(f"{LOG_PREFIX} {message}")
    else:
        print(LOG_PREFIX)


# ---------------------------------------------------------------------------
# TensorBoard extraction
# ---------------------------------------------------------------------------


def extract_losses_from_tensorboard(tb_dir: str) -> dict[int, float]:
    """Extract ``loss/mean`` scalars from TensorBoard event files.

    Intentionally RL-specific (not imported from ``scripts/loss_compare.py``):
    the RL ``MetricsProcessor`` writes events directly under ``dump_folder``
    -- there is no per-run ``tb_baseline``/``tb_test`` timestamped subdir --
    and the RL loss is logged under the ``loss/mean`` tag rather than the
    pre-training ``loss_metrics/global_avg_loss`` tag.

    Args:
        tb_dir: Directory containing TensorBoard event files.  The RL
            ``MetricsProcessor`` writes events directly under ``dump_folder``,
            so this is typically the dump folder itself.

    Returns:
        Dictionary mapping step number to full-precision loss value.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    if not os.path.exists(tb_dir):
        raise FileNotFoundError(f"TensorBoard path does not exist: {tb_dir}")

    event_acc = EventAccumulator(tb_dir)
    event_acc.Reload()

    available_tags = event_acc.Tags().get("scalars", [])
    if TB_LOSS_TAG not in available_tags:
        raise KeyError(
            f"Scalar tag '{TB_LOSS_TAG}' not found in TensorBoard events at "
            f"{tb_dir}. Available tags: {available_tags}"
        )

    scalars = event_acc.Scalars(TB_LOSS_TAG)
    losses = {scalar.step: scalar.value for scalar in scalars}

    log_print(f"Extracted {len(losses)} steps from TensorBoard events")
    return losses


# Reference-file I/O (``read_losses_from_file`` / ``export_losses_to_file``) is
# reused from ``scripts/loss_compare.py`` -- imported lazily in ``main`` so this
# module only needs the repo root on ``sys.path`` when actually run.


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def run_training(
    hf_assets_path: str,
    dump_folder: str,
    config: str,
) -> None:
    """Run RL GRPO training as a subprocess."""
    cmd = [
        sys.executable,
        "-m",
        "torchtitan.experiments.rl.train",
        "--module",
        "alphabet_sort",
        "--config",
        config,
        f"--hf_assets_path={hf_assets_path}",
        f"--dump_folder={dump_folder}",
        "--trainer.debug.deterministic",
        "--generator.debug.deterministic",
        "--trainer.debug.seed=42",
        "--generator.debug.seed=42",
        "--async-loop.num-training-steps=10",
        "--async-loop.validation.num-samples=0",
        "--metrics.enable-tensorboard",
        "--metrics.no-enable-wandb",
    ]

    log_print(f"Running: {' '.join(cmd)}")

    result = subprocess.run(cmd, env=os.environ)
    if result.returncode != 0:
        log_print(f"Training failed with return code {result.returncode}")
        sys.exit(1)

    log_print("Training completed successfully")


# ---------------------------------------------------------------------------
# Assertion
# ---------------------------------------------------------------------------


def assert_losses_equal(
    actual: dict[int, float],
    expected: dict[int, float],
) -> None:
    """Assert exact floating-point equality between actual and expected losses."""
    log_print(
        f"Comparing {len(actual)} actual steps against {len(expected)} expected steps"
    )

    class LossEqualityTest(unittest.TestCase):
        def test_losses_equal(self):
            actual_steps = set(actual.keys())
            expected_steps = set(expected.keys())
            self.assertEqual(
                actual_steps,
                expected_steps,
                f"Steps mismatch: actual has {len(actual_steps)} steps, "
                f"expected has {len(expected_steps)} steps",
            )

            for step in sorted(actual_steps):
                self.assertEqual(
                    actual[step],
                    expected[step],
                    f"Loss mismatch at step {step}: "
                    f"actual={repr(actual[step])}, expected={repr(expected[step])}",
                )

    suite = unittest.TestLoader().loadTestsFromTestCase(LossEqualityTest)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if not result.wasSuccessful():
        log_print("Loss assertion FAILED!")
        log_print()
        log_print(
            "Actual losses (use these to update the reference file if "
            "the change is intentional):"
        )
        log_print("WARNING: verify the loss curve change is not a regression first!")
        for step in sorted(actual.keys()):
            print(f"{step} {repr(actual[step])}")
        log_print()
        sys.exit(1)
    else:
        log_print("All losses match. Assertion passed!")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare RL training losses against a reference file.",
    )
    parser.add_argument(
        "--hf-assets-path",
        required=True,
        help="Path to HF model checkpoint (weights, tokenizer, config).",
    )
    parser.add_argument(
        "--dump-folder",
        required=True,
        help="Directory for training outputs and TensorBoard events.",
    )
    parser.add_argument(
        "--config",
        default="rl_grpo_qwen3_0_6b_varlen_batch_invariant",
        help="RL config name (default: rl_grpo_qwen3_0_6b_varlen_batch_invariant).",
    )
    parser.add_argument(
        "--export-result",
        default="",
        help="Export losses to this file path after training.",
    )
    parser.add_argument(
        "--import-result",
        default="",
        help="Reference file to compare against (requires --assert-equal).",
    )
    parser.add_argument(
        "--assert-equal",
        action="store_true",
        help="Assert losses match the reference file exactly.",
    )

    args = parser.parse_args()

    # Reuse the reference-file helpers from the pre-training loss_compare so we
    # don't maintain two copies. That script lives at the repo root under
    # scripts/, which the editable install does not expose on sys.path, so add
    # the repo root before importing (works whether run as a script or -m).
    repo_root = os.path.abspath(
        os.path.join(os.path.dirname(__file__), *([os.pardir] * 4))
    )
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    from scripts.loss_compare import export_losses_to_file, read_losses_from_file

    export_result = args.export_result or None
    import_result = args.import_result or None

    if not export_result and not import_result:
        parser.error("one of --export-result / --import-result is required")

    if args.assert_equal and not import_result:
        parser.error("--assert-equal requires --import-result")

    if import_result:
        args.assert_equal = True

    # Run training
    run_training(args.hf_assets_path, args.dump_folder, args.config)

    # Extract losses from TensorBoard
    actual_losses = extract_losses_from_tensorboard(args.dump_folder)

    if not actual_losses:
        log_print("Error: no losses found in TensorBoard events")
        sys.exit(1)

    # Export if requested
    if export_result:
        export_losses_to_file(actual_losses, export_result)

    # Compare if requested
    if args.assert_equal:
        expected_losses = read_losses_from_file(import_result)
        assert_losses_equal(actual_losses, expected_losses)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This script compares training losses between different git commits
and/or different training configurations. --debug.deterministic is
always enabled and seed checkpoint is also enabled by default for
reproducible comparisons. You can disable seed checkpoint with
--no-seed-checkpoint if you don't need it to speed up comparisons.
If --output-folder is specified, all outputs are organized in that
folder with detailed analysis and statistical summaries.

The --assert-equal flag can be used for CI testing to verify that
losses are identical between runs. If losses differ, the script will
exit with a non-zero status code.

Example usages:
1. Compare losses between two different git commits with default config:
   loss_compare.py main my_branch

2. Compare losses between two commits with custom config and options:
   loss_compare.py main my_branch \
       --baseline-config='llama3_8b' \
       --baseline-options='--parallelism.tensor_parallel_degree=2' \
       --output-folder=my_comparison

3. Compare losses between two commits using a different model module:
   loss_compare.py main my_branch \
       --baseline-module='qwen3' --baseline-config='qwen3_debugmodel'

4. Compare commits with the same command but skip seed checkpoint for
   faster execution:
   loss_compare.py main my_branch --no-seed-checkpoint

5. Compare the same commit with different training configurations:
   loss_compare.py . . \
       --baseline-options='--parallelism.dp=1' \
       --test-options='--parallelism.dp=2'

6. Assert that losses are equal (for CI testing):
   loss_compare.py main my_branch --assert-equal

7. Run baseline only and compare against imported losses (baseline-only mode):
   loss_compare.py . . --assert-equal --import-result=expected_losses.txt

8. Run baseline only with specific config and compare against imported losses:
   loss_compare.py . . --assert-equal --import-result=expected_losses.txt \
       --baseline-config='llama3_8b'

9. Run baseline only and export the losses (no comparison):
   loss_compare.py . . --export-result=baseline_losses.txt

10. Run baseline with specific options and export the losses:
    loss_compare.py . . --baseline-options='--parallelism.dp=2' \
        --export-result=my_config_losses.txt
"""

import argparse
import os
import shutil
import subprocess
import sys
import unittest
from typing import Any

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

LOG_PREFIX = "[LOSS_COMPARE]"

# TensorBoard scalar tag used to extract loss values
TB_LOSS_TAG = "loss_metrics/global_avg_loss"

# Fixed options that are always appended
FIXED_OPTIONS = "--debug.deterministic --debug.seed=42 --metrics.enable_tensorboard --metrics.log_freq=1"


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def log_print(message: str = "") -> None:
    """Print message with LOG_PREFIX."""
    if message:
        print(f"{LOG_PREFIX} {message}")
    else:
        print(f"{LOG_PREFIX}")


def get_log_path(scenario: str, output_folder: str | None) -> str:
    """Get log file path for a scenario."""
    if output_folder:
        return f"{output_folder}/{scenario}_training.log"
    return f"/tmp/{scenario}_training.log"


def build_base_command(
    module: str, config: str, options: str, job_dump_folder: str
) -> str:
    """Build the base command from module, config, and options."""
    cmd = f"MODULE='{module}' CONFIG='{config}' ./run_train.sh"
    cmd += f" --dump_folder={job_dump_folder}"
    if options:
        cmd += f" {options}"
    return cmd


def run_with_realtime_output(cmd: str, logfile: str, env: dict[str, Any]) -> None:
    """Run command with real-time output to both console and log file."""
    log_print(f"Executing: {cmd}")

    # Set PYTHONUNBUFFERED for better output handling
    env["PYTHONUNBUFFERED"] = "1"

    # Run command and tee output to both stdout and log file
    with open(logfile, "w") as log_f:
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
            bufsize=1,
        )

        # pyrefly: ignore [not-iterable]
        for line in process.stdout:
            print(line, end="")
            log_f.write(line)
            log_f.flush()

        process.wait()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)


def extract_losses_from_tensorboard(
    job_dump_folder: str, tb_folder: str
) -> dict[int, float]:
    """Extract full-precision loss values from TensorBoard event files.

    The TB directory is cleared before each run (see ``run_training``), so
    there is exactly one timestamped subdirectory.  We find it and point
    ``EventAccumulator`` at it directly.

    Args:
        job_dump_folder: The --job-dump-folder value (e.g., "outputs")
        tb_folder: The TB subfolder name (e.g., "tb_baseline")

    Returns:
        Dictionary mapping step number to full-precision loss value.
    """
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    base_path = os.path.join(job_dump_folder, tb_folder)
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"TensorBoard path does not exist: {base_path}")

    # Find the single timestamped subdirectory (e.g., "20260306-1618")
    subdirs = [
        d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))
    ]
    if len(subdirs) == 1:
        event_dir = os.path.join(base_path, subdirs[0])
    else:
        # Should not happen since we clear the directory before each run
        raise RuntimeError(
            f"Expected exactly one subdirectory under {base_path}, "
            f"found {len(subdirs)}: {subdirs}"
        )

    log_print(f"Loading TensorBoard events from: {event_dir}")

    event_acc = EventAccumulator(event_dir)
    event_acc.Reload()

    scalar_tag = TB_LOSS_TAG
    available_tags = event_acc.Tags().get("scalars", [])

    if scalar_tag not in available_tags:  # pyrefly: ignore [not-iterable]
        raise KeyError(
            f"Scalar tag '{scalar_tag}' not found in TensorBoard events. "
            f"Available tags: {available_tags}"
        )

    scalars = event_acc.Scalars(scalar_tag)
    losses = {scalar.step: scalar.value for scalar in scalars}

    log_print(f"Extracted {len(losses)} steps from TensorBoard events")
    return losses


def log_and_save(message: str, stats_file: str | None) -> None:
    """Output message to both stdout and stats file if provided."""
    print(message)
    if stats_file:
        with open(stats_file, "a") as f:
            f.write(message + "\n")


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_arguments(
    baseline_commit: str,
    test_commit: str,
    baseline_module: str,
    baseline_config: str,
    baseline_options: str,
    test_module: str,
    test_config: str,
    test_options: str,
    steps: int,
    assert_equal: bool,
    export_result: str | None,
    import_result: str | None,
) -> bool:
    """Validate command line arguments.

    Returns:
        True if baseline-only mode (all settings identical with import_result),
        False otherwise.
    """
    # Validate that we are comparing different settings
    commits_differ = baseline_commit != test_commit
    configs_differ = baseline_config != test_config
    modules_differ = baseline_module != test_module
    options_differ = baseline_options != test_options

    all_identical = not (
        commits_differ or configs_differ or modules_differ or options_differ
    )

    # Determine baseline-only mode:
    # - With --export-result: always run baseline only (export the losses)
    # - With --import-result and --assert-equal: run baseline, compare against imported
    baseline_only = export_result is not None or (
        all_identical and import_result is not None and assert_equal
    )

    if export_result:
        log_print("Baseline-only mode: --export-result specified")
        log_print("Will run baseline only and export the losses")
    elif all_identical and import_result and assert_equal:
        log_print("Baseline-only mode: all settings identical with --import-result")
        log_print("Will run baseline only and compare against imported losses")
    elif all_identical:
        log_print("Error: All settings are identical")
        log_print("       Cannot compare identical configurations")
        log_print(
            "       Please provide different commits, configs, modules, or options"
        )
        log_print(
            "       Or use --import-result with --assert-equal "
            "or --export-result to run baseline-only mode"
        )
        sys.exit(1)

    # Validate steps is a positive integer
    if steps <= 0:
        log_print(f"Error: --steps must be a positive integer, got: {steps}")
        sys.exit(1)

    # Validate import-result requires assert-equal
    if import_result and not assert_equal:
        log_print("Error: --import-result requires --assert-equal")
        log_print("       Import is used to verify all losses match")
        sys.exit(1)

    # Validate export-result and import-result are mutually exclusive
    if export_result and import_result:
        log_print(
            "Error: --export-result and --import-result cannot be " "used together"
        )
        log_print(
            "       Use export to save results or import to compare "
            "against saved results"
        )
        sys.exit(1)

    # Validate import file exists
    if import_result and not os.path.exists(import_result):
        log_print(f"Error: Import file does not exist: {import_result}")
        sys.exit(1)

    # Return whether we're in baseline-only mode
    return baseline_only


# =============================================================================
# SETUP FUNCTIONS
# =============================================================================


def setup_output_directory(output_folder: str | None) -> str | None:
    """Setup output directory and return stats file path.
    Returns None if no output folder specified.
    """
    if not output_folder:
        return None

    # Check if output folder already exists
    if os.path.exists(output_folder):
        log_print(f"Error: Output folder '{output_folder}' already exists")
        log_print(f"Please delete it first: rm -rf {output_folder}")
        sys.exit(1)

    # Create the output folder
    log_print(f"Creating output folder: {output_folder}")
    os.makedirs(output_folder)

    # Set statistics file path
    stats_file = os.path.join(output_folder, "comparison_statistics.txt")
    return stats_file


def build_training_command(
    module: str,
    config: str,
    options: str,
    steps: int,
    enable_seed_checkpoint: bool,
    job_dump_folder: str,
    tb_folder: str = "tb",
) -> str:
    """Build the final training command with all options."""
    base_cmd = build_base_command(module, config, options, job_dump_folder)
    cmd = f"{base_cmd} {FIXED_OPTIONS} --training.steps={steps}"
    cmd += f" --metrics.save_tb_folder={tb_folder}"
    if enable_seed_checkpoint:
        cmd += (
            " --checkpoint.enable --checkpoint.export_dtype=bfloat16"
            " --checkpoint.load_only"
        )
    return cmd


def print_configuration(
    baseline_commit: str,
    test_commit: str,
    baseline_module: str,
    baseline_config: str,
    baseline_options: str,
    test_module: str,
    test_config: str,
    test_options: str,
    steps: int,
    enable_seed_checkpoint: bool,
    job_dump_folder: str,
    baseline_only_mode: bool = False,
    baseline_tb_folder: str = "tb_baseline",
    test_tb_folder: str = "tb_test",
) -> None:
    """Print configuration summary."""
    if baseline_only_mode:
        log_print(f"Starting baseline-only run with commit: {baseline_commit}")
    else:
        log_print(
            f"Starting loss comparison between baseline commit: "
            f"{baseline_commit} and test commit: {test_commit}"
        )
    log_print(f"Training steps: {steps}")
    log_print(f"Seed checkpoint enabled: {enable_seed_checkpoint}")
    log_print()

    # Build and display final commands
    baseline_final_cmd = build_training_command(
        baseline_module,
        baseline_config,
        baseline_options,
        steps,
        enable_seed_checkpoint,
        job_dump_folder,
        tb_folder=baseline_tb_folder,
    )

    log_print("Baseline command:")
    log_print(f"  {baseline_final_cmd}")
    log_print()

    if not baseline_only_mode:
        test_final_cmd = build_training_command(
            test_module,
            test_config,
            test_options,
            steps,
            enable_seed_checkpoint,
            job_dump_folder,
            tb_folder=test_tb_folder,
        )
        log_print("Test command:")
        log_print(f"  {test_final_cmd}")
        log_print()


# =============================================================================
# GIT OPERATIONS
# =============================================================================


def check_git_clean_state() -> None:
    """Check if git working directory is clean before switching commits.

    Raises SystemExit if there are uncommitted changes to tracked files.
    Untracked files are ignored.
    """
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        capture_output=True,
        text=True,
        check=True,
    )

    # Filter out untracked files (lines starting with "??")
    modified_tracked_files = []
    for line in result.stdout.strip().split("\n"):
        if line and not line.startswith("??"):
            modified_tracked_files.append(line)

    if modified_tracked_files:
        log_print(
            "Error: Git working directory has uncommitted changes to tracked files"
        )
        log_print("       Cannot switch commits with uncommitted changes")
        log_print("")
        log_print("Modified tracked files:")
        for line in modified_tracked_files:
            log_print(f"  {line}")
        log_print("")
        log_print(
            "Please commit, stash, or discard your changes before running this script"
        )
        log_print("  - To commit: git add -A && git commit -m 'message'")
        log_print("  - To stash: git stash")
        log_print("  - To discard: git checkout -- . && git clean -fd")
        sys.exit(1)


def checkout_commit(commit: str, commit_name: str) -> None:
    """Checkout git commit."""
    if commit != ".":
        log_print(f"Checking out {commit_name} commit: {commit}")
        subprocess.run(["git", "checkout", commit], check=True)
    else:
        log_print(f"Using current working directory for {commit_name} (commit: '.')")


def get_current_commit() -> str:
    """Get the current git commit hash or branch name.

    Returns the current branch name if on a branch, otherwise returns the commit hash.
    """
    # Try to get current branch name
    result = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    ref = result.stdout.strip()

    # If in detached HEAD state, ref will be "HEAD", so get the commit hash instead
    if ref == "HEAD":
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        ref = result.stdout.strip()

    return ref


def restore_original_commit(original_commit: str) -> None:
    """Restore the original git commit/branch."""
    log_print(f"Restoring original commit/branch: {original_commit}")
    subprocess.run(["git", "checkout", original_commit], check=True)


# =============================================================================
# TRAINING OPERATIONS
# =============================================================================


def create_seed_checkpoint(
    enable_seed_checkpoint: bool,
    module: str,
    config: str,
    output_folder: str | None,
    job_dump_folder: str,
) -> None:
    """Create seed checkpoint."""
    if enable_seed_checkpoint:
        log_file = get_log_path("seed", output_folder)
        log_print(f"Creating seed checkpoint and logging output to {log_file}")

        # Build seed checkpoint command
        seed_cmd = (
            f"MODULE='{module}' CONFIG='{config}' "
            f"./run_train.sh --dump_folder={job_dump_folder} "
            f"--checkpoint.create_seed_checkpoint "
            f"--checkpoint.enable {FIXED_OPTIONS}"
        )

        env = os.environ.copy()
        env["NGPU"] = "1"

        run_with_realtime_output(seed_cmd, log_file, env)


def run_training(
    scenario: str,
    module: str,
    config: str,
    options: str,
    steps: int,
    enable_seed_checkpoint: bool,
    output_folder: str | None,
    job_dump_folder: str,
    ngpus: int,
    tb_folder: str = "tb",
) -> str:
    """Run training for a specific scenario. Returns the log file path."""
    log_file = get_log_path(scenario, output_folder)
    log_print(
        f"Running training with {scenario} commit and logging output " f"to {log_file}"
    )

    # Clear stale TensorBoard data so EventAccumulator only sees this run
    tb_dir = os.path.join(job_dump_folder, tb_folder)
    if os.path.exists(tb_dir):
        log_print(f"Removing stale TensorBoard directory: {tb_dir}")
        shutil.rmtree(tb_dir)

    # Build the final command
    full_cmd = build_training_command(
        module,
        config,
        options,
        steps,
        enable_seed_checkpoint,
        job_dump_folder,
        tb_folder=tb_folder,
    )

    env = os.environ.copy()
    env["NGPU"] = str(ngpus)

    run_with_realtime_output(full_cmd, log_file, env)

    return log_file


# =============================================================================
# LOG PROCESSING AND ANALYSIS
# =============================================================================


def read_losses_from_file(loss_file: str) -> dict[int, float]:
    """Read losses from a processed loss file."""
    losses = {}
    with open(loss_file, "r") as f:
        for line in f:
            step, loss = line.strip().split()
            losses[int(step)] = float(loss)
    return losses


def export_losses_to_file(losses: dict[int, float], export_path: str) -> None:
    """Export losses to file and stdout.

    Uses repr() for float formatting to preserve full round-trip precision.

    Args:
        losses: Dictionary mapping step numbers to loss values
        export_path: Path to export file
    """
    log_print(f"Exporting losses to {export_path}")

    # Write to file and collect output for stdout
    with open(export_path, "w") as f:
        for step in sorted(losses.keys()):
            loss = losses[step]
            line = f"{step} {repr(loss)}"
            f.write(line + "\n")

    log_print(f"Exported {len(losses)} loss values:")
    log_print()

    # Output to stdout in same format
    for step in sorted(losses.keys()):
        loss = losses[step]
        print(f"{step} {repr(loss)}")

    log_print()
    log_print(f"Losses saved to: {export_path}")


def generate_step_comparison(
    baseline_losses: dict[int, float],
    test_losses: dict[int, float],
    stats_file: str | None,
) -> None:
    """Generate step-by-step comparison."""
    log_and_save("", stats_file)
    log_and_save(f"{LOG_PREFIX} Step-by-step loss comparison:", stats_file)
    log_and_save(
        f"{LOG_PREFIX} Step    Baseline Loss    Test Loss   Difference",
        stats_file,
    )
    log_and_save(
        f"{LOG_PREFIX} ----    -------------    ---------   ----------",
        stats_file,
    )

    # Generate comparison for common steps
    for step in sorted(set(baseline_losses.keys()) & set(test_losses.keys())):
        baseline_loss = baseline_losses[step]
        test_loss = test_losses[step]
        diff = test_loss - baseline_loss

        formatted_line = (
            f"{LOG_PREFIX} {step:<6}  {baseline_loss:<13}    "
            f"{test_loss:<14}   {diff:.6f}"
        )
        log_and_save(formatted_line, stats_file)


def generate_summary_statistics(
    baseline_losses: dict[int, float],
    test_losses: dict[int, float],
    stats_file: str | None,
) -> None:
    """Generate summary statistics."""
    log_and_save(f"{LOG_PREFIX}", stats_file)
    log_and_save(f"{LOG_PREFIX} Summary statistics:", stats_file)

    # Calculate average losses
    def calculate_average(losses: dict[int, float]) -> float | None:
        """Calculate average loss from losses dict."""
        if not losses:
            return None
        return sum(losses.values()) / len(losses)

    baseline_avg = calculate_average(baseline_losses)
    test_avg = calculate_average(test_losses)

    baseline_avg_str = f"{baseline_avg}" if baseline_avg is not None else "N/A"
    test_avg_str = f"{test_avg}" if test_avg is not None else "N/A"

    log_and_save(f"{LOG_PREFIX} Average baseline loss:  {baseline_avg_str}", stats_file)
    log_and_save(f"{LOG_PREFIX} Average test loss: {test_avg_str}", stats_file)

    # Calculate overall difference if both averages are available
    if baseline_avg is not None and test_avg is not None:
        avg_diff = test_avg - baseline_avg
        log_and_save(f"{LOG_PREFIX} Average difference:     {avg_diff:.6f}", stats_file)


def perform_loss_analysis(
    baseline_losses: dict[int, float],
    test_losses: dict[int, float],
    stats_file: str | None,
) -> None:
    """Perform loss comparison analysis."""
    # Initialize stats file and add header
    log_and_save(f"{LOG_PREFIX} ==========================================", stats_file)
    log_and_save(f"{LOG_PREFIX} LOSS COMPARISON ANALYSIS", stats_file)
    log_and_save(f"{LOG_PREFIX} ==========================================", stats_file)

    # Check if losses were extracted successfully
    name_losses = [("baseline", baseline_losses), ("test", test_losses)]
    for name, losses in name_losses:
        if not losses:
            log_and_save(
                f"{LOG_PREFIX} Warning: No loss data for {name}.",
                stats_file,
            )
            log_and_save(
                f"{LOG_PREFIX} Please check that the training completed "
                "successfully.",
                stats_file,
            )
            return

    # Generate comparison outputs
    generate_step_comparison(baseline_losses, test_losses, stats_file)
    generate_summary_statistics(baseline_losses, test_losses, stats_file)


def assert_losses_equal(
    baseline_losses: dict[int, float],
    test_losses: dict[int, float] | None = None,
    import_result: str | None = None,
) -> None:
    """Assert that losses are equal between baseline and test using unittest.

    Args:
        baseline_losses: Baseline loss values extracted from TensorBoard.
        test_losses: Test loss values extracted from TensorBoard. If None,
            only compares baseline against imported losses (baseline-only mode).
        import_result: Path to imported losses file for comparison.

    In baseline-only mode (test_losses is None), import_result must be provided.
    """
    log_print("Asserting losses are equal...")
    log_print(f"Baseline: {len(baseline_losses)} steps")
    if test_losses is not None:
        log_print(f"Test: {len(test_losses)} steps")
    else:
        log_print("Test: None (baseline-only mode)")
    if import_result:
        log_print(f"Import file: {import_result}")

    # Validate baseline-only mode has import_result
    if test_losses is None and import_result is None:
        log_print("Error: baseline-only mode requires --import-result")
        sys.exit(1)

    if not baseline_losses:
        log_print("Error: No losses found in baseline")
        sys.exit(1)

    if test_losses is not None and not test_losses:
        log_print("Error: No losses found in test")
        sys.exit(1)

    # Load imported losses if provided
    imported_losses = None
    if import_result:
        imported_losses = read_losses_from_file(import_result)
        log_print(f"Loaded {len(imported_losses)} steps from import file")
        if not imported_losses:
            log_print("Error: No losses found in import file")
            sys.exit(1)

    # Create a test case
    class LossEqualityTest(unittest.TestCase):
        def test_losses_equal(self):
            baseline_steps = set(baseline_losses.keys())

            # Check baseline vs test if test exists
            if test_losses is not None:
                test_steps = set(test_losses.keys())
                self.assertEqual(
                    baseline_steps,
                    test_steps,
                    f"Steps mismatch: baseline has {len(baseline_steps)} steps, "
                    f"test has {len(test_steps)} steps",
                )

            # If imported losses exist, check steps match
            if imported_losses:
                imported_steps = set(imported_losses.keys())
                self.assertEqual(
                    baseline_steps,
                    imported_steps,
                    f"Steps mismatch: baseline has {len(baseline_steps)} steps, "
                    f"imported has {len(imported_steps)} steps",
                )

            # Check that losses are equal for each step
            for step in sorted(baseline_steps):
                baseline_loss = baseline_losses[step]

                # Compare baseline vs test (if test exists)
                if test_losses is not None:
                    test_loss = test_losses[step]
                    self.assertEqual(
                        baseline_loss,
                        test_loss,
                        f"Loss mismatch at step {step}: "
                        f"baseline={repr(baseline_loss)}, test={repr(test_loss)}",
                    )

                # Compare baseline vs imported (if provided)
                if imported_losses:
                    imported_loss = imported_losses[step]
                    self.assertEqual(
                        baseline_loss,
                        imported_loss,
                        f"Loss mismatch at step {step}: "
                        f"baseline={repr(baseline_loss)}, "
                        f"imported={repr(imported_loss)}",
                    )

    # Run the test
    suite = unittest.TestLoader().loadTestsFromTestCase(LossEqualityTest)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if not result.wasSuccessful():
        log_print("Loss assertion failed!")
        log_print()
        log_print(
            "Actual baseline losses (can be used to update import file if "
            "the loss curve change is expected):"
        )
        log_print(
            "Note that you should verify the loss curve change is not a "
            "regression first!!!"
        )
        for step in sorted(baseline_losses.keys()):
            loss = baseline_losses[step]
            print(f"{step} {repr(loss)}")
        log_print()
        sys.exit(1)
    else:
        if test_losses is not None and import_result:
            log_print(
                "All losses are equal (baseline, test, and imported). "
                "Assertion passed!"
            )
        elif test_losses is not None:
            log_print("All losses are equal (baseline and test). Assertion passed!")
        else:
            log_print("All losses are equal (baseline and imported). Assertion passed!")


# =============================================================================
# OUTPUT FUNCTIONS
# =============================================================================


def print_completion_summary(
    output_folder: str | None,
    enable_seed_checkpoint: bool,
    baseline_only_mode: bool = False,
) -> None:
    """Print completion summary."""
    log_print()
    if output_folder:
        if baseline_only_mode:
            log_print(f"Baseline run complete. Results saved in {output_folder}/:")
        else:
            log_print(f"Loss comparison complete. Results saved in {output_folder}/:")
        log_print("  - baseline_outputs/")
        if not baseline_only_mode:
            log_print("  - test_outputs/")
        if enable_seed_checkpoint:
            log_print("  - seed_checkpoint_outputs/")
        log_print()
        log_print(f"Training logs saved in {output_folder}/:")
        if enable_seed_checkpoint:
            log_print("  - seed_checkpoint.log")
        log_print("  - baseline_training.log")
        if not baseline_only_mode:
            log_print("  - test_training.log")
        log_print()
        log_print(f"All outputs organized in: {output_folder}/")
    else:
        if baseline_only_mode:
            log_print(
                "Baseline run complete. No results saved "
                "(no output folder specified)."
            )
        else:
            log_print(
                "Loss comparison complete. No results saved "
                "(no output folder specified)."
            )


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare training losses between different git commits "
            "and/or different training configurations."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s abc123 def456
  %(prog)s abc123 def456 --steps=200
  %(prog)s abc123 def456 --baseline-config='llama3_8b' \\
      --baseline-options='--parallelism.tensor_parallel_degree=2' --steps=50
  %(prog)s abc123 def456 --no-seed-checkpoint
  %(prog)s . . --baseline-options='--parallelism.dp=1' \\
      --test-options='--parallelism.dp=2' --steps=30
        """,
    )

    parser.add_argument("baseline_commit", help="Git commit hash for baseline")
    parser.add_argument("test_commit", help="Git commit hash for test")
    parser.add_argument(
        "--baseline-module",
        default="llama3",
        help="Module name for baseline run (default: llama3)",
    )
    parser.add_argument(
        "--test-module",
        default="",
        help="Module name for test run (default: uses baseline-module)",
    )
    parser.add_argument(
        "--baseline-config",
        default="llama3_debugmodel",
        help="Config name for baseline run (default: llama3_debugmodel)",
    )
    parser.add_argument(
        "--test-config",
        default="",
        help="Config name for test run (default: uses baseline-config)",
    )
    parser.add_argument(
        "--baseline-options",
        default="",
        help="Additional CLI arguments for baseline run (default: empty)",
    )
    parser.add_argument(
        "--test-options",
        default="",
        help="Additional CLI arguments for test run (default: empty)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=100,
        help="Number of training steps (default: 100)",
    )
    parser.add_argument(
        "--no-seed-checkpoint",
        action="store_true",
        help=("Disable seed checkpoint creation and checkpoint functionality"),
    )
    parser.add_argument(
        "--output-folder",
        default="",
        help=(
            "Output folder for results (optional, if not specified, "
            "results will not be saved)"
        ),
    )
    parser.add_argument(
        "--assert-equal",
        action="store_true",
        help=(
            "Assert that all losses are equal (for CI testing). "
            "Script exits with error if losses differ."
        ),
    )
    parser.add_argument(
        "--export-result",
        default="",
        help=(
            "Export losses to specified file path (requires --assert-equal). "
            "Exports only when losses match. Format: '{step} {loss}' per line."
        ),
    )
    parser.add_argument(
        "--import-result",
        default="",
        help=(
            "Import losses from specified file path for comparison "
            "(requires --assert-equal). "
            "Compares imported losses with both baseline and test "
            "(all 3 must match)."
        ),
    )
    parser.add_argument(
        "--job-dump-folder",
        default="outputs",
        help="Job dump folder path (default: outputs)",
    )
    parser.add_argument(
        "--baseline-ngpus",
        type=int,
        default=8,
        help="Number of GPUs for baseline run (default: 8)",
    )
    parser.add_argument(
        "--test-ngpus",
        type=int,
        default=8,
        help="Number of GPUs for test run (default: 8)",
    )

    args = parser.parse_args()

    # Set default values if not provided
    if not args.test_module:
        args.test_module = args.baseline_module

    if not args.test_config:
        args.test_config = args.baseline_config

    # Convert empty output_folder to None
    if not args.output_folder:
        args.output_folder = None

    # Convert empty export_result to None
    if not args.export_result:
        args.export_result = None

    # Convert empty import_result to None
    if not args.import_result:
        args.import_result = None

    return args


def run_scenario(
    scenario: str,
    commit: str,
    module: str,
    config: str,
    options: str,
    steps: int,
    enable_seed_checkpoint: bool,
    output_folder: str | None,
    job_dump_folder: str,
    ngpus: int,
    tb_folder: str = "tb",
) -> str:
    """Run training for a specific scenario (baseline or test).

    Args:
        scenario: Name of the scenario ("baseline" or "test")
        commit: Git commit to checkout
        module: Module name (e.g., "llama3")
        config: Config name (e.g., "llama3_debugmodel")
        options: Additional CLI options
        steps: Number of training steps
        enable_seed_checkpoint: Whether to use seed checkpoint
        output_folder: Output folder for results
        job_dump_folder: Job dump folder path
        ngpus: Number of GPUs to use
        tb_folder: TensorBoard subfolder name for this scenario

    Returns:
        Path to the log file
    """
    checkout_commit(commit, scenario)

    log_file = run_training(
        scenario,
        module,
        config,
        options,
        steps,
        enable_seed_checkpoint,
        output_folder,
        job_dump_folder,
        ngpus,
        tb_folder=tb_folder,
    )

    return log_file


def main() -> None:
    """Main function that orchestrates the entire comparison process."""
    # Parse and validate arguments
    args = parse_arguments()
    baseline_only_mode = validate_arguments(
        args.baseline_commit,
        args.test_commit,
        args.baseline_module,
        args.baseline_config,
        args.baseline_options,
        args.test_module,
        args.test_config,
        args.test_options,
        args.steps,
        args.assert_equal,
        args.export_result,
        args.import_result,
    )

    # Setup environment
    stats_file = setup_output_directory(args.output_folder)
    enable_seed_checkpoint = not args.no_seed_checkpoint

    # Define per-scenario TensorBoard folder names
    baseline_tb_folder = "tb_baseline"
    test_tb_folder = "tb_test"

    print_configuration(
        args.baseline_commit,
        args.test_commit,
        args.baseline_module,
        args.baseline_config,
        args.baseline_options,
        args.test_module,
        args.test_config,
        args.test_options,
        args.steps,
        enable_seed_checkpoint,
        args.job_dump_folder,
        baseline_only_mode,
        baseline_tb_folder=baseline_tb_folder,
        test_tb_folder=test_tb_folder,
    )

    # Check if git working directory is clean before switching commits
    # Skip check only if both commits are "." (comparing configs on same commit)
    needs_git_checkout = args.baseline_commit != "." or args.test_commit != "."
    if needs_git_checkout:
        check_git_clean_state()

    # Save original commit if we're going to do checkouts
    original_commit = None
    if needs_git_checkout:
        original_commit = get_current_commit()
        log_print(f"Saving original commit/branch: {original_commit}")
        log_print()

    try:
        create_seed_checkpoint(
            enable_seed_checkpoint,
            args.baseline_module,
            args.baseline_config,
            args.output_folder,
            args.job_dump_folder,
        )
        # Run baseline training
        baseline_log = run_scenario(
            "baseline",
            args.baseline_commit,
            args.baseline_module,
            args.baseline_config,
            args.baseline_options,
            args.steps,
            enable_seed_checkpoint,
            args.output_folder,
            args.job_dump_folder,
            args.baseline_ngpus,
            tb_folder=baseline_tb_folder,
        )

        # Extract baseline losses from TensorBoard (full precision)
        baseline_losses = extract_losses_from_tensorboard(
            args.job_dump_folder, baseline_tb_folder
        )

        # Run test training (skip in baseline-only mode)
        test_log = None
        test_losses = None
        if not baseline_only_mode:
            test_log = run_scenario(
                "test",
                args.test_commit,
                args.test_module,
                args.test_config,
                args.test_options,
                args.steps,
                enable_seed_checkpoint,
                args.output_folder,
                args.job_dump_folder,
                args.test_ngpus,
                tb_folder=test_tb_folder,
            )

            # Extract test losses from TensorBoard (full precision)
            test_losses = extract_losses_from_tensorboard(
                args.job_dump_folder, test_tb_folder
            )
        log_print()

        # Assert losses are equal if requested
        if args.assert_equal:
            assert_losses_equal(baseline_losses, test_losses, args.import_result)

            # Export losses if requested (only after assertion passes)
            if args.export_result:
                export_losses_to_file(baseline_losses, args.export_result)

        # Export losses in baseline-only mode without assertion
        # (when --export-result is used with identical settings)
        if args.export_result and baseline_only_mode and not args.assert_equal:
            export_losses_to_file(baseline_losses, args.export_result)

        # Analysis and reporting (skip in baseline-only mode as there's no test to compare)
        if not baseline_only_mode and test_losses is not None:
            perform_loss_analysis(baseline_losses, test_losses, stats_file)
        print_completion_summary(
            args.output_folder, enable_seed_checkpoint, baseline_only_mode
        )
    finally:
        # Restore original commit if we did checkouts
        if original_commit is not None:
            log_print()
            restore_original_commit(original_commit)


if __name__ == "__main__":
    main()

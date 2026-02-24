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
       --baseline-config='./custom.py' \
       --baseline-options='--parallelism.tensor_parallel_degree=2' \
       --output-folder=my_comparison

3. Compare commits with the same command but skip seed checkpoint for
   faster execution:
   loss_compare.py main my_branch --no-seed-checkpoint

4. Compare the same commit with different training configurations:
   loss_compare.py . . \
       --baseline-options='--parallelism.dp=1' \
       --test-options='--parallelism.dp=2'

5. Compare with different train files:
   loss_compare.py main my_branch \
       --baseline-train-file='torchtitan.train' \
       --test-train-file='torchtitan.custom_train'

6. Assert that losses are equal (for CI testing):
   loss_compare.py main my_branch --assert-equal

7. Run baseline only and compare against imported losses (baseline-only mode):
   loss_compare.py . . --assert-equal --import-result=expected_losses.txt

8. Run baseline only with specific config and compare against imported losses:
   loss_compare.py . . --assert-equal --import-result=expected_losses.txt \
       --baseline-config='./custom.py'

9. Run baseline only and export the losses (no comparison):
   loss_compare.py . . --export-result=baseline_losses.txt

10. Run baseline with specific options and export the losses:
    loss_compare.py . . --baseline-options='--parallelism.dp=2' \
        --export-result=my_config_losses.txt
"""

import argparse
import os
import re
import subprocess
import sys
import unittest
from typing import Any

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

LOG_PREFIX = "[LOSS_COMPARE]"

# Fixed options that are always appended
FIXED_OPTIONS = "--debug.deterministic --debug.seed=42"


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


def get_loss_file_path(scenario: str, output_folder: str) -> str:
    """Get loss file path for a scenario."""
    return f"{output_folder}/{scenario}_losses.txt"


def get_clean_log_path(scenario: str, output_folder: str) -> str:
    """Get cleaned log file path for a scenario."""
    return f"{output_folder}/{scenario}_training_clean.log"


def build_base_command(
    config_file: str, train_file: str, options: str, job_dump_folder: str
) -> str:
    """Build the base command from config file, train file, and options."""
    cmd = f"TRAIN_FILE='{train_file}' CONFIG_FILE='{config_file}' ./run_train.sh"
    cmd += f" --dump_folder={job_dump_folder}"
    if options:
        cmd += f" {options}"
    return cmd


def strip_ansi_codes(input_file: str, output_file: str) -> None:
    """Strip ANSI escape codes from log files."""
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")
    with open(input_file, "r") as f_in:
        with open(output_file, "w") as f_out:
            for line in f_in:
                f_out.write(ansi_escape.sub("", line))


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
    baseline_config: str,
    baseline_train_file: str,
    baseline_options: str,
    test_config: str,
    test_train_file: str,
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
    train_files_differ = baseline_train_file != test_train_file
    options_differ = baseline_options != test_options

    all_identical = not (
        commits_differ or configs_differ or train_files_differ or options_differ
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
            "       Please provide different commits, configs, train files, "
            "or options"
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
    config_file: str,
    train_file: str,
    options: str,
    steps: int,
    enable_seed_checkpoint: bool,
    job_dump_folder: str,
) -> str:
    """Build the final training command with all options."""
    base_cmd = build_base_command(config_file, train_file, options, job_dump_folder)
    cmd = f"{base_cmd} {FIXED_OPTIONS} --training.steps={steps}"
    if enable_seed_checkpoint:
        cmd += (
            " --checkpoint.enable --checkpoint.export_dtype=bfloat16"
            " --checkpoint.load_only"
        )
    return cmd


def print_configuration(
    baseline_commit: str,
    test_commit: str,
    baseline_config: str,
    baseline_train_file: str,
    baseline_options: str,
    test_config: str,
    test_train_file: str,
    test_options: str,
    steps: int,
    enable_seed_checkpoint: bool,
    job_dump_folder: str,
    baseline_only_mode: bool = False,
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
        baseline_config,
        baseline_train_file,
        baseline_options,
        steps,
        enable_seed_checkpoint,
        job_dump_folder,
    )

    log_print("Baseline command:")
    log_print(f"  {baseline_final_cmd}")
    log_print()

    if not baseline_only_mode:
        test_final_cmd = build_training_command(
            test_config,
            test_train_file,
            test_options,
            steps,
            enable_seed_checkpoint,
            job_dump_folder,
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
    config_file: str,
    train_file: str,
    output_folder: str | None,
    job_dump_folder: str,
) -> None:
    """Create seed checkpoint."""
    if enable_seed_checkpoint:
        log_file = get_log_path("seed", output_folder)
        log_print(f"Creating seed checkpoint and logging output to {log_file}")

        # Build seed checkpoint command
        seed_cmd = (
            f"TRAIN_FILE='{train_file}' CONFIG_FILE='{config_file}' "
            f"./run_train.sh --dump_folder={job_dump_folder} "
            f"--checkpoint.create_seed_checkpoint "
            f"--checkpoint.enable {FIXED_OPTIONS}"
        )

        env = os.environ.copy()
        env["NGPU"] = "1"

        run_with_realtime_output(seed_cmd, log_file, env)


def run_training(
    scenario: str,
    config_file: str,
    train_file: str,
    options: str,
    steps: int,
    enable_seed_checkpoint: bool,
    output_folder: str | None,
    job_dump_folder: str,
    ngpus: int,
) -> str:
    """Run training for a specific scenario. Returns the log file path."""
    log_file = get_log_path(scenario, output_folder)
    log_print(
        f"Running training with {scenario} commit and logging output " f"to {log_file}"
    )

    # Build the final command
    full_cmd = build_training_command(
        config_file, train_file, options, steps, enable_seed_checkpoint, job_dump_folder
    )

    env = os.environ.copy()
    env["NGPU"] = str(ngpus)

    run_with_realtime_output(full_cmd, log_file, env)

    return log_file


# =============================================================================
# LOG PROCESSING AND ANALYSIS
# =============================================================================


def extract_losses_from_log(log_file: str) -> dict[int, float]:
    """Extract step and loss pairs from a log file."""
    losses = {}
    step_loss_pattern = re.compile(r"step:\s*(\d+)\s*loss:\s*(\d+\.\d+)")
    ansi_escape = re.compile(r"\x1b\[[0-9;]*m")

    with open(log_file, "r") as f:
        for line in f:
            # Strip ANSI codes before matching
            clean_line = ansi_escape.sub("", line)
            match = step_loss_pattern.search(clean_line)
            if match:
                step, loss = match.groups()
                losses[int(step)] = float(loss)

    return losses


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

    Args:
        losses: Dictionary mapping step numbers to loss values
        export_path: Path to export file
    """
    log_print(f"Exporting losses to {export_path}")

    # Write to file and collect output for stdout
    with open(export_path, "w") as f:
        for step in sorted(losses.keys()):
            loss = losses[step]
            line = f"{step} {loss}"
            f.write(line + "\n")

    log_print(f"Exported {len(losses)} loss values:")
    log_print()

    # Output to stdout in same format
    for step in sorted(losses.keys()):
        loss = losses[step]
        print(f"{step} {loss}")

    log_print()
    log_print(f"Losses saved to: {export_path}")


def extract_loss_data(output_folder: str | None) -> None:
    """Extract loss data from logs."""
    if not output_folder:
        return

    log_print("Cleaning ANSI escape codes from log files...")

    # Strip ANSI escape codes from log files before processing
    scenarios = ["baseline", "test"]
    for scenario in scenarios:
        strip_ansi_codes(
            get_log_path(scenario, output_folder),
            get_clean_log_path(scenario, output_folder),
        )

    # Extract step and loss from cleaned logs
    step_loss_pattern = re.compile(r"step:\s*(\d+)\s*loss:\s*(\d+\.\d+)")

    for scenario in scenarios:
        with open(get_clean_log_path(scenario, output_folder), "r") as f_in:
            with open(get_loss_file_path(scenario, output_folder), "w") as f_out:
                for line in f_in:
                    match = step_loss_pattern.search(line)
                    if match:
                        step, loss = match.groups()
                        f_out.write(f"{step} {loss}\n")


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
    baseline_log: str, test_log: str, stats_file: str | None
) -> None:
    """Perform loss comparison analysis."""
    # Initialize stats file and add header
    log_and_save(f"{LOG_PREFIX} ==========================================", stats_file)
    log_and_save(f"{LOG_PREFIX} LOSS COMPARISON ANALYSIS", stats_file)
    log_and_save(f"{LOG_PREFIX} ==========================================", stats_file)

    # Extract losses directly from log files
    baseline_losses = extract_losses_from_log(baseline_log)
    test_losses = extract_losses_from_log(test_log)

    # Check if losses were extracted successfully
    name_losses = [("baseline", baseline_losses), ("test", test_losses)]
    for name, losses in name_losses:
        if not losses:
            log_and_save(
                f"{LOG_PREFIX} Warning: Could not extract loss data from "
                f"{name} training log.",
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
    baseline_log: str,
    test_log: str | None = None,
    import_result: str | None = None,
) -> None:
    """Assert that losses are equal between baseline and test using unittest.

    Args:
        baseline_log: Path to baseline training log file.
        test_log: Path to test training log file. If None, only compares
            baseline against imported losses (baseline-only mode).
        import_result: Path to imported losses file for comparison.

    In baseline-only mode (test_log is None), import_result must be provided.
    """
    log_print("Asserting losses are equal...")
    log_print(f"Baseline log: {baseline_log}")
    if test_log:
        log_print(f"Test log: {test_log}")
    else:
        log_print("Test log: None (baseline-only mode)")
    if import_result:
        log_print(f"Import file: {import_result}")

    # Validate baseline-only mode has import_result
    if test_log is None and import_result is None:
        log_print("Error: baseline-only mode requires --import-result")
        sys.exit(1)

    # Extract losses from baseline log
    baseline_losses = extract_losses_from_log(baseline_log)
    log_print(f"Extracted {len(baseline_losses)} steps from baseline log")

    if not baseline_losses:
        log_print("Error: No losses found in baseline log")
        sys.exit(1)

    # Extract losses from test log if provided
    test_losses = None
    if test_log:
        test_losses = extract_losses_from_log(test_log)
        log_print(f"Extracted {len(test_losses)} steps from test log")
        if not test_losses:
            log_print("Error: No losses found in test log")
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
                        f"baseline={baseline_loss}, test={test_loss}",
                    )

                # Compare baseline vs imported (if provided)
                if imported_losses:
                    imported_loss = imported_losses[step]
                    self.assertEqual(
                        baseline_loss,
                        imported_loss,
                        f"Loss mismatch at step {step}: "
                        f"baseline={baseline_loss}, imported={imported_loss}",
                    )

    # Run the test
    suite = unittest.TestLoader().loadTestsFromTestCase(LossEqualityTest)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if not result.wasSuccessful():
        log_print("Loss assertion failed!")
        sys.exit(1)
    else:
        if test_log and import_result:
            log_print(
                "All losses are equal (baseline, test, and imported). "
                "Assertion passed!"
            )
        elif test_log:
            log_print("All losses are equal (baseline and test). Assertion passed!")
        else:
            log_print("All losses are equal (baseline and imported). Assertion passed!")


def cleanup_temp_files(output_folder: str | None) -> None:
    """Cleanup temporary files."""
    if not output_folder:
        return

    scenarios = ["baseline", "test"]
    for scenario in scenarios:
        for temp_file in [
            get_loss_file_path(scenario, output_folder),
            get_clean_log_path(scenario, output_folder),
        ]:
            if os.path.exists(temp_file):
                os.remove(temp_file)


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
  %(prog)s abc123 def456 --baseline-config='./custom.py' \\
      --baseline-options='--parallelism.tensor_parallel_degree=2' --steps=50
  %(prog)s abc123 def456 --no-seed-checkpoint
  %(prog)s . . --baseline-options='--parallelism.dp=1' \\
      --test-options='--parallelism.dp=2' --steps=30
        """,
    )

    parser.add_argument("baseline_commit", help="Git commit hash for baseline")
    parser.add_argument("test_commit", help="Git commit hash for test")
    parser.add_argument(
        "--baseline-config",
        default="./torchtitan/models/llama3/config_registry.py",
        help=(
            "Config file for baseline run "
            "(default: ./torchtitan/models/llama3/config_registry.py)"
        ),
    )
    parser.add_argument(
        "--test-config",
        default="",
        help="Config file for test run (default: uses baseline-config)",
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
        "--baseline-train-file",
        default="torchtitan.train",
        help=(
            "Train file (Python module path) for baseline run "
            "(default: torchtitan.train)"
        ),
    )
    parser.add_argument(
        "--test-train-file",
        default="",
        help=(
            "Train file (Python module path) for test run "
            "(default: uses baseline-train-file)"
        ),
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
    if not args.test_config:
        args.test_config = args.baseline_config

    if not args.test_train_file:
        args.test_train_file = args.baseline_train_file

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
    config_file: str,
    train_file: str,
    options: str,
    steps: int,
    enable_seed_checkpoint: bool,
    output_folder: str | None,
    job_dump_folder: str,
    ngpus: int,
) -> str:
    """Run training for a specific scenario (baseline or test).

    Args:
        scenario: Name of the scenario ("baseline" or "test")
        commit: Git commit to checkout
        config_file: Config file path
        train_file: Train file (Python module path)
        options: Additional CLI options
        steps: Number of training steps
        enable_seed_checkpoint: Whether to use seed checkpoint
        output_folder: Output folder for results
        job_dump_folder: Job dump folder path
        ngpus: Number of GPUs to use

    Returns:
        Path to the log file
    """
    checkout_commit(commit, scenario)

    log_file = run_training(
        scenario,
        config_file,
        train_file,
        options,
        steps,
        enable_seed_checkpoint,
        output_folder,
        job_dump_folder,
        ngpus,
    )

    return log_file


def main() -> None:
    """Main function that orchestrates the entire comparison process."""
    # Parse and validate arguments
    args = parse_arguments()
    baseline_only_mode = validate_arguments(
        args.baseline_commit,
        args.test_commit,
        args.baseline_config,
        args.baseline_train_file,
        args.baseline_options,
        args.test_config,
        args.test_train_file,
        args.test_options,
        args.steps,
        args.assert_equal,
        args.export_result,
        args.import_result,
    )

    # Setup environment
    stats_file = setup_output_directory(args.output_folder)
    enable_seed_checkpoint = not args.no_seed_checkpoint
    print_configuration(
        args.baseline_commit,
        args.test_commit,
        args.baseline_config,
        args.baseline_train_file,
        args.baseline_options,
        args.test_config,
        args.test_train_file,
        args.test_options,
        args.steps,
        enable_seed_checkpoint,
        args.job_dump_folder,
        baseline_only_mode,
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
            args.baseline_config,
            args.baseline_train_file,
            args.output_folder,
            args.job_dump_folder,
        )
        # Run baseline training
        baseline_log = run_scenario(
            "baseline",
            args.baseline_commit,
            args.baseline_config,
            args.baseline_train_file,
            args.baseline_options,
            args.steps,
            enable_seed_checkpoint,
            args.output_folder,
            args.job_dump_folder,
            args.baseline_ngpus,
        )

        # Run test training (skip in baseline-only mode)
        test_log = None
        if not baseline_only_mode:
            test_log = run_scenario(
                "test",
                args.test_commit,
                args.test_config,
                args.test_train_file,
                args.test_options,
                args.steps,
                enable_seed_checkpoint,
                args.output_folder,
                args.job_dump_folder,
                args.test_ngpus,
            )
        log_print()

        # Assert losses are equal if requested
        if args.assert_equal:
            # Pass test_log (None in baseline-only mode) and import_result
            assert_losses_equal(baseline_log, test_log, args.import_result)

            # Export losses if requested (only after assertion passes)
            if args.export_result:
                # Extract baseline losses (they equal test losses since assertion passed)
                baseline_losses = extract_losses_from_log(baseline_log)
                export_losses_to_file(baseline_losses, args.export_result)

        # Export losses in baseline-only mode without assertion
        # (when --export-result is used with identical settings)
        if args.export_result and baseline_only_mode and not args.assert_equal:
            baseline_losses = extract_losses_from_log(baseline_log)
            export_losses_to_file(baseline_losses, args.export_result)

        # Analysis and reporting (skip in baseline-only mode as there's no test to compare)
        if not baseline_only_mode and test_log is not None:
            perform_loss_analysis(baseline_log, test_log, stats_file)
            cleanup_temp_files(args.output_folder)
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

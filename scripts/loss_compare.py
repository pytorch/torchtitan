#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This script compares training losses between different git commits and/or different training configurations.
--debug.deterministic is always enabled and seed checkpoint is also enabled by default for reproducible
comparisons. You can disable seed checkpoint with --no-seed-checkpoint if you don't need it to speed up comparisons.
If --output-folder is specified, all outputs are organized in that folder with detailed analysis and statistical summaries.

The --assert-equal flag can be used for CI testing to verify that losses are identical between runs.
If losses differ, the script will exit with a non-zero status code.

Example usages:
1. Compare losses between two different git commits with default command:
   loss_compare.py main my_branch

2. Compare losses between two commits with custom command and save results:
   loss_compare.py main my_branch \
       --baseline-cmd="CONFIG_FILE='./custom.toml' ./run_train.sh --parallelism.tensor_parallel_degree=2" \
       --output-folder=my_comparison

3. Compare commits with the same command but skip seed checkpoint for faster execution:
   loss_compare.py main my_branch --no-seed-checkpoint

4. Compare the same commit with different training configurations:
   loss_compare.py . . \
       --baseline-cmd="CONFIG_FILE='./llama3_8b.toml' ./run_train.sh --parallelism.dp=1" \
       --test-cmd="CONFIG_FILE='./llama3_8b.toml' ./run_train.sh --parallelism.dp=2"

5. Assert that losses are equal (for CI testing):
   loss_compare.py main my_branch --assert-equal
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import unittest
from typing import Any

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================

LOG_PREFIX = "[LOSS_COMPARE]"

# Default configuration values
DEFAULT_RUN_CMD = "CONFIG_FILE='./torchtitan/models/llama3/train_configs/llama3_8b.toml' ./run_train.sh"
DEFAULT_STEPS = 100

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


def extract_config_file(cmd: str) -> str | None:
    """Extract CONFIG_FILE value from command string."""
    # Match CONFIG_FILE=value with optional quotes
    patterns = [
        r"CONFIG_FILE='([^']+)'",  # Single quotes
        r'CONFIG_FILE="([^"]+)"',  # Double quotes
        r"CONFIG_FILE=(\S+)",  # No quotes
    ]

    for pattern in patterns:
        match = re.search(pattern, cmd)
        if match:
            return match.group(1)

    return None


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

        for line in process.stdout:
            print(line, end="")
            log_f.write(line)
            log_f.flush()

        process.wait()

        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)


def log_and_save(message: str, stats_file: str) -> None:
    """Output message to both stdout and stats file."""
    print(message)
    with open(stats_file, "a") as f:
        f.write(message + "\n")


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================


def validate_arguments(
    baseline_commit: str, test_commit: str, baseline_cmd: str, test_cmd: str, steps: int
) -> None:
    """Validate command line arguments."""
    # Validate commit arguments - if one is ".", both must be "."
    if (baseline_commit == "." and test_commit != ".") or (
        baseline_commit != "." and test_commit == "."
    ):
        log_print("Error: If one commit is '.', both commits must be '.'")
        log_print(f"       Got baseline: '{baseline_commit}', test: '{test_commit}'")
        log_print(
            "       Use '.' for both commits to compare different configurations on current working directory"
        )
        sys.exit(1)

    # Validate CONFIG_FILE is specified in both commands
    baseline_config = extract_config_file(baseline_cmd)
    if not baseline_config:
        log_print("Error: CONFIG_FILE not found in baseline command")
        log_print(f"       Baseline command: {baseline_cmd}")
        log_print(
            "       Please specify CONFIG_FILE in the format: CONFIG_FILE='path/to/config.toml' ./run_train.sh"
        )
        sys.exit(1)

    test_config = extract_config_file(test_cmd)
    if not test_config:
        log_print("Error: CONFIG_FILE not found in test command")
        log_print(f"       Test command: {test_cmd}")
        log_print(
            "       Please specify CONFIG_FILE in the format: CONFIG_FILE='path/to/config.toml' ./run_train.sh"
        )
        sys.exit(1)

    # Validate that commits and commands are not both identical
    if baseline_commit == test_commit and baseline_cmd == test_cmd:
        log_print("Error: Both commits and commands are identical")
        log_print("       Cannot compare identical configurations")
        log_print(
            "       Please provide either different commits or different commands"
        )
        sys.exit(1)

    # Validate steps is a positive integer
    if steps <= 0:
        log_print(f"Error: --steps must be a positive integer, got: {steps}")
        sys.exit(1)


# =============================================================================
# SETUP FUNCTIONS
# =============================================================================


def setup_output_directory(output_folder: str | None) -> str | None:
    """Setup output directory and return stats file path. Returns None if no output folder specified."""
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
    base_cmd: str, steps: int, enable_seed_checkpoint: bool
) -> str:
    """Build the final training command with all options."""
    cmd = f"{base_cmd} {FIXED_OPTIONS} --training.steps={steps}"
    if enable_seed_checkpoint:
        cmd += " --checkpoint.enable --checkpoint.export_dtype=bfloat16 --checkpoint.load_only"
    return cmd


def print_configuration(
    baseline_commit: str,
    test_commit: str,
    baseline_cmd: str,
    test_cmd: str,
    steps: int,
    enable_seed_checkpoint: bool,
) -> None:
    """Print configuration summary."""
    log_print(
        f"Starting loss comparison between baseline commit: {baseline_commit} and test commit: {test_commit}"
    )
    log_print(f"Training steps: {steps}")
    log_print(f"Seed checkpoint enabled: {enable_seed_checkpoint}")
    log_print()

    # Build and display final commands
    baseline_final_cmd = build_training_command(
        baseline_cmd, steps, enable_seed_checkpoint
    )
    test_final_cmd = build_training_command(test_cmd, steps, enable_seed_checkpoint)

    log_print("Baseline command:")
    log_print(f"  {baseline_final_cmd}")
    log_print()
    log_print("Test command:")
    log_print(f"  {test_final_cmd}")
    log_print()


# =============================================================================
# GIT OPERATIONS
# =============================================================================


def checkout_commit(commit: str, commit_name: str) -> None:
    """Checkout git commit."""
    if commit != ".":
        log_print(f"Checking out {commit_name} commit: {commit}")
        subprocess.run(["git", "checkout", commit], check=True)
    else:
        log_print(f"Using current working directory for {commit_name} (commit: '.')")


# =============================================================================
# TRAINING OPERATIONS
# =============================================================================


def create_seed_checkpoint(
    enable_seed_checkpoint: bool, baseline_cmd: str, output_folder: str | None
) -> None:
    """Create seed checkpoint."""
    if enable_seed_checkpoint:
        log_file = get_log_path("seed_checkpoint", output_folder)
        log_print(f"Creating seed checkpoint and logging output to {log_file}")

        # Extract CONFIG_FILE from baseline command
        config_file = extract_config_file(baseline_cmd)
        if not config_file:
            log_print("Warning: Could not extract CONFIG_FILE from baseline command")
            log_print(f"         Baseline command: {baseline_cmd}")
            sys.exit(1)

        # Build seed checkpoint command
        seed_cmd = (
            f"CONFIG_FILE={config_file} ./run_train.sh "
            f"--checkpoint.create_seed_checkpoint --checkpoint.enable {FIXED_OPTIONS}"
        )

        env = os.environ.copy()
        env["NGPU"] = "1"

        run_with_realtime_output(seed_cmd, log_file, env)

        # Backup the seed checkpoint
        if output_folder:
            shutil.copytree("outputs", f"{output_folder}/seed_checkpoint_outputs")


def restore_seed_checkpoint(
    enable_seed_checkpoint: bool, output_folder: str | None
) -> None:
    """Restore seed checkpoint."""
    if enable_seed_checkpoint and output_folder:
        if os.path.exists("outputs"):
            shutil.rmtree("outputs")
        shutil.copytree(f"{output_folder}/seed_checkpoint_outputs", "outputs")


def run_training(
    scenario: str,
    cmd: str,
    steps: int,
    enable_seed_checkpoint: bool,
    output_folder: str | None,
) -> str:
    """Run training for a specific scenario. Returns the log file path."""
    log_file = get_log_path(scenario, output_folder)
    log_print(
        f"Running training with {scenario} commit and logging output to {log_file}"
    )

    # Build the final command
    full_cmd = build_training_command(cmd, steps, enable_seed_checkpoint)

    env = os.environ.copy()

    run_with_realtime_output(full_cmd, log_file, env)

    # Backup the outputs
    if output_folder:
        shutil.move("outputs", f"{output_folder}/{scenario}_outputs")
    else:
        # Clean up outputs if not saving
        if os.path.exists("outputs"):
            shutil.rmtree("outputs")

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


def generate_step_comparison(output_folder: str, stats_file: str) -> None:
    """Generate step-by-step comparison."""
    log_and_save("", stats_file)
    log_and_save(f"{LOG_PREFIX} Step-by-step loss comparison:", stats_file)
    log_and_save(
        f"{LOG_PREFIX} Step    Baseline Loss    Test Loss   Difference", stats_file
    )
    log_and_save(
        f"{LOG_PREFIX} ----    -------------    ---------   ----------", stats_file
    )

    # Read baseline and test losses
    baseline_losses = read_losses_from_file(
        get_loss_file_path("baseline", output_folder)
    )
    test_losses = read_losses_from_file(get_loss_file_path("test", output_folder))

    # Generate comparison for common steps
    for step in sorted(set(baseline_losses.keys()) & set(test_losses.keys())):
        baseline_loss = baseline_losses[step]
        test_loss = test_losses[step]
        diff = test_loss - baseline_loss

        formatted_line = f"{LOG_PREFIX} {step:<6}  {baseline_loss:<13}    {test_loss:<14}   {diff:.6f}"
        log_and_save(formatted_line, stats_file)


def generate_summary_statistics(output_folder: str, stats_file: str) -> None:
    """Generate summary statistics."""
    log_and_save(f"{LOG_PREFIX}", stats_file)
    log_and_save(f"{LOG_PREFIX} Summary statistics:", stats_file)

    # Calculate average losses
    def calculate_average(losses: dict[int, float]) -> float | None:
        """Calculate average loss from losses dict."""
        if not losses:
            return None
        return sum(losses.values()) / len(losses)

    baseline_losses = read_losses_from_file(
        get_loss_file_path("baseline", output_folder)
    )
    test_losses = read_losses_from_file(get_loss_file_path("test", output_folder))

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


def perform_loss_analysis(output_folder: str | None, stats_file: str | None) -> None:
    """Perform loss comparison analysis."""
    if not output_folder or not stats_file:
        log_print("Skipping loss analysis (no output folder specified)")
        return

    # Initialize stats file and add header
    log_and_save(f"{LOG_PREFIX} ==========================================", stats_file)
    log_and_save(f"{LOG_PREFIX} LOSS COMPARISON ANALYSIS", stats_file)
    log_and_save(f"{LOG_PREFIX} ==========================================", stats_file)

    # Extract loss data from training logs
    extract_loss_data(output_folder)

    # Check if loss files were created successfully
    scenarios = ["baseline", "test"]
    for scenario in scenarios:
        loss_path = get_loss_file_path(scenario, output_folder)
        if not os.path.exists(loss_path) or os.path.getsize(loss_path) == 0:
            log_and_save(
                f"{LOG_PREFIX} Warning: Could not extract loss data from training logs.",
                stats_file,
            )
            log_and_save(
                f"{LOG_PREFIX} Please check that the training completed successfully.",
                stats_file,
            )
            return

    # Generate comparison outputs
    generate_step_comparison(output_folder, stats_file)
    generate_summary_statistics(output_folder, stats_file)


def assert_losses_equal(baseline_log: str, test_log: str) -> None:
    """Assert that losses are equal between baseline and test using unittest."""
    log_print("Asserting losses are equal...")
    log_print(f"Baseline log: {baseline_log}")
    log_print(f"Test log: {test_log}")

    # Extract losses from both logs
    baseline_losses = extract_losses_from_log(baseline_log)
    test_losses = extract_losses_from_log(test_log)

    log_print(f"Extracted {len(baseline_losses)} steps from baseline log")
    log_print(f"Extracted {len(test_losses)} steps from test log")

    if not baseline_losses:
        log_print("Error: No losses found in baseline log")
        sys.exit(1)

    if not test_losses:
        log_print("Error: No losses found in test log")
        sys.exit(1)

    # Create a test case
    class LossEqualityTest(unittest.TestCase):
        def test_losses_equal(self):
            # Check that both have the same steps
            baseline_steps = set(baseline_losses.keys())
            test_steps = set(test_losses.keys())

            self.assertEqual(
                baseline_steps,
                test_steps,
                f"Steps mismatch: baseline has {len(baseline_steps)} steps, test has {len(test_steps)} steps",
            )

            # Check that losses are equal for each step
            for step in sorted(baseline_steps):
                baseline_loss = baseline_losses[step]
                test_loss = test_losses[step]
                self.assertEqual(
                    baseline_loss,
                    test_loss,
                    f"Loss mismatch at step {step}: baseline={baseline_loss}, test={test_loss}",
                )

    # Run the test
    suite = unittest.TestLoader().loadTestsFromTestCase(LossEqualityTest)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    if not result.wasSuccessful():
        log_print("Loss assertion failed!")
        sys.exit(1)
    else:
        log_print("All losses are equal. Assertion passed!")


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
    output_folder: str | None, enable_seed_checkpoint: bool
) -> None:
    """Print completion summary."""
    log_print()
    if output_folder:
        log_print(f"Loss comparison complete. Results saved in {output_folder}/:")
        log_print("  - baseline_outputs/")
        log_print("  - test_outputs/")
        if enable_seed_checkpoint:
            log_print("  - seed_checkpoint_outputs/")
        log_print()
        log_print(f"Training logs saved in {output_folder}/:")
        if enable_seed_checkpoint:
            log_print("  - seed_checkpoint.log")
        log_print("  - baseline_training.log")
        log_print("  - test_training.log")
        log_print()
        log_print(f"All outputs organized in: {output_folder}/")
    else:
        log_print(
            "Loss comparison complete. No results saved (no output folder specified)."
        )


# =============================================================================
# MAIN EXECUTION
# =============================================================================


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare training losses between different git commits and/or different training configurations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s abc123 def456
  %(prog)s abc123 def456 --steps=200
  %(prog)s abc123 def456 --baseline-cmd="CONFIG_FILE='./custom.toml' ./run_train.sh" --steps=50
  %(prog)s abc123 def456 --no-seed-checkpoint
  %(prog)s . . --baseline-cmd="CONFIG_FILE='./llama3_8b.toml' ./run_train.sh --parallelism.dp=1" \\
      --test-cmd="CONFIG_FILE='./llama3_8b.toml' ./run_train.sh --parallelism.dp=2" --steps=30
        """,
    )

    parser.add_argument("baseline_commit", help="Git commit hash for baseline")
    parser.add_argument("test_commit", help="Git commit hash for test")
    parser.add_argument(
        "--baseline-cmd",
        default="",
        help=f"Full command for baseline run (default: {DEFAULT_RUN_CMD})",
    )
    parser.add_argument(
        "--test-cmd",
        default="",
        help="Full command for test run (default: uses baseline-cmd)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Number of training steps (default: {DEFAULT_STEPS})",
    )
    parser.add_argument(
        "--no-seed-checkpoint",
        action="store_true",
        help="Disable seed checkpoint creation and checkpoint functionality",
    )
    parser.add_argument(
        "--output-folder",
        default="",
        help="Output folder for results (optional, if not specified, results will not be saved)",
    )
    parser.add_argument(
        "--assert-equal",
        action="store_true",
        help="Assert that all losses are equal (for CI testing). Script exits with error if losses differ.",
    )

    args = parser.parse_args()

    # Set default commands if not provided
    if not args.baseline_cmd:
        args.baseline_cmd = DEFAULT_RUN_CMD

    if not args.test_cmd:
        args.test_cmd = args.baseline_cmd
        log_print("Note: Using baseline command for both baseline and test runs")

    # Convert empty output_folder to None
    if not args.output_folder:
        args.output_folder = None

    return args


def run_scenario(
    scenario: str,
    commit: str,
    cmd: str,
    steps: int,
    enable_seed_checkpoint: bool,
    output_folder: str | None,
    is_baseline: bool = False,
) -> str:
    """Run training for a specific scenario (baseline or test).

    Args:
        scenario: Name of the scenario ("baseline" or "test")
        commit: Git commit to checkout
        cmd: Command to run
        steps: Number of training steps
        enable_seed_checkpoint: Whether to use seed checkpoint
        output_folder: Output folder for results
        is_baseline: Whether this is the baseline run (handles seed checkpoint creation)

    Returns:
        Path to the log file
    """
    checkout_commit(commit, scenario)

    if is_baseline:
        create_seed_checkpoint(enable_seed_checkpoint, cmd, output_folder)

    log_file = run_training(scenario, cmd, steps, enable_seed_checkpoint, output_folder)

    if is_baseline:
        restore_seed_checkpoint(enable_seed_checkpoint, output_folder)

    return log_file


def main() -> None:
    """Main function that orchestrates the entire comparison process."""
    # Parse and validate arguments
    args = parse_arguments()
    validate_arguments(
        args.baseline_commit,
        args.test_commit,
        args.baseline_cmd,
        args.test_cmd,
        args.steps,
    )

    # Setup environment
    stats_file = setup_output_directory(args.output_folder)
    enable_seed_checkpoint = not args.no_seed_checkpoint
    print_configuration(
        args.baseline_commit,
        args.test_commit,
        args.baseline_cmd,
        args.test_cmd,
        args.steps,
        enable_seed_checkpoint,
    )

    # Run baseline and test training
    baseline_log = run_scenario(
        "baseline",
        args.baseline_commit,
        args.baseline_cmd,
        args.steps,
        enable_seed_checkpoint,
        args.output_folder,
        is_baseline=True,
    )

    test_log = run_scenario(
        "test",
        args.test_commit,
        args.test_cmd,
        args.steps,
        enable_seed_checkpoint,
        args.output_folder,
        is_baseline=False,
    )
    log_print()

    # Assert losses are equal if requested
    if args.assert_equal:
        assert_losses_equal(baseline_log, test_log)

    # Analysis and reporting
    perform_loss_analysis(args.output_folder, stats_file)
    cleanup_temp_files(args.output_folder)
    print_completion_summary(args.output_folder, enable_seed_checkpoint)


if __name__ == "__main__":
    main()

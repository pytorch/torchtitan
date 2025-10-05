#!/bin/bash

# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# This script compares training losses between different git commits and/or different training configurations.
# --training.deterministic is always enabled and seed checkpoint is also enabled by default for reproducible
# comparisons. You can disable seed checkpoint with --no-seed-checkpoint if you don't need it to speed up comparisons.
# All outputs are organized in the loss_compare/ folder with detailed analysis and statistical summaries.
#
# Example usages:
# 1. Compare losses between two different git commits with the same training options:
#    loss_compare.sh main my_branch --baseline-train-options="--parallelism.tensor_parallel_degree=2"
#
# 2. Compare commits with the same options but skip seed checkpoint for faster execution:
#    loss_compare.sh main my_branch --baseline-train-options="--parallelism.tensor_parallel_degree=2" --no-seed-checkpoint
#
# 3. Compare the same commit with different training options:
#    loss_compare.sh . . --baseline-train-options="--parallelism.tensor_parallel_degree=1" --benchmark-train-options="--parallelism.tensor_parallel_degree=2"

# Exit immediately if any command exits with a non-zero status
set -e
set -u
set -o pipefail

#=============================================================================
# GLOBAL CONFIGURATION
#=============================================================================

# Define log prefix for easy identification
LOG_PREFIX="[LOSS_COMPARE]"

# Default configuration values
CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml"
STEPS=100
BASELINE_TRAIN_OPTIONS=""
BENCHMARK_TRAIN_OPTIONS=""
ENABLE_SEED_CHECKPOINT=true
OUTPUT_FOLDER="loss_compare"

# Command templates
FIXED_OPTIONS="--training.deterministic --training.seed=42"
SEED_CHECKPOINT_CMD="./run_train.sh --checkpoint.create_seed_checkpoint --checkpoint.enable ${FIXED_OPTIONS}"
TRAIN_CMD_WITH_CHECKPOINT="./run_train.sh --checkpoint.enable --checkpoint.export_dtype=bfloat16 ${FIXED_OPTIONS}"
TRAIN_CMD_WITHOUT_CHECKPOINT="./run_train.sh ${FIXED_OPTIONS}"

# Global variables set by argument parsing
BASELINE_COMMIT=""
BENCHMARK_COMMIT=""
STATS_FILE=""

#=============================================================================
# UTILITY FUNCTIONS
#=============================================================================

# Function to strip ANSI escape codes from log files
strip_ansi_codes() {
    local input_file="$1"
    local output_file="$2"

    # Remove ANSI escape sequences using sed
    sed 's/\x1b\[[0-9;]*m//g' "$input_file" > "$output_file"
}

# Function to run command with real-time output (fallback options)
run_with_realtime_output() {
    local cmd="$1"
    local logfile="$2"

    # Method 1: Try script command (creates pseudo-TTY)
    if command -v script >/dev/null 2>&1; then
        PYTHONUNBUFFERED=1 script -q -c "$cmd" /dev/null 2>&1 | tee "$logfile"
    # Method 2: Try unbuffer (from expect package)
    elif command -v unbuffer >/dev/null 2>&1; then
        PYTHONUNBUFFERED=1 unbuffer $cmd 2>&1 | tee "$logfile"
    # Method 3: Fallback to stdbuf
    elif command -v stdbuf >/dev/null 2>&1; then
        PYTHONUNBUFFERED=1 stdbuf -oL -eL $cmd 2>&1 | tee "$logfile"
    # Method 4: Basic approach (may not be real-time)
    else
        echo "$LOG_PREFIX Warning: No real-time output tools available, output may be delayed"
        PYTHONUNBUFFERED=1 $cmd 2>&1 | tee "$logfile"
    fi
}

# Function to output message to both stdout and stats file
log_and_save() {
    local message="$1"
    echo "$message"
    echo "$message" >> "$STATS_FILE"
}

#=============================================================================
# HELP AND VALIDATION FUNCTIONS
#=============================================================================

# Function to show usage
show_usage() {
    echo "$LOG_PREFIX Usage: $0 <baseline_commit> <benchmark_commit> [OPTIONS]"
    echo "$LOG_PREFIX"
    echo "$LOG_PREFIX Required arguments:"
    echo "$LOG_PREFIX   baseline_commit              Git commit hash for baseline"
    echo "$LOG_PREFIX   benchmark_commit             Git commit hash for benchmark"
    echo "$LOG_PREFIX"
    echo "$LOG_PREFIX Optional arguments:"
    echo "$LOG_PREFIX   --config=PATH                Path to config file (default: ./torchtitan/models/llama3/train_configs/llama3_8b.toml)"
    echo "$LOG_PREFIX   --steps=N                    Number of training steps (default: 100)"
    echo "$LOG_PREFIX   --baseline-train-options=\"\"   Additional training options for baseline run"
    echo "$LOG_PREFIX   --benchmark-train-options=\"\"  Additional training options for benchmark run"
    echo "$LOG_PREFIX   --no-seed-checkpoint         Disable seed checkpoint creation and checkpoint functionality"
    echo "$LOG_PREFIX"
    echo "$LOG_PREFIX Note: If only --baseline-train-options is provided, both runs will use the same options"
    echo "$LOG_PREFIX       Seed checkpoint is enabled by default for reproducible comparisons"
    echo "$LOG_PREFIX"
    echo "$LOG_PREFIX Examples:"
    echo "$LOG_PREFIX   $0 abc123 def456"
    echo "$LOG_PREFIX   $0 abc123 def456 --steps=200"
    echo "$LOG_PREFIX   $0 abc123 def456 --config=./custom.toml --steps=50"
    echo "$LOG_PREFIX   $0 abc123 def456 --no-seed-checkpoint"
    echo "$LOG_PREFIX   $0 . . --baseline-train-options=\"--parallelism.data_parallel_replicate_degree=1\" --benchmark-train-options=\"--parallelism.data_parallel_replicate_degree=2\" --steps=30"
    echo "$LOG_PREFIX"
}

# Function to validate arguments
validate_arguments() {
    # Check minimum argument count
    if [ $# -lt 2 ]; then
        show_usage
        exit 1
    fi

    # Validate commit arguments - if one is ".", both must be "."
    if [[ "$BASELINE_COMMIT" == "." && "$BENCHMARK_COMMIT" != "." ]] || [[ "$BASELINE_COMMIT" != "." && "$BENCHMARK_COMMIT" == "." ]]; then
        echo "$LOG_PREFIX Error: If one commit is '.', both commits must be '.'"
        echo "$LOG_PREFIX       Got baseline: '$BASELINE_COMMIT', benchmark: '$BENCHMARK_COMMIT'"
        echo "$LOG_PREFIX       Use '.' for both commits to compare different configurations on current working directory"
        exit 1
    fi

    # Validate steps is a number
    if ! [[ "$STEPS" =~ ^[0-9]+$ ]]; then
        echo "$LOG_PREFIX Error: --steps must be a positive integer, got: $STEPS"
        exit 1
    fi
}

#=============================================================================
# ARGUMENT PARSING
#=============================================================================

# Function to parse command line arguments
parse_arguments() {
    # Check if at least two arguments are provided
    if [ $# -lt 2 ]; then
        show_usage
        exit 1
    fi

    # Parse required arguments
    BASELINE_COMMIT=$1
    BENCHMARK_COMMIT=$2
    shift 2

    # Parse optional named arguments
    while [ $# -gt 0 ]; do
        case $1 in
            --config=*)
                CONFIG_FILE="${1#*=}"
                ;;
            --steps=*)
                STEPS="${1#*=}"
                ;;
            --baseline-train-options=*)
                BASELINE_TRAIN_OPTIONS="${1#*=}"
                ;;
            --benchmark-train-options=*)
                BENCHMARK_TRAIN_OPTIONS="${1#*=}"
                ;;
            --no-seed-checkpoint)
                ENABLE_SEED_CHECKPOINT=false
                ;;
            --help|-h)
                show_usage
                exit 0
                ;;
            *)
                echo "$LOG_PREFIX Error: Unknown argument '$1'"
                show_usage
                exit 1
                ;;
        esac
        shift
    done

    # If only baseline train options are provided, use them for both
    if [ -n "$BASELINE_TRAIN_OPTIONS" ] && [ -z "$BENCHMARK_TRAIN_OPTIONS" ]; then
        BENCHMARK_TRAIN_OPTIONS="$BASELINE_TRAIN_OPTIONS"
        echo "$LOG_PREFIX Note: Using baseline train options for both baseline and benchmark runs"
    fi
}

#=============================================================================
# SETUP FUNCTIONS
#=============================================================================

# Function to setup output directory
setup_output_directory() {
    # Check if output folder already exists
    if [ -d "$OUTPUT_FOLDER" ]; then
        echo "$LOG_PREFIX Error: Output folder '$OUTPUT_FOLDER' already exists"
        echo "$LOG_PREFIX Please delete it first: rm -rf $OUTPUT_FOLDER"
        exit 1
    fi

    # Create the output folder
    echo "$LOG_PREFIX Creating output folder: $OUTPUT_FOLDER"
    mkdir -p "$OUTPUT_FOLDER"

    # Set statistics file path
    STATS_FILE="$OUTPUT_FOLDER/comparison_statistics.txt"
}

# Function to print configuration summary
print_configuration() {
    echo "$LOG_PREFIX Starting loss comparison between baseline commit: $BASELINE_COMMIT and benchmark commit: $BENCHMARK_COMMIT"
    echo "$LOG_PREFIX Config file: $CONFIG_FILE"
    echo "$LOG_PREFIX Training steps: $STEPS"
    echo "$LOG_PREFIX Seed checkpoint enabled: $ENABLE_SEED_CHECKPOINT"
    echo "$LOG_PREFIX Baseline train options: $BASELINE_TRAIN_OPTIONS"
    echo "$LOG_PREFIX Benchmark train options: $BENCHMARK_TRAIN_OPTIONS"
    echo "$LOG_PREFIX "
}

#=============================================================================
# GIT OPERATIONS
#=============================================================================

# Function to checkout git commit
checkout_commit() {
    local commit="$1"
    local commit_name="$2"

    if [[ "$commit" != "." ]]; then
        echo "$LOG_PREFIX Checking out $commit_name commit: $commit"
        git checkout $commit
    else
        echo "$LOG_PREFIX Using current working directory for $commit_name (commit: '.')"
    fi
}

#=============================================================================
# TRAINING OPERATIONS
#=============================================================================

# Function to create seed checkpoint
create_seed_checkpoint() {
    if [ "$ENABLE_SEED_CHECKPOINT" = true ]; then
        echo "$LOG_PREFIX Creating seed checkpoint and logging output to $OUTPUT_FOLDER/seed_checkpoint.log"
        NGPU=1 CONFIG_FILE="$CONFIG_FILE" run_with_realtime_output "$SEED_CHECKPOINT_CMD" "$OUTPUT_FOLDER/seed_checkpoint.log"

        # Backup the seed checkpoint
        cp -r outputs "$OUTPUT_FOLDER/seed_checkpoint_outputs"
    fi
}

# Function to get appropriate training command
get_training_command() {
    if [ "$ENABLE_SEED_CHECKPOINT" = true ]; then
        echo "$TRAIN_CMD_WITH_CHECKPOINT"
    else
        echo "$TRAIN_CMD_WITHOUT_CHECKPOINT"
    fi
}

# Function to restore seed checkpoint
restore_seed_checkpoint() {
    if [ "$ENABLE_SEED_CHECKPOINT" = true ]; then
        cp -r "$OUTPUT_FOLDER/seed_checkpoint_outputs" outputs
    fi
}

# Function to run training for a specific scenario
run_training() {
    local scenario="$1"
    local train_options="$2"
    local train_cmd=$(get_training_command)

    echo "$LOG_PREFIX Running training with $scenario commit and logging output to $OUTPUT_FOLDER/${scenario}_training.log"
    CONFIG_FILE="$CONFIG_FILE" run_with_realtime_output "$train_cmd --training.steps=$STEPS $train_options" "$OUTPUT_FOLDER/${scenario}_training.log"

    # Backup the outputs
    mv outputs "$OUTPUT_FOLDER/${scenario}_outputs"
}

#=============================================================================
# LOG PROCESSING AND ANALYSIS
#=============================================================================

# Function to extract loss data from logs
extract_loss_data() {
    echo "$LOG_PREFIX Cleaning ANSI escape codes from log files..."

    # Strip ANSI escape codes from log files before processing
    strip_ansi_codes "$OUTPUT_FOLDER/baseline_training.log" "$OUTPUT_FOLDER/baseline_training_clean.log"
    strip_ansi_codes "$OUTPUT_FOLDER/benchmark_training.log" "$OUTPUT_FOLDER/benchmark_training_clean.log"

    # Extract step and loss from cleaned logs
    grep "step:" "$OUTPUT_FOLDER/baseline_training_clean.log" | sed -E 's/.*step:[[:space:]]*([0-9]+)[[:space:]]*loss:[[:space:]]*([0-9]+\.[0-9]+).*/\1 \2/' > "$OUTPUT_FOLDER/baseline_losses.txt"
    grep "step:" "$OUTPUT_FOLDER/benchmark_training_clean.log" | sed -E 's/.*step:[[:space:]]*([0-9]+)[[:space:]]*loss:[[:space:]]*([0-9]+\.[0-9]+).*/\1 \2/' > "$OUTPUT_FOLDER/benchmark_losses.txt"
}

# Function to generate step-by-step comparison
generate_step_comparison() {
    log_and_save ""
    log_and_save "$LOG_PREFIX Step-by-step loss comparison:"
    log_and_save "$LOG_PREFIX Step    Baseline Loss    Benchmark Loss   Difference"
    log_and_save "$LOG_PREFIX ----    -------------    --------------   ----------"

    # Join the two files and calculate differences
    join "$OUTPUT_FOLDER/baseline_losses.txt" "$OUTPUT_FOLDER/benchmark_losses.txt" | while read step baseline_loss benchmark_loss; do
        # Calculate difference using awk for floating point arithmetic
        diff=$(awk "BEGIN {printf \"%.6f\", $benchmark_loss - $baseline_loss}")
        local formatted_line=$(printf "$LOG_PREFIX %-6s  %-13s    %-14s   %s" "$step" "$baseline_loss" "$benchmark_loss" "$diff")
        log_and_save "$formatted_line"
    done
}

# Function to generate summary statistics
generate_summary_statistics() {
    log_and_save "$LOG_PREFIX"
    log_and_save "$LOG_PREFIX Summary statistics:"

    # Calculate average losses
    local baseline_avg=$(awk '{sum+=$2; count++} END {if(count>0) print sum/count; else print "N/A"}' "$OUTPUT_FOLDER/baseline_losses.txt")
    local benchmark_avg=$(awk '{sum+=$2; count++} END {if(count>0) print sum/count; else print "N/A"}' "$OUTPUT_FOLDER/benchmark_losses.txt")

    log_and_save "$LOG_PREFIX Average baseline loss:  $baseline_avg"
    log_and_save "$LOG_PREFIX Average benchmark loss: $benchmark_avg"

    # Calculate overall difference if both averages are available
    if [[ "$baseline_avg" != "N/A" && "$benchmark_avg" != "N/A" ]]; then
        local avg_diff=$(awk "BEGIN {printf \"%.6f\", $benchmark_avg - $baseline_avg}")
        log_and_save "$LOG_PREFIX Average difference:     $avg_diff"

        # Determine performance comparison
        if (( $(awk "BEGIN {print ($avg_diff < 0)}") )); then
            log_and_save "$LOG_PREFIX Benchmark performs BETTER (lower loss) than baseline"
        elif (( $(awk "BEGIN {print ($avg_diff > 0)}") )); then
            log_and_save "$LOG_PREFIX Baseline performs BETTER (lower loss) than benchmark"
        else
            log_and_save "$LOG_PREFIX Performance is EQUIVALENT between baseline and benchmark"
        fi
    fi
}

# Function to perform loss comparison analysis
perform_loss_analysis() {
    # Initialize stats file and add header
    log_and_save "$LOG_PREFIX =========================================="
    log_and_save "$LOG_PREFIX LOSS COMPARISON ANALYSIS"
    log_and_save "$LOG_PREFIX =========================================="

    # Extract loss data from training logs
    extract_loss_data

    # Check if loss files were created successfully
    if [ ! -s "$OUTPUT_FOLDER/baseline_losses.txt" ] || [ ! -s "$OUTPUT_FOLDER/benchmark_losses.txt" ]; then
        log_and_save "$LOG_PREFIX Warning: Could not extract loss data from training logs."
        log_and_save "$LOG_PREFIX Please check that the training completed successfully."
        return
    fi

    # Generate comparison outputs
    generate_step_comparison
    generate_summary_statistics
}

# Function to cleanup temporary files
cleanup_temp_files() {
    rm -f "$OUTPUT_FOLDER/baseline_losses.txt" "$OUTPUT_FOLDER/benchmark_losses.txt"
    rm -f "$OUTPUT_FOLDER/baseline_training_clean.log" "$OUTPUT_FOLDER/benchmark_training_clean.log"
}

#=============================================================================
# OUTPUT FUNCTIONS
#=============================================================================

# Function to print completion summary
print_completion_summary() {
    echo "$LOG_PREFIX"
    echo "$LOG_PREFIX Loss comparison complete. Results saved in $OUTPUT_FOLDER/:"
    echo "$LOG_PREFIX   - baseline_outputs/"
    echo "$LOG_PREFIX   - benchmark_outputs/"
    if [ "$ENABLE_SEED_CHECKPOINT" = true ]; then
        echo "$LOG_PREFIX   - seed_checkpoint_outputs/"
    fi
    echo "$LOG_PREFIX"
    echo "$LOG_PREFIX Training logs saved in $OUTPUT_FOLDER/:"
    if [ "$ENABLE_SEED_CHECKPOINT" = true ]; then
        echo "$LOG_PREFIX   - seed_checkpoint.log"
    fi
    echo "$LOG_PREFIX   - baseline_training.log"
    echo "$LOG_PREFIX   - benchmark_training.log"
    echo "$LOG_PREFIX"
    echo "$LOG_PREFIX All outputs organized in: $OUTPUT_FOLDER/"
}

#=============================================================================
# MAIN EXECUTION
#=============================================================================

# Main function that orchestrates the entire comparison process
main() {
    # Parse and validate arguments
    parse_arguments "$@"
    validate_arguments "$@"

    # Setup environment
    setup_output_directory
    print_configuration

    # Baseline training
    checkout_commit "$BASELINE_COMMIT" "baseline"
    create_seed_checkpoint
    run_training "baseline" "$BASELINE_TRAIN_OPTIONS"
    restore_seed_checkpoint

    # Benchmark training
    checkout_commit "$BENCHMARK_COMMIT" "benchmark"
    run_training "benchmark" "$BENCHMARK_TRAIN_OPTIONS"
    echo "$LOG_PREFIX "

    # Analysis and reporting
    perform_loss_analysis
    cleanup_temp_files
    print_completion_summary
}

# Run the main function with all arguments
main "$@"

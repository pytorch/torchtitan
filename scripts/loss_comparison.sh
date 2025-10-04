#!/bin/bash

# Exit immediately if any command exits with a non-zero status
set -e

# Exit on undefined variables
set -u

# Make pipe failures detectable
set -o pipefail

# Optional: Print commands as they are executed (uncomment for debugging)
# set -x

# Define log prefix for easy identification
LOG_PREFIX="[LOSS_COMPARE]"

# Default values
STEP_COUNT=100
CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml"

# Command configurations for better readability
SEED_CHECKPOINT_CMD="./run_train.sh --checkpoint.create_seed_checkpoint --checkpoint.enable "
TRAIN_CMD="./run_train.sh --checkpoint.enable "
TRAIN_OPTIONS="--training.seq_len=4096 "

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

# Function to show usage
show_usage() {
    echo "$LOG_PREFIX Usage: $0 <baseline_commit> <benchmark_commit> [OPTIONS]"
    echo "$LOG_PREFIX"
    echo "$LOG_PREFIX Required arguments:"
    echo "$LOG_PREFIX   baseline_commit     Git commit hash for baseline"
    echo "$LOG_PREFIX   benchmark_commit    Git commit hash for benchmark"
    echo "$LOG_PREFIX"
    echo "$LOG_PREFIX Optional arguments:"
    echo "$LOG_PREFIX   --steps=N           Number of training steps (default: 100)"
    echo "$LOG_PREFIX   --config=PATH       Path to config file (default: ./torchtitan/models/llama3/train_configs/llama3_8b.toml)"
    echo "$LOG_PREFIX"
    echo "$LOG_PREFIX Examples:"
    echo "$LOG_PREFIX   $0 abc123 def456"
    echo "$LOG_PREFIX   $0 abc123 def456 --steps=200"
    echo "$LOG_PREFIX   $0 abc123 def456 --config=./custom.toml"
    echo "$LOG_PREFIX   $0 abc123 def456 --steps=50 --config=./custom.toml"
    echo "$LOG_PREFIX   $0 abc123 def456 --config=./custom.toml --steps=200"
}

# Check if at least two arguments are provided
if [ $# -lt 2 ]; then
    show_usage
    exit 1
fi

# Parse required arguments
BASELINE_COMMIT=$1
BENCHMARK_COMMIT=$2

# Parse optional named arguments
shift 2  # Remove the first two arguments
while [ $# -gt 0 ]; do
    case $1 in
        --steps=*)
            STEP_COUNT="${1#*=}"
            ;;
        --config=*)
            CONFIG_FILE="${1#*=}"
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

# Validate step count is a number
if ! [[ "$STEP_COUNT" =~ ^[0-9]+$ ]]; then
    echo "$LOG_PREFIX Error: --steps must be a positive integer, got: $STEP_COUNT"
    exit 1
fi

echo "$LOG_PREFIX Starting loss comparison between baseline commit: $BASELINE_COMMIT and benchmark commit: $BENCHMARK_COMMIT"
echo "$LOG_PREFIX Training steps: $STEP_COUNT"
echo "$LOG_PREFIX Config file: $CONFIG_FILE"
echo "$LOG_PREFIX "

# Create a seed checkpoint
echo "$LOG_PREFIX Creating seed checkpoint and logging output to seed_checkpoint.log"
NGPU=1 CONFIG_FILE="$CONFIG_FILE" run_with_realtime_output "$SEED_CHECKPOINT_CMD" "seed_checkpoint.log"

# Backup the seed checkpoint
cp -r outputs seed_checkpoint_outputs

# Checkout the baseline git commit
echo "$LOG_PREFIX Checking out baseline commit: $BASELINE_COMMIT"
git checkout $BASELINE_COMMIT

# Run the training with the baseline git commit
echo "$LOG_PREFIX Running training with baseline commit and logging output to baseline_training.log"
CONFIG_FILE="$CONFIG_FILE" run_with_realtime_output "$TRAIN_CMD --training.steps=$STEP_COUNT" "baseline_training.log"

# Backup the baseline outputs
mv outputs baseline_outputs

# Copy seed checkpoint back
cp -r seed_checkpoint_outputs outputs

# Checkout the benchmark git commit
echo "$LOG_PREFIX Checking out benchmark commit: $BENCHMARK_COMMIT"
git checkout $BENCHMARK_COMMIT

# Run the training with the benchmark git commit
echo "$LOG_PREFIX Running training with benchmark commit and logging output to benchmark_training.log"
CONFIG_FILE="$CONFIG_FILE" run_with_realtime_output "$TRAIN_CMD --training.steps=$STEP_COUNT" "benchmark_training.log"
echo "$LOG_PREFIX "

# Backup the benchmark outputs
mv outputs benchmark_outputs

echo "$LOG_PREFIX Loss comparison complete. Results saved in:"
echo "$LOG_PREFIX   - baseline_outputs/"
echo "$LOG_PREFIX   - benchmark_outputs/"
echo "$LOG_PREFIX   - seed_checkpoint_outputs/"
echo "$LOG_PREFIX"
echo "$LOG_PREFIX Training logs saved in:"
echo "$LOG_PREFIX   - seed_checkpoint.log"
echo "$LOG_PREFIX   - baseline_training.log"
echo "$LOG_PREFIX   - benchmark_training.log"
echo "$LOG_PREFIX"

# Extract and compare loss values
echo "$LOG_PREFIX =========================================="
echo "$LOG_PREFIX LOSS COMPARISON ANALYSIS"
echo "$LOG_PREFIX =========================================="

# Extract step and loss from baseline log
grep "step:" baseline_training.log | sed -E 's/.*step:[[:space:]]*([0-9]+)[[:space:]]*loss:[[:space:]]*([0-9]+\.[0-9]+).*/\1 \2/' > baseline_losses.txt

# Extract step and loss from benchmark log
grep "step:" benchmark_training.log | sed -E 's/.*step:[[:space:]]*([0-9]+)[[:space:]]*loss:[[:space:]]*([0-9]+\.[0-9]+).*/\1 \2/' > benchmark_losses.txt

# Check if loss files were created successfully
if [ ! -s baseline_losses.txt ] || [ ! -s benchmark_losses.txt ]; then
    echo "$LOG_PREFIX Warning: Could not extract loss data from training logs."
    echo "$LOG_PREFIX Please check that the training completed successfully."
else
    # Create comparison output
    echo "$LOG_PREFIX Step-by-step loss comparison:"
    echo "$LOG_PREFIX Step    Baseline Loss    Benchmark Loss   Difference"
    echo "$LOG_PREFIX ----    -------------    --------------   ----------"

    # Join the two files and calculate differences
    join baseline_losses.txt benchmark_losses.txt | while read step baseline_loss benchmark_loss; do
        # Calculate difference using awk for floating point arithmetic
        diff=$(awk "BEGIN {printf \"%.6f\", $benchmark_loss - $baseline_loss}")
        printf "%-6s  %-13s    %-14s   %s\n" "$step" "$baseline_loss" "$benchmark_loss" "$diff"
    done

    echo "$LOG_PREFIX"
    echo "$LOG_PREFIX Summary statistics:"

    # Calculate average losses
    baseline_avg=$(awk '{sum+=$2; count++} END {if(count>0) print sum/count; else print "N/A"}' baseline_losses.txt)
    benchmark_avg=$(awk '{sum+=$2; count++} END {if(count>0) print sum/count; else print "N/A"}' benchmark_losses.txt)

    echo "$LOG_PREFIX Average baseline loss:  $baseline_avg"
    echo "$LOG_PREFIX Average benchmark loss: $benchmark_avg"

    # Calculate overall difference if both averages are available
    if [[ "$baseline_avg" != "N/A" && "$benchmark_avg" != "N/A" ]]; then
        avg_diff=$(awk "BEGIN {printf \"%.6f\", $benchmark_avg - $baseline_avg}")
        echo "$LOG_PREFIX Average difference:     $avg_diff"

        if (( $(awk "BEGIN {print ($avg_diff < 0)}") )); then
            echo "$LOG_PREFIX Benchmark performs BETTER (lower loss) than baseline"
        elif (( $(awk "BEGIN {print ($avg_diff > 0)}") )); then
            echo "$LOG_PREFIX Baseline performs BETTER (lower loss) than benchmark"
        else
            echo "$LOG_PREFIX Performance is EQUIVALENT between baseline and benchmark"
        fi
    fi
fi

echo "$LOG_PREFIX"
echo "$LOG_PREFIX Additional comparison commands:"
echo "$LOG_PREFIX   diff baseline_training.log benchmark_training.log"
echo "$LOG_PREFIX   grep -E 'loss|step' baseline_training.log benchmark_training.log"

# Clean up temporary files
rm -f baseline_losses.txt benchmark_losses.txt

#!/usr/bin/env python
# Copyright (c) Nous Research.
# All rights reserved.

"""
Standalone script for running lm-evaluation-harness on torchtitan checkpoints.

This script provides a command-line interface for running evaluations with
full reproducibility through seed control and configuration logging.

Usage:
    python scripts/eval/run_lm_eval.py \
        --checkpoint /path/to/checkpoint \
        --model_name qwen3 \
        --model_flavor 10B-A1B \
        --tasks hellaswag,arc_easy,arc_challenge,winogrande \
        --output_dir ./eval_results \
        --seed 42

For full reproducibility, all configuration is saved alongside results.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path


def setup_paths(torchtitan_path: str | None, lm_eval_path: str | None) -> None:
    """Setup Python paths for imports."""
    if torchtitan_path and torchtitan_path not in sys.path:
        sys.path.insert(0, torchtitan_path)

    if lm_eval_path and lm_eval_path not in sys.path:
        sys.path.insert(0, lm_eval_path)

    # Auto-detect torchtitan path if not provided
    if not torchtitan_path:
        possible_paths = [
            Path(__file__).parent.parent.parent,  # scripts/eval/run_lm_eval.py
            Path("/home/phuc/workspace/moe/online_evals/torchtitan"),
        ]
        for path in possible_paths:
            if (path / "torchtitan").exists():
                sys.path.insert(0, str(path))
                break


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run lm-evaluation-harness on torchtitan checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python run_lm_eval.py --checkpoint /path/to/checkpoint --model_name qwen3 --model_flavor 10B-A1B

    # With custom tasks and output
    python run_lm_eval.py \\
        --checkpoint /path/to/checkpoint \\
        --model_name qwen3 \\
        --model_flavor 10B-A1B \\
        --tasks hellaswag,arc_easy,arc_challenge,winogrande \\
        --output_dir ./my_eval_results \\
        --seed 42

    # Quick test with limited samples
    python run_lm_eval.py \\
        --checkpoint /path/to/checkpoint \\
        --model_name llama3 \\
        --model_flavor 8B \\
        --tasks hellaswag \\
        --limit 100
        """,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint directory (HF safetensors format)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["llama3", "qwen3"],
        help="Model architecture name",
    )
    parser.add_argument(
        "--model_flavor",
        type=str,
        required=True,
        help="Model flavor/size (e.g., 8B, 70B, 10B-A1B)",
    )

    # Optional arguments
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Path to tokenizer (defaults to checkpoint path)",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="hellaswag,arc_easy,arc_challenge,winogrande",
        help="Comma-separated list of evaluation tasks",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples per task (None = full evaluation)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./eval_results",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--log_samples",
        action="store_true",
        default=True,
        help="Log individual sample predictions",
    )
    parser.add_argument(
        "--no_log_samples",
        action="store_false",
        dest="log_samples",
        help="Don't log individual sample predictions",
    )

    # Seed arguments for reproducibility
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for all RNGs",
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=None,
        help="Override Python random seed",
    )
    parser.add_argument(
        "--numpy_seed",
        type=int,
        default=None,
        help="Override NumPy random seed",
    )
    parser.add_argument(
        "--torch_seed",
        type=int,
        default=None,
        help="Override PyTorch random seed",
    )
    parser.add_argument(
        "--fewshot_seed",
        type=int,
        default=None,
        help="Override fewshot sampler seed",
    )

    # Path configuration
    parser.add_argument(
        "--torchtitan_path",
        type=str,
        default=None,
        help="Path to torchtitan installation",
    )
    parser.add_argument(
        "--lm_eval_path",
        type=str,
        default=None,
        help="Path to lm-evaluation-harness installation",
    )

    # Other options
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Data type for model",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run evaluation on",
    )

    return parser.parse_args()


def get_seeds(args: argparse.Namespace) -> tuple[int, int, int, int]:
    """Get seeds from args, falling back to base seed."""
    return (
        args.random_seed if args.random_seed is not None else args.seed,
        args.numpy_seed if args.numpy_seed is not None else args.seed,
        args.torch_seed if args.torch_seed is not None else args.seed,
        args.fewshot_seed if args.fewshot_seed is not None else args.seed,
    )


def get_seed_string(args: argparse.Namespace) -> str:
    """Get seed string for lm_eval CLI format."""
    seeds = get_seeds(args)
    return f"{seeds[0]},{seeds[1]},{seeds[2]},{seeds[3]}"


def save_eval_config(args: argparse.Namespace, output_dir: Path) -> None:
    """Save evaluation configuration for reproducibility."""
    seeds = get_seeds(args)

    config = {
        "timestamp": datetime.now().isoformat(),
        "checkpoint": args.checkpoint,
        "tokenizer_path": args.tokenizer_path or args.checkpoint,
        "model_name": args.model_name,
        "model_flavor": args.model_flavor,
        "tasks": args.tasks,
        "num_fewshot": args.num_fewshot,
        "batch_size": args.batch_size,
        "max_seq_len": args.max_seq_len,
        "limit": args.limit,
        "dtype": args.dtype,
        "device": args.device,
        "seeds": {
            "random_seed": seeds[0],
            "numpy_seed": seeds[1],
            "torch_seed": seeds[2],
            "fewshot_seed": seeds[3],
        },
        "command": " ".join(sys.argv),
    }

    config_path = output_dir / "eval_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Configuration saved to: {config_path}")


def main() -> None:
    """Main entry point."""
    args = parse_args()

    # Setup paths
    setup_paths(args.torchtitan_path, args.lm_eval_path)

    import lm_eval

    # Import after path setup
    import numpy as np
    import torch

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get seeds
    seeds = get_seeds(args)
    random_seed, numpy_seed, torch_seed, fewshot_seed = seeds

    # Set seeds for reproducibility
    random.seed(random_seed)
    np.random.seed(numpy_seed)
    torch.manual_seed(torch_seed)

    # Print configuration
    print("=" * 60)
    print("LM-EVALUATION-HARNESS")
    print("=" * 60)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Model: {args.model_name} ({args.model_flavor})")
    print(f"Tasks: {args.tasks}")
    print(f"Batch size: {args.batch_size}")
    print(f"Max seq len: {args.max_seq_len}")
    print(f"Limit: {args.limit or 'None (full evaluation)'}")
    print(
        f"Seeds: random={random_seed}, numpy={numpy_seed}, torch={torch_seed}, fewshot={fewshot_seed}"
    )
    print(f"Output: {output_dir}")
    print("=" * 60)

    # Save configuration
    save_eval_config(args, output_dir)

    # Build model args
    tokenizer_path = args.tokenizer_path or args.checkpoint
    model_args = (
        f"pretrained={args.checkpoint},"
        f"tokenizer_path={tokenizer_path},"
        f"model_name={args.model_name},"
        f"model_flavor={args.model_flavor},"
        f"dtype={args.dtype},"
        f"max_seq_len={args.max_seq_len},"
        f"device={args.device}"
    )

    print(f"\nModel args: {model_args}")
    print("\nStarting evaluation...")

    # Run evaluation
    results = lm_eval.simple_evaluate(
        model="torchtitan",
        model_args=model_args,
        tasks=args.tasks.split(","),
        num_fewshot=args.num_fewshot,
        limit=args.limit,
        batch_size=args.batch_size,
        log_samples=args.log_samples,
    )

    # Save results
    results_path = output_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to: {results_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    for task_name, task_results in results.get("results", {}).items():
        print(f"\n{task_name}:")
        for metric, value in task_results.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")

    # Save summary to text file
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"LM-Evaluation-Harness Results\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Model: {args.model_name} ({args.model_flavor})\n")
        f.write(f"Tasks: {args.tasks}\n")
        f.write(f"Seeds: {get_seed_string(args)}\n\n")

        for task_name, task_results in results.get("results", {}).items():
            f.write(f"\n{task_name}:\n")
            for metric, value in task_results.items():
                if isinstance(value, (int, float)):
                    f.write(f"  {metric}: {value:.4f}\n")

    print(f"\nSummary saved to: {summary_path}")
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

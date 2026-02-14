"""
CLI entry point for async RL training experiments.

Wires together actors and training loops from actors.py and training.py.

Usage:
    python experiments/reinforcement_learning/run.py --num-steps 20 --num-generators 2 --mode both
    python experiments/reinforcement_learning/run.py --help
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Run RL training with Monarch actors.",
    )
    parser.add_argument(
        "--num-steps", type=int, default=20,
        help="Number of training steps per loop (default: 20)",
    )
    parser.add_argument(
        "--num-generators", type=int, default=2,
        help="Number of generator workers (default: 2)",
    )
    parser.add_argument(
        "--num-zorplex", type=int, default=2,
        help="Number of Zorplex tool workers (default: 2)",
    )
    parser.add_argument(
        "--eval-samples", type=int, default=10,
        help="Number of evaluation samples (default: 10)",
    )
    parser.add_argument(
        "--mode", choices=["sync", "async", "both"], default="sync",
        help="Training mode: sync, async, or both (default: sync)",
    )
    parser.add_argument(
        "--plot", type=str, default=None,
        help="Save timeline plot to this path (requires matplotlib; only with --mode both)",
    )
    args = parser.parse_args()

    # Environment setup
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Add src/ to sys.path so actors can import zorplex_rl, rl_primitives, etc.
    _src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "src"))
    if _src_dir not in sys.path:
        sys.path.insert(0, _src_dir)

    # Also add the directory containing actors.py and training.py
    _this_dir = os.path.abspath(os.path.dirname(__file__))
    if _this_dir not in sys.path:
        sys.path.insert(0, _this_dir)

    # Set PYTHONPATH for Monarch subprocesses
    _existing = os.environ.get("PYTHONPATH", "")
    _paths_to_add = f"{_src_dir}:{_this_dir}"
    os.environ["PYTHONPATH"] = f"{_paths_to_add}:{_existing}" if _existing else _paths_to_add

    from training import (
        evaluate,
        print_comparison,
        plot_timeline,
        run_async_loop,
        run_sync_loop,
        setup_actors,
        teardown_actors,
    )

    batch_size = args.num_generators  # Train on exactly one round of generation per step

    if args.mode == "sync":
        _run_sync(args, batch_size, evaluate, run_sync_loop, setup_actors, teardown_actors)
    elif args.mode == "async":
        _run_async(args, batch_size, evaluate, run_async_loop, setup_actors, teardown_actors)
    else:
        _run_both(args, batch_size, evaluate, print_comparison, plot_timeline,
                  run_sync_loop, run_async_loop, setup_actors, teardown_actors)


def _run_sync(args, batch_size, evaluate, run_sync_loop, setup_actors, teardown_actors):
    actors = setup_actors(
        num_generators=args.num_generators,
        num_zorplex=args.num_zorplex,
    )

    print("Evaluating pre-training baseline...")
    pre_eval = evaluate(actors, num_samples=args.eval_samples)
    print(f"Pre-training accuracy: {pre_eval['accuracy']:.0%}")

    sync_stats = run_sync_loop(actors, args.num_steps, args.num_generators)
    print(f"\nSync complete: {sync_stats.wall_time:.2f}s, "
          f"{sync_stats.total_generations} generations, "
          f"{sync_stats.gens_per_second:.2f} gens/s")

    print("Evaluating post-sync performance...")
    post_eval = evaluate(actors, num_samples=args.eval_samples)
    print(f"Post-sync accuracy: {post_eval['accuracy']:.0%}")

    teardown_actors(actors)


def _run_async(args, batch_size, evaluate, run_async_loop, setup_actors, teardown_actors):
    actors = setup_actors(
        num_generators=args.num_generators,
        num_zorplex=args.num_zorplex,
    )

    print("Evaluating pre-training baseline...")
    pre_eval = evaluate(actors, num_samples=args.eval_samples)
    print(f"Pre-training accuracy: {pre_eval['accuracy']:.0%}")

    async_stats = run_async_loop(actors, args.num_steps, args.num_generators, batch_size)
    print(f"\nAsync complete: {async_stats.wall_time:.2f}s, "
          f"{async_stats.total_generations} generations, "
          f"{async_stats.gens_per_second:.2f} gens/s")

    print("Evaluating post-async performance...")
    post_eval = evaluate(actors, num_samples=args.eval_samples)
    print(f"Post-async accuracy: {post_eval['accuracy']:.0%}")

    teardown_actors(actors)


def _run_both(args, batch_size, evaluate, print_comparison, plot_timeline,
              run_sync_loop, run_async_loop, setup_actors, teardown_actors):
    # --- Sync run ---
    actors = setup_actors(
        num_generators=args.num_generators,
        num_zorplex=args.num_zorplex,
    )

    print("Evaluating pre-training baseline...")
    pre_eval = evaluate(actors, num_samples=args.eval_samples)
    print(f"Pre-training accuracy: {pre_eval['accuracy']:.0%}")

    sync_stats = run_sync_loop(actors, args.num_steps, args.num_generators)
    print(f"\nSync complete: {sync_stats.wall_time:.2f}s, "
          f"{sync_stats.total_generations} generations, "
          f"{sync_stats.gens_per_second:.2f} gens/s")

    print("Evaluating post-sync performance...")
    sync_post_eval = evaluate(actors, num_samples=args.eval_samples)
    print(f"Post-sync accuracy: {sync_post_eval['accuracy']:.0%}")

    # --- Tear down and re-spawn for async ---
    teardown_actors(actors)
    print("Re-spawning actors for async run...")
    async_actors = setup_actors(
        num_generators=args.num_generators,
        num_zorplex=args.num_zorplex,
    )

    async_stats = run_async_loop(async_actors, args.num_steps, args.num_generators, batch_size)
    print(f"\nAsync complete: {async_stats.wall_time:.2f}s, "
          f"{async_stats.total_generations} generations, "
          f"{async_stats.gens_per_second:.2f} gens/s")

    print("Evaluating post-async performance...")
    async_post_eval = evaluate(async_actors, num_samples=args.eval_samples)
    print(f"Post-async accuracy: {async_post_eval['accuracy']:.0%}")

    teardown_actors(async_actors)

    # --- Comparison ---
    print_comparison(sync_stats, async_stats, pre_eval, sync_post_eval, async_post_eval)

    if args.plot:
        plot_timeline(sync_stats, async_stats, save_path=args.plot)


if __name__ == "__main__":
    main()

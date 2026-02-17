# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
CLI entry point and training loop for RL experiments.

Usage:
    python experiments/reinforcement_learning/main.py --num-steps 20 --num-generators 2
    python experiments/reinforcement_learning/main.py --help
"""

import argparse
import os
import sys
import time


# ---------------------------------------------------------------------------
# Actor lifecycle
# ---------------------------------------------------------------------------


def setup_actors(
    num_generators: int = 2,
) -> dict:
    """Spawn and initialize all actors.

    Returns a dict with keys: trainer, generators, _proc_meshes.
    """
    from generator import Generator
    from monarch.actor import this_host
    from trainer import Trainer

    host = this_host()

    # 1. Generators -- plain ActorMesh
    gen_procs = host.spawn_procs(per_host={"procs": num_generators})
    generators = gen_procs.spawn("generators", Generator)

    # 2. Trainer
    trainer_procs = host.spawn_procs(per_host={"procs": 1})
    trainer = trainer_procs.spawn("trainer", Trainer)

    # Initialize actors that need setup
    print("[SETUP] Setting up generator workers...")
    generators.setup.call().get()  # broadcast setup to all generators

    print("[SETUP] Setting up trainer...")
    trainer.setup.call_one().get()

    print(f"[SETUP] All actors ready! {num_generators} generators")

    proc_meshes = [gen_procs, trainer_procs]

    return {
        "trainer": trainer,
        "generators": generators,
        "_proc_meshes": proc_meshes,
    }


def teardown_actors(actors: dict) -> None:
    """Stop all ProcMeshes, releasing processes and GPU memory."""
    for pm in actors.get("_proc_meshes", []):
        try:
            pm.stop("teardown for re-init").get()
        except Exception:
            pass  # Best-effort cleanup
    print("[TEARDOWN] All actors stopped.")


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(actors: dict, num_samples: int = 10, seed: int = 42) -> dict:
    """Run evaluation on the trainer's current model.

    Returns a dict with accuracy, correct, total, avg_turns, avg_tools,
    format_rate, and failure_modes.
    """
    return (
        actors["trainer"]
        .evaluate_zorplex.call_one(
            num_samples=num_samples,
            seed=seed,
        )
        .get()
    )


def _print_eval(label: str, eval_result: dict) -> None:
    """Print all evaluation stats."""
    fm = eval_result["failure_modes"]
    print(f"\n{label}:")
    print(
        f"  Accuracy:          {eval_result['accuracy']:.0%} ({eval_result['correct']}/{eval_result['total']})"
    )
    print(f"  Format compliance: {eval_result['format_rate']:.0%}")
    print(f"  Avg turns:         {eval_result['avg_turns']:.1f}")
    print(f"  Avg tool calls:    {eval_result['avg_tools']:.1f}")
    print(
        f"  Failure modes:     success={fm['success']} wrong_format={fm['wrong_format']} "
        f"tool_spam={fm['tool_spam']} wrong_answer={fm['wrong_answer']}"
    )


# ---------------------------------------------------------------------------
# Sync training loop
# ---------------------------------------------------------------------------


def run_sync_loop(
    actors: dict,
    num_steps: int,
    num_generators: int,
    verbose: bool = False,
) -> dict:
    """Run synchronous training: broadcast generate -> train -> repeat.

    Uses .call() to broadcast generate_trajectory to all generators
    simultaneously, waits for ALL generators to finish, then trains.

    Returns a dict with wall_time, total_generations, and gens_per_second.
    """
    print("\n" + "=" * 60)
    print("SYNC MODE: Broadcast Generate -> Train")
    print("=" * 60)

    trainer = actors["trainer"]
    generators = actors["generators"]

    total_generations = 0
    advantage_baseline = 0.5
    t0 = time.perf_counter()

    for step in range(num_steps):
        # Generate trajectories -- broadcast to ALL generators
        gen_start = time.perf_counter()
        traj_mesh = generators.generate_trajectory.call().get()

        # Collect into a plain list
        batch = list(traj_mesh.values())

        # Print a sample of generations
        if verbose:
            for t in batch[:2]:
                print(f"  Q: {t.task_question}")
                print(f"  A: {t.response_text[:300]}")
                print(
                    f"  correct={t.is_correct} reward={t.reward:.2f} format={t.has_answer_tag}"
                )
                print()

        gen_time = time.perf_counter() - gen_start
        total_generations += num_generators

        # Train directly on the batch (no buffer in sync mode)
        train_start = time.perf_counter()
        if batch:
            metrics = trainer.train_step.call_one(batch, advantage_baseline).get()
            advantage_baseline = 0.9 * advantage_baseline + 0.1 * metrics.avg_reward

            # Sync weights to all generators (broadcast)
            state_dict, ver = trainer.get_state_dict.call_one().get()
            generators.sync_weights.call(state_dict, ver).get()

        train_time = time.perf_counter() - train_start

        correct_count = sum(1 for t in traj_mesh.values() if t.is_correct)
        format_count = sum(1 for t in traj_mesh.values() if t.has_answer_tag)
        loss_str = (
            f" loss={metrics.loss:.4f} reward={metrics.avg_reward:.2f}" if batch else ""
        )
        print(
            f"[SYNC {step + 1:2d}] {correct_count}/{num_generators} correct "
            f"{format_count}/{num_generators} formatted{loss_str} "
            f"gen={gen_time * 1000:.0f}ms train={train_time * 1000:.0f}ms"
        )

    wall_time = time.perf_counter() - t0
    gens_per_second = total_generations / wall_time if wall_time > 0 else 0

    return {
        "wall_time": wall_time,
        "total_generations": total_generations,
        "gens_per_second": gens_per_second,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run RL training with Monarch actors.",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=20,
        help="Number of training steps (default: 20)",
    )
    parser.add_argument(
        "--num-generators",
        type=int,
        default=2,
        help="Number of generator workers (default: 2)",
    )
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=16,
        help="Number of evaluation samples (default: 10)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Print sample generations on each step",
    )
    args = parser.parse_args()

    # Environment setup
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Add the experiment directory to sys.path so actors can import modules
    _this_dir = os.path.abspath(os.path.dirname(__file__))
    if _this_dir not in sys.path:
        sys.path.insert(0, _this_dir)

    # Set PYTHONPATH for Monarch subprocesses
    _existing = os.environ.get("PYTHONPATH", "")
    os.environ["PYTHONPATH"] = f"{_this_dir}:{_existing}" if _existing else _this_dir

    actors = setup_actors(
        num_generators=args.num_generators,
    )

    print("Evaluating pre-training baseline...")
    pre_eval = evaluate(actors, num_samples=args.eval_samples)
    _print_eval("Pre-training baseline", pre_eval)

    stats = run_sync_loop(
        actors, args.num_steps, args.num_generators, verbose=args.verbose
    )
    print(
        f"\nTraining complete: {stats['wall_time']:.2f}s, "
        f"{stats['total_generations']} generations, "
        f"{stats['gens_per_second']:.2f} gens/s"
    )

    print("Evaluating post-training performance...")
    post_eval = evaluate(actors, num_samples=args.eval_samples)
    _print_eval("Post-training results", post_eval)

    teardown_actors(actors)


if __name__ == "__main__":
    main()

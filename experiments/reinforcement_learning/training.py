"""
Training loops and orchestration for async RL.

Provides:
- TimingEvent / TimingStats: Timing infrastructure for profiling
- setup_actors / teardown_actors: Actor lifecycle management
- run_sync_loop / run_async_loop: Sync and async training loops
- evaluate: Run evaluation on the current model
- print_comparison: Text summary of sync vs async results
- plot_timeline: Matplotlib timeline visualization
"""

import threading
import time
from dataclasses import dataclass, field

from monarch.actor import this_host

from actors import (
    GeneratorWorker,
    ReplayBuffer,
    TrainerActor,
    ZorplexWorker,
)
from monarch_utils.services import Service, register_service


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TimingEvent:
    """A single timed event for timeline visualization."""
    actor_id: str
    event_type: str  # "generate", "train", "sync"
    start_time: float
    duration: float


@dataclass
class TimingStats:
    """Timing statistics for a training run."""
    mode: str
    num_generators: int
    num_steps: int
    total_generations: int
    wall_time: float
    gen_times: list = field(default_factory=list)
    train_times: list = field(default_factory=list)
    events: list = field(default_factory=list)  # List of TimingEvent
    rdma_syncs: int = 0
    direct_syncs: int = 0
    staleness: list = field(default_factory=list)  # policy_version gaps per train batch

    @property
    def gens_per_second(self) -> float:
        return self.total_generations / self.wall_time if self.wall_time > 0 else 0

    @property
    def steps_per_second(self) -> float:
        return self.num_steps / self.wall_time if self.wall_time > 0 else 0


# ---------------------------------------------------------------------------
# Actor lifecycle
# ---------------------------------------------------------------------------

def setup_actors(
    num_generators: int = 2,
    num_zorplex: int = 2,
    buffer_max_size: int = 500,
) -> dict:
    """Spawn and initialize all actors.

    Returns a dict with keys: trainer, buffer, generators, zorplex_svc, _proc_meshes.
    """
    host = this_host()

    # 1. ZorplexWorkers -- wrapped in a Service for health tracking and round-robin
    zorplex_worker_procs = host.spawn_procs(per_host={"procs": num_zorplex})
    zorplex_svc_procs = host.spawn_procs(per_host={"procs": 1})
    zorplex_svc = zorplex_svc_procs.spawn(
        "zorplex_svc", Service,
        service_name="zorplex", worker_class=ZorplexWorker,
        procs=zorplex_worker_procs, procs_per_replica=1,
        difficulty="easy",
    )

    # 2. Generators -- plain ActorMesh
    gen_procs = host.spawn_procs(per_host={"procs": num_generators})
    generators = gen_procs.spawn("generators", GeneratorWorker)

    # 3. ReplayBuffer
    buffer_procs = host.spawn_procs(per_host={"procs": 1})
    buffer = buffer_procs.spawn("buffer", ReplayBuffer, max_size=buffer_max_size)

    # 4. Trainer
    trainer_procs = host.spawn_procs(per_host={"procs": 1})
    trainer = trainer_procs.spawn("trainer", TrainerActor)

    # Initialize actors that need setup
    zorplex_svc.ping.call_one().get()

    print("[SETUP] Setting up generator workers...")
    generators.setup.call().get()  # broadcast setup to all generators

    buffer.stats.call_one().get()

    print("[SETUP] Setting up trainer...")
    trainer.setup.call_one().get()

    register_service("zorplex", zorplex_svc)

    print(f"[SETUP] All actors ready! {num_generators} generators, {num_zorplex} zorplex workers")

    proc_meshes = [zorplex_worker_procs, zorplex_svc_procs, gen_procs, buffer_procs, trainer_procs]

    return {
        "trainer": trainer,
        "buffer": buffer,
        "generators": generators,
        "zorplex_svc": zorplex_svc,
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
    return actors["trainer"].evaluate_zorplex.call_one(
        num_samples=num_samples, seed=seed,
    ).get()


# ---------------------------------------------------------------------------
# Sync training loop
# ---------------------------------------------------------------------------

def run_sync_loop(
    actors: dict,
    num_steps: int,
    num_generators: int,
) -> TimingStats:
    """Run synchronous training: broadcast generate -> train -> repeat.

    Uses .call() to broadcast generate_trajectory to all generators
    simultaneously, waits for ALL generators to finish, then trains.
    """
    print("\n" + "=" * 60)
    print("SYNC MODE: Broadcast Generate -> Train")
    print("=" * 60)

    trainer = actors["trainer"]
    generators = actors["generators"]

    stats = TimingStats(
        mode="SYNC",
        num_generators=num_generators,
        num_steps=num_steps,
        total_generations=0,
        wall_time=0,
    )

    baseline = 0.5
    t0 = time.perf_counter()

    for step in range(num_steps):
        # Generate trajectories -- broadcast to ALL generators
        gen_start = time.perf_counter()
        traj_mesh = generators.generate_trajectory.call().get()

        # Collect into a plain list
        batch = list(traj_mesh.values())

        gen_time = time.perf_counter() - gen_start
        stats.gen_times.append(gen_time)
        stats.total_generations += num_generators

        # Record one event per generator (they ran in parallel)
        for gi in range(num_generators):
            stats.events.append(TimingEvent(
                actor_id=f"Gen{gi}",
                event_type="generate",
                start_time=gen_start - t0,
                duration=gen_time,
            ))

        # Train directly on the batch (no buffer in sync mode)
        train_start = time.perf_counter()
        if batch:
            metrics = trainer.train_step.call_one(batch, baseline).get()
            baseline = 0.9 * baseline + 0.1 * metrics.avg_reward

            batch_staleness = [metrics.policy_version - t.policy_version for t in batch]
            stats.staleness.extend(batch_staleness)

            # Sync weights to all generators (broadcast)
            try:
                handle, param_meta, version, total_bytes = trainer.get_weight_handle.call_one().get()
                if handle is not None:
                    generators.sync_weights_from_buffer.call(handle, param_meta, version, total_bytes).get()
                else:
                    state_dict, ver = trainer.get_state_dict.call_one().get()
                    generators.sync_weights.call(state_dict, ver).get()
            except Exception:
                pass  # Non-fatal: generators will use slightly stale weights

        train_time = time.perf_counter() - train_start
        stats.train_times.append(train_time)
        stats.events.append(TimingEvent(
            actor_id="Train",
            event_type="train",
            start_time=train_start - t0,
            duration=train_time,
        ))

        correct_count = sum(1 for t in traj_mesh.values() if t.is_correct)
        format_count = sum(1 for t in traj_mesh.values() if t.has_answer_tag)
        print(f"[SYNC {step + 1:2d}] {correct_count}/{num_generators} correct "
              f"{format_count}/{num_generators} formatted "
              f"gen={gen_time * 1000:.0f}ms train={train_time * 1000:.0f}ms")

    stats.wall_time = time.perf_counter() - t0
    return stats


# ---------------------------------------------------------------------------
# Async training loop
# ---------------------------------------------------------------------------

def run_async_loop(
    actors: dict,
    num_steps: int,
    num_generators: int,
    batch_size: int,
) -> TimingStats:
    """Run asynchronous training: generators and trainer run concurrently.

    - 1 thread per generator (each uses .slice() to address its generator)
    - Training in main thread
    - Each generator pulls latest weights before each trajectory
    """
    print("\n" + "=" * 60)
    print(f"ASYNC MODE: {num_generators} Generators + 1 Trainer (Concurrent)")
    print("=" * 60)

    trainer = actors["trainer"]
    buffer = actors["buffer"]
    generators = actors["generators"]

    stats = TimingStats(
        mode="ASYNC",
        num_generators=num_generators,
        num_steps=num_steps,
        total_generations=0,
        wall_time=0,
    )

    lock = threading.Lock()
    stop_flag = threading.Event()
    t0 = time.perf_counter()

    def generation_loop(gen_idx):
        """Each generator gets its own thread, using .slice() for individual access."""
        gen = generators.slice(procs=gen_idx)
        while not stop_flag.is_set():
            gen_start = time.perf_counter()

            try:
                # Pull latest weights before generating.
                handle, param_meta, version, total_bytes = trainer.get_weight_handle.call_one().get()
                if handle is not None:
                    synced = gen.sync_weights_from_buffer.call_one(handle, param_meta, version, total_bytes).get()
                    if synced:
                        with lock:
                            stats.rdma_syncs += 1
                else:
                    state_dict, ver = trainer.get_state_dict.call_one().get()
                    synced = gen.sync_weights.call_one(state_dict, ver).get()
                    if synced:
                        with lock:
                            stats.direct_syncs += 1

                # Generate trajectory
                traj = gen.generate_trajectory.call_one().get()
                buffer.add.call_one(traj).get()

                gen_time = time.perf_counter() - gen_start
                with lock:
                    stats.gen_times.append(gen_time)
                    stats.total_generations += 1
                    count = stats.total_generations
                    stats.events.append(TimingEvent(
                        actor_id=f"Gen{gen_idx}",
                        event_type="generate",
                        start_time=gen_start - t0,
                        duration=gen_time,
                    ))

                status = "correct" if traj.is_correct else traj.failure_mode
                print(f"[GEN{gen_idx} #{count:2d}] {status} gen={gen_time * 1000:.0f}ms")

            except Exception as e:
                print(f"[GEN{gen_idx}] Error: {e}, retrying...")
                continue

    # Start 1 thread per generator
    gen_threads = []
    for idx in range(num_generators):
        t = threading.Thread(target=generation_loop, args=(idx,), daemon=True)
        t.start()
        gen_threads.append(t)

    # Training in main thread
    train_steps_done = 0
    baseline = 0.5

    while train_steps_done < num_steps:
        # Wait for enough samples
        while True:
            size = buffer.size.call_one().get()
            if size >= batch_size:
                break
            time.sleep(0.02)

        train_start = time.perf_counter()
        batch = buffer.sample.call_one(batch_size).get()
        if batch:
            metrics = trainer.train_step.call_one(batch, baseline).get()
            baseline = 0.9 * baseline + 0.1 * metrics.avg_reward

            batch_staleness = [metrics.policy_version - t.policy_version for t in batch]
            with lock:
                stats.staleness.extend(batch_staleness)

        train_time = time.perf_counter() - train_start
        with lock:
            stats.train_times.append(train_time)
            stats.events.append(TimingEvent(
                actor_id="Train",
                event_type="train",
                start_time=train_start - t0,
                duration=train_time,
            ))
        train_steps_done += 1

        print(f"[TRAIN {train_steps_done:2d}] time={train_time * 1000:.0f}ms buffer={size}")

    stop_flag.set()

    for t in gen_threads:
        t.join(timeout=2.0)

    stats.wall_time = time.perf_counter() - t0
    return stats


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_comparison(
    sync_stats: TimingStats,
    async_stats: TimingStats,
    pre_eval: dict,
    sync_post_eval: dict,
    async_post_eval: dict,
) -> None:
    """Print a text summary comparing sync and async training results."""
    speedup = sync_stats.wall_time / async_stats.wall_time if async_stats.wall_time > 0 else 0
    gen_ratio = async_stats.gens_per_second / sync_stats.gens_per_second if sync_stats.gens_per_second > 0 else 0

    def _avg(lst):
        return sum(lst) / len(lst) if lst else 0.0

    avg_sync_gen = _avg(sync_stats.gen_times) * 1000
    avg_async_gen = _avg(async_stats.gen_times) * 1000
    avg_sync_train = _avg(sync_stats.train_times) * 1000
    avg_async_train = _avg(async_stats.train_times) * 1000

    async_syncs = async_stats.rdma_syncs + async_stats.direct_syncs

    print("\n" + "=" * 70)
    print("SYNC vs ASYNC COMPARISON")
    print("=" * 70)
    print(f"{'Metric':<25} {'SYNC':>12} {'ASYNC':>12} {'Ratio':>15}")
    print("-" * 70)
    print(f"{'Wall time':<25} {sync_stats.wall_time:>11.2f}s {async_stats.wall_time:>11.2f}s {speedup:>13.2f}x speedup")
    print(f"{'Generations':<25} {sync_stats.total_generations:>12} {async_stats.total_generations:>12} {async_stats.total_generations / max(sync_stats.total_generations, 1):>13.1f}x")
    print(f"{'Gens/second':<25} {sync_stats.gens_per_second:>12.2f} {async_stats.gens_per_second:>12.2f} {gen_ratio:>13.1f}x throughput")
    print(f"{'Avg gen time':<25} {avg_sync_gen:>10.0f}ms {avg_async_gen:>10.0f}ms")
    print(f"{'Avg train time':<25} {avg_sync_train:>10.0f}ms {avg_async_train:>10.0f}ms")
    print(f"{'Weight syncs':<25} {sync_stats.total_generations:>12} {async_syncs:>12}")

    # Staleness
    sync_avg_stale = _avg(sync_stats.staleness)
    async_avg_stale = _avg(async_stats.staleness)
    async_max_stale = max(async_stats.staleness) if async_stats.staleness else 0
    print(f"\n{'Avg staleness':<25} {sync_avg_stale:>12.1f} {async_avg_stale:>12.1f}")
    print(f"{'Max staleness':<25} {max(sync_stats.staleness) if sync_stats.staleness else 0:>12} {async_max_stale:>12}")

    # Evaluation results
    def _delta(post, pre, key):
        return post[key] - pre[key]

    def _dir(d):
        if d > 0:
            return "improved"
        elif d == 0:
            return "unchanged"
        return "decreased"

    sync_acc_d = _delta(sync_post_eval, pre_eval, "accuracy")
    async_acc_d = _delta(async_post_eval, pre_eval, "accuracy")

    print(f"\n{'':=<70}")
    print("TRAINING RESULTS: Baseline vs Sync vs Async")
    print(f"{'':=<70}")
    print(f"{'Metric':<25} {'Baseline':>12} {'After Sync':>12} {'After Async':>12}")
    print("-" * 70)
    print(f"{'Accuracy':<25} {pre_eval['accuracy']:>11.0%} {sync_post_eval['accuracy']:>11.0%} {async_post_eval['accuracy']:>11.0%}")
    print(f"{'Format compliance':<25} {pre_eval['format_rate']:>11.0%} {sync_post_eval['format_rate']:>11.0%} {async_post_eval['format_rate']:>11.0%}")
    print(f"{'Avg turns':<25} {pre_eval['avg_turns']:>12.1f} {sync_post_eval['avg_turns']:>12.1f} {async_post_eval['avg_turns']:>12.1f}")
    print(f"{'Avg tool calls':<25} {pre_eval['avg_tools']:>12.1f} {sync_post_eval['avg_tools']:>12.1f} {async_post_eval['avg_tools']:>12.1f}")

    print(f"\nFailure modes:")
    for mode in ["success", "wrong_format", "tool_spam", "wrong_answer"]:
        pre_v = pre_eval["failure_modes"][mode]
        sync_v = sync_post_eval["failure_modes"][mode]
        async_v = async_post_eval["failure_modes"][mode]
        print(f"  {mode:<20} {pre_v:>8} {sync_v:>12} {async_v:>12}")

    print(f"\nSync accuracy {_dir(sync_acc_d)} by {abs(sync_acc_d):.0%}.")
    print(f"Async accuracy {_dir(async_acc_d)} by {abs(async_acc_d):.0%}.")


def plot_timeline(
    sync_stats: TimingStats,
    async_stats: TimingStats,
    save_path: str | None = None,
) -> None:
    """Plot Gantt-chart timelines for sync and async training runs.

    If matplotlib is not available, prints a warning and returns.
    If save_path is provided, saves to that file; otherwise calls plt.show().
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        print("[plot_timeline] matplotlib not available, skipping timeline plot.")
        return

    def _plot_one(stats, ax, title):
        color_map = {
            "generate": "#4CAF50",  # green
            "train": "#E91E63",     # pink
            "sync": "#9C27B0",      # purple
        }

        actor_ids = []
        for ev in stats.events:
            if ev.actor_id not in actor_ids:
                actor_ids.append(ev.actor_id)

        gen_ids = sorted([a for a in actor_ids if a.startswith("Gen")])
        other_ids = [a for a in ["Train", "Sync"] if a in actor_ids]
        actor_ids = gen_ids + other_ids

        y_map = {aid: i for i, aid in enumerate(actor_ids)}

        for ev in stats.events:
            if ev.actor_id in y_map:
                y = y_map[ev.actor_id]
                color = color_map.get(ev.event_type, "#999999")
                ax.barh(y, ev.duration, left=ev.start_time, height=0.6,
                        color=color, alpha=0.8, edgecolor="white", linewidth=0.5)

        ax.set_yticks(range(len(actor_ids)))
        ax.set_yticklabels(actor_ids)
        ax.set_xlabel("Wall time (seconds)")
        ax.set_title(title)
        ax.invert_yaxis()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

    _plot_one(sync_stats, ax1, f"SYNC ({sync_stats.wall_time:.1f}s)")
    _plot_one(async_stats, ax2, f"ASYNC ({async_stats.wall_time:.1f}s)")

    legend_patches = [
        mpatches.Patch(color="#4CAF50", label="Generate"),
        mpatches.Patch(color="#E91E63", label="Train"),
    ]
    fig.legend(handles=legend_patches, loc="upper right", framealpha=0.9)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot_timeline] Saved to {save_path}")
    else:
        plt.show()

    plt.close(fig)

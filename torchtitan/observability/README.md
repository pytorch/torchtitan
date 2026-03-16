# TorchTitan Observability

Structured logging and metrics for distributed training. Works for both
SPMD pretraining and RL with multiple actors (no shared process group).

## Overview

Mental model: Every metric first goes to structured JSONL using python's stdout logger. Then, a background subprocess collects and aggregates for dashboards.

There are two types of metrics:
- **Experiment metrics** — training values (loss, throughput, memory) that get
  aggregated across ranks and sent to WandB/TB/console.
- **System metrics** — per-rank timing and scalar snapshots for debugging tools
  (Perfetto, DuckDB, LLM agents).

```
Experiment: record_metric(key, Metric) → logger → experiment.jsonl → logging subprocess → aggregate → WandB/TB/console

System:     record_span / record_event → logger → system.jsonl     → analysis tools (Perfetto, DuckDB, LLM agents)
```

This JSONL-first design gives us two advantages:

1. **No metric plumbing.** Each process calls `record_metric(key, value)`
   locally. No passing dicts between functions or actors. The logging
   subprocess reads all JSONL files in the background.

2. **Debuggability.** Every rank's raw metrics are preserved as JSONL on disk.
   An LLM agent or DuckDB query can answer "what happened on rank 47 at step
   3000?"

### Quickstart

```python
from torchtitan.observability import (
    init_observability, set_step, add_step_tag,
    record_span, record_event, record_metric,
    EventType, MeanMetric, NoOpMetric,
)
from torchtitan.tools.logging import init_logger

# Console logging (stdout with [titan] format)
init_logger()

# JSONL file handlers for structured logging
init_observability(source="trainer", output_dir="./outputs")

for step in range(steps):
    # Stamp all subsequent JSONL entries with step=step
    set_step(step)

    if should_garbage_collect:
        add_step_tag("gc")
        with record_span("trainer_time/gc_s", EventType.GC_COLLECT):
            run_gc()

    with record_span("trainer_time/forward_backward_s", EventType.FWD_BWD):
        reduced_loss = model.fwd_bwd(batch)
        record_metric("training/loss_mean", NoOpMetric(value=reduced_loss))
```

## Setup

```python
from torchtitan.tools.logging import init_logger
from torchtitan.observability import init_observability

init_logger()  # console StreamHandler (stdout, [titan] prefix)
init_observability(source="trainer", output_dir="./outputs")  # JSONL file handlers
```

Call once per process, before any logging calls. `init_logger()` sets up
console output. `init_observability()` adds JSONL file handlers, which can be modified to
a different backend (e.g. Grafana).

```
                    ┌─ Console StreamHandler (stdout, [titan] format)
Root Logger ────────┤    set up by init_logger()
                    │
                    │         ┌─ StructuredLoggingHandler → system.jsonl
System Logger ──────┤─────────┤─ StructuredJSONFormatter (rank, source, caller, timestamp)
                    │         └─ InflightEventTrackingHandler (tracks open spans for crash forensics)
                    │
                    │         ┌─ ExperimentLoggingHandler → experiment.jsonl
Experiment Logger ──┤─────────┤─ ExperimentJSONFormatter (rank, source, caller, timestamp)
```

Rank and source are baked into the formatters at init time. Every JSONL entry
automatically includes `rank`, `source`, `caller` (file:line:function), and
`timestamp`.



<<<<<<< Updated upstream
set_step(42)              # stamps all subsequent JSONL entries with step=42
add_step_tag("gc")        # annotate: GC ran this step
add_step_tag("eval")      # annotate: validation ran this step
clear_step_tags()         # reset tags (called at the start of each step)
```

Step and tags are stored as module-level globals — one value per process.
These tags and steps could be used for custom aggregation, e.g. in RL
aggregate scores per policy version.

## 4. Experiment Metrics: record_metric

```python
from torchtitan.observability.metrics import (
    record_metric, MeanMetric, MaxMetric, SumMetric, MinMetric, NoOpMetric,
)

record_metric("trainer_throughput/tps_mean", MeanMetric(sum=1234.5, weight=262))
record_metric("trainer_gradient/norm_max", MaxMetric(value=12.3))
record_metric("trainer_memory/ooms_sum", SumMetric(value=0))
record_metric("loss/trainer_loss_mean", NoOpMetric(value=0.038))  # already all_reduced
record_metric("trainer_schedule/lr", NoOpMetric(value=3e-4))      # same on all ranks
```
=======
## Experiment Metrics: record_metric
>>>>>>> Stashed changes

Each call serializes to experiment JSONL immediately. Step comes from
`set_step()`. Rank, source, caller, and timestamp are added automatically.

Example: two ranks log the same metric on the same step, then the subprocess
aggregates them:

```python
record_metric("trainer_throughput/tps_mean", MeanMetric(sum=tps))
```

```json
// dump_dir/experiment_logs/trainer_rank_0_experiment.jsonl
{"key": "trainer_throughput/tps_mean", "reduce": "MeanMetric",
 "sum": 1234.5, "weight": 1.0,
 "step": 42, "rank": 0, "source": "trainer",
 "caller": "torchtitan/trainer.py:542:train_step", "timestamp": 1708200121.724}

// dump_dir/experiment_logs/trainer_rank_1_experiment.jsonl
{"key": "trainer_throughput/tps_mean", "reduce": "MeanMetric",
 "sum": 1180.2, "weight": 1.0,
 "step": 42, "rank": 1, "source": "trainer",
 "caller": "torchtitan/trainer.py:542:train_step", "timestamp": 1708200121.725}

// After aggregation: (1234.5 + 1180.2) / (1.0 + 1.0) = 1207.35
```

### Reduce types

| Type | Constructor | Aggregation |
|------|------------|-------------|
| `MeanMetric` | `MeanMetric(sum=x, weight=1)` | `sum(sums) / sum(weights)` |
| `MaxMetric` | `MaxMetric(value=x)` | `max(values)` |
| `MinMetric` | `MinMetric(value=x)` | `min(values)` |
| `SumMetric` | `SumMetric(value=x)` | `sum(values)` |
| `NoOpMetric` | `NoOpMetric(value=x)` | Pass through (in aggregation, selects random row) |

Custom reduce types can be registered in `REDUCE_REGISTRY`.

### Known limitations

- Does not work with tensors on accelerators (GPU) or inside `torch.compile`
  regions. All values must be Python floats/ints (call `.item()` first).
  The choice is who does the cross-rank reduction:

  a) **You reduce, then log.** Call `all_reduce` + `.item()` yourself, then
  record with `NoOpMetric` (the subprocess takes the value as-is):
  ```python
  dist.all_reduce(my_tensor, op=dist.ReduceOp.SUM, group=dp_mesh.get_group())
  record_metric("my_metric", NoOpMetric(value=my_tensor.item()))
  ```

  b) **Log per-rank, let the subprocess reduce.** Each rank logs its local
  value and the subprocess aggregates across all rank JSONL files:
  ```python
  if rank in dp_mesh.get_group():
    record_metric("my_metric", SumMetric(value=my_tensor.item()))
  ```

An API that is compile-safe and friendly for distributed tensors is planned.

## System Metrics: record_span and record_event

### record_span — timing a code region

On enter: writes a START event to system JSONL.
On exit: writes an END event with duration in milliseconds.

```python
from torchtitan.observability import record_span, EventType

with record_span("trainer_time/forward_backward_s", EventType.FWD_BWD):
    output = model(batch)
    loss.backward()
```

### System JSONL format

Values are grouped into four typed columns (`int`, `normal`, `double`,
`normvector`) for easy ingestion into Grafana, DuckDB, or Scuba:

```json
{"int": {"step": 42, "rank": 0, "time_us": 1708200121724000},
 "normal": {"log_type_name": "fwd_bwd_end", "source": "trainer",
            "message": "[step 42] trainer_time/forward_backward_s fwd_bwd_end took 123.45 ms",
            "caller": "torchtitan/trainer.py:730:train_step"},
 "double": {"value": 123.45},
 "normvector": {"step_tags": ["gc"]}}
```

EventType is optional — description is used as the event type if omitted:
```python
with record_span("trainer_time/forward_backward_s"):
    rollouts = generate(prompts)
```

By default `log_to_metrics=True`, the exit also calls `record_metric` with
the duration in **seconds** as a `MeanMetric`, so timing data flows to
WandB/TB alongside other experiment metrics.

To disable:
```python
with record_span("rl_time/rollout_s", log_to_metrics=False):
    rollouts = generate(prompts)
```


### record_event — point-in-time scalars

```python
from torchtitan.observability import record_event

record_event({"train.step": 42, "train.tflops": 45.6})
```

Writes to system JSONL only. Does NOT flow to experiment JSONL or WandB.
Used for per-rank diagnostic data (TFLOPS, loss, grad_norm every step).



## Aggregation (Logging Subprocess)

The training process never reads JSONL or writes to backends directly.
A background subprocess on rank 0 handles all aggregation and output:

```
                  Training Process (all ranks)
                  ┌─────────────────────────────────┐
                  │  record_metric("loss", ...)     │──→ experiment.jsonl (per rank)
                  │  record_span("fwd_bwd", ...)    │──→ system.jsonl (per rank)
                  │                                 │
                  │  # on log steps:                │
                  │  barrier()                      │
                  │  log_queue.put(step) ───────────│──┐  (~0.1ms, non-blocking)
                  └─────────────────────────────────┘  │
                                                       │
                  Logging Subprocess (rank 0 only)     │
                  ┌─────────────────────────────────┐  │
                  │  step = queue.get() ◄───────────│──┘
                  │  read all experiment.jsonl files│
                  │  aggregate by key (reduce)      │
                  │  write to WandB / TensorBoard   │
                  │  print to console               │
                  └─────────────────────────────────┘
```

The subprocess is spawned via `logging_worker` from `aggregation.py`. Here's
the RL pattern where the controller owns the subprocess directly:

```python
import multiprocessing
from torchtitan.observability import logging_worker

# Spawn the logging subprocess
log_queue = multiprocessing.Queue()
log_process = multiprocessing.Process(
    target=logging_worker,
    args=(log_queue, OUTPUT_DIR),
    kwargs={
        "enable_wandb": True,
        "enable_tensorboard": False,
        "console_log_metric_keys": [
            "training/loss_mean",
            "training/grad_norm_max",
            "rl/reward_mean",
            "trainer_throughput/tps_mean",
        ],
    },
    daemon=True,
)
log_process.start()

# Each actor calls record_metric locally — writes go to JSONL
# ...

# After each step, signal the subprocess to read + aggregate + flush
log_queue.put((step, False))  # (step, is_validation)

# Shutdown
log_queue.put(None)
log_process.join()
```

For SPMD training, `MetricsProcessor` wraps this lifecycle automatically —
see `observability/metrics_processor.py`.

`aggregate()` groups entries by key and delegates reduction to the
`REDUCE_REGISTRY` based on each entry's `"reduce"` field. Console output
also comes from the subprocess (not from `logger.info` in the training process).

**Timing (local filesystem, 100 metrics/rank):**

Aggregation benchmarks are measured on local filesystem. NFS will be slower
for the read, but shouldn't be an issue since it is non-blocking.

| Scale | Read | Aggregate | Total |
|-------|------|-----------|-------|
| 10 files (1K entries) | 9ms | 0.5ms | 10ms |
| 100 files (10K entries) | 88ms | 6ms | 94ms |
| 500 files (50K entries) | 333ms | 35ms | 368ms |

None of this blocks training. Training pays only the signal cost (~0.1ms).

## Analysis Tools

### Gantt chart from system JSONL

```python
from torchtitan.observability.analysis import generate_gantt_trace

generate_gantt_trace("outputs/system_logs/", "outputs/analysis/gantt.json")
```

Reads all system JSONL files (all ranks), produces a Chrome Trace JSON.
Open in `chrome://tracing` or Perfetto to see a Gantt chart of every
`record_span` across all ranks.

Example output from toy_rl (view in chrome://tracing or https://ui.perfetto.dev):

```
<<<<<<< Updated upstream
observability/
    __init__.py             # Public API re-exports
    step_state.py           # Globals: _STEP, _STEP_TAGS, set_step, add_step_tag, clear_step_tags
    _constants.py           # Logger names, metric entry markers (import cycle breaker)
    structured_logging.py   # System pipeline: init_observability, record_span, record_event, EventType
    metrics.py              # Experiment pipeline: record_metric, MetricValue types, REDUCE_REGISTRY
    aggregation.py          # aggregate() + logging_worker subprocess + JSONL readers
    logging_boundary.py     # EveryNSteps schedule
    rollout_logger.py       # RL: RolloutLogger, filter_top_bottom
    analysis.py             # Post-training: generate_gantt_trace (Gantt chart from system JSONL)
    README.md               # This file
=======
                       step 1                              step 2
                  ┌──────────────────────────┐       ┌───────────────────────────┐
trainer rank 0    │ ▓▓ fwd_bwd ▓▓  ▓ optim ▓ │       │ ▓ fwd_bwd ▓▓ ▓▓ optim ▓   │
trainer rank 1    │  ▓▓ fwd_bwd ▓ ▓ optim ▓  │       │  ▓▓ fwd_bwd ▓▓ ▓ optim ▓  │
trainer rank 2    │ ▓ fwd_bwd ▓▓ ▓ optim ▓▓  │       │  ▓ fwd_bwd ▓▓▓ ▓ optim ▓▓ │
trainer rank 3    │  ▓▓ fwd_bwd ▓▓▓ ▓ optim ▓│       │ ▓ fwd_bwd ▓▓ ▓ optim ▓    │
                  └──────────────────────────┘       └───────────────────────────┘
controller        ▓▓▓▓▓▓ training_s ▓▓▓▓▓▓▓ ▓ scoring ▓ ▓▓▓▓▓▓ training_s ▓▓▓▓▓▓
rollouter         ▓▓▓                                     ▓▓▓
reward                                       ▓▓▓                              ▓▓▓
                  ├──────────────────────────┼───────────┼──────────────────────┤
                  0s                         3s          4s                     7s
```

Each `record_span` becomes a bar. Every actor type is a separate process
row in Perfetto, with threads per rank — stragglers and idle gaps are
immediately visible.

## RL: RolloutLogger

Logs RL rollouts to JSONL with optional filtering:

```python
from torchtitan.observability import RolloutLogger

rollout_logger = RolloutLogger(
    output_dir="outputs/rollouts",
    filter_fn=lambda records: filter_top_bottom(records, key="reward", k=2),
)

rollouts = [
     {"prompt": "what is 2+2?", "response": "4", "reward": 0.95},
     {"prompt": "write a poem", "response": "roses...", "reward": 0.3}
     ]

# Log rollout dicts with optional metadata
rollout_logger.log(
    rollouts,
    metadata={"step": 42, "batch_size": 64},
)
rollout_logger.close()
```

Output JSONL (metadata stored under `__metadata__` to avoid key collisions):
```json
{"prompt": "what is 2+2?", "response": "4", "reward": 0.95, "__metadata__": {"step": 42, "batch_size": 64}}
{"prompt": "write a poem", "response": "roses...", "reward": 0.3, "__metadata__": {"step": 42, "batch_size": 64}}
```

`RolloutLogger.log(records, metadata=None, filter_fn=None)` takes a
`list[dict]` — no schema enforced. Metadata is stored under `__metadata__`
in each record to avoid key collisions.

## Output and File Layout

### Output folder

```
{dump_folder}/
├── system_logs/                       ← record_span + record_event
│   ├── trainer_rank_0_system.jsonl
│   ├── trainer_rank_1_system.jsonl
│   └── ...
├── experiment_logs/                   ← record_metric
│   ├── trainer_rank_0_experiment.jsonl
│   └── ...
├── rollouts/                          ← RolloutLogger (RL only)
│   └── rollouts.jsonl
├── analysis/
│   └── system_metrics_gantt.json      ← analysis.py:generate_gantt_trace output
└── tb/                                ← TensorBoard logs (when enabled)
>>>>>>> Stashed changes
```

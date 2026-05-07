# TorchTitan RL Metrics

Actors and the controller emit `Metric(key, reduction)` entries at the callsite; the controller aggregates per step and forwards the resulting `dict[str, float]` to every backend.

## Usage

```python
import time

from torchtitan.tools.logging import logger
from torchtitan.experiments.rl.observability import metrics as m

self.metric_logger = m.MetricLogger.build(
    config.metrics,
    log_dir=config.dump_folder,
    config_dict=config.to_dict(),
)

def calculate_rewards(self, myinputs)

    rewards = []
    metrics = []
    for myinput in myinputs:
        ...
        rewards.append(myreward)

    # add reward metrics using `.from_list`
    metrics.append(m.Metric("rollout/reward", m.Mean.from_list(rewards)))
    metrics.append(m.Metric("rollout/reward", m.Max.from_list(rewards)))

    return rewards, metrics

for step in range(num_steps):
    metrics = []

    ...

    rewards, metrics_rewards = calculate_rewards(myinputs)


    # add Mean episode length. Prefer `.from_list` single entry over adding N entries.
    for episode in episodes:
        metrics.append(m.Metric("rollout/episode_length", m.Mean(episode.length)))

    # train metrics are already aggregated in the trainer. Use `m.NoReduce` to pass them through.
    already_reduced_metrics: dict[str, float] = TrainerActor.step.call(batch)
    metrics += [m.Metric(k, m.NoReduce(v)) for k, v in already_reduced_metrics.items()]

    start = time.perf_counter()
    self.metric_logger.log(step, metrics)
    logger.info("metric logging took %.2f ms", (time.perf_counter() - start) * 1000)

self.metric_logger.close()
```

## API at a glance

```
Metric(key, reduction)
    │
    reduction is one of:
    │
    ├── Mean(value, count=1.0)   ──►  key/mean
    ├── Max(value)               ──►  key/max
    ├── Min(value)               ──►  key/min
    ├── Std(value)               ──►  key/std
    ├── Stats(value)             ──►  key/_{max,mean,min,std,sum}
    └── NoReduce(value)          ──►  key

Each reduction also has `Reduction.from_list(values)` for many observations.

aggregate_metrics(records)  ──►  dict[str, float]
    groups by (key, reduction type), filters NaN entries,
    raises on duplicate output keys.

MetricLogger(backends).log(
    step, records,
    *,
    console_allow_list=None,   # None = all, [] = silent, list[regex] = filtered
    console_prefix="",         # e.g. "validate " for the validation line
)
    │
    ├── log_to_console(...)                    ── stdout (in-process)
    ├── WandbMetricLogger(...)                 ── wandb.log(metrics, step)
    └── (your custom MetricBackend subclass)
```

## Custom backends

Any class implementing `log(metrics, step)` and `close()` is a backend. Inherit `MetricBackend` for clarity:

```python
import json

from torchtitan.experiments.rl.observability.metrics import MetricBackend

class JsonlMetricBackend(MetricBackend):
    def __init__(self, path):
        self._fh = open(path, "a")

    def log(self, metrics, step):
        self._fh.write(json.dumps({"step": step, **metrics}) + "\n")
        self._fh.flush()

    def close(self):
        self._fh.close()
```

## Non-goals

- Multi-axis metrics (`step_axis`/`step_value`).
- Per-rollout structured logging (a `RolloutLogger` is a planned follow-up).
- Cross-actor distributed reduction. `NoReduce` assumes the trainer already all-reduced upstream.
- TensorBoard integration in the RL path.

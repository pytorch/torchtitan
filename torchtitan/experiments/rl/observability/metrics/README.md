# TorchTitan RL Metrics

Actors and the controller emit typed `Metric(key, value)` records. Reduction is done lazily, at the MetricLogger.log call. The logger reduces those records once per step and sends the flat `dict[str, float]` to console and backend loggers.

```text
loss / trainer / controller
        |
        v
list[Metric(key, MetricValue)]
        |
        v
MetricLogger.log(step, metrics) -> MetricLogger._aggregate_metrics(...)
        |
        v
console + W&B / TensorBoard backends
```

## Usage

```python
from torchtitan.experiments.rl.observability import metrics as m

metric_logger = config.metrics.build(
    log_dir=config.dump_folder,
    config_dict=config.to_dict(),
)

metrics = []

# Same key with two value types.
# Their final names are `key/max` and `key/mean`.
for response_length in [12, 18]:
    metrics += [
        m.Metric("rollout/response_length", m.Max(response_length)),
        m.Metric("rollout/response_length", m.Mean(response_length))
    ]

# `from_list` is preferred when observations are already in a list,
# e.g. the example above.
metrics.append(m.Metric("reward", m.SummaryStats.from_list([0.0, 0.5, 1.0])))

# Already-reduced scalars from an actor pass through with NoReduce.
metrics.append(m.Metric("loss/total", m.NoReduce(0.42)))

# Log the metrics at step 7.
metric_logger.log(step=7, metrics=metrics, is_validation=False)
```

## Metric values

| Constructor                                                    | Output keys                                          |
| -------------------------------------------------------------- | ---------------------------------------------------- |
| `m.Mean(value)` / `m.Mean.from_list(values)`                   | `key/mean`                                           |
| `m.Max(value)` / `m.Max.from_list(values)`                     | `key/max`                                            |
| `m.Min(value)` / `m.Min.from_list(values)`                     | `key/min`                                            |
| `m.Sum(value)` / `m.Sum.from_list(values)`                     | `key/sum`                                            |
| `m.Std(value)` / `m.Std.from_list(values)`                     | `key/std`                                            |
| `m.SummaryStats(value)` / `m.SummaryStats.from_list(values)`   | `key/_max`, `key/_mean`, `key/_min`, `key/_std`, `key/_sum` |
| `m.NoReduce(value)`                                            | `key`                                                |

`SummaryStats` uses leading underscores so its outputs do not collide with
standalone `Mean`/`Max`/`Min`/`Std`/`Sum` records under the same key.

## Console output

`MetricLogger.log(...)` reads the configured allow list:

```python
m.MetricsConfig(
    console_log_keys_train=["loss", "grad_norm"],
    console_log_keys_validation=["loss"],
)

metric_logger.log(step=step, metrics=train_metrics)
# prints "Train | Step:  N | loss:  0.42 | grad_norm:  0.01

metric_logger.log(step=step, metrics=val_metrics, is_validation=True)
# prints "Validation | Step:  N | loss:  0.42"
```

## Backends

`MetricsConfig.enable_wandb` and `MetricsConfig.enable_tensorboard` add
the corresponding backend at build time. Both require `log_dir` to be
passed to `MetricsConfig.build(...)`.

For ad-hoc backends (a custom JSONL writer, a metrics-tagging proxy,
etc.), pass them through the constructor:

```python
import json

class JsonlBackend(m.MetricBackend):
    def __init__(self, path):
        self._log_file = open(path, "a")

    def log(self, metrics, step):
        self._log_file.write(json.dumps({"step": step, **metrics}) + "\n")
        self._log_file.flush()

    def close(self):
        self._log_file.close()

metric_logger = m.MetricLogger(
    m.MetricsConfig(),
    backends=[JsonlBackend("metrics.jsonl")],
)
```

# TorchTitan RL Metrics

Actors and the controller emit typed `Metric(key, value)` records. Reduction
is done lazily, at the MetricLogger.log call. The logger reduces those
records once per step and sends the flat `dict[str, float]` to console and
backend loggers.

```text
loss / trainer / controller
        |
        v
metrics = list[Metric(key, MetricValue)]
        |
        v
MetricLogger.log(step, metrics) -> MetricLogger._aggregate_metrics(metrics)
        |
        v
console + W&B / TensorBoard backends
```

## Usage

```python
from torchtitan.experiments.rl.observability import metrics as m

# The allow list is used to only log to console some keys, instead of all of them
config.metrics.console_log_keys_train = ["loss/mean", "rollout/prompt_length/mean"]
metric_logger = config.metrics.build(
    log_dir=config.dump_folder,
    job_config=config.to_dict(),
)

metrics = []

# Add to a list multiple records with the same key
# They will be later reduced to: {"rollout/prompt_length": 200}
for prompt_length in [100,300]:
    metrics.append(
        m.Metric("rollout/prompt_length", m.Mean(prompt_length)),
    )

# `from_list` is preferred when observations are already in a list
# since it created a single record, instead of N.
metrics.append(m.Metric("reward", m.SummaryStats.from_list([0.0, 1.0])))

# You can define multiples key with different value types.
# Their final names are `key/max` and `key/mean`.
response_lengths = [1000, 2000]
metrics += [
        m.Metric("rollout/response_length", m.Max.from_list(response_lengths)),
        m.Metric("rollout/response_length", m.Mean.from_list(response_lengths))
    ]

# Your train already returns reduced metrics
# You can use NoReduce as a no-op metric type.
metrics.append(m.Metric("loss/mean", m.NoReduce(0.42)))

# Log the metrics at step 7.
metric_logger.log(step=7, metrics=metrics, is_validation=False)
```

On the log call, it will aggregate all metrics, and produce the dictionary:
```python
{
    "rollout/prompt_length/mean": 200.0,        # mean of [100, 300]
    "reward/_max": 1.0,                         # SummaryStats expansion
    "reward/_mean": 0.5,
    "reward/_min": 0.0,
    "reward/_std": 0.5,
    "reward/_sum": 1.0,
    "rollout/response_length/max": 2000.0,      # max of [1000, 2000]
    "rollout/response_length/mean": 1500.0,     # mean of [1000, 2000]
    "loss/mean": 0.42,                          # NoReduce pass-through
}
```

The full dictionary is forwarded to every backend (W&B, TensorBoard). Console
output is filtered by the configured allow list. With the allow list above
(`["rollout/prompt_length/mean", "loss/total"]`), the printed line is:
```text
----------
Train | Step:  7  loss/mean: 0.42   rollout/prompt_length/mean: 200.0
```

## Metric values

```text
Mean
  Metric("reward", Mean.from_list([1.0, 3.0]))
  -> reward/mean = 2.0

Max
  Metric("response_length", Max.from_list([1.0, 3.0]))
  -> response_length/max = 3.0

Min
  Metric("response_length", Min.from_list([1.0, 3.0]))
  -> response_length/min = 1.0

Sum
  Metric("tokens", Sum.from_list([1.0, 3.0]))
  -> tokens/sum = 4.0

Std
  Metric("reward", Std.from_list([1.0, 3.0]))
  -> reward/std = 1.0

SummaryStats
  Metric("reward", SummaryStats.from_list([1.0, 3.0]))
  -> reward/_max, reward/_mean, reward/_min, reward/_std, reward/_sum

NoReduce
  Metric("loss/total", NoReduce(0.42))
  -> loss/total = 0.42
```

`SummaryStats` uses leading-underscore output names so its sub-keys
don't collide with bare `Mean`/`Max`/`Min`/`Std`/`Sum` records under
the same key.

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

`MetricsConfig.enable_wandb` and `MetricsConfig.enable_tensorboard` add the
corresponding backend at build time. Both require `log_dir` to be passed to
`MetricsConfig.build(...)`. `WANDB_PROJECT` defaults to `titan_rl`; set the
env var or `wandb_project=` on the config to override.

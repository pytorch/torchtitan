# TorchTitan RL Metrics

Actors and the controller emit typed `Metric(key, value)` records. Reduction
is done lazily, at the MetricsProcessor.log call. The logger reduces those
records once per step and sends the flat `dict[str, float]` to console and
backend loggers.

```text
loss / trainer / controller
        |
        v
metrics = list[Metric(key, MetricValue)]
        |
        v
MetricsProcessor.log(step, metrics) -> MetricsProcessor._aggregate_metrics(metrics)
        |
        v
console + W&B / TensorBoard backends
```

## Usage

```python
from torchtitan.experiments.rl.observability import metrics as m

# The allow list is used to only log to console some keys, instead of all of them
config.metrics.console_log_keys_train = ["loss/mean", "rollout/prompt_length/mean"]
metrics_processor = config.metrics.build(
    log_dir=config.dump_folder,
    job_config=config.to_dict(),
)

metrics = []

# Add to a list multiple records with the same key
# They will be later reduced to: {"rollout/prompt_length/mean": 200}
for prompt_length in [100,300]:
    metrics.append(
        m.Metric("rollout/prompt_length", m.Mean(prompt_length)),
    )

# `from_list` is preferred when observations are already in a list
# since it creates a single record, instead of N.
metrics.append(m.Metric("reward", m.SummaryStats.from_list([0.0, 1.0])))

# You can define multiples key with different value types.
# Their final names have a different suffix `key/max` and `key/mean`.
response_lengths = [1000, 2000]
metrics += [
        m.Metric("rollout/response_length", m.Max.from_list(response_lengths)),
        m.Metric("rollout/response_length", m.Mean.from_list(response_lengths))
    ]

# Your train already returns reduced metrics
# You can use NoReduce as a no-op metric type.
metrics.append(m.Metric("loss/mean", m.NoReduce(0.42)))

# Log the metrics at step 7.
metrics_processor.log(step=7, metrics=metrics, is_validation=False)
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
(`["rollout/prompt_length/mean", "loss/mean"]`), the printed line is:
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
  Metric("reward", Max.from_list([1.0, 3.0]))
  -> reward/max = 3.0

Min
  Metric("reward", Min.from_list([1.0, 3.0]))
  -> reward/min = 1.0

Sum
  Metric("reward", Sum.from_list([1.0, 3.0]))
  -> reward/sum = 4.0

Std
  Metric("reward", Std.from_list([1.0, 3.0]))
  -> reward/std = 1.0

SummaryStats
  Metric("reward", SummaryStats.from_list([1.0, 3.0]))
  -> reward/_max, reward/_mean, reward/_min, reward/_std, reward/_sum

NoReduce
  Metric("loss/mean", NoReduce(0.42))
  -> loss/mean = 0.42
```

`SummaryStats` uses leading-underscore output names so its sub-keys
don't collide with bare `Mean`/`Max`/`Min`/`Std`/`Sum` records under
the same key.

## Console output

`MetricsProcessor.log(...)` reads the configured allow list:

```python
m.MetricsProcessor.Config(
    console_log_keys_train=["loss", "grad_norm"],
    console_log_keys_validation=["loss"],
)

metrics_processor.log(step=step, metrics=train_metrics)
# prints "Train | Step:  N | loss:  0.42 | grad_norm:  0.01

metrics_processor.log(step=step, metrics=val_metrics, is_validation=True)
# prints "Validation | Step:  N | loss:  0.42"
```

## Backends

`MetricsProcessor.Config.enable_wandb` and `MetricsProcessor.Config.enable_tensorboard`
add the corresponding backend at build time. Both require `log_dir` to be
passed to `MetricsProcessor.Config.build(...)`. `WANDB_PROJECT` defaults to
`titan_rl`; set the env var or `wandb_project=` on the config to override.

# TorchTitan RL Metrics

Actors and the controller emit typed `Metric(key, value)` records. Reduction
is done lazily, at the MetricsProcessor.log call. The logger reduces those
records once per step and sends the flat `dict[str, float]` to console and
backend loggers.

vLLM engine-level metrics use a separate OpenTelemetry pipeline described in
[vLLM engine metrics](#vllm-engine-metrics).

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

## vLLM engine metrics

`VllmOtelStatLogger` exports engine-level rollout metrics through
[OpenTelemetry](https://opentelemetry.io/). The logger is registered by default,
but stays inactive unless `OTEL_METRICS_EXPORTER` is explicitly set to `jsonl`
or `otlp`. Only TP rank 0 in each DP replica creates the logger because all TP
ranks see the same engine-aggregate statistics.

The exported metrics are grouped by OpenTelemetry instrument type:

- Counters: `vllm.generation_tokens`, `vllm.prompt_tokens`,
  `vllm.cached_prompt_tokens`, `vllm.preempted_requests`,
  `vllm.finished_requests`, `vllm.prefix_cache_queries`, and
  `vllm.prefix_cache_hits`.
- Histograms: `vllm.time_to_first_token`, `vllm.inter_token_latency`,
  `vllm.decode_time`, `vllm.queue_time`, and `vllm.e2e_latency`.
- Gauges: `vllm.kv_cache_usage`, `vllm.num_running_requests`, and
  `vllm.num_waiting_requests`.

Counters are cumulative; derive throughput and prefix-cache hit rate from their
rates in the backend. Histogram values use milliseconds. Gauges report the
latest scheduler state observed by the logger.

#### Prefix-cache hit rate

The logger exports prefix-cache hits and queries as separate monotonic counters.
Compute the hit rate over the selected query window from their rates:

```text
prefix_cache_hit_rate =
    rate(vllm.prefix_cache_hits) / rate(vllm.prefix_cache_queries)
```

Multiply the result by 100 to display a percentage. When querying multiple
generators, sum each counter's rates before dividing:

```text
prefix_cache_hit_rate_percent =
    100 * sum(rate(vllm.prefix_cache_hits))
        / sum(rate(vllm.prefix_cache_queries))
```

Return no value when the query rate is zero. Using rates instead of the raw
cumulative counter values limits the result to the selected time window and
handles counter resets.

### Choose the right exporter option

Both exporters use the same OpenTelemetry instruments and metric names; only
the destination changes:

- Use `jsonl` for local development, debugging, and short or small-scale runs
  that do not have an OpenTelemetry collector.
- Use `otlp` for distributed or production runs that need centralized
  aggregation, retention, dashboards, or alerts.

#### Write JSONL locally

After completing the [DAPO Math setup](../../examples/dapo_math/README.md#setup),
use `jsonl` for local inspection without a collector:

```bash
OTEL_METRICS_EXPORTER=jsonl \
VLLM_LOG_STATS_INTERVAL=10 \
python -m torchtitan.experiments.rl.train \
  --module dapo_math \
  --config rl_dapo_qwen3_4b_math_8k \
  --metrics.no-enable-wandb
```

Each TP-rank-0 generator writes compact JSON records to
`{dump_folder}/vllm_metrics/{generator_name}.rank{rank}.jsonl`. This mode
requires a configured output directory.

#### Export with OTLP

Use `otlp` to send metrics directly to an OpenTelemetry collector or compatible
backend:

```bash
OTEL_METRICS_EXPORTER=otlp \
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
VLLM_LOG_STATS_INTERVAL=10 \
python -m torchtitan.experiments.rl.train \
  --module dapo_math \
  --config rl_dapo_qwen3_4b_math_8k \
  --metrics.no-enable-wandb
```

The logger uses the OTLP HTTP/protobuf exporter. If a protocol environment
variable is set, `OTEL_EXPORTER_OTLP_METRICS_PROTOCOL` or
`OTEL_EXPORTER_OTLP_PROTOCOL` must be `http/protobuf`. Set either
`OTEL_EXPORTER_OTLP_ENDPOINT` or `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT`; the
metrics-specific setting takes precedence.

The OpenTelemetry SDK exports on a background thread. The export cadence is
controlled by `VLLM_LOG_STATS_INTERVAL` in seconds (default: 10). Standard
OpenTelemetry environment variables remain available, including:

- `OTEL_METRIC_EXPORT_TIMEOUT`: per-export time budget in milliseconds.
- `OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE`: OTLP aggregation
  temporality (`CUMULATIVE`, `DELTA`, or `LOWMEMORY`; default: `CUMULATIVE`).
- `OTEL_SERVICE_NAME`: service name attached to exported metrics.
- `OTEL_RESOURCE_ATTRIBUTES`: additional resource attributes.
- `OTEL_SDK_DISABLED=true`: disable OpenTelemetry entirely.

The logger also attaches `model_name`, `hostname`, `rank`, `local_rank`,
`world_size`, `dp_rank`, `tp_rank`, and `generator_name` resource attributes.

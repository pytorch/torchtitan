We support automatically collecting metrics such as
1. High level system metrics such as MFU, average loss, max loss and words per second along with some
2. Memory metrics to measure max VRAM consumption and the number of OOMs
3. Timing metrics to measure data loading bottlenecks

Those metrics can then be visualized in a TensorBoard, W&B, or MLflow dashboard.

## TensorBoard

To visualize TensorBoard metrics of models trained on a remote server via a local web browser:

1. Add `TensorBoardLogger.Config()` to `metrics.loggers` in your training recipe.

2. Set up SSH tunneling, by running the following from local CLI
```
ssh -L 6006:127.0.0.1:6006 [username]@[hostname]
```

3. Inside the SSH tunnel that logged into the remote server, go to the torchtitan repo, and start the TensorBoard backend
```
tensorboard --logdir=./outputs/tb
```

4. In the local web browser, go to the URL it provides OR to http://localhost:6006/.

## Weights and Biases

Weights and Biases will automatically send metrics to a remote server if you login with `wandb login`

Add `WandBLogger.Config()` to `metrics.loggers` in your recipe. Set
`WANDB_PROJECT` to select the W&B project; the default is `torchtitan`.

Each log call is its own W&B row, with the training step in `trainer/step`, the
x-axis of the generated charts. W&B's `_step` (`Step` in the UI) counts rows, not
training steps: plot saved panels against `trainer/step`, and for
`WANDB_RESUME_FROM` / `WANDB_FORK_FROM` (`<run>?_step=N`) look up the `_step` of
the row with the `trainer/step` you want.

For an example you can inspect the Llama 3 [recipe](../../torchtitan_recipes/models/llama3.py)

## MLflow

Install the optional tracking client with `pip install mlflow-skinny`
(or the full `mlflow` package).

Add `MLFlowLogger.Config()` to `metrics.loggers` in your recipe and set `MLFLOW_TRACKING_URI`
and `MLFLOW_EXPERIMENT_NAME` to select the tracking server and experiment.

## Configure metric exporters

```python
from torchtitan.observability.metrics import (
    MetricsProcessor,
    TensorBoardLogger,
    WandBLogger,
)

metrics = MetricsProcessor.Config(
    loggers=[WandBLogger.Config(), TensorBoardLogger.Config()],
)
```

Loggers are built and called in list order. An empty list disables metric
export while preserving console reporting. The supervised processor writes
files under `dump_folder/save_tb_folder`; `save_for_all_ranks=True` exports
from every rank, except MLflow always uses the rank that owns the loss.
Logger construction errors propagate to the caller.

Custom exporters subclass `BaseLogger`, define a nested
`Config(BaseLogger.Config)`, and implement `log(metrics, step)` and `close()`.
Their constructors receive their own config and the runtime arguments
`log_dir`, `config_dict`, and `tag`. Set `Config.metrics_rank_only = True`
when an exporter must only run on the metrics rank.

The RL metrics processor accepts the same `loggers` list, builds it on the
controller, and retains its own aggregation and console filters. RL uses the
same W&B project default without changing environment variables.

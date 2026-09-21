We support automatically collecting metrics such as
1. High level system metrics such as MFU, average loss, max loss and words per second along with some
2. Memory metrics to measure max VRAM consumption and the number of OOMs
3. Timing metrics to measure data loading bottlenecks

Those metrics can then be visualized in either a TensorBoard or WandDB dashboard

## TensorBoard

To visualize TensorBoard metrics of models trained on a remote server via a local web browser:

1. Make sure `metrics.enable_tensorboard` option is set to true in model training (either from a config_registry function or from CLI).

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

So all you need to do is make sure that `metrics.enable_wandb` is enabled

For an example you can inspect the Llama 3 [config_registry.py](../torchtitan/models/llama3/config_registry.py)

If both W&B and TensorBoard are enabled, both loggers run.

## FLOPs, throughput, and MFU

TorchTitan estimates model-wide logical training FLOPs from each raw input
batch and accumulates them over the reporting interval. At the logging
boundary, it averages FLOPs across DP x CP ranks; TorchFT also averages across
the active fault-tolerance group. It then divides the logical FLOP rate by
`CP * TP * PP` to report per-device TFLOPS and MFU.

Each trainer retains its existing logical-token throughput convention.
For standard Trainer and TorchFT, each yielded GA/PP microbatch contributes
`training.num_tokens_per_microbatch_per_dp_rank` to throughput and advances
`TrainingEngine.ntokens_seen` using that configured logical count. RL
throughput instead uses its existing local `labels.numel()` count.
`num_valid_tokens` remains dedicated to loss normalization and loss metrics,
so padding or ignored labels can affect loss normalization without changing
logical token throughput.

For the estimator callback contract and model-authoring guidance, see
[Batch FLOP estimation](../torchtitan/models/README.md#batch-flop-estimation).

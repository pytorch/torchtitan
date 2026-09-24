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

Models estimate the logical training FLOPs for each raw input batch. Trainers sum
those values between log events. At each log event they:

1. Average the interval FLOPs across DP x CP ranks. TorchFT also averages over
   its active fault-tolerance group.
2. Divide by the interval duration to get the logical model FLOP rate.
3. Divide by `CP x TP x PP` to get the average rate per device.

In formula form:

```text
per_device_flops_per_second =
    mean_interval_flops / interval_seconds / (CP * TP * PP)
TFLOPS = per_device_flops_per_second / 1e12
MFU = per_device_flops_per_second / device_peak_flops
```

DP is averaged rather than divided again because each DP rank processes different
data. CP ranks see the same raw batch before sharding, so averaging removes the
duplicate estimates; dividing by CP then assigns the logical work across the CP
devices. TP and PP are divided for the same model-work-to-device conversion.

Token throughput is tracked separately from FLOPs:

- Standard Trainer and TorchFT use
  `training.num_tokens_per_microbatch_per_dp_rank` for each yielded GA/PP
  microbatch.
- RL uses the existing local `labels.numel()` count.
- `num_valid_tokens` is only for loss normalization and loss metrics. Padding or
  ignored labels therefore do not change logical token throughput.

For the model estimator interface, ownership rules, and examples, see
[Batch FLOP estimation](../torchtitan/models/README.md#batch-flop-estimation).

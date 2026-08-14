# Tensor logging

Tensor logging records compact statistics about model tensors without retaining the tensors themselves.

```text
tensor [0, 1, -2, 3]
       |
       +-> count=4, zero_count=1, abs_sum=6, square_sum=14, abs_max=3
       |
       +-> fixed device row -> one packed drain -> TensorBoard/W&B
```

Use it to find exploding activations and dead gradients across a distributed training job.

## Mental model

```text
1. Register metric names while modules are constructed.
2. `init()` freezes one globally ordered row for every name.
3. `log_stats()` writes sufficient statistics into those fixed rows.
4. On a selected step, `collect()` reduces the packed rows and derives scalars.
5. TorchTitan publishes the scalars through its existing loggers.
```

The ordinary path does not inspect TP, CP, DP, EP, or PP meshes. It reports statistics over the tensor occurrences emitted by participating ranks. A metric that needs a particular semantic population reconstructs that value first, then uses the same `log_stats()` call.

## Enable it

Enable tensor logging and a metrics sink on an existing recipe:

```bash
NGPU=8 MODULE=qwen3 CONFIG=qwen3_debugmodel ./run_train.sh \
  --metrics.enable-wandb \
  --metrics.tensor-logging.enabled \
  --metrics.tensor-logging.freq 5
```

See the [observability README](../README.md) for TensorBoard and Weights & Biases setup.

Tensor work runs only when both its requested cadence and ordinary `metrics.log_freq` select the step. For tensor cadence 15 and scalar cadence 10, tensor metrics publish at steps 30, 60, and so on. Tensor logging does not inherit ordinary metrics' special step-1 publication.

The default tensor cadence is 5. The default publication filter keeps `numel`, `nonfinite_count`, `abs_mean`, `square_mean`, and `abs_max`.

## Record a tensor

Register names in module construction, then record their current values at the producer:

```python
from torchtitan.observability import tensor_logging


class Attention(nn.Module):
    def __init__(self, ...):
        super().__init__()
        tensor_logging.register(self, ["scores"])

    def forward(self, x):
        scores = self.compute_scores(x)
        tensor_logging.log_stats(self, scores=scores)
        return self.apply_scores(scores)
```

`register()` declares the public name. The trainer calls `init()` after model parallelization and optimizer construction, assigning every name a fixed buffer row. Emitting an unregistered name is an error.

### Record forward and backward values

```python
class Attention(nn.Module):
    def __init__(self, ...):
        super().__init__()
        tensor_logging.register_fwd_bwd(self, ["xq"])

    def forward(self, x):
        xq = self.query(x)
        tensor_logging.log_fwd_bwd_stats(self, xq=xq)
        return self.attend(xq)
```

This publishes:

```text
<module>.xq.x.<statistic>   forward tensor
<module>.xq.dx.<statistic>  gradient arriving at xq during backward
```

The call returns `None`; continue using the original `xq` tensor. The backward recorder is attached directly to it.

### Boundaries and internal tensors use the same API

```python
# Residual-branch boundary
tensor_logging.log_fwd_bwd_stats(
    transformer_block,
    attn_stream=residual,
    attn_out=attention_output,
)

# Internal attention projection
tensor_logging.log_fwd_bwd_stats(attention, xq=xq)
```

“Boundary” describes where the tensor sits in the model; it is not a different logging operation. TorchTitan does not infer a reduction mesh from either call.

## From rows to metrics

Each observation contributes mergeable sufficient statistics:

```text
counts = [numel, nonfinite_count, zero_count, observation_count]
sums   = [abs_sum, square_sum, fourth_moment_sum]
maxima = [abs_max]
```

For finite values, these rows derive `zero_frac`, `abs_mean`, `square_mean`, RMS, `abs_max`, and excess kurtosis. For `[0, 1, -2, 3]` recorded once:

```text
numel=4
zero_frac=1/4
abs_mean=(0+1+2+3)/4=1.5
square_mean=(0+1+4+9)/4=3.5
abs_max=3
kurtosis=(0+1+16+81)/4/3.5^2 - 3 = -1
```

Adding another ordinary metric does not add another collective. All rows share three packed buffers on a selected step:

```text
int64 counts   --SUM--> exact counts above float32's integer range
float32 sums   --SUM--> moments and means
float32 maxima --MAX--> absolute maxima
```

Every rank allocates the same row order. Under PP, ranks that do not own a metric contribute identity values for its row.

Ordinary rows count physical observations. A replicated tensor contributes once per rank holding it, so absolute counts include replica multiplicity. Ratios such as `abs_mean`, `zero_frac`, and kurtosis are unchanged by uniform replication.

## Publication filter

`metrics.tensor_logging.publish_filter_regex` is an allowlist over dotted metric names:

```text
layers.0.attention.xq.x.abs_max
```

The filter controls which tensor rows reach TensorBoard/W&B. It does not avoid the GPU statistic calculation, so a narrow filter reduces sink volume but not all collection work.

## Execution modes

- Full and selective activation checkpointing preserve the original forward mutation so recomputation does not double-count statistics.
- Regional full-graph `torch.compile` uses a device-resident enabled flag, allowing selected and unselected steps to reuse one graph.
- In-process Graph Trainer tracing and CUDA-graph replay use the live model-owned rows. Separately produced or loaded precompiled artifacts are unsupported because registered owners and live buffers are not portable across artifacts.
- Pipeline model parts can share global prefixes so names remain model paths such as `layers.7.attention.xq.x`, independent of rank-local part indices.
- CUDA uses a lazily imported Triton accumulator; CPU uses the eager reference path. ROCm source compatibility is not a hardware-validation claim.

## Current limitations

- GraphPP forward-statistic support is partial because copied FX graphs do not yet preserve every forward buffer mutation.
- The publication filter does not skip GPU collection.
- Publication is synchronous with the training step.
- Separately precompiled Graph Trainer artifacts are unsupported.
- Optional visualizations, asynchronous publication, and additional built-in metrics are follow-up work.

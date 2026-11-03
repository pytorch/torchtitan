# Tensor logger

The tensor logger publishes statistics of chosen activations and their gradients,
for example the RMS of every layer's output, every `log_freq` steps. It is off by
default. When it is off, each logging call returns after a cheap check.

```text
tensor_logger/layers.3.attention.xq.fwd.rms        0.92   forward value of xq in layer 3
tensor_logger/layers.3.attention.xq.bwd.rms        1e-05  gradient arriving at xq
tensor_logger/layers.3.feed_forward.output.fwd.abs_max  7.5
```

## Enable it

Set the field in a recipe. `log_freq` must be a multiple of `metrics.log_freq`.

```python
from torchtitan.observability.tensor_logger import TensorLogger
from torchtitan.trainer import Trainer
from torchtitan_recipes.tests.models.llama3 import llama3_debugmodel


def llama3_debugmodel_tensor_logger() -> Trainer.Config:
    config = llama3_debugmodel()
    config.tensor_logger = TensorLogger.Config(log_freq=10)
    return config
```

Launch it with `MODULE=my_project.recipes CONFIG=llama3_debugmodel_tensor_logger ./run_train.sh`.
The statistics go to the same loggers as the other metrics ([METRICS.md](../METRICS.md)).

## Log a tensor from a module

Register names in `__init__`, then log tensors with those names in `forward`:

```python
from torchtitan.observability import tensor_logger


class Router(nn.Module):
    def __init__(self, ...):
        super().__init__()
        tensor_logger.register_fwd(self, "scores")  # row <path>.scores.fwd
        tensor_logger.register_fwd_bwd(self, "hidden")  # rows <path>.hidden.fwd and .bwd

    def forward(self, x):
        hidden = self.proj(x)
        scores = self.gate(hidden)
        tensor_logger.log_fwd(self, scores=scores)
        tensor_logger.log_fwd_bwd(self, hidden=hidden)  # adds a gradient hook
        ...
```

`<path>` is the module's name in the full model, such as `layers.3.feed_forward.router`.
Pipeline stages keep these names, so a run with PP publishes the same keys as one without.

Rules for callsites:

- Log full values or a shard of them. In spmd_types terms, `log_fwd` takes `R`, `I`, `V`
  or `S` tensors. A partial sum (`P`, e.g. a row-parallel output before its all-reduce)
  gives wrong rows.
- `log_fwd_bwd` also logs the gradient, so it takes `I`, `V` or `S`, not `R`: the gradient
  of an `R` tensor is a partial sum on each rank.
- Pass tensors made in this forward to `log_fwd_bwd`. A tensor that lives across steps would
  collect one more gradient hook per call; a parameter (a leaf that requires grad) raises.
- Keep callsites outside `local_compile` regions: the logging calls are tested in eager
  code and in accelerator graphs only.

## Published fields

Each row publishes `tensor_logger/<path>.<name>.<fwd|bwd>.<field>`:

| field | meaning |
|---|---|
| `numel` | elements observed, summed over calls and ranks |
| `observation_count` | number of (call, rank) pairs that logged this row |
| `nonfinite_count` | `nan` and `inf` elements |
| `zero_frac` | zeros / finite elements |
| `abs_mean` | mean of abs over finite elements |
| `rms` | sqrt(mean of squares) over finite elements |
| `kurtosis` | mean(x^4) / mean(x^2)^2 - 3: 0 for a zero-mean Gaussian, -2 when all values have the same abs, large with rare spikes |
| `abs_max` | largest finite abs |

A row nobody observed on a recorded step publishes nothing.

## How rows combine over ranks and calls

Each rank adds every call into its local row. On a recorded step, `collect()` runs one
SUM all-reduce over all ranks for the sums and counts and one MAX for `abs_max`.
Every rank holds the same sorted list of rows. A rank that never logs a row, such as a
pipeline stage without that layer, keeps 0 for the sums and `-inf` for the maximum:

```text
rows:     [layers.0.output, layers.1.output, layers.2.output, layers.3.output]
PP rank 0 (layers 2, 3): [0,  0,  v1, v2]
PP rank 1 (layers 0, 1): [v3, v4, 0,  0 ]
SUM                    : [v3, v4, v1, v2]
```

So a row describes all elements logged under its name in the step. Sharded tensors
(TP, CP, FSDP data) add up to the full tensor. A tensor that several ranks hold in full
(e.g. a layer output under TP without sequence parallel) is counted once per rank:
`numel` and `observation_count` grow with the number of copies, while `zero_frac`,
`abs_mean`, `rms`, `kurtosis` and `abs_max` stay exact.

Gradients are the ones each rank's backward sees. With data parallel, the loss averages
over all DP ranks, so `.bwd` values shrink as the DP degree grows.

## Cadence and cost

- Steps that are multiples of `log_freq` record.
- With accelerator graphs on (the default), the logging calls run on every step, because
  a graph replays exactly what ran when it was captured. A device flag makes the kernel
  exit at once on steps that do not record. A tensor with gaps in memory is still copied
  first on those steps. With graphs off, steps that do not record launch nothing.
- A recorded step reads each logged tensor and gradient once, then runs two all-reduces
  and two small device-to-host copies, after the forward and backward.
- Memory: `rows * 8 * 4` bytes on each GPU (about 2.3 KB for the 73 rows of the llama3 debug model).
- Sums are float32. `numel` is exact up to 2^24 elements per row and step, and rounds
  to about 7 significant digits above that. `kurtosis` is wrong for values below about
  1e-9, because their fourth powers underflow float32.
- The GPU kernel adds float sums with atomics, so two otherwise identical runs can differ
  in the last bits of `abs_mean`, `rms` and `kurtosis`.
- The buffers are not saved in checkpoints. Each recorded step starts from zero, so a
  resumed run records normally from the next multiple of `log_freq`.

## Activation remat, PP and validation

- `torch_remat` replays a block's forward during backward. Logging calls skip the replay,
  so each forward counts once with every remat policy.
- PP metadata inference runs real forwards and backwards at the start of the first step and
  after each switch from eval back to train. The logger clears what they recorded.
- Validation records nothing: it runs outside the recorded scope, and `log_fwd_bwd`
  returns under `no_grad`.

## Not supported

Each of these raises `ValueError` at config time:

- PP schedules that split the backward into input- and weight-gradient passes (e.g.
  `ZBVZeroBubble`, `DualPipeV`), and schedule CSV files. A split backward runs the
  gradient hooks of linear outputs twice.
- `sdc_replayer`: the replay reruns forward and backward and would record twice.
- `GraphTrainer`: its traced graphs do not run the logging calls yet.
- `FaultTolerantTrainer`: its training step does not record or collect.

## Covered models

Shared modules log these rows (`fwd` and `bwd`, unless marked):

| module | rows |
|---|---|
| `Decoder.forward` | `tok_embeddings.output` (`fwd` only), `layers.<i>.output` |
| `GQAttention` | `xq`, `xk`, `xv`, `head_out` |
| `DenseFeedForward` | `act_out` |
| `MoE` | `output` (the full output, after any TP reduction) |

A model publishes the rows of the modules it runs: for example, llama3 logs every row except
`MoE`'s, and deepseek_v3 every row except `GQAttention`'s. A model that overrides `Decoder.forward`
or one of these modules' `forward` must make the same logging call (kimi_k3 and qwen3_5
do). Otherwise its registered rows publish nothing.

## Plot a run

```python
import re

import matplotlib.pyplot as plt
import wandb

run = wandb.Api().run("<entity>/<project>/<run id>")
pattern = re.compile(r"tensor_logger/layers\.(\d+)\.output\.fwd\.rms")
keys = sorted((k for k in run.summary.keys() if pattern.fullmatch(k)),
              key=lambda k: int(pattern.fullmatch(k)[1]))
history = run.history(keys=keys, pandas=True)
plt.imshow(history[keys].T, aspect="auto")
plt.xlabel("recorded step")
plt.ylabel("layer")
plt.colorbar(label="rms")
plt.savefig("layer_output_rms.png")
```

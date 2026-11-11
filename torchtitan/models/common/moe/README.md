# Choosing an expert-parallel backend

TorchTitan supports two ways to execute routed experts: a token dispatcher
followed by grouped expert computation, or Dist-MoE, which owns dispatch,
expert computation, activation storage, and combine together. The router and
checkpoint parameter names stay in TorchTitan.

## Selection

| Workload | Backend | Constraints |
| --- | --- | --- |
| Correctness reference or a new accelerator | `AllToAllTokenDispatcher` | Uses PyTorch collectives. Variable token counts require a CPU sync; disable accelerator graphs. |
| Quantized grouped GEMMs with the reference dispatcher | `TorchAOTokenDispatcher` | Adds per-expert padding to all-to-all. Also supports EP=1. |
| NVIDIA training or inference with a separate grouped-GEMM implementation | `DeepEPTokenDispatcher` | Requires DeepEP v2. Training uses compact dispatch with a CPU sync; disable training accelerator graphs. Inference under `no_grad` can use `cuda_graph_compatible=True` for the static expand layout. |
| NVIDIA training with graph-captured EP communication, including graph trainer | `HybridEPTokenDispatcher` | Requires the HybridEP branch and a supported NVLink/network topology. Set `non_blocking_capacity_factor` for CPU-free dispatch. |
| Blackwell BF16 or MXFP8 training with integrated communication and compute | Dist-MoE | Requires SM100 or newer and `precision.param_dtype="bfloat16"`. Start with the [Dist-MoE recipes and runtime guide](../dist_moe/README.md). |

Without expert parallelism, `LocalTokenDispatcher` only reorders local tokens.
DeepEP and HybridEP require EP>1. TorchTitan does not expose a separate DeepEP
v1 or direct NCCL EP dispatcher.

HybridEP is developed by NVIDIA with CUDA graph support and has TorchTitan
[graph-trainer](../../../experiments/graph_trainer/README.md) recipes and
integration tests. DeepEP's graph-trainer compatibility has not been validated
here; its current training adapter's CPU sync prevents CUDA graph capture.

Both DeepEP v2 and HybridEP use NVIDIA-specific kernels and require a supported
communication fabric. DeepEP v2 requires Hopper (SM90) or compatible newer GPUs;
HybridEP uses TMA and targets Hopper/Blackwell systems. For other accelerators,
start with the PyTorch all-to-all reference and check upstream hardware and
fabric requirements before choosing an optimized backend.

Choose by hardware, precision, graph requirements, and memory budget, then
measure training step time and peak memory on the actual routing distribution.
Dispatch/combine microbenchmarks alone do not measure overlap with expert GEMMs
or the cost of activation rematerialization. Run at least 10 training steps so
startup does not dominate the comparison.

See [router-score absorption](ROUTER_SCORE_ABSORPTION.md) for diagrams of DeepEP
score placement, biased experts, postprocessing, and checkpoint dependencies.

## Configure a dispatcher

Use `TokenDispatcherTransform` to replace the dispatchers in an existing recipe:

```python
from torchtitan.config.transform import apply_transforms
from torchtitan.config.transform.token_dispatcher import TokenDispatcherTransform
from torchtitan.models.common.moe.token_dispatcher import HybridEPTokenDispatcher
from torchtitan_recipes.models.deepseek_v3 import deepseek_v3_671b

config = deepseek_v3_671b(seq_len=4096)
config = apply_transforms(
    config,
    [
        TokenDispatcherTransform(
            dispatcher=HybridEPTokenDispatcher,
            num_max_tokens_per_rank=(
                config.training.num_tokens_per_microbatch_per_dp_rank
                // (
                    config.parallelism.context_parallel_degree
                    * config.parallelism.tensor_parallel_degree
                )
            ),
            kwargs={"non_blocking_capacity_factor": 1.0},
        ),
    ],
)
```

`num_max_tokens_per_rank` bounds the input tokens one EP rank dispatches per
forward for the lifetime of its communication buffer. It is not the number of
tokens after top-k fan-out. Inference engines must use their own per-rank token
budget. Dist-MoE uses `DistMoeTransform` and its runtime instead of this transform.

For graph-captured training, also provide fixed-shape attention metadata and
enable accelerator graphs as in the existing HybridEP recipes. A CPU-free
dispatcher alone does not make the whole model capturable.

### HybridEP capacity

- `None`: blocking dispatch sizes the permuted output from actual token counts.
- `1.0`: nonblocking dispatch reserves worst-case unpadded token fan-out, at a
  higher memory cost.
- Below `1.0`: a smaller permuted output can drop tokens on overflow. Load
  balancing does not guarantee that arbitrary routing stays within capacity.

The capacity factor sizes the permuted output, not the persistent communication
buffer. Quantized experts also need the appropriate `pad_multiple`. Per-expert
padding can overflow the allocation even at factor `1.0`. Check
`check_hybridep_over_budget()` after the step for capacity below `1.0` or padded
nonblocking dispatch, and reset the flag between measurements; it stays on the
device until explicitly read.

## Shared-expert overlap

A MoE can run its shared experts on a side stream while the routed combine is in
flight. The dispatcher publishes the stream position at the start of `combine`
with an optional `combine_ready_event`; the MoE takes it once and starts the
shared experts after the routed experts finish:

```text
routed experts -> combine starts (event recorded) -> routed combine (main stream)
                              \-> shared experts (side stream) -> wait -> add
```

Enable it on a dispatcher that implements the publisher:

```python
from torchtitan.models.common.moe.token_dispatcher import (
    AllToAllTokenDispatcher,
    CombineReadyEvent,
)

token_dispatcher = AllToAllTokenDispatcher.Config(
    num_experts=8,
    top_k=1,
    combine_ready_event=CombineReadyEvent.Config(),
)
```

Pass this config as `RoutedExperts.Config.token_dispatcher`.

`AllToAllTokenDispatcher`, `DeepEPTokenDispatcher`, `HybridEPTokenDispatcher`,
and `TorchAOTokenDispatcher` publish the event; the EP=1 local-only
`LocalTokenDispatcher` does not. The MoE must have `shared_experts`; otherwise
the config raises `ValueError` because the event would have no consumer. The
overlap requires eager execution: `torch.compile` raises at the first forward,
accelerator graph capture is not supported, and `DeepEPTokenDispatcher` rejects
`combine_ready_event` together with `cuda_graph_compatible=True`. Without the
option, the MoE keeps the synchronous path.

When replacing dispatchers with `TokenDispatcherTransform`, pass
`combine_ready_event` through its `kwargs`; the transform fills only structural
fields from the existing config.

## Activation rematerialization

The default remat policy saves dispatcher communication and recomputes the
routed-expert `w13` projection. To retain communication under a custom policy:

```python
from torchtitan.distributed.activation_remat import ActivationRematConfig

config.activation_remat = ActivationRematConfig(
    save_regions=["*token_dispatcher.ep_communication"],
)
```

DeepEP uses `ep_communication` to control dispatch replay and always saves the
complete combine region, including unpermute. Full remat replays dispatch with
deterministic receive ordering, adding a sort. HybridEP saves or replays
dispatch and combine together under `ep_communication`; selecting just its
`.dispatch` or `.combine` suffix does not control the pair. It derives token
positions from prefix scans. The all-to-all dispatcher instead has independent
`dispatch` and `combine` regions that include local reordering and collectives.

Dist-MoE always retains its outer region and owns its internal activation
save/recompute decisions. See [REMAT.md](../../../distributed/REMAT.md) for
policy details.

## Dependencies

DeepEP v2 and HybridEP currently live on incompatible branches of the same
`deep_ep` package. Use separate environments when comparing them. Consult the
upstream installation and fabric requirements:

- [DeepEP v2](https://github.com/deepseek-ai/DeepEP)
- [HybridEP](https://github.com/deepseek-ai/DeepEP/blob/hybrid-ep/docs/README_Hybrid-EP.md)
- [Dist-MoE](../dist_moe/README.md)

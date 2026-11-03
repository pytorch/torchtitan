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
| NVIDIA training with a separate grouped-GEMM implementation | `DeepEPTokenDispatcher` | Requires DeepEP v2. Compact dispatch supports backward but synchronizes with the CPU; disable training accelerator graphs. |
| NVIDIA training with graph-captured EP communication | `HybridEPTokenDispatcher` | Requires the HybridEP branch and a supported NVLink/network topology. Set `non_blocking_capacity_factor` for CPU-free dispatch. |
| Blackwell BF16 or MXFP8 training with integrated communication and compute | Dist-MoE | Requires SM100 or newer and `training.mixed_precision_param="bfloat16"`. Start with the [Dist-MoE recipes and runtime guide](../dist_moe/README.md). |
| Graph-captured inference with DeepEP | `DeepEPTokenDispatcher` | Set `cuda_graph_compatible=True`. Its static expand layout is inference-only; grad-enabled execution uses compact dispatch. |

Without expert parallelism, `LocalTokenDispatcher` only reorders local tokens.
DeepEP and HybridEP require EP>1. TorchTitan does not expose a separate DeepEP
v1 or direct NCCL EP dispatcher.

Choose by hardware, precision, graph requirements, and memory budget, then
measure training step time and peak memory on the actual routing distribution.
Dispatch/combine microbenchmarks alone do not measure overlap with expert GEMMs
or the cost of activation rematerialization. Run at least 10 training steps so
startup does not dominate the comparison.

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

## Activation rematerialization

The default remat policy saves dispatcher communication and recomputes the
routed-expert `w13` projection. To retain communication under a custom policy:

```python
from torchtitan.distributed.activation_remat import ActivationRematConfig

config.activation_remat = ActivationRematConfig(
    save_regions=["*token_dispatcher.ep_communication"],
)
```

DeepEP and HybridEP save or replay dispatch and combine together under
`ep_communication`; selecting just its `.dispatch` or `.combine` suffix does
not control the pair. Full remat replays both. DeepEP then enables deterministic
receive ordering, adding a sort; HybridEP derives token positions from prefix
scans. The all-to-all dispatcher instead has independent `dispatch` and
`combine` regions that include local reordering and collectives.

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

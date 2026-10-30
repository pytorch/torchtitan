# Kimi K3 MXFP8 recipes

These are text-only, full 93-layer Kimi K3 throughput recipes. Both keep FP32
resident parameters, gradient accumulation/reduction and optimizer state,
with BF16 FSDP compute views and dense/shared GEMMs. Only routed-expert GEMMs
use fast-math MXFP8 E4M3. The model's native attention-residual, FA4 and KDA
implementations remain selected by its `varlen` flavor.

| Recipe | Hardware | PP / VPP / DP / CP / TP / EP | Tokens per microbatch | Microbatches / update | Global tokens / update |
| --- | --- | --- | ---: | ---: | ---: |
| `kimi_k3_dist_moe_mxfp8_gb300_256` | 256 GB300 | 4 / 2 / 64 / 1 / 1 / 64 | 4,096 | 32 | 8,388,608 |
| `kimi_k3_dist_moe_mxfp8_gb200_512` | 512 GB200 | 8 / 1 / 64 / 1 / 1 / 64 | 4,096 | 64 | 16,777,216 |

Both use Interleaved1F1B, one schedule call per update, CUDA-graph F/B replay,
32-document fixed metadata, and twelve validation updates. Eight explicit
chunks preserve original layer FQNs: seven contain twelve layers and the
last contains nine. The 256-GPU loop layout puts chunks `r` and `r + 4` on
physical PP rank `r`; the 512-GPU layout has one chunk per rank.

These are **forced-routing performance controls**, not learned-router
pretraining recommendations. Expert choices are round-robin, phased by EP
source to balance destination token counts even for a partial final cycle.
Gate scores, their normalization and gradients remain learned. No token is
dropped. Changing to learned routing needs a separately validated scratch
budget; the factor-one control budget must not be reused without checking.

## Memory policies

| Policy | 256 GB300 | 512 GB200 |
| --- | --- | --- |
| FSDP after forward | Reshard | Retain |
| FSDP after each backward | Reshard through the native PP API | Native retention default |
| Deferred final reduce-scatter stream | Current stream | Native communication stream |
| Active unsharded chunks | Two | One |
| Optimizer redistribution | Transient, optimizer-only CTA cap 1 | Transient, optimizer-only CTA cap 8 |
| Additional remat saves | Shared `w13`/`w2` everywhere; KDA core and Q/K/V/output-gate projections at layers 58/70/82/90 | None |

Explicit backward prefetch is enabled. The 256-GPU recipe releases unused
startup cache once per local stage before its first training backward;
neither recipe empties the cache per action or per update. Native Dist-MoE
uses activation-slot factor `None`, scratch factor `1.0`, and VMM factor
`None`; the outer remat policy honors its native non-recomputed boundary.
Dist-MoE's backward owns expert recomputation. See the
[integration contract](../../torchtitan/models/common/dist_moe/README.md) and
[remat policy](../../torchtitan/distributed/REMAT.md).

The projection-retention 256-GPU source recipe completed twelve updates at
1,079 tokens/s/GPU, with 255.64 GiB maximum PyTorch reservation. The 512-GPU
Interleaved/VPP1 source completed twelve updates at 1,009.5 tokens/s/GPU,
with 158.41 GiB maximum reservation. These describe the validated source
workloads, not fresh performance measurements of this public port or total
device memory. The public port uses the existing public C4 source rather than
a deployment-specific data path.

## Dependencies and launch

Install TorchTitan's dependencies, FA4/KDA dependencies for the model's native
backend, and public [Dist-MoE](https://github.com/meta-pytorch/dist_moe)
with SiTU-GLU (`situ_glu_gate_beta`, `situ_glu_up_beta`) and MXFP8 support.
The native dependency validated for this stack is commit
`285763b5a0e01059a62c96b0990754a60a6ed94a`.

The 256-GPU recipe additionally requires PyTorch exposing
`PipelineStage.set_reshard_after_backward` and
`FSDPModule.set_reduce_scatter_on_current_stream`. It fails at recipe
construction if either is absent; it does not silently change the memory
policy. The public source commits are
[backward resharding](https://github.com/shuhuayu/pytorch/commit/4d03ce8382b)
and [current-stream reduction](https://github.com/shuhuayu/pytorch/commit/836e2870159).
The complete public [PyTorch stack](https://github.com/shuhuayu/pytorch/tree/k3-dev)
also contains the singleton FSDP reduction-allocation fix used in the original
CUDA-graph runs. The small recipe-specific model subclass only configures
these native APIs on the final reconstructed PP stages; it does not implement
a second pipeline or alter the model's forward/checkpoint layout.

Provide the Kimi tokenizer assets at `./assets/hf/Kimi-K3`, or replace
`hf_assets_path` in a local wrapper recipe. Model weights initialize from
scratch. The bundled dataset is the public `allenai/c4` training source.

Use these validated process-level settings before starting the workers.
`NCCL_GRAPH_REGISTER=0` is the graph-registration workaround used with the
validated NCCL 2.30.7 dependency; revalidate it when updating NCCL.

```bash
export NCCL_RUNTIME_CONNECT=0
export NCCL_GRAPH_REGISTER=0
export PYTORCH_ALLOC_CONF=expandable_segments:True,graph_capture_record_stream_reuse:True
export TORCHINDUCTOR_COMPILE_THREADS=1

# Select one config with your ordinary multi-node torchrun launcher:
python -m torchtitan.train --module torchtitan_recipes.models.kimi_k3 \
  --config kimi_k3_dist_moe_mxfp8_gb300_256
# Or: --config kimi_k3_dist_moe_mxfp8_gb200_512
```

Profiling is selective by default: update six on one rank per PP stage
(`0, 64, ...`). Other ranks do not profile. Run the existing FakePG mode at
each PP coordinate before a full-scale launch; it checks local geometry and
memory, not remote collective performance or learned-routing balance.

# DeepSeek V4.1 Flash validation

## Source and reference

The stack starts from upstream commit
`a87db08ea73cbdd567ceb055bd9118fd4a4507b2` and follows the `add_model` workflow
in PR 5636 at revision `5805da3246b2bf8f479eed3794ab1a3f02760044`.

Reference code is pinned to DeepSeek-V4.1-Flash HF revision
`314113f8a7dadb388215396ba5fbee2c610a04d6`. The comparison harness verifies the
source hashes and uses independent differentiable PyTorch replacements for the
author's inference kernels. Both models use the same initialized text weights.
It compares block and full-model outputs, normalized next-token cross entropy,
parameter gradients, gradient presence, and shared KV/index state.

The vision comparison also checks the ViT encoder, padded unshuffle aligner,
image marker replacement, image logits, and visual parameter gradients.
For the image arithmetic check, each author's attention quantizer receives the
same activation as the native quantizer. All 34 independent quantizer results
must agree bitwise, and incoming arithmetic must pass FP32 tolerances. The raw
independently rounded image-logit error is reported separately. In this fixture,
a `1.49e-8` attention difference straddles an E4M3 midpoint and propagates to a
raw maximum logit error of `1.594e-3`; the common-input comparison is `3.58e-7`.
The text and drafter comparisons retain independent quantization decisions.

The drafter reference checks a five-position next-token cross entropy plus a
confidence BCE objective. It compares all 102 logical HF gradient tensors,
including six zero gradients for routed experts skipped by the author's loop.
The maximum gradient error is `1.49e-8`; both objective values are
`8.35378646850586`. The debug backbone has eight experts and the drafter has four,
exercising the released model's differing backbone and drafter expert counts.

## Feature and checkpoint coverage

CPU feature tests cover quantization rounding and STE gradients, single-pass
mHC, bounded SwiGLU, causal Engram boundaries, hierarchical index selection,
packed documents, empty compressed pools, causal input gradients, exact HF
round trips, and unaligned quantized checkpoint slices across files. Vision
tests compare against an independent SDPA oracle, isolate packed image grids,
check padded channel-major unshuffle, and verify that image tokens receive no
Engram residual update.

The CPU pipeline test runs two different packed microbatches before backward.
Splits exercise local-only boundaries and shared global KV producers followed by
consumers on later stages. No-remat, default, and full rematerialization have
bitwise-identical logits, normalized loss, every parameter gradient, gradient
presence, and buffers. The real boundary tensors also match the supplied static
metadata for shape, dtype, stride, and gradient presence.
Endpoint ownership and the fixed document bound required by these payloads are
also checked; unsupported custom splits and unbounded PP metadata fail early.

| Component | Forward and backward coverage |
| --- | --- |
| CSA2 / shared KV / mHC / Engram | Independent author arithmetic and 547 full-text gradients; causal and packed-document CPU tests |
| MoE / indexer | Independent routed computation, balancing/count reset through real optimizer hooks, native auxiliary-loss training |
| Vision / aligner / image markers | Independent SDPA and author encoder comparison, isolated visual gradients, mixed-rank FSDP training and exact HF round trip |
| DSpark / Markov / confidence | Independent CE+BCE with 102 logical HF gradients, stopped backbone gradients, CPU and FSDP remat parity |
| Quantized HF storage | FP8 and packed FP4 unaligned slices across scale blocks and files; exact native/HF debug round trips |
| Pipeline state | Two outstanding forward graphs across KV boundaries, every gradient and buffer, static boundary metadata |

## Local GPU validation

The reference comparisons and isolated topology checks use the published ARM nightly
`torch==2.16.0.dev20261030+cu132` (git
`bf270c3251d8e84723cedbf5b60eaa60cacaa3f2`), NCCL 2.32.3, CUDA 13.2,
`spmd_types==0.3.1`, and `torch_remat==0.2.2` on NVIDIA GB300 GPUs.
`torchao==0.19.0.dev20261007+cu132`, attention-gym 0.0.17, and
`torch_checkpointing` commit `ca688ec44cd55863198ad8400aae917cd63c8703`
(wheel metadata 0.1.0) complete this runtime. The published PyTorch files are
unmodified. The FP32 reference compares 547 text parameter gradients and the vision path;
its complete measurements are in `tests/assets/deepseek_v4_1/reference_parity.json`.

The mixed-rank vision GPU test runs ten optimization steps with images on only
one of two FSDP ranks, default rematerialization, and the model's optimizer
hooks. It checks visual gradient connectivity, reset routing counters, and
identical image-balancing biases across ranks. It then exports the distributed
state to HF, restores every tensor exactly, loads it into the model, and checks
identical logits. This is a connectivity and checkpoint test; it does not claim
reconstructed multimodal-gradient parity.
The same test then trains the drafter for ten steps with its own optimizer and
routing hooks, checks reconstructed draft projection and confidence gradients,
and verifies that every frozen backbone parameter remains exactly unchanged.
Enabling drafter rematerialization within this same FSDP2 run preserves all ten
full-precision losses and global gradient norms on both ranks bitwise. All 69
reconstructed native drafter parameter gradients are bitwise identical at every
step, with maximum absolute error zero. The CPU rematerialization test also
holds two distinct context tensors live through backward to check their lifetime.

Real two-rank FSDP training with default rematerialization has loss identical to
the single-rank baseline and reconstructed parameter-gradient maximum error
`2.79e-9`. The optional SPMD checker currently rejects upstream DTensor identity
views of empty cache tensors; the FSDP comparison explicitly disables it.
Checker-enabled FSDP validation is pending that upstream fix.

The native CPU seed/checkpoint and fused TP2/EP2 integration recipe use a complete
ARM Python 3.12 environment with the same published PyTorch build,
`torch_checkpointing==0.1.1`, and CUTLASS DSL 4.7.0. Before computation, every
native child verifies all 1,131 source files, the validation helpers, dependency
versions and module origins, and 2,587 published PyTorch Python files. Each node
also verifies all 12,597 published PyTorch payload files, including compiled
libraries. The recipe completes ten C4 optimizer steps with SPMD checking
enabled. All losses and gradient norms are finite; loss changes from
`7.677572250366211` to `6.613213539123535`. The 69 CPU model and integration-definition
tests also pass in this complete runtime.

For the current FSDP validation recipe, set the checker override explicitly:

```python
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan_recipes.tests.models.deepseek_v4_1 import deepseek_v4_1_debugmodel

config = deepseek_v4_1_debugmodel(
    parallelism=ParallelismConfig(data_parallel_shard_degree=2),
)
config.debug.spmd_typechecking = False
```

## Parallelism numerical matrix

Every local topology starts from the same checkpoint and eight 32-token packed
microbatches, with 216 global valid CE labels and 224 routing tokens. The actual
training engine and optimizer hooks run with seed 42 and strict deterministic
execution. All 285 logical parameter gradients are captured before clipping
and reconstructed according to their DTensor placements, counting each logical
parameter once. The baseline is a single real GPU with reference attention and
default rematerialization.

| Topology | Compute | Loss absolute error | Gradient maximum error | Gradient norm ratio | Router counts / biases |
| --- | --- | --- | --- | --- | --- |
| FSDP2 | BF16 | 0 | 2.79e-9 | 1.00000000018 | Exact |
| FSDP2 + EP2 | BF16 | 0 | 4.66e-7 | 1.00000000367 | Exact |
| PP2, four virtual stages | BF16 | 0 | 1.53e-6 | 0.99999999155 | Exact |
| TP2 + EP2 | BF16 | 5.25e-5 | 2.54e-4 | 0.99997558 | 18 count differences; maximum bias difference 8.75e-4 |
| PP2, four virtual stages | FP32 | 0 | 5.82e-11 | 1.00000000000 | Exact |
| TP2 + EP2 | FP32 | 4.77e-7 | 4.72e-6 | 1.00000000034 | Exact |
| FSDP2 + TP2 + PP2 + EP2, GB200 | BF16 | 5.29e-5 | 2.54e-4 | 0.99997561 | 18 count differences; maximum bias difference 8.75e-4 |
| FSDP2 + TP2 + PP2 + EP2, GB200 | FP32 | 0 | 4.72e-6 | 1.00000000690 | Exact |

BF16 TP changes eleven valid tokens' routing choices at near-tied score
boundaries; each changed top-k gap is within twice the measured score error.
Holding only expert IDs fixed restores exact counters and biases, but BF16
partial reductions and activation quantization still produce gradient differences.
The FP32 TP/EP comparison passes PyTorch's default FP32 elementwise tolerance for
all 285 reconstructed gradients; gradient cosine is 0.999999999678.
FP32 PP gradient cosine is 0.99999999999998. This does not claim bitwise
optimizer updates across different topologies: AdamW can magnify a tiny
near-zero gradient sign change through its epsilon, even in the FP32 comparison.

The combined topology comparison runs on eight real GB200 GPUs. Its FP32 loss
matches the single-GPU baseline exactly; all 285 reconstructed parameter
gradients pass PyTorch's default FP32 elementwise tolerance, with gradient
cosine 0.999999999609. Router counts and updated balancing biases are exact.
All eight parent ranks and 18 BF16/FP32 matrix children verify the pinned
source and runtime, and their results are bound to the submitted MAST app ID.

## Remote CI and limits

The combined real eight-GPU test selects FSDP2, TP2, EP2, and PP2 with four
virtual stages, four microbatches, default rematerialization, and AdamW. Its
numerical recipe explicitly selects reference attention; a separate `_fused`
factory selects attention-gym kernels. TP/EP's two-GPU integration test enables
SPMD checking. The pipeline recipe disables it, as required by the existing
training engine. The B200 and H100 suites both select the combined factory.

The complete published runtime passes a two-GB200 bootstrap with verified source
and PyTorch payloads and a successful NCCL all-reduce. The eight-GB200 gradient
matrix also passes. Two independent native `loss_compare.py` launches each
complete ten C4 optimizer steps on eight GB200 GPUs. Every full-precision loss
and grad_norm value agrees bitwise across launches, and their CPU seed payloads
have identical SHA256 hashes. Loss changes from `7.664351940155029` to
`6.559906482696533`. All eight parent ranks and nine native seed/training children
verify the frozen source and published runtime before computation.

The B200 numerical entry registers
`tests/assets/losses/real_pg/deepseek_v4_1_flash_b200.txt`. Its golden is generated
only after the two-launch repeatability proof. A distinct native import replay
with `--assert-equal` completes ten steps on eight GB300 GPUs in two four-GPU
hosts. Every full-precision loss and grad_norm value matches the eight-GB200
golden bitwise, and the CPU seed payload is identical. This comparison covers
the reference-attention debug recipe with the same published runtime.
The golden SHA256 is
`3f21fa384049a96ab70ab855b90e59c4941da6f9ebd6796770ffe3ea9e150213`.
Pinned source/runtime information, native job IDs, and proof hashes
are recorded in
[`numerical_golden.json`](../../../tests/assets/deepseek_v4_1/numerical_golden.json).

### Current upstream source

The golden producers and first GB300 replay use frozen source
`da6b26f9fd7571c6c34bd9a88d647eb1cf597ac1`. The current stack on upstream
`a87db08ea73cbdd567ceb055bd9118fd4a4507b2` is validated separately at
`14ba8df0bf471b97c14fe5c3a76c87db7a5ada4a`. Its four API migrations use the
direct `ParallelismConfig` import and explicit `TensorBoardLogger` configuration.
All 14 model Python files remain byte-identical to the golden source.

A distinct native `--assert-equal` replay of this current source completes ten
steps on eight GB300 GPUs in DCT. Every full-precision loss and grad_norm value
matches the original golden bitwise, and the CPU seed payload is identical.
All eight parent ranks and nine native seed/training children verify 1,134 source
files and the same published runtime. The source comparison verifies the four
API migrations and all unchanged model files explicitly. All 69 CPU tests and
all-file lint pass on the current upstream base. The `current_main_import_replay`
entry in `numerical_golden.json` records this source, scheduler status, native
command, and proof hashes separately from the original golden production.

Released checkpoint manifest names are checked without downloading the released
parameter values. DSpark requires TP=1 and context collection requires an unsplit backbone.
Full-scale convergence, H100 execution, context parallelism
above degree one, packed incremental caches, and optimized serving prefill have
not been validated or implemented here. Ten debug C4 steps validate training
execution and decreasing loss; they are not full-size convergence evidence.

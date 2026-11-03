# Validation

Reference code is pinned to DeepSeek-V4.1-Flash HF revision
`314113f8a7dadb388215396ba5fbee2c610a04d6`. The comparison harness verifies the
source hashes and uses independent differentiable PyTorch replacements for the
author's inference kernels. Both models use the same initialized text weights.
It compares block and full-model outputs, normalized next-token cross entropy,
parameter gradients, gradient presence, and shared KV/index state.

The vision comparison also checks the ViT encoder, padded unshuffle aligner,
image marker replacement, image logits, and visual parameter gradients.

CPU feature tests cover quantization rounding and STE gradients, single-pass
mHC, bounded SwiGLU, causal Engram boundaries, hierarchical index selection,
packed documents, empty compressed pools, causal input gradients, exact HF
round trips, and unaligned quantized checkpoint slices across files. Vision
tests compare against an independent SDPA oracle, isolate packed image grids,
check padded channel-major unshuffle, and verify that image tokens receive no
Engram residual update.

The current local validation uses the published ARM nightly
`torch==2.16.0.dev20261030+cu132` (git
`bf270c3251d8e84723cedbf5b60eaa60cacaa3f2`), NCCL 2.32.3, CUDA 13.2,
`spmd_types==0.3.1`, and `torch_remat==0.2.2` on NVIDIA GB300 GPUs.
The FP32 reference compares 547 text parameter gradients and the vision path;
its complete measurements are in `tests/assets/deepseek_v4_1/reference_parity.json`.

The mixed-rank vision GPU test runs ten optimization steps with images on only
one of two FSDP ranks, default rematerialization, and the model's optimizer
hooks. It checks visual gradient connectivity, reset routing counters, and
identical image-balancing biases across ranks. It then exports the distributed
state to HF, restores every tensor exactly, loads it into the model, and checks
identical logits. This is a connectivity and checkpoint test; it does not claim
reconstructed multimodal-gradient parity.

Real two-rank FSDP training with default rematerialization has loss identical to
the single-rank baseline and reconstructed parameter-gradient maximum error
`2.79e-9`. The optional SPMD checker currently rejects upstream DTensor identity
views of empty cache tensors; the FSDP comparison explicitly disables it.
Checker-enabled FSDP validation is pending that upstream fix.

For the current FSDP validation recipe, set the checker override explicitly:

```python
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan_recipes.tests.models.deepseek_v4_1 import deepseek_v4_1_debugmodel

config = deepseek_v4_1_debugmodel(
    parallelism=ParallelismConfig(data_parallel_shard_degree=2),
)
config.debug.spmd_typechecking = False
```

Released checkpoint manifest keys are checked without downloading
all released values. Full-size convergence has not been validated.

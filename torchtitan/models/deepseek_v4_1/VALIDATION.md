# Validation

Reference code is pinned to DeepSeek-V4.1-Flash HF revision
`314113f8a7dadb388215396ba5fbee2c610a04d6`. The comparison harness verifies the
source hashes and uses independent differentiable PyTorch replacements for the
author's inference kernels. Both models use the same initialized text weights.
It compares block and full-model outputs, normalized next-token cross entropy,
parameter gradients, gradient presence, and shared KV/index state.

CPU feature tests cover quantization rounding and STE gradients, single-pass
mHC, bounded SwiGLU, causal Engram boundaries, hierarchical index selection,
packed documents, empty compressed pools, causal input gradients, exact HF
round trips, and unaligned quantized checkpoint slices across files.

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

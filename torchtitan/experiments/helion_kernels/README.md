# Experimental Helion Kernels

This experiment hosts prototype Helion kernels for TorchTitan model hotspots.
Nothing in core TorchTitan depends on this package.

## RoPE Candidates

The first target is the cos/sin RoPE path used by Qwen3, GPT-OSS, and Qwen3-VL.
The current core implementation materializes the broadcasted/gathered RoPE cache,
casts Q and K to fp32, builds `rotate_half` with `torch.cat`, then computes Q and K
rotations. A Helion kernel can fuse the position lookup, rotation, fp32 math, and
cast-back into one GPU kernel for both Q and K.

Implemented prototype:

```python
from torchtitan.experiments.helion_kernels.rope import (
    apply_rotary_emb_cos_sin_helion,
)
```

Supported fast path:

- `xq` and `xk`: contiguous CUDA tensors shaped `(batch, seq_len, heads, head_dim)`
- `rope_cache`: contiguous CUDA tensor shaped `(max_seq_len, head_dim * 2)`
- `positions`: contiguous CUDA tensor shaped `(batch, seq_len)`

Other cases fall back to `torchtitan.models.common.rope.apply_rotary_emb_cos_sin`.

## RMSNorm Candidate

Standalone RMSNorm now has an experimental forward-only Helion path:

```python
from torchtitan.experiments.helion_kernels.rmsnorm import rms_norm_helion
```

Supported fast path:

- contiguous CUDA tensors with affine `weight`
- input dtype and weight dtype in `{float16, bfloat16, float32}`
- inference/no-grad execution

The public wrapper is selective for the current B200 Qwen3 sweep: it uses
Helion for short hidden-size norms `(num_rows <= 256, dim=5120)` and the
largest Q/K norm rows `(num_rows >= 4096, dim=128)`, and falls back to
`torch.nn.functional.rms_norm` elsewhere. Use `rms_norm_helion_raw` to
benchmark the raw Helion kernel without this routing.

Benchmark the pasted Qwen3 shape/count sweep with:

```bash
python -m torchtitan.experiments.helion_kernels.bench_rmsnorm \
  --backends aten helion helion_raw compile
```

## Q/K RMSNorm + RoPE Fusion

Qwen-style Q/K normalization can be fused with the cos/sin RoPE application:

```python
from torchtitan.experiments.helion_kernels.rope import (
    apply_qk_rmsnorm_rotary_emb_cos_sin_helion,
)
```

Supported fast path:

- `xq` and `xk`: contiguous CUDA tensors shaped `(batch, seq_len, heads, head_dim)`
- `q_weight` and `k_weight`: contiguous CUDA tensors shaped `(head_dim,)`
- `rope_cache`: contiguous CUDA tensor shaped `(max_seq_len, head_dim * 2)`
- `positions`: contiguous CUDA tensor shaped `(batch, seq_len)`
- inference/no-grad execution

Other cases fall back to `F.rms_norm` followed by
`torchtitan.models.common.rope.apply_rotary_emb_cos_sin`.

Benchmark with:

```bash
python -m torchtitan.experiments.helion_kernels.bench_qk_rmsnorm_rope
```

Sweep Helion row-tile and warp configs with:

```bash
python -m torchtitan.experiments.helion_kernels.bench_qk_rmsnorm_rope --sweep
```

Next candidates:

- A real-valued replacement for complex RoPE, since Helion currently does not support
  complex tensors directly.
- MRoPE/Qwen3-VL support for pre-broadcast 4D cos/sin caches.

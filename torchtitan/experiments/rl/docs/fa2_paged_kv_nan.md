# FA2 auto num_splits + paged KV produces NaN

## Summary

Flash Attention 2's auto `num_splits` heuristic intermittently produces NaN when used with paged KV cache (`block_table`). This was introduced by [pytorch/pytorch#179760](https://github.com/pytorch/pytorch/pull/179760), which added `num_splits` support to FA2. The auto heuristic can select a split configuration that corrupts output for paged KV layouts.

Setting `num_splits=1` explicitly avoids the issue. Notably, other explicit values (2, 4, 8, ...) also work — only the auto heuristic triggers NaN.

**Tracking**: [pytorch/pytorch#179760](https://github.com/pytorch/pytorch/pull/179760)

## Impact

In vLLM's decode path, the KV cache is paged (managed via `block_table`). Each decode step has a ~1% chance of producing NaN with the default auto `num_splits`. Over hundreds of decode steps per generation, this accumulates and corrupts model outputs, causing RL training to produce zero or NaN loss.

## Affected configurations

- **GPU**: Observed on A100 (SM 8.0) and A10G (SM 8.6)
- **PyTorch**: Nightly 2.13+ (post pytorch/pytorch#179760)
- **FA impl**: FA2 (default on SM < 9.0). FA3 is not affected by this bug but has a separate non-determinism issue with `num_splits > 1`.

## Reproduce

```python
"""
FA2 + auto num_splits + paged KV → NaN.
Mirrors vLLM's decode pattern: pre-allocated KV cache with one new
token appended per sequence per step.
Expected: num_splits=auto produces NaN, num_splits=1 does not.
"""
import torch
from torch.nn.attention.varlen import varlen_attn_out

device = torch.device("cuda:0")
dtype = torch.bfloat16
num_heads, num_kv_heads, head_dim = 16, 8, 64
block_size = 256
scale = head_dim**-0.5

# 10 sequences with varying prompt lengths
num_seqs = 10
prompt_lens = [512, 1024, 2048, 300, 700, 1500, 3000, 4096, 800, 2500]

# Pre-allocate paged KV cache
N = 500  # decode steps
max_bps = max((s + N + block_size - 1) // block_size for s in prompt_lens)
total_blocks = num_seqs * max_bps + 4
key_cache = torch.zeros(total_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=dtype)
value_cache = torch.zeros_like(key_cache)

# Block table
block_table = torch.zeros(num_seqs, max_bps, dtype=torch.int32, device=device)
bi = 0
for i, plen in enumerate(prompt_lens):
    for j in range((plen + N + block_size - 1) // block_size):
        block_table[i, j] = bi
        bi += 1

# Prefill KV cache
for i, plen in enumerate(prompt_lens):
    left = plen
    for j in range((plen + block_size - 1) // block_size):
        blk = block_table[i, j].item()
        fill = min(left, block_size)
        key_cache[blk, :fill] = torch.randn(fill, num_kv_heads, head_dim, device=device, dtype=dtype)
        value_cache[blk, :fill] = torch.randn(fill, num_kv_heads, head_dim, device=device, dtype=dtype)
        left -= fill

cu_q = torch.arange(num_seqs + 1, dtype=torch.int32, device=device)

# Simulate decode steps
for label, kwargs in [("auto", {}), ("1", {"num_splits": 1})]:
    kv_lens = list(prompt_lens)
    nan_count = 0
    for step in range(N):
        q = torch.randn(num_seqs, num_heads, head_dim, device=device, dtype=dtype)
        for i in range(num_seqs):
            pos = kv_lens[i]
            blk = block_table[i, pos // block_size].item()
            key_cache[blk, pos % block_size] = torch.randn(num_kv_heads, head_dim, device=device, dtype=dtype)
            value_cache[blk, pos % block_size] = torch.randn(num_kv_heads, head_dim, device=device, dtype=dtype)
            kv_lens[i] += 1
        seqused_k = torch.tensor(kv_lens, dtype=torch.int32, device=device)
        cu_k = torch.zeros(num_seqs + 1, dtype=torch.int32, device=device)
        cu_k[1:] = torch.cumsum(seqused_k, dim=0)
        out = torch.empty_like(q)
        result = varlen_attn_out(
            out, q, key_cache, value_cache, cu_q, cu_k, 1, max(kv_lens),
            scale=scale, window_size=(-1, 0), block_table=block_table,
            seqused_k=seqused_k, enable_gqa=True, **kwargs,
        )
        if torch.isnan(result).any():
            nan_count += 1
    print(f"num_splits={label:>4}: {nan_count}/{N} decode steps had NaN")
```

Example output on A100:
```
num_splits=auto:   5/500 decode steps had NaN
num_splits=   1:   0/500 decode steps had NaN
```

The NaN rate is low per step (~1%) but accumulates over a full generation.

## Workaround

Force `num_splits=1` when calling FA2 with paged KV:
```python
extra_kwargs["num_splits"] = 1
```

Applied in `torchtitan/experiments/rl/models/attention.py`.

## CI evidence

- Fix PR: https://github.com/pytorch/torchtitan/pull/3041
- CI run showing NaN before fix: https://github.com/pytorch/torchtitan/actions/runs/25398907645/job/74493004250?pr=3041

# SGLang generator engine

**Status:** an alternative to the vLLM engine. Verified only at SGLang
`851642ade`, which `scripts/install_rl_env.sh --sglang` installs; CI runs
`test_sglang_bitwise_parity.py` on B200 (`integration_test_rl_b200.yaml`). It
relies on private SGLang APIs (see the TODOs linking sgl-project/sglang issues),
so other SGLang versions may break it.

`SGLangEngine` runs the generator's TorchTitan model inside
[SGLang](https://github.com/sgl-project/sglang), as `VLLMEngine` does inside
vLLM. SGLang provides KV-cache management (paged KV pool, radix prefix cache)
and scheduling (continuous batching, chunked prefill, priority, overlap
scheduling, CUDA graphs); the model and its attention kernel are TorchTitan's.

```
Generator engine loop (every rank, SPMD)
  SGLangEngine
    add_requests ----> Scheduler.process_input_requests   (no tokenizer/detokenizer procs)
    step ------------> one iteration of Scheduler.event_loop_overlap
                       (event_loop_normal for GDN models)
                          ModelRunner -> SGLangModelWrapper.forward(input_ids, positions, forward_batch)
                                           TorchTitan Decoder
                                             Attention -> SGLangInnerAttention -> RadixAttention
                                                            TorchTitanVarlenAttnBackend:
                                                            SGLang's paged-KV metadata,
                                                            torch varlen FlashAttention (the trainer's kernel)
                                             GDN -> SGLangInnerGDN -> RadixLinearAttention
                                                      TorchTitanGDNAttnBackend: SGLang's state
                                                      slots, the trainer's Attention Gym kernels
                                           lm_head + TP gather -> SGLang LogitsProcessor / Sampler
    load_model_state_dict -> ModelStateDictLoader (FSDP groups, in place: CUDA graphs stay valid)
```

Each rank builds its own SGLang `Scheduler` in-process. SGLang's TP
schedulers make identical scheduling decisions, which is the contract the
engine loop already relies on.

## Use it

Select it by config type:

```python
from torchtitan.rl.generator.sglang.engine import SGLangEngine

generator=Generator.Config(..., engine=SGLangEngine.Config())
```

`torchtitan_recipes.rl.alphabet_sort.rl_grpo_qwen3_0_6b_varlen_sglang` is a
complete recipe (6 GPUs: 4 generator + 2 trainer).

## Environment

SGLang pins an older PyTorch than TorchTitan's nightly, so install it into its
own environment on top of the RL environment, without replacing PyTorch:

```bash
uv venv --python 3.12 .venv-sglang && source .venv-sglang/bin/activate
# The RL environment: PyTorch and vLLM nightlies (batch-invariant mode uses vLLM's
# batch-invariant bmm), TorchStore and FlashAttention; then SGLang at the verified
# commit, keeping the RL environment's PyTorch.
scripts/install_rl_env.sh --sglang
```

## Supported

| Feature | Status |
| --- | --- |
| Dense softmax attention, varlen backend (Qwen3, Llama 3) | Supported |
| Tensor parallelism | Supported; outside batch-invariant mode, the model's TP all-reduces run on SGLang's TP all-reduce (custom all-reduce) instead of NCCL |
| CUDA graphs (decode, and breakable prefill graphs), overlap scheduling, chunked prefill | Supported |
| Priority scheduling by rollout group | Supported |
| Model state dict loads (`cpu_prefetch`, `gpu_fetch`) | Supported |
| Policy-version prefix cache isolation | Supported (`cache_salt`) |
| `reset_kv_cache_on_model_state_dict_load` | Supported: running requests are retracted and recompute their KV |
| Multi-turn session KV | Supported, always on; see below |
| CPU KV offload (SGLang HiCache) | Supported through `extra_server_args`; see below |
| Batch-invariant mode | Supported; see below |
| Per-request metrics (queue time, TTFT, prefill, decode, preemptions, cached tokens) | Supported |
| Data parallelism within a generator | Not supported: run more generators |
| Sliding-window attention (e.g. Muse Glimmer) | Supported; KV of every token stays cached (no sliding-window KV pool) |
| GDN linear attention (Qwen3.5) | Supported; see below |
| MLA (DeepSeek V3) | Supported with per-head K/V in the KV cache, as the vLLM engine runs MLA in batch-invariant mode; SGLang's compressed-latent MLA cache is not used |
| Flex attention, KDA, attention sinks (GPT-OSS) | Not supported |
| MoE with expert parallelism over the TP group, Dist-MoE | Supported; see below |
| Sequence parallelism | Not supported |

Unsupported configurations raise at engine build.

## CUDA graphs

`SGLangEngine.Config.cuda_graph` picks what SGLang captures. SGLang runs decode
and prefill in separate batches; `"decode"` captures full graphs for decode
batches. The default, `"breakable"`, also captures prefill batches as SGLang's
breakable CUDA graphs: the model runs in graph segments, with attention eager in
between.
Without them, prefill is bound by the eager model's CPU overhead (about 7x the
GPU time of a 16K-token prefill chunk of Qwen3-0.6B at TP 2). Capture adds
about 30 s to startup; `"decode"` skips it.

Attention kernels compile on first use for each kernel variant, so the first
prefill batches of a run can take a few seconds.

## MoE

With expert parallelism, the MoE layers shard each step's tokens over the TP
ranks, so the model pads a step to a multiple of the TP degree; attention skips
the padding rows. The all-to-all token dispatcher reads its split sizes on the
host, which CUDA graphs cannot capture: it needs `cuda_graph="none"`. Dist-MoE
(`Generator.Config.dist_moe_runtime`) runs under both kinds of CUDA graphs. It
shards a step's tokens over the TP ranks even without expert parallelism, so with
TP > 1 it needs expert parallelism too. The other EP token dispatchers (DeepEP,
HybridEP) are not supported.

## GDN

SGLang keeps each request's GDN conv and recurrent state in its state pool, and
its radix cache snapshots them, so a session's next turn resumes from its cached
prefix as with KV. The trainer's Attention Gym kernels advance the states
(`TorchTitanGDNAttnBackend`, registered through SGLang's linear attention model
registry). The backend registers without SGLang's `extra_buffer` mamba cache
strategy, which needs the kernels to write state snapshots at intermediate
positions. SGLang then snapshots a request's state only when it caches the
request, and disables overlap scheduling: the engine steps SGLang's normal event
loop for these models.

## Multi-turn sessions

Each request carries its rollout's `routing_session_id` as an SGLang session.
When a turn finishes, SGLang's radix cache references the session's cached
prefix (prompt and completion) until the rollout ends and the engine loop
releases the session or its group. Eviction takes unreferenced KV first and
referenced KV only when nothing else is left, so the next turn usually reuses
the whole history; no request ever waits for a held session.

## CPU KV offload

SGLang's hierarchical cache keeps KV evicted from GPU memory in host memory and
loads it back on a prefix hit:

```python
SGLangEngine.Config(
    extra_server_args={
        "enable_hierarchical_cache": True,
        "hicache_size": 64,  # GB of host memory per rank
        "hicache_write_policy": "write_back",
    }
)
```

KV loaded back is bitwise identical to recomputed KV (checked in batch-invariant
mode on a multi-turn workload), and a reset on model state dict load clears the
host cache too. For GDN models, it keeps the conv and recurrent states of evicted
prefixes in host memory too: a cached turn whose prefix was loaded back matched
the trainer's prefill bitwise. HiCache pays off when recomputing a prefix costs
more than copying its KV: for Qwen3-0.6B, recompute is faster.

## Seeds

SGLang samples with each request's seed only in its deterministic mode, which
the engine turns on with `DebugConfig(seed=...)`; otherwise a request with a seed
is rejected. A seed makes sampling reproducible for given logits; the logits
themselves are reproducible across batches only in batch-invariant mode.

## Batch-invariant mode

With `DebugConfig(batch_invariant=True)`, the generator matches the trainer's
logprobs bitwise at matched TP (see `torchtitan/rl/BITWISE_PARITY.md`):

- attention runs the trainer's varlen FlashAttention with `num_splits=1`
  (`torchtitan/rl/generator/attention/attention.py`, shared with the vLLM engine);
- the sampled tokens' logprobs come from the trainer's `compute_logprobs` on
  `logits / temperature` (`batch_invariance.py`);
- SGLang's custom all-reduce is off, and the model's TP all-reduces run on NCCL.

`tests/rl/unit_tests/gpu/test_sglang_bitwise_parity.py` checks, for each
supported model family, trainer vs. SGLang decode parity (also for a session's
next turn, which resumes its cached prefix, and after an in-place model state
dict load) and SGLang decode batch invariance.

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
                          ModelRunner -> SGLangModelWrapper.forward(input_ids, positions, forward_batch)
                                           TorchTitan Decoder
                                             Attention -> SGLangInnerAttention -> RadixAttention
                                                            TorchTitanVarlenAttnBackend:
                                                            SGLang's paged-KV metadata,
                                                            torch varlen FlashAttention (the trainer's kernel)
                                             GDN / KDA -> SGLangInnerGDN / SGLangInnerKDA -> RadixLinearAttention
                                                            TorchTitanGDNAttnBackend / TorchTitanKDAAttnBackend:
                                                            SGLang's state slots, the trainer's Attention Gym kernels
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
| Tensor parallelism | Supported; outside batch-invariant mode, the model's TP all-reduces run on SGLang's attention-TP all-reduce (custom all-reduce) instead of NCCL |
| CUDA graphs (decode and prefill), overlap scheduling, chunked prefill | Supported |
| Priority scheduling by rollout group | Supported |
| Prompt KV reuse within a rollout group | Supported; linear attention (GDN, KDA) reuses the state at the prompt's last 64-token boundary; see below |
| Model state dict loads (`cpu_prefetch`, `gpu_fetch`) | Supported |
| Policy-version prefix cache isolation | Supported (`cache_salt`) |
| `reset_kv_cache_on_model_state_dict_load` | Supported: running requests are retracted and recompute their KV |
| Multi-turn session KV | Supported, always on; see below |
| CPU KV offload (SGLang HiCache) | Supported through `extra_server_args`; see below |
| KV shared across generators (Mooncake Store) | Supported through `extra_server_args`; see below |
| Batch-invariant mode | Supported; see below |
| Per-request metrics (queue time, TTFT, prefill, decode, preemptions, cached tokens) | Supported |
| Data parallelism within a generator | Supported with expert parallelism, as SGLang's attention DP; see below |
| Sliding-window attention (e.g. Muse Glimmer) | Supported; KV of every token stays cached (no sliding-window KV pool) |
| GDN linear attention (Qwen3.5) | Supported; see below |
| KDA linear attention (Kimi K3) | Supported; see below |
| MLA (DeepSeek V3) | Supported with per-head K/V in the KV cache, as the vLLM engine runs MLA in batch-invariant mode; SGLang's compressed-latent MLA cache is not used |
| Flex attention, attention sinks (GPT-OSS) | Not supported |
| MoE with expert parallelism over the DP and TP ranks, Dist-MoE | Supported; see below |
| Sequence parallelism | Not supported |

Unsupported configurations raise at engine build.

## CUDA graphs

`SGLangEngine.Config.cuda_graph` picks what SGLang captures. SGLang runs decode
and prefill in separate batches; `"decode"` captures full graphs for decode
batches. The default, `"full"`, also captures prefill batches as SGLang's full
prefill CUDA graphs, one per token-count bucket, with a request slot per running
request. Without them, prefill is bound by the eager model's CPU overhead (about
7x the GPU time of a 16K-token prefill chunk of Qwen3-0.6B at TP 2). SGLang's
linear attention backend fails to capture full prefill graphs, so linear
attention models (GDN, KDA) raise with `"full"` and need `"breakable"`: SGLang's
breakable prefill graphs, where the model runs in graph segments, with attention
eager in between, at a CPU cost per layer in every prefill step. Capture adds
about 30 s to startup; `"decode"` skips it.

Attention kernels compile on first use for each kernel variant, so the first
prefill batches of a run can take a few seconds.

## MoE

With expert parallelism, the MoE layers shard each step's tokens over the TP
ranks, so the model pads a step to a multiple of the TP degree; attention skips
the padding rows. The all-to-all token dispatcher reads its split sizes on the
host, which CUDA graphs cannot capture: it needs `cuda_graph="none"`. DeepEP with
`cuda_graph_compatible` and HybridEP with `non_blocking_capacity_factor` dispatch
without a host sync, so they run under CUDA graphs; without those options they
need `cuda_graph="none"` too. Their `num_max_tokens_per_rank` must cover a step's
tokens per EP rank, `(chunked_prefill_size + max_running_requests) / TP`, with
the per-DP-rank values; the engine checks it at build. HybridEP needs an EP degree divisible by 4: a step can
have any token count, which HybridEP's 16-byte alignment otherwise rejects.
DeepEP's deterministic dispatch (`DebugConfig.deterministic`) cannot run in CUDA
graphs, and HybridEP's `non_blocking_capacity_factor` must be 1.0 (below it,
HybridEP drops tokens). `scripts/install_rl_env.sh` installs neither DeepEP nor
HybridEP; build them against the environment's PyTorch, as
`.github/scripts/install_deepep_v2.sh` (DeepEP) and
`.ci/docker/common/install_deepep.sh` (HybridEP) do. SGLang's own DeepEP wheel
fails its NCCL version check under the PyTorch nightly. Dist-MoE
(`Generator.Config.dist_moe_runtime`) runs under CUDA graphs: decode graphs, and
breakable or full prefill graphs. It shards a step's tokens over the TP ranks
even without expert parallelism, so with TP > 1 it needs expert parallelism too.

## Data parallelism

`data_parallel_degree` > 1 supplies ranks for expert parallelism (Generator.Config
requires `expert_parallel_degree == data_parallel_degree * tensor_parallel_degree`).
The engine runs it as SGLang's attention DP: one SGLang instance over the DP x TP
world (`tp_size = DP * TP`, `attn_dp_size = DP`). Each DP rank runs its own
requests, with attention heads sharded over its attention-TP group, and the MoE
layers run EP over every rank. For DP 2 x TP 2:

```
world = 4 ranks, DP 2 x TP 2
SGLang TP group (= world):   [0 1 2 3]   <- EP 4 for the MoE layers
attention-TP groups:         [0 1] [2 3] <- heads sharded over these
DP rank 0 = ranks 0, 1       DP rank 1 = ranks 2, 3
```

NOTE: SGLang names its groups after the whole instance. SGLang's "TP" (`tp_size`,
`tp_group`, `tensor_model_parallel_all_reduce`) spans all DP x TP ranks, while its
"attention TP" (`attn_tp_size`, `attn_tp_group`, `attn_tp_all_reduce`) is one DP
rank's TP ranks: torchtitan's `tensor_parallel_degree`. So the engine's TP code
(attention head sharding, the model's routed TP all-reduce) uses the attention-TP
names. The two coincide at DP = 1.

The EP collectives span the DP ranks, so every DP rank steps together: SGLang runs
a DP rank without requests on an empty batch, which the model pads to one row per
TP rank, and `has_unfinished_requests` is True on every rank while any DP rank has
requests. `max_num_seqs` and `max_num_batched_tokens` are per DP rank.

## GDN and KDA

SGLang keeps each request's GDN or KDA conv and recurrent state in its state
pool, and its radix cache snapshots them, so a session's next turn resumes from
its cached prefix as with KV. The trainer's Attention Gym kernels advance the
states (`TorchTitanGDNAttnBackend` and `TorchTitanKDAAttnBackend`, registered
through SGLang's linear attention model registry). They use SGLang's
`extra_buffer` mamba cache strategy, which keeps overlap scheduling: SGLang
gives each request two track slots, and the backends snapshot the request's
state into one at the positions the radix cache tracks, so the next batch can
keep advancing the request's own slot.

- Prefill: at the last 64-token boundary of the new tokens. The conv history is
  copied from the inputs, and Attention Gym writes the recurrent state
  (`StateCheckpoints`).
- Decode: SGLang copies the states every `mamba_track_interval` tokens (64;
  SGLang's default is 256).

A request with a cached prompt reuses the state at the prompt's last 64-token
boundary, and a session's next turn resumes from the previous turn's last
snapshot, at most 63 tokens before its end.

In batch-invariant mode, a KDA state is the state after the last complete
64-token chunk plus Attention Gym's replay state of the unfinished chunk, as in
the vLLM engine. SGLang's state pool holds the replay tensors as extra conv
state entries of each slot (`TorchTitanKDAConfig`), so its radix cache snapshots
them with the state; at head dim 128, they take 1.25x the memory of the
recurrent states. SGLang counts prefill snapshot positions from the start of an
extend, which chunked prefill can leave off the 64-token grid, so a KDA prefill
snapshot is the state at its last complete chunk (`StateCheckpoints`) plus the
inputs after that chunk as its replay state (`copy_replay_tails`). Decode
snapshots are at multiples of 64 tokens, where the replay state is empty;
SGLang does not copy the replay entries, so the backend zeroes the track slot's
replay count.

### KV and states share one pool

For GDN and KDA models, the engine turns on SGLang's unified memory: the full
attention layers' KV cache and the GDN / KDA states share one pool, so memory
that no state uses can hold KV, and the other way around. A running request
takes 5 state slots (3, plus the 2 track slots of `extra_buffer`), so SGLang runs
at most `state slots // 5` requests (about 428 for Qwen3.5-9B at TP 1,
`gpu_memory_limit=0.86`, on GB300); for more, raise `max_mamba_cache_size` in
`extra_server_args`. Unified memory costs about 1 ms of a ~25 ms decode step.

## Multi-turn sessions

Each request carries its rollout's `routing_session_id` as an SGLang session.
When a turn finishes, SGLang's radix cache references the session's cached
prefix (prompt and completion) until the rollout ends and the engine loop
releases the session or its group. Eviction takes unreferenced KV first and
referenced KV only when nothing else is left, so the next turn usually reuses
the whole history; no request ever waits for a held session.

## Rollout groups

The samples of a rollout group share a prompt. SGLang inserts a request's KV
into its radix cache only after the request's prefill, so if a group's samples
are prefilled in the same batch, each computes the whole prompt. (SGLang's
in-batch prefix caching only reorders the waiting queue, under the `lpm`
policy, which priority scheduling excludes.)

So the engine puts the first request with a given prompt into SGLang's waiting
queue and holds the others outside it until that request's prefill is done.
The held requests then read the prompt's KV from the radix cache and compute
only the prompt's last token (with pages larger than one token, its last
partial page):

```
add_requests([g7/s0, g7/s1, ..., g7/s15])   same prompt, same cache salt
  waiting queue: g7/s0                      held: g7/s1 .. g7/s15
step()  prefill g7/s0 -> its prompt KV enters the radix cache
  waiting queue: g7/s1 .. g7/s15            each prefills only the last token
```

For Qwen3-4B (TP 1) and 8 prompts x 16 samples admitted at once, the cached
share of prompt tokens goes from 0% to 94% for 1K-token prompts (TTFT p50
0.46 s -> 0.10 s) and from 59% to 94% for 4K-token prompts (0.73 s -> 0.25 s).
The held requests keep their priority and wait queue entry times, so groups
still run oldest first. Linear attention (GDN, KDA) models hold requests too: a
held request resumes from the leader's state snapshot at the prompt's last
64-token boundary and recomputes the rest of the prompt (see [GDN and KDA](#gdn-and-kda)).

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

KDA models raise with HiCache for now (#6015). TODO: with HiCache's `write_through`
policy (SGLang's default), one run in 18 of the KDA parity class mismatched the
trainer (max logprob delta 3.1e-2); not root-caused.

## Shared KV across generators (Mooncake Store)

[Mooncake Store](https://github.com/kvcache-ai/Mooncake) adds a KV tier below
HiCache's host memory that all generators share, so a session's next turn that
lands on another generator loads its history's KV instead of recomputing it:

```text
turn 1 on generator A: prefill, decode; HiCache copies the KV to host memory and the Store
turn 2 on generator B: GPU and host miss, Store hit -> load the history's KV,
                       prefill only the new tokens
```

`scripts/install_rl_env.sh` already installs the Mooncake SDK (for the
transport API). The controller starts one `mooncake_master` for the job on its
host, on a free port, and adds its address to each generator's
`hicache_storage_backend_extra_config`; the master exits with the controller.
Configure the generators; each generator contributes the Store memory:

```python
SGLangEngine.Config(
    extra_server_args={
        "enable_hierarchical_cache": True,
        "hicache_size": 64,  # GB of host memory per rank
        "hicache_storage_backend": "mooncake",
        "hicache_storage_backend_extra_config": json.dumps(
            {
                "metadata_server": "P2PHANDSHAKE",
                "protocol": "rdma",
                "device_name": "mlx5_0",
                # Host memory this generator contributes, split across its TP
                # ranks. It and the KV in it go away when the generator exits.
                "global_segment_size": "32GB",
                "extra_backend_tag": f"my-run-{uuid.uuid4().hex}",
            }
        ),
    }
)
```

- Each rank serves its Store memory at `MOONCAKE_LOCAL_HOSTNAME` (default
  `localhost`). With generators on more than one host, set it on each host to
  an address the other hosts reach (an IP where hosts cannot resolve each
  other's names), plus `MC_USE_IPV6=1` for an IPv6 address: without it,
  Mooncake's handshake cannot resolve IPv6 addresses.
- The Store sits below HiCache's host memory (a load goes Store -> host -> GPU),
  so it needs `enable_hierarchical_cache`. Keep the default `write_through`
  policy: with `write_back`, KV reaches the Store only when evicted from GPU.
- A Store key is `extra_backend_tag`, the model path, the request's cache salt
  (its pinned policy version), the token prefix and the TP rank. So:
  - `extra_backend_tag` scopes Store keys to a run (the engine raises without
    one): the Store outlives a run and a new run's policy versions restart at
    0. A resume can keep its tag to reuse KV, but it would also read KV the
    previous launch computed after the checkpoint, with weights the resumed
    run retrains. Use a new tag unless training is reproducible.
  - A request reuses another generator's KV only under the same pinned policy
    version, and generators sharing a tag need the same TP degree.
- `reset_kv_cache_on_model_state_dict_load=True` raises with a storage backend:
  it clears KV at each load instead of salting requests, but SGLang's KV reset
  clears GPU and host memory, not the Store, so unsalted requests would read KV
  of earlier weights. SGLang never deletes Store keys; Mooncake evicts them when
  the Store is full.
- With 1-token pages (see `_build_server_args`), the Store holds one object per
  token, K and V, and TP rank.

Checked on H100, TP 2:

- **Correctness** (Qwen3-0.6B, batch-invariant mode): a second turn on another
  generator loaded 649 of its 650 history tokens from the Store and decoded the
  same tokens and logprobs, bitwise, as recomputing them. Another tag or policy
  version loaded nothing.
- **Performance**: second turn of 4 sessions of 16K tokens on another generator,
  timed until all finish. The Store side uses one RDMA NIC and 16 GB of host
  memory per rank; the recompute side runs without HiCache.

| Model | Recompute (s) | Load from the Store (s) | Store vs. recompute |
| --- | ---: | ---: | --- |
| Qwen3-8B | 1.38 | 1.93 | 1.4x slower |
| Qwen3-32B | 5.24 | 2.45 | 2.1x faster |

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

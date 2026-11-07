# Local CPU KV cache

Titan RL uses vLLM's [SimpleCPUOffloadConnector](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py)
to reload KV from each worker's CPU DRAM after GPU eviction. Independent
generators and DP replicas have separate caches; use sticky routing for reuse
between turns.

## Setup

The tested vLLM version rejects expandable allocations with KV connectors.
Set these before importing PyTorch:

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:False
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
```

These job-wide exports also override the trainer's allocator defaults. Disabling
expandable segments can increase reserved-memory fragmentation, leaving less
headroom for DeepEP buffers and CUDA graph pools.

## Generator configuration

```python
from vllm.config import KVTransferConfig

from torchtitan.rl.generator.vllm.engine import VLLMEngine

engine = VLLMEngine.Config(
    kv_transfer_config=KVTransferConfig(
        kv_connector="SimpleCPUOffloadConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"cpu_bytes_to_use_per_rank": 1 << 30},
    ),
)
```

Capacity is **per GPU worker**: TP=2 uses 2 GiB here. Leave room for weight
buffers and other host processes. Prefix caching is required. Default eager
offload saves eligible prompt and decode blocks, including single-turn requests.

Eight-GPU recipe:
`torchtitan_recipes.tests.rl.alphabet_sort.rl_grpo_qwen3_5_debug_varlen_cpu_offload`.

## Weight and cache lifecycle

- CPU blocks share the GPU prefix cache's salt, the request's
  `min_policy_version`: a group reuses its offloaded KV across weight updates
  exactly as it reuses GPU KV, and new groups start under the current version.
  The salt is the group's starting version, not the weights that computed a
  block: with hot swap, or across turns of a group pinned to an older version,
  cached GPU and CPU blocks may hold KV computed under different weights. For
  strict recompute, use `reset_kv_cache_on_model_state_dict_load=True`, which
  KV offload does not support.
- Weight updates may hot swap (`generator_router.hot_swap=True`). Saves and
  loads only copy KV blocks, which a weight load does not touch; the engine
  keeps stepping pending transfers after responses return.
- Keep `generator.reset_kv_cache_on_model_state_dict_load=False`: a reset cannot
  free GPU blocks that in-flight saves still hold.

## Session release

Enable the [session holder](session_kv_holder.py) with
`session_kv_holder=VLLMSessionKVHolder.Config()` to hold GPU prefixes between turns
and forward session/group release through
`connector.release_session_cache(released_hashes, active_hashes)`. With
`session_kv_holder=None`, CPU offload still works, but eviction follows the
connector's normal LRU policy without session release hints.

```text
Session/group ends -> notify generators -> wait for requests and saves
                  -> prioritize unused CPU blocks for eviction
```

A session release reaches the rank 0 of the generator the session is pinned to
(sticky routing); a group release reaches every generator. Rank 0 broadcasts
the release to its workers. A generator the session left after a repin
releases its KV when the group ends. Without the holder, generators ignore
releases.

Released blocks remain reusable until reclaimed. Hashes and aliases used by
active sessions on the local DP replica are protected. This is a one-shot hint;
later requests can cache the prefix again. GPU pressure alone does not release
CPU KV.

## Multi-GPU coordination

Each TP scheduler queries its local CPU cache. The
[SimpleCPU adapter](simple_cpu_offload.py) scopes save completion to its TP group;
the [shared scheduler](cpu_kv_offload.py) coordinates request order and completion:

```text
Local lookup -> allocate GPU blocks -> each worker loads its KV shard
             -> aggregate TP completions -> resume requests / release references
```

All generator ranks synchronize whether requests, scheduler work, or queued
outputs remain. Idle DP replicas join required EP forwards with dummy batches
until all work drains. Independent generators do not join these collectives.

## Validation

The full matrix requires eight available GPUs. Run from the repo root:

```bash
pytest -q tests/rl/unit_tests/gpu/test_kv_offload.py
```

Tests cover CPU restoration, session release, policy isolation, hot swap,
TP/DP/EP, and hybrid tails, all without CUDA graphs. The RL multi-GPU CI job
runs TP=2/DP=2/EP=4 and hybrid TP=2 cases. FULL CUDA graphs were benchmarked
on Qwen3-8B at TP=2 and TP=4 (identical tokens and logprobs between CPU reload
and GPU prefix reuse) but are not covered by CI. `MultiConnector`, custom
connectors, and standalone `generate.py` are unsupported. Disk offload remains
unvalidated.

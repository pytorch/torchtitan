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
from torchtitan.rl.routing.inter_generator import InterGeneratorRouter

engine = VLLMEngine.Config(
    kv_transfer_config=KVTransferConfig(
        kv_connector="SimpleCPUOffloadConnector",
        kv_role="kv_both",
        kv_connector_extra_config={"cpu_bytes_to_use_per_rank": 1 << 30},
    ),
)
generator_router = InterGeneratorRouter.Config(hot_swap=False)
```

Capacity is **per GPU worker**: TP=2 uses 2 GiB here. Leave room for weight
buffers and other host processes. Prefix caching is required. Default eager
offload saves eligible prompt and decode blocks, including single-turn requests.

Eight-GPU recipe:
`torchtitan_recipes.tests.rl.alphabet_sort.rl_grpo_qwen3_5_debug_varlen_cpu_offload`.

## Weight and cache lifecycle

- Load trainer weights before the first request. Titan salts cache keys with
  the loaded policy version to prevent stale KV reuse after weight updates.
  This applies to both GPU and CPU cache: after an update, requests cannot reuse
  the group's older KV. Without offload, Titan permits stale GPU KV within a rollout group.
- Keep `generator_router.hot_swap=False` and
  `generator.reset_kv_cache_on_model_state_dict_load=False`. Weight updates drain
  requests; the engine also finishes pending transfers after responses return.

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

Session releases notify each generator's rank 0, which broadcasts the release
to its workers. With the holder disabled, session release only removes the routing pin.

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

Tests cover CPU restoration, session release, policy isolation, TP/DP/EP, and
hybrid tails. The RL multi-GPU CI job runs TP=2/DP=2/EP=4 and hybrid TP=2 cases.
`MultiConnector`, custom connectors, and standalone `generate.py`
are unsupported. Disk offload and CUDA graphs remain unvalidated.

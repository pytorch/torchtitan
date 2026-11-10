# Distributed CPU KV cache (Mooncake Store)

Titan RL uses vLLM's
[`MooncakeStoreConnector`](https://docs.vllm.ai/en/latest/features/mooncake_store_connector_usage/)
to share prompt and decode KV across generators through Mooncake Store, a
distributed CPU DRAM pool. A turn routed to a different generator than the
previous turn (non-sticky routing, repin after overload) loads its prefix from
the Store instead of recomputing it. When every turn of a session stays on one
generator, the local [SimpleCPU offload](SIMPLE_CPU_OFFLOAD.md) is simpler.

```text
turn 1 on generator A: prefill, save KV to the Store
turn 2 on generator B: look up prefix -> load KV from the Store -> prefill only new tokens
```

## Setup

Install the Mooncake SDK (`mooncake-transfer-engine`) on each generator host,
start a Master (`mooncake_master --rpc_port=50051`), and point
`MOONCAKE_CONFIG_PATH` in every generator process at a Store config, e.g.:

```json
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "master_server_address": "MASTER_HOST:50051",
  "global_segment_size": "16GB",
  "local_buffer_size": "64MB",
  "protocol": "rdma",
  "device_name": "mlx5_0",
  "enable_offload": false
}
```

`global_segment_size` is the CPU DRAM each GPU process contributes. As with
SimpleCPU offload, set `expandable_segments:False` (see
[Setup](SIMPLE_CPU_OFFLOAD.md#setup)).

## Generator configuration

```python
import uuid

from vllm.config import KVTransferConfig

from torchtitan.rl.generator.vllm.engine import VLLMEngine

engine = VLLMEngine.Config(
    kv_transfer_config=KVTransferConfig(
        kv_connector="MooncakeStoreConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "cache_prefix": f"my-run-{uuid.uuid4().hex}",
            "save_decode_cache": True,
        },
    ),
)
```

- `cache_prefix` namespaces every Store key. Use a **new value for each launch**,
  including checkpoint resume: the Store outlives a run and policy versions
  restart, so an old namespace could serve KV computed by other weights. Store
  keys encode a worker's TP rank but not the TP degree, model layout or dtype;
  generators of one run share one engine config, so these match.
- `save_decode_cache=True` also saves generated tokens, so the next turn reuses
  the previous completion.
- Titan derives `lookup_rpc_port` (an IPC socket name) per generator; do not set
  it.

## Cache keys and weight updates

Store keys carry the same salt as the GPU prefix cache and SimpleCPU offload,
with the same weight-update semantics, including hot swap and mixed-weight KV
(see [Weight and cache lifecycle](SIMPLE_CPU_OFFLOAD.md#weight-and-cache-lifecycle)).
Because the Store is shared, a turn reuses another generator's KV only when both
generators pinned the group to the same `min_policy_version`.

## Multi-GPU coordination

Generators with TP or DP > 1 use the shared scheduler from
[SimpleCPU offload](SIMPLE_CPU_OFFLOAD.md#multi-gpu-coordination). The
[Mooncake adapter](mooncake.py) adds: the TP leader's Store lookup is broadcast
to the other TP ranks, a save completes once this TP group's workers report, and
each DP replica starts its own lookup server.

## Compared with local CPU offload

Neither connector knows about sessions: keys are prefix hashes, so turn 2 reuses
turn 1's blocks because its prompt starts with turn 1's prompt and completion.
Session handling lives in Titan:

| | Local CPU offload | Mooncake Store |
| --- | --- | --- |
| Shared across generators | No | Yes (generators on one Master) |
| Sticky routing | Yes | Yes |
| Hold a session's prefix on GPU between turns (`session_kv_holder`) | Yes | Yes |
| Evict a released session's offloaded blocks first | Yes | No: a generator cannot tell whether another still needs a prefix, so Titan never deletes Store keys |
| Blocks saved | With `lazy_offload=True`, only blocks the GPU is about to evict | Every computed block |
| Eviction | CPU LRU plus session release hints | Master watermark eviction (approximate LRU) |

Between turns (e.g. during tool calls) a session's Store blocks compete with
every new write, so a long gap can find the prefix evicted.

## Performance

Mooncake pays off when turns change generator and loading a prefix from the
Store is faster than recomputing it. Under sticky routing (the default) few
turns change generator, so local [CPU offload](SIMPLE_CPU_OFFLOAD.md) reuses
more KV.

### Agent RL training (sticky routing)

Terminal-Bench: Qwen3.5-27B, 64K context, sticky routing, 2
generator hosts with 8 generators x 1 GB300 (TP=1, `gpu_memory_limit=0.30`) or
4 generators x 2 GB300 (TP=2, `gpu_memory_limit=0.21`). Mooncake over RDMA (4
NICs per host), 40 GB per worker, one Master per host; local CPU offload with
`lazy_offload=True`, 40 GiB per rank. The three jobs of each TP started
together; means over training steps 11-24. Each cell shows TP=1 / TP=2:

| Connector | KV hit rate, TP=1 / TP=2 | Prefill ms, TP=1 / TP=2 | Inter-token latency ms, TP=1 / TP=2 |
| --- | ---: | ---: | ---: |
| No offload | 8.2% / 4.0% | 1,076 / 830 | 48.0 / 59.8 |
| Mooncake | 35.0% / 36.3% | 670 / 497 | 38.5 / 41.1 |
| Local CPU offload | 68.3% / 71.1% | 398 / 289 | 30.2 / 30.4 |

Mooncake saves every computed block, so the Store turns over in about half a
minute instead of keeping the blocks the GPU is about to evict. Training step
time differences were within step-to-step noise.

### Generator microbenchmark (every turn on a different generator)

A separate, generation-only benchmark of the case Mooncake targets: no
training, and every turn of a session is forced onto a different generator, so
local offload never helps. Qwen3-8B, 4 generators on one 8x H100 host, all
Store traffic through one 200 Gb/s NIC. Each session starts from an 8K or 32K
token context and has 4 turns; turn t of session s runs on generator
(s + t) % 4 and adds 192 input and 32 output tokens. Time to run 24 sessions
(96 requests):

| Context | TP per generator | No offload (s) | Mooncake (s) | Speedup |
| --- | --- | ---: | ---: | ---: |
| 32K | 1 | 41.3 | 30.7 | 1.35x |
| 8K | 1 | 11.3 | 10.6 | 1.07x |
| 32K | 2 | 24.8 | 30.9 | 0.80x (slower) |

Mooncake helps when recomputing the context costs more than loading its KV:
at 32K and TP=1, a turn saves about 1.8 s of prefill, while the Store lookup
and load add about 0.85 s of queueing. At TP=2 prefill is fast enough that
loading 32K tokens of KV through the shared NIC takes longer.

## Validation

```bash
pytest -q tests/rl/unit_tests/cpu/test_mooncake_coordination.py
```

The RL CI image has no Mooncake SDK, so end-to-end runs with a real Store are
manual. Validated with `mooncake-transfer-engine==0.3.13.post1`.

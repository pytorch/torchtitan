# Model state dict transport

`ModelStateDictTransport` (`model_state_dict_transport.py`) moves the trainer's
model state dict into the generators' prefetched model state dicts with
one-sided RDMA reads over a `torch.distributed._transport` backend
(`backend="mooncake"`, the default, or `"nixl"`). It replaces TorchStore when
`Controller.Config.model_state_dict_transport` is set:

```python
config.model_state_dict_transport = ModelStateDictTransport.Config(backend="nixl")
```

It is a separate path rather than a TorchStore transport: the trainer and
generator shardings are known at init, so it plans every read once and stages
straight into registered memory, with no StorageVolume, key lookup or per-step
metadata exchange. The trainer stages a pinned CPU copy already resharded per
generator rank, so the optimizer can step while generators read and the
trainer keeps no persistent GPU copy.

`None` keeps TorchStore, which is still needed for
`model_state_dict_load_mode="gpu_fetch"`. TODO: delete the TorchStore path once
this one covers `gpu_fetch`.

Mooncake needs `mooncake-transfer-engine-cuda13==0.3.14` (installed by
`scripts/install_rl_env.sh`) and a PyTorch nightly from 2026-11-10 or later
(https://github.com/pytorch/pytorch/pull/203623; older ones can drop a
completed read and time out). The vLLM wheel pinned by
`scripts/install_rl_env.sh` pins torch 2026-11-07, which is too old. TODO:
drop this note once that pin moves past 2026-11-10. Without
https://github.com/pytorch/pytorch/pull/204113, host memory on other NUMA nodes
uses node 0's NICs, which can cut read bandwidth several times. On IPv6-only
hosts set `MC_USE_IPV6=1`; Mooncake's handshake otherwise resolves peer
hostnames over IPv4.

NIXL needs `nixl==1.5.0` and `nixl-cu13==1.5.0` (installed by
`scripts/install_rl_env.sh`) and uses its UCX plugin; select NICs and
transports with the usual UCX variables, e.g. `UCX_NET_DEVICES`, `UCX_TLS` and
`UCX_MAX_RMA_RAILS`.

## Layout

Every generator has the same sharding, so each trainer rank stages one copy of
its shards, laid out per generator rank (g0, g1). A piece is the intersection
of a trainer rank's box of a tensor with a generator rank's box, cast to the
generator's dtype. Generator ranks read their pieces from every trainer rank
straight into their registered prefetch buffers.

Example: one tensor of shape (8, 6), trainer FSDP 4 on dim 0, generator TP 2
on dim 1.

```
        trainer staging buffers (pinned, registered)
        rank 0        rank 1        rank 2        rank 3
       +------+------+------+------+------+------+------+------+
       | g0   | g1   | g0   | g1   | g0   | g1   | g0   | g1   |
       | 2x3  | 2x3  | 2x3  | 2x3  | 2x3  | 2x3  | 2x3  | 2x3  |
       +------+------+------+------+------+------+------+------+
          |      |      |      |      |      |      |      |
          v      |      v      |      v      |      v      |   read
       generator rank 0 (g0): (8, 3), rows 0-1, 2-3, 4-5, 6-7
                 v             v             v             v
       generator rank 1 (g1): (8, 3), rows 0-1, 2-3, 4-5, 6-7
```

Each piece is one contiguous read per index of the dims before the last one it
does not span. Where a trainer shard cuts a generator shard on a dim after the
trainer's first sharded dim, the trainer restages the DTensor with its shards of
later dims moved to that dim, one DTensor redistribute per such tensor per
push: a row-parallel weight (FSDP `Shard(0)`, TP `Shard(1)`) read by generator
ranks split on dim 0 is staged as `(Shard(0), Shard(0))`, one read instead of
one per row. With generator TP on dim 1 matching the trainer's it is staged as
is, also one read. A stacked `[2, F, D]` weight sharded on dim 1 takes 2
reads. When several trainer ranks stage the same piece (HSDP replicas),
generator ranks are spread across them round-robin. Init raises if the trainer
does not cover every element of a generator shard, or on `Partial` and uneven
`_StridedShard` placements.

## Lifecycle

1. Init (`ActorSystem.initialize_model_state_dict_transport`): every generator
   reports its ranks' `shard_layout`s, which must match
   (`shared_rank_layouts`); each trainer rank allocates and registers its
   staging buffer (`register_trainer_source`); every generator rank registers
   its prefetch buffers, connects to every trainer rank and plans its reads
   (`connect_generator`).
2. Push: the trainer casts and, where needed, restages its shards and copies
   them into the staging buffer (`stage`). Restaging is a collective, so it
   runs on the actor thread before the push's first `await`. The controller
   sends the push before the next forward/backward and Monarch starts them in
   that order, so every rank issues it before its training collectives.
3. Pull: each generator rank issues its reads (`read`) and waits for them,
   raising after `read_timeout_s` (600 s). A timeout leaves the reads in flight;
   restart the job.

The controller finishes pull N before push N+1, so a single staging buffer is
enough.

## Deltas

With `delta_max_fraction` set, a pull reads only the elements whose bits
changed since the previous push:

```python
config.model_state_dict_transport = ModelStateDictTransport.Config(
    backend="nixl", delta_max_fraction=0.1
)
```

Each staged piece gets a delta row after the pieces in the staging buffer:
an int64 `[count, version]` header, then up to `capacity` values and their
int32 flat indices in the piece, `capacity` being `delta_max_fraction` of the
piece's elements. On each push, `stage` copies the piece's previous version
from the staging buffer back to the device, in chunks, compares it bitwise
with the new one, and writes the changed elements into the row with
`nonzero_static`, without host syncs, before overwriting the piece. `count` is
the number of changed elements, even past `capacity`.

A generator rank reads its rows first, each from the trainer rank it reads the
piece from. For each piece, if its prefetch buffers hold the previous version
and `count` is at most `capacity`, it scatters the row's values into them;
otherwise it reads the piece in full. The first pull, and the one after a
skipped version, reads every piece in full. Bitwise comparison makes every
pull exact, including `-0.0` and NaN payloads.

Each generator rank logs, per pull, the fraction of elements that changed and
the largest fraction in one piece; set `delta_max_fraction` just above the
latter. The cost, on both sides, is pinned memory of about
`delta_max_fraction * (1 + 4 / itemsize)` of the staging buffer. On the
trainer, each push also copies every piece host to device on the push stream
and holds a device copy of one row at a time. A pull reads the rows plus the
pieces that changed more than `capacity`.

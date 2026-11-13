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

Each piece is one read, into the generator rank's prefetch buffer of that key
at consecutive offsets in piece offset order. If every piece of a key is one
contiguous run of the generator shard, as above, that is where it belongs.
Otherwise the key's prefetch buffer holds its pieces packed, and the load
copies them into place: e.g. a row-parallel weight (FSDP `Shard(0)`, TP
`Shard(1)`) read by generator ranks split on dim 0 arrives in pieces that span
only part of each row. This keeps one read per piece, with no per-row reads
and no trainer collective, at the cost of a strided copy on load. When several
trainer ranks stage the same piece (HSDP replicas),
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
2. Push: the trainer casts its shards and copies them into the staging buffer
   (`stage`), with no collectives.
3. Pull: each generator rank issues its reads (`read`) and waits for them,
   raising after `read_timeout_s` (600 s). A timeout leaves the reads in flight;
   restart the job.
4. Load: the generator copies its weights from the prefetch buffers, unpacking
   packed keys (`copy_prefetched_`).

The controller finishes pull N before push N+1, so a single staging buffer is
enough.

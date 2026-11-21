# Model state dict transport

`ModelStateDictTransport` (`model_state_dict_transport.py`) moves the trainer's
model state dict into the generators' prefetched model state dicts with
one-sided RDMA reads over a `torch.distributed._transport` backend
(`backend="nixl"`, the default, or `"mooncake"`). It is the default
`Controller.Config.model_state_dict_transport`; select Mooncake with:

```python
config.model_state_dict_transport = ModelStateDictTransport.Config(backend="mooncake")
```

This preserves full synchronization. Set `sync_mode="delta"` to transfer only
the words changed since the preceding push after the first pull.

It is a separate path rather than a TorchStore transport: the trainer and
generator shardings are known at init, so it plans every read once and stages
straight into registered memory, with no StorageVolume or key lookup. The
trainer stages a pinned CPU copy already resharded per generator rank, so the
optimizer can step while generators read and the trainer keeps no persistent
GPU copy.

## Delta sync

At RL learning rates an optimizer step changes few bf16 values (about 2%), but
they are spread over every tensor. Delta sync sends only the changed 2-byte
words, as NeMo RL's sparse refit does: the gap before each changed word and the
word XOR its old value, compressed with zstd. A 1-ulp change leaves the high
bits of the XOR zero, so it compresses better than the new value. Each staged
piece's changes are one record, addressed in the piece a generator rank already
reads, as other systems address changes per tensor.

```text
Trainer rank, per 64 MiB window of the staging buffer

 staging buffer (last push)            this push (GPU)
+------------+---------------+   +------------+---------------+
| k1: 7 3 9  | k2: 4 8 1     |   | k1: 7 5 9  | k2: 4 2 1     |
+------------+---------------+   +------------+---------------+
                  |  compare (GPU), per window
                  v
  positions [1, 4]  XORs [3^5, 8^2]
                  |
                  v  split by piece, gaps, zstd (CPU threads): one record per piece
  k1: gaps [1]  XORs [3^5]     k2: gaps [1]  XORs [8^2]
                  |
                  v
  delta buffer: [k1's record | k2's record | ...]
                  |
                  v
  Controller forwards a DeltaRecord per changed piece (offset, size, gap width)
                  |
                  v
Generator
  prefetch: RDMA the records of its pieces -> unzstd (CPU threads)
            -> each FSDP group's changes packed into one pinned buffer
                  |
                  v
  load, per FSDP group:
    prefetch buffer (last push) -- copy to GPU, as for full sync --> weights
    group's changes -> GPU; positions = running sum of (gap + 1), per piece
    weights[positions] ^= XORs  -> k1: 7 5 9   k2: 4 2 1
                  |
                  v  after the load, in the background (CPU threads)
  prefetch buffer[positions] ^= XORs
```

The load applies the changes on the GPU to the weights it copies anyway, so a
pull moves only the compressed changes and the load's own copy; the prefetch
buffer catches up off the critical path, before the next read.

The trainer compares the staging buffer in 64 MiB windows, which bounds its GPU
memory, and caps the changes it sends at 1/16 of each window's words, shared by
the pieces in it. Unchanged pieces are not sent. The first push, a generator
that missed a push, and a push with more than 1/16 of a window's words changed
are read whole.

`None` uses TorchStore instead, which is still needed for
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
hostnames over IPv4. Without RDMA devices Mooncake falls back to its NVLink
transport, which cannot read pinned CPU memory, and reads fail with
`Mooncake transfer submission failed`; set `MC_FORCE_TCP=1`.

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
   (`connect_generator`). Delta sync also registers each trainer rank's delta
   buffer and each generator rank's receive buffer.
2. Push: the trainer casts its shards and copies them into the staging buffer
   (`stage`), with no collectives. Delta sync first compares each window with
   the staging buffer on the GPU and stages the changed words' positions and
   XORs, which `finish_stage` compresses on CPU threads into one record per
   piece; the push returns a `DeltaRecord` per changed piece (offset, size, gap
   width), which the controller forwards to the generators.
3. Pull: each generator rank issues its reads (`read`) and waits for them,
   raising after `read_timeout_s` (600 s). A timeout leaves the reads in flight;
   restart the job. Delta sync reads the staging buffer whole on its first pull,
   then only the records of its pieces (one read per run of adjacent records),
   decompresses them on CPU threads and packs each FSDP group's changes into one
   pinned buffer.
4. Load: the generator copies its weights from the prefetch buffers, unpacking
   packed keys (`copy_prefetched_`). After a delta pull it copies each FSDP
   group's changes to the GPU, computes their positions there, and XORs them into
   the copies; then CPU threads XOR them into the prefetch buffers in the
   background, which the next pull waits for.

The controller finishes pull N before push N+1, so a single staging buffer is
enough.

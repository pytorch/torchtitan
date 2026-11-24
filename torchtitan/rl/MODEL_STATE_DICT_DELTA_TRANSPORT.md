# Model state dict delta transport

`ModelStateDictDeltaTransport` builds on `ModelStateDictTransport`
(`MODEL_STATE_DICT_TRANSPORT.md`): after the first pull, each pull reads only
the words that changed since the preceding push. Select it by placing
`ModelStateDictDeltaTransport.Config` where `ModelStateDictTransport.Config`
goes.

## Windows and pieces

The staging buffer is `ModelStateDictTransport`'s: each `StagedPiece` at its
`byte_offset`, padded to 16 bytes. `_delta_windows` cuts it into windows;
`_windows_of` lists the windows a piece overlaps. A window can hold several
pieces, and a piece can span windows.

```text
 staging buffer (32 words = 64 bytes; word = 2 bytes)

 word:   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31
        [ piece A "wq"     |pad  | piece B "wk"                                |pad        | C "norm"  |pad        ]
        |<------------------ _Window 0 (lo=0, hi=32 bytes) ----->|<----------- _Window 1 (lo=32, hi=64) ---------->|

 trainer rank's local tensors (bf16): "wq" (2, 3), "wk" (2, 6), "norm" (8,)

 StagedPiece  key     index                        byte_offset  bytes  generator rank   _windows_of
 A            "wq"    (slice(0, 2), slice(0, 3))   0            12     0                [0]
 B            "wk"    (slice(0, 2), slice(0, 6))   16           24     0                [0, 1]
 C            "norm"  (slice(4, 8),)               48           8      1                [1]

 _Window 0: lo=0,  hi=32, pieces=[A, B]
 _Window 1: lo=32, hi=64, pieces=[B, C]
```

`_Window.pieces` entries are `(key, index in the trainer's local tensor,
byte_offset, bytes)`.

## Trainer: compare a window

`_TrainerDelta.stage` runs on the GPU stream, with no host synchronization. Per
window: read the window's previous contents back from the staging buffer
(`old`), assemble this push's window (`new`), stage the changes, then write
`new` back.

Comparing one window at a time bounds the trainer's GPU memory: the temporaries
are about 180 MiB per 64 MiB window, whatever the model size, instead of two
copies of the whole staging buffer (about 2 x 12.5 GiB per trainer rank on
Qwen3.5-27B).

```text
 per window, on the GPU (reused across windows):
   old      = staging window -> GPU        64 MiB
   new      = old + this push's pieces     64 MiB
   changed  = new != old                   32 MiB   (bool per word)
   positions = nonzero_static(changed)     16 MiB   (int64, capacity 2M)
   XORs                                     4 MiB
```

```text
 _Window 0, push N+1                              (int16 words of the window)

 window word:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
 old:          a  b  c  d  e  f  .  .  g  h  i  j  k  l  m  n      <- staging buffer (push N)
 new:          a  B  c  d  E  f  .  .  g  h  I  j  k  l  m  n      <- shards (push N+1)
 changed:         ^        ^                 ^

 _diff_window(new, old):
   changed   = new != old
   count     = changed.sum()                            -> _counts[0] = 3
   positions = nonzero_static(changed, size=capacity)   -> [1, 4, 10, 0]   (padded to capacity 4)
   xors      = new[positions] ^ old[positions]          -> [b^B, e^E, i^I, 0]
   copy (non_blocking) to the pinned _Window.positions and _Window.xors

 _Window 1, push N+1:
 window word:  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
 changed:         ^                       ^
               (= staging word 17, in B)  (= staging word 25, in C)
   _counts[1] = 2,  positions = [1, 9, 0, 0],  xors = [p^P, q^Q, 0, 0]
```

The padding words between pieces come from `old`, so they never change. The
first push only assembles and writes back: there is nothing to compare yet.

## Trainer: one record per piece

`_TrainerDelta.compress_records` (called by `prepare_transfer`) runs on CPU
threads, one piece per task. It finds each piece's part of each window it
overlaps with `np.searchsorted`, makes positions relative to the piece, turns
them into gaps, and compresses one record.

A position is a changed word's index; a gap is the number of unchanged words
before it, since the previous changed word (or since the start). Positions
grow with the buffer, so they need 4 bytes; gaps stay small, so they usually
fit in 2 bytes and compress well.

```text
 piece B words:  0  1  2  3  4  5  6  7  8  9  10 11
 changed:              ^                    ^
                 |----|  |---------------|
                 2 unchanged  6 unchanged

 positions [2, 9]    index of each changed word
 gaps      [2, 6]    unchanged words before each changed word
 position = previous position + gap + 1     2 = -1 + 2 + 1,   9 = 2 + 6 + 1
```

```text
 piece A (words 0..5 of the staging buffer), window 0:
   window positions [1, 4, 10]  (first _counts[0] = 3)
   A's words in window 0: [0, 6)  -> searchsorted -> [1, 4]
   relative to A:  [1, 4]

 piece B (words 8..19), spans windows 0 and 1:
   window 0: words [8, 16)  -> [10]   -> relative to B: 10 - 8       = [2]
   window 1: words [0, 4)   -> [1]    -> relative to B: 1 + (16 - 8)  = [9]
   concatenated: [2, 9]

 piece C (words 24..27), window 1:
   window words [8, 12)     -> [9]    -> relative to C: 9 - 8 = [1]

 gaps = positions - previous position - 1   (np.diff(positions, prepend=-1) - 1)

            positions   gaps     XORs
 A          [1, 4]      [1, 2]   [b^B, e^E]
 B          [2, 9]      [2, 6]   [i^I, p^P]
 C          [1]         [1]      [q^Q]
```

A record is one zstd frame of `[gaps | XORs]`. Gaps are uint16, or uint32 if a
gap reaches 65536 (`gap_itemsize`). A 1-ulp change leaves the high bits of the
XOR zero, so XORs compress better than new values.

```text
 record of B (before zstd):
 +-------------+-------------+-------------+-------------+
 | gap 2 (u16) | gap 6 (u16) | i^I (i16)   | p^P (i16)   |
 +-------------+-------------+-------------+-------------+
   gap_itemsize x count        2 B x count

 delta buffer: records of changed pieces, back to back in staging order
 offset:  0          nA              nA+nB
          [ rec A    | rec B         | rec C   |        unused       ]

 ModelStateDictDeltaTransferMetadata(
     delta_records={
          0: DeltaRecord(offset=0,     nbytes=nA, gap_itemsize=2),   # key: StagedPiece.byte_offset
         16: DeltaRecord(offset=nA,    nbytes=nB, gap_itemsize=2),
         48: DeltaRecord(offset=nA+nB, nbytes=nC, gap_itemsize=2),
     },
     push=N+1,
 )
```

A piece with no changes has no record and no `delta_records` entry. The frame
stores its own size, so a generator knows the change count after
decompressing: `count = size / (gap_itemsize + 2)`.

## When a push is read whole

A generator reads every piece from the staging buffer, as full sync does, when:

- it is the first push, or the first read since `connect_generator`;
- the generator missed a push (`push` is not its last push + 1);
- a window had more than 1/16 of its words changed (`Config.capacity_divisor`,
  16 by default; `delta_records=None`);
- the records did not fit the trainer's delta buffer (`delta_records=None`);
- this generator rank's records from a trainer rank do not fit its share of the
  receive buffer (this rank only).


## Generator: which records to read

`connect_generator` turns each of the rank's `Read`s (one
per piece) into a `_DeltaPlan` entry per trainer rank:

```text
 Read(source=0, byte_offset=0,  nbytes=12, key="wq", local_byte_offset=0)
 Read(source=0, byte_offset=16, nbytes=24, key="wk", local_byte_offset=64)

 _DeltaPlan (trainer rank 0), generator rank 0:
   records          = {0: Read(... key="wq" ...), 16: Read(... key="wk" ...)}
                      # trainer staging byte_offset -> the piece's Read
                      # "wk": local_byte_offset 64 = word 32; its first 64 bytes come from trainer rank 1
   delta_buffer     = DeltaTrainerSource.delta_remote_buffer
   receive_offset   = where trainer rank 0's records go in the receive buffer
   receive_capacity = _delta_capacity_nbytes(sizes of A, B)
```

`plan_pull` keeps the records the rank needs and reads each run of adjacent
records with one RDMA read:

```text
 metadata.delta_records keys:   0 (A)     16 (B)     48 (C)
 _DeltaPlan.records keys:       0         16                    -> needed: A, B

 delta buffer (trainer rank 0)  [ rec A | rec B | rec C ]
                                 \_____________/
                                  one run -> one RDMA read
                                       |
                                       v
 receive buffer                 [ rec A | rec B |   ...   ]
                                ^ receive_offset

 generator rank 1 needs only C  -> one read of [ rec C ]

 a needed record next to a skipped one splits the run:
 delta buffer                   [ rec A | rec X | rec B ]   (X: another rank's piece)
                                  read 1          read 2
```

## Generator: decompress and group

After the reads land, `decode` decompresses every record on CPU threads and
packs the changes per load group: the keys one `copy_prefetched_` call of a
load gets (`load_groups`, e.g. one FSDP group).

```text
 receive buffer [ rec A | rec B ]
                    |       |   _decompress_record (zstd, CPU threads)
                    v       v
          ("wq", 0):  gaps [1, 2]   XORs [b^B, e^E]
          ("wk", 32): gaps [2, 6]   XORs [i^I, p^P]
                    |
                    |   load_groups = [["wq", "wk"], ...]  -> both in group 0
                    v
 _Group.pack  (pinned buffers, one per group)
   gaps     [1, 2, 2, 6]                  # int32, every changed piece of the group
   xors     [b^B, e^E, i^I, p^P]          # int16
   starts   [0, 2]                        # index in gaps of each piece's first change
   shifts   [0, 27]                       # see next section
   keys     {"wq": (0, 2), "wk": (2, 4)}  # each key's slice of gaps / xors

 _GeneratorDelta._pending = {"wq": group 0, "wk": group 0}
```

## The running sum: gaps to positions

A load copies a whole group's changes to the GPU once and turns gaps into
positions with one `cumsum`. Each piece restarts at its own word offset in its
key: `shifts` adds the jump from where the previous piece ended.

```text
 position of change i = position of change i-1 + gap_i + 1
                      = cumsum(gap + 1) - 1, restarted at each piece's offset

 pack():  previous_end = 0
   piece A: offset 0,  gaps [1, 2]
            shift = offset - previous_end = 0 - 0 = 0
            previous_end = offset + sum(gaps) + count = 0 + 3 + 2 = 5
   piece B: offset 32, gaps [2, 6]
            shift = 32 - 5 = 27
            previous_end = 32 + 8 + 2 = 42

 positions_on(device):                      index:   0     1     2     3
   gaps                                             [1,    2,    2,    6]
   + 1                                              [2,    3,    3,    7]
   index_add_(starts=[0, 2], shifts=[0, 27])        [2,    3,   30,    7]
   cumsum                                           [2,    5,   35,   42]
   - 1                                              [1,    4,   34,   41]
                                                     \____/     \____/
                                                    "wq" 1, 4  "wk" 34, 41
 check: "wk" = 32 + [2, 9] = [34, 41]
```

## Load: apply the changes

`ModelStateDictLoader.load` calls `copy_prefetched_` once per load group, on
each FSDP group's sharded bf16 parameters, before FSDP all-gathers (and
quantizes) them.

```text
 copy_prefetched_({"wq": param, "wk": param, ...})        one load group
   |
   +-- take_group(keys) -> the group (moves to _loaded)
   +-- _Group.positions_on(GPU): one copy, one running sum
   |     -> "wq": positions [1, 4],   XORs [b^B, e^E]
   |        "wk": positions [34, 41], XORs [i^I, p^P]
   |
   +-- "wq", plain tensor, prefetch buffer has its layout:
   |     param.copy_(prefetch["wq"])                 H2D, as full sync
   |     param.view(int16)[[1, 4]] ^= [b^B, e^E]     in place on the GPU
   |
   +-- "wk", packed key, or a tensor subclass (e.g. quantized parameter):
   |     tmp = prefetch["wk"].to(GPU)                contiguous copy
   |     tmp.view(int16)[[34, 41]] ^= [i^I, p^P]
   |     param.copy_(tmp)   or unpack tmp into param (ModelStateDictTransport._packed)
   |
   +-- keys with no changes: ModelStateDictTransport.copy_prefetched_
   |
   +-- start_write_back_after_load(): nothing pending -> _write_back per loaded group

 ModelStateDictLoader.load, per group:
   reshard -> copy_prefetched_ (bf16 = push N+1) -> FSDP all-gather (+ quantize) -> forward
```

## Prefetch buffer write-back

After the load, the GPU copies hold push N+1 but the prefetch buffers still
hold push N. CPU threads XOR the same changes into them in the background
(`_write_back`, `positions_on(cpu)`), while the generator serves.

```text
 time ------------------------------------------------------------------------------------------>

 pull N+1                     |  generating with push N+1             | pull N+2
 read: RDMA, decode | load    |                                       | read
                    |  _write_back x groups (CPU threads) ------------+ |
                    |                                                 | |
                    |  prefetch buffers: push N ...... push N+1 <-----+ |
                                                                        |
 read N+2 starts:                                                       v
   delta read:  RDMA + decode (only the receive buffer)  -> overlaps the write-backs
                wait_write_backs()                       -> before the load copies prefetch
   whole read:  wait_write_backs() first                 -> the read overwrites prefetch
```

## Metadata path

```text
 Trainer.push_model_state_dict
   transport.stage(shards)                 -> staging buffer, _Window.positions/xors
   transport.prepare_transfer()            -> ModelStateDictDeltaTransferMetadata
        |                                     (one per trainer rank)
        v
 controller: metadata_by_trainer_rank = (rank 0, rank 1, ...)
        |
        v
 router.pull_model_state_dict(version, metadata_by_trainer_rank)
        |
        v
 Generator.prefetch_model_state_dict(metadata_by_trainer_rank)
   transport.read(metadata_by_trainer_rank)
     all ranks on the same push?         (assert)
     push == last push + 1, all delta_records set?
       yes -> plan_pull -> RDMA -> decode -> wait_write_backs
       no  -> wait_write_backs -> whole read (ModelStateDictTransport.read)
```

The controller forwards the metadata unchanged. With full sync, every push's
metadata is `ModelStateDictTransferMetadata()` and every read is whole.

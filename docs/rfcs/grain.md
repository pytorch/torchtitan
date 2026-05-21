# RFC: Replace HF Datasets / StatefulDataLoader with a Grain-based Pipeline

## Summary

torchtitan's data configs cover source selection well — dataset name, path,
per-source weights, seed — but processing behavior is not configurable at all:
the text field, tokenization options, packing algorithm, and any filtering are
hardcoded inside the dataset classes. Changing any of them means editing Python. This RFC replaces torchtitan's current data stack with a Grain-based pipeline where both source selection
and processing behavior are expressed as composable `Op`s in a dataclass
config. The old stack is deleted entirely.

## Motivation

Four concrete problems drive this change.

**Processing behavior is not configurable.** `HuggingFaceTextDataLoader.Config`
and `InterleavedHuggingFaceTextDataLoader.Config` expose source selection
(dataset name, path, weights, seed) but nothing about how records are
processed. The text field is hardcoded to `sample["text"]`, `add_bos` and
`add_eos` are always `True`, the packing algorithm is fixed per class, and
there is no concept of filtering. `ChatDataLoader.Config.sample_processor` is
`tyro.conf.Suppress` — it cannot be set from CLI or a config file at all.
Adding any processing capability (a length filter, a different tokenization
strategy, a new packing algorithm) means editing the dataset classes or adding
a new one.

**Text and chat paths duplicate the same skeleton independently.** Both
`HuggingFaceTextDataset.__iter__` and `ChatDataset._iter_greedy_packed` share
the same structure: iterate over data, tokenize, accumulate into a buffer, slice
when full, handle infinite re-loop with per-epoch shuffle. Each maintains its
own state dict with different keys, its own backward-compat fallbacks (e.g. the
`positions_buffer` warning in `HuggingFaceTextDataset.load_state_dict`), and its
own bugs.

**Resume logic is genuinely complex.** `HuggingFaceTextDataset` and
`ChatDataset` each carry separate code paths for map-style vs. iterable
datasets, with manual `set_epoch` synchronization on restore. `ChatDataset`
adds `_pending_input_ids`/`_pending_label_ids` to survive the edge case where a
sample straddles a packing boundary at checkpoint time. This complexity is
inherent to hand-rolling stateful iteration on top of HuggingFace's own
iterator state.

## The Core Idea

Grain's API is already a good data pipeline description language. Here is a
complete multi-source pretrain pipeline in raw Grain:

```python
def make_corpus(file_glob, seed):
    src = grain.sources.ArrayRecordDataSource(sorted(glob.glob(file_glob)))
    ds  = grain.MapDataset.source(src).seed(seed).shuffle().repeat()
    ds  = ds.map(parse_and_tokenize)
    return ds

web   = make_corpus("/prep_ar/web/*.array_record",   seed=11)
code  = make_corpus("/prep_ar/code/*.array_record",  seed=22)
books = make_corpus("/prep_ar/books/*.array_record", seed=33)

mixed = grain.MapDataset.mix([web, code, books], weights=[0.70, 0.20, 0.10])
mixed = mixed[dp_rank :: dp_world_size]
it    = mixed.to_iter_dataset(grain.ReadOptions(num_threads=8))
it    = FirstFitPackIterDataset(it, length_struct={"input_ids": seq_len}, num_packing_bins=64)
it    = it.batch(local_batch_size, drop_remainder=True)
it    = it.mp_prefetch(grain.MultiprocessingOptions(num_workers=16))
```

This code already tells the whole story. The problem is that it is not
configurable: `parse_and_tokenize`, the packing bin count, the weights, the
seed — all hardcoded. Adding a source or swapping the packing algorithm means
editing this function. Sharing it across training recipes means copy-paste.

The proposed design makes each step a named, typed `Op` in a dataclass config.
`make_corpus` becomes `Source(pipeline=[ArrayRecordReader(...), Shuffle(),
Repeat(), TextTokenize()])`. The post-mix chain becomes
`pipeline=[FirstFitPack(num_packing_bins=64), Batch(), Prefetch(num_workers=16)]`.
The mix, DP slice, and Map→Iter crossover are injected automatically by the
orchestrator — they are framework concerns, not user concerns. Every parameter
that was hardcoded is now a config field, overridable from CLI.

## Proposed Design

The design has three interlocking concepts. An **`Op`** is the unit of
composition — a dataclass that wraps one Grain operation and declares two
independent contracts: a **Stage** contract (`accepts`/`produces`) tracking
the shape and readiness of the records flowing through it, and a **Phase**
contract tracking the Grain backing it requires (`MapDataset` vs
`IterDataset`). The two axes are independent by design: Stage catches
record-shape errors like tokenize-after-pack at config-parse time; Phase drives
the Map→Iter crossover automatically at build time. A **Pipeline** is an
ordered list of Ops with those contracts validated end-to-end: each source
runs its own pipeline up to the mixing boundary, the framework mixes the
sources, and a shared post-mix pipeline carries the mixed stream through to a
training batch.

### The stage ladder

Every record flowing through the pipeline is in one of five stages:

```
SOURCE  →  RAW  →  ENCODED  →  TRAINSAMPLE  →  TRAINBATCH
```

This tracks *granularity and readiness*, not modality:

| Stage | Meaning |
|---|---|
| `SOURCE` | Empty sentinel before any data exists. Only a reader op accepts it. |
| `RAW` | One undecoded record as storage emits it — a dict of bytes or strings. |
| `ENCODED` | Processed model-input features for one source item. Variable shape, any modality. think tokens for text |
| `TRAINSAMPLE` | One training-ready item conformed to the model's per-step shape (`seq_len` for text tokens). |
| `TRAINBATCH` | Stacked tensors in the `(input_dict, labels)` shape the trainer consumes. |

The pipeline validates this sequence at `__post_init__`. A missing reader, a
reader in the wrong position, or tokenize-after-pack all surface as
`ValueError`s before any data is touched.

### The `Op` abstraction

`Op` is the single extension point. Every op declares two independent contracts
as class-level metadata:

```python
class Op(Configurable.Config):
    accepts:  ClassVar[frozenset[Stage] | Literal["any"]]      # record shape accepted
    produces: ClassVar[Stage | Literal["same"]]                 # record shape emitted
    phase:    ClassVar[Literal["map", "iter", "any"]] = "any"  # grain backing required

    def apply(self, ds, ctx: PipelineContext): ...
```

`accepts`/`produces` is the **stage contract** — it catches record-shape errors
like tokenize-after-pack. `phase` is the **grain backing contract** — it tells
the orchestrator whether an op requires a `MapDataset`, an `IterDataset`, or
works on either. The two axes are genuinely independent: `Prefetch` is
stage-agnostic (`"any"`/`"same"`) but `"iter"`-only, because `mp_prefetch`
does not exist on a `MapDataset`.

A concrete op is typically a few lines:

```python
@dataclass
class Repeat(Op):
    accepts, produces = "any", "same"
    num_epochs: int | None = None   # None → infinite

    def apply(self, ds, ctx):
        return ds.repeat(self.num_epochs)
```

`Op.__init_subclass__` auto-registers every subclass by class name. The
registry is what will allow a YAML loader (see [What This Unlocks](#what-this-unlocks))
to construct an arbitrary pipeline from a file with no additional wiring.

### Pipeline config shape

Two dataclasses describe a complete data run:

```python
@dataclass
class Source:
    pipeline: list   # [Reader, ...ops...]; SOURCE → ENCODED
    weight: float = 1.0

@dataclass
class DataPipeline.Config:
    sources: list    # list[Source]
    pipeline: list   # list[Op]; ENCODED → TRAINBATCH
    seed: int = 42
```

A `Source` is a reader-headed op list plus its mixing weight. The post-mix
`pipeline` runs over the combined stream after all sources are mixed. Both are
validated at `__post_init__`: per-source pipelines must run `SOURCE → ENCODED`;
the post-mix pipeline must run `ENCODED → TRAINBATCH`.

The fields are typed as bare `list` — not `list[Op]` — matching torchtitan's
existing tyro pattern for heterogeneous lists. Pipeline structure is built
before tyro runs (in `config_registry.py` or from a future YAML file), and
tyro layers leaf overrides on top. CLI overrides stay clean:

```
--sources.0.weight=0.65 --pipeline.0.num_packing_bins=128
```

### Framework-internal concerns

These are handled inside `DataPipeline.Config.build(ctx)` and never appear in
user configs.

**Source tagging.** Each record gets `{"source_idx": i}` injected immediately
after its reader, so per-source observability is available throughout the
entire pipeline.

**DP sharding.** After mixing, `mixed[dp_rank :: dp_world_size]` is applied
before the Map→Iter crossover. Sharding is a free index-slice on the mixed
`MapDataset` — no extra state, perfectly balanced.

**Map→Iter crossover.** The orchestrator tracks the current grain backing. The
first time it meets an `"iter"` op while still on a `MapDataset`, it injects
`.to_iter_dataset(...)` automatically. A `"map"` op encountered after the
crossover raises. `"any"` ops before the crossover stay on `MapDataset`, which
is cheaper.

**Mixing.** `MapDataset.mix(sources, weights, seed)` is structural — it is
many-to-one and does not fit the `list[Op]` model, so it is not an op.

**Checkpointing.** Delegated entirely to Grain's `IterDataset` state dict.

The runtime object is a `DataPipeline` shim (~15 lines) that wraps the
terminal `IterDataset`, exposes `__iter__` and `Stateful`, and refuses to
load a checkpoint whose `dp_world_size` differs from the current run. Grain
does not support resharding, and silently restoring the wrong iterator state
would be worse than a clean failure.

### Text instantiation

Text is the simplest path through the ladder:

- **`ENCODED`:** `{input_ids, positions}` — one tokenized document, variable
  length. No labels yet.
- **`TRAINSAMPLE`:** those fields packed or padded to `seq_len + 1`. After
  `NextTokenShift`: `{input_ids, positions, label_ids}`.
- **`TRAINBATCH`:** those fields stacked into `(input_dict, labels)`.

Labels are derived late, by `NextTokenShift` (`label_ids = input_ids[1:]`),
because plain causal-LM labels are a pure mechanical shift — there is nothing
to compute at encode time.

The concrete text ops are: `ArrayRecordReader`, `Shuffle`, `Repeat`,
`TextTokenize`, `MinLengthFilter`, `FirstFitPack`, `GreedyPack`,
`NextTokenShift`, `Batch`, `Prefetch`.

Canonical pretrain pipeline:

```python
DataPipeline.Config(
    sources=[Source(pipeline=[ArrayRecordReader(...), Shuffle(), Repeat(), TextTokenize()])],
    pipeline=[FirstFitPack(num_packing_bins=64), NextTokenShift(), Batch(), Prefetch(num_workers=4)],
)
```

See the [Appendix](#appendix-code-sketch) for multi-source, SFT, and
validation examples.

### Multimodal instantiation

Vision-text reuses the same five-stage ladder unchanged. Two things differ.

**Labels are produced at encode time, not shifted later.** For vision-text,
`ENCODED` carries `{input_ids, positions, pixel_values, label_ids}`. Masking
vision and placeholder tokens requires knowing which tokens they are — masking
is intrinsic to encoding, not a late mechanical shift. Vision-text therefore
**skips `NextTokenShift`**.

**Two ops specialize:**

- `VisionTextEncode` (`RAW → ENCODED`): decodes images, inserts
  vision-placeholder tokens, tokenizes, and masks vision-token labels in one
  step. Processing parameters (patch size, image normalization, etc.) live on
  the op, not on `ctx` — context stays lean.
- `MMPack` (`ENCODED → TRAINSAMPLE`): payload-aware bin-packing that keeps
  each sample's `pixel_values` attached.
- `MMBatch` (`TRAINSAMPLE → TRAINBATCH`): a collator, not a stack — it
  patchifies images, pads ragged image counts, builds `grid_thw`, and caps
  images per batch.

**Payloads survive shared ops transparently.** `MapOp` re-attaches a reserved
set of framework keys (`_PASSTHROUGH = {"source_idx"}`) after every
`transform`, so framework metadata can never be accidentally dropped. For
payload fields like `pixel_values`, the contract is simpler: they survive
because no op between producer and consumer rewrites the record. A custom op
that must carry a payload is responsible for doing so explicitly.

The full `MMBatch.combine` internals are deferred to the multimodal PR.

### Customization

The standard ops cover the common cases, but the design is not a closed box.
Four family bases make custom ops easy to slot in:

- **`MapOp`** — 1:1 record transform. Implement `transform(sample, ctx) → dict`;
  the base handles `ds.map(...)` wiring and re-attaches framework metadata.
- **`FilterOp`** — record predicate. Implement `keep(sample, ctx) → bool`.
- **`PackOp`** — length-normalize `ENCODED → TRAINSAMPLE`. Implement `apply(ds, ctx)`.
- **`BatchOp`** — collate `TRAINSAMPLE → TRAINBATCH`. Implement `combine(samples, ctx) → dict`;
  the base handles batching and the universal `(input_dict, labels)` split.

For cases that don't fit any family — a new modality, bespoke Grain code, a
stage jump — subclass `Op` directly and write raw Grain in `apply`. The minimal
custom pipeline is two ops:

```python
class MyEncode(MapOp):            # RAW → ENCODED: custom processing
    accepts, produces = frozenset({Stage.RAW}), Stage.ENCODED
    def transform(self, sample, ctx): ...

class MyPost(Op):                 # ENCODED → TRAINBATCH in one op
    accepts, produces, phase = frozenset({Stage.ENCODED}), Stage.TRAINBATCH, "iter"
    def apply(self, ds, ctx): ...

DataPipeline.Config(
    sources=[Source(pipeline=[ArrayRecordReader(file_glob=...), MyEncode()])],
    pipeline=[MyPost()],
)
```

Mixing, DP sharding, the Map→Iter crossover, and checkpointing are all
injected by the framework around your ops — you write only the parts that are actually custom.

For the ultimate escape hatch — an exotic topology, a custom sharding strategy, iterable sources that don't fit the Op model at all — subclass `DataPipeline.Config` and redefine `build`. The trainer calls `config.data.build(ctx)` and gets a `DataPipeline` back; it does not care how it was constructed:

```python
@dataclass
class MyPipeline(DataPipeline.Config):
    my_param: int = 32

    def build(self, ctx: PipelineContext) -> DataPipeline:
        # full raw Grain — any topology, any source format
        src = grain.sources.ArrayRecordDataSource(...)
        ds  = grain.MapDataset.source(src).seed(self.seed).shuffle().repeat()
        ds  = ds.map(lambda s: my_custom_processing(s, ctx, self.my_param))
        ds  = ds[ctx.dp_rank :: ctx.dp_world_size]
        it  = ds.to_iter_dataset(grain.ReadOptions(num_threads=8))
        it  = my_custom_pack(it, ctx.seq_len)
        it  = it.batch(ctx.local_batch_size, drop_remainder=True)
        it  = it.mp_prefetch(grain.MultiprocessingOptions(num_workers=8))
        return DataPipeline(it, dp_world_size=ctx.dp_world_size)
```

The only contract: `build` must return a `DataPipeline` — the thin shim
described under *Framework-internal concerns* above, which wraps a terminal `IterDataset`, exposes `__iter__` and `Stateful`, and guards against resharding on restore. The full implementation is in the Appendix. Stage validation, the Op registry, and framework concerns are all bypassed — full power, full responsibility.

## Key Design Decisions

**MapDataset-only sources.** All sources are expressed as `grain.MapDataset`
— indexable and deterministic. Streaming-only formats go through an offline
preprocessing step that emits ArrayRecord shards. The payoff: global shuffle,
exact DP sharding, and exact resume. An opt-in iterable-sources mode for
streaming sources is supported by the phase system, with documented tradeoffs
(see [What This Unlocks](#what-this-unlocks)).

**Mix at the MapDataset level.** Mixing happens before the Map→Iter crossover,
so DP sharding is a free index-slice and the mixed stream is globally shuffled.
Iterable-level mixing is the opt-in alternative for streaming sources.

**No dataset/dataloader split.** The old boundary was an artifact of the torch
DataLoader contract (`Dataset` produces samples, `DataLoader` batches and
prefetches). In Grain the pipeline is both. The split no longer reflects a real
architectural boundary.

**Full deletion, no wrapping.** A compatibility shim would preserve the config
duplication problem this design exists to solve. The old classes are deleted;
behaviors are reimplemented as ops only where they fit the pipeline model.

## What Gets Deleted / What Changes

**Files deleted:**

- `torchtitan/hf_datasets/*`
- `torchtitan/components/dataloader.py`

**Knobs that go away:**

| Old knob | Replacement |
|---|---|
| `num_workers`, `persistent_workers`, `pin_memory`, `prefetch_factor` | `Prefetch(num_workers=..., prefetch_buffer_size=...)` op |
| `snapshot_every_n_steps` | Grain snapshots are cheap; no cadence knob needed |
| `infinite` | Presence or absence of `Repeat()` |
| `stopping_strategy` | Implicit; `Repeat()` on every source gives an infinite mix |
| `set_epoch` | Not a concept in Grain; reshuffles come from `Shuffle` + the pipeline seed |

**Trainer delta:**

1. Rename `dataloader → data` in the config.
2. Construct a `PipelineContext` (tokenizer, seq_len, dp_rank, dp_world_size, local_batch_size).
3. Call `config.data.build(ctx)` instead of constructing a `StatefulDataLoader`.

The iteration loop and checkpointer registration are unchanged — the
`DataPipeline` shim presents the same `Iterable + Stateful` contract.

## What This Unlocks

### Iterable-sources mode

The `phase` system already makes a fully-iterable pipeline expressible. If a
reader returns an `IterDataset`, the Map→Iter crossover is never triggered and
every op applies directly. Three framework concerns adapt:

- **Mixing** dispatches to `IterDataset.mix`. All sources must share the same
  backing — a mix of `MapDataset` and `IterDataset` sources is a config error.
- **DP sharding** becomes a stateful strided op: each rank keeps element `i`
  where `i % dp_world_size == dp_rank`, rather than an index-slice.
- **Shuffle** degrades to `WindowShuffle` — an approximate, buffer-based
  shuffle instead of a global permutation.

The cost: each rank reads the full stream and discards `(N-1)/N` of it
(read amplification), and shuffle quality is approximate. The gain: streaming
sources without an offline preprocessing step.

This mode is strictly opt-in. Its tradeoffs are documented at the config site.
It is never a silent fallback.

### File-based pipeline YAML

Three properties of this design make a YAML loader a pure addition — no
changes to existing code needed when it arrives:

1. The op-name registry (auto-populated by `Op.__init_subclass__`) maps class
   names to constructors.
2. `to_dict` emits `op = type(self).__name__` per step, so any config is
   serializable and round-trippable.
3. Bare-`list` field typing already accepts externally-constructed entries.

A future loader reads a YAML file, looks up each `op` key in the registry,
pops it, and instantiates the dataclass — running `__post_init__` stage
validation on the way. A malformed file-defined pipeline fails the same way a
code-defined one does. The target shape:

```yaml
data:
  seed: 42
  sources:
    - weight: 0.7
      pipeline:
        - op: ArrayRecordReader
          file_glob: /data/web/*.array_record
        - op: Shuffle
        - op: Repeat
        - op: TextTokenize
          text_field: text
    - weight: 0.3
      pipeline:
        - op: ArrayRecordReader
          file_glob: /data/code/*.array_record
        - op: Shuffle
        - op: Repeat
        - op: TextTokenize
          text_field: content
          add_eos: false
  pipeline:
    - op: FirstFitPack
      num_packing_bins: 128
    - op: NextTokenShift
    - op: Batch
    - op: Prefetch
      num_workers: 16
```

## Migration Plan

**PR 1 — This RFC.** Design alignment. No code changes.

**PR 2 — Core implementation.** The following land together in a single PR:
- ArrayRecord preprocessing tool (HF / Parquet / JSON → ArrayRecord shards)
- Core framework: `Stage`, `Op` family bases, `DataPipeline`, text ops
  (`TextTokenize`, `ChatTokenize`, `FirstFitPack`, `GreedyPack`,
  `NextTokenShift`, `Batch`, `Prefetch`)
- Trainer wiring: rename `dataloader → data`, construct `PipelineContext`,
  call `config.data.build(ctx)`
- Delete `torchtitan/hf_datasets/` and `torchtitan/components/dataloader.py`
- Update `config_registry.py` for all models

Bundling the preprocessing tool and the deletion means there is no gap where
the old stack is gone but users have no path to prepare their data.

**PR 3 — Multimodal.** `VisionTextEncode`, `MMPack`, `MMBatch`, and
payload-aware packing. Depends on PR 2.

**PR 4 — Iterable-sources mode.** The opt-in streaming path: `IterDataset`
mixing, strided DP sharding, `WindowShuffle`. Depends on PR 2.

**PR 5 (tentative) — File-based YAML loader.** The op-name registry and
`to_dict` contract are committed in PR 2; the loader itself and the
file→config merge precedence are deferred to this PR.

## Open Questions

**1. Should PRs 2–5 follow the sequencing above, or is a different ordering preferred?**

The plan above bundles the preprocessing tool with the deletion (PR 2) to
avoid a gap where the old stack is gone but users have no path to prepare
their data. If the preprocessing tool needs more design time, an alternative
is to land the framework and text ops first while keeping the old stack, then
delete once the tool ships.

**2. Document weights vs. token weights.**

`weight` in `Source` controls how often a *document* is drawn from each
source, not the realized token proportion. Because documents vary in length,
the actual token mix will diverge from the declared weights.

Token-true weighting is achievable: pack each source to `seq_len` before
mixing, so every drawn sample represents exactly `seq_len` tokens. But packing
is an `"iter"`-phase op — it requires crossing to `IterDataset` first. Crossing
per-source, before the mix, forfeits `MapDataset.mix`: mixing must then happen
at the `IterDataset` level, losing the global shuffle and the free index-slice
DP sharding that the base design relies on. Is document-frequency weighting
acceptable as the default? If so, should the config emit an explicit warning?

**3. Full deletion vs. a deprecation period.**

There is no migration path for existing data configs. Names like
`HuggingFaceTextDataset`, `ChatDataset`, and `StatefulDataLoader` disappear.
Is a clean break the right call?

---

## Appendix: Code Sketch

Shape only — not production code. Intended to show what an op looks like and
how the orchestrator wires the pieces together.

### Examples

#### Single-source pretrain

```python
DataPipeline.Config(
    sources=[
        Source(
            pipeline=[
                ArrayRecordReader(file_glob="./data/c4/*.array_record"),
                Shuffle(),
                Repeat(),           # infinite
                TextTokenize(text_field="text"),
            ],
        ),
    ],
    pipeline=[
        FirstFitPack(num_packing_bins=64),
        NextTokenShift(),
        Batch(),
        Prefetch(num_workers=4),
    ],
)
```

#### Multi-source weighted mix

```python
DataPipeline.Config(
    sources=[
        Source(
            pipeline=[ArrayRecordReader(file_glob="/data/web/*.array_record"),
                      Shuffle(), Repeat(), TextTokenize()],
            weight=0.70,
        ),
        Source(
            pipeline=[ArrayRecordReader(file_glob="/data/code/*.array_record"),
                      Shuffle(), Repeat(), TextTokenize(text_field="content", add_eos=False)],
            weight=0.20,
        ),
        Source(
            pipeline=[ArrayRecordReader(file_glob="/data/books/*.array_record"),
                      Shuffle(), Repeat(), TextTokenize()],
            weight=0.10,
        ),
    ],
    pipeline=[
        FirstFitPack(num_packing_bins=128),
        NextTokenShift(),
        Batch(),
        Prefetch(num_workers=16),
    ],
    seed=42,
)
```

#### SFT

```python
def process_sample(sample):
    return [
        {"role": "user",      "content": sample["question"]},
        {"role": "assistant", "content": sample["answer"]},
    ]

DataPipeline.Config(
    sources=[
        Source(
            pipeline=[
                ArrayRecordReader(file_glob="./data/sft/*.array_record"),
                Shuffle(),
                Repeat(),
                ChatTokenize(sample_processor=process_sample),
            ],
        ),
    ],
    pipeline=[
        GreedyPack(),           # sequential fill + EOS pad
        NextTokenShift(),
        Batch(),
        Prefetch(num_workers=4),
    ],
)
```

#### Validation (finite stream)

```python
DataPipeline.Config(
    sources=[
        Source(
            pipeline=[
                ArrayRecordReader(file_glob="/data/c4_val/*.array_record"),
                TextTokenize(),
            ],
        ),
    ],
    pipeline=[FirstFitPack(), NextTokenShift(), Batch(), Prefetch(num_workers=2)],
)
```

---

### Full sketch

```python
# ── generic core (modality-agnostic) ─────────────────────────────────────────

class Stage(Enum):
    SOURCE      = "source"
    RAW         = "raw"
    ENCODED     = "encoded"
    TRAINSAMPLE = "trainsample"
    TRAINBATCH  = "trainbatch"


class Op(Configurable.Config):
    accepts:  ClassVar[frozenset[Stage] | Literal["any"]]
    produces: ClassVar[Stage | Literal["same"]]
    phase:    ClassVar[Literal["map", "iter", "any"]] = "any"

    def apply(self, ds, ctx: PipelineContext): ...


# Framework metadata re-attached by MapOp after every transform.
_PASSTHROUGH = frozenset({"source_idx"})


class MapOp(Op):
    """1:1 record transform. Subclass implements transform(); base re-attaches
    _PASSTHROUGH so transform() can return a clean dict."""
    def apply(self, ds, ctx):
        def _fn(s):
            out = self.transform(s, ctx)
            for k in _PASSTHROUGH:
                if k in s and k not in out:
                    out[k] = s[k]
            return out
        return ds.map(_fn)

    def transform(self, sample, ctx): ...


class FilterOp(Op):
    accepts, produces = "any", "same"

    def apply(self, ds, ctx):
        return ds.filter(lambda s: self.keep(s, ctx))

    def keep(self, sample, ctx) -> bool: ...


class PackOp(Op):
    """ENCODED → TRAINSAMPLE. Subclasses implement the packing algorithm."""
    accepts, produces, phase = frozenset({Stage.ENCODED}), Stage.TRAINSAMPLE, "iter"


class BatchOp(Op):
    """TRAINSAMPLE → TRAINBATCH. Subclasses implement combine(); base owns the
    grain wiring and the universal (input_dict, labels) split."""
    accepts, produces, phase = frozenset({Stage.TRAINSAMPLE}), Stage.TRAINBATCH, "iter"
    drop_remainder: bool = True

    def apply(self, ds, ctx):
        return ds.batch(
            ctx.local_batch_size,
            drop_remainder=self.drop_remainder,
            batch_fn=lambda g: self._to_train_batch(self.combine(g, ctx)),
        )

    def combine(self, samples, ctx) -> dict: ...  # modality-specific merge

    @staticmethod
    def _to_train_batch(batched: dict):
        labels = batched.pop("label_ids")
        inputs = {("input" if k == "input_ids" else k): v for k, v in batched.items()}
        return inputs, labels


class Reader(Op):
    """Root op: no input, produces RAW, creates the MapDataset."""
    accepts, produces, phase = frozenset({Stage.SOURCE}), Stage.RAW, "map"


# ── structural ops ────────────────────────────────────────────────────────────

@dataclass
class Shuffle(Op):
    accepts, produces, phase = "any", "same", "map"  # global shuffle is MapDataset-only

    def apply(self, ds, ctx):
        return ds.shuffle()


@dataclass
class Repeat(Op):
    accepts, produces = "any", "same"
    num_epochs: int | None = None

    def apply(self, ds, ctx):
        return ds.repeat(self.num_epochs)


@dataclass
class Prefetch(Op):
    accepts, produces, phase = "any", "same", "iter"  # mp_prefetch is IterDataset-only
    num_workers: int = 0
    prefetch_buffer_size: int = 64

    def apply(self, ds, ctx):
        return ds.mp_prefetch(
            grain.MultiprocessingOptions(num_workers=self.num_workers),
            buffer_size=self.prefetch_buffer_size,
        )


@dataclass
class ArrayRecordReader(Reader):
    file_glob: str

    def apply(self, ds, ctx):  # ds is None — reader is the root
        return grain.MapDataset.source(
            grain.sources.ArrayRecordDataSource(sorted(glob.glob(self.file_glob)))
        )


@dataclass
class Batch(BatchOp):
    def combine(self, samples, ctx):
        return {k: stack([s[k] for s in samples]) for k in samples[0]}


# ── text ops ──────────────────────────────────────────────────────────────────

@dataclass
class TextTokenize(MapOp):
    accepts, produces = frozenset({Stage.RAW}), Stage.ENCODED
    text_field: str = "text"
    add_bos: bool = True
    add_eos: bool = True

    def transform(self, sample, ctx):
        ids = ctx.tokenizer.encode(
            sample[self.text_field], add_bos=self.add_bos, add_eos=self.add_eos
        )
        return {"input_ids": ids, "positions": list(range(len(ids)))}


@dataclass
class NextTokenShift(MapOp):
    accepts, produces = frozenset({Stage.TRAINSAMPLE}), "same"

    def transform(self, sample, ctx):
        return {
            "input_ids": sample["input_ids"][:-1],
            "positions": sample["positions"][:-1],
            "label_ids": sample["input_ids"][1:],
        }


@dataclass
class MinLengthFilter(FilterOp):
    min_tokens: int = 1

    def keep(self, sample, ctx) -> bool:
        return len(sample["input_ids"]) >= self.min_tokens


@dataclass
class FirstFitPack(PackOp):
    num_packing_bins: int = 64

    def apply(self, ds, ctx):
        return grain.experimental.FirstFitPackIterDataset(
            ds,
            length_struct={"input_ids": ctx.seq_len + 1, "positions": ctx.seq_len + 1},
            num_packing_bins=self.num_packing_bins,
        )


@dataclass
class GreedyPack(PackOp):
    """Sequential fill until the next sample won't fit, then EOS-pad and start
    fresh. Default for SFT."""

    def apply(self, ds, ctx):
        return GreedyPackIterDataset(ds, length=ctx.seq_len + 1, pad_id=ctx.tokenizer.eos_id)


# ── multimodal ops (signatures only; internals in the MM PR) ──────────────────

@dataclass
class VisionTextEncode(MapOp):
    accepts, produces = frozenset({Stage.RAW}), Stage.ENCODED
    patch_size: int
    spatial_merge_size: int
    min_pixels: int
    max_pixels: int
    image_mean: tuple[float, ...]
    image_std: tuple[float, ...]

    def transform(self, sample, ctx):
        # decode images → insert vision placeholders → tokenize → mask vision labels
        return {"input_ids": ..., "positions": ..., "label_ids": ..., "pixel_values": ...}


@dataclass
class MMPack(PackOp):
    def apply(self, ds, ctx):
        return MMSamplePackIterDataset(ds, length=ctx.seq_len + 1)


@dataclass
class MMBatch(BatchOp):
    max_images_per_batch: int = 0

    def combine(self, samples, ctx) -> dict:
        ...  # patchify, pad ragged images, grid_thw, cap images, special_tokens


# ── orchestrator ──────────────────────────────────────────────────────────────
@dataclass
class PipelineContext:
    tokenizer: BaseTokenizer
    seq_len: int
    dp_rank: int
    dp_world_size: int
    local_batch_size: int

@dataclass
class DataPipeline:

    @dataclass
    class Config:
        sources: list = field(default_factory=list)  # list[Source]
        pipeline: list = field(default_factory=list)  # list[Op]
        seed: int = 42

        def __post_init__(self):
            for src in self.sources:
                _validate_stages(src.pipeline, start=Stage.SOURCE, end=Stage.ENCODED,
                                 where="source")
            _validate_stages(self.pipeline, start=Stage.ENCODED, end=Stage.TRAINBATCH,
                             where="post-mix")

        def build(self, ctx: PipelineContext) -> "DataPipeline":
            per_source = []
            for idx, src in enumerate(self.sources):
                reader, *rest = src.pipeline
                ds = reader.apply(None, ctx)
                ds = ds.seed(self.seed + idx)
                ds = ds.map(lambda s, i=idx: {**s, "source_idx": i})
                for op in rest:
                    ds = op.apply(ds, ctx)
                per_source.append(ds)

            mixed = grain.MapDataset.mix(per_source, weights=[s.weight for s in self.sources])
            mixed = mixed[ctx.dp_rank :: ctx.dp_world_size]

            ds = mixed
            for op in self.pipeline:
                if op.phase == "iter" and isinstance(ds, grain.MapDataset):
                    ds = ds.to_iter_dataset(grain.ReadOptions(num_threads=8))
                elif op.phase == "map" and isinstance(ds, grain.IterDataset):
                    raise ValueError(
                        f"{type(op).__name__} requires MapDataset but the pipeline "
                        "already crossed to IterDataset"
                    )
                ds = op.apply(ds, ctx)

            return DataPipeline(ds, dp_world_size=ctx.dp_world_size)

    def __init__(self, ds, dp_world_size: int):
        self._ds = ds
        self._dp_world_size = dp_world_size

    def __iter__(self):
        return iter(self._ds)

    def state_dict(self):
        return {"ds": self._ds.state_dict(), "dp_world_size": self._dp_world_size}

    def load_state_dict(self, sd):
        if sd["dp_world_size"] != self._dp_world_size:
            raise ValueError("dp_world_size changed; resharding is not supported")
        self._ds.load_state_dict(sd["ds"])
```

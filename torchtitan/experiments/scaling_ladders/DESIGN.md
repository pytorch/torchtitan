# Llama3 Scaling Ladder -- Design (as built)

This documents the as-built design of the scaling-ladder experiment. For the
high-level overview, the OLMo-core <-> TorchTitan bridge table, the rung family,
and the baseline extrapolation result, see [`README.md`](README.md). The policy
math and WSD-S curve are a faithful port of OLMo-core (see **References**).

## Architecture: resolve once, carry one spec

`Llama3Ladder` is the single source of truth for rungs, policy, compute spec, and
build knobs (attention backend, gradient-reduce dtype, fp8, compile). The
read/plan side runs in-process; the run side goes through **one** launcher that
resolves the schedule once and hands the worker a config file. The dry-run plan,
the launched run, and the read-back metrics therefore cannot disagree -- not
because every layer re-derives the same thing, but because the resolved plan is
computed once and carried.

```text
                       +------------------+
                       |   Llama3Ladder   |   config + resolution (no GPU)
                       | rungs + policy + |   _resolve / trainer_config / run_dir
                       | compute + knobs  |
                       +--------+---------+
   plan/status/metrics/compare  |  build_spec
        (in-process, on meta)   |
        +-----------------------+---------------------------+
        |                       |                           |
config_registry.py        launcher.run_jobs             cli.py / run_perf_campaign
(nullary recipes for      (resolve once -> spec.json    (thin adapters: build Jobs,
 run_train.sh, native)     -> torchrun --config-file     emit JSON / drive campaigns)
                            -> train.run_from_spec)
```

- **Read/plan side is in-process, no distributed.** `plan`, `status`, `metrics`,
  and `compare` only build models on `meta` (for param counts) and read files.
- **Run side is one launcher.** `launcher.run_jobs(ladder, jobs, total_gpus=N)`
  bin-packs jobs onto the node (sequential is just `run_rung`, i.e. one job at
  `total_gpus = its width`); each job is its own `torchrun` group with a private
  rendezvous and masked `CUDA_VISIBLE_DEVICES`. The launcher resolves the
  schedule **once** (`build_spec`), writes the plan plus every build knob to
  `{run_dir}/launch_spec.json`, and spawns `torchrun ... train --config-file
  PATH`. The worker (`train.py` -> `run_from_spec`) loads that spec and builds the
  `Trainer.Config` directly -- config never round-trips through argv, so a new
  knob is one spec key, not a flag threaded through five files.
- **Resilience lives in the one launcher.** OOM -> shrink `local_batch_size`
  (schedule-invariant) and retry; a transient streaming failure (e.g. an HF C4
  shard fetch) -> retry at the same batch; an already-complete run (final
  checkpoint present) -> skip. All callers (cli `run`/`sweep --execute`,
  `showcase.run_rungs`, `run_perf_campaign`) inherit this from the single path.
- **Default recipe** is reachable the native way, with no launcher or plumbing:
  `MODULE=scaling_ladders CONFIG=llama3_ladder_100m ./run_train.sh`, where the
  config function is `return LADDER.trainer_config("100M")`.

## TorchTitan integration constraints

The bridge is a set of thin adapters over these TorchTitan facts:

- The top-level config is `Trainer.Config` (no `JobConfig`); the model-registration
  object is `ModelSpec`. `ConfigManager` resolves `--module/--config` by importing
  `torchtitan.experiments.<m>.config_registry` and calling the nullary function;
  registration is one entry in `_supported_experiments`
  (`torchtitan/experiments/__init__.py`) -- the only core touch.
- Training is strictly **step-based** (`step < training.steps`); there is no
  train-by-tokens mode, so the planner converts every token budget to steps.
- `training.global_batch_size` is a **sequence** count; the trainer asserts
  `global_batch_size % (local_batch_size * dp_degree) == 0` and derives
  gradient accumulation from it.
- Custom scheduler / checkpoint / validator plug in **without core edits** via the
  `Configurable._owner` mechanism: a nested `Config` subclass is auto-wired so
  `config.X.build(...)` constructs the owning subclass.
- `ParallelDims.from_config(parallelism, world_size)` is pure arithmetic (resolves
  `dp_shard=-1`, no process group), so the planner computes the data-parallel
  degree at config-build time.
- `Validator.should_validate` is freq-modulo only; metrics are TensorBoard scalars
  (`loss_metrics/global_avg_loss`, `grad_norm`, `validation_metrics/loss`) read
  back with `EventAccumulator` (as `scripts/loss_compare.py` does).

## Package layout

```text
torchtitan/experiments/scaling_ladders/
  README.md             # overview + baseline + perf findings (+ assets/*.png)
  DESIGN.md             # this document
  __init__.py           # exports Llama3Ladder, the default LADDER, model_registry
  ladder.py             # Llama3Ladder: config + resolution; plan/status/metrics/sweep/compare
  policy.py             # WSD-S / Chinchilla policy (OVERRIDABLE_FIELDS + formulas + periods)
  planner.py            # policy + compute spec -> Trainer.Config and plan dict
  model.py              # experiment-local Llama3 rungs + model_registry + fp8_converter
  lr_scheduler.py       # WSDSScheduler(LRSchedulersContainer)
  checkpoint.py         # LadderCheckpointManager(CheckpointManager), explicit steps
  validate.py           # LadderValidator(Validator), fires at fixed steps
  metrics.py            # TensorBoard read-back -> per-checkpoint records + throughput
  config_registry.py    # nullary default recipes + debug config (native run_train.sh)
  launcher.py           # ONE scheduler: build_spec / run_jobs / run_rung / run_from_spec
  train.py              # torchrun worker entry: --config-file -> run_from_spec
  cli.py                # thin CLI over Llama3Ladder + launcher
  showcase.py           # loss-vs-compute fit + variant/perf comparison + plotting
  run_perf_campaign.py  # iso-quality throughput campaign driver (--variant flex_flash|fp8|...)
```

`launcher.py` holds all launch concerns (one scheduler, one cmd builder, the
OOM/transient backoff, the spec serialization); `ladder.py` no longer spawns
anything. `train.py` is ~15 lines.

Constructing the module-level `LADDER` stays cheap: rungs are built on `meta`
lazily (on first `plan`/`trainer_config`), not at import time.

## Scaling policy (`policy.py`)

`WSDSChinchillaPolicy` defaults: `tokens_per_param=20`, `chinchilla_multiple=4`,
`decay_fraction=0.1`, `lr_multiplier=1.0`, `weight_decay=0.1`,
`stepped_schedule=False`, `seq_len=4096`. Formulas, ported verbatim
(`N = ladder_params`):

```text
target_token_batch = round(2048 * 160 * (N / 108_000_000) ** (2 / 3))
train_tokens       = chinchilla_multiple * tokens_per_param * N
warmup_tokens      = N
peak_lr            = 0.0047 * (N / 108_000_000) ** (-1 / 3) / 2 * lr_multiplier
beta2              = 0.95 if actual_token_batch >= 524288 else 0.99
```

Notes carried in code:

- The `2048` in `target_token_batch` is OLMo's anchor sequence length baked into
  the constant; it is **not** the ladder `seq_len`. The result is tokens; convert
  to sequences with `/ seq_len`.
- The `/ 2` on `peak_lr` is OLMo-core's empirical halving (near-optimal for the
  stepped WSD-S schedule).
- `beta2` keys off the **actual** (rounded) token batch, not the target.
- WSD-S periods for `chinchilla_multiple = c`: `[2**p for p in range(-1,
  log2(c)+1)]` (e.g. `[0.5, 1, 2, 4]`). The stepped-LR variant
  (`period_lr_multipliers = 1/sqrt(c)`) is supported but defaults off.
- Validations: `chinchilla_multiple` a power of 2 and `>= 0.5`;
  `0 < decay_fraction < 0.5`; and the planner guards `warmup + first-period decay
  <= first-period length`.

**Divergence from OLMo-core:** OLMo uses `SkipStepAdamW` (skips loss/grad-spike
steps); this experiment uses plain `AdamW` built via an `OptimizersContainer.Config`
with two `ParamGroupConfig`s -- a `tok_embeddings` group with `weight_decay=0.0`
and a catch-all (`.*`) group with the policy `weight_decay` -- matching OLMo's
embedding-no-decay group override.

## Rungs and parameter accounting (`model.py`)

Rungs reuse `Llama3Model`, `Llama3TransformerBlock`, the public common builders
(`make_gqa_config`, `make_ffn_config`, `get_attention_config`,
`Embedding/RMSNorm/Linear/ComplexRoPE` configs), `parallelize_llama`,
`pipeline_llm`, and `Llama3StateDictAdapter`. Only the small per-layer init dicts
are reproduced locally. `model_registry(rung) -> ModelSpec` mirrors
`models/llama3/__init__.py`.

The eight rungs (60M .. 8B) are listed with their `(dim, layers, heads, kv heads,
hidden dim, ladder_params)` in [`README.md`](README.md). Key choices:

- All rungs use **untied embeddings** and Llama3 vocab `128256`;
  `head_dim = dim / n_heads`. Untying is load-bearing: it keeps `lm_head` in the
  count, matching OLMo's untied ladder and the transfer of its fitted constants.
- `ladder_params = total_params - vocab_size * dim` (OLMo's
  `num_non_embedding_params`), built on `meta`; the planner audits it and warns if
  it drifts >5% from the nominal label (all rungs are within ~2%).
- Per-rung parallelism and `local_batch_size` are derived from each rung's memory
  footprint by `planner.auto_compute_spec` (no hardcoded table): a rung that fits
  on one GPU runs as DDP replicas (cheap all-reduce), larger rungs shard (FSDP)
  only as much as needed; `local_batch_size` is sized to the largest microbatch
  whose activations (estimated by `model.activation_gib_per_seq`) fit beside the
  resident model+optimizer state, capped at ~1 gradient-accumulation step -- so the
  run fits on the first try and the launcher's OOM backoff is only a backstop for
  residual estimate error (without it, e.g. 8B started at an unfittable lbs and
  walked the probe down through many expensive recompiles). All of this is
  throughput-only -- the global batch, token budget, and loss are unchanged.

## Planner (`planner.py`)

Both `plan()` (dict) and `trainer_config()` (`Trainer.Config`) call `resolve()`:

1. `ladder_params N` from `count_ladder_params` (meta build, audited).
2. `dp_degree = dp_replicate * dp_shard` from `ParallelDims.from_config`.
3. `global_batch_size` (sequences) = target token batch / seq_len, rounded to a
   multiple of `local_batch_size * dp_degree`; `actual_token_batch =
   global_batch_size * seq_len` (warn if it deviates >10% from target).
4. `peak_lr`, `beta2` from policy (using `actual_token_batch`).
5. One step-rounded period table drives **both** the scheduler periods and the
   checkpoint steps, so they stay aligned: cumulative Chinchilla period ends are
   computed in steps; `period_lengths` are their incremental spans; `steps =
   last cumulative end`; pre-decay step `= period_end - round(decay_fraction *
   period_length)`; post-decay step `= period_end`. The pre-decay checkpoint thus
   lands exactly where the scheduler begins decaying.
6. Emit `Trainer.Config`: ladder `ModelSpec`; `TrainingConfig`; `parallelism`;
   the two-group AdamW optimizer; `WSDSScheduler.Config`;
   `LadderCheckpointManager.Config` (explicit steps, `keep_latest_k=0`);
   `LadderValidator.Config` (`fixed_steps` = post-decay steps);
   `ChunkedCELoss`; TensorBoard on; `CompileConfig(enable=...)`;
   `mixed_precision_reduce` (gradient all-reduce dtype) and optional fp8
   converters; dataset + `hf_assets_path`; `dump_folder` (the run's unique dir);
   `seed`.

## WSD-S scheduler (`lr_scheduler.py`)

`WSDSScheduler(LRSchedulersContainer)` with a `Config` adding `period_lengths`
(in steps), `decay_fraction`, and optional `period_lr_multipliers`. Its `build()`
returns a `LambdaLR` whose multiplier reproduces OLMo-core's `WSDS.get_lr`
exactly -- a continuous sawtooth:

- Warmup once (linear to peak) over `warmup_steps`, subtracted from period 0 only.
- Within each period: hold peak, then linearly decay to 0 over
  `round(decay_fraction * period_length)` steps.
- At each period boundary the LR jumps straight back to peak (no re-warmup); with
  `period_lr_multipliers`, peak is scaled per period.

It is a pure function of the global step, so the inherited `state_dict` /
`load_state_dict` (just `last_epoch`) suffice. The LambdaLR `last_epoch` is
0-indexed; the lambda applies the `+1` convention used by torchtitan's default
scheduler so the curve matches OLMo's 1-indexed `get_lr`. A unit test compares
the full curve against an inline copy of OLMo's `get_lr`.

## Checkpointing, validation, metrics

- `LadderCheckpointManager(CheckpointManager)` adds `checkpoint_steps: list[int]`
  and overrides `_should_save` to fire only on those steps (and the final step),
  bypassing the inherited interval modulo. `keep_latest_k=0` retains all rungs'
  pre/post-decay pairs (the WSD-S deliverable).
- `LadderValidator(Validator)` adds `fixed_steps: list[int]` and overrides
  `should_validate` to fire at those steps (the post-decay steps) plus step 1, so
  `validation_metrics/loss` exists at the comparison points. `validator.steps` is
  a fixed positive number (not `-1`) to avoid multi-rank hangs.
- `metrics.py` reads TensorBoard scalars into per-checkpoint records keyed by
  step, with `tokens`, `chinchilla_multiple`, and `phase` attached. `val_loss` is
  exact-at-step (the validator logs at the checkpoint step); `train_loss` and
  `grad_norm` are the nearest *earlier* logged value (train metrics are only
  written every `log_freq` steps). Structured contract:

  ```json
  {"rung": "100M", "ladder_params": 99230208,
   "checkpoints": [
     {"step": 6729, "tokens": 1984462848, "chinchilla_multiple": 1.0,
      "phase": "post-decay", "train_loss": 3.66, "val_loss": 3.66,
      "grad_norm": 0.4}]}
  ```

## Agent-facing API and CLI

`Llama3Ladder` (config + resolution; methods accept the policy overrides in
`policy.OVERRIDABLE_FIELDS` -- `lr_multiplier`, `weight_decay`,
`chinchilla_multiple`, `tokens_per_param`, `decay_fraction` -- plus `seed`):

```text
plan(rung, **overrides)            -> dict   # params, batch, steps, lr, beta2, ckpt_steps, ...
trainer_config(rung, **overrides)  -> Trainer.Config
run_dir(rung, **overrides)         -> str    # unique dump folder for (rung, overrides, seed)
status(rung, **overrides)          -> dict   # ckpt/metric steps present, pct complete
metrics(rung, **overrides)         -> dict   # per-checkpoint structured records
sweep(rungs, grid)                 -> list   # expand to (rung, overrides) specs (pure)
compare(runs, metric, at_xC)       -> dict   # rank a sweep's runs at matched xC
```

The ladder does not launch anything; that is `launcher.py`:

```text
launcher.run_jobs(ladder, jobs, total_gpus=N) -> list[result]  # the one scheduler
launcher.run_rung(ladder, rung, overrides)    -> result        # sequential (N=1)
launcher.build_spec(ladder, job)              -> dict          # resolve once -> spec
launcher.Job(rung, gpus, overrides=, fp8=, attn_backend=, reduce_dtype=, base_dump_folder=)
```

Run identity is `(rung, overrides, seed)`; `run_dir` encodes it into a unique
folder (e.g. `.../100M/wd0.05_seed1/`, `.../60M/cm1.0_seed0/`) so concurrent jobs
never collide and `status`/`metrics`/the persisted `launch_spec.json` resolve the
same folder. Per-job build knobs (`fp8`, `attn_backend`, `reduce_dtype`,
`base_dump_folder`) let one `run_jobs` call mix A/B arms on the node.

`cli.py` read commands emit JSON; all accept the override flags:

```text
dry-run --size 100M / dry-run-all
run --size 100M [overrides]            # launcher.run_rung (blocks)
launch-command --size 100M             # prints the run_train.sh command
status / status-all, metrics / metrics-all
sweep --sizes 60M,100M --grid weight_decay=0.05,0.1,0.2 [--execute]
compare --runs <sweep-output.json> --metric val_loss --at 1xC   # NxC or N
```

`train.py` takes only `--config-file PATH`; all build options live in the spec.

The hillclimbing loop an agent drives: `sweep` the `(rung, override, seed)`
specs -> `launcher.run_jobs` -> poll `status` to completion -> `metrics` parse
the objective at matched `xC` -> `compare` argmin -> escalate the winner up the
ladder. The unique `run_dir`, `status`, and JSON contract make this loop possible
without screen-scraping or run collisions.

## Config registry (`config_registry.py`)

One nullary function per rung (`llama3_ladder_60m` .. `llama3_ladder_8b`), each
`return LADDER.trainer_config("<rung>")`, plus `llama3_ladder_debug`: a tiny rung
on `c4_test` with the test tokenizer, capped to `training.steps=10` and with
checkpointing disabled (DCP cannot build a multi-rank save plan from a single
`fake_backend` process), for fake-backend and smoke tests. Real-rung recipes use
`c4` and a Llama3 tokenizer/assets dir with TensorBoard on.

## Experiments and status (`showcase.py`, `run_perf_campaign.py`)

- **Loss-vs-compute extrapolation (executed).** Fit `L(C) = E + A*(C/c_ref)^-alpha`
  on the small rungs' post-decay points (each `cm=1` run yields 0.5xC and 1xC),
  predict a held-out larger rung, and verify by running it. The baseline run (fit
  60M-370M, held-out 760M) is reported in [`README.md`](README.md): the prediction
  holds to ~1%.
- **Code-variant comparison (Flavor-Q, executed).** `compare_variants` and
  `plot_loss_vs_compute` support an "agent edits code, the ladder judges it" loop:
  a worktree-isolated architecture change is trained on the small rungs and its
  loss-vs-compute curve compared against the *reused* baseline (no retraining of
  the baseline). The first variant (QK-norm) lowered the curve at every measured
  point; the result and figure are in [`README.md`](README.md).
- **Weight-decay hillclimb (designed, not run).** Sweep `weight_decay` on the
  small rungs to matched `xC`, select the argmin, and check transfer up-ladder via
  `sweep`/`compare`. Methodology: always compare at matched `xC`; use multiple
  seeds and a noise threshold; record the resolved plan per run.

### Performance experiments (`run_perf_campaign.py`)

The same harness doubles as an iso-quality throughput A/B: the ladder is the
fitness function, a perf knob is the variable. `run_perf_campaign --variant V`
trains a baseline and a variant arm (per-rung GPU widths, both arms identical so
each rung's pair shares a data-parallel degree), then `showcase.compare_perf`
reads steady-state `throughput(tps)` from TensorBoard and checks the loss curves
coincide. `metrics.read_run_throughput` and `showcase.plot_perf` support it.
Findings on this B200 node (see [`README.md`](README.md) for plots):

- **`flex_flash` attention -- adopted (~1.02-1.15x, grows with size, iso-quality).**
  Switching `attn_backend` from the default Triton `flex` to the FlexAttention
  FLASH kernel is faster at every rung at numerically-identical loss. (Needs
  flash_attn 2.8.4's CUTE kernels; the stock 2.8.3 is incompatible.)
- **bf16 gradient all-reduce -- modest win (~1.03-1.05x on multi-GPU rungs).**
  `reduce_dtype="bfloat16"` halves the gradient-comm bytes; helps only where there
  is an all-reduce (1.00x on 1-GPU rungs). Modest because the all-reduce is mostly
  overlapped with the backward pass.
- **fp8 (rowwise) -- rejected for this ladder (<=760M).** A net loss in every
  config: fp8-everywhere ~0.93x, fp8 on the MLP GEMMs only 0.915x at 760M; the
  GEMMs at dim <=1536 are below the fp8 crossover on B200's fast bf16, and fp8 on
  `lm_head` OOMs (it breaks `ChunkedCELoss`'s logit fusion). `fp8_converter` and
  the wiring remain for larger models. torchao's `auto_filter_small_kn` is
  incompatible with the config-conversion path (it requires real `nn.Linear`).

## Tests

`tests/unit_tests/test_scaling_ladders.py` (CPU-only) covers: parameter accounting
(within 5%, tied not double-counted); policy math cross-checked against an inline
OLMo-core reference; batch rounding validity; the WSD-S LR curve vs OLMo's
`get_lr` across warmup/stable/decay/jump-back, and that `build()` drives a real
`LambdaLR`; checkpoint/validator step gating; `ConfigManager` loading the default
recipe; the `plan`/`compare`/`compare_variants` contracts and run-dir uniqueness;
and the extrapolation fit + prediction on synthetic data.

```bash
pytest tests/unit_tests/test_scaling_ladders.py tests/unit_tests/test_config_manager.py
COMM_MODE=fake_backend NGPU=8 MODULE=scaling_ladders CONFIG=llama3_ladder_debug ./run_train.sh
```

## Divergences and non-goals

- Plain `AdamW`, not `SkipStepAdamW` (skip-step behavior is a possible later add).
- Loss / grad-norm / validation-loss only; no downstream task evaluation.
- No Slurm/Beaker launcher; runs are local `torchrun` subprocesses.
- Rungs stay experiment-local; core `torchtitan.models.llama3` is unchanged.
- No auto-expansion of device count to hit the target batch (the compute spec is
  taken as given, with a launch-time `NGPU == world_size` guard).

## References

- OLMo-core ladder configurator (formulas, periods, checkpoint pairs):
  `OLMo-core/src/olmo_core/model_ladder/wsds_chinchilla_run_configurator.py`
- OLMo-core `WSDS` scheduler (exact LR curve):
  `OLMo-core/src/olmo_core/optim/scheduler.py`
- OLMo-core ladder object + CLI (API shape mirrored):
  `OLMo-core/src/olmo_core/model_ladder/base.py`, `.../internal/ladder.py`
- Batch-size and peak-LR scaling fits: SemanticScholar CorpusID:270764838.
- WSD-S schedule: arXiv:2410.05192.

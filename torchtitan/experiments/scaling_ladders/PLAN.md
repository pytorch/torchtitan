# TorchTitan Llama3 Scaling Ladder Plan

## Summary

Add OLMo-style scaling ladder infrastructure under
`torchtitan/experiments/scaling_ladders/`.

A scaling ladder runs a family of models from small research sizes up to ~8B,
each trained with Chinchilla-matched compute and a WSD-S (warmup-stable-decay,
simplified) learning-rate schedule, so that cheap small-rung experiments can
predict good choices for expensive large-rung runs. This is a faithful port of
OLMo-core's `WSDSChinchillaRunConfigurator` ladder onto TorchTitan's native
training stack (FSDP2/DTensor/DCP, TP/PP/CP). See **References**.

The first showcase family uses TorchTitan's Llama3 decoder as the architecture
template, but defines experiment-local ladder rungs instead of modifying
`torchtitan.models.llama3`.

The design is **Python-API-first**: a single `Llama3Ladder` object is the source
of truth for rungs, policy, and compute spec. The config registry, the CLI, and
the agent-facing run/plan/status/metrics API all derive from it. This makes the
ladder drivable by an automated hillclimbing loop (human or agent), not just by
hand.

Ladder size means non-embedding parameters: `total_params - input_embedding`.
The input token embedding matrix is excluded from the rung label and from
Chinchilla scaling calculations. The output `lm_head` is counted unless it is
tied to the input embeddings. This matches OLMo-core's
`num_non_embedding_params = num_params - embeddings.weight.numel()`.

No Slurm/Beaker support in v1. The plan/status/metrics/compare side runs
in-process (no distributed); each training run is a `torchrun`-spawned subprocess.
Support local dry-run, local launch-command generation, local run, status polling,
and structured metric aggregation.

## Verified Repo Facts

These were re-verified against the current `main`. (The earlier draft of this
plan assumed a flat `dim/n_layers/n_heads` model-args API that no longer exists.)

- There is **no `JobConfig`**: the top-level config dataclass is `Trainer.Config`
  (`torchtitan/trainer.py`). The model-registration object is **`ModelSpec`**
  (`torchtitan/protocols/model_spec.py`), not `TrainSpec`.
- `ConfigManager` resolves `--module <m> --config <c>` (and the `MODULE`/`CONFIG`
  env vars set by `run_train.sh`) by importing
  `torchtitan.experiments.<m>.config_registry` and calling `getattr(module, c)()`
  **with no arguments**. A config function returns a `Trainer.Config` (or a
  subclass). To register, add the experiment name to `_supported_experiments` in
  `torchtitan/experiments/__init__.py` (the one sanctioned core touch).
- `Llama3Model.Config` is a nested `Decoder.Config` (`models/llama3/model.py`),
  assembled by helpers, not flat args. Building a rung means composing
  `Llama3Model.Config(dim, vocab_size, tok_embeddings=Embedding.Config(...),
  norm=RMSNorm.Config(...), lm_head=Linear.Config(...), layers=[...])`, where the
  layers come from `make_gqa_config` + `make_ffn_config` +
  `compute_ffn_hidden_dim` (public, `models/common/config_utils.py` and
  `models/common`) wrapped in `Llama3TransformerBlock.Config`. The per-layer init
  dicts (`_LINEAR_INIT`, `_NORM_INIT`, `_depth_init`, ...) are module-private in
  `models/llama3/__init__.py`; v1 reproduces these ~6 lines experiment-locally to
  avoid coupling to private internals. Existing public sizes are reachable via
  `llama3_configs["1B"]()` etc.
- Parameter counting already exists: `get_dense_model_nparams_and_flops`
  (`models/utils.py`) returns total `nparams` (tie-aware via `parameters()`
  dedup). Reuse it; do not reimplement.
- Custom LR scheduler and checkpoint manager can be plugged in **without core
  edits**: `config.lr_scheduler.build(...)` and `config.checkpoint.build(...)`
  construct whatever class owns the `Config` (the `Configurable._owner`
  mechanism). `TorchFTCheckpointManager` already overrides `_should_save` this
  way, and `GraphTrainer`/`FaultTolerantTrainer` subclass `Trainer`.
- `ParallelDims.from_config(parallelism_cfg, world_size)` is **pure arithmetic**
  (resolves `dp_shard=-1`, asserts the product equals world size) and needs **no
  process group** -- `init_device_mesh` is lazy. So the planner can compute the
  data-parallel degree at config-build time given a world size.
- `training.global_batch_size` is a **sequence** count. The trainer asserts
  `global_batch_size % (local_batch_size * dp_degree) == 0` and derives
  `gradient_accumulation_steps` from it (`trainer.py`). Training is **strictly
  step-based** (`should_continue_training` checks `step < training.steps`); there
  is no train-by-tokens mode, so the planner converts all token budgets to steps.
- Validation exists (`components/validate.py`) but `Validator.should_validate` is
  **freq-modulo only** (`step == 1 or step % freq == 0`). It cannot fire at an
  arbitrary list of steps today; v1 subclasses it to add `fixed_steps` (see
  **Validation at Matched Checkpoints**).
- Metrics are written to TensorBoard under
  `{dump_folder}/{metrics.save_tb_folder}/{timestamp}/` with tags
  `loss_metrics/global_avg_loss`, `grad_norm`, and (when validation runs)
  `validation_metrics/loss`. `scripts/loss_compare.py` already reads these back
  via `EventAccumulator`; the metric aggregator reuses that approach.

## References

The policy math and WSD-S schedule are ported from OLMo-core. Cite these so the
implementation can be validated and the magic constants understood:

- OLMo-core ladder run configurator (formulas, periods, checkpoint pairs):
  `OLMo-core/src/olmo_core/model_ladder/wsds_chinchilla_run_configurator.py`
- OLMo-core `WSDS` scheduler (exact LR curve):
  `OLMo-core/src/olmo_core/optim/scheduler.py`
- OLMo-core ladder object + CLI (API shape we mirror):
  `OLMo-core/src/olmo_core/model_ladder/base.py`, `.../internal/ladder.py`
- Batch-size and peak-LR scaling fits: SemanticScholar CorpusID:270764838
  ("Language models scale reliably with over-training and on downstream tasks").
- WSD-S schedule: arXiv:2410.05192.

## Architecture: Python-API-First

`Llama3Ladder` is the single source of truth. Everything else is a thin adapter
over it, so the dry-run plan, the launched run, and the read-back metrics can
never disagree.

```text
                       +------------------+
                       |   Llama3Ladder   |   single source of truth
                       |  rungs + policy  |
                       |  + compute spec  |
                       +--------+---------+
        plan/trainer_config     | run / status / metrics / sweep / compare
        +-----------------------+------------------------+
        |                       |                        |
config_registry.py        train.py                    cli.py
(nullary default recipe   (torchrun entry point:      (thin wrapper: spawns
 for run_train.sh +        per-rank ladder.run)        torchrun, JSON output,
 ConfigManager test)                                   overrides)
```

- **Read/plan side is in-process, no distributed.** `plan`, `status`, `metrics`,
  and `compare` only build models on `meta` (for param counts) and read files, so
  the agent or CLI calls them directly and gets structured data back.
- **Run side is a `torchrun` subprocess.** A run is multi-process (one rank per
  GPU). The agent or CLI `run` / `sweep --execute` spawns
  `torchrun --nproc_per_node=<world_size> -m torchtitan.experiments.scaling_ladders.train --rung 100M --weight-decay 0.05 --seed 1`;
  the `train.py` entry point rebuilds the ladder per rank and calls
  `ladder.run(rung, **overrides)`, whose body is
  `config.build()` -> `trainer.train()` -> `trainer.close()`.
- **Overrides go through the planner, not tyro.** `lr_multiplier`, `weight_decay`,
  etc. change derived LR/steps, so a post-hoc tyro override of an already-derived
  field could not re-run the policy. `train.py` parses them from argv and threads
  them into `ladder.run` so the policy re-runs cleanly.
- **Default recipe** (no overrides) is also reachable the standard TorchTitan way:
  `MODULE=scaling_ladders CONFIG=llama3_ladder_100m ./run_train.sh`; that config
  function is a one-liner: `return LADDER.trainer_config("100M")`.

## Package Shape

```text
torchtitan/experiments/scaling_ladders/
  PLAN.md
  __init__.py            # exports Llama3Ladder, the default LADDER instance, model_registry
  ladder.py             # Llama3Ladder: plan/run/status/metrics/sweep/compare
  policy.py             # WSD-S Chinchilla policy (formulas + periods + checkpoints)
  planner.py            # policy + compute spec -> resolved Trainer.Config and a plan dict
  model.py              # experiment-local Llama3 rung builders + model_registry
  lr_scheduler.py       # WSDSScheduler(LRSchedulersContainer)
  checkpoint.py         # LadderCheckpointManager(CheckpointManager) with explicit steps
  validate.py           # LadderValidator(Validator) firing at explicit steps
  metrics.py            # TensorBoard read-back -> structured per-checkpoint metrics
  config_registry.py    # thin nullary wrappers for the default recipe + debug config
  train.py              # custom entry point for API-driven / swept runs
  cli.py                # thin CLI over Llama3Ladder
```

Register `"scaling_ladders"` in `torchtitan/experiments/__init__.py`. Keep the
implementation experiment-local for v1; do not add ladder rungs to
`torchtitan.models.llama3` until APIs settle (see **Promotion Criteria**).

Constructing the module-level `LADDER` must stay cheap: audit each rung on `meta`
lazily (on first `plan`/`trainer_config` for that rung), not at import time, so
`ConfigManager` import is not slowed.

## Llama3 Ladder Family

Define experiment-local Llama3-style rungs in `model.py`, reusing `Llama3Model`,
`Llama3TransformerBlock`, the public common builders (`make_gqa_config`,
`make_ffn_config`, `compute_ffn_hidden_dim`, `Embedding/RMSNorm/Linear/ComplexRoPE`
configs), `parallelize_llama`, `pipeline_llm`, and `Llama3StateDictAdapter`. Only
the tiny per-layer init dicts are reproduced locally.

`model_registry(rung) -> ModelSpec` mirrors `models/llama3/__init__.py`:
build the rung's `Llama3Model.Config`, then return
`ModelSpec(name="scaling_ladders/llama3", flavor=rung, model=config,
parallelize_fn=parallelize_llama, pipelining_fn=pipeline_llm,
post_optimizer_build_fn=None, state_dict_adapter=Llama3StateDictAdapter)`.

Candidate rungs, all using **untied embeddings** and Llama3 vocab `128256`.
`hidden_dim` is passed directly to `make_ffn_config`; `head_dim = dim / n_heads`
(64 for the small rungs, 128 for 3B/8B). These are intentionally experiment-local,
chosen for hardware-friendly dims and a smooth small-to-large ladder ending near
Llama3-8B.

The `ladder_params` column was **verified by building each rung on `meta`** (untied,
so `lm_head` is counted, matching OLMo's `num_non_embedding_params`); every rung
lands within +/-2% of its nominal label, well inside the 5% tolerance. The planner
still re-verifies on `meta` at config-build time as a guard.

| Rung | dim | layers | heads | kv heads | head dim | hidden dim | ladder_params (meta) | vs nominal |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `60M` | 384 | 5 | 6 | 6 | 64 | 1536 | 61,051,008 | +1.8% |
| `100M` | 512 | 8 | 8 | 8 | 64 | 2048 | 99,230,208 | -0.8% |
| `190M` | 768 | 10 | 12 | 12 | 64 | 3072 | 192,888,576 | +1.5% |
| `370M` | 1024 | 14 | 16 | 16 | 64 | 4096 | 366,244,864 | -1.0% |
| `760M` | 1536 | 15 | 24 | 24 | 64 | 6144 | 763,279,872 | +0.4% |
| `1B` | 2048 | 11 | 32 | 32 | 64 | 8192 | 1,000,912,896 | +0.1% |
| `3B` | 3072 | 26 | 24 | 8 | 128 | 8192 | 3,011,410,944 | +0.4% |
| `8B` | 4096 | 34 | 32 | 8 | 128 | 14336 | 7,941,148,672 | -0.7% |

Untied embeddings are load-bearing: tying would drop the ~`vocab*dim` `lm_head`
from `ladder_params` (the 60M rung would fall to ~12M) and break both the size
labels and the transfer of OLMo's fitted constants. The smallest rungs are
therefore lm_head-dominated (e.g. 60M: ~49M lm_head + ~12M transformer), which is
exactly how OLMo's untied ladder behaves.

## Parameter Accounting

Reuse the existing counter; do not reimplement.

- Build the rung on `meta`, take total `nparams` from
  `cfg.get_nparams_and_flops(model, seq_len)[0]` (it fills `n_layers`/`n_heads`/
  `head_dims` from the config), or simply
  `sum(p.numel() for p in model.parameters())`.
- `ladder_params = nparams - vocab_size * dim`. This equals OLMo-core's
  `num_params - embeddings.weight.numel()` for both tied and untied: for untied,
  the separate `lm_head` stays counted; for tied, `parameters()` counts the shared
  matrix once and we subtract it once.
- v1 rungs are all untied (see the rung table), so only the untied branch is
  exercised. A tied unit test should build on CPU, not `meta`:
  `Decoder.init_states` re-ties on the real device, and the meta-time tie is only
  guaranteed right after `__init__`.
- Warn (or fail) if actual `ladder_params` differs from the rung target by more
  than 5%. (OLMo-core warns at 5%; our verified rungs are within 2%.)

## Scaling Policy

`policy.py` implements the OLMo WSD-S / Chinchilla policy. Defaults:

- `tokens_per_param = 20` (may need tuning for C4; OLMo-core flags this as
  dataset/optimizer-dependent)
- `chinchilla_multiple = 4` (power of two, the max horizon)
- `decay_fraction = 0.1`
- `lr_multiplier = 1.0`
- `weight_decay = 0.1` (embedding param group forced to `0.0`)
- `seq_len = 4096` for the first TorchTitan ladder

Formulas (ported verbatim; `N = ladder_params`):

```text
target_token_batch = round(2048 * 160 * (N / 108_000_000) ** (2 / 3))
train_tokens       = chinchilla_multiple * tokens_per_param * N
warmup_tokens      = N
peak_lr            = 0.0047 * (N / 108_000_000) ** (-1 / 3) / 2 * lr_multiplier
beta2              = 0.95 if actual_token_batch >= 524288 else 0.99
```

Notes that must appear in code comments:

- The `2048` in `target_token_batch` is OLMo's **anchor** sequence length baked
  into the constant; it is **not** the ladder `seq_len`. The result is a token
  count; convert to sequences with `/ seq_len`.
- The `/ 2` on `peak_lr` is OLMo-core's empirical halving ("near optimal for OLMo
  models, especially with the stepped schedule"); keep it and cite the source.
- `beta2` keys off the **actual** (rounded) token batch, not the target.
- Optimizer: plain `AdamW` (`default_adamw`) with the computed `lr`,
  `betas=(0.9, beta2)`, `weight_decay`, plus an embedding param group with
  `weight_decay=0.0`. This diverges from OLMo-core's `SkipStepAdamW` (which skips
  loss/grad-spike steps); document the divergence. Skip-step behavior is a
  possible later addition.

WSD-S periods for `chinchilla_multiple = 4`: `[0.5, 1, 2, 4] x C`, generated as
`2**p for p in range(-1, log2(chinchilla_multiple) + 1)`. For each period, save a
pre-decay checkpoint and a post-decay checkpoint; pre-decay is `decay_fraction`
of that period's length before its end. The stepped-LR variant
(`period_lr_multipliers = 1/sqrt(c)`) is supported by the scheduler but defaults
off in v1.

Policy validations (mirror OLMo-core): `chinchilla_multiple` must be a power of 2
and `>= 0.5`; `0 < decay_fraction < 0.5`; and `warmup + first-period decay <=
first-period length` (holds for the defaults, where `2N <= 10N`, but guard it).

## Compute Spec

The user supplies the compute/mesh budget when constructing the ladder
(answering "where does the data-parallel degree come from"). It is per-rung
(60M fits on 1 GPU; 8B needs sharding), with a default and per-rung overrides.

```text
ComputeSpec(
  world_size,                 # GPUs this rung's runs use (== NGPU at launch)
  parallelism,                # ParallelismConfig (dp_shard=-1, tp, pp, cp, ...)
  local_batch_size,           # per-device microbatch, in sequences
)
```

The planner derives `dp_degree` from
`ParallelDims.from_config(parallelism, world_size)` (pure arithmetic, no process
group) as the product of the data-parallel axes (`dp_replicate * dp_shard`).

Robustness:

- Because `global_batch_size` is fixed in **sequences**, the effective token
  batch is **world-size-invariant**: a different `NGPU` only shifts grad-accum vs
  data-parallel degree. World size is needed only to pick a memory-feasible
  `local_batch_size` and a valid divisor for the batch rounding.
- Add a launch-time guard that asserts `NGPU == compute_spec.world_size` for the
  rung, so a mismatched launch fails loudly (OLMo-core does the same).
- Auto-expanding device count to best hit the target batch (as OLMo-core does)
  is a possible later enhancement; v1 takes the spec as given.

## Planner

`planner.py` converts policy outputs + compute spec into a resolved
`Trainer.Config` and a structured plan dict. Both `plan()` (dict, for
dry-run/JSON) and `trainer_config()` (the actual config) call the same code path.

Steps:

1. Resolve rung -> `ModelSpec` -> `ladder_params` (built on `meta`, audited).
2. `target_token_batch = policy.target_batch(ladder_params)`.
3. `dp_degree` from `ParallelDims.from_config`.
4. `global_batch_size` (sequences) `= round(target_token_batch / seq_len)`, then
   rounded to the nearest multiple of `local_batch_size * dp_degree` (min 1).
   `actual_token_batch = global_batch_size * seq_len`. Warn if
   `|actual - target| / target > 0.10`.
5. `peak_lr` and `beta2` from policy (using `actual_token_batch`).
6. `steps = round(train_tokens / actual_token_batch)`;
   `warmup_steps = round(warmup_tokens / actual_token_batch)`.
7. Build one step-rounded period table and derive BOTH the scheduler periods and
   the checkpoint steps from it, so they stay aligned:
   - `period_lengths` are the INCREMENTAL spans between successive Chinchilla
     periods (`period_lengths[0] = 0.5xC`, then `c[i]xC - c[i-1]xC`), each
     converted to steps via `actual_token_batch`.
   - pre-decay step `= period_end_step - round(decay_fraction * period_steps)`;
     post-decay step `= period_end_step`. Using the same rounded periods ensures
     the pre-decay checkpoint lands exactly where the scheduler begins decaying.
8. Emit `Trainer.Config` with: ladder `ModelSpec`; `TrainingConfig`
   (`local_batch_size`, `global_batch_size`, `seq_len`, `steps`); `parallelism`
   from the compute spec; `default_adamw(...)` + embedding `wd=0` group;
   `WSDSScheduler.Config`; `LadderCheckpointManager.Config` (explicit steps);
   `LadderValidator.Config` (`fixed_steps` = post-decay steps); TensorBoard
   metrics enabled; dataset + `hf_assets_path`; `dump_folder`; `seed`.

Use existing TorchTitan abstractions; do not reimplement mesh rules
(`ParallelDims.from_config` provides the degrees).

## WSD-S Scheduler

`lr_scheduler.py` defines `WSDSScheduler(LRSchedulersContainer)` with a nested
`Config(LRSchedulersContainer.Config)` adding `period_lengths` (in **steps**),
`decay_fraction`, optional `period_lr_multipliers`, and `warmup_steps`. Its
`build()` returns the scheduler; the trainer picks it up via
`config.lr_scheduler.build(...)` with no core edits.

The LR curve must replicate OLMo-core's `WSDS.get_lr` exactly -- it is a sawtooth
in one continuous run, not branched runs:

- Warmup once (linear to peak) over `warmup_steps`, subtracted from period 0 only.
- Within each period: hold peak LR for the stable portion, then linearly decay to
  `decay_min_lr = 0` over `decay = round(decay_fraction * period_length)`.
- At the start of the next period the LR **jumps straight back to peak (no
  re-warmup)**; with `period_lr_multipliers`, peak is scaled per period.
- It is a pure function of the global step (works with `LambdaLR`), so the
  inherited `state_dict`/`load_state_dict` (just `last_epoch`) suffice.

This deliberately does not overload the single-period `LRSchedulersContainer`.

## Checkpointing

`checkpoint.py` defines `LadderCheckpointManager(CheckpointManager)` with a
`Config(CheckpointManager.Config)` adding `checkpoint_steps: list[int]`. Override
`_should_save` to fire only on `checkpoint_steps` (and `last_step`), bypassing the
inherited `interval` modulo. This mirrors `TorchFTCheckpointManager`.

Behavior:

- Save at each explicit pre-decay and post-decay step, and at the final step.
- Keep existing load, model-only final save, async mode, and retention.
- Set `keep_latest_k` high enough to retain all ladder checkpoints (the WSD-S
  pre-decay/post-decay pairs are the deliverable).
- Do not change core `CheckpointManager`.

## Validation at Matched Checkpoints

The showcase selects winners by validation loss at matched Chinchilla
checkpoints, but TorchTitan's `Validator` is freq-modulo only. `validate.py`
defines `LadderValidator(Validator)` adding `fixed_steps: list[int]` and
overriding `should_validate` to fire when `step in fixed_steps` (plus `step == 1`).
Set `fixed_steps` to the post-decay steps so `validation_metrics/loss` exists
exactly at the comparison points. Use a fixed `validator.steps` (not `-1`) to
avoid multi-rank hangs.

## Metric Aggregation

`metrics.py` reads back results with no bespoke parsing:

- Use `EventAccumulator` (as `scripts/loss_compare.py` does) on
  `{run_dir}/{save_tb_folder}/{timestamp}/` (where `run_dir` is the unique
  per-`(rung, overrides, seed)` folder) to read `loss_metrics/global_avg_loss`,
  `validation_metrics/loss`, and `grad_norm`.
- Map to structured per-checkpoint records keyed by step, with `tokens`,
  `chinchilla_multiple`, and `phase` (pre-/post-decay) attached from the plan.
  `val_loss` is populated only at post-decay steps (where `LadderValidator`
  fires); pre-decay records carry `train_loss`/`grad_norm` only.

Structured output (the agent-facing contract; values illustrative,
`tokens = step * actual_token_batch`):

```json
{"rung": "100M", "ladder_params": 99230208,
 "checkpoints": [
   {"step": 6376, "tokens": 1984823296, "chinchilla_multiple": 1.0,
    "phase": "post-decay", "train_loss": 2.81, "val_loss": 2.86,
    "grad_norm": 0.42}]}
```

Loss-only (plus grad norm) is the v1 scope. Downstream task evals -- the deeper
reason OLMo ladders exist -- are out of scope for v1 (see Non-Goals) and noted as
the main follow-up for "real iteration".

## Agent-Facing API and CLI

`Llama3Ladder` (in `ladder.py`) is the programmatic surface a hillclimbing loop
drives. All methods accept policy/compute overrides (`lr_multiplier`,
`chinchilla_multiple`, `tokens_per_param`, `decay_fraction`, `weight_decay`
(applied to the non-embedding group; embeddings stay `0.0`), `seed`, and per-rung
shape overrides):

```text
plan(rung, **overrides)              -> dict   # params, batch, steps, lr, beta2, ckpt_steps, dp_degree, grad_accum
trainer_config(rung, **overrides)    -> Trainer.Config
run(rung, **overrides)               -> None   # per-rank body: config.build() -> trainer.train() -> trainer.close()
run_dir(rung, **overrides)           -> str    # unique dump folder for this (rung, overrides, seed)
status(rung, **overrides)            -> dict   # ckpt/metric steps present for this run, pct complete
metrics(rung, **overrides)           -> dict   # per-checkpoint structured records for this run
sweep(rungs, grid, *, execute=False) -> list   # emit (default) or run the (rung, overrides) run specs
compare(runs, metric, at_xC)         -> dict   # rank a sweep's runs by the swept override, argmin at matched xC
```

Run identity is `(rung, overrides, seed)`. `run_dir` hashes it into a unique
`dump_folder` (e.g. `.../100M/wd0.05_seed1/`) so concurrent sweep points never
overwrite each other and `status`/`metrics` resolve the same folder. `compare`
takes the run specs a `sweep` produced (not bare rungs) and ranks by the swept
override dimension, optionally aggregating across the small rungs.

`cli.py` is a thin wrapper (precedent: OLMo-core `internal/ladder.py`). The read
commands emit JSON; all accept the override flags:

```text
dry-run --size 100M          # plan() for one rung
dry-run-all                  # plan() for every rung
run --size 100M [overrides]  # spawns torchrun -m ...scaling_ladders.train (blocks)
launch-command --size 100M   # print the run_train.sh command for the default recipe
status --size 100M [overrides]   # / status-all
metrics --size 100M [overrides]  # / metrics-all
sweep --sizes 60M,100M --grid weight_decay=0.05,0.1,0.2  # emits run specs; --execute to run
compare --runs <sweep-output> --metric val_loss --at 1xC         # rank by swept override
```

`launch-command` prints the default-recipe command with `NGPU` set to the rung's
`compute_spec.world_size` (the launch-time guard asserts they match):

```bash
NGPU=8 MODULE=scaling_ladders CONFIG=llama3_ladder_100m ./run_train.sh
```

Hillclimbing loop (how an agent or a Workflow drives it): `pipeline` over the
`(rung, override, seed)` specs from `sweep` -- stage 1 spawn `torchrun ... train`,
stage 2 poll `status(rung, **overrides)` to completion, stage 3
`metrics(rung, **overrides)` parse the objective at matched `xC` -- then a
synthesis stage takes the `compare` argmin and escalates the winner to the next
rung. The unique `run_dir`, the `status` command, and the JSON contract are what
make this loop possible without screen-scraping or run collisions.

## Config Registry

`config_registry.py` exposes one nullary function per rung for the **default
recipe** (so the standard `run_train.sh` path and the ConfigManager-load test
work). Each is a one-liner over the shared ladder:

```text
llama3_ladder_60m  -> LADDER.trainer_config("60M")
llama3_ladder_100m -> LADDER.trainer_config("100M")
... 190m, 370m, 760m, 1b, 3b, 8b
```

Also `llama3_ladder_debug`: `tests/assets/tokenizer`, `c4_test`, a tiny rung, and
a short run for fake-backend and smoke tests.

Each default config sets `HuggingFaceTextDataLoader.Config(dataset="c4")`,
`hf_assets_path` to a Llama3 tokenizer/assets dir, TensorBoard on, W&B off.

## Showcase Experiments

Two stages. The first validates the infrastructure honestly; the second is the
agentic hillclimb. The original LR-multiplier sweep is **demoted to a sanity
check**: because peak LR is already parameterized by `N`, the small-rung argmin
should land near `1.0`; if it does not, the port is wrong.

### Stage 1 -- Infrastructure validation (extrapolation)

Fit loss vs compute on the small rungs, then predict a held-out larger rung's
post-decay loss and verify by running it. The held-out target is a parameter;
default to `1B` (fit on `60M -> 760M`), because verifying the prediction means
actually training the held-out rung, and `3B`/`8B` to `4xC` costs a real pretrain
(`4xC` on 3B is `~240B` tokens). Falsifiable claim: predicted loss within a stated
tolerance. This proves the WSD-S/Chinchilla wiring and exercises the extrapolation
that is the whole point of a ladder; `3B`/`8B` held-out verification is a stretch
goal once the cheaper prediction holds.

### Stage 2 -- Hillclimb a knob the recipe does not prescribe (weight decay)

The LR formula pins LR; it says nothing about weight decay. The loop:

- Sweep `weight_decay` (e.g. `0.05, 0.1, 0.2`) on `60M, 100M, 190M` to matched
  `1xC` (preferably `2xC`).
- Select the argmin validation loss at the matched `xC` checkpoint.
- Validate that the small-rung-selected value transfers to `760M` or `1B`.
- Apply the selected value on `3B` (optionally `8B`).

This is genuine model iteration with a non-circular, falsifiable result (whether
the optimum transfers or shifts predictably is informative either way). The same
harness supports other knobs (batch-size multiplier, warmup length,
`decay_fraction`).

### Methodology (required for credibility)

- Always compare at **matched `xC`** (equal training maturity), never equal steps
  or wall-clock.
- Use `>= 2-3` seeds per point and a noise threshold for the "winner" decision;
  small-rung, short-horizon deltas are often within seed noise (this is why OLMo
  trains to `4xC`).
- Record the full resolved plan per run (in the run's dump folder) so comparisons
  are reproducible.

## Test Plan

Unit tests:

- Parameter accounting excludes input embeddings; tied embeddings are not
  double-counted; actual params within 5% of each rung target (built on `meta`).
- Policy math matches expected batch, duration, LR, beta2, and periods -- include
  a numeric cross-check against OLMo-core for at least one rung.
- Batch rounding produces a valid `global_batch_size`
  (`% (local_batch_size * dp_degree) == 0`) for a representative compute spec.
- `WSDSScheduler` matches expected LR across warmup, stable, decay, and the
  jump-back-to-peak boundary (compare to OLMo-core `WSDS.get_lr`).
- Explicit checkpoint steps trigger saves at pre-decay and post-decay steps; the
  inherited `interval` does not fire.
- `LadderValidator` fires at `fixed_steps`.
- `ConfigManager` loads `--module scaling_ladders --config llama3_ladder_100m`.
- `Llama3Ladder.plan()` and `compare()` emit the documented JSON schema.
- Distinct `(rung, overrides, seed)` map to distinct `run_dir`s; `status`/`metrics`
  resolve the right one.
- `compare` ranks a sweep's runs by the swept override and returns the argmin.
- Planner aligns checkpoint steps with scheduler decay boundaries (the pre-decay
  step is exactly where decay begins).

Smoke tests:

- `dry-run-all` prints every rung's plan (the CLI always emits JSON).
- `COMM_MODE=fake_backend` run for `llama3_ladder_debug`. Note `run_train.sh`
  forces `--training.steps 1` under `COMM_MODE`, so this checks 'builds + steps
  once', not the schedule/checkpoint logic (covered by unit tests).
- GPU smoke run for `60M` on `c4_test`, then `status` and `metrics`.

Suggested commands after implementation:

```bash
pytest tests/unit_tests/test_config_manager.py
pytest tests/unit_tests/test_scaling_ladders.py
COMM_MODE=fake_backend NGPU=8 MODULE=scaling_ladders CONFIG=llama3_ladder_debug ./run_train.sh
```

## Non-Goals For V1

- No Slurm or Beaker launcher.
- No changes to core `torchtitan.models.llama3` (rungs stay experiment-local).
- No `SkipStepAdamW` (plain AdamW; divergence documented).
- No downstream task evaluation (loss/grad-norm only).
- No automatic architecture search; rung shapes are fixed.
- No auto-expansion of device count to hit the target batch.
- No public promise that the experiment-local rung shapes are final.

## Promotion Criteria

Consider moving pieces into core only after:

- The Llama3 ladder runs at least through the small rungs and Stage 1 reproduces
  a clean loss-vs-compute trend.
- The `WSDSScheduler`, `LadderCheckpointManager`, and `LadderValidator` APIs are
  stable enough to be reused by another model family.
- The `Llama3Ladder` API and metric aggregation are useful outside this
  experiment (e.g. an agent-driven sweep on a second architecture).
- Existing TorchTitan model and integration tests remain unaffected.
```

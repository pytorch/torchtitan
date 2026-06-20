# Llama3 Scaling Ladder (WSD-S / Chinchilla)

An OLMo-style **scaling ladder** for TorchTitan: a family of Llama3 models from
small research sizes up to ~8B, each trained with Chinchilla-matched compute and
a WSD-S (Warmup-Stable-Decay, Simplified) learning-rate schedule, so that cheap
small-rung experiments predict good choices for expensive large-rung runs.

The design doc is [`DESIGN.md`](DESIGN.md); this README covers the *why*, the
OLMo-core <-> TorchTitan bridge, the code that was added, the rung family, and
the baseline extrapolation result.

## Why this is needed

Training a large model is expensive, and most design decisions (architecture
tweaks, hyperparameters, data choices) cannot be afforded at full scale. A
scaling ladder turns "guess and pray at 8B" into "measure on 60M-760M and
extrapolate": if a quantity follows a smooth loss-vs-compute trend across the
small rungs, you can both (a) **predict** a larger model's loss before paying for
it, and (b) **select** choices on cheap rungs that transfer up the ladder.

OLMo-core ships exactly this machinery. TorchTitan -- the PyTorch-native training
stack (FSDP2/DTensor/DCP, TP/PP/CP) -- did not. This experiment ports OLMo-core's
WSD-S / Chinchilla ladder onto TorchTitan's native stack so the same disciplined,
predictive workflow is available here, and so it can be driven by an automated
hillclimbing loop (human or agent), not just by hand.

## OLMo-core vs. TorchTitan: the gap, and how we bridge it

The *policy* (the scaling-law math and the WSD-S curve) is model-agnostic and was
ported verbatim. The *plumbing* differs substantially; the bridge is a set of
thin adapters over TorchTitan's existing abstractions.

| Concern | OLMo-core | TorchTitan | Bridge in this experiment |
|---|---|---|---|
| Run config | `RunConfigurator` family | `Trainer.Config` (no `JobConfig`); models are `ModelSpec` | `planner.py` emits a resolved `Trainer.Config` |
| Training duration | train-by-tokens (`Duration`) | strictly **step-based** (`step < training.steps`) | planner converts every token budget to steps |
| Batch size | tokens | **sequences** (`global_batch_size`), grad-accum derived | planner rounds the target token-batch to a valid sequence count |
| LR schedule | `WSDS` over tokens | `LRSchedulersContainer` (`LambdaLR`, per-step) | `WSDSScheduler` reproduces OLMo's `get_lr` as a step-indexed lambda (bit-identical curve) |
| Checkpointing | interval/duration list | `CheckpointManager` (interval modulo) | `LadderCheckpointManager` saves at explicit pre/post-decay steps |
| Validation | flexible | `Validator` is **freq-modulo only** | `LadderValidator` fires at the matched-Chinchilla post-decay steps |
| Optimizer | `SkipStepAdamW` (skips loss/grad spikes) | plain `AdamW` | plain `AdamW` with an embedding `weight_decay=0` group (divergence documented) |
| Plug-in mechanism | config classes | `Configurable._owner` auto-wiring | scheduler/checkpoint/validator are `Configurable` subclasses -- **no core edits** |
| Mesh degrees | -- | `ParallelDims.from_config` is pure arithmetic | planner computes the data-parallel degree at config-build time (no process group) |

The only sanctioned core touch is registering the experiment name in
`torchtitan/experiments/__init__.py`. Everything else lives in this folder and
**reuses** Llama3 (`Llama3Model`, `parallelize_llama`, `pipeline_llm`,
`Llama3StateDictAdapter`) and the public common builders.

Parameter accounting matches OLMo-core exactly: the ladder size is *non-embedding*
parameters, `ladder_params = total_params - vocab_size * dim`
(OLMo's `num_non_embedding_params`). All rungs use untied embeddings, so the
`lm_head` is counted.

## Salient code changes (high level)

Everything is under `torchtitan/experiments/scaling_ladders/`:

- `policy.py` -- WSD-S / Chinchilla policy: target batch size, training duration,
  peak LR, beta2, and Chinchilla periods. Ported from OLMo-core and
  unit-cross-checked against an inline copy of its formulas.
- `model.py` -- experiment-local Llama3 rungs (the table below), reusing
  `Llama3Model` and the public common builders; only the tiny per-layer init
  dicts are reproduced locally. `count_ladder_params` builds each rung on `meta`.
- `lr_scheduler.py` / `checkpoint.py` / `validate.py` -- `WSDSScheduler`,
  `LadderCheckpointManager` (explicit checkpoint steps), and `LadderValidator`
  (validation at matched-Chinchilla steps), each a `Configurable` subclass that
  plugs in without core edits.
- `planner.py` -- the single resolution path: policy + compute spec ->
  `Trainer.Config` and a plan dict. The same step-rounded period table drives
  *both* the scheduler's decay boundaries and the checkpoint steps, so the
  pre-decay checkpoint lands exactly where decay begins.
- `ladder.py` -- `Llama3Ladder`, the source of truth and the agent-facing API:
  `plan / trainer_config / run / status / metrics / sweep / compare`. Run
  identity is `(rung, overrides, seed)`, hashed into a unique `run_dir`.
- `config_registry.py` / `train.py` / `cli.py` -- nullary default recipes for the
  standard `run_train.sh` path, a torchrun entry point for API-driven runs, and a
  thin CLI.
- `metrics.py` -- reads TensorBoard scalars back into structured per-checkpoint
  records (reusing the `scripts/loss_compare.py` approach).
- `showcase.py` -- the experiment drivers: loss-vs-compute fit + extrapolation,
  and `compare_variants` / `plot_loss_vs_compute` for code-variant A/Bs.

A nuance worth calling out: throughput on these small models is grad-accumulation
overhead bound at `local_batch_size=1`, so the per-rung `lbs` defaults are tuned
for ~1 grad-accum step under 8-way data parallelism. This is purely a throughput
choice -- the global batch, token budget, and loss curve are unchanged.

## The Llama3 ladder family

Eight rungs, all untied embeddings, Llama3 vocab `128256`, `head_dim = dim/n_heads`.
`ladder_params` is the non-embedding count built on `meta`; every rung is within
~2% of its nominal label.

| Rung | dim | layers | heads | kv heads | hidden dim | ladder_params |
|---|---:|---:|---:|---:|---:|---:|
| 60M  | 384  | 5  | 6  | 6  | 1536  | 61,051,008 |
| 100M | 512  | 8  | 8  | 8  | 2048  | 99,230,208 |
| 190M | 768  | 10 | 12 | 12 | 3072  | 192,888,576 |
| 370M | 1024 | 14 | 16 | 16 | 4096  | 366,244,864 |
| 760M | 1536 | 15 | 24 | 24 | 6144  | 763,279,872 |
| 1B   | 2048 | 11 | 32 | 32 | 8192  | 1,000,912,896 |
| 3B   | 3072 | 26 | 24 | 8  | 8192  | 3,011,410,944 |
| 8B   | 4096 | 34 | 32 | 8  | 14336 | 7,941,148,672 |

The policy (defaults: `tokens_per_param=20`, `chinchilla_multiple=4`,
`decay_fraction=0.1`, `seq_len=4096`) derives, per rung, the target token batch,
training steps, peak LR (`~ N^-1/3`), beta2, and the WSD-S period table. Because
WSD-S decays to zero at the end of every Chinchilla period, a single run yields a
*converged* post-decay checkpoint at each period (e.g. 0.5xC and 1xC), which the
read-back turns into matched-compute comparison points.

## Baseline experiment: loss-vs-compute extrapolation

The headline validation of the infrastructure: fit a Chinchilla curve on the
small rungs and predict a held-out larger rung's loss, then run it to check.

- **Fit** on 60M / 100M / 190M / 370M at `chinchilla_multiple=1` -- each run
  contributes its 0.5xC and 1xC post-decay validation loss, giving 8
  (compute, loss) points.
- **Held out**: 760M, predicted from the fit, then trained once to verify.
- Curve: `L(C) = E + A * (C / c_ref)^(-alpha)` with `C = 6 N D` (FLOPs).
- Real C4 train + c4_validation, 8x B200, FSDP, `torch.compile`, ~7.8 hr total.

Fitted curve: `E = 1.482`, `alpha = 0.108`, fit RMSE = `0.026`.

| Held-out point | predicted val loss | actual val loss | relative error |
|---|---:|---:|---:|
| 760M @ 0.5xC | 2.976 | 2.983 | **0.25%** |
| 760M @ 1xC   | 2.868 | 2.901 | **1.15%** |

A curve fit only on rungs up to 370M predicts the 760M validation loss to ~1%.
The falsifiable claim -- "predicted loss within tolerance" -- holds.

![Loss-vs-compute extrapolation: fit on 60M-370M predicts held-out 760M](assets/extrapolation.png)

## Reproducing

```bash
# Dry-run plans for every rung (in-process, no GPU):
python -m torchtitan.experiments.scaling_ladders.cli dry-run-all

# Default-recipe single rung via the standard launcher:
NGPU=8 MODULE=scaling_ladders CONFIG=llama3_ladder_100m ./run_train.sh

# The extrapolation experiment (trains the rungs, then fits + predicts):
python -m torchtitan.experiments.scaling_ladders.showcase \
    --fit-rungs 60M,100M,190M,370M --held-out-rungs 760M --chinchilla-multiple 1.0
```

C4 streams from the HF hub, so training needs network egress configured.

## Status and next steps

- **Done:** the ladder infrastructure, CPU unit tests, and the baseline
  loss-vs-compute extrapolation above.
- **In progress:** a Flavor-Q "agent edits code, ladder judges it" loop -- a
  worktree-isolated architecture change (QK-norm) trained on the small rungs and
  compared against the *reused* baseline curve via `compare_variants` /
  `plot_loss_vs_compute`.
- **Out of v1 scope:** the weight-decay hillclimb, multi-seed noise bands, the
  1B/3B verification, and downstream task evals (see `DESIGN.md`).

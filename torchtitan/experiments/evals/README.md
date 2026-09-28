# Pretraining evals

Evaluate the checkpoints of a training run with
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness)
while it trains, without touching the training job and without converting
checkpoints to Hugging Face format.

- **Out of band.** The eval job runs next to the training job on its own GPUs
  and only reads the `step-N` checkpoints training writes. Training is never
  paused or slowed.
- **Native checkpoints.** Each DCP checkpoint is loaded straight into the
  torchtitan model built from the training config, so what is evaluated is
  exactly what was trained. Convert to HF once, for the final model only
  (`scripts/checkpoint_conversion/convert_to_hf.py`).
- **Same numerics as training.** Requests are packed into token buffers in the
  training collator's layout and scored through the model's own
  `preprocess_inputs` and attention masking. On training documents the eval
  path reproduces the trainer's cross-entropy exactly.

## Setup

```bash
pip install -r torchtitan/experiments/evals/requirements.txt
```

This adds lm-eval to the training env; it does not touch the installed torch.

## Usage

Pass the eval options, then `--`, then the exact training arguments (the same
`--module`, `--config` and overrides as the training job, so the eval finds
its `dump_folder` and builds the same model):

```bash
TRAIN_ARGS="--module llama3 --config llama3_8b --dump_folder ./outputs/run7"

# Watch mode: evaluate each checkpoint as it lands; exits once the final
# training step (training.steps) has been evaluated.
torchrun --nproc_per_node=8 -m torchtitan.experiments.evals.evaluate \
    --tasks trainstation_quick --watch -- $TRAIN_ARGS

# Backfill: every completed checkpoint that has no results for the suite yet.
torchrun --nproc_per_node=8 -m torchtitan.experiments.evals.evaluate \
    --tasks trainstation_full -- $TRAIN_ARGS

# Specific steps.
torchrun --nproc_per_node=8 -m torchtitan.experiments.evals.evaluate \
    --tasks trainstation_full --steps 20000 40000 -- $TRAIN_ARGS
```

Results go to `<dump_folder>/evals/step-N/<suite>.json` (lm-eval's `results`,
plus step, dtype and lm-eval version). A step with results is skipped on
later runs, so re-running is cheap and a crashed job can simply be restarted.

### Models that do not fit on one GPU

`--tensor-parallel-degree N` shards each model replica over N GPUs with the
model's own tensor-parallel plan; the remaining GPUs are data-parallel
replicas that lm-eval splits the examples across. For example, 8 GPUs with
`--tensor-parallel-degree 4` run two replicas of four GPUs each. The model is
parallelized the same way as for training, so the checkpoint loads through
DCP resharding whatever layout it was trained with.

Other axes of the training layout (FSDP sharding, EP, CP, PP) are not used by
the eval job: lm-eval gives each data-parallel rank different requests, and
those axes communicate across ranks in the forward pass.

### W&B

`--wandb` logs every result as `eval/<task>/<metric>` to a W&B run named
`<run name>-eval-<suite>` in the training run's group. It reads the same
environment variables as the training job's W&B logger (`WANDB_TEAM`,
`WANDB_PROJECT`, `WANDB_RUN_NAME`, `WANDB_RUN_GROUP`). Metrics are plotted
against `train_step`, so backfilled or out-of-order results land in the right
place, and a restarted eval job resumes the same run.

### Other options

`--limit` (examples per task, for smoke tests), `--dtype`,
`--num-tokens-per-batch`, `--poll-interval`, `--output-folder`; see `--help`.

Make sure `checkpoint.keep_latest_k` does not delete checkpoints before they
are evaluated; `checkpoint.purge_exempt` can keep selected steps.

## Suites

Suites are lm-eval groups defined in [`tasks/`](./tasks); any lm-eval task or
group name also works with `--tasks`.

| Suite | Tasks |
|---|---|
| `trainstation_quick` | WikiText (bits per byte), LAMBADA, HellaSwag, ARC-Easy, PIQA |
| `trainstation_full` | `trainstation_quick` + ARC-Challenge, WinoGrande, BoolQ, OpenBookQA, SciQ, `trainstation_mmlu` |
| `trainstation_mmlu` | MMLU, 5-shot, accuracy weighted by subject size |

Paloma is a good addition to `trainstation_quick` once your Hugging Face
account has accepted its terms (`allenai/paloma` is gated).

## Limitations

- Log-likelihood tasks only. Generative tasks (GSM8K, HumanEval, ...) need a
  generation loop and are not supported yet.
- No FSDP, EP, CP or PP in the eval job (see above), so MoE models too large
  for tensor parallelism alone are not supported yet.
- Scoring in the training loop is not implemented yet; `TrainstationLM` is
  written so a validator can reuse it.

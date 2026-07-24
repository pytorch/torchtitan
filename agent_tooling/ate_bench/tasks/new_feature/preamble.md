You are integrating a newly published model architecture into the **TorchTitan**
training framework. Use your full toolset (Read, Grep, Glob, Bash, Edit, Write).

You are given the same materials an engineer would consult: the architecture's
arXiv paper and its reference implementation (see `references.md`). Produce a
**training script that integrates the new feature into the base MoE model** and
runs on the `{{DATASET}}` dataset under the config: pipeline parallel = 4, expert
parallel = 2, data-parallel shard = auto (`-1`; EP is carved from the data axis,
so on {{NGPU}} GPUs the data axis is 2 and EP=2 shards experts across it), sequence
length 2048, global batch {{GLOBAL_BATCH_SIZE}}, BF16, with `MODULE={{MODULE}}`,
`CONFIG={{CONFIG}}`.

Launch training with the provided script, e.g.:
```
STEPS=64 GLOBAL_BATCH_SIZE={{GLOBAL_BATCH_SIZE}} bash {{TRAIN_SH}} --dataloader.dataset={{DATASET}} 2>&1 | tee {{WORKSPACE}}/{{LABEL}}/{{TASK_ID}}/train.log
```

Run training for **64 steps**. Your change is correct only if **both** hold:
1. cross-entropy loss **decreases** across the 64-step run and stays **finite**
   (no NaN, no explosion);
2. your implementation clearly exhibits the architecture's **three distinguishing
   components** (from the paper) in your `git diff` against `main` — these are
   judged independently, so make the intended mechanism evident in the diff (a
   passing loss from an unchanged model does not count).

Where the relevant code lives in TorchTitan:
- attention: `torchtitan/models/common/attention.py`, `.../common/decoder.py`
- MoE routing + experts: `torchtitan/models/common/moe.py`
  (`TokenChoiceTopKRouter`), `.../common/token_dispatcher.py`
- per-model layer assembly: `torchtitan/models/{{MODULE}}/model.py`

Save the 64-step training log to `{{WORKSPACE}}/{{LABEL}}/{{TASK_ID}}/train.log`
so the loss-curve check can read it.

**Stop condition (important).** The complete deliverable is: your code edits + ONE
successful 64-step training run whose loss decreases and stays finite. As soon as
that single run succeeds, you are DONE — stop and report. Do NOT run repeated
trainings, hyperparameter sweeps, ablations, baseline comparisons, or extended
verification. If the first run fails, fix and retry, but keep iterations minimal.

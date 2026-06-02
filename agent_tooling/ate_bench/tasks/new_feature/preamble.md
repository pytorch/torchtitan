You are integrating a newly published model architecture into the **TorchTitan**
training framework. Use your full toolset (Read, Grep, Glob, Bash, Edit, Write).

You are given the same materials an engineer would consult: the architecture's
arXiv paper and its reference implementation (see `references.md`). Produce a
**training script that integrates the new feature into the base MoE model** and
runs on the C4 dataset under the fixed config: pipeline parallel = 4, expert
parallel = 2, data-parallel shard = 1, sequence length 2048, global batch 1024,
BF16, on {{NGPU}} GPUs, with `MODULE={{MODULE}}`, `CONFIG={{CONFIG}}`.

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

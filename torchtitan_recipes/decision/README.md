# Typed-decision (Jev-style) fine-tuning

Decision models such as TypeSafe's Jev answer typed questions about a state
(`choice`, yes/no `noul`, ordinal `score`) with a probability per option in one
forward pass instead of generating text. This recipe trains one with the
simplest open approach, a letter-slot readout:

```text
Context:
<state>

Question (yes/no): Did source annotators rate this comment as toxic (at least 0.7)?
Options:
A) Yes
B) No
Answer: A          <- only this token is supervised
```

At inference, the next-token logits at `Answer:` restricted to the valid
letters give the option distribution, so every question is answered in one
forward pass with no generated text. Each question is its own packed
document, so questions about the same state cannot see each other.

## Run

```bash
python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-1.7B-Base --assets tokenizer safetensors config
python -m torchtitan_recipes.decision.prepare_data     # -> ./assets/hf/bev-decision-150K/
python -m torchtitan_recipes.decision.evaluate --hf-assets ./assets/hf/Qwen3-1.7B-Base
NGPU=2 MODULE=torchtitan_recipes.decision.qwen3 CONFIG=qwen3_1_7b_decision \
  ./run_train.sh --output-dir outputs/decision
python -m torchtitan_recipes.decision.evaluate --hf-assets ./assets/hf/Qwen3-1.7B-Base \
  --checkpoint outputs/decision/checkpoint/step-600
```

Data is [`avbiswas/bev-decision-150K`](https://huggingface.co/datasets/avbiswas/bev-decision-150K)
(247K training questions after filtering). Training is a 600-step full
fine-tune, about 3 minutes on 2 GB300 GPUs.

Rows are short (~180 tokens) with one supervised token each, which changes
what dominates a step. The recipe therefore applies `lm_head` only at the
supervised positions (`AnswerTokenLoss`), reads data with a single reader
thread instead of grain's 16, prefetches 16 microbatches, skips
activation recompute, and caps documents per microbatch at 256. On one GB300
this takes throughput from 68.7K to 105.5K tokens/s (steps 21-80 of 80).

RMSNorm is the next largest cost: eager `F.rms_norm` runs ATen's LayerNorm
kernels, which are slow for Qwen3's 128-wide q/k norms. Recent PyTorch nightlies
replace them with CuTeDSL (quack) kernels when `nvidia-cutlass-dsl` and
`apache-tvm-ffi` are installed at a version PyTorch lists as known-good
(4.8.0 here; other versions silently fall back). With that, throughput is
114.9K tokens/s with no recipe change.

## Results

1,500 held-out BEV test questions (BEV's train and test splits share no
states); temperature fitted on 760 test questions from 250 other states.

| type | n | accuracy | Brier (lower is better) | ECE (lower is better) | answered at 5% error |
|---|---|---|---|---|---|
| all | 1500 | 0.608 -> 0.873 | 0.494 -> 0.177 | 0.058 -> 0.021 | 0.15 -> 0.80 |
| choice | 576 | 0.549 -> 0.882 | 0.542 -> 0.162 | 0.028 -> 0.019 | 0.23 -> 0.84 |
| noul | 609 | 0.780 -> 0.936 | 0.346 -> 0.099 | 0.092 -> 0.014 | 0.04 -> 0.95 |
| score | 315 | 0.384 -> 0.737 | 0.692 -> 0.356 | 0.077 -> 0.060 | 0.00 -> 0.37 |

These are in-distribution numbers: the test questions come from the same
source datasets as training.

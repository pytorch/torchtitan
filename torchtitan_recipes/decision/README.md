# Typed-decision (Jev-style) fine-tuning

Fine-tune Qwen3-1.7B-Base to answer typed questions about a state without
generating text, following TypeSafe's Jev: `choice` (categorical), `noul`
(yes/no), and `score` (ordered scale). Each question gets one probability per
option from a single forward pass, read at the answer slot:

```text
Context:
<state>

Question (choice): Which team should handle this ticket?
Options:
A) Payments and refunds
B) Login and account access
C) Something else
Answer:  --one forward pass--> next-token logits
                               keep only " A", " B", " C"
                               softmax(logits / temperature)
                               -> {payments: 0.91, account: 0.06, other: 0.03}
```

Training supervises only the answer-letter token. Each question is packed as
a separate document, so questions cannot attend to each other.

## Run

```bash
python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-1.7B-Base --assets tokenizer safetensors config
python -m torchtitan_recipes.decision.prepare_data
python -m torchtitan_recipes.decision.evaluate --hf-assets ./assets/hf/Qwen3-1.7B-Base
NGPU=2 MODULE=torchtitan_recipes.decision.qwen3 CONFIG=qwen3_1_7b_decision ./run_train.sh --output-dir outputs/decision
python -m torchtitan_recipes.decision.evaluate --hf-assets ./assets/hf/Qwen3-1.7B-Base --checkpoint outputs/decision/checkpoint/exports/step-600
```

`prepare_data` downloads [`avbiswas/bev-decision-150K`](https://huggingface.co/datasets/avbiswas/bev-decision-150K)
and writes one prompt per question to `./assets/hf/bev-decision-150K/`
(about 247K training questions). A seeded shuffle randomizes `choice` options
and `noul` Yes/No order in every split; `score` levels stay ordered. Training
is a 600-step full fine-tune.

Supported parallelism: FSDP by default; pass a `ParallelismConfig` to
`qwen3_1_7b_decision` from your own config function for tensor parallelism
(with loss parallel) or context parallelism (Ulysses varlen attention). Pipeline
parallelism is unsupported because Qwen3-1.7B ties its embedding and output
weights.

Rows average ~180 tokens with one supervised token. `AnswerTokenLoss` applies
`lm_head` only at supervised positions; the recipe also uses a single data
reader, prefetches 16 microbatches, disables activation recomputation, and
caps documents per microbatch at 256. On one GB300 these raise throughput
from 68.7K to 105.5K tokens/s (mean of steps 21-80 of an 80-step run) over the
same recipe with `ChunkedLossWrapper`, default recompute and data loading, and
512 documents per microbatch.

## Results

1,500 test questions; temperature fitted on questions from 250 other test
states. Training, calibration, and evaluation share no states. Pairs are
base -> fine-tuned.

| type | n | accuracy (higher is better) | Brier (lower is better) | ECE (lower is better) | answered at 5% error (higher is better) |
|---|---|---|---|---|---|
| all | 1500 | 0.611 -> 0.873 | 0.491 -> 0.177 | 0.062 -> 0.016 | 0.10 -> 0.79 |
| choice | 576 | 0.552 -> 0.889 | 0.528 -> 0.158 | 0.053 -> 0.019 | 0.25 -> 0.86 |
| noul | 609 | 0.783 -> 0.933 | 0.352 -> 0.097 | 0.103 -> 0.015 | 0.00 -> 0.96 |
| score | 315 | 0.384 -> 0.727 | 0.690 -> 0.369 | 0.086 -> 0.065 | 0.00 -> 0.34 |

"Answered" is the largest fraction of questions, taken most-confident first,
whose error rate stays at or below 5%. These are in-distribution results: test
questions come from the same source datasets as training.

## Candidate-distribution fine-tuning

The Qwen3.5 recipe reads GS1-format `request`/ `targets` JSONL. Preparation
aligns targets by candidate ID, fixes option order, and uses Jev's `Answer:`
readout. Both sources write `target_probs` (one-hot for Jev), use the same
4,000-character state limit, and reject over-length tokenized prompts. GS1
preparation checks that sampled states are disjoint across the three input splits.

```bash
python scripts/download_hf_assets.py --repo_id Qwen/Qwen3.5-0.8B-Base --assets tokenizer safetensors config
python -m torchtitan_recipes.decision.prepare_data \
    --input /path/to/train.jsonl --calibration /path/to/calibration.jsonl --eval /path/to/final_test.jsonl
NGPU=1 MODULE=torchtitan_recipes.decision.qwen3 CONFIG=qwen3_5_distribution ./run_train.sh
python -m torchtitan_recipes.decision.evaluate --hf-assets ./assets/hf/Qwen3.5-0.8B-Base \
    --checkpoint /path/to/export --data ./assets/hf/gs1-distribution
```

Both objectives sample one answer from the soft targets per question.
Pass `loss_fn=CrossEntropyLoss.Config()` to `qwen3_5_distribution` for CE on
the sampled token. The default objective uses the sampled slot and option
count for candidate NLL plus expected multiclass Brier, with configurable
`brier_weight`. The Qwen3.5 recipe uses data parallelism and four contexts
per microbatch to bound activation memory.


For the comparison, take the first 10,000 records from distribution-v4.5's
training JSONL with `head -n 10000` and use that file as `--input`. Train on
3,000 sampled questions with the default 600 steps and 32,768 tokens per
microbatch, with CUDA graphs enabled. For each objective run seeds 42 and 43
by setting `config.debug.seed` in a recipe function. Fit temperature on 500
calibration questions and evaluate on 500 final-test questions. Accuracy
and ECE use the target probability of the predicted option as its expected
correctness.

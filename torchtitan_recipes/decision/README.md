# Typed-decision (Jev-style) fine-tuning

Fine-tune Qwen3 or Qwen3.5 to answer typed questions about a state without
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

## Qwen3.5 candidate-distribution fine-tuning

`qwen3_5_distribution` defaults to the same public
[BEV dataset](https://huggingface.co/datasets/avbiswas/bev-decision-150K) as Jev.
It uses 120 steps and 16,384 tokens per microbatch with data parallelism.

```bash
python scripts/download_hf_assets.py --repo_id Qwen/Qwen3.5-0.8B-Base --assets tokenizer safetensors config
python -m torchtitan_recipes.decision.prepare_data
NGPU=1 MODULE=torchtitan_recipes.decision.qwen3 CONFIG=qwen3_5_distribution ./run_train.sh --output-dir outputs/distribution
python -m torchtitan_recipes.decision.evaluate --hf-assets ./assets/hf/Qwen3.5-0.8B-Base \
    --checkpoint outputs/distribution/checkpoint/exports/step-120
```

The candidate objective computes NLL plus `brier_weight` times expected
multiclass Brier over the listed options. Pass `CrossEntropyLoss.Config()`
as `loss_fn` to train on the same answer tokens with full-vocabulary CE.
Both objectives sample an answer from the target probabilities, which are
one-hot for BEV.

### Reproduce the objective comparison

Run each objective with seeds 42 and 43, setting both `config.debug.seed`
and `config.dataloader.seed` in a recipe function. Each run uses one A800
and CUDA graphs. Fit temperature by minimizing NLL on the calibration split
and report metrics on the 1,500 disjoint evaluation questions.
The benchmark uses BEV revision `11ed9b5b879e7f9524e7c89c1037d498c54206f0`
and Qwen3.5-0.8B-Base revision `dc7cdfe2ee4154fa7e30f5b51ca41bfa40174e68`.

| Objective | Seed | Accuracy ↑ | Brier ↓ | ECE ↓ | Temperature |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-0.8B-Base | — | 0.4780 | 0.6036 | 0.0542 | 2.103 |
| Full-vocabulary CE | 42 | 0.7553 | 0.3214 | 0.0122 | 0.987 |
| Full-vocabulary CE | 43 | 0.7520 | 0.3313 | 0.0157 | 0.990 |
| Candidate NLL + Brier | 42 | 0.7733 | 0.3183 | 0.0242 | 1.001 |
| Candidate NLL + Brier | 43 | 0.7420 | 0.3456 | 0.0205 | 1.052 |

Both objectives improved accuracy, Brier, and ECE over the base checkpoint
in both seeds. Candidate NLL + Brier had the lower Brier at seed 42, and
full-vocabulary CE had the lower Brier at seed 43.

### Use GS1-format soft targets

For local GS1-format data, pass separate train, calibration, and evaluation
JSONL files to `prepare_data`. Each record has a `request` with typed
questions and a `targets` mapping keyed by question ID. Each target gives
`candidate_ids` and a `distribution`. Preparation aligns probabilities by
candidate ID, applies the shared prompt format and state length limit, and
excludes sampled calibration and evaluation states from training.

```bash
python -m torchtitan_recipes.decision.prepare_data \
    --input /path/to/train.jsonl --calibration /path/to/calibration.jsonl --eval /path/to/final_test.jsonl
```

Point `config.dataloader.dataset.dataset.source.load_dataset_kwargs["data_files"]`
to `./assets/hf/gs1-distribution/train.jsonl` in your recipe function and pass
`--data ./assets/hf/gs1-distribution` to the evaluator. Accuracy and ECE use the
target probability of the predicted option as its expected correctness.

# Candidate-distribution training experiment

This experiment fine-tunes **Qwen3.5**, using GS1-format `request`/`targets`
records and sampled distribution NLL plus weighted expected multiclass Brier.
It does not load native GS1 checkpoints or reproduce GS1's decision head.

- `prepare_data.py` selects text questions and aligns targets by candidate ID.
- `recipe.py` contains the processor, loss, and `qwen3_5_distribution` config.
- There is no evaluation entry point for this experiment yet. The Jev
  `evaluate.py` uses a different prompt and must not be used for these exports.

The existing `torchtitan_recipes/decision/` directory remains the validated
Jev pipeline: `prepare_data.py` produces prompts for `qwen3.py`, and
`evaluate.py` evaluates those checkpoints. The experiment shares Jev’s answer-position loss wrapper.

```bash
python scripts/download_hf_assets.py --repo_id Qwen/Qwen3.5-0.8B-Base --assets tokenizer safetensors config
python -m torchtitan.experiments.decision_distribution.prepare_data --input /path/to/train.jsonl
NGPU=2 MODULE=torchtitan.experiments.decision_distribution.recipe CONFIG=qwen3_5_distribution ./run_train.sh
```

Preparation accepts a JSONL file with one `request` and `targets` object per
line. Each question target supplies `candidate_ids` and `distribution`. The
distribution-v4.5 export used for validation is local to the GS1 project.

On one A800, Qwen3.5-0.8B-Base completed 10 optimizer steps and exported its
final weights. The run used 3,000 sampled questions (preparation seed 0),
8,192 tokens per microbatch, and accelerator graphs disabled. Training loss
was 1.93938 at step 1 and 1.50018 at step 10. Held-out calibration and
comparisons with full-vocabulary CE remain to be measured.

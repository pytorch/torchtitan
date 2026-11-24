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
`evaluate.py` evaluates those checkpoints. Neither pipeline imports the other.

```bash
python scripts/download_hf_assets.py --repo_id Qwen/Qwen3.5-0.8B --assets tokenizer safetensors config
python -m torchtitan.experiments.decision_distribution.prepare_data --input /path/to/train.jsonl
NGPU=2 MODULE=torchtitan.experiments.decision_distribution.recipe CONFIG=qwen3_5_distribution ./run_train.sh
```

The preparation default of 3,000 questions is a configurable smoke-run subset,
not a convergence recommendation. GPU training and checkpoint parity remain
unverified; this experiment is not a validated public recipe.

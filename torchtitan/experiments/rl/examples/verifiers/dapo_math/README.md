# DAPO Math with Verifiers

This example keeps the existing [DAPO Math](../../dapo_math) recipe unchanged
and replaces only its rollout path with
[Verifiers](https://github.com/PrimeIntellect-ai/verifiers).

It uses Verifiers' null harness and a local subprocess runtime for a
single-turn math task. No tools or sandbox are provided, so do not use this
configuration for untrusted code execution.

## Setup

Follow the [TitanRL setup](../../../README.md), then install this example's dependencies:

```bash
pip install -r torchtitan/experiments/rl/examples/verifiers/dapo_math/requirements.txt

python scripts/download_hf_assets.py \
  --repo_id Qwen/Qwen3-4B-Base \
  --local_dir torchtitan/experiments/rl/example_checkpoint \
  --all
```

## Run

```bash
python -m torchtitan.experiments.rl.train \
  --module verifiers.dapo_math \
  --config rl_dapo_qwen3_4b_verifiers_8k
```

Use `rl_dapo_qwen3_4b_verifiers_32k` for the 32K response variant.

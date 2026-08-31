# Verifiers

This example keeps the existing [DAPO Math](../dapo_math) recipe unchanged and replaces only its rollout path with [Verifiers](https://github.com/PrimeIntellect-ai/verifiers). Training still uses the filtered DAPO-Math dataset, AIME 2025 validation, DAPO loss, and the Qwen3-4B-Base model.

Verifiers runs a single-turn math task with its `null` harness. The runtime is a local subprocess; there is no Docker or remote sandbox and no tools are exposed. Do not use this configuration for untrusted code execution.

Verifiers is optional, and all Verifiers integration code lives in this example.
Other TitanRL recipes do not require it.

## Setup

Follow the [TitanRL setup](../../README.md), then install this example's dependencies:

```bash
pip install -r torchtitan/experiments/rl/examples/verifiers/requirements.txt

python scripts/download_hf_assets.py \
  --repo_id Qwen/Qwen3-4B-Base \
  --local_dir torchtitan/experiments/rl/example_checkpoint \
  --all
```

## Run

```bash
python -m torchtitan.experiments.rl.train \
  --module verifiers \
  --config rl_dapo_qwen3_4b_verifiers_8k
```

Use `rl_dapo_qwen3_4b_verifiers_32k` for the 32K response variant.

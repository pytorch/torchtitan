# Qwen3-4B-Base on DAPO-Math

This example trains Qwen3-4B-Base on verifiable math problems using DAPO loss. It fits one eight-GPU node by assigning two GPUs to the trainer and one GPU to each of six independent generators.

## Configuration

```text
model:             Qwen3-4B-Base
trainer:           TP=2, fp32 language-model head
generators:        6 replicas, TP=1
rollouts:          8 prompts x 16 completions per optimizer step
prompt budget:     2,048 tokens
response budget:   8,192 tokens
packing length:    10,240 tokens
off-policy limit:  4 trainer versions
learning rate:     1e-6, constant
```

Training uses the 12,643-row [filtered DAPO-Math dataset](https://huggingface.co/datasets/hamishivi/DAPO-Math-17k-Processed_filtered). Validation uses all 30 problems from [AIME 2025](https://huggingface.co/datasets/opencompass/AIME2025). Math-Verify assigns a binary reward from the final `Answer:` expression.

The default configuration runs 1,000 optimizer steps, or 128,000 sampled completions. `max_offpolicy_steps=4` bounds policy lag and permits 40 prompt groups in the active rollout buffer.

## Setup

Follow the [RL environment setup](../../README.md), then download the base checkpoint:

```bash
python scripts/download_hf_assets.py \
  --repo_id Qwen/Qwen3-4B-Base \
  --local_dir torchtitan/experiments/rl/example_checkpoint \
  --all
```

## Run

The launcher uses the TP=2 trainer configuration and defaults to 150 steps:

```bash
./torchtitan/experiments/rl/examples/dapo_math/run_train.sh
```

Override the duration or output directory with environment variables:

```bash
TRAINING_STEPS=10 \
DUMP_FOLDER=outputs/rl/qwen3_4b_dapo_math_smoke \
./torchtitan/experiments/rl/examples/dapo_math/run_train.sh
```

The replicated DP=2 trainer configuration is also available:

```bash
python -m torchtitan.experiments.rl.train \
  --module dapo_math \
  --config rl_dapo_qwen3_4b_math_8k
```

## 150-step result

The single-node TP=2 trial completed optimizer steps 0 through 149 in 1 hour 46 minutes. This trial used the earlier 7,168-token response and 9,216-token packing limits; the runnable configuration above restores the full 8,192-token response budget.

```text
metric                              first 10 steps    last 10 steps
rollout reward, mean                0.067             0.401
rollout total length, mean          988 tokens        3,284 tokens
rollout truncation rate, mean       1.75%             14.55%
```

The final recorded step had mean reward `0.302`, mean total length `3,999`, and truncation rate `18.75%`. The maximum observed mean reward was `0.632`; the maximum per-step truncation rate was `30.73%`. Initial AIME validation scored `0/30` before training.

<!-- Plot placeholder: rollout_reward/_mean versus optimizer step. -->

<!-- Plot placeholder: rollout/total_length/{mean,max} and rollout/truncation_rate/mean versus optimizer step. -->

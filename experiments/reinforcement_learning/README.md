# Reinforcement Learning

## Setup

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install torch --extra-index-url https://download.pytorch.org/whl/nightly/cu126
uv pip install transformers>=4.40.0
uv pip install torchmonarch==0.3.0
```

## Usage

Run the CLI entry point to execute training:

```bash
# Show all options
python experiments/reinforcement_learning/main.py --help

# Run training with sum digits task
python experiments/reinforcement_learning/main.py --task sum_digits --num-steps 20 --num-generators 4

# Run training with reverse digits task
python experiments/reinforcement_learning/main.py --task reverse_digits --num-steps 20 --num-generators 4
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--task` | sum_digits | Task to train on (`sum_digits` or `reverse_digits`) |
| `--num-steps` | 20 | Number of training steps |
| `--num-generators` | 2 | Number of generator workers |
| `--eval-samples` | 20 | Number of evaluation samples |
| `--verbose` | off | Print sample generations on each step |

## Tasks

### Sum Digits

Given a sequence of 2-4 integers (10-999), the model must compute the total digit sum.

Example: "What is the total digit sum of [123, 45, 67]?" → 1+2+3 + 4+5 + 6+7 = 28

The system prompt includes a worked example and instructs the model to solve step by step.

### Reverse Digits

Given a 6-7 digit integer (100000-9999999), the model must reverse its digits.

Example: "What is 123456 reversed?" → 654321

Trailing zeros become leading zeros and are dropped (e.g., 1200 → 0021 → 21).

## Training

Training uses REINFORCE with per-token normalized log-probabilities. Each step:

1. All generators produce trajectories in parallel (one per generator)
2. The trainer computes a policy gradient update on the batch
3. Updated weights are synced back to all generators

### Reward

| Condition | Reward |
|-----------|--------|
| Correct answer | +1.0 |
| Wrong answer | -1.0 |

### Evaluation

Evaluation runs on the trainer's model before and after training. Accuracy is measured by `extract_answer`, which parses the model's output using a fallback chain:

1. `[ANSWER] <number>` tag (preferred)
2. Patterns like "the answer is <number>" or "= <number>"
3. Last number in the text (fallback)

Format compliance measures the fraction of responses using the `[ANSWER]` tag.

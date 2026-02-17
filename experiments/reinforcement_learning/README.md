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

# Run training
python experiments/reinforcement_learning/main.py --num-steps 20 --num-generators 4
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--num-steps` | 20 | Number of training steps |
| `--num-generators` | 2 | Number of generator workers |
| `--eval-samples` | 16 | Number of evaluation samples |
| `--verbose` | off | Print sample generations on each step |

## Zorplex Task

Zorplex is a synthetic benchmark for training LLMs on multi-step tool use. Each word in a fixed vocabulary (apple, banana, cat, ...) maps to a hidden integer value via a seeded lookup table. The model must use a `LOOKUP[word]` tool to discover values — it cannot know them in advance.

Each task picks two words and asks the model to add their zorplex values: "What is zorplex('apple') + zorplex('banana')?" → two LOOKUPs, then add the results.

### Multi-turn Generation

Each trajectory runs for up to `max_turns` turns (default: 4). On each turn, the model generates text until a tool call (`LOOKUP[word]`) is detected. The tool result is injected back into the context, and the model continues. When no tool call is produced, the turn is treated as the final response.

The ideal trajectory is 3 turns:
1. `LOOKUP[apple]` → gets value
2. `LOOKUP[banana]` → gets value
3. Computes sum, emits `[ANSWER] <number>`

### Reward Calculation

Rewards are assigned per trajectory in `Generator._run_generation`:

| Condition | Reward |
|-----------|--------|
| Correct answer | +1.0 |
| Correct + `[ANSWER]` tag | +1.2 |
| Wrong answer | 0.0 |

The `[ANSWER]` format bonus (+0.2) is only awarded when the answer is also correct, preventing the model from gaming the reward by emitting `[ANSWER]` without doing the work.

### Evaluation

Evaluation runs on the trainer's model before and after training (`--eval-samples` controls the number of tasks).

**Accuracy** is measured by `extract_answer`, which parses the model's final answer using a fallback chain:

1. `[ANSWER] <number>` tag (preferred)
2. Patterns like "the answer is <number>" or "= <number>"
3. Last number in the text (fallback)

If the extracted number matches the correct answer, the trajectory counts as correct. A model can get accuracy credit without using `[ANSWER]` formatting.

**Format compliance** measures the fraction of trajectories where the model emits an `[ANSWER]` tag. A model that learns to use `LOOKUP` tools correctly but never wraps its final answer in `[ANSWER]` will show high accuracy but low format compliance.

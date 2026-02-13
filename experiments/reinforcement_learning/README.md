# Reinforcement Learning

## Setup

```bash
conda create -n torchtitan-rl python=3.12 -y
conda activate torchtitan-rl
pip install torch --extra-index-url https://download.pytorch.org/whl/nightly/cu126
pip install transformers>=4.40.0
pip install torchmonarch==0.3.0
```

## Usage

Run the CLI entry point to execute training:

```bash
# Show all options
python experiments/reinforcement_learning/main.py --help

# Run training
python experiments/reinforcement_learning/main.py --num-steps 40 --num-generators 4
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--num-steps` | 20 | Number of training steps |
| `--num-generators` | 2 | Number of generator workers |
| `--eval-samples` | 10 | Number of evaluation samples |
| `--verbose` | off | Print sample generations on each step |

## Zorplex Task

Zorplex is a synthetic benchmark for training LLMs on multi-step tool use. Each word in a fixed vocabulary (apple, banana, cat, ...) maps to a hidden integer value via a seeded lookup table. The model must use a `LOOKUP[word]` tool to discover values — it cannot know them in advance.

### Task Types

- **Simple**: Single lookup. "What is the zorplex value of 'apple'?" → `LOOKUP[apple]` → `[ANSWER] 82`
- **Compositional**: Multiple lookups + arithmetic. "What is zorplex('apple') + zorplex('banana')?" → two LOOKUPs, then add the results.

The default task is **compositional** with **easy** difficulty (2 words, addition only).

### Multi-turn Generation

Each trajectory runs for up to `max_turns` turns (default: 4). On each turn, the model generates text until a tool call (`LOOKUP[word]`) is detected. The tool result is injected back into the context, and the model continues. When no tool call is produced, the turn is treated as the final response.

For compositional-easy, the ideal trajectory is 3 turns:
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

### Accuracy Evaluation

Accuracy is measured by `extract_answer`, which tries to parse the model's final answer using a fallback chain:

1. `[ANSWER] <number>` tag (preferred)
2. Patterns like "the answer is <number>" or "= <number>"
3. Last number in the text (fallback)

If the extracted number matches the correct answer, the trajectory counts as correct. This means a model can get accuracy credit without using `[ANSWER]` formatting, but only gets the full 1.2 reward with the tag.

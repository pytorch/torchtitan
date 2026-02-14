# Reinforcement Learning

## Setup

```bash
conda env create -f experiments/reinforcement_learning/environment.yml
conda activate torchtitan-rl
```

## Usage

Run the CLI entry point to execute sync, async, or both training loops:

```bash
# Show all options
python experiments/reinforcement_learning/run.py --help

# Run only sync training
python experiments/reinforcement_learning/run.py --num-steps 5 --num-generators 1 --mode sync

# Run only async training
python experiments/reinforcement_learning/run.py --num-steps 10 --num-generators 3 --mode async

# Run both sync and async training
python experiments/reinforcement_learning/run.py --num-steps 20 --num-generators 2 --mode both

# Save a timeline plot (requires matplotlib, only with --mode both)
python experiments/reinforcement_learning/run.py --mode both --plot timeline.png
```

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--num-steps` | 20 | Number of training steps per loop |
| `--num-generators` | 2 | Number of generator workers |
| `--num-zorplex` | 2 | Number of Zorplex tool workers |
| `--eval-samples` | 10 | Number of evaluation samples |
| `--mode` | sync | Training mode: `sync`, `async`, or `both` |
| `--plot` | None | Save timeline plot to this path |

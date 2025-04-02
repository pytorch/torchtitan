# Model Generation Check

The `test_generate` script provides a straightforward way to validate models, tokenizers, checkpoints, and device compatibility by running a single forward pass. This script functions as a sanity check to ensure everything is set up correctly.

While **torchtitan** focuses on advanced features for distributed pre-training, this script acts as a lightweight integration test to verify runtime setup. For more extensive inference and generation capabilities, consider tools like [pytorch/torchchat](https://github.com/pytorch/torchchat/).

## Purpose and Use Case

This script is ideal for users who need to:

- **Run Sanity Checks**: Confirm that models, tokenizers, and checkpoints load without errors.
- **Test Compatibility**: Execute a forward pass to assess model response and memory usage.
- **Evaluate Device Scaling**: Optionally test distributed generation using tensor parallel (TP) to confirm multi-device functionality.

## Usage Instructions

#### Run on a single GPU.

```bash
NGPU=1 CONFIG_FILE=./torchtitan/models/llama3/train_configs/llama3_8b.toml CHECKPOINT_DIR=./outputs/checkpoint/ \
PROMPT="What is the meaning of life?" \
./scripts/generate/run_llama_generate.sh --max_new_tokens=32 --temperature=0.8 --seed=3
```

#### Run on 4 GPUs and pipe results to a json file.

```bash
NGPU=4 CONFIG_FILE=./torchtitan/models/llama3/train_configs/llama3_8b.toml CHECKPOINT_DIR=./outputs/checkpoint/ \
PROMPT="What is the meaning of life?" \
./scripts/generate/run_llama_generate.sh --max_new_tokens=32 --temperature=0.8 --seed=3 --out > output.json
```

#### View Available Arguments

```bash
> python -m scripts.generate.test_generate --help
```

# Model Generation Check

The `test_generate` script provides a straightforward way to validate models, tokenizers, checkpoints, and device compatibility by running a single forward pass. This script functions as a sanity check to ensure everything is set up correctly.

While **torchtitan** focuses on advanced features for distributed pre-training, this tool acts as a lightweight integration test to verify runtime setup. For more extensive inference and generation capabilities, consider tools like [pytorch/torchchat](https://github.com/pytorch/torchchat/).

## Purpose and Use Case

This script is ideal for users who need to:

- **Run Sanity Checks**: Confirm that models, tokenizers, and checkpoints load without errors.
- **Test Compatibility**: Execute a forward pass to assess model response and memory usage.
- **Evaluate Device Scaling**: Optionally test distributed generation using tensor parallel (TP) to confirm multi-device functionality.

## Usage Instructions

#### Run on a single GPU.

```bash
NGPU=1 CONFIG_FILE=./train_configs/llama3_8b.toml CHECKPOINT_DIR=./outputs/checkpoint/ \
PROMPT="What is the meaning of life?" \
./test/generate/run_llama_pred.sh --max_new_tokens=32 --temperature=0.8 --seed=3
```

#### Run on 4 GPUs

```bash
NGPU=4 CONFIG_FILE=./train_configs/llama3_8b.toml CHECKPOINT_DIR=./outputs/checkpoint/ \
PROMPT="What is the meaning of life?" \
./test/generate/run_llama_pred.sh --max_new_tokens=32 --temperature=0.8 --seed=3
```

#### View Available Arguments

```bash
> python ./test/generate/test_generate.py --help
```

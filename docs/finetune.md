## Fine-tuning from an existing checkpoint

You first need to download the Llama checkpoint. Here are the commands:

```bash
# Configure these paths as needed
export HF_TOKEN=... # get your HF token from https://huggingface.co/settings/tokens
export ORIGINAL_MODEL_DIR="tmp"
export TOKENIZER_DIR="assets/tokenizer"
export DCP_MODEL_DIR="assets/models/dcp/llama3.1-8B"

# Download the tokenizer and model weights
rm -rf $ORIGINAL_MODEL_DIR
huggingface-cli download meta-llama/Llama-3.1-8B original/tokenizer.model --local-dir $ORIGINAL_MODEL_DIR
huggingface-cli download meta-llama/Llama-3.1-8B original/consolidated.00.pth --local-dir $ORIGINAL_MODEL_DIR
huggingface-cli download meta-llama/Llama-3.1-8B original/params.json --local-dir $ORIGINAL_MODEL_DIR
# Convert the model weights to the DCP format and move it and the tokenizer to the target directories
mkdir -p $TOKENIZER_DIR && cp $ORIGINAL_MODEL_DIR/original/tokenizer.model $TOKENIZER_DIR/Meta-Llama-3.1-8B-tokenizer.model
python -m scripts.convert_llama_to_dcp $ORIGINAL_MODEL_DIR/original/ $DCP_MODEL_DIR
```

Then you can fine-tune from the checkpoint:

```bash
export TOKENIZER_DIR="assets/tokenizer"
export DCP_MODEL_DIR="assets/models/dcp/llama3.1-8B"
CONFIG_FILE="./torchtitan/models/llama3/train_configs/llama3_8b.toml" uv run ./run_train.sh \
  --model.tokenizer_path $TOKENIZER_DIR/Meta-Llama-3.1-8B-tokenizer.model \
  --checkpoint.initial_load_path $DCP_MODEL_DIR \
  --checkpoint.enable_checkpoint
```

You should see something like this:

```bash
...
l batch size 8, gradient accumulation steps 1, sequence length 8192, total steps 1000 (warmup 200).
[rank0]:[titan] 2025-06-20 19:13:25,465 - root - INFO - Loading the checkpoint from assets/models/dcp/llama3.1-8B.
[rank0]:[titan] 2025-06-20 19:13:39,662 - root - INFO - [GC] GC collection for checkpoint loading. 0.01 seconds.
[rank0]:[titan] 2025-06-20 19:13:39,663 - root - INFO - Finished loading the checkpoint in 14.20 seconds.
[rank0]:[titan] 2025-06-20 19:13:39,663 - root - INFO - Training starts at step 1.
```
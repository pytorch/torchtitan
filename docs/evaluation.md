# Evaluation in HF

Torchtitan provides direct and indirect support for validation to support user's training goals. Direct support is provided by the `Validator` class which interacts directly with the training loop, and indirect support is provided through [huggingface checkpoint conversion](https://github.com/pytorch/torchtitan/blob/main/docs/checkpoint.md#huggingface) for users who want to asynchronously do evaluation using external tools such as ELeutherAI's `lm_eval`.

## Validation
For users who want to perform validation directly during the training loop, we provide the `Validator` class which can be conveniently overloaded through the `train_spec` or configured in `job_config`. The validator class has access to and reuses many of the trainer's functions such as its parallelization, including pipelining.

Below is an example validation config:

```toml
[validation]
enabled = true
dataset = "c4_validation"
freq = 500
steps = -1 #consume the entire validation set
```

## Third-Party Evaluation
With `./scripts/checkpoint_conversion/convert_to_hf`, `torchtitan` offers support for converting checkpoints from dcp to safetensors format. Using this script, users can perform efficient evaluation asynchronously from their training using external libraries that support HuggingFace e.g. `lm_eval` with `vllm` backend.

### Example usage of `lm_eval` with `vllm`:
To use this specific setup make sure to include a huggingface `config.json` file which is not provided by conversion script or `last_save_in_hf` option. The hf config file can be downloaded by running `python ./scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets config`.

Note that pip installing `lm-eval` may result in breaking `torchtitan` dev environment so we recommend creating a separate env.
```bash
pip install "lm-eval[vllm]"
lm_eval --model vllm \
    --model_args pretrained=./outputs/checkpoint/step-1000,tensor_parallel_size=8,dtype=auto,gpu_memory_utilization=0.8, \
    --tasks lambada_openai \
    --batch_size auto
```

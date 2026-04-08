# RL Training with TorchTitan and vLLM

This directory contains code for RL training using TorchTitan model definitions with vLLM inference engine for fast rollout generation.

> **Note:** This experiment is under active development. APIs and configurations may change.

## Overview
The integration consists of the following components:

1. **vLLM Model Wrapper** (`models/vllm_wrapper.py`): Adapts TorchTitan models for vLLM's inference engine
2. **RL Training Loop** (`simple_grpo_sum_digits.py`): GRPO-based RL training with Monarch actors
3. **Inference Script** (`inference_example.py`): Standalone inference using the vLLM engine


## Key features available

1. **Unified model definition**: Canonical TorchTitan model definition shared by both trainer (TorchTitan) and generator (vLLM), enabling fast iteration, shared optimizations, and straightforward bitwise parity verification
2. **[Monarch](https://github.com/meta-pytorch/monarch) as controller**: Distributed actor framework for orchestrating trainer and generator on separate GPU meshes with async communication
3. **[TorchStore](https://github.com/meta-pytorch/torchstore) for weight sync**: Efficient weight synchronization between trainer and generator, supporting direct GPU-to-GPU RDMA transfers

## Quick Start
### Prerequisites

0. Create and activate environment with uv:
```bash
pip install uv
uv venv --python 3.12 titan-rl
source titan-rl/bin/activate
```

1. Install Monarch and TorchStore from main:
```bash
uv pip install torchmonarch==0.5.0.dev20260403
uv pip install --no-deps "git+https://github.com/meta-pytorch/torchstore.git@main"
uv pip install pygtrie portpicker
```

2. Install Flash Attention 3 kernels:
```bash
# Flash Attention v3 (recommended for H100/H200 and newer GPUs)
uv pip install flash-attn-3 --extra-index-url=https://download.pytorch.org/whl/test/cu128
```

**NOTE:** FA2 is bundled with PyTorch and will be used automatically on older GPUs (e.g. A100) that don't support FA3.

3. Install batch-invariant ops if you need to run batch-invariant mode (Triton kernels for bitwise-reproducible training):
```bash
uv pip install --no-deps "git+https://github.com/thinking-machines-lab/batch_invariant_ops.git@main"
```

4. Install PyTorch nightly for torchtitan, and pre-built vllm wheels (based on PyTorch nightly version).
```bash
# Install vllm with nightly torch
uv pip install torch vllm xformers  --pre \
--extra-index-url https://download.pytorch.org/whl/nightly/cu128 \
--index-strategy unsafe-best-match
```

**NOTE:** The pre-built vLLM wheels are only compatible with CUDA 12.8, though they should work with most older CUDA versions. Alternatively, you can install the corresponding vLLM pre-built wheels directly from https://download.pytorch.org/whl/nightly/cu128, for example: `uv pip install vllm-1.0.0.dev20260219+cu130-<suffix>.whl`. Ensure the build version number (e.g., `dev20260219`) matches your PyTorch nightly installation.


5. Install TorchTitan in editable mode:
```bash
uv pip install -e .
```

6. Download `Qwen/Qwen3-0.6B` (or `Qwen/Qwen3-1.7B`) checkpoint from HuggingFace to `torchtitan/experiments/rl/example_checkpoint` folder.
```bash
python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-0.6B --local_dir torchtitan/experiments/rl/example_checkpoint --all --hf_token=...

python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-1.7B --local_dir torchtitan/experiments/rl/example_checkpoint --all --hf_token=...
```

7. Run inference with torchtitan model definition:
```bash
torchrun --nproc_per_node=2 torchtitan/experiments/rl/inference_example.py
```

**NOTE:**: Set `--nproc_per_node` to the world size, which should match the `tensor_parallel_degree` in the `VLLMGenerator` config.

8. Run simple GRPO RL loop to learn sum digits task
```bash
python torchtitan/experiments/rl/simple_grpo_sum_digits.py --module rl --config rl_grpo_qwen3_0_6b
```

**NOTE:** If you downloaded your HF model to a different path than the one in step 4, specify it in your command with `--hf_assets_path=<path_to_model_checkpoint>`.

We use a unified model definition from torchtitan for the trainer and generator, ensuring bitwise-identical models to address a class of subtle correctness bugs in RL for LLMs.

## Reproducibility

We provide two independent tools for debugging and reproducibility. They address different sources of non-determinism and can be used separately or together.

### Batch-invariant mode

Batch-invariant mode guarantees that a model's output for a given input is **identical regardless of what other inputs are in the batch**. This is critical for RL training because the generator computes log-probs in one batch composition (e.g. 8 completions), while the trainer recomputes them in a different batch composition (e.g. 2 completions after DP sharding). Without batch-invariant mode, the same input can produce different log-probs in different batch contexts due to floating-point accumulation order differences.

When enabled, batch-invariant mode will:
- Replaces `mm`, `addmm`, `log_softmax`, and `mean.dim` with Triton kernels that use a fixed tile iteration order (via [batch_invariant_ops](https://github.com/thinking-machines-lab/batch_invariant_ops))
- Forces NCCL to use Ring all-reduce with a single channel for deterministic inter-GPU collectives
- Disables reduced-precision reductions and TF32 to prevent batch-size-dependent rounding
- Forces `num_splits=1` in flash attention to prevent non-deterministic split-k reductions


### Verifying generator/trainer logprob parity
If you want to run true on-policy mode in TorchTitan RL and debug generator/trainer log-prob parity, you should enable `batch-invariant-mode` to eliminate potential numerical differences caused by batch-size discrepancies between the generator and trainer. The `batch-invariant-mode` provides run-to-run determinism for both the trainer and generator. If the model has randomness (e.g., dropout), you should also ensure consistent behavior between the trainer and generator by specifying a `seed`.

Now we only support logprob bitwise parity when trainer and generator are under the same parallelism.
Example:
```bash
python torchtitan/experiments/rl/simple_grpo_sum_digits.py --module rl --config  rl_grpo_qwen3_0_6b_batch_invariant
```

This config sets `DebugConfig(batch_invariant=True, deterministic=True)` and `training.dtype="bfloat16"` (required so the trainer computes in the same precision as the generator, as a limitation because TP only doesn't naturally support mixed precision training).

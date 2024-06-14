[![4 GPU Integration Test](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_4gpu.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_4gpu.yaml?query=branch%3Amain)
[![8 GPU Integration Test](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu.yaml?query=branch%3Amain)

# torchtitan

`torchtitan` is currently in a pre-release state and under extensive development.

`torchtitan` is a proof-of-concept for Large-scale LLM training using native PyTorch. It is (and will continue to be) a repo to showcase PyTorch's latest distributed training features in a clean, minimal codebase. torchtitan is complementary to and not a replacement for any of the great large-scale LLM training codebases such as Megatron, Megablocks, LLM Foundry, Deepspeed, etc. Instead, we hope that the features showcased in torchtitan will be adopted by these codebases quickly. torchtitan is unlikely to ever grow a large community around it.

Our guiding principles when building `torchtitan`:

* Designed to be easy to understand, use and extend for different training purposes.
* Minimal changes to the model code when applying 1D, 2D, or (soon) 3D Parallel.
* Modular components instead of a monolithic codebase.
* Get started in minutes, not hours!

### Intro video - learn more about torchtitan in under 4 mins:

[![Welcome to torchtitan!](assets/images/titan_play_video.png)](https://youtu.be/ee5DOEqD35I?si=_B94PbVv0V5ZnNKE "Welcome to torchtitan!")

## Pre-Release Updates:
#### (4/25/2024): `torchtitan` is now public but in a pre-release state and under development.
Currently we showcase pre-training **Llama 3 and Llama 2** LLMs of various sizes from scratch. `torchtitan` is tested and verified with the PyTorch nightly version `torch-2.4.0.dev20240412`. (We recommend latest PyTorch nightly).

### Key features available

1. [FSDP2 with per param sharding](docs/fsdp.md)
2. [Tensor Parallel](https://pytorch.org/docs/stable/distributed.tensor.parallel.html)
3. Selective layer and operator activation checkpointing
4. Distributed checkpointing
5. 2 datasets pre-configured (45K - 144M)
6. GPU usage, MFU, tokens per second and more displayed via TensorBoard
6. Learning rate scheduler, meta init, Optional Fused RMSNorm
7. All options easily configured via [toml files](train_configs/)
8. [Interoperable checkpoints](docs/checkpoint.md) which can be loaded directly into [`torchtune`](https://github.com/pytorch/torchtune) for fine tuning

We report our [Performance](docs/performance.md) verified on 64 A100 GPUs


### Coming soon

1. Async checkpointing
2. FP8 support
3. Context Parallel
4. 3D Pipeline Parallel
5. `torch.compile` support
6. Scalable data loading solution


## Installation

```bash
git clone https://github.com/pytorch/torchtitan
cd torchtitan
pip install -r requirements.txt
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121 # or cu118
pip3 install --pre torchdata --index-url https://download.pytorch.org/whl/nightly
```

### Downloading a tokenizer

`torchtitan` currently supports training Llama 3 (8B, 70B), and Llama 2 (7B, 13B, 70B) out of the box. To get started training these models, we need to download a tokenizer.model. Follow the instructions on the official [meta-llama](https://huggingface.co/meta-llama/Meta-Llama-3-8B) repository to ensure you have access to the Llama model weights.

Once you have confirmed access, you can run the following command to download the Llama 3 / Llama 2 tokenizer to your local machine.

```bash
# Get your HF token from https://huggingface.co/settings/tokens

# llama3 tokenizer.model
python torchtitan/datasets/download_tokenizer.py --repo_id meta-llama/Meta-Llama-3-8B --tokenizer_path "original" --hf_token=...

# llama2 tokenizer.model
python torchtitan/datasets/download_tokenizer.py --repo_id meta-llama/Llama-2-13b-hf --hf_token=...
```

### Start a training run
Llama 3 8B model locally on 8 GPUs

```bash
CONFIG_FILE="./train_configs/llama3_8b.toml" ./run_llama_train.sh
```


## TensorBoard

To visualize TensorBoard metrics of models trained on a remote server via a local web browser:

1. Make sure `metrics.enable_tensorboard` option is set to true in model training (either from a .toml file or from CLI).

2. Set up SSH tunneling, by running the following from local CLI
```
ssh -L 6006:127.0.0.1:6006 [username]@[hostname]
```

3. Inside the SSH tunnel that logged into the remote server, go to the torchtitan repo, and start the TensorBoard backend
```
tensorboard --logdir=./outputs/tb
```

4. In the local web browser, go to the URL it provides OR to http://localhost:6006/.


## Multi-Node Training
For training on ParallelCluster/Slurm type configurations, you can use the `multinode_trainer.slurm` file to submit your sbatch job.

To get started adjust the number of nodes and GPUs
```
#SBATCH --ntasks=2
#SBATCH --nodes=2
```

Then start a run where `nnodes` is your total node count, matching the sbatch node count above.

```
srun torchrun --nnodes 2
```

If your gpu count per node is not 8, adjust:

```--nproc_per_node```

 in the torchrun command and

```#SBATCH --gpus-per-task```

in the SBATCH command section.

## License

This code is made available under [BSD 3 license](./LICENSE). However you may have other legal obligations that govern your use of other content, such as the terms of service for third-party models, data, etc.

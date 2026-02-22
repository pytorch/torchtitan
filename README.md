<div align="center">

# torchtitan

#### A PyTorch native platform for training generative AI models

[![8 GPU Feature Tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_features.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_features.yaml?query=branch%3Amain)
[![8 GPU Model Tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_models.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_models.yaml?query=branch%3Amain)
[![arXiv](https://img.shields.io/badge/arXiv-2410.06511-b31b1b.svg)](https://arxiv.org/abs/2410.06511)
[![ICLR](https://img.shields.io/badge/ICLR-2025-violet.svg)](https://iclr.cc/virtual/2025/poster/29620)
[![forum](https://img.shields.io/badge/pytorch-forum-DE3412.svg)](https://discuss.pytorch.org/c/distributed/torchtitan/44)
[![license](https://img.shields.io/badge/license-BSD_3--Clause-lightgrey.svg)](./LICENSE)
[![pip](https://img.shields.io/pypi/v/torchtitan?color=blue)](https://pypi.org/project/torchtitan/)
[![conda](https://img.shields.io/conda/vn/conda-forge/torchtitan?color=green)](https://anaconda.org/conda-forge/torchtitan)


</div>

`torchtitan` is under extensive development. To use the latest features of `torchtitan`, we recommend using the most recent PyTorch nightly.


## Latest News
- [2025/11] AMD released an [optimized fork](https://github.com/AMD-AGI/torchtitan-amd/tree/main) of `torchtitan` for AMD GPUs.
- [2025/10] We released `torchtitan` [v0.2.0](https://github.com/pytorch/torchtitan/releases).
- [2025/10] SkyPilot now supports `torchtitan`! See the tutorial [here](https://docs.skypilot.co/en/latest/examples/training/torchtitan.html).
- [2025/07] We published [instructions](/torchtitan/models/README.md) on how to add a model to `torchtitan`.
- [2025/04] Our paper was accepted by [ICLR 2025](https://iclr.cc/virtual/2025/poster/29620).
- [2024/12] GPU MODE [lecture](https://www.youtube.com/watch?v=VYWRjcUqW6w) on torchtitan.
- [2024/07] [Presentation](https://pytorch2024.sched.com/event/1fHn3) at PyTorch Conference 2024.


## Overview

`torchtitan` is a PyTorch native platform designed for **rapid experimentation and large-scale training** of generative AI models. As a minimal clean-room implementation of PyTorch native scaling techniques, `torchtitan` provides a flexible foundation for developers to build upon. With `torchtitan` [extension points](docs/extension.md), one can easily create custom extensions tailored to specific needs.

Our mission is to accelerate innovation in the field of generative AI by empowering researchers and developers to explore new modeling architectures and infrastructure techniques.

The Guiding Principles when building `torchtitan`
* Designed to be easy to understand, use and extend for different training purposes.
* Minimal changes to the model code when applying multi-dimensional parallelism.
* Bias towards a clean, minimal codebase while providing basic reusable / swappable components.

`torchtitan` has been showcasing PyTorch's latest distributed training features, via support for pretraining Llama 3.1 LLMs of various sizes.

## Contributing

We look forward to your contributions!

* To accelerate contributions to and innovations around torchtitan, we host an [`experiments`](torchtitan/experiments) folder. New ideas should start there. To contribute, follow the [`experiments guidelines`](torchtitan/experiments/README.md).
* For fixes and contributions to core, follow these [`guidelines`](CONTRIBUTING.md).

## Llama 3.1 training

### Key features available

1. Multi-dimensional composable parallelisms
   - [FSDP2](docs/fsdp.md) with per-parameter sharding
   - [Tensor Parallel](https://pytorch.org/docs/stable/distributed.tensor.parallel.html) (including [async TP](https://discuss.pytorch.org/t/distributed-w-torchtitan-introducing-async-tensor-parallelism-in-pytorch/209487))
   - [Pipeline Parallel](https://discuss.pytorch.org/t/distributed-w-torchtitan-training-with-zero-bubble-pipeline-parallelism/214420)
   - [Context Parallel](https://discuss.pytorch.org/t/distributed-w-torchtitan-breaking-barriers-training-long-context-llms-with-1m-sequence-length-in-pytorch-using-context-parallel/215082)
2. [Meta device](https://pytorch.org/docs/stable/meta.html) initialization
3. Selective (layer or operator) and full activation checkpointing
4. [Distributed checkpointing](https://discuss.pytorch.org/t/distributed-w-torchtitan-optimizing-checkpointing-efficiency-with-pytorch-dcp/211250) (including async checkpointing)
   - [Interoperable checkpoints](docs/checkpoint.md) which can be loaded directly into [`torchtune`](https://github.com/pytorch/torchtune) for fine-tuning
5. `torch.compile` support
6. [Float8](https://discuss.pytorch.org/t/distributed-w-torchtitan-enabling-float8-all-gather-in-fsdp2/209323) support ([how-to](docs/float8.md))
7. [MXFP8 training for dense and MoE models](docs/mxfp8.md) on Blackwell GPUs.
7. DDP and HSDP
8. [TorchFT](https://github.com/pytorch/torchft) integration
9. Checkpointable data-loading, with the C4 dataset pre-configured (144M entries) and support for [custom datasets](docs/datasets.md)
10. Gradient accumulation, enabled by giving an additional `--training.global_batch_size` argument on the CLI
11. Flexible learning rate scheduler (warmup-stable-decay)
12. Loss, GPU memory, throughput (tokens/sec), TFLOPs, and MFU displayed and logged via [Tensorboard or Weights & Biases](/docs/metrics.md)
13. [Debugging tools](docs/debugging.md) including CPU/GPU profiling, memory profiling, Flight Recorder, etc.
14. All options easily configured via [Python config registry](torchtitan/models/llama3/config_registry.py) with `--module` and `--config` CLI flags
15. [Helper scripts](scripts/) to
    - download tokenizers from Hugging Face
    - convert original Llama 3 checkpoints into the expected DCP format
    - estimate FSDP/HSDP memory usage without materializing the model
    - run distributed inference with Tensor Parallel

We report [performance](benchmarks/llama3_h100_202412_torchtitan.md) on up to 512 GPUs, and verify [loss converging](docs/converging.md) correctness of various techniques.

### Dive into the code

You may want to see how the model is defined or how parallelism techniques are applied. For a guided tour, see these files first:
* [torchtitan/train.py](torchtitan/train.py) - the main training loop and high-level setup code
* [torchtitan/models/llama3/model/model.py](torchtitan/models/llama3/model/model.py) - the Llama 3.1 model definition
* [torchtitan/models/llama3/infra/parallelize.py](torchtitan/models/llama3/infra/parallelize.py) - helpers for applying Data Parallel, Tensor Parallel, activation checkpointing, and `torch.compile` to the model
* [torchtitan/models/llama3/infra/pipeline.py](torchtitan/models/llama3/infra/pipeline.py) - helpers for applying Pipeline Parallel to the model
* [torchtitan/components/checkpoint.py](torchtitan/components/checkpoint.py) - utils for saving/loading distributed checkpoints
* [torchtitan/components/quantization/float8.py](torchtitan/components/quantization/float8.py) - utils for applying Float8 techniques


## Installation

One can directly run the source code, or install `torchtitan` from a nightly build, or a stable release.

### From source

This method requires the nightly build of PyTorch, or the latest PyTorch built [from source](https://github.com/pytorch/pytorch?tab=readme-ov-file#from-source).

```bash
git clone https://github.com/pytorch/torchtitan
cd torchtitan
pip install -r requirements.txt
```

### Nightly builds

This method requires the nightly build of PyTorch. You can replace `cu126` with another version of cuda (e.g. `cu128`) or an AMD GPU (e.g. `rocm6.3`).

```sh
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu126 --force-reinstall
pip install --pre torchtitan --index-url https://download.pytorch.org/whl/nightly/cu126
```

### Stable releases
One can install the latest [stable release](https://github.com/pytorch/torchtitan/releases) of `torchtitan` via `pip` or `conda`.
```sh
pip install torchtitan
```
```sh
conda install conda-forge::torchtitan
```
Note that each stable release pins the nightly versions of `torch` and `torchao`. Please see [release.md](docs/release.md) for more details.

### Downloading a tokenizer

`torchtitan` currently supports training Llama 3.1 (8B, 70B, 405B) out of the box. To get started training these models, we need to download the tokenizer. Follow the instructions on the official [meta-llama](https://huggingface.co/meta-llama/Llama-3.1-8B) repository to ensure you have access to the Llama model weights.

Once you have confirmed access, you can run the following command to download the Llama 3.1 tokenizer to your local machine.

```bash
# Get your HF token from https://huggingface.co/settings/tokens

# Llama 3.1 tokenizer
python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer --hf_token=...
```

### Start a training run
Llama 3 8B model locally on 8 GPUs

```bash
MODULE=llama3 CONFIG=llama3_8b ./run_train.sh
```

### Multi-Node Training
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

If your gpu count per node is not 8, adjust `--nproc_per_node` in the torchrun command and `#SBATCH --gpus-per-task` in the SBATCH command section.


## Citation

We provide a detailed look into the parallelisms and optimizations available in `torchtitan`, along with summary advice on when to use various techniques.

[TorchTitan: One-stop PyTorch native solution for production ready LLM pre-training](https://openreview.net/forum?id=SFN6Wm7YBI)
```
@inproceedings{
   liang2025torchtitan,
   title={TorchTitan: One-stop PyTorch native solution for production ready {LLM} pretraining},
   author={Wanchao Liang and Tianyu Liu and Less Wright and Will Constable and Andrew Gu and Chien-Chin Huang and Iris Zhang and Wei Feng and Howard Huang and Junjie Wang and Sanket Purandare and Gokul Nadathur and Stratos Idreos},
   booktitle={The Thirteenth International Conference on Learning Representations},
   year={2025},
   url={https://openreview.net/forum?id=SFN6Wm7YBI}
}
```


## License

Source code is made available under a [BSD 3 license](./LICENSE), however you may have other legal obligations that govern your use of other content linked in this repository, such as the license or terms of service for third-party data and models.

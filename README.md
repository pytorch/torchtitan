<div align="center">

# torchtitan

**A PyTorch-native platform for pretraining and post-training generative AI models at scale**

[![arXiv](https://img.shields.io/badge/arXiv-2410.06511-b31b1b.svg)](https://arxiv.org/abs/2410.06511)
[![ICLR](https://img.shields.io/badge/ICLR-2025-violet.svg)](https://iclr.cc/virtual/2025/poster/29620)
[![devlogs](https://img.shields.io/badge/PyTorch-DevLog-DE3412.svg)](https://docs.pytorch.org/devlogs/)
[![license](https://img.shields.io/badge/license-BSD_3--Clause-lightgrey.svg)](./LICENSE)
[![pip](https://img.shields.io/pypi/v/torchtitan?color=blue)](https://pypi.org/project/torchtitan/)
[![conda](https://img.shields.io/conda/vn/conda-forge/torchtitan?color=green)](https://anaconda.org/conda-forge/torchtitan)

[Quick start](#quick-start) · [Models](#supported-models) · [Features](#features) · [RL](torchtitan/rl/README.md) · [Docs](#documentation) · [Contributing](#contributing)

</div>

`torchtitan` is a clean-room, PyTorch-native implementation of large-scale
training. One readable codebase takes a model from pretraining through
supervised fine-tuning to reinforcement learning, composing FSDP, tensor,
pipeline, context, and expert parallelism without rewriting the model.

- **Composable parallelism.** Models declare how they shard; FSDP2, TP, PP, CP,
  and EP stack into N-D layouts through `DeviceMesh` and DTensor.
- **Built for research.** Minimal, hackable code; every run is a Python
  [recipe](torchtitan/config/README.md) you can read, copy, and modify.
- **Pretraining to RL in one stack.** [TitanRL](torchtitan/rl/README.md) runs the
  same model definitions and kernels in the trainer and in vLLM generation, with
  a bitwise-reproducible batch-invariant mode.
- **Frontier models and techniques.** DeepSeek, Qwen, Kimi, gpt-oss, Llama, and
  FLUX, with MXFP8/NVFP4 low precision, Dist-MoE, CUDA graphs, and more.

> [!NOTE]
> `torchtitan` is under active development. To use the latest features, use a
> recent PyTorch nightly.

## Latest news

- **[2026/08]** [TitanRL](torchtitan/rl) is a hackable RL stack for scaling and debugging. It reuses TorchTitan model definitions and kernels across training and vLLM generation and supports batch-invariant mode.
- **[2025/11]** AMD released an [optimized fork](https://github.com/AMD-AGI/torchtitan-amd/tree/main) of `torchtitan` for AMD GPUs.
- **[2025/10]** We released `torchtitan` [v0.2.0](https://github.com/pytorch/torchtitan/releases).
- **[2025/10]** SkyPilot now supports `torchtitan`! See the [tutorial](https://docs.skypilot.co/en/latest/examples/training/torchtitan.html).

<details>
<summary>Older news</summary>

- **[2025/07]** We published [instructions](torchtitan/models/README.md) on how to add a model to `torchtitan`.
- **[2025/04]** Our paper was accepted by [ICLR 2025](https://iclr.cc/virtual/2025/poster/29620).
- **[2024/12]** GPU MODE [lecture](https://www.youtube.com/watch?v=VYWRjcUqW6w) on torchtitan.
- **[2024/07]** [Presentation](https://pytorch2024.sched.com/event/1fHn3) at PyTorch Conference 2024.

</details>

## Quick start

**1. Install** with a PyTorch nightly (Python 3.11+). Replace `cu132` with your CUDA or ROCm build.

```bash
git clone https://github.com/pytorch/torchtitan && cd torchtitan
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu132
pip install -r requirements.txt
```

**2. Train a debug model.** It uses the tokenizer bundled with the tests, so no download is needed.

```bash
NGPU=1 ./run_train.sh   # Llama 3 debug model, 10 steps
```

**3. Train a real model.** Download the tokenizer (Llama weights require
[access on Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-8B)), then
select a recipe by module and function name:

```bash
python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer --hf_token=...
MODULE=torchtitan_recipes.models.llama3 CONFIG=llama3_8b ./run_train.sh   # 8 GPUs
```

**4. Make it yours.** A recipe is a Python function returning a complete config,
so changing a run means writing a function, not passing flags:

```python
# my_recipes.py
from torchtitan_recipes.models.llama3 import llama3_8b

def llama3_8b_tp2():
    config = llama3_8b()
    config.parallelism.tensor_parallel_degree = 2
    return config
```

```bash
MODULE=my_recipes CONFIG=llama3_8b_tp2 ./run_train.sh
```

See the [configuration guide](torchtitan/config/README.md) for the full model,
and [`torchtitan_recipes`](torchtitan_recipes/README.md) for the verified recipes.

## Supported models

| Model | Sizes | Modality | Verified recipes | RL |
| --- | --- | --- | --- | :---: |
| [Llama 3](torchtitan/models/llama3) | 1B – 405B | Text | `llama3_8b`, `llama3_70b` | ✅ |
| [Qwen3](torchtitan/models/qwen3) | 0.6B – 32B dense; 30B-A3B, 235B-A22B MoE | Text | `qwen3_14b` | ✅ |
| [Qwen3.5](torchtitan/models/qwen3_5) / [3.6](torchtitan/models/qwen3_6) / [3.8](torchtitan/models/qwen3_8) | 0.8B – 397B-A17B; 2.4T-A95B (3.8) | Text, vision | | ✅ text |
| [DeepSeek V3](torchtitan/models/deepseek_v3) | 16B, 236B, 671B | Text | `deepseek_v3_671b` (+ Dist-MoE, NVFP4) | |
| [DeepSeek V4](torchtitan/models/deepseek_v4) | Flash, Pro | Text | | Soon |
| [gpt-oss](torchtitan/models/gpt_oss) | 20B, 120B | Text | | ✅ |
| [Kimi K2.7](torchtitan/models/kimi_k2_7) | Moonlight-16B-A3B, Kimi-VL-A3B, K2.5 | Text, vision | | |
| [Kimi K3](torchtitan/models/kimi_k3) | K3 | Text, vision | | Soon |
| [Muse Glimmer](torchtitan/models/muse_glimmer) | 30B | Text, vision | `muse_glimmer_30b` | ✅ |
| [FLUX](torchtitan/models/flux) | dev, schnell | Text-to-image | `flux_dev`, `flux_schnell` | |

Every model has debug configurations exercised in CI. Verified recipes have
been run in training, convergence, or performance workflows. To add a model, see
the [model guide](torchtitan/models/README.md).

## Features

<table>
<tr>
<td valign="top" width="50%">

**Parallelism**
- [FSDP2](torchtitan/distributed/FSDP.md), HSDP, and DDP
- Tensor parallel, including async TP
- [Pipeline parallel](torchtitan/distributed/PIPELINE_PARALLEL.md): 1F1B, Interleaved 1F1B, GPipe, and other `torch.distributed.pipelining` schedules
- Context parallel: all-gather KV and Ulysses
- Expert parallel with DeepEP / HybridEP dispatch
- [Dist-MoE](torchtitan/models/common/dist_moe/README.md): fused dispatch, expert compute, and combine on Blackwell

**Memory and performance**
- [Activation checkpointing](torchtitan/distributed/REMAT.md): full, selective, and regional
- Regional `torch.compile` and CUDA graphs
- [MXFP8](torchtitan/quantization/mxfp8/README.md) and [NVFP4](torchtitan/quantization/nvfp4/README.md) training
- FlashAttention 3/4, FlexAttention, and varlen attention
- [BF16 optimizer states](torchtitan/components/optim/BF16_OPTIMIZER_STATES.md) and FSDP CPU offload

**Training**
- [Grain data pipeline](torchtitan/components/data/README.md) with packing, mixing, and checkpointable state
- Pretraining, SFT on chat datasets, and LoRA
- Muon (DistMuon), EMA, multi-token prediction
- Warmup-stable-decay LR schedule and [validation](docs/evaluation.md)

</td>
<td valign="top" width="50%">

**Post-training with [TitanRL](torchtitan/rl/README.md)**
- One model definition shared by the trainer and vLLM
- [Bitwise trainer/generator parity](torchtitan/rl/BITWISE_PARITY.md) in batch-invariant mode
- Async rollouts orchestrated with Monarch; trainer-to-generator model state dict transfer over TorchStore
- GRPO and DAPO losses; [DAPO Math](torchtitan/rl/examples/dapo_math), [Search-R1](torchtitan/rl/examples/search_r1), and [Verifiers](torchtitan/rl/examples/verifiers) examples

**Checkpointing**
- [Distributed checkpointing](torchtitan/components/checkpointer/README.md), including async saves
- Load and save Hugging Face safetensors directly
- [Conversion scripts](scripts/checkpoint_conversion) between HF and DCP

**Debugging and observability**
- [Metrics](torchtitan/observability/METRICS.md): loss, memory, throughput, TFLOPs, and MFU in TensorBoard or W&B
- [Profiling, memory snapshots, and Flight Recorder](docs/debugging.md)
- Fake-backend dry runs of large layouts on one GPU
- Deterministic mode and [loss comparison](scripts/loss_compare.py) across commits
- [SDC replay](torchtitan/observability/SILENT_DATA_CORRUPTION.md) and [structured logging](torchtitan/observability/structured_logger/README.md)

**Extensibility**
- Python [recipes](torchtitan/config/README.md) selected with `--module` and `--config`
- [Component overrides](torchtitan/config/OVERRIDE.md) for custom kernels and modules
- [Experiments](torchtitan/experiments): GraphTrainer, TorchFT, Transformers modeling backend

</td>
</tr>
</table>

We report [performance](docs/benchmarks/llama3_h100_202412_torchtitan.md) on up
to 512 GPUs and verify [loss convergence](torchtitan/models/llama3/README.md#loss-convergence)
across parallelisms and techniques.

## Test status

| Hardware | Integration tests | Unit tests |
| --- | --- | --- |
| CPU | | [![CPU Unit Test](https://github.com/pytorch/torchtitan/actions/workflows/unit_test_cpu.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/unit_test_cpu.yaml?query=branch%3Amain) |
| NVIDIA GPU | [![Integration Tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test.yaml?query=branch%3Amain) [![H100 Tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_h100.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_h100.yaml?query=branch%3Amain) [![B200 Tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_b200.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_b200.yaml?query=branch%3Amain) | [![GPU Unit Tests](https://github.com/pytorch/torchtitan/actions/workflows/unit_test_gpu.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/unit_test_gpu.yaml?query=branch%3Amain) |
| AMD GPU (ROCm) | [![Integration Tests (ROCm)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_rocm.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_rocm.yaml?query=branch%3Amain) | |
| RL (NVIDIA GPU) | [![RL Integration Tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_rl.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_rl.yaml?query=branch%3Amain) [![RL B200 Tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_rl_b200.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_rl_b200.yaml?query=branch%3Amain) | [![RL GPU Unit Tests](https://github.com/pytorch/torchtitan/actions/workflows/unit_test_gpu_rl.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/unit_test_gpu_rl.yaml?query=branch%3Amain) |

## Installation

The [quick start](#quick-start) installs from source, which is the recommended
way to get the latest features. Other options:

<details>
<summary><b>Nightly builds</b></summary>

```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu132 --force-reinstall
pip install --pre torchtitan --index-url https://download.pytorch.org/whl/nightly/cu132
```

Replace `cu132` with another CUDA version or a ROCm build (e.g. `rocm10.0`).

</details>

<details>
<summary><b>Stable releases</b></summary>

```bash
pip install torchtitan
# or
conda install conda-forge::torchtitan
```

Each stable release pins the nightly versions of `torch` and `torchao` it was
validated against; see [RELEASE.md](RELEASE.md).

</details>

<details>
<summary><b>Optional dependencies</b></summary>

- **Low precision (MXFP8, NVFP4):** install a `torchao` nightly matching your
  `torch` build. It is not in `requirements.txt` so it cannot be resolved
  against a different `torch`:
  ```bash
  USE_CPP=0 pip install --pre --upgrade torchao --index-url https://download.pytorch.org/whl/nightly/cu132
  ```
- **RL:** `scripts/install_rl_env.sh` installs the vLLM, Monarch, and TorchStore
  nightlies and the FlashAttention build for your GPU. See the
  [RL quick start](torchtitan/rl/README.md#quick-start).
- **Extras** for vision, FLUX, RL, and development are also available as
  `pip install -e ".[vlm,flux,rl,dev]"`.
- **Importing from elsewhere:** the source tree runs as is; to import
  `torchtitan` from another directory, install it in editable mode without
  re-resolving dependencies: `pip install -e . --no-deps`.

</details>

<details>
<summary><b>Multi-node training with Slurm</b></summary>

Submit [`scripts/multinode_trainer.slurm`](scripts/multinode_trainer.slurm) with
`sbatch` from the repository root. Set the node count in both the `#SBATCH`
header and the `torchrun --nnodes` argument:

```bash
#SBATCH --ntasks=2
#SBATCH --nodes=2

srun torchrun --nnodes 2 ...
```

If a node does not have 8 GPUs, also adjust `--nproc_per_node` and
`#SBATCH --gpus-per-task`.

</details>

## Documentation

| Topic | Where |
| --- | --- |
| Configuring runs and writing recipes | [torchtitan/config](torchtitan/config/README.md) |
| Adding a model | [torchtitan/models](torchtitan/models/README.md) |
| Parallelisms and distributed techniques | [torchtitan/distributed](torchtitan/distributed/README.md) |
| Datasets and data loading | [torchtitan/components/data](torchtitan/components/data/README.md) |
| Checkpointing and HF conversion | [torchtitan/components/checkpointer](torchtitan/components/checkpointer/README.md) |
| Reinforcement learning | [torchtitan/rl](torchtitan/rl/README.md) |
| Debugging, reproducibility, and validating numerics | [docs/debugging.md](docs/debugging.md) |
| Validation and evaluation | [docs/evaluation.md](docs/evaluation.md) |
| Benchmarks | [docs/benchmarks](docs/benchmarks) |
| Tests and CI | [tests](tests/README.md) |

## Contributing

We look forward to your contributions!

- New ideas should start in the [`experiments`](torchtitan/experiments) folder;
  follow the [experiments guidelines](torchtitan/experiments/README.md).
- For fixes and contributions to core, follow the [contributing guidelines](CONTRIBUTING.md).

## Citation

Our paper gives a detailed look at the parallelisms and optimizations in
`torchtitan`, with advice on when to use each technique:
[TorchTitan: One-stop PyTorch native solution for production ready LLM pre-training](https://openreview.net/forum?id=SFN6Wm7YBI).

```bibtex
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

Source code is made available under a [BSD 3 license](./LICENSE). You may have
other legal obligations that govern your use of other content linked in this
repository, such as the license or terms of service for third-party data and
models.

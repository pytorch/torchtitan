<div align="center">

# torchtitan

**面向大规模生成式 AI 模型预训练与后训练的原生 PyTorch 平台**

[![arXiv](https://img.shields.io/badge/arXiv-2410.06511-b31b1b.svg)](https://arxiv.org/abs/2410.06511)
[![ICLR](https://img.shields.io/badge/ICLR-2025-violet.svg)](https://iclr.cc/virtual/2025/poster/29620)
[![devlogs](https://img.shields.io/badge/PyTorch-DevLog-DE3412.svg)](https://docs.pytorch.org/devlogs/)
[![license](https://img.shields.io/badge/license-BSD_3--Clause-lightgrey.svg)](./LICENSE)
[![pip](https://img.shields.io/pypi/v/torchtitan?color=blue)](https://pypi.org/project/torchtitan/)
[![conda](https://img.shields.io/conda/vn/conda-forge/torchtitan?color=green)](https://anaconda.org/conda-forge/torchtitan)

<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

[快速上手](#快速上手) · [支持的模型](#支持的模型) · [核心特性](#核心特性) · [强化学习 (RL)](torchtitan/rl/README.md) · [文档](#文档) · [参与贡献](#参与贡献)

</div>

`torchtitan` 是一个全新构建、完全原生基于 PyTorch 的大规模训练平台。单一且清晰易读的代码库即可支持模型从预训练（Pretraining）、监督微调（SFT）到强化学习（RL）的全流程，并能在不重写模型代码的前提下自由组合 FSDP、张量并行（TP）、流水线并行（PP）、上下文并行（CP）与专家并行（EP）。

- **可自由组合的并行化（Composable parallelism）**：模型自行声明分片逻辑；FSDP2、TP、PP、CP 与 EP 通过 `DeviceMesh` 和 DTensor 堆叠组合为任意 N 维分布式布局。
- **专为研究而设计（Built for research）**：极简且高度可定制（hackable）的代码结构；每次训练运行都是一个独立的 Python [配方（recipe）](torchtitan/config/README.md)，可直接阅读、复制和修改。
- **单一技术栈贯通预训练与强化学习（Pretraining to RL in one stack）**：[TitanRL](torchtitan/rl/README.md) 在训练器（Trainer）与 vLLM 生成端复用完全相同的模型定义与计算内核，并支持按位完全可复现的批次无关（batch-invariant）模式。
- **支持前沿模型与尖端技术（Frontier models and techniques）**：原生支持 DeepSeek、Qwen、Kimi、gpt-oss、Llama 与 FLUX，集成 MXFP8/NVFP4 低精度、Dist-MoE、CUDA Graph 等进阶特性。

> [!NOTE]
> `torchtitan` 处于积极迭代开发中。若要体验最新功能，建议使用最新的 PyTorch nightly 版本。

## 最新动态

- **[2026/08]** 推出 [TitanRL](torchtitan/rl)，这是一个用于大规模扩展与调试的高可定制化强化学习技术栈。它在训练器和 vLLM 推理生成间复用 TorchTitan 的模型定义和内核，并支持批次无关（batch-invariant）模式。
- **[2025/11]** AMD 发布了面向 AMD GPU 优化的 `torchtitan` [分支版本](https://github.com/AMD-AGI/torchtitan-amd/tree/main)。
- **[2025/10]** 发布 `torchtitan` [v0.2.0](https://github.com/pytorch/torchtitan/releases)。
- **[2025/10]** SkyPilot 现已支持 `torchtitan`！详见[教程](https://docs.skypilot.co/en/latest/examples/training/torchtitan.html)。

<details>
<summary>历史动态</summary>

- **[2025/07]** 发布了关于如何向 `torchtitan` 添加新模型的[指南说明](torchtitan/models/README.md)。
- **[2025/04]** 论文被 [ICLR 2025](https://iclr.cc/virtual/2025/poster/29620) 录用。
- **[2024/12]** GPU MODE 关于 torchtitan 的公开[讲座](https://www.youtube.com/watch?v=VYWRjcUqW6w)。
- **[2024/07]** 在 PyTorch Conference 2024 上的[演讲展示](https://pytorch2024.sched.com/event/1fHn3)。

</details>

## 快速上手

**1. 安装** PyTorch nightly 版本（需要 Python 3.11+）。请根据你的 CUDA 或 ROCm 构建版本替换 `cu132`。

```bash
git clone https://github.com/pytorch/torchtitan && cd torchtitan
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu132
pip install -r requirements.txt
```

**2. 训练调试模型**。此步骤使用测试集自带的分词器（tokenizer），无需额外下载任何权重：

```bash
NGPU=1 ./run_train.sh   # Llama 3 调试模型，运行 10 步
```

**3. 训练真实模型**。下载对应分词器（Llama 权重需在 [Hugging Face 上申请访问权限](https://huggingface.co/meta-llama/Llama-3.1-8B)），然后通过模块名和函数名选择训练配方（recipe）：

```bash
python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer --hf_token=...
MODULE=torchtitan_recipes.models.llama3 CONFIG=llama3_8b ./run_train.sh   # 8 块 GPU
```

**4. 打造你的专属配方**。配方是一个返回完整配置字典的 Python 函数，因此修改运行参数只需编写一个函数，而无需传递繁琐的命令行参数：

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

关于完整的配置选项，请参阅[配置指南](torchtitan/config/README.md)；已验证的配方列表请参阅 [`torchtitan_recipes`](torchtitan_recipes/README.md)。

## 支持的模型

| 模型 | 规格 | 模态 | 已验证配方 | RL (强化学习) |
| --- | --- | --- | --- | :---: |
| [Llama 3](torchtitan/models/llama3) | 1B – 405B | 文本 | `llama3_8b`, `llama3_70b` | ✅ |
| [Qwen3](torchtitan/models/qwen3) | 0.6B – 32B 稠密; 30B-A3B, 235B-A22B MoE | 文本 | `qwen3_14b` | ✅ |
| [Qwen3.5](torchtitan/models/qwen3_5) / [3.6](torchtitan/models/qwen3_6) / [3.8](torchtitan/models/qwen3_8) | 0.8B – 397B-A17B; 2.4T-A95B (3.8) | 文本、多模态视觉 | | ✅ 仅文本 |
| [DeepSeek V3](torchtitan/models/deepseek_v3) | 16B, 236B, 671B | 文本 | `deepseek_v3_671b` (+ Dist-MoE, NVFP4) | |
| [DeepSeek V4](torchtitan/models/deepseek_v4) | Flash, Pro | 文本 | | 即将推出 |
| [gpt-oss](torchtitan/models/gpt_oss) | 20B, 120B | 文本 | | ✅ |
| [Kimi K2.7](torchtitan/models/kimi_k2_7) | Moonlight-16B-A3B, Kimi-VL-A3B, K2.5 | 文本、多模态视觉 | | |
| [Kimi K3](torchtitan/models/kimi_k3) | K3 | 文本、多模态视觉 | | 即将推出 |
| [Muse Glimmer](torchtitan/models/muse_glimmer) | 30B | 文本、多模态视觉 | `muse_glimmer_30b` | ✅ |
| [FLUX](torchtitan/models/flux) | dev, schnell | 文生图 | `flux_dev`, `flux_schnell` | |

所有模型均具备在 CI 中测试的调试配置。已验证配方均已在训练、收敛性或性能工作流中通过验证。如需添加新模型，请参阅[模型接入指南](torchtitan/models/README.md)。

## 核心特性

<table>
<tr>
<td valign="top" width="50%">

**分布式并行技术**
- [FSDP2](torchtitan/distributed/FSDP.md)、HSDP 与 DDP
- 张量并行（Tensor Parallelism），包括异步 TP
- [流水线并行（Pipeline Parallelism）](torchtitan/distributed/PIPELINE_PARALLEL.md)：1F1B、交错 1F1B（Interleaved 1F1B）、GPipe 及其他 `torch.distributed.pipelining` 调度策略
- 上下文并行（Context Parallelism）：All-Gather KV 与 Ulysses
- 专家并行（Expert Parallelism）：支持 DeepEP / HybridEP 分派
- [Dist-MoE](torchtitan/models/common/dist_moe/README.md)：Blackwell 架构上的融合分派、专家计算与结果合并

**显存与性能优化**
- [激活重计算 / 检查点（Activation Checkpointing）](torchtitan/distributed/REMAT.md)：完全重计算、选择性重计算与区域重计算
- 区域化 `torch.compile` 与 CUDA Graph
- [MXFP8](torchtitan/quantization/mxfp8/README.md) 与 [NVFP4](torchtitan/quantization/nvfp4/README.md) 低精度训练
- FlashAttention 3/4、FlexAttention 与可变长注意力（varlen attention）
- [BF16 优化器状态](torchtitan/components/optim/BF16_OPTIMIZER_STATES.md) 与 FSDP CPU Offload

**训练工作流**
- 基于 [Grain 的数据处理流水线](torchtitan/components/data/README.md)，支持样本拼接（packing）、多源混合与可恢复状态检查点
- 预训练、对话数据集 SFT 监督微调与 LoRA
- Muon (DistMuon)、EMA（指数移动平均）、多 Token 预测（Multi-Token Prediction）
- Warmup-Stable-Decay 学习率调度策略与[验证评估](docs/evaluation.md)

</td>
<td valign="top" width="50%">

**基于 [TitanRL](torchtitan/rl/README.md) 的强化学习后训练**
- 训练器（Trainer）与 vLLM 共享完全相同的单一模型定义
- 批次无关模式下[按位完全一致的训练器/生成端对齐](torchtitan/rl/BITWISE_PARITY.md)
- 由 Monarch 编排的异步 Rollout；基于 TorchStore 实现权重同步
- GRPO 与 DAPO 损失函数；提供 [DAPO Math](torchtitan/rl/examples/dapo_math)、[Search-R1](torchtitan/rl/examples/search_r1) 与 [Verifiers](torchtitan/rl/examples/verifiers) 完整示例

**检查点存取（Checkpointing）**
- [分布式检查点（Distributed Checkpointing, DCP）](torchtitan/components/checkpointer/README.md)，包含异步保存支持
- 原生直接加载与保存 Hugging Face safetensors 格式
- HF 与 DCP 之间的[格式转换脚本](scripts/checkpoint_conversion)

**调试与可观测性**
- [监控指标（Metrics）](torchtitan/observability/METRICS.md)：在 TensorBoard 或 W&B 中记录 Loss、显存占用、吞吐量、TFLOPs 与 MFU
- [性能分析（Profiling）、显存快照与 Flight Recorder](docs/debugging.md)
- 单卡模拟（Fake-backend dry runs）超大规模分布式拓扑
- 确定性模式与跨 Commit 的 [Loss 差异对比工具](scripts/loss_compare.py)
- [静默数据损坏（SDC）重放](torchtitan/observability/SILENT_DATA_CORRUPTION.md) 与[结构化日志](torchtitan/observability/structured_logger/README.md)

**高度可扩展性**
- 通过 `--module` 与 `--config` 自由选择 Python [配方（recipes）](torchtitan/config/README.md)
- 支持[组件覆盖（Component overrides）](torchtitan/config/OVERRIDE.md)，无缝接入自定义算子与模块
- [实验性特性（Experiments）](torchtitan/experiments)：GraphTrainer、TorchFT、Transformers 建模后端

</td>
</tr>
</table>

我们在高达 512 块 GPU 的规模上发布了[性能基准测试报告](docs/benchmarks/llama3_h100_202412_torchtitan.md)，并在多种并行方式与优化技术下全面验证了[损失收敛性（Loss Convergence）](torchtitan/models/llama3/README.md#loss-convergence)。

## 测试状态

| 硬件环境 | 集成测试 | 单元测试 |
| --- | --- | --- |
| CPU | | [![CPU Unit Test](https://github.com/pytorch/torchtitan/actions/workflows/unit_test_cpu.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/unit_test_cpu.yaml?query=branch%3Amain) |
| NVIDIA GPU | [![Integration Tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test.yaml?query=branch%3Amain) [![H100 Tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_h100.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_h100.yaml?query=branch%3Amain) [![B200 Tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_b200.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_b200.yaml?query=branch%3Amain) | [![GPU Unit Tests](https://github.com/pytorch/torchtitan/actions/workflows/unit_test_gpu.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/unit_test_gpu.yaml?query=branch%3Amain) |
| AMD GPU (ROCm) | [![Integration Tests (ROCm)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_rocm.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_rocm.yaml?query=branch%3Amain) | |
| RL (NVIDIA GPU) | [![RL Integration Tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_rl.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_rl.yaml?query=branch%3Amain) [![RL B200 Tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_rl_b200.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_rl_b200.yaml?query=branch%3Amain) | [![RL GPU Unit Tests](https://github.com/pytorch/torchtitan/actions/workflows/unit_test_gpu_rl.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/unit_test_gpu_rl.yaml?query=branch%3Amain) |

## 安装方式

[快速上手](#快速上手)中推荐的从源码安装方式是体验最新特性的最佳途径。其他安装途径包括：

<details>
<summary><b>Nightly 预构建版本</b></summary>

```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu132 --force-reinstall
pip install --pre torchtitan --index-url https://download.pytorch.org/whl/nightly/cu132
```

可以将 `cu132` 替换为其他 CUDA 版本或 ROCm 版本（例如 `rocm10.0`）。

</details>

<details>
<summary><b>稳定版本（Stable Releases）</b></summary>

```bash
pip install torchtitan
# 或
conda install conda-forge::torchtitan
```

每个稳定版本都会锁定对应测试验证过的 `torch` 和 `torchao` nightly 版本；详情请参阅 [RELEASE.md](RELEASE.md)。

</details>

<details>
<summary><b>可选依赖</b></summary>

- **低精度量化训练（MXFP8, NVFP4）：** 安装与当前 `torch` 构建版本匹配的 `torchao` nightly 版本。该依赖未写入 `requirements.txt`，以避免与不同版本的 `torch` 产生冲突：
  ```bash
  USE_CPP=0 pip install --pre --upgrade torchao --index-url https://download.pytorch.org/whl/nightly/cu132
  ```
- **强化学习（RL）：** 运行 `scripts/install_rl_env.sh` 安装 vLLM、Monarch、TorchStore 的 nightly 版本以及适用于当前 GPU 的 FlashAttention 构建版本。详情请参阅 [RL 快速入门](torchtitan/rl/README.md#quick-start)。
- **附加拓展项（Extras）：** 针对视觉模型、FLUX、RL 及开发工具的扩展，可通过 `pip install -e ".[vlm,flux,rl,dev]"` 安装。
- **从外部导入：** 仓库源码开箱即用；若要从其他目录直接导入 `torchtitan`，可以以可编辑模式安装且不重新解析依赖项：`pip install -e . --no-deps`。

</details>

<details>
<summary><b>使用 Slurm 进行多节点训练</b></summary>

在仓库根目录下使用 `sbatch` 提交 [`scripts/multinode_trainer.slurm`](scripts/multinode_trainer.slurm)。在 `#SBATCH` 标头与 `torchrun --nnodes` 参数中同步配置节点数：

```bash
#SBATCH --ntasks=2
#SBATCH --nodes=2

srun torchrun --nnodes 2 ...
```

如果单个节点配备的不是 8 块 GPU，请相应调整 `--nproc_per_node` 与 `#SBATCH --gpus-per-task`。

</details>

## 文档

| 主题 | 路径导航 |
| --- | --- |
| 运行配置与编写配方 | [torchtitan/config](torchtitan/config/README.md) |
| 添加新模型 | [torchtitan/models](torchtitan/models/README.md) |
| 并行策略与分布式技术 | [torchtitan/distributed](torchtitan/distributed/README.md) |
| 数据集与数据加载 | [torchtitan/components/data](torchtitan/components/data/README.md) |
| 检查点存取与 HF 权重转换 | [torchtitan/components/checkpointer](torchtitan/components/checkpointer/README.md) |
| 强化学习（RL） | [torchtitan/rl](torchtitan/rl/README.md) |
| 调试、数值可复现性与数值验证 | [docs/debugging.md](docs/debugging.md) |
| 验证与评测 | [docs/evaluation.md](docs/evaluation.md) |
| 性能基准测试 | [docs/benchmarks](docs/benchmarks) |
| 测试体系与 CI | [tests](tests/README.md) |

## 参与贡献

热烈欢迎社区贡献！

- 新想法与新实验建议先在 [`experiments`](torchtitan/experiments) 目录下发起；请遵循[实验指南规范](torchtitan/experiments/README.md)。
- 核心代码库的 Bug 修复与功能贡献，请遵循[贡献指南](CONTRIBUTING.md)。

## 引用

我们的论文详细介绍了 `torchtitan` 中的并行化技术与系统优化，并对各项技术的适用场景提供了最佳实践指导：
[TorchTitan: One-stop PyTorch native solution for production ready LLM pre-training](https://openreview.net/forum?id=SFN6Wm7YBI)。

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

## 许可证

本仓库源代码在 [BSD 3 许可证](./LICENSE) 下开源。使用本仓库链接的第三方数据与模型时，请遵守其适用的法律条款及服务协议。

---
*注：本文档由社区志愿者维护（贡献者：[@JasonYeYuhe](https://github.com/JasonYeYuhe)），内容与英文官方文档保持同步（更新时间：2026年10月12日）。如有任何疑问或更新建议，欢迎提交 PR 或 Issue。*

# DeepEP Expert Parallelism for TorchTitan

Achieve **near-linear scaling** from single node to 16 nodes (128 GPUs), **33% faster** than TorchTitan's default expert parallelism (14,796 vs 9,930 tok/s/GPU at EP=8). And Almost no throughput degradation when scaling expert parallel degree from 1 → 8, and all numbers below are reported for Qwen3-30B-A3B.

## 🚀 Performance Highlights

### Scaling Results: Qwen3-30B (128 experts, top-k=8)

**Strong Scaling** (fixed batch size, increasing nodes):

| Configuration | Nodes | GPUs | Tokens/sec | TFLOPS | Memory/GPU |
|---------------|-------|------|------------|--------|------------|
| 1 node | 1 | 8 | 14,796 | 341 | 167.93 GiB (94.15%) |
| 2 nodes | 2 | 16 | 14,380 | 331 | 138.75 GiB (77.78%) |
| 4 nodes | 4 | 32 | 14,276 | 329 | 124.58 GiB (69.84%) |
| 8 nodes | 8 | 64 | 14,107 | 325 | 117.50 GiB (65.88%) |
| 16 nodes | 16 | 128 | 13,856 | 319 | 114.78 GiB (64.35%) |

**Weak Scaling** (optimized batch size for 16 nodes):


| LBS | Nodes | GPUs | Tokens/sec | TFLOPS | Memory/GPU | Failure Reason |
|-----|-------|------|------------|--------|------------|----------------|
| 8 | 16 | 128 | 13,856 | 319 | 114.78 GiB (64.35%) | - |
| 10 | 16 | 128 | 14,123 | 326 | 142.21 GiB (79.73%) | - |
| 12 | 16 | 128 | - | - | - | RendezvousConnectionError (C10d store connection lost) |
| 14 | 16 | 128 | - | - | - | RendezvousConnectionError (C10d store connection lost) |
| 16 | 16 | 128 | - | - | - | CUDA OOM (tried to allocate 2.00 GiB, only 820 MiB free) |
| 18 | 16 | 128 | - | - | - | RendezvousConnectionError (C10d store connection lost) |
| 20 | 16 | 128 | - | - | - | CUDA OOM (tried to allocate 2.50 GiB, only 1.06 GiB free) |


With expert parallelism optimized to remain within a single node for the 30B-A3B configuration, throughput is expected to scale near-linearly to 256 GPUs. Any throughput degradation at this scale is attributed to non-expert parallelism factors.



### What does this enable?

With **256 GPUs** :
- Expected aggregate throughput: 3.6M tokens/sec (14,123 × 256 GPUs), translating to 10T tokens/month
- Identified upcoming optimization (not upstream yet): it could bump the throughput by 30%, this means 10T tokens in 20 days, or equivalently **30T tokens in 2 months**. For reference: Qwen3 was trained on 36T tokens

---

## ⚡ Quick Start

### Basic Usage

```bash
# Single-node training (8 GPUs)
NGPU=8 \
CONFIG_FILE="./torchtitan/models/qwen3/train_configs/qwen3_10b_a1b_with_deepep.toml" \
PYTORCH_ALLOC_CONF="expandable_segments:True,max_split_size_mb:128,garbage_collection_threshold:0.8" \
./run_train.sh
```

**Expected performance**: ~14,768 tok/s/GPU, ~340 TFLOPS on 8× B200s

> **Note**: `PYTORCH_ALLOC_CONF` is required because PyTorch's default cache allocator reserves more memory than needed. This setting uses a smaller buffer.

---

## 📝 Configuration

### Enabling DeepEP in Your Model

DeepEP is enabled through the model flavor configuration. Here's how to enable it:

#### Option 1: Use Pre-configured DeepEP Flavor

For Qwen3 models, use the `-deepep` flavor suffix:

```toml
[model]
name = "qwen3"
flavor = "30B-A3B-deepep"  # DeepEP enabled
hf_assets_path = "./assets/hf/Qwen3-32B"
```

**What this does**: Sets `use_deepep=True` in the model's `MoEArgs`, which automatically enables the DeepEP token dispatcher.

#### Option 2: Programmatic Configuration

For custom models, set `use_deepep=True` in your `MoEArgs`:

```python
from torchtitan.models.moe.moe import MoEArgs

moe_args = MoEArgs(
    num_experts=128,
    top_k=8,
    use_deepep=True,  # Enable DeepEP
    # ... other MoE settings
)
```

### Expert Parallelism Configuration

Configure parallelism degrees in the `[parallelism]` section:

```toml
[parallelism]
...
expert_parallel_degree = 8       # Number of GPUs for expert parallelism
```


### Complete Configuration Diff

Here's the minimal set of changes to enable DeepEP on an existing config:

```diff
 [model]
 name = "qwen3"
+flavor = "30B-A3B-deepep"

 [training]
+debug_moe_force_load_balance = true  # For sanity testing (see Tips)

 [parallelism]
-expert_parallel_degree = 1
+expert_parallel_degree = 8
```

### Command-Line Override

Override config values without editing the file:

```bash
NGPU=8 CONFIG_FILE="path/to/config.toml" ./run_train.sh \
  --parallelism.expert_parallel_degree 8 \
  --training.debug_moe_force_load_balance
```

---

## 🔧 Installation

### Prerequisites

- CUDA 12.8+
- Python 3.10+
- PyTorch 2.9.0+ (with CUDA 12.8 support)
- NVIDIA B200 or H100 GPUs (other architectures may work but are untested)

### Step 1: Install NVSHMEM

DeepEP requires NVIDIA NVSHMEM for multi-node communication:

```bash
# Install NVSHMEM (required for DeepEP)
uv pip install nvidia-nvshmem-cu12

# Set environment variables for DeepEP compilation
export NVSHMEM_DIR=$(uv run python -c "import importlib.util; print(importlib.util.find_spec('nvidia.nvshmem').submodule_search_locations[0])")
export LD_LIBRARY_PATH="${NVSHMEM_DIR}/lib:$LD_LIBRARY_PATH"
export PATH="${NVSHMEM_DIR}/bin:$PATH"
```

### Step 2: Install DeepEP

```bash
# Set CUDA architecture for your GPUs
export TORCH_CUDA_ARCH_LIST="10.0"  # For B200
# export TORCH_CUDA_ARCH_LIST="9.0"  # For H100/H800

# Install DeepEP from source
uv pip install git+https://github.com/deepseek-ai/DeepEP.git --no-build-isolation
```

> **Troubleshooting**: If you see `RuntimeError: Failed: CUDA error ... named symbol not found`, ensure:
> 1. NVSHMEM is installed correctly
> 2. `NVSHMEM_DIR` and `LD_LIBRARY_PATH` are set
> 3. `TORCH_CUDA_ARCH_LIST` matches your GPU architecture
>
> See [GitHub Issue #224](https://github.com/deepseek-ai/DeepEP/issues/224#issuecomment-2985783610)

### Step 3: Verify Installation

```bash
python -c "from deep_ep import Buffer, Config; print('DeepEP installed successfully')"
```


---

## 💡 Tips & Best Practices

### Router Load Balancing

**Problem**: MoE routers start with unbalanced token distributions, causing low throughput during initial training steps. If you observe low throughput initially, this is expected. Once the router converges to uniform distribution, throughput should approach the target.

**Solution**: Use `debug_moe_force_load_balance` for testing and initial benchmarking:

```toml
[training]
debug_moe_force_load_balance = true  # Forces uniform token distribution across experts
```

Or via command line:

```bash
./run_train.sh --training.debug_moe_force_load_balance
```

### DeepEP Tuning

The default DeepEP configuration is optimized for **B200 GPUs with EP=8**. If your setup differs, you may need to tune:

**When to tune**:
- Using different GPU hardware (H100/H800/A100)
- Different expert parallel degree (EP ≠ 8)
- Different model architecture (hidden_size, num_experts, top_k)

**How to tune**: See [scripts/deepep/torchtitan_deepep_tune/README.md](../../../scripts/deepep/torchtitan_deepep_tune/README.md)

---


### For Debugging Purposes

The numbers were reproduced with the following dependencies versions:

```
CUDA & GPU:
├─ NVCC: 12.8 (/usr/local/cuda)
├─ Driver: 570.124.06
└─ GPU: NVIDIA B200 (Blackwell)

Python & Core:
├─ Python: 3.10.19
├─ PyTorch: 2.9.0+cu128
│  ├─ Commit: 0fabc3ba44823f257e70ce397d989c8de5e362c1
│  ├─ CUDA compiled: 12.8
│  └─ CUDNN: 91002
├─ Triton: 3.5.0
└─ Numpy: 2.2.6

DeepEP:
├─ Version: Local installation
├─ Commit: bfded34800dfec415b71503f8205181de90b2480
└─ NVSHMEM: 3.3.20
```

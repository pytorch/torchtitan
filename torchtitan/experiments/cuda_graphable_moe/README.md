# CUDA-Graphable MoE with Paged Stashing

End-to-end CUDA graph capture for MoE training with paged stashing to reduce memory consumption from dynamic expert routing.

1. **CUDA-graphable MoE**: HybridEP eliminates CPU-GPU synchronization in expert parallel dispatch, making the entire MoE forward+backward capturable in a CUDA graph.
2. **Paged Stashing**: A memory planning pass (similar to SAC/activation offloading) which reduces the number of buffers used by MoE models when CUDA graph'd. Improves memory consumption from O(layers × hybrid worst case dispatch size) to O(worst case + actual usage).
3. **3-level overflow defense**: Host spillover, cross-rank detection, and retry with buffer growth — mirrors Megatron-LM's approach.

## Setup

**Platform**: 4+ NVIDIA GPUs with NVLink (tested on 4x GB200, CUDA 13.2, aarch64).

### Step 1: Install DeepEP

```bash
cd /tmp && git clone --branch hybrid-ep https://github.com/deepseek-ai/deepep.git
cd /tmp/deepep && CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="10.0" pip install -e .
```

Verify: `python -c "import deep_ep; print(deep_ep.__version__)"`

> **Note**: Adjust `TORCH_CUDA_ARCH_LIST` to your GPU architecture (e.g., `"9.0"` for H100).

### Step 2: Install torchtitan

```bash
cd /workspace/torchtitan && pip install -e .
```

### Step 3: Verify

```bash
CUDA_HOME=/usr/local/cuda NCCL_GRAPH_REGISTER=0 NGPU=4 \
  MODULE=cuda_graphable_moe.deepseek_v3 CONFIG=paged_stash_deepseek_v3_debugmodel \
  ./run_train.sh \
  --compile.mode aot_fx_trace \
  --parallelism.data_parallel_shard_degree=2 \
  --parallelism.tensor_parallel_degree=2 \
  --parallelism.expert_parallel_degree=2 \
  --training.steps=10
```

Expected output:
```
[titan] 2026-05-04 13:58:14,761 - root - INFO - torchtitan version: 0.2.2 (0.0.0 means __version__ is not defined correctly).
[titan] 2026-05-04 13:58:15,359 - root - WARNING - ENV[TORCH_NCCL_ASYNC_ERROR_HANDLING] = 1 will be overridden to 3 based on job config
[titan] 2026-05-04 13:58:15,361 - root - INFO - Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=2, cp=1, tp=2, ep=2
[titan] 2026-05-04 13:58:15,383 - root - INFO - Successfully created meshes with active dimensions: ['batch', 'loss', 'fsdp', 'tp', 'ep', 'efsdp']
[titan] 2026-05-04 13:58:15,384 - root - INFO - [GC] Initial GC collection took 0.00 seconds
[titan] 2026-05-04 13:58:17,108 - root - INFO - Loading tokenizer from tokenizer.json
[titan] 2026-05-04 13:58:17,112 - root - INFO - Preparing c4_test dataset from tests/assets/c4_test
[titan] 2026-05-04 13:58:17,194 - root - INFO - Building graph_trainer/deepseek_v3 debugmodel
[titan] 2026-05-04 13:58:17,219 - root - INFO - CUDA capacity: NVIDIA GB200 with 187.28GiB memory
[titan] 2026-05-04 13:58:17,242 - root - INFO - Total parameter count: dense 23,173,376, sparse 9,840,640, active 28,098,816
[titan] 2026-05-04 13:58:17,242 - root - INFO - Model graph_trainer/deepseek_v3 debugmodel size: 33,014,016 total parameters
[titan] 2026-05-04 13:58:17,242 - root - INFO - Compiling the loss function with torch.compile
[titan] 2026-05-04 13:58:17,311 - root - INFO - Applied Data Parallel (simple_fsdp) (dp mode=fully_shard) to the model
[titan] 2026-05-04 13:58:18,254 - root - INFO - Pre-initialized HybridEP buffer (hidden_dim=256, num_tokens=16384, num_local_experts=4)
[titan] 2026-05-04 13:58:18,254 - root - INFO - aot_fx_trace compile mode: graph capture will happen at training time
[titan] 2026-05-04 13:58:19,607 - root - INFO - Peak FLOPS used for computing MFU: 2.500e+15
[titan] 2026-05-04 13:58:19,607 - root - INFO - CUDA memory usage for model: 0.04GiB(0.02%)
[titan] 2026-05-04 13:58:19,608 - root - WARNING - model.safetensors.index.json not found at hf_assets_path: ./tests/assets/tokenizer/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
[titan] 2026-05-04 13:58:21,899 - root - INFO - Trainer is initialized with local batch size 8, global batch size 16, gradient accumulation steps 1, sequence length 2048, total steps 10 (warmup 2)
[titan] 2026-05-04 13:58:22,031 - root - INFO - Created 1 paged stash buffers (max_tokens=98304, num_expert_modules=5, ops_per_key={(torch.bfloat16, 256): 25}, page_size=64, device=cuda, host_buffer=no)
[titan] 2026-05-04 13:58:22,031 - root - INFO - Graph-based paged SAC enabled
[titan] 2026-05-04 13:58:22,031 - root - INFO - aot_fx_trace mode: paged stash pass will be applied at trace time
[titan] 2026-05-04 13:58:22,031 - root - INFO - Training starts at step 1
[titan] 2026-05-04 13:58:26,712 - root - INFO - Removed 62 aten.detach.default node(s) from the graph
[titan] 2026-05-04 13:58:26,774 - root - INFO - Removed 669 identity view/reshape node(s) from the graph
[titan] 2026-05-04 13:58:26,775 - root - INFO - Removed 66 identity slice node(s)
[titan] 2026-05-04 13:58:26,942 - root - INFO - Applied selective activation checkpointing (SAC) graph pass.
[titan] 2026-05-04 13:58:26,942 - root - INFO -   Forced 5 nodes to MUST_SAVE at layer boundaries
[titan] 2026-05-04 13:58:26,942 - root - INFO -   Layer non-layer: 6 MUST_SAVE, 58 PREFER_RECOMPUTE
[titan] 2026-05-04 13:58:26,942 - root - INFO -   Layer 0: 22 MUST_SAVE, 78 PREFER_RECOMPUTE
[titan] 2026-05-04 13:58:26,942 - root - INFO -   Layer 1: 30 MUST_SAVE, 104 PREFER_RECOMPUTE
[titan] 2026-05-04 13:58:26,942 - root - INFO -   Layer 2: 30 MUST_SAVE, 104 PREFER_RECOMPUTE
[titan] 2026-05-04 13:58:26,942 - root - INFO -   Layer 3: 30 MUST_SAVE, 104 PREFER_RECOMPUTE
[titan] 2026-05-04 13:58:26,942 - root - INFO -   Layer 4: 30 MUST_SAVE, 104 PREFER_RECOMPUTE
[titan] 2026-05-04 13:58:26,942 - root - INFO -   Layer 5: 30 MUST_SAVE, 104 PREFER_RECOMPUTE
[titan] 2026-05-04 13:58:27,056 - root - INFO - Inserted paged stash ops: 10 copy + wait in fwd, 10 pop + wait in bwd
[titan] 2026-05-04 13:58:27,056 - root - INFO -   layers.1: 2 stashed activations
[titan] 2026-05-04 13:58:27,056 - root - INFO -   layers.2: 2 stashed activations
[titan] 2026-05-04 13:58:27,056 - root - INFO -   layers.3: 2 stashed activations
[titan] 2026-05-04 13:58:27,056 - root - INFO -   layers.4: 2 stashed activations
[titan] 2026-05-04 13:58:27,056 - root - INFO -   layers.5: 2 stashed activations
[titan] 2026-05-04 13:58:28,390 - root - INFO - [CUSTOM_CODEGEN] Saving code to /tmp/torchtitan_fx_codegen_0
[titan] 2026-05-04 13:58:28,608 - root - INFO - [CUSTOM_CODEGEN] Dumped new file: /tmp/torchtitan_fx_codegen_0/fx_7c1a0c517192260c_rank0.py
[titan] 2026-05-04 13:58:28,721 - root - INFO - [CUSTOM_CODEGEN] Loaded module from /tmp/torchtitan_fx_codegen_0/fx_7c1a0c517192260c_rank0.py, hash: 7c1a0c51
[titan] 2026-05-04 13:58:28,724 - root - INFO - Applied cudagraph pass.
[titan] 2026-05-04 13:58:41,402 - root - INFO - step:  1  loss:  4.09409  grad_norm:  3.6453  memory:  3.18GiB(1.70%)  tps: 339  tflops: 0.18  mfu: 0.01%
[titan] 2026-05-04 13:58:41,403 - root - INFO - Synchronizing and adjusting timeout for all ProcessGroups to 0:01:40
[titan] 2026-05-04 13:58:41,664 - root - INFO - step:  2  loss:  3.07209  grad_norm:  4.5404  memory:  3.18GiB(1.70%)  tps: 31,371  tflops: 17.03  mfu: 0.68%
[titan] 2026-05-04 13:58:41,729 - root - INFO - step:  3  loss:  2.47806  grad_norm:  4.1407  memory:  3.18GiB(1.70%)  tps: 125,180  tflops: 67.96  mfu: 2.72%
[titan] 2026-05-04 13:58:41,793 - root - INFO - step:  4  loss:  2.42329  grad_norm:  2.8168  memory:  3.18GiB(1.70%)  tps: 129,655  tflops: 70.39  mfu: 2.82%
[titan] 2026-05-04 13:58:41,863 - root - INFO - step:  5  loss:  2.23090  grad_norm:  2.6348  memory:  3.18GiB(1.70%)  tps: 116,451  tflops: 63.23  mfu: 2.53%
[titan] 2026-05-04 13:58:41,935 - root - INFO - step:  6  loss:  2.12690  grad_norm:  2.2734  memory:  3.18GiB(1.70%)  tps: 114,268  tflops: 62.04  mfu: 2.48%
[titan] 2026-05-04 13:58:41,999 - root - INFO - step:  7  loss:  2.02902  grad_norm:  1.9232  memory:  3.18GiB(1.70%)  tps: 129,971  tflops: 70.57  mfu: 2.82%
[titan] 2026-05-04 13:58:42,061 - root - INFO - step:  8  loss:  2.00265  grad_norm:  2.2169  memory:  3.18GiB(1.70%)  tps: 131,613  tflops: 71.46  mfu: 2.86%
[titan] 2026-05-04 13:58:42,123 - root - INFO - step:  9  loss:  2.21702  grad_norm:  2.0541  memory:  3.18GiB(1.70%)  tps: 132,235  tflops: 71.80  mfu: 2.87%
[titan] 2026-05-04 13:58:42,191 - root - INFO - step: 10  loss:  1.93761  grad_norm:  1.7217  memory:  3.18GiB(1.70%)  tps: 121,163  tflops: 65.78  mfu: 2.63%
[titan] 2026-05-04 13:58:42,191 - root - INFO - Sleeping 2 seconds for other ranks to complete
[titan] 2026-05-04 13:58:44,191 - root - INFO - Training completed
[titan] 2026-05-04 13:58:45,710 - root - INFO - Process group destroyed
```

## Why Paged Stashing

### The Problem

MoE expert activations have **dynamic token counts** — the number of tokens
routed to each expert varies per-batch due to learned routing decisions.
Under HybridEP with capacity-factor padding, these activations are oversized
(padded to worst-case). Standard SAC either:

1. **Recomputes** them (expensive: grouped GEMM + SwiGLU per layer per backward step), or
2. **Saves** them (wasteful: each layer must reserve its own worst-case-sized buffer,
   resulting in O(layers × worst case dispatch size) memory consumption — even though
   only one layer executes at a time inside CUDA graphs where the pool is fixed at
   capture time).

### The Solution

Paged stashing replaces per-layer worst-case buffers with a single shared paged
buffer plus one worst-case dispatch buffer, reducing memory from
O(layers × worst case dispatch size) to O(worst case + actual usage). Activations
are stored in the paged buffer managed by Triton kernels, and compact fixed-size
**page_record** handles (int64 tensors encoding page IDs) are saved in the graph.

**Key benefits**:
- **No recomputation**: avoids the cost of re-running grouped GEMM + SwiGLU in backward
- **Reduced fragmentation**: one shared paged buffer across all layers instead of per-layer worst-case allocations (note: internal fragmentation within pages still exists, but is bounded by page size)
- **CUDA graph compatible**: page_record handles are fixed-size, paged buffer addresses
  are stable, Triton kernels are capturable
- **Async stream overlap**: copy/pop kernels run on a dedicated transfer stream
  (ao's `_get_or_create_transfer_stream`), with `ao.wait_tensor` for synchronization

## Experiments

All experiments use `aot_fx_trace` compilation + CUDAGraph + HybridEP on the DeepSeek V3 debugmodel (4 GPUs, DP=2, TP=2, EP=2). SAC and CUDAGraph are always enabled in `aot_fx_trace` mode.

### Experiment 0: Baseline (save only, no paged stash)

Baseline: HybridEP + SAC + CUDAGraph. MoE expert activations are saved as regular tensors (per-layer worst-case buffers).

```bash
CUDA_HOME=/usr/local/cuda NCCL_GRAPH_REGISTER=0 NGPU=4 \
  MODULE=cuda_graphable_moe.deepseek_v3 CONFIG=paged_stash_deepseek_v3_debugmodel \
  ./run_train.sh \
  --compile.mode aot_fx_trace \
  --parallelism.data_parallel_shard_degree=2 \
  --parallelism.tensor_parallel_degree=2 \
  --parallelism.expert_parallel_degree=2 \
  --training.steps=10 \
  --compile.memory_policy=paged_stash_save_only
```

Expected output:
```
[titan] 2026-05-04 14:24:29,601 - root - INFO - torchtitan version: 0.2.2 (0.0.0 means __version__ is not defined correctly).
[titan] 2026-05-04 14:24:30,184 - root - WARNING - ENV[TORCH_NCCL_ASYNC_ERROR_HANDLING] = 1 will be overridden to 3 based on job config
[titan] 2026-05-04 14:24:30,186 - root - INFO - Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=2, cp=1, tp=2, ep=2
[titan] 2026-05-04 14:24:30,208 - root - INFO - Successfully created meshes with active dimensions: ['batch', 'loss', 'fsdp', 'tp', 'ep', 'efsdp']
[titan] 2026-05-04 14:24:30,209 - root - INFO - [GC] Initial GC collection took 0.00 seconds
[titan] 2026-05-04 14:24:31,938 - root - INFO - Loading tokenizer from tokenizer.json
[titan] 2026-05-04 14:24:31,942 - root - INFO - Preparing c4_test dataset from tests/assets/c4_test
[titan] 2026-05-04 14:24:32,077 - root - INFO - Building graph_trainer/deepseek_v3 debugmodel
[titan] 2026-05-04 14:24:32,101 - root - INFO - CUDA capacity: NVIDIA GB200 with 187.28GiB memory
[titan] 2026-05-04 14:24:32,122 - root - INFO - Total parameter count: dense 23,173,376, sparse 9,840,640, active 28,098,816
[titan] 2026-05-04 14:24:32,122 - root - INFO - Model graph_trainer/deepseek_v3 debugmodel size: 33,014,016 total parameters
[titan] 2026-05-04 14:24:32,122 - root - INFO - Compiling the loss function with torch.compile
[titan] 2026-05-04 14:24:32,192 - root - INFO - Applied Data Parallel (simple_fsdp) (dp mode=fully_shard) to the model
[titan] 2026-05-04 14:24:33,222 - root - INFO - Pre-initialized HybridEP buffer (hidden_dim=256, num_tokens=16384, num_local_experts=4)
[titan] 2026-05-04 14:24:33,223 - root - INFO - aot_fx_trace compile mode: graph capture will happen at training time
[titan] 2026-05-04 14:24:34,685 - root - INFO - Peak FLOPS used for computing MFU: 2.500e+15
[titan] 2026-05-04 14:24:34,686 - root - INFO - CUDA memory usage for model: 0.04GiB(0.02%)
[titan] 2026-05-04 14:24:34,686 - root - WARNING - model.safetensors.index.json not found at hf_assets_path: ./tests/assets/tokenizer/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
[titan] 2026-05-04 14:24:36,674 - root - INFO - Trainer is initialized with local batch size 8, global batch size 16, gradient accumulation steps 1, sequence length 2048, total steps 10 (warmup 2)
[titan] 2026-05-04 14:24:36,674 - root - INFO - Training starts at step 1
[titan] 2026-05-04 14:24:43,275 - root - INFO - Removed 62 aten.detach.default node(s) from the graph
[titan] 2026-05-04 14:24:43,342 - root - INFO - Removed 669 identity view/reshape node(s) from the graph
[titan] 2026-05-04 14:24:43,343 - root - INFO - Removed 66 identity slice node(s)
[titan] 2026-05-04 14:24:43,525 - root - INFO - Applied selective activation checkpointing (SAC) graph pass.
[titan] 2026-05-04 14:24:43,525 - root - INFO -   Forced 5 nodes to MUST_SAVE at layer boundaries
[titan] 2026-05-04 14:24:43,525 - root - INFO -   Layer non-layer: 6 MUST_SAVE, 58 PREFER_RECOMPUTE
[titan] 2026-05-04 14:24:43,525 - root - INFO -   Layer 0: 22 MUST_SAVE, 78 PREFER_RECOMPUTE
[titan] 2026-05-04 14:24:43,525 - root - INFO -   Layer 1: 32 MUST_SAVE, 102 PREFER_RECOMPUTE
[titan] 2026-05-04 14:24:43,525 - root - INFO -   Layer 2: 32 MUST_SAVE, 102 PREFER_RECOMPUTE
[titan] 2026-05-04 14:24:43,525 - root - INFO -   Layer 3: 32 MUST_SAVE, 102 PREFER_RECOMPUTE
[titan] 2026-05-04 14:24:43,525 - root - INFO -   Layer 4: 32 MUST_SAVE, 102 PREFER_RECOMPUTE
[titan] 2026-05-04 14:24:43,525 - root - INFO -   Layer 5: 32 MUST_SAVE, 102 PREFER_RECOMPUTE
[titan] 2026-05-04 14:24:45,226 - root - INFO - [CUSTOM_CODEGEN] Saving code to /tmp/torchtitan_fx_codegen_0
[titan] 2026-05-04 14:24:45,458 - root - INFO - [CUSTOM_CODEGEN] Dumped new file: /tmp/torchtitan_fx_codegen_0/fx_e3617c8d825ea2c9_rank0.py
[titan] 2026-05-04 14:24:45,576 - root - INFO - [CUSTOM_CODEGEN] Loaded module from /tmp/torchtitan_fx_codegen_0/fx_e3617c8d825ea2c9_rank0.py, hash: e3617c8d
[titan] 2026-05-04 14:24:45,581 - root - INFO - Applied cudagraph pass.
[titan] 2026-05-04 14:24:55,703 - root - INFO - step:  1  loss:  4.03431  grad_norm:  3.3690  memory:  2.34GiB(1.25%)  tps: 347  tflops: 0.19  mfu: 0.01%
[titan] 2026-05-04 14:24:55,703 - root - INFO - Synchronizing and adjusting timeout for all ProcessGroups to 0:01:40
[titan] 2026-05-04 14:24:55,850 - root - INFO - step:  2  loss:  3.03884  grad_norm:  4.4647  memory:  2.34GiB(1.25%)  tps: 55,850  tflops: 30.32  mfu: 1.21%
[titan] 2026-05-04 14:24:55,915 - root - INFO - step:  3  loss:  2.49329  grad_norm:  3.4046  memory:  2.34GiB(1.25%)  tps: 126,876  tflops: 68.89  mfu: 2.76%
[titan] 2026-05-04 14:24:55,979 - root - INFO - step:  4  loss:  2.40819  grad_norm:  2.7550  memory:  2.34GiB(1.25%)  tps: 127,997  tflops: 69.49  mfu: 2.78%
[titan] 2026-05-04 14:24:56,047 - root - INFO - step:  5  loss:  2.23368  grad_norm:  2.5997  memory:  2.34GiB(1.25%)  tps: 121,253  tflops: 65.83  mfu: 2.63%
[titan] 2026-05-04 14:24:56,116 - root - INFO - step:  6  loss:  2.12019  grad_norm:  2.1802  memory:  2.34GiB(1.25%)  tps: 118,256  tflops: 64.21  mfu: 2.57%
[titan] 2026-05-04 14:24:56,181 - root - INFO - step:  7  loss:  2.04957  grad_norm:  3.0099  memory:  2.34GiB(1.25%)  tps: 126,996  tflops: 68.95  mfu: 2.76%
[titan] 2026-05-04 14:24:56,245 - root - INFO - step:  8  loss:  2.02824  grad_norm:  2.0838  memory:  2.34GiB(1.25%)  tps: 128,149  tflops: 69.58  mfu: 2.78%
[titan] 2026-05-04 14:24:56,309 - root - INFO - step:  9  loss:  2.23342  grad_norm:  1.8721  memory:  2.34GiB(1.25%)  tps: 129,482  tflops: 70.30  mfu: 2.81%
[titan] 2026-05-04 14:24:56,377 - root - INFO - step: 10  loss:  1.97393  grad_norm:  1.7784  memory:  2.34GiB(1.25%)  tps: 119,540  tflops: 64.90  mfu: 2.60%
[titan] 2026-05-04 14:24:56,378 - root - INFO - Sleeping 2 seconds for other ranks to complete
[titan] 2026-05-04 14:24:58,378 - root - INFO - Training completed
[titan] 2026-05-04 14:25:00,514 - root - INFO - Process group destroyed
```

### Experiment 1: Paged stash (default config)

Paged stash enabled. The paged stash pass runs before SAC+remat — it redirects
backward consumers away from eligible activations, so remat naturally ignores them.

```bash
CUDA_HOME=/usr/local/cuda NCCL_GRAPH_REGISTER=0 NGPU=4 \
  MODULE=cuda_graphable_moe.deepseek_v3 CONFIG=paged_stash_deepseek_v3_debugmodel \
  ./run_train.sh \
  --compile.mode aot_fx_trace \
  --parallelism.data_parallel_shard_degree=2 \
  --parallelism.tensor_parallel_degree=2 \
  --parallelism.expert_parallel_degree=2 \
  --training.steps=10 \
  --compile.memory_policy=paged_stash
```

Expected output:
```
[titan] 2026-05-04 13:58:14,761 - root - INFO - torchtitan version: 0.2.2 (0.0.0 means __version__ is not defined correctly).
[titan] 2026-05-04 13:58:15,359 - root - WARNING - ENV[TORCH_NCCL_ASYNC_ERROR_HANDLING] = 1 will be overridden to 3 based on job config
[titan] 2026-05-04 13:58:15,361 - root - INFO - Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=2, cp=1, tp=2, ep=2
[titan] 2026-05-04 13:58:15,383 - root - INFO - Successfully created meshes with active dimensions: ['batch', 'loss', 'fsdp', 'tp', 'ep', 'efsdp']
[titan] 2026-05-04 13:58:15,384 - root - INFO - [GC] Initial GC collection took 0.00 seconds
[titan] 2026-05-04 13:58:17,108 - root - INFO - Loading tokenizer from tokenizer.json
[titan] 2026-05-04 13:58:17,112 - root - INFO - Preparing c4_test dataset from tests/assets/c4_test
[titan] 2026-05-04 13:58:17,194 - root - INFO - Building graph_trainer/deepseek_v3 debugmodel
[titan] 2026-05-04 13:58:17,219 - root - INFO - CUDA capacity: NVIDIA GB200 with 187.28GiB memory
[titan] 2026-05-04 13:58:17,242 - root - INFO - Total parameter count: dense 23,173,376, sparse 9,840,640, active 28,098,816
[titan] 2026-05-04 13:58:17,242 - root - INFO - Model graph_trainer/deepseek_v3 debugmodel size: 33,014,016 total parameters
[titan] 2026-05-04 13:58:17,242 - root - INFO - Compiling the loss function with torch.compile
[titan] 2026-05-04 13:58:17,311 - root - INFO - Applied Data Parallel (simple_fsdp) (dp mode=fully_shard) to the model
[titan] 2026-05-04 13:58:18,254 - root - INFO - Pre-initialized HybridEP buffer (hidden_dim=256, num_tokens=16384, num_local_experts=4)
[titan] 2026-05-04 13:58:18,254 - root - INFO - aot_fx_trace compile mode: graph capture will happen at training time
[titan] 2026-05-04 13:58:19,607 - root - INFO - Peak FLOPS used for computing MFU: 2.500e+15
[titan] 2026-05-04 13:58:19,607 - root - INFO - CUDA memory usage for model: 0.04GiB(0.02%)
[titan] 2026-05-04 13:58:19,608 - root - WARNING - model.safetensors.index.json not found at hf_assets_path: ./tests/assets/tokenizer/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
[titan] 2026-05-04 13:58:21,899 - root - INFO - Trainer is initialized with local batch size 8, global batch size 16, gradient accumulation steps 1, sequence length 2048, total steps 10 (warmup 2)
[titan] 2026-05-04 13:58:22,031 - root - INFO - Created 1 paged stash buffers (max_tokens=98304, num_expert_modules=5, ops_per_key={(torch.bfloat16, 256): 25}, page_size=64, device=cuda, host_buffer=no)
[titan] 2026-05-04 13:58:22,031 - root - INFO - Graph-based paged SAC enabled
[titan] 2026-05-04 13:58:22,031 - root - INFO - aot_fx_trace mode: paged stash pass will be applied at trace time
[titan] 2026-05-04 13:58:22,031 - root - INFO - Training starts at step 1
[titan] 2026-05-04 13:58:26,712 - root - INFO - Removed 62 aten.detach.default node(s) from the graph
[titan] 2026-05-04 13:58:26,774 - root - INFO - Removed 669 identity view/reshape node(s) from the graph
[titan] 2026-05-04 13:58:26,775 - root - INFO - Removed 66 identity slice node(s)
[titan] 2026-05-04 13:58:26,942 - root - INFO - Applied selective activation checkpointing (SAC) graph pass.
[titan] 2026-05-04 13:58:26,942 - root - INFO -   Forced 5 nodes to MUST_SAVE at layer boundaries
[titan] 2026-05-04 13:58:26,942 - root - INFO -   Layer non-layer: 6 MUST_SAVE, 58 PREFER_RECOMPUTE
[titan] 2026-05-04 13:58:26,942 - root - INFO -   Layer 0: 22 MUST_SAVE, 78 PREFER_RECOMPUTE
[titan] 2026-05-04 13:58:26,942 - root - INFO -   Layer 1: 30 MUST_SAVE, 104 PREFER_RECOMPUTE
[titan] 2026-05-04 13:58:26,942 - root - INFO -   Layer 2: 30 MUST_SAVE, 104 PREFER_RECOMPUTE
[titan] 2026-05-04 13:58:26,942 - root - INFO -   Layer 3: 30 MUST_SAVE, 104 PREFER_RECOMPUTE
[titan] 2026-05-04 13:58:26,942 - root - INFO -   Layer 4: 30 MUST_SAVE, 104 PREFER_RECOMPUTE
[titan] 2026-05-04 13:58:26,942 - root - INFO -   Layer 5: 30 MUST_SAVE, 104 PREFER_RECOMPUTE
[titan] 2026-05-04 13:58:27,056 - root - INFO - Inserted paged stash ops: 10 copy + wait in fwd, 10 pop + wait in bwd
[titan] 2026-05-04 13:58:27,056 - root - INFO -   layers.1: 2 stashed activations
[titan] 2026-05-04 13:58:27,056 - root - INFO -   layers.2: 2 stashed activations
[titan] 2026-05-04 13:58:27,056 - root - INFO -   layers.3: 2 stashed activations
[titan] 2026-05-04 13:58:27,056 - root - INFO -   layers.4: 2 stashed activations
[titan] 2026-05-04 13:58:27,056 - root - INFO -   layers.5: 2 stashed activations
[titan] 2026-05-04 13:58:28,390 - root - INFO - [CUSTOM_CODEGEN] Saving code to /tmp/torchtitan_fx_codegen_0
[titan] 2026-05-04 13:58:28,608 - root - INFO - [CUSTOM_CODEGEN] Dumped new file: /tmp/torchtitan_fx_codegen_0/fx_7c1a0c517192260c_rank0.py
[titan] 2026-05-04 13:58:28,721 - root - INFO - [CUSTOM_CODEGEN] Loaded module from /tmp/torchtitan_fx_codegen_0/fx_7c1a0c517192260c_rank0.py, hash: 7c1a0c51
[titan] 2026-05-04 13:58:28,724 - root - INFO - Applied cudagraph pass.
[titan] 2026-05-04 13:58:41,402 - root - INFO - step:  1  loss:  4.09409  grad_norm:  3.6453  memory:  3.18GiB(1.70%)  tps: 339  tflops: 0.18  mfu: 0.01%
[titan] 2026-05-04 13:58:41,403 - root - INFO - Synchronizing and adjusting timeout for all ProcessGroups to 0:01:40
[titan] 2026-05-04 13:58:41,664 - root - INFO - step:  2  loss:  3.07209  grad_norm:  4.5404  memory:  3.18GiB(1.70%)  tps: 31,371  tflops: 17.03  mfu: 0.68%
[titan] 2026-05-04 13:58:41,729 - root - INFO - step:  3  loss:  2.47806  grad_norm:  4.1407  memory:  3.18GiB(1.70%)  tps: 125,180  tflops: 67.96  mfu: 2.72%
[titan] 2026-05-04 13:58:41,793 - root - INFO - step:  4  loss:  2.42329  grad_norm:  2.8168  memory:  3.18GiB(1.70%)  tps: 129,655  tflops: 70.39  mfu: 2.82%
[titan] 2026-05-04 13:58:41,863 - root - INFO - step:  5  loss:  2.23090  grad_norm:  2.6348  memory:  3.18GiB(1.70%)  tps: 116,451  tflops: 63.23  mfu: 2.53%
[titan] 2026-05-04 13:58:41,935 - root - INFO - step:  6  loss:  2.12690  grad_norm:  2.2734  memory:  3.18GiB(1.70%)  tps: 114,268  tflops: 62.04  mfu: 2.48%
[titan] 2026-05-04 13:58:41,999 - root - INFO - step:  7  loss:  2.02902  grad_norm:  1.9232  memory:  3.18GiB(1.70%)  tps: 129,971  tflops: 70.57  mfu: 2.82%
[titan] 2026-05-04 13:58:42,061 - root - INFO - step:  8  loss:  2.00265  grad_norm:  2.2169  memory:  3.18GiB(1.70%)  tps: 131,613  tflops: 71.46  mfu: 2.86%
[titan] 2026-05-04 13:58:42,123 - root - INFO - step:  9  loss:  2.21702  grad_norm:  2.0541  memory:  3.18GiB(1.70%)  tps: 132,235  tflops: 71.80  mfu: 2.87%
[titan] 2026-05-04 13:58:42,191 - root - INFO - step: 10  loss:  1.93761  grad_norm:  1.7217  memory:  3.18GiB(1.70%)  tps: 121,163  tflops: 65.78  mfu: 2.63%
[titan] 2026-05-04 13:58:42,191 - root - INFO - Sleeping 2 seconds for other ranks to complete
[titan] 2026-05-04 13:58:44,191 - root - INFO - Training completed
[titan] 2026-05-04 13:58:45,710 - root - INFO - Process group destroyed
```

### Experiment 2: Host spillover test

Undersized CUDA buffer forces activations to spill to pinned host memory (Level 1 overflow defense).

```bash
CUDA_HOME=/usr/local/cuda NCCL_GRAPH_REGISTER=0 NGPU=4 \
  MODULE=cuda_graphable_moe.deepseek_v3 CONFIG=paged_stash_deepseek_v3_debugmodel \
  ./run_train.sh \
  --compile.mode aot_fx_trace \
  --parallelism.data_parallel_shard_degree=2 \
  --parallelism.tensor_parallel_degree=2 \
  --parallelism.expert_parallel_degree=2 \
  --training.steps=10 \
  --compile.memory_policy=paged_stash_spillover
```

Expected output:
```
[titan] 2026-05-04 14:25:17,832 - root - INFO - torchtitan version: 0.2.2 (0.0.0 means __version__ is not defined correctly).
[titan] 2026-05-04 14:25:18,419 - root - WARNING - ENV[TORCH_NCCL_ASYNC_ERROR_HANDLING] = 1 will be overridden to 3 based on job config
[titan] 2026-05-04 14:25:18,421 - root - INFO - Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=2, cp=1, tp=2, ep=2
[titan] 2026-05-04 14:25:18,439 - root - INFO - Successfully created meshes with active dimensions: ['batch', 'loss', 'fsdp', 'tp', 'ep', 'efsdp']
[titan] 2026-05-04 14:25:18,440 - root - INFO - [GC] Initial GC collection took 0.00 seconds
[titan] 2026-05-04 14:25:20,092 - root - INFO - Loading tokenizer from tokenizer.json
[titan] 2026-05-04 14:25:20,096 - root - INFO - Preparing c4_test dataset from tests/assets/c4_test
[titan] 2026-05-04 14:25:20,231 - root - INFO - Building graph_trainer/deepseek_v3 debugmodel
[titan] 2026-05-04 14:25:20,254 - root - INFO - CUDA capacity: NVIDIA GB200 with 187.28GiB memory
[titan] 2026-05-04 14:25:20,272 - root - INFO - Total parameter count: dense 23,173,376, sparse 9,840,640, active 28,098,816
[titan] 2026-05-04 14:25:20,272 - root - INFO - Model graph_trainer/deepseek_v3 debugmodel size: 33,014,016 total parameters
[titan] 2026-05-04 14:25:20,272 - root - INFO - Compiling the loss function with torch.compile
[titan] 2026-05-04 14:25:20,344 - root - INFO - Applied Data Parallel (simple_fsdp) (dp mode=fully_shard) to the model
[titan] 2026-05-04 14:25:21,337 - root - INFO - Pre-initialized HybridEP buffer (hidden_dim=256, num_tokens=16384, num_local_experts=4)
[titan] 2026-05-04 14:25:21,337 - root - INFO - aot_fx_trace compile mode: graph capture will happen at training time
[titan] 2026-05-04 14:25:22,603 - root - INFO - Peak FLOPS used for computing MFU: 2.500e+15
[titan] 2026-05-04 14:25:22,603 - root - INFO - CUDA memory usage for model: 0.04GiB(0.02%)
[titan] 2026-05-04 14:25:22,604 - root - WARNING - model.safetensors.index.json not found at hf_assets_path: ./tests/assets/tokenizer/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
[titan] 2026-05-04 14:25:25,132 - root - INFO - Trainer is initialized with local batch size 8, global batch size 16, gradient accumulation steps 1, sequence length 2048, total steps 10 (warmup 2)
[titan] 2026-05-04 14:25:25,240 - root - INFO - Created 1 paged stash buffers (max_tokens=98304, num_expert_modules=5, ops_per_key={(torch.bfloat16, 256): 25}, page_size=64, device=cuda, host_buffer=yes)
[titan] 2026-05-04 14:25:25,240 - root - INFO - Graph-based paged SAC enabled
[titan] 2026-05-04 14:25:25,240 - root - INFO - aot_fx_trace mode: paged stash pass will be applied at trace time
[titan] 2026-05-04 14:25:25,240 - root - INFO - Training starts at step 1
[titan] 2026-05-04 14:25:30,815 - root - INFO - Removed 62 aten.detach.default node(s) from the graph
[titan] 2026-05-04 14:25:30,886 - root - INFO - Removed 669 identity view/reshape node(s) from the graph
[titan] 2026-05-04 14:25:30,887 - root - INFO - Removed 66 identity slice node(s)
[titan] 2026-05-04 14:25:31,081 - root - INFO - Applied selective activation checkpointing (SAC) graph pass.
[titan] 2026-05-04 14:25:31,081 - root - INFO -   Forced 5 nodes to MUST_SAVE at layer boundaries
[titan] 2026-05-04 14:25:31,081 - root - INFO -   Layer non-layer: 6 MUST_SAVE, 58 PREFER_RECOMPUTE
[titan] 2026-05-04 14:25:31,081 - root - INFO -   Layer 0: 22 MUST_SAVE, 78 PREFER_RECOMPUTE
[titan] 2026-05-04 14:25:31,081 - root - INFO -   Layer 1: 30 MUST_SAVE, 104 PREFER_RECOMPUTE
[titan] 2026-05-04 14:25:31,081 - root - INFO -   Layer 2: 30 MUST_SAVE, 104 PREFER_RECOMPUTE
[titan] 2026-05-04 14:25:31,081 - root - INFO -   Layer 3: 30 MUST_SAVE, 104 PREFER_RECOMPUTE
[titan] 2026-05-04 14:25:31,081 - root - INFO -   Layer 4: 30 MUST_SAVE, 104 PREFER_RECOMPUTE
[titan] 2026-05-04 14:25:31,081 - root - INFO -   Layer 5: 30 MUST_SAVE, 104 PREFER_RECOMPUTE
[titan] 2026-05-04 14:25:31,217 - root - INFO - Inserted paged stash ops: 10 copy + wait in fwd, 10 pop + wait in bwd
[titan] 2026-05-04 14:25:31,217 - root - INFO -   layers.1: 2 stashed activations
[titan] 2026-05-04 14:25:31,217 - root - INFO -   layers.2: 2 stashed activations
[titan] 2026-05-04 14:25:31,217 - root - INFO -   layers.3: 2 stashed activations
[titan] 2026-05-04 14:25:31,217 - root - INFO -   layers.4: 2 stashed activations
[titan] 2026-05-04 14:25:31,217 - root - INFO -   layers.5: 2 stashed activations
[titan] 2026-05-04 14:25:32,643 - root - INFO - [CUSTOM_CODEGEN] Saving code to /tmp/torchtitan_fx_codegen_0
[titan] 2026-05-04 14:25:32,863 - root - INFO - [CUSTOM_CODEGEN] Dumped new file: /tmp/torchtitan_fx_codegen_0/fx_0c9976ea635ff3f6_rank0.py
[titan] 2026-05-04 14:25:32,976 - root - INFO - [CUSTOM_CODEGEN] Loaded module from /tmp/torchtitan_fx_codegen_0/fx_0c9976ea635ff3f6_rank0.py, hash: 0c9976ea
[titan] 2026-05-04 14:25:32,978 - root - INFO - Applied cudagraph pass.
[titan] 2026-05-04 14:25:44,035 - root - INFO - step:  1  loss:  4.08671  grad_norm:  3.5197  memory:  2.24GiB(1.20%)  tps: 345  tflops: 0.19  mfu: 0.01%
[titan] 2026-05-04 14:25:44,035 - root - INFO - Synchronizing and adjusting timeout for all ProcessGroups to 0:01:40
[titan] 2026-05-04 14:25:44,229 - root - INFO - step:  2  loss:  3.12314  grad_norm:  4.5676  memory:  2.24GiB(1.20%)  tps: 42,286  tflops: 22.96  mfu: 0.92%
[titan] 2026-05-04 14:25:44,294 - root - INFO - step:  3  loss:  2.45927  grad_norm:  2.7225  memory:  2.24GiB(1.20%)  tps: 126,461  tflops: 68.66  mfu: 2.75%
[titan] 2026-05-04 14:25:44,359 - root - INFO - step:  4  loss:  2.36184  grad_norm:  2.5735  memory:  2.24GiB(1.20%)  tps: 126,487  tflops: 68.67  mfu: 2.75%
[titan] 2026-05-04 14:25:44,425 - root - INFO - step:  5  loss:  2.18472  grad_norm:  2.3589  memory:  2.24GiB(1.20%)  tps: 123,546  tflops: 67.08  mfu: 2.68%
[titan] 2026-05-04 14:25:44,496 - root - INFO - step:  6  loss:  2.11336  grad_norm:  2.0354  memory:  2.24GiB(1.20%)  tps: 116,576  tflops: 63.29  mfu: 2.53%
[titan] 2026-05-04 14:25:44,560 - root - INFO - step:  7  loss:  2.04445  grad_norm:  1.9576  memory:  2.24GiB(1.20%)  tps: 128,574  tflops: 69.81  mfu: 2.79%
[titan] 2026-05-04 14:25:44,624 - root - INFO - step:  8  loss:  2.00092  grad_norm:  1.8438  memory:  2.24GiB(1.20%)  tps: 128,578  tflops: 69.81  mfu: 2.79%
[titan] 2026-05-04 14:25:44,686 - root - INFO - step:  9  loss:  2.19546  grad_norm:  1.5113  memory:  2.24GiB(1.20%)  tps: 131,673  tflops: 71.49  mfu: 2.86%
[titan] 2026-05-04 14:25:44,754 - root - INFO - step: 10  loss:  1.94691  grad_norm:  1.6922  memory:  2.24GiB(1.20%)  tps: 121,408  tflops: 65.92  mfu: 2.64%
[titan] 2026-05-04 14:25:44,754 - root - INFO - Sleeping 2 seconds for other ranks to complete
[titan] 2026-05-04 14:25:46,754 - root - INFO - Training completed
[titan] 2026-05-04 14:25:49,146 - root - INFO - Process group destroyed
```

### Experiment 3: Overflow retry test

Extremely undersized buffer triggers full overflow and retry with buffer growth (Level 3 defense).

```bash
CUDA_HOME=/usr/local/cuda NCCL_GRAPH_REGISTER=0 NGPU=4 \
  MODULE=cuda_graphable_moe.deepseek_v3 CONFIG=paged_stash_deepseek_v3_debugmodel \
  ./run_train.sh \
  --compile.mode aot_fx_trace \
  --parallelism.data_parallel_shard_degree=2 \
  --parallelism.tensor_parallel_degree=2 \
  --parallelism.expert_parallel_degree=2 \
  --training.steps=10 \
  --compile.memory_policy=paged_stash_overflow_test
```

Expected output:
```
[titan] 2026-05-04 14:26:13,799 - root - INFO - torchtitan version: 0.2.2 (0.0.0 means __version__ is not defined correctly).
[titan] 2026-05-04 14:26:14,398 - root - WARNING - ENV[TORCH_NCCL_ASYNC_ERROR_HANDLING] = 1 will be overridden to 3 based on job config
[titan] 2026-05-04 14:26:14,399 - root - INFO - Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=2, cp=1, tp=2, ep=2
[titan] 2026-05-04 14:26:14,423 - root - INFO - Successfully created meshes with active dimensions: ['batch', 'loss', 'fsdp', 'tp', 'ep', 'efsdp']
[titan] 2026-05-04 14:26:14,424 - root - INFO - [GC] Initial GC collection took 0.00 seconds
[titan] 2026-05-04 14:26:16,129 - root - INFO - Loading tokenizer from tokenizer.json
[titan] 2026-05-04 14:26:16,134 - root - INFO - Preparing c4_test dataset from tests/assets/c4_test
[titan] 2026-05-04 14:26:16,319 - root - INFO - Building graph_trainer/deepseek_v3 debugmodel
[titan] 2026-05-04 14:26:16,344 - root - INFO - CUDA capacity: NVIDIA GB200 with 187.28GiB memory
[titan] 2026-05-04 14:26:16,361 - root - INFO - Total parameter count: dense 23,173,376, sparse 9,840,640, active 28,098,816
[titan] 2026-05-04 14:26:16,362 - root - INFO - Model graph_trainer/deepseek_v3 debugmodel size: 33,014,016 total parameters
[titan] 2026-05-04 14:26:16,362 - root - INFO - Compiling the loss function with torch.compile
[titan] 2026-05-04 14:26:16,430 - root - INFO - Applied Data Parallel (simple_fsdp) (dp mode=fully_shard) to the model
[titan] 2026-05-04 14:26:17,438 - root - INFO - Pre-initialized HybridEP buffer (hidden_dim=256, num_tokens=16384, num_local_experts=4)
[titan] 2026-05-04 14:26:17,438 - root - INFO - aot_fx_trace compile mode: graph capture will happen at training time
[titan] 2026-05-04 14:26:18,783 - root - INFO - Peak FLOPS used for computing MFU: 2.500e+15
[titan] 2026-05-04 14:26:18,783 - root - INFO - CUDA memory usage for model: 0.04GiB(0.02%)
[titan] 2026-05-04 14:26:18,784 - root - WARNING - model.safetensors.index.json not found at hf_assets_path: ./tests/assets/tokenizer/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
[titan] 2026-05-04 14:26:20,892 - root - INFO - Trainer is initialized with local batch size 8, global batch size 16, gradient accumulation steps 1, sequence length 2048, total steps 10 (warmup 2)
[titan] 2026-05-04 14:26:20,908 - root - INFO - Created 1 paged stash buffers (max_tokens=98304, num_expert_modules=5, ops_per_key={(torch.bfloat16, 256): 25}, page_size=64, device=cuda, host_buffer=no)
[titan] 2026-05-04 14:26:20,908 - root - INFO - Graph-based paged SAC enabled
[titan] 2026-05-04 14:26:20,908 - root - INFO - aot_fx_trace mode: paged stash pass will be applied at trace time
[titan] 2026-05-04 14:26:20,908 - root - INFO - Training starts at step 1
[titan] 2026-05-04 14:26:27,823 - root - INFO - Removed 62 aten.detach.default node(s) from the graph
[titan] 2026-05-04 14:26:27,889 - root - INFO - Removed 669 identity view/reshape node(s) from the graph
[titan] 2026-05-04 14:26:27,890 - root - INFO - Removed 66 identity slice node(s)
[titan] 2026-05-04 14:26:28,068 - root - INFO - Applied selective activation checkpointing (SAC) graph pass.
[titan] 2026-05-04 14:26:28,068 - root - INFO -   Forced 5 nodes to MUST_SAVE at layer boundaries
[titan] 2026-05-04 14:26:28,068 - root - INFO -   Layer non-layer: 6 MUST_SAVE, 58 PREFER_RECOMPUTE
[titan] 2026-05-04 14:26:28,068 - root - INFO -   Layer 0: 22 MUST_SAVE, 78 PREFER_RECOMPUTE
[titan] 2026-05-04 14:26:28,068 - root - INFO -   Layer 1: 30 MUST_SAVE, 104 PREFER_RECOMPUTE
[titan] 2026-05-04 14:26:28,068 - root - INFO -   Layer 2: 30 MUST_SAVE, 104 PREFER_RECOMPUTE
[titan] 2026-05-04 14:26:28,068 - root - INFO -   Layer 3: 30 MUST_SAVE, 104 PREFER_RECOMPUTE
[titan] 2026-05-04 14:26:28,068 - root - INFO -   Layer 4: 30 MUST_SAVE, 104 PREFER_RECOMPUTE
[titan] 2026-05-04 14:26:28,068 - root - INFO -   Layer 5: 30 MUST_SAVE, 104 PREFER_RECOMPUTE
[titan] 2026-05-04 14:26:28,211 - root - INFO - Inserted paged stash ops: 10 copy + wait in fwd, 10 pop + wait in bwd
[titan] 2026-05-04 14:26:28,211 - root - INFO -   layers.1: 2 stashed activations
[titan] 2026-05-04 14:26:28,211 - root - INFO -   layers.2: 2 stashed activations
[titan] 2026-05-04 14:26:28,211 - root - INFO -   layers.3: 2 stashed activations
[titan] 2026-05-04 14:26:28,211 - root - INFO -   layers.4: 2 stashed activations
[titan] 2026-05-04 14:26:28,211 - root - INFO -   layers.5: 2 stashed activations
[titan] 2026-05-04 14:26:29,921 - root - INFO - [CUSTOM_CODEGEN] Saving code to /tmp/torchtitan_fx_codegen_0
[titan] 2026-05-04 14:26:30,161 - root - INFO - [CUSTOM_CODEGEN] Dumped new file: /tmp/torchtitan_fx_codegen_0/fx_0d5fe9e4ccaa9a1f_rank0.py
[titan] 2026-05-04 14:26:30,281 - root - INFO - [CUSTOM_CODEGEN] Loaded module from /tmp/torchtitan_fx_codegen_0/fx_0d5fe9e4ccaa9a1f_rank0.py, hash: 0d5fe9e4
[titan] 2026-05-04 14:26:30,286 - root - INFO - Applied cudagraph pass.
[titan] 2026-05-04 14:26:39,805 - root - WARNING - Paged stash: stash buffer overflow on 2 rank(s).
[titan] 2026-05-04 14:26:39,805 - root - WARNING - Paged stash: retrying step (attempt 2/2).
[titan] 2026-05-04 14:26:39,830 - root - WARNING - PagedStashBuffer grown: 7680 -> 15360 CUDA pages (hidden_size=256, dtype=torch.bfloat16)
[titan] 2026-05-04 14:26:41,599 - root - INFO - step:  1  loss:  4.09682  grad_norm:  3.5324  memory:  2.57GiB(1.37%)  tps: 325  tflops: 0.18  mfu: 0.01%
[titan] 2026-05-04 14:26:41,600 - root - INFO - Synchronizing and adjusting timeout for all ProcessGroups to 0:01:40
[titan] 2026-05-04 14:26:41,764 - root - INFO - step:  2  loss:  3.17136  grad_norm:  4.5833  memory:  2.57GiB(1.37%)  tps: 49,984  tflops: 27.14  mfu: 1.09%
[titan] 2026-05-04 14:26:41,829 - root - INFO - step:  3  loss:  2.53983  grad_norm:  4.2956  memory:  2.38GiB(1.27%)  tps: 125,115  tflops: 67.93  mfu: 2.72%
[titan] 2026-05-04 14:26:41,893 - root - INFO - step:  4  loss:  2.45230  grad_norm:  2.9074  memory:  2.38GiB(1.27%)  tps: 129,271  tflops: 70.19  mfu: 2.81%
[titan] 2026-05-04 14:26:41,957 - root - INFO - step:  5  loss:  2.26690  grad_norm:  2.7325  memory:  2.38GiB(1.27%)  tps: 129,172  tflops: 70.13  mfu: 2.81%
[titan] 2026-05-04 14:26:42,025 - root - INFO - step:  6  loss:  2.16931  grad_norm:  2.4381  memory:  2.38GiB(1.27%)  tps: 120,492  tflops: 65.42  mfu: 2.62%
[titan] 2026-05-04 14:26:42,088 - root - INFO - step:  7  loss:  2.06017  grad_norm:  2.0171  memory:  2.38GiB(1.27%)  tps: 130,491  tflops: 70.85  mfu: 2.83%
[titan] 2026-05-04 14:26:42,150 - root - INFO - step:  8  loss:  2.03083  grad_norm:  2.2023  memory:  2.38GiB(1.27%)  tps: 132,635  tflops: 72.01  mfu: 2.88%
[titan] 2026-05-04 14:26:42,213 - root - INFO - step:  9  loss:  2.25326  grad_norm:  2.1300  memory:  2.38GiB(1.27%)  tps: 130,591  tflops: 70.90  mfu: 2.84%
[titan] 2026-05-04 14:26:42,288 - root - INFO - step: 10  loss:  1.97317  grad_norm:  1.9390  memory:  2.38GiB(1.27%)  tps: 108,788  tflops: 59.06  mfu: 2.36%
[titan] 2026-05-04 14:26:42,288 - root - INFO - Sleeping 2 seconds for other ranks to complete
[titan] 2026-05-04 14:26:44,289 - root - INFO - Training completed
[titan] 2026-05-04 14:26:46,167 - root - INFO - Process group destroyed
```

### Numerics validation

Paged stash produces **identical** loss to baseline SAC (non-computation change).

```bash
# Baseline (SAC with fc1 _grouped_mm saves, no paged stash):
CUDA_HOME=/usr/local/cuda NCCL_GRAPH_REGISTER=0 NGPU=4 \
  MODULE=cuda_graphable_moe.deepseek_v3 CONFIG=paged_stash_deepseek_v3_debugmodel \
  ./run_train.sh \
  --compile.mode aot_fx_trace \
  --parallelism.data_parallel_shard_degree=2 \
  --parallelism.tensor_parallel_degree=2 \
  --parallelism.expert_parallel_degree=2 \
  --training.steps=10 \
  --compile.memory_policy=paged_stash_save_only \
  --debug.seed=42 --debug.deterministic
```

Expected output:
```
[titan] 2026-05-04 14:34:52,373 - root - INFO - torchtitan version: 0.2.2 (0.0.0 means __version__ is not defined correctly).
[titan] 2026-05-04 14:34:52,848 - root - WARNING - ENV[TORCH_NCCL_ASYNC_ERROR_HANDLING] = 1 will be overridden to 3 based on job config
[titan] 2026-05-04 14:34:52,850 - root - INFO - Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=2, cp=1, tp=2, ep=2
[titan] 2026-05-04 14:34:52,870 - root - INFO - Successfully created meshes with active dimensions: ['batch', 'loss', 'fsdp', 'tp', 'ep', 'efsdp']
[titan] 2026-05-04 14:34:52,871 - root - INFO - [GC] Initial GC collection took 0.00 seconds
[titan] 2026-05-04 14:34:52,871 - root - INFO - Deterministic algorithm enabled (expect perf degradation).
[titan] 2026-05-04 14:34:52,882 - root - INFO - Loading tokenizer from tokenizer.json
[titan] 2026-05-04 14:34:52,886 - root - INFO - Preparing c4_test dataset from tests/assets/c4_test
[titan] 2026-05-04 14:34:53,019 - root - INFO - Building graph_trainer/deepseek_v3 debugmodel
[titan] 2026-05-04 14:34:53,043 - root - INFO - CUDA capacity: NVIDIA GB200 with 187.28GiB memory
[titan] 2026-05-04 14:34:53,064 - root - INFO - Total parameter count: dense 23,173,376, sparse 9,840,640, active 28,098,816
[titan] 2026-05-04 14:34:53,064 - root - INFO - Model graph_trainer/deepseek_v3 debugmodel size: 33,014,016 total parameters
[titan] 2026-05-04 14:34:53,064 - root - INFO - Compiling the loss function with torch.compile
[titan] 2026-05-04 14:34:53,130 - root - INFO - Applied Data Parallel (simple_fsdp) (dp mode=fully_shard) to the model
[titan] 2026-05-04 14:34:54,366 - root - INFO - Pre-initialized HybridEP buffer (hidden_dim=256, num_tokens=16384, num_local_experts=4)
[titan] 2026-05-04 14:34:54,366 - root - INFO - aot_fx_trace compile mode: graph capture will happen at training time
[titan] 2026-05-04 14:34:55,576 - root - INFO - Peak FLOPS used for computing MFU: 2.500e+15
[titan] 2026-05-04 14:34:55,577 - root - INFO - CUDA memory usage for model: 0.04GiB(0.02%)
[titan] 2026-05-04 14:34:55,577 - root - WARNING - model.safetensors.index.json not found at hf_assets_path: ./tests/assets/tokenizer/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
[titan] 2026-05-04 14:34:58,641 - root - INFO - Trainer is initialized with local batch size 8, global batch size 16, gradient accumulation steps 1, sequence length 2048, total steps 10 (warmup 2)
[titan] 2026-05-04 14:34:58,641 - root - INFO - Training starts at step 1
[titan] 2026-05-04 14:35:03,518 - root - INFO - Removed 62 aten.detach.default node(s) from the graph
[titan] 2026-05-04 14:35:03,589 - root - INFO - Removed 669 identity view/reshape node(s) from the graph
[titan] 2026-05-04 14:35:03,591 - root - INFO - Removed 66 identity slice node(s)
[titan] 2026-05-04 14:35:03,784 - root - INFO - Applied selective activation checkpointing (SAC) graph pass.
[titan] 2026-05-04 14:35:03,784 - root - INFO -   Forced 5 nodes to MUST_SAVE at layer boundaries
[titan] 2026-05-04 14:35:03,784 - root - INFO -   Layer non-layer: 6 MUST_SAVE, 58 PREFER_RECOMPUTE
[titan] 2026-05-04 14:35:03,784 - root - INFO -   Layer 0: 23 MUST_SAVE, 107 PREFER_RECOMPUTE
[titan] 2026-05-04 14:35:03,784 - root - INFO -   Layer 1: 33 MUST_SAVE, 131 PREFER_RECOMPUTE
[titan] 2026-05-04 14:35:03,784 - root - INFO -   Layer 2: 33 MUST_SAVE, 131 PREFER_RECOMPUTE
[titan] 2026-05-04 14:35:03,784 - root - INFO -   Layer 3: 33 MUST_SAVE, 131 PREFER_RECOMPUTE
[titan] 2026-05-04 14:35:03,784 - root - INFO -   Layer 4: 33 MUST_SAVE, 131 PREFER_RECOMPUTE
[titan] 2026-05-04 14:35:03,784 - root - INFO -   Layer 5: 33 MUST_SAVE, 131 PREFER_RECOMPUTE
[titan] 2026-05-04 14:35:05,268 - root - INFO - [CUSTOM_CODEGEN] Saving code to /tmp/torchtitan_fx_codegen_0
[titan] 2026-05-04 14:35:05,529 - root - INFO - [CUSTOM_CODEGEN] Dumped new file: /tmp/torchtitan_fx_codegen_0/fx_cbeded9119e4bba1_rank0.py
[titan] 2026-05-04 14:35:05,655 - root - INFO - [CUSTOM_CODEGEN] Loaded module from /tmp/torchtitan_fx_codegen_0/fx_cbeded9119e4bba1_rank0.py, hash: cbeded91
[titan] 2026-05-04 14:35:05,657 - root - INFO - Applied cudagraph pass.
[titan] 2026-05-04 14:35:16,843 - root - INFO - step:  1  loss:  3.99615  grad_norm:  3.7530  memory: 11.36GiB(6.06%)  tps: 345  tflops: 0.19  mfu: 0.01%
[titan] 2026-05-04 14:35:16,844 - root - INFO - Synchronizing and adjusting timeout for all ProcessGroups to 0:01:40
[titan] 2026-05-04 14:35:17,175 - root - INFO - step:  2  loss:  3.05336  grad_norm:  4.2603  memory: 11.36GiB(6.07%)  tps: 24,733  tflops: 13.43  mfu: 0.54%
[titan] 2026-05-04 14:35:17,310 - root - INFO - step:  3  loss:  2.57188  grad_norm:  3.2958  memory: 11.36GiB(6.07%)  tps: 61,139  tflops: 33.19  mfu: 1.33%
[titan] 2026-05-04 14:35:17,446 - root - INFO - step:  4  loss:  2.45969  grad_norm:  2.8077  memory: 11.36GiB(6.07%)  tps: 60,128  tflops: 32.65  mfu: 1.31%
[titan] 2026-05-04 14:35:17,592 - root - INFO - step:  5  loss:  2.29445  grad_norm:  2.7451  memory: 11.36GiB(6.07%)  tps: 56,352  tflops: 30.60  mfu: 1.22%
[titan] 2026-05-04 14:35:17,729 - root - INFO - step:  6  loss:  2.19078  grad_norm:  2.4130  memory: 11.36GiB(6.07%)  tps: 59,679  tflops: 32.40  mfu: 1.30%
[titan] 2026-05-04 14:35:17,861 - root - INFO - step:  7  loss:  2.09918  grad_norm:  2.1302  memory: 11.36GiB(6.07%)  tps: 62,186  tflops: 33.76  mfu: 1.35%
[titan] 2026-05-04 14:35:17,993 - root - INFO - step:  8  loss:  2.07602  grad_norm:  2.1352  memory: 11.36GiB(6.07%)  tps: 62,562  tflops: 33.97  mfu: 1.36%
[titan] 2026-05-04 14:35:18,123 - root - INFO - step:  9  loss:  2.25745  grad_norm:  1.9888  memory: 11.36GiB(6.07%)  tps: 62,745  tflops: 34.07  mfu: 1.36%
[titan] 2026-05-04 14:35:18,260 - root - INFO - step: 10  loss:  2.02040  grad_norm:  1.9490  memory: 11.36GiB(6.07%)  tps: 60,056  tflops: 32.61  mfu: 1.30%
[titan] 2026-05-04 14:35:18,260 - root - INFO - Sleeping 2 seconds for other ranks to complete
[titan] 2026-05-04 14:35:20,260 - root - INFO - Training completed
[titan] 2026-05-04 14:35:22,300 - root - INFO - Process group destroyed
```

```bash
# Paged stash (same model, same seed):
CUDA_HOME=/usr/local/cuda NCCL_GRAPH_REGISTER=0 NGPU=4 \
  MODULE=cuda_graphable_moe.deepseek_v3 CONFIG=paged_stash_deepseek_v3_debugmodel \
  ./run_train.sh \
  --compile.mode aot_fx_trace \
  --parallelism.data_parallel_shard_degree=2 \
  --parallelism.tensor_parallel_degree=2 \
  --parallelism.expert_parallel_degree=2 \
  --training.steps=10 \
  --compile.memory_policy=paged_stash \
  --debug.seed=42 --debug.deterministic
```

Expected output:
```
[titan] 2026-05-04 14:36:29,237 - root - INFO - torchtitan version: 0.2.2 (0.0.0 means __version__ is not defined correctly).
[titan] 2026-05-04 14:36:29,848 - root - WARNING - ENV[TORCH_NCCL_ASYNC_ERROR_HANDLING] = 1 will be overridden to 3 based on job config
[titan] 2026-05-04 14:36:29,849 - root - INFO - Building device mesh with parallelism: pp=1, dp_replicate=1, dp_shard=2, cp=1, tp=2, ep=2
[titan] 2026-05-04 14:36:29,867 - root - INFO - Successfully created meshes with active dimensions: ['batch', 'loss', 'fsdp', 'tp', 'ep', 'efsdp']
[titan] 2026-05-04 14:36:29,868 - root - INFO - [GC] Initial GC collection took 0.00 seconds
[titan] 2026-05-04 14:36:29,868 - root - INFO - Deterministic algorithm enabled (expect perf degradation).
[titan] 2026-05-04 14:36:29,878 - root - INFO - Loading tokenizer from tokenizer.json
[titan] 2026-05-04 14:36:29,882 - root - INFO - Preparing c4_test dataset from tests/assets/c4_test
[titan] 2026-05-04 14:36:29,966 - root - INFO - Building graph_trainer/deepseek_v3 debugmodel
[titan] 2026-05-04 14:36:29,990 - root - INFO - CUDA capacity: NVIDIA GB200 with 187.28GiB memory
[titan] 2026-05-04 14:36:30,013 - root - INFO - Total parameter count: dense 23,173,376, sparse 9,840,640, active 28,098,816
[titan] 2026-05-04 14:36:30,013 - root - INFO - Model graph_trainer/deepseek_v3 debugmodel size: 33,014,016 total parameters
[titan] 2026-05-04 14:36:30,013 - root - INFO - Compiling the loss function with torch.compile
[titan] 2026-05-04 14:36:30,071 - root - INFO - Applied Data Parallel (simple_fsdp) (dp mode=fully_shard) to the model
[titan] 2026-05-04 14:36:31,321 - root - INFO - Pre-initialized HybridEP buffer (hidden_dim=256, num_tokens=16384, num_local_experts=4)
[titan] 2026-05-04 14:36:31,321 - root - INFO - aot_fx_trace compile mode: graph capture will happen at training time
[titan] 2026-05-04 14:36:32,716 - root - INFO - Peak FLOPS used for computing MFU: 2.500e+15
[titan] 2026-05-04 14:36:32,717 - root - INFO - CUDA memory usage for model: 0.04GiB(0.02%)
[titan] 2026-05-04 14:36:32,718 - root - WARNING - model.safetensors.index.json not found at hf_assets_path: ./tests/assets/tokenizer/model.safetensors.index.json.                     Defaulting to saving a single safetensors file if checkpoint is saved in HF format
[titan] 2026-05-04 14:36:34,877 - root - INFO - Trainer is initialized with local batch size 8, global batch size 16, gradient accumulation steps 1, sequence length 2048, total steps 10 (warmup 2)
[titan] 2026-05-04 14:36:35,042 - root - INFO - Created 1 paged stash buffers (max_tokens=98304, num_expert_modules=5, ops_per_key={(torch.bfloat16, 256): 25}, page_size=64, device=cuda, host_buffer=no)
[titan] 2026-05-04 14:36:35,042 - root - INFO - Graph-based paged SAC enabled
[titan] 2026-05-04 14:36:35,042 - root - INFO - aot_fx_trace mode: paged stash pass will be applied at trace time
[titan] 2026-05-04 14:36:35,042 - root - INFO - Training starts at step 1
[titan] 2026-05-04 14:36:39,953 - root - INFO - Removed 62 aten.detach.default node(s) from the graph
[titan] 2026-05-04 14:36:40,025 - root - INFO - Removed 669 identity view/reshape node(s) from the graph
[titan] 2026-05-04 14:36:40,026 - root - INFO - Removed 66 identity slice node(s)
[titan] 2026-05-04 14:36:40,221 - root - INFO - Applied selective activation checkpointing (SAC) graph pass.
[titan] 2026-05-04 14:36:40,221 - root - INFO -   Forced 5 nodes to MUST_SAVE at layer boundaries
[titan] 2026-05-04 14:36:40,221 - root - INFO -   Layer non-layer: 6 MUST_SAVE, 58 PREFER_RECOMPUTE
[titan] 2026-05-04 14:36:40,221 - root - INFO -   Layer 0: 23 MUST_SAVE, 107 PREFER_RECOMPUTE
[titan] 2026-05-04 14:36:40,221 - root - INFO -   Layer 1: 31 MUST_SAVE, 133 PREFER_RECOMPUTE
[titan] 2026-05-04 14:36:40,221 - root - INFO -   Layer 2: 31 MUST_SAVE, 133 PREFER_RECOMPUTE
[titan] 2026-05-04 14:36:40,221 - root - INFO -   Layer 3: 31 MUST_SAVE, 133 PREFER_RECOMPUTE
[titan] 2026-05-04 14:36:40,221 - root - INFO -   Layer 4: 31 MUST_SAVE, 133 PREFER_RECOMPUTE
[titan] 2026-05-04 14:36:40,221 - root - INFO -   Layer 5: 31 MUST_SAVE, 133 PREFER_RECOMPUTE
[titan] 2026-05-04 14:36:40,346 - root - INFO - Inserted paged stash ops: 10 copy + wait in fwd, 10 pop + wait in bwd
[titan] 2026-05-04 14:36:40,346 - root - INFO -   layers.1: 2 stashed activations
[titan] 2026-05-04 14:36:40,347 - root - INFO -   layers.2: 2 stashed activations
[titan] 2026-05-04 14:36:40,347 - root - INFO -   layers.3: 2 stashed activations
[titan] 2026-05-04 14:36:40,347 - root - INFO -   layers.4: 2 stashed activations
[titan] 2026-05-04 14:36:40,347 - root - INFO -   layers.5: 2 stashed activations
[titan] 2026-05-04 14:36:41,837 - root - INFO - [CUSTOM_CODEGEN] Saving code to /tmp/torchtitan_fx_codegen_0
[titan] 2026-05-04 14:36:42,113 - root - INFO - [CUSTOM_CODEGEN] Dumped new file: /tmp/torchtitan_fx_codegen_0/fx_2c36023b9a7ad7aa_rank0.py
[titan] 2026-05-04 14:36:42,241 - root - INFO - [CUSTOM_CODEGEN] Loaded module from /tmp/torchtitan_fx_codegen_0/fx_2c36023b9a7ad7aa_rank0.py, hash: 2c36023b
[titan] 2026-05-04 14:36:42,246 - root - INFO - Applied cudagraph pass.
[titan] 2026-05-04 14:36:53,952 - root - INFO - step:  1  loss:  3.99615  grad_norm:  3.7530  memory: 12.36GiB(6.60%)  tps: 342  tflops: 0.19  mfu: 0.01%
[titan] 2026-05-04 14:36:53,953 - root - INFO - Synchronizing and adjusting timeout for all ProcessGroups to 0:01:40
[titan] 2026-05-04 14:36:54,281 - root - INFO - step:  2  loss:  3.05336  grad_norm:  4.2603  memory: 12.36GiB(6.60%)  tps: 24,961  tflops: 13.55  mfu: 0.54%
[titan] 2026-05-04 14:36:54,417 - root - INFO - step:  3  loss:  2.57188  grad_norm:  3.2958  memory: 12.36GiB(6.60%)  tps: 60,305  tflops: 32.74  mfu: 1.31%
[titan] 2026-05-04 14:36:54,556 - root - INFO - step:  4  loss:  2.45969  grad_norm:  2.8077  memory: 12.36GiB(6.60%)  tps: 59,166  tflops: 32.12  mfu: 1.28%
[titan] 2026-05-04 14:36:54,691 - root - INFO - step:  5  loss:  2.29445  grad_norm:  2.7451  memory: 12.36GiB(6.60%)  tps: 60,667  tflops: 32.94  mfu: 1.32%
[titan] 2026-05-04 14:36:54,830 - root - INFO - step:  6  loss:  2.19078  grad_norm:  2.4130  memory: 12.36GiB(6.60%)  tps: 58,804  tflops: 31.93  mfu: 1.28%
[titan] 2026-05-04 14:36:54,964 - root - INFO - step:  7  loss:  2.09918  grad_norm:  2.1302  memory: 12.36GiB(6.60%)  tps: 61,296  tflops: 33.28  mfu: 1.33%
[titan] 2026-05-04 14:36:55,098 - root - INFO - step:  8  loss:  2.07602  grad_norm:  2.1352  memory: 12.36GiB(6.60%)  tps: 61,581  tflops: 33.43  mfu: 1.34%
[titan] 2026-05-04 14:36:55,232 - root - INFO - step:  9  loss:  2.25745  grad_norm:  1.9888  memory: 12.36GiB(6.60%)  tps: 61,176  tflops: 33.21  mfu: 1.33%
[titan] 2026-05-04 14:36:55,370 - root - INFO - step: 10  loss:  2.02040  grad_norm:  1.9490  memory: 12.36GiB(6.60%)  tps: 59,202  tflops: 32.14  mfu: 1.29%
[titan] 2026-05-04 14:36:55,371 - root - INFO - Sleeping 2 seconds for other ranks to complete
[titan] 2026-05-04 14:36:57,371 - root - INFO - Training completed
[titan] 2026-05-04 14:36:59,260 - root - INFO - Process group destroyed
```

Both runs produce identical loss and grad_norm at every step (verified: 10 steps, 5-digit match at stdout precision).

## How It Works

### HybridEP: CUDA-graph-compatible MoE dispatch

Standard EP dispatch requires CPU-GPU sync to learn per-rank token counts. HybridEP pre-sizes the buffer using a capacity factor:

```
num_permuted_tokens = num_tokens * ep_size * min(num_local_experts, top_k) * capacity_factor
```

With `capacity_factor=1.0`, the buffer is worst-case sized. No D2H sync needed.

### Paged stash: joint-graph pass

1. **Op-target matching**: The graph pass identifies stash-eligible `_grouped_mm` nodes by matching `aten._grouped_mm.default` whose output feeds `silu` (gate projection) or feeds a `mul` whose other operand is a `silu` output (up projection, i.e. the SwiGLU gate `silu(fc1_1) * fc1_2`). The down projection (w2\*h) feeds a different `mul` (combine with shared experts) and is excluded. No annotations or monkey-patching required.

2. **Graph pass** (`apply_paged_stash_pass`): Operates on the joint fwd+bwd graph in `aot_fx_trace` mode. Uses `autograd_backward` metadata for fwd/bwd classification. For each eligible forward node:
   - Inserts `paged_stash.copy` + `ao.wait_tensor(page_record, keepalive=activation)` after the producer
   - Inserts `paged_stash.pop` + `ao.wait_tensor(pop_output)` before backward consumers
   - Redirects backward consumers via `replace_input_with`

3. **After surgery**: The large activation has no backward users. `remat_using_tags_for_fwd_loss_bwd_graph` naturally ignores these nodes.

4. **Stream overlap**: Copy/pop ops use ao's `_get_or_create_transfer_stream` internally. Triton kernels launch on the transfer stream; `ao.wait_tensor` synchronizes the compute stream. Captured into CUDA graphs.

5. **Buffer access**: Via `_PAGED_STASH_REGISTRY[buffer_id]` inside the op implementations — the graph only carries integer `buffer_id` constants.

### 3-level overflow defense

Mirrors Megatron-LM's approach for handling routing skew:

| Level | Mechanism | Trigger | Effect |
|---|---|---|---|
| 1 | Host spillover | CUDA pages exhausted | Triton kernel copies to pinned host; warning logged |
| 2 | Cross-rank detection | Any rank overflows/over-budget | `all_reduce(SUM)` of 3 flags ensures all ranks agree |
| 3 | Retry | Both CUDA + host exhausted, or HybridEP over-budget | Zero grads, grow buffers 2x, reset CUDA graphs, rerun step |

### Buffer sizing

```python
estimated_tokens = max_tokens / capacity_factor   # balanced estimate
cuda_tokens = estimated_tokens * buffer_size_factor * num_ops
host_tokens = estimated_tokens * host_buffer_size_factor * num_ops  # 0 = off
```

`page_record` format: `[num_tokens, spilled_to_host, page_id_0, page_id_1, ...]`

## File Structure

```
cuda_graphable_moe/
├── README.md                   # This file
├── paged_stashing_guide.md     # In-depth technical guide (Megatron comparison, design rationale)
├── configs.py                  # PagedStashActivationCheckpointConfig
├── train.py                    # PagedStashTrainer — overflow detection + retry loop
├── paged_stash_ops.py          # Triton kernels, PagedStashBuffer, _PAGED_STASH_REGISTRY,
│                               #   custom ops (paged_stash::copy/pop), ao stream integration
├── paged_stash_graph_pass.py   # Joint-graph pass (apply_paged_stash_pass) + utility passes
└── deepseek_v3/
    ├── __init__.py             # Model registry
    └── config_registry.py      # Pre-built configs with hybridep defaults
```

## Configuration

### Environment Variables

| Variable | Required | Description |
|---|---|---|
| `CUDA_HOME` | Yes | Path to CUDA toolkit (e.g., `/usr/local/cuda`) for DeepEP JIT |
| `NCCL_GRAPH_REGISTER` | No | Set to `0` to disable NCCL graph registration if needed |

### Memory Policies

Buffer sizing and overflow behavior are controlled by `--compile.memory_policy`:

| Policy | Description |
|---|---|
| `paged_stash` | Default: CUDA buffer at 1.1x estimated tokens, overflow detection + retry |
| `paged_stash_save_only` | Baseline: no paged buffers, SAC saves fc1 activations as regular tensors |
| `paged_stash_spillover` | Tight CUDA buffer (0.3x) with pinned host spillover |
| `paged_stash_overflow_test` | Tiny CUDA buffer (0.2x) to trigger overflow → retry → grow |

### Default Config Settings

The default config (`paged_stash_deepseek_v3_debugmodel`) uses `aot_fx_trace` mode.
SAC and CUDAGraph are built into `construct_default_graph_passes`; the paged stash
pass is injected by `PagedStashTrainer` at trace time.

| Setting | Value |
|---|---|
| `compile.mode` | `aot_fx_trace` (via `--compile.mode aot_fx_trace`) |
| `parallelism.expert_parallel_comm_backend` | `"hybridep"` |
| `non_blocking_capacity_factor` | `1.0` |

## Available Configs

| Config | Description |
|---|---|
| `paged_stash_deepseek_v3_debugmodel` | Debug-scale model (default) |
| `paged_stash_deepseek_v3_671b` | DeepSeek V3 671B |

## Further Reading

See [Scalable Training of Mixture-of-Experts Models with Megatron Core](https://arxiv.org/html/2603.07685v2).

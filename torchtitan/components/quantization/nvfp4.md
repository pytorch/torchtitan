## Experimental NVFP4 Training on Blackwell GPUs

NVFP4 training dynamically quantizes linear GEMM activations, weights, and
gradients to NVFP4 through TorchAO's training prototype. The model weights and
distributed collectives remain in bf16. This reduces memory use and can improve
throughput on NVIDIA Blackwell GPUs.

> [!WARNING]
> NVFP4 training is experimental. It depends on a TorchAO prototype and has no
> backward-compatibility guarantees. NVFP4 training is not guarded by CI; the
> results below are evidence from one 200M-token training run, not broad
> numerical or performance validation.

### Requirements

- NVIDIA Blackwell SM100 or later GPU with CUDA.
- A PyTorch and TorchAO build that provides
  `torchao.prototype.moe_training.nvfp4_training`.
- `torch.compile` for competitive performance. The provided
  `llama3_8b_nvfp4_mixed` recipe enables model compilation automatically.
- Local GEMM dimensions divisible by 128. The LM head is therefore kept in
  bf16 because its vocabulary dimension does not meet this requirement.

### Llama 3 8B Usage

Use the `llama3_8b_nvfp4_mixed` config for the supported Llama 3 8B recipe:

```bash
torchrun --standalone --nproc_per_node 4 \
  -m torchtitan.train \
  --module llama3 \
  --config llama3_8b_nvfp4_mixed \
  --parallelism.tensor_parallel_degree 1 \
  --parallelism.data_parallel_shard_degree 4 \
  --training.local_batch_size 32 \
  --training.seq_len 2048 \
  --training.steps 763 \
  --dataloader.dataset c4 \
  --metrics.log_freq 10 \
  --optimizer.param-groups.0.optimizer-kwargs.lr 0.0003 \
  --hf-assets-path ./tests/assets/tokenizer
```

The mixed recipe follows [Pretraining Large Language Models with NVFP4](https://arxiv.org/abs/2509.25149). It converts the leading 85% of decoder layers to NVFP4 and leaves the final 15% plus the LM head in bf16 for stability. For the 32-layer Llama 3 8B model, layers 0-26 use NVFP4 and layers 27-31 remain bf16.

### 200M-Token Evidence

The following eager + `torch.compile` runs trained Llama 3 8B on C4 for 200M
tokens with global batch size 128 (local batch size 32, FSDP degree 4). Results
are reported at step 760.

| Run | Loss | Tokens/sec (per GPU) | Memory |
| --- | ---: | ---: | ---: |
| NVFP4 (bf16 tail) | 1.2715 | 30,040 | 110.95 GiB (60.2%) |
| NVFP4 (full) | 1.2687 | 31,758 | 103.00 GiB (55.9%) |
| MXFP8 | 1.2671 | 28,084 | 179.94 GiB (97.6%) |
| BF16 | 1.2738 | 21,919 | 174.49 GiB (94.7%) |

The bf16 tail costs about 5% throughput and 8 GiB compared with full NVFP4,
while remaining faster and using less memory than the MXFP8 and bf16 baselines
in this run. Its final loss is on par with the compared precisions.

![Llama 3 8B NVFP4, MXFP8, and BF16 training loss curves](../../../assets/images/nvfp4_vs_mxfp8_vs_bf16_eager_compile_200m_tokens.png)

*Llama 3 8B training loss through 200M tokens at global batch size 128. All runs use eager execution with model compilation.*

### Versioned Environment

These results and instructions use the container's current upstream builds:

- PyTorch: `2.14.0a0+gitd9abf9e`
- TorchAO: `0.18.0+gitcb76f29`
- TorchTitan: `8d6877e129566fb2da3e1769daaa8eb02292d922`

### Known Limitations

- NVFP4 is a TorchAO prototype and is experimental.
- There is no NVFP4 end-to-end GPU training coverage in CI.
- It supports SM100 or later only.
- NVFP4 quantizes GEMMs only; tensor-parallel all-gather and reduce-scatter remain in bf16.
- The 200M-token result is limited to the documented Llama 3 8B C4 configuration. Validate convergence and performance for each new model, parallelism, and hardware configuration.

## Experimental NVFP4 Training on Blackwell GPUs

NVFP4 training dynamically quantizes linear GEMM activations, weights, and
gradients to NVFP4 through TorchAO's training prototype. The model weights and
distributed collectives remain in bf16. This reduces memory use and can improve
throughput on NVIDIA Blackwell GPUs.

> [!WARNING]
> NVFP4 training is experimental. It depends on a TorchAO prototype and has no
> backward-compatibility guarantees. NVFP4 training is not guarded by CI; the
> results below are evidence from a small set of 200M-token training runs, not
> broad numerical or performance validation.

### Requirements

- NVIDIA Blackwell SM100 or later GPU with CUDA.
- A PyTorch and TorchAO build that provides
  `torchao.prototype.moe_training.nvfp4_training`.
- `torch.compile` for competitive performance. The provided
  `llama3_8b_nvfp4_mixed` recipe enables model compilation automatically.
- Local GEMM dimensions divisible by 128. The LM head is therefore kept in
  bf16 because its vocabulary dimension does not meet this requirement.

### NVFP4 Training Recommendations

1. Train with NVFP4 for most of the run.
2. When exact loss recovery matters, switch the linear-layer GEMM inputs to a higher precision shortly before learning-rate decay begins.
3. Prefer switching only the forward-pass GEMMs to bf16 or, potentially, MXFP8.
4. When maximum NVFP4 utilization matters, remain in NVFP4 until nearly the end, then use a very brief high-precision finish. This can meaningfully improve loss, but may not completely recover it because the learning rate is already small.
5. When downstream accuracy is already sufficient, the end switch is optional.
6. The mixed recipe follows [Pretraining Large Language Models with NVFP4](https://arxiv.org/abs/2509.25149). It converts the leading 85% of decoder layers to NVFP4 and leaves the final 15% plus the LM head in bf16 for stability.

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

For the 32-layer Llama 3 8B model, layers 0-26 use NVFP4 and layers 27-31 remain bf16.


### Llama 3 8B 200M-Token Evidence

The following eager + `torch.compile` runs trained Llama 3 8B on C4 for 200M
tokens with global batch size 128 (local batch size 32, FSDP degree 4). Results
are reported at step 760.

| Run | Loss | Tokens/sec (per GPU) | Memory |
| --- | ---: | ---: | ---: |
| NVFP4 (bf16 tail) | 1.2715 | 30,040 | 110.95 GiB (60.2%) |
| MXFP8 | 1.2671 | 28,084 | 179.94 GiB (97.6%) |
| BF16 | 1.2738 | 21,919 | 174.49 GiB (94.7%) |

The bf16 tail costs about 5% throughput and 8 GiB compared with full NVFP4,
while remaining faster and using less memory than the MXFP8 and bf16 baselines
in this run. Its final loss is on par with the compared precisions.

![Llama 3 8B NVFP4, MXFP8, and BF16 training loss curves](../../../assets/images/nvfp4_vs_mxfp8_vs_bf16_eager_compile_200m_tokens.png)

*Llama 3 8B training loss through 200M tokens at global batch size 128. All runs use eager execution with model compilation.*

### Qwen3 8B 200M-Token Evidence

The following eager + `torch.compile` runs trained Qwen3 8B on C4 for
200,015,872 tokens with global batch size 64 (local batch size 16, FSDP degree
4), sequence length 2048, and 1,526 optimizer steps. Results are reported at
the final logged step, 1,520. The mixed NVFP4 recipe converts layers 0-29 to
NVFP4 and leaves layers 30-35 plus the LM head in bf16.

#### Random Initialization

| Run | Loss | Tokens/sec (per GPU) | Peak Reserved Memory |
| --- | ---: | ---: | ---: |
| NVFP4 (bf16 tail) | 3.82986 | 26,732 | 78.63 GiB (42.67%) |
| MXFP8 | 3.81545 | 27,587 | 112.63 GiB (61.12%) |
| BF16 | 3.81439 | 20,913 | 112.88 GiB (61.25%) |

In this run, NVFP4 delivered 28% more throughput than bf16 while using 34.25
GiB (30%) less peak reserved memory. It was 3% slower than MXFP8 while using
34.00 GiB (30%) less memory. Its final logged loss was 0.01547 above bf16 and
0.01441 above MXFP8.

![Qwen3 8B random-initialization NVFP4, MXFP8, and BF16 training loss curves](../../../assets/images/qwen3_8b_random_init_nvfp4_vs_mxfp8_vs_bf16_eager_compile_200m_tokens.png)

*Qwen3 8B random-initialization training loss through 200M tokens at global batch size 64.*

The random-initialization NVFP4 and MXFP8 runs used TorchTitan revision`20a66c9a108af41444222169982e15105de4c0e9`; the bf16 run used `34c805f3224dca3b9ea4188cd53b0a25c68bde34`.

### Versioned Environment

The Llama results and instructions use the container's current upstream builds:

- PyTorch: `2.14.0a0+gitd9abf9e`
- TorchAO: `0.18.0+gitcb76f29`
- TorchTitan: `8d6877e129566fb2da3e1769daaa8eb02292d922`

### Known Limitations

- NVFP4 is a TorchAO prototype and is experimental.
- There is no NVFP4 end-to-end GPU training coverage in CI.
- It supports SM100 or later only.
- NVFP4 quantizes GEMMs only; tensor-parallel all-gather and reduce-scatter remain in bf16.
- The 200M-token results are limited to the documented Llama 3 8B and Qwen3 8B C4 configurations. Validate convergence and performance for each new model, parallelism, and hardware configuration.

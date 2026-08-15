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
  `llama3_8b_first_85_pct_layers_nvfp4` recipe enables model compilation automatically.
- Local GEMM dimensions divisible by 128. A Linear whose local in/out features
  are not a multiple of 128 (after TP sharding) is rejected by the NVFP4 kernels
  and must be excluded from the converter. The mixed recipe converts only
  decoder layers, so the token embeddings and LM head always remain in bf16.

### NVFP4 Training Recommendations

1. Use NVFP4 for most of pretraining.

2. Keep a small set of numerically sensitive linear layers in higher precision throughout training. As a general rule, **leave approximately the final 15% of decoder blocks in BF16**, and keep the LM head in BF16. The layer-selection policy follows [What Matters for NVFP4 Training? A Scaling Study of Low-Precision Pre-Training Recipes](https://openreview.net/pdf?id=jlkIyaG32w). This is also consistent with [Pretraining Large Language Models with NVFP4](https://arxiv.org/abs/2509.25149), which identifies the final blocks as the most precision-sensitive and recommends keeping a small fraction—fewer than approximately 15%—of the final layers in BF16. Its conservative 12B training run additionally kept the first two blocks in BF16.

3. When matching higher-precision training loss is important, Appendix D recommends “switching to high precision shortly before the onset of learning rate decay” for full loss recovery. A switch performed only at the very end can still improve loss, but may not completely close the gap because the learning rate is already small.

4. [Appendix D. Switching to Higher Precision](https://arxiv.org/abs/2509.25149) finds that most of the loss gap comes from quantization in the forward pass. Switching only the forward-pass GEMM inputs to BF16, while leaving Dgrad and Wgrad in NVFP4, reduced the paper's relative loss error from approximately 1.5% to 0.5%. The authors observed no corresponding benefit from a backward-only switch, and the forward-only policy placed only approximately 6% of total training computation in higher precision.

5. The current TorchTitan NVFP4 integration does not support independently switching Fprop to BF16 while retaining NVFP4 for Dgrad and Wgrad. The practical fallback is a full-BF16 finish, which is more expensive than the paper's preferred forward-only policy and approximates the Appendix D experiment.

* Save a TorchTitan checkpoint at the desired precision-switch boundary.
* Restart from that checkpoint with the NVFP4 converter or override disabled.
* Set the training dtype to BF16.
* Restore the model, optimizer, learning-rate scheduler, dataloader, and trainer-step state, and continue the same learning-rate schedule.

5. For maximum NVFP4 utilization, defer the switch until very near the end of training. A short BF16 finish can still improve loss, although it may not completely recover the higher-precision baseline because the learning rate is already small.

6. Skip the end-of-training switch when the NVFP4 model already meets the desired downstream quality.

The following recommendations are based on
* [Pretraining Large Language Models with NVFP4](https://arxiv.org/abs/2509.25149),
particularly Appendix D. Switching to Higher Precision
* [What Matters for NVFP4 Training? A Scaling Study of Low-Precision Pre-Training Recipes](https://openreview.net/pdf?id=jlkIyaG32w)

### Llama 3 8B Usage

Use the `llama3_8b_first_85_pct_layers_nvfp4` config for the supported Llama 3 8B recipe:

```bash
torchrun --standalone --nproc_per_node 4 \
  -m torchtitan.train \
  --module llama3 \
  --config llama3_8b_first_85_pct_layers_nvfp4 \
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

In this run nvfp4 remained faster and used less memory than both the MXFP8 and bf16 baselines, and its final loss is on par with them.

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

### Versioned Environment

The Llama results and instructions use the container's current upstream builds:

- PyTorch: `2.14.0a0+gitd9abf9e`
- TorchAO: `0.18.0+gitcb76f29`

### Known Limitations

- NVFP4 is a TorchAO prototype and is experimental.
- There is no NVFP4 end-to-end GPU training coverage in CI.
- It supports SM100 or later only.
- NVFP4 quantizes GEMMs only; tensor-parallel all-gather and reduce-scatter remain in bf16.
- The 200M-token results are limited to the documented Llama 3 8B and Qwen3 8B C4 configurations. Validate convergence and performance for each new model, parallelism, and hardware configuration.

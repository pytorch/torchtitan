# DeepSeek V3 vLLM Engine Inference Benchmark

## Benchmarking inference only using vllm Engine on MoE EP, Dense TP+SP

This log tracks inference-only benchmark runs using the vLLM engine with
TorchTitan model definitions. Add one row per run so different parallelism,
compile, CUDA graph, batch size, and token settings can be compared directly.

### Results

| Date | Host | Model | GPUs | DP | Dense TP+SP | MoE EP | Batch size | Max tokens | Mode | Profile | Throughput tokens/s | Latency ms/token | Peak memory GB | Total tokens | Runs used | Output |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 2026-07-29 | devgpu006.rva3 | DeepSeek V3 16B | 8x H100 | 4 | 2 | 8 | 64 | 128 | compile, async TP, no vLLM cudagraph | yes | 667.38 | 1.50 | 85.66 | 8,192 | 1 | [json](../benchmark_deepseek_v3_16b_bs64_mt128_async_tp.json), [traces](../outputs/outputs_profile_deepseek_v3_16b_vllm_async_tp/vllm_torchtitan_deepseek_v3_16B_dp4_tp2_ep8_compile_bsz64_maxtok128) |

### Run Command

```bash
source .venv/bin/activate

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
LD_LIBRARY_PATH="$PWD/.venv/lib/python3.12/site-packages/torch/lib:$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH:-}" \
LD_PRELOAD="$PWD/.venv/lib/python3.12/site-packages/nvidia/cu13/lib/libcublas.so.13" \
NCCL_NVLS_ENABLE=0 \
PYTHONPATH=$PWD \
.venv/bin/torchrun --nproc_per_node=8 \
  torchtitan/experiments/rl/scripts/benchmarking_perf.py \
  --model-family deepseek_v3 \
  --model-size 16B \
  --model-path /home/jessicazhong/torchtitan/assets/hf/deepseek-moe-16b-base \
  --dp 4 \
  --tp 2 \
  --ep 8 \
  --spmd-backend spmd_types \
  --enable-async-tp \
  --use-compile-cudagraph \
  --compile-backend inductor \
  --batch-size 64 \
  --max-tokens 128 \
  --warmup-runs 2 \
  --num-runs 5 \
  --test-cases vllm-torchtitan \
  --profile \
  --profile-dir outputs/outputs_profile_deepseek_v3_16b_vllm_async_tp \
  --output benchmark_deepseek_v3_16b_bs64_mt128_async_tp.json
```

### Notes

- The benchmark request passed `--use-compile-cudagraph`, but vLLM CUDA graph
  capture is disabled for async TP because the compiled collective path performs
  host-side event synchronization. Torch compile remains enabled.
- HF weight loading is disabled for the `spmd_types` TP benchmark path, so this
  run uses initialized weights while exercising the same model shapes, sharding,
  compile path, async TP path, and vLLM attention path.
- Profiling was limited to one active iteration, with uncompressed Chrome traces
  and CUDA time table dumping disabled, to avoid profiler stop/post-processing
  stalls on large traces.
- Final profile traces were written as one `.pt.trace.json` file per rank under
  the trace directory linked in the results table.


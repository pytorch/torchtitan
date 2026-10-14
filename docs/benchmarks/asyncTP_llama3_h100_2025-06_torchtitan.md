The following performance benchmarks were done by the PyTorch team in June 2025, to measure the performance improvements of async TP over the vanilla TP baseline.

### Models

Llama 3.1 8B, 70B

### Hardware

We ran our performance benchmarks on the [Grand Teton platform](https://engineering.fb.com/2022/10/18/open-source/ocp-summit-2022-grand-teton/), where
- Each host has 8 NVIDIA H100 GPUs fully connected with NVLink.
- Each H100 GPU is equipped with 96GB HBM2e with 2.4 TB/sec peak memory bandwidth.
- Hosts are inter-connected with backend RDMA network with 400 Gb/s per GPU.
- We used the default 500W power limit, although tuning it up to 700W TDP can potentially provide further speedups.


### Results

Detailed performance results and training configurations can be found in the tables below:

#### Llama3 70b on 256 H100s with FSDP=32, TP=8, torch.compile, full AC,  local batch size 16

| Quantization      | Vanilla TP tokens/sec | Async TP tokens/sec |  Async TP speedup
| :---------------- | :---- | :---- | :--- |
| None (bfloat16)   | 597.3 | 652.4 | 1.09 |
| float8 tensorwise | 809.8 | 942.4 | 1.16 |
| float8 rowwise    | 599.6 | 624.8 | 1.04 |

#### Llama3 8b on 64 H100s with FSDP=8, TP=8, torch.compile, per op SAC, local batch size 12

| Quantization      | Vanilla TP tokens/sec | Async TP tokens/sec |  Async TP speedup
| :---------------- | :----- | :----- | :--- |
| None (bfloat16)   | 4378   | 4809.4 | 1.10 |
| float8 tensorwise | 5078.1 | 5570.1 | 1.10 |
| float8 rowwise    | 3708.5 | 3914.9 | 1.06 |

**Note**: the low baseline performance of the vanilla TP float8 rowwise training is being addressed here: https://github.com/pytorch/torchtitan/issues/1207

### Versions and Dates

| repo | commit | date |
| --- | --- | --- |
| torch | [38410cf9](https://github.com/pytorch/pytorch/commit/38410cf9b57079f3360c1e79601973a01cb2588c) | 2025/06/14 |
| torchao | [6243040](https://github.com/pytorch/ao/commit/6243040807b9ceee889a58cba8e68c5fc4e2ebd8) | 2024/06/13 |
| torchtitan | [820504e](https://github.com/pytorch/torchtitan/commit/820504e20d1149fbf0b98c567af24c4b0433b22d) | 2024/06/13 |

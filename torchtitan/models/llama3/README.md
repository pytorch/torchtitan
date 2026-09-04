# Llama 3

Llama 3 is the reference decoder model in torchtitan. Training recipes cover
Llama 3.1 8B, 70B, and 405B, plus a small debug model used by CI.

## Download the tokenizer

Follow the access instructions on the official
[meta-llama](https://huggingface.co/meta-llama/Llama-3.1-8B) repository, then:

```bash
python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer
```

The 8B, 70B, and 405B recipes expect tokenizer assets under
`./assets/hf/Llama-3.1-8B` (and the matching 70B / 405B paths).

## Training

```bash
# Debug model (used by integration tests)
MODULE=llama3 CONFIG=llama3_debugmodel ./run_train.sh

# Llama 3.1 8B
MODULE=llama3 CONFIG=llama3_8b ./run_train.sh
```

Other recipes include `llama3_70b` and `llama3_405b`. See
[`config_registry.py`](./config_registry.py).

## Supported Parallelisms

Coverage below matches `parallelize.py` and the Llama 3 jobs in
`tests/integration_tests/features.py` (plus Float8 jobs in
`tests/integration_tests/h100.py`).

| Feature | Notes |
|---------|-------|
| FSDP / HSDP | Default data-parallel path |
| Tensor Parallel (TP) | Including sequence parallel; async TP is exercised on H100 |
| Context Parallel (CP) | Composes with FSDP, HSDP, DDP, and TP |
| Pipeline Parallel (PP) | 1F1B, Interleaved1F1B, and GPipe. Zero-bubble / split-backward PP tests are disabled in `tests/integration_tests/features.py` because FlexAttention `BlockMask` is not a Tensor |
| DDP | Including DDP+CP |
| Activation checkpointing | Selective and full |
| `torch.compile` | 1D and multi-dimensional jobs |
| Float8 | H100 integration tests; `llama3_debugmodel_float8` |
| MXFP8 | Recipe `llama3_8b_mxfp8` exists; not in the default GPU feature suite |

## Numerical checks

Llama 3 is the baseline used to validate distributed training techniques. See
`tests/integration_tests` and [docs/converging.md](/docs/converging.md) rather
than treating any single published KL or MFU number as a parity claim.

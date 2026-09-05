## Available features
#### Dense Model
- Qwen3 dense model:
    - supports FSDP/HSDP, TP, CP, DDP.
    - Supports AC, torch.compile.
- Qwen3 MoE model:
    - Supports FSDP/HSDP, TP, CP, DDP, EP.
    - Supports AC, torch.compile.
    - MoE models use Token Choice routing. Load-balancing follows the original
      Qwen3 training recipe (auxiliary loss).

Dense and MoE debug models are covered by `tests/integration_tests/models.py`
(FSDP+TP+CP, and FSDP+TP+CP+EP for MoE).

Model architectures exist for 4B, 8B, and 235B-A22B. 8B has a pretrain
recipe (`qwen3_8b`) and an SFT recipe (`sft_qwen3_8b_math`). 4B and
235B-A22B do not yet have pretrain `config_registry` recipes.

## Download Qwen3 tokenizer
```python scripts/download_hf_assets.py --repo_id <hf_repo_name> --assets tokenizer```

eg, for Qwen3 0.6B model, the HF repo name is `Qwen/Qwen3-0.6B`. For 1.7B model, the HF repo name is `Qwen/Qwen3-1.7B`.


## Remaining work
- Add `config_registry` recipes for 4B and 235B-A22B.
- Verify learning rate and schedule on longer training jobs, or cite official
  references.
- Compare against established performance benchmarks.

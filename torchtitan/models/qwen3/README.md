**The Qwen3 model is still under development.**


## Available features
#### Dense Model
- Qwen3 dense model:
    - supports FSDP/HSDP, TP, DDP.
    - Supports AC, torch.compile.
- Qwen3 MoE model:
    - Supports FSDP/HSDP, TP, DDP, EP.
    - Supports AC, torch.compile.
    - MoE models use Token Choice routing, which is using auxiluary-loss-free load balancing algorithm.


Other model sizes are added to the configs, but config_registry entries need to be added and tested.

## Download Qwen3 tokenizer
```python scripts/download_hf_assets.py --repo_id <hf_repo_name> --assets tokenizer```

eg, for Qwen3 0.6B model, the HF repo name is `Qwen/Qwen3-0.6B`. For 1.7B model, the HF repo name is `Qwen/Qwen3-1.7B`.


## To be added
- Modeling
    - CP is not supported currently because of RoPE embedding implementation details.

- Testing
    - Learning rate verifying: verify learning rate and schedule with real training jobs (eg, 3k stps), or find official references.
    - The model should be tested against established performance benchmarks
    - CI integration

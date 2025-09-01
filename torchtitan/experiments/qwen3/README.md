**The Qwen3 model is still under development.**


#### Available features
QWEN3 0.6B Dense model is available for:

- FSDP/HSDP, TP, DDP, AC, compile support

Other model sizes are added to the args, but toml file configs need to be added and tested.

#### Download Qwen3 tokenizer
```python scripts/download_hf_assets.py --repo_id <hf_repo_name> --assets tokenizer```

eg, for Qwen3 0.6B model, the HF repo name is `Qwen/Qwen3-0.6B`. For 1.7B model, the HF repo name is `Qwen/Qwen3-1.7B`.

#### Parity with HF

Model parity test has been done and results suggest parity with HF implementation.

#### To be added
- Modeling
    - Variants of Dense models up to 32B
    - MoE alternatives

- Testing
    - Learning rate verifying: verify learning rate and schedule with real training jobs (eg, 3k stps), or find official references.
    - The model should be tested against established performance benchmarks
    - CI integration

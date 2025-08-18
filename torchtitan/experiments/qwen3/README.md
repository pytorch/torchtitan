**The Qwen3 model is still under development.**


#### Available features
QWEN3 0.6B Dense model is available for:

- FSDP/HSDP, TP, DDP, AC, compile support

Other model sizes are added to the args, but toml file configs need to be added and tested. Further testing is needed to check the coistency of the parallelism implementations.

#### Download Qwen3 tokenizer

```python scripts/download_tokenizer.py --repo_id Qwen/Qwen3-0.6B```


#### Parity with HF

Model parity test has been done and results suggest parity with HF implementation. Further investigation is needed to check the sanity of the Rope function.

#### To be added
- Modeling
    - Variants of Dense models up to 32B
    - MoE alternatives
    - Weight tying
- Testing
    - The model should be tested against established performance benchmarks
    - CI integration

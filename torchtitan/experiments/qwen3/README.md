**The Qwen3 model is still under development.**



#### Available features
QWEN3 0.6B Dense model - Other model sizes are added to the args, but toml file configs need to be added and tested
for FSDP support.

#### Download Qwen3 tokenizer

python scripts/download_tokenizer.py --repo_id Qwen/Qwen3-0.6B


#### Parity with HF

Model parity test has been done and results suggest parity with HF implementation.

#### To be added
- Modeling
    - variants of Dense models up to 32B
    - MoE alternatives
- Parallelism
    - TP, CP, and DDP
- Testing
    - Model needs to be tested across different metrics
    - CI integration

**The Qwen3 model is still under development.**



#### Available features
QWEN3 0.6B Dense model - Other model sizes are also added to the args, but toml file configs need to be added and tested
FSDP support

#### Download Qwen3 tokenizer

python scripts/download_tokenizer.py --repo_id Qwen/Qwen3-0.6B

#### To be added
- Modeling
    - variants of Dense models up to 32B
    - MoE alternatives 
- Parallelism
    - TP, CP, and DDP
- Testing
    - Model needs to be tested across different metrics
    - CI integration

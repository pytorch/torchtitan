**The Llama 4 folder is still under development.**

#### Issue tracking
https://github.com/pytorch/torchtitan/issues/1118

#### Available features
- Llama 4 model (text-only), including a token-choice MoE architecture with efficient bfloat16 Grouped MM kernels and auxiliary-loss-free load balancing
- FSDP, TP, PP, CP support
- DCP checkpoint conversion scripts

#### Download Llama 4 tokenizer
```bash
# Llama 4 tokenizer.model
python scripts/download_tokenizer.py --repo_id meta-llama/Llama-4-Scout-17B-16E --tokenizer_path "" --hf_token=...
```

#### To be added
- Modeling
    - alternative expert-choice MoE
    - multimodal support
- Parallelism
    - Context Parallel support for FlexAttention and multimodal inputs
    - Expert Parallel support
- torch.compile
    - for MoE layers
- Quantization
    - efficient float8 Grouped MM kernels (from torchao)
- Testing
    - perfomance and loss converging tests
    - CI integration

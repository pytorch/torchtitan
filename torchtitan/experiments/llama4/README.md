**The Llama 4 folder is still under development.**

#### Available features
- Llama 4 model definition (text-only), including the MoE architecture with token-choice routing
- Basic FSDP, TP, PP, CP support
- DCP checkpoint conversion scripts

#### Download Llama 4 tokenizer
```bash
# Llama 4 tokenizer.model
python scripts/download_tokenizer.py --repo_id meta-llama/Llama-4-Scout-17B-16E --tokenizer_path "" --hf_token=...
```

#### To be added
- Modeling
    - iRoPE implementation
    - load balance loss for token-choice MoE
    - alternative expert-choice MoE
    - multimodal support
- Kernel integration
    - efficient bfloat16 GroupedGEMM kernels (from PyTorch core)
    - efficient float8 GroupedGEMM kernels (from torchao)
- Parallelism
    - performant TP implementation and torch.compile support for MoE layers
    - Context Parallel support for FlexAttention, iRoPE, and multimodal inputs
    - Expert Parallel support
- Testing
    - perfomance and loss converging tests
    - CI integration

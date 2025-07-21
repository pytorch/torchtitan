The Qwen3 model is still under development.



Available features
QWEN3 0.6B Dense model - Other model sizes are also added to the args, but toml file configs need to be added and tested
FSDP support

Download Qwen3 tokenizer
# Qwen 3 tokenizer.model
python scripts/download_tokenizer.py --repo_id Qwen/Qwen3-0.6B

To be added
Other modes of TP,CP, and DDP
Needs to be tested for bigger size models

Modeling
alternative expert-choice MoE
multimodal support
Parallelism
Context Parallel support for FlexAttention and multimodal inputs
torch.compile
for MoE layers
Quantization
efficient float8 Grouped MM kernels (from torchao)
Testing
perfomance and loss converging tests
CI integration

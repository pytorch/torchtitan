# GRPO

GRPO instructions

## Installation instructions
```shell
pip install uv
uv pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu129
uv pip install -r requirements.txt
pip install --pre -U xformers
export VLLM_COMMIT=2918c1b49c88c29783c86f78d2c4221cb9622379
uv pip install vllm torch==2.9.0 --torch-backend=cu129 --prerelease=allow --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT} --extra-index-url https://download.pytorch.org/whl/cu129
pip install flashinfer-python==0.4.1 flashinfer-cubin==0.4.1
pip install flashinfer-jit-cache==0.4.1 --index-url https://flashinfer.ai/whl/cu129
```

## Configuration instructions

see `torchtitan/grop/configs/qwen25-7b-math.toml` for good initial values

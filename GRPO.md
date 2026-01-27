# GRPO

GRPO instructions

## Installation instructions
```shell
mkdir logs
chmod g+rw ./logs
pip install uv
uv pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu129
uv pip install -r requirements.txt
pip install --pre -U xformers
export VLLM_COMMIT=2918c1b49c88c29783c86f78d2c4221cb9622379
uv pip install vllm torch==2.9.0 --torch-backend=cu129 --prerelease=allow --extra-index-url https://wheels.vllm.ai/${VLLM_COMMIT} --extra-index-url https://download.pytorch.org/whl/cu129
pip install flashinfer-python==0.4.1 flashinfer-cubin==0.4.1
pip install flashinfer-jit-cache==0.4.1 --index-url https://flashinfer.ai/whl/cu129
pip install transformers==4.57.1
```

## Configuration instructions

see `torchtitan/grop/configs/qwen25-7b-math.toml` for good initial values

## sbatch script

`online_multinode_vllm.slurm` contains some paths to edit,
- TRAIN_PATH - where this is installed on the cluster
- TRAIN_ENV - if you don't init the venv to .venv, this needs to be changed to that venv
- VLLM_ENV - same as TRAIN_ENV unless you're doing something different
- API_ENV - atropos venv

One that's done, you can do something like
```bash
sbatch --export=ALL,CONFIG_FILE=/home/dakota/github/torchtitan/torchtitan/grpo/configs/qwen25-7b-math.toml,MODEL_NAME=Qwen/Qwen2.5-7B,PYTHON_SCRIPT=/home/dakota/github/atropos/environments/math_server_zero.py,WANDB_PROJECT=qwen7b_debug online_multinode_vllm.slurm
```
to launch a run

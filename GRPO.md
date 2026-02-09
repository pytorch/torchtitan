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

see `torchtitan/grpo/configs/qwen25-7b-math.toml` for good initial values

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

## LoRA Training (Multi-Node with vLLM)

For LoRA-based GRPO training you can use the multi-node Slurm launcher which
splits nodes between training (torchtitan + torchrun) and inference (vLLM with
hot-reloaded LoRA adapters).

### Train config

LoRA parameters live in the `[peft]` section of your torchtitan TOML config:

```toml
[peft]
enable_peft = true
use_lora = true
lora_rank = 64
lora_alpha = 1.0
lora_dropout = 0.0
train_embeddings = true
train_output_layer = true
```

The PEFT setup script (`scripts/setup_peft_base.py`) reads these values
directly from the config file, so there is no need to specify them separately.

### PEFT base adapter

Before inference can begin, vLLM needs a PEFT adapter directory that defines
the LoRA structure (rank, alpha, target modules, etc.). The actual trained
weights are loaded at runtime, but the scaffold must exist on disk first.

`scripts/setup_peft_base.py` handles this automatically:

```shell
# Standalone usage — reads everything from the train config:
python scripts/setup_peft_base.py \
    --config-file path/to/train_config.toml \
    --base-dir /home/shared/peft_bases

# Or with explicit args:
python scripts/setup_peft_base.py \
    --base-model moonshotai/Kimi-K2-Instruct \
    --rank 16 --alpha 1 --dropout 0.0
```

The script prints shell-eval-friendly output to stdout:

```
PEFT_PATH=/home/shared/peft_bases/moonshotai__Kimi-K2-Instruct/r16_a1.0_d0.0_abc123
BASE_MODEL=moonshotai/Kimi-K2-Instruct
LORA_RANK=16
LORA_ALPHA=1.0
LORA_DROPOUT=0.0
```

The Slurm script uses `eval $(...)` to capture these as environment variables
which are then forwarded to every node via `srun --export=ALL`.

### Launching a multi-node LoRA run

A ready-to-use LoRA config lives at
`torchtitan/grpo/configs/nous-3ba-30b-math-lora.toml`. Point `CONFIG_FILE` at
it (or your own copy) when launching:

```shell
CONFIG_FILE=torchtitan/grpo/configs/nous-3ba-30b-math-lora.toml \
MODEL_NAME=Qwen/Qwen3-30B-A3B \
WANDB_PROJECT=my-grpo-lora \
NUM_TRAINING_NODES=1 \
NUM_INFERENCE_NODES=1 \
sbatch --export=ALL online_multinode_vllm_lora.slurm
```

Key environment variables accepted by the Slurm script:

| Variable | Default | Description |
|---|---|---|
| `CONFIG_FILE` | *(required)* | Path to torchtitan TOML train config |
| `MODEL_NAME` | `"default_model"` | HuggingFace model name / path |
| `NUM_TRAINING_NODES` | `1` | Nodes allocated to torchrun training |
| `NUM_INFERENCE_NODES` | `1` | Nodes allocated to vLLM inference |
| `PEFT_BASE_DIR` | `/home/shared/peft_bases` | Where PEFT adapter scaffolds are stored |
| `TRAINING_ARGS` | `""` | Extra CLI args forwarded to torchtitan |
| `WANDB_PROJECT` | `"torchtitan-grpo"` | Weights & Biases project name |


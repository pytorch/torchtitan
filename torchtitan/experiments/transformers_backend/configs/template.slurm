#!/bin/bash
#SBATCH --job-name={{ name }}
#SBATCH --output={{ root_path }}/slurm_%j.out
#SBATCH --error={{ root_path }}/slurm_%j.out
#SBATCH --nodes={{ nodes }}
#SBATCH --ntasks-per-node={{ n_proc_per_node }}
#SBATCH --gpus-per-task=1
#SBATCH --qos={{ qos }}
#SBATCH --cpus-per-task=12

# Misc initializations.
echo "========================"
echo "START TIME: $(date)"
source /etc/profile.d/modules.sh
source /fsx/ferdinandmom/ferdinand-hf/huggingface/torchtitan/env_torchtitan_official/bin/activate
echo python3 version = $(python3 --version)
echo "==========="

# Slurm stuff
export HOSTNAMES=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((1024 + RANDOM % 64511))

export TMPDIR=/scratch
export TORCH_HOME="/fsx/ferdinandmom/cache/torch"
export HF_HOME="/fsx/ferdinandmom/cache/huggingface"
export HF_DATASETS_CACHE="/fsx/ferdinandmom/cache/huggingface/datasets"
export TRANSFORMERS_CACHE="/fsx/ferdinandmom/cache/huggingface/transformers"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export CUDA_DEVICE_MAX_CONNECTIONS="1"
export UV_CACHE_DIR="/fsx/ferdinandmom/.cache/uv"

module load cuda/12.4

echo "Running training job: {{ name }}"
echo "Config file: {{ config_path }}"

{% if name == "seed_checkpoint" %}
python /fsx/ferdinandmom/ferdinand-hf/huggingface/torchtitan/scripts/download_hf_assets.py --repo_id {{ repo_id }} --local_dir {{ root_path }} --assets tokenizer
{% endif %}

torchrun \
   --nproc_per_node {{ n_proc_per_node }} \
   --nnodes {{ nodes }} \
   --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
   --rdzv_backend c10d \
   --max_restarts 0 \
   --tee 3 \
   -m torchtitan.train \
   --checkpoint.enable \
   {% if name == "seed_checkpoint" %} --checkpoint.create_seed_checkpoint {% else %} --checkpoint.initial_load_path {{ initial_load_path }} {% endif %} \
   --training.seed 42 \
   --training.deterministic \
   --job.config_file {{ config_path }}

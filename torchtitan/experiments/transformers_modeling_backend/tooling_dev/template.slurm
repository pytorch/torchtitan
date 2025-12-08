#!/bin/bash
#SBATCH --job-name={{ name }}
#SBATCH --output={{ root_path }}/slurm_%j.out
#SBATCH --error={{ root_path }}/slurm_%j.out
#SBATCH --nodes={{ nodes }}
#SBATCH --gres=gpu:{{ n_proc_per_node }}
#SBATCH --ntasks-per-node=1
#SBATCH --qos={{ qos }}
#SBATCH --cpus-per-task=12
#SBATCH --partition=hopper-extra
#SBATCH --time=00:30:00

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

# EFA settings
export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1
export FI_EFA_ENABLE_SHM_TRANSFER=1
export NCCL_PROTO=simple
export NCCL_SOCKET_IFNAME=enp

module load cuda/12.4

echo "Running training job: {{ name }}"
echo "Config file: {{ config_path }}"

# Function to update status based on squeue output
update_status() {
    job_id=$1
    status_file=$2
    # For unknown reasons, it doenst update status for pending. It only works for running 
    while true; do
        job_status=$(squeue --job $job_id --noheader --format=%T)
        echo "Job status: $job_status"
        if [ -z "$job_status" ]; then
            # Job has finished or is not found
            break
        elif [ "$job_status" = "RUNNING" ]; then
            printf "running" > $status_file
            break
        fi
        sleep 10
    done
}

# Update status to "pending" or "running" in the background
update_status $job_id {{ root_path }}/status.txt &

# LOG_DIR="{{ root_path }}/logs"
# mkdir -p ${LOG_DIR}

# CMD="torchrun \
#    --nproc_per_node {{ n_proc_per_node }} \
#    --nnodes {{ nodes }} \
#    --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
#    --rdzv_backend c10d \
#    --max_restarts 0 \
#    --log-dir ${LOG_DIR} \
#    --role rank \
#    --tee 3 \
#    -m torchtitan.train \
#    --checkpoint.enable \
#    {% if name == "seed_checkpoint" %} --checkpoint.create_seed_checkpoint {% else %} --checkpoint.initial_load_path {{ initial_load_path }} {% endif %} \
#    --debug.seed 42 \
#    --debug.deterministic \
#    --training.steps 1 \
#    --job.config_file {{ config_path }}"


{% if name == "seed_checkpoint" %}python /fsx/ferdinandmom/ferdinand-hf/huggingface/torchtitan/scripts/download_hf_assets.py --repo_id {{ repo_id }} --assets tokenizer --local_dir {{ root_path }}/../../assets/hf {% endif %}

CMD="torchrun \
   --nproc_per_node {{ n_proc_per_node }} \
   --nnodes {{ nodes }} \
   --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
   --rdzv_backend c10d \
   --max_restarts 0 \
   --role rank \
   --local_ranks_filter {{ n_proc_per_node - 1 }} \
   --tee 3 \
   -m torchtitan.train \
   --checkpoint.enable \
   {% if name == "seed_checkpoint" %} --checkpoint.create_seed_checkpoint {% else %} --checkpoint.initial_load_path {{ initial_load_path }} {% endif %} \
   --debug.seed 42 \
   --debug.deterministic \
   {% if model_type == "transformers_modeling_backend" %}--job.custom_config_module=torchtitan.experiments.transformers_modeling_backend.job_config {% endif %}
   --job.config_file {{ config_path }}"

# Run the main command
echo "Running command: srun -u $CMD"
srun -u $CMD
exit_status=$?


# Update status based on the exit status of `srun`
if [ $exit_status -eq 0 ]; then
   printf "completed" > {{ root_path }}/status.txt
else
   printf "fail" > {{ root_path }}/status.txt
fi
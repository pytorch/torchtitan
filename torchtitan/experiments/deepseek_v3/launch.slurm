#!/bin/bash
#SBATCH --job-name=torchtune
#SBATCH --ntasks=4
#SBATCH --nodes=4
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=96
#SBATCH --output=/traindata/%u/runs/torchtune/slurm_logs/slurm-%j.out
#SBATCH --error=/traindata/%u/runs/torchtune/slurm_logs/slurm-%j.out

nodes=( $( scontrol show hostnames $SLURM_JOB_NODELIST ) )
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(srun --nodes=1 --ntasks=1 -w "$head_node" hostname --ip-address)

echo Node IP: $head_node_ip

# Job specific environment variables.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# Debug environment variables.
export LOGLEVEL=INFO
export NCCL_DEBUG=WARN
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0

# Network fabric environment variables.
export FI_PROVIDER="efa"
export NCCL_IB_DISABLE=1
export LD_LIBRARY_PATH=/opt/amazon/efa/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/local/lib/:$LD_LIBRARY_PATH
export NCCL_SOCKET_IFNAME="eth0,en,eth,em,bond"
export NCCL_BUFFSIZE=2097152
export FI_EFA_SET_CUDA_SYNC_MEMOPS=0

srun --container-image=/traindata/enroot_images/torchtune.sqsh \
  --container-mounts=/traindata:/traindata \
  bash -c "cd /traindata/$USER/torchtitan && \
    source /traindata/eugen/torchtune/.venv/bin/activate && \
    torchrun \
      --nnodes $SLURM_NNODES \
      --nproc_per_node 8 \
      --rdzv_id $SLURM_JOB_ID \
      --rdzv_backend c10d \
      --rdzv_endpoint \"$head_node_ip:29500\" \
      torchtitan/experiments/deepseek_v3/run.py"
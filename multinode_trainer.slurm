nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)
head_node=${nodes_array[0]}
head_node_ip=$(cat /etc/hosts | grep -w "$head_node" | awk '{print $1}')
echo $head_node

## env config
GPUS_PER_NODE=8
MASTER_ADDR=$head_node_ip
MASTER_PORT=29500
NNODES=$SLURM_NNODES

torchrun --nproc_per_node 8 --nnodes $NNODES --node_rank $SLURM_PROCID --master_addr $MASTER_ADDR --master_port $MASTER_PORT \
--rdzv_backend c10d --rdzv_id 101 --rdzv_endpoint "$head_node_ip:29500" --local-ranks-filter 0 \
./train.py --job.config_file ./train_configs/llama2_70b.toml
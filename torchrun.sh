CONFIG_FILE=${CONFIG_FILE:-"./train_configs/debug_model.toml"}
echo "torchrun --nproc-per-node=1 --nnodes=$ENV_WORLD_SIZE --node-rank=$ENV_NODE_RANK --master-addr="$ENV_MASTER_ADDR" --master-port=29500 train.py --job.config_file ${CONFIG_FILE}
"
torchrun --nproc-per-node=1 --nnodes=$ENV_WORLD_SIZE --node-rank=$ENV_NODE_RANK --master-addr="$ENV_MASTER_ADDR" --master-port=29500 train.py --job.config_file ${CONFIG_FILE}
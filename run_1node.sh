srun -p llm_s -n 1 -N 1 --ntasks-per-node=1 --gpus-per-task=8 torchrun --nproc_per_node=8 --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter 0 --role rank --tee 3 \
train.py --job.config_file ./train_configs/llama2_7b.toml
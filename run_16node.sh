srun -p llm_s -n 16 -N 16 --ntasks-per-node=1 --gpus-per-task=8 -w "HOST-10-140-60-[1-14,16,188]" \
torchrun --nproc_per_node 8 --rdzv_backend c10d --rdzv_id 101 --rdzv_endpoint "10.140.60.12:29500" \
--local-ranks-filter 0 ./train.py --job.config_file ./train_configs/llama2_7b.toml
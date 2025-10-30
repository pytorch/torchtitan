help:
	@echo "Choose another target"

fsdp-debug:
	export CONFIG_FILE=./torchtitan/models/deepseek_v3/train_configs/debug_model.toml && \
	./run_train.sh \
		--parallelism.data_parallel_shard_degree -1 \
		--parallelism.tensor_parallel_degree 1 \
		--parallelism.expert_parallel_degree 1 \

fsdp-16b:
	export CONFIG_FILE=./torchtitan/models/deepseek_v3/train_configs/deepseek_v3_16b.toml && \
	./run_train.sh \
		--parallelism.data_parallel_shard_degree -1 \
		--parallelism.tensor_parallel_degree 1 \
		--parallelism.expert_parallel_degree 1 \

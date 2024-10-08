## Enable Memory Profiling

Launch training job with the following command (or alternatively set configs in toml files)
```
CONFIG_FILE="./train_configs/debug_model.toml" ./run_llama_train.sh --profiling.enable_memory_snapshot --profiling.save_memory_snapshot_folder output_folder
```
* `--profiling.enable_memory_snapshot`: enable memory snapshot
* `--profiling.save_memory_snapshot_folder`: dump memory snapshots in to output folder, default to be `memory_snapshot`.
	+ If in case of OOMs. output folder is `memory_snapshot/iteration_x_exit`.
	+ If regularly according to `profile_freq`. output folder is `memory_snapshot/iteration_x`.

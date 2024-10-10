## Enable Memory Profiling

Launch training job with the following command (or alternatively set configs in toml files)
```
CONFIG_FILE="./train_configs/debug_model.toml" ./run_llama_train.sh --profiling.enable_memory_snapshot --profiling.save_memory_snapshot_folder memory_snapshot
```
* `--profiling.enable_memory_snapshot`: to enable memory profiling
* `--profiling.save_memory_snapshot_folder`: configures the folder which memory snapshots are dumped into (`./outputs/memory_snapshot/` by default)
	+ In case of OOMs, the snapshots will be in `./outputs/memory_snapshot/iteration_x_exit`.
	+ Regular snapshots (taken every `profiling.profile_freq` iterations) will be in `memory_snapshot/iteration_x`.

Once you have dumped the memory profiler, you will find the saved pickle files in your output folder.
To visualize a snapshot file, you can drag and drop it to <https://pytorch.org/memory_viz>. To learn more details on memory profiling, please visit this [tutorial](https://pytorch.org/blog/understanding-gpu-memory-1/).
